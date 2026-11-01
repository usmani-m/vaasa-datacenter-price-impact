"""Scenario engine: price effect of a data center on Finnish day-ahead prices.

Driven by config/scenarios.yaml (data_center, ppa, curtailment, run) and the coupling setup in
config/coupling.yaml. For each market year (historical price and coupling basis), model variant,
PPA option and data center size:

1. Data center load per hour: size x ramp-up share (target_year, else full size), flat, with an
   optional summer daytime cooling bump.
2. PPA (optional): new wind sized to match the data center's annual energy; output = fitted wind
   capacity factor (logistic wind curve, models/renewables_wind_2023_2025.json) for the market
   year's weather x PPA MW. Net added load = DC load - PPA output (can be negative).
3. Curtailment rule: in hours where the actual price was <= 0 and wind was curtailed (fitted
   uncurtailed wind - actual wind), extra load first absorbs the curtailed wind with no price
   effect; surplus PPA wind in such hours is curtailed too (no price effect).
4. Price change per hour by regime (src/models/coupling.py):
     coupled (incl. SE3-only), added load > 0: up to the spare import capacity at the coupled
       price (variant's coupled-hour model), the rest on the Finnish curve; the hours where the
       load exceeds the spare capacity "switch" from coupled to constrained.
     coupled, added load < 0 (PPA surplus): exported at the coupled price.
     import-constrained / other separated: Finnish curve.
     export-constrained: Finnish (export-constrained) curve; increases capped at the gap to SE1.
5. Outputs: hourly price change, annual / monthly / hour-of-day means, the price-duration curve
   with and without the data center (actual prices + change), switching hours, and the extra
   annual cost to existing Finnish consumers = sum(price change x consumption) in EUR million.
   This is a transfer at wholesale level, not a welfare measure; the data center's own load is
   not included.

Model variants: central (GBM; upper end of the likely range: hydro and import response not
modelled), binned, structural (low end).

Results do NOT scale linearly with size: larger data centers exceed the spare import capacity in
more coupled hours (switching), where the steeper Finnish curve applies.

Usage:
    python -m src.scenarios.engine
    python -m src.scenarios.engine --years 2024 --variants central
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models import coupling as K
from src.models import price as P
from src.models.renewables import logistic

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "scenarios.yaml"
WIND_MODEL_PATH = REPO_ROOT / "models" / "renewables_wind_2023_2025.json"
OUT_DIR = REPO_ROOT / "outputs" / "scenarios"
HOURLY_DIR = REPO_ROOT / "data" / "scenarios"

# variant -> (model for the imported part in coupled hours, Finnish-curve model family)
VARIANTS = {"central": ("gbm", "gbm"), "binned": ("binned", "binned"), "structural": ("structural", "gbm")}
PDC_POINTS = np.linspace(0, 100, 101)  # price-duration curve: price exceeded in p% of hours


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def wind_cf(ws100: pd.Series) -> pd.Series:
    """Fitted (uncurtailed) wind capacity factor from mean 100 m wind speed."""
    p = json.loads(WIND_MODEL_PATH.read_text())["logistic_pooled"]
    return pd.Series(logistic(ws100.to_numpy(), p["L"], p["k"], p["v0"], p["c"]), index=ws100.index)


def dc_load(d: pd.DataFrame, size_mw: float, cfg: dict) -> pd.Series:
    """Data center load (MW) per hour: size x ramp-up share, flat, plus optional cooling bump."""
    dc = cfg["data_center"]
    share = dc["ramp_up"].get(dc["target_year"], 1.0) if dc.get("target_year") else 1.0
    load = pd.Series(size_mw * share, index=d.index, dtype=float)
    bump = dc.get("cooling_bump", {})
    if bump.get("enabled"):
        on = d["month_local"].isin(bump["months"]) & d["hour_local"].isin(bump["hours_local"])
        load[on] *= 1 + bump["extra_share"]
    return load


class Year:
    """One market year: data, regimes and cached base predictions."""

    def __init__(self, t: pd.DataFrame, year: int):
        d = t[t["year"] == year].copy()
        self.d = d
        self.cf = wind_cf(d["ws100_mean"])
        # curtailed wind in hours with price <= 0: fitted uncurtailed output - actual output
        uncurtailed = self.cf * d["wind_capacity"]
        self.curtailed = (uncurtailed - d["wind"]).clip(lower=0).where(d["price_eur_mwh"] <= 0, 0.0).fillna(0.0)
        self._base: dict[int, np.ndarray] = {}

    def base(self, model) -> np.ndarray:
        key = id(model)
        if key not in self._base:
            self._base[key] = model.predict(self.d)
        return self._base[key]

    def delta(self, model, x: np.ndarray, sel: np.ndarray) -> np.ndarray:
        """Price change for added residual load x (array over all hours), on the rows in sel."""
        sub = self.d[sel]
        return model.predict(sub, x[sel]) - self.base(model)[sel]


def price_change(y: Year, x: np.ndarray, models: dict, variant: str) -> tuple[np.ndarray, np.ndarray]:
    """Hourly price change for added load x (MW, per hour), and the coupled hours that switch."""
    d = y.d
    cm, fm = VARIANTS[variant]
    imp_m, exp_m, full_m = (models["import_constrained"][fm], models["export_constrained"][fm],
                            models["full"][fm])
    coupled_m = K.ScaledDelta(imp_m, d["fi_load_share"]) if cm == "structural" else models["coupled"][cm]
    reg = d["regime"].to_numpy()
    head = d["headroom"].to_numpy()
    out = np.zeros(len(d))
    coupled = np.isin(reg, K.COUPLED)

    pos_c = coupled & (x > 0)
    if pos_c.any():
        cov = np.minimum(head, x)
        rest = np.where(pos_c, x - cov, 0.0)
        cov = np.where(pos_c, cov, 0.0)
        out[pos_c] = y.delta(coupled_m, cov, pos_c) if cm != "structural" else _scaled(y, coupled_m, cov, pos_c)
        has_rest = pos_c & (rest > 0)
        if has_rest.any():
            out[has_rest] += y.delta(imp_m, rest, has_rest)
    neg_c = coupled & (x < 0)
    if neg_c.any():
        out[neg_c] = y.delta(coupled_m, x, neg_c) if cm != "structural" else _scaled(y, coupled_m, x, neg_c)
    sep = np.isin(reg, ["import_constrained", "separated_other"]) & (x != 0)
    if sep.any():
        out[sep] = y.delta(imp_m, x, sep)
    exp = (reg == "export_constrained") & (x != 0)
    if exp.any():
        dp = y.delta(exp_m, x, exp)
        gap = np.clip(d["price_se1"].to_numpy()[exp] - d["price_eur_mwh"].to_numpy()[exp], 0, None)
        out[exp] = np.where(x[exp] > 0, np.minimum(dp, gap), dp)
    unk = (reg == "unknown") & (x != 0)
    if unk.any():
        out[unk] = y.delta(full_m, x, unk)
    switching = pos_c & (x > head)
    return out, switching


def _scaled(y: Year, scaled: K.ScaledDelta, x: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Structural variant: FI load share x Finnish-curve change, on the rows in sel."""
    share = y.d["fi_load_share"].to_numpy()[sel]
    return share * y.delta(scaled.base, x, sel)


def run_scenario(y: Year, size_mw: float, ppa: bool, variant: str, models: dict, cfg: dict) -> dict:
    d = y.d
    load = dc_load(d, size_mw, cfg)
    ppa_mw, ppa_out = 0.0, pd.Series(0.0, index=d.index)
    if ppa and size_mw > 0:
        cf = y.cf.clip(lower=0)
        ppa_mw = float(load.sum() / cf.sum())
        ppa_out = cf * ppa_mw
    x = (load - ppa_out).to_numpy()
    absorbed = np.zeros(len(d))
    if cfg["curtailment"]["enabled"]:
        neg_price = (d["price_eur_mwh"] <= 0).to_numpy()
        curt = y.curtailed.to_numpy()
        pos = neg_price & (x > 0)
        absorbed[pos] = np.minimum(x[pos], curt[pos])
        x = np.where(pos, x - absorbed, x)
        x = np.where(neg_price & (x < 0), 0.0, x)  # surplus PPA wind curtailed at price <= 0
    dp, switching = price_change(y, x, models, variant) if size_mw > 0 else (np.zeros(len(d)), np.zeros(len(d), bool))
    missing = np.isnan(dp)
    dp = np.nan_to_num(dp)
    s = pd.Series(dp, index=d.index)
    price = d["price_eur_mwh"]
    with_dc = price + s
    cons = d["consumption"].fillna(0)
    return {
        "hourly": s,
        "summary": {
            "market_year": int(d["year"].iloc[0]), "size_mw": size_mw, "ppa": ppa, "variant": variant,
            "ppa_mw": round(ppa_mw, 1),
            "dc_energy_twh": round(float(load.sum()) / 1e6, 4),
            "absorbed_curtailed_gwh": round(float(absorbed.sum()) / 1e3, 2),
            "mean_dprice": round(float(dp.mean()), 3),
            "consumption_weighted_dprice": round(float((dp * cons).sum() / cons.sum()), 3),
            "monthly_dprice": [round(float(v), 3) for v in s.groupby(d["month_local"]).mean().reindex(range(1, 13))],
            "hour_of_day_dprice": [round(float(v), 3) for v in s.groupby(d["hour_local"]).mean().reindex(range(24))],
            "switching_hours": int(switching.sum()),
            "consumer_cost_meur": round(float((dp * cons).sum()) / 1e6, 2),
            "pdc_with": [round(float(v), 2) for v in np.nanpercentile(with_dc, 100 - PDC_POINTS)],
            "hours_missing_inputs": int(missing.sum()),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", type=int, nargs="+", help="market years (default: config run.market_years)")
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS), help="default: config run.variants")
    args = p.parse_args()
    pd.set_option("display.width", 200, "display.max_columns", 30)

    cfg = load_config()
    years = sorted(args.years or cfg["run"]["market_years"])
    variants = args.variants or cfg["run"]["variants"]
    sizes = cfg["data_center"]["sizes_mw"]
    ppa_opts = cfg["ppa"]["enabled_options"]
    fit_years = cfg["run"]["fit_years"]

    ccfg = K.load_cfg()
    t = K.prepare(sorted(set(fit_years) | set(years)), ccfg)
    fit = t[t["year"].isin(fit_years)]
    models = K.fit_regime_models(fit, P.fit_mask(fit), fit["regime"])
    print(f"Models fitted on {fit_years}; running {len(years) * len(variants) * len(ppa_opts) * len(sizes)} scenarios")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HOURLY_DIR.mkdir(parents=True, exist_ok=True)
    results, baseline = [], {}
    for year in years:
        y = Year(t, year)
        price = y.d["price_eur_mwh"]
        baseline[str(year)] = {
            "mean_price": round(float(price.mean()), 2),
            "consumption_twh": round(float(y.d["consumption"].sum()) / 1e6, 3),
            "pdc_without": [round(float(v), 2) for v in np.nanpercentile(price, 100 - PDC_POINTS)],
            "coupled_share_pct": round(float(y.d["regime"].isin(K.COUPLED).mean() * 100), 1),
            "curtailed_wind_gwh_at_price_le_0": round(float(y.curtailed.sum()) / 1e3, 1),
        }
        for variant in variants:
            for ppa in ppa_opts:
                hourly = {}
                for size in sizes:
                    r = run_scenario(y, size, ppa, variant, models, cfg)
                    results.append(r["summary"])
                    hourly[f"{size}MW"] = r["hourly"].round(4)
                pd.DataFrame(hourly).to_parquet(HOURLY_DIR / f"dprice_{year}_{variant}_{'ppa' if ppa else 'noppa'}.parquet")
        print(f"  {year} done")

    out = {
        "description": "FI day-ahead price change from a data center (EUR/MWh) by market year, model variant, "
                       "PPA option and size. Δprice = modelled change added to actual historical prices.",
        "units": {"dprice": "EUR/MWh", "consumer_cost_meur": "EUR million per year (existing consumption)",
                  "pdc": "price exceeded in p% of hours, p = 0..100"},
        "variants": {"central": "GBM; upper end of the likely range (hydro and import response not modelled)",
                     "binned": "binned supply curve", "structural": "low end"},
        "config": {k: cfg[k] for k in ("data_center", "ppa", "curtailment", "run")},
        "pdc_points_pct": PDC_POINTS.tolist(),
        "baseline": baseline,
        "scenarios": results,
    }
    path = OUT_DIR / "grid.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} kB, {len(results)} scenarios); hourly in {HOURLY_DIR}")

    df = pd.DataFrame(results)
    for variant in variants:
        v = df[df["variant"] == variant]
        for metric, label in (("mean_dprice", "mean price change (EUR/MWh)"),
                              ("consumer_cost_meur", "extra cost to consumers (EUR million/yr)"),
                              ("switching_hours", "coupled hours switching to constrained")):
            tab = v.pivot_table(index="size_mw", columns=["market_year", "ppa"], values=metric)
            tab.columns = [f"{y} {'PPA' if p else 'no PPA'}" for y, p in tab.columns]
            print(f"\n=== {variant}: {label} ===")
            print(tab.round(2).to_string())


if __name__ == "__main__":
    main()

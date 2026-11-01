"""Export compact JSON for the static dashboard (dashboard/data/).

  impact.json      scenario grid (from outputs/scenarios/grid.json, run src.scenarios.engine first),
                   reshaped per (market year, PPA, variant) with one array entry per size
  history.json     2019-2025: monthly energy mix, residual load vs price sample + medians,
                   coupling regime shares, annual price stats
  renewables.json  wind capacity factor vs wind speed sample + fitted curve, monthly wind and
                   solar capacity factors, calculator defaults

All numbers are rounded; no API keys or raw data are exported.

Usage:
    python -m src.process.export_dashboard
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import coupling as K
from src.models import price as P

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "dashboard" / "data"
GRID = REPO_ROOT / "outputs" / "scenarios" / "grid.json"
WIND_MODEL = REPO_ROOT / "models" / "renewables_wind_2023_2025.json"
HIST_YEARS = list(range(2019, 2026))
PDC_STEP = 2  # keep every 2nd percentile point of the price-duration curve (0, 2, ..., 100)
SCATTER_PER_YEAR = 600
WIND_SCATTER = 1500
MIX = ["wind", "solar", "nuclear", "hydro", "other", "net_import"]


def r(x, nd=2):
    """Round floats (or nested lists) for compact JSON; NaN -> None."""
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return [r(v, nd) for v in x]
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), nd)


def write(name: str, obj: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False))
    print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} kB)")


def export_impact() -> None:
    g = json.loads(GRID.read_text())
    sizes = g["config"]["data_center"]["sizes_mw"]
    keep = slice(None, None, PDC_STEP)
    runs = {}
    df = pd.DataFrame(g["scenarios"])
    for (y, ppa, v), s in df.groupby(["market_year", "ppa", "variant"]):
        s = s.set_index("size_mw").reindex(sizes)
        runs[f"{y}|{int(ppa)}|{v}"] = {
            "mean": r(s["mean_dprice"]), "cost": r(s["consumer_cost_meur"], 1),
            "ppa_mw": r(s["ppa_mw"], 0), "switching": [int(x) for x in s["switching_hours"]],
            "absorbed_gwh": r(s["absorbed_curtailed_gwh"], 0),
            "monthly": [r(m) for m in s["monthly_dprice"]], "hour": [r(h) for h in s["hour_of_day_dprice"]],
            "pdc": [r(p[keep], 1) for p in s["pdc_with"]],
        }
    baseline = {y: {"mean_price": b["mean_price"], "consumption_twh": b["consumption_twh"],
                    "coupled_share_pct": b["coupled_share_pct"], "pdc_without": r(b["pdc_without"][keep], 1)}
                for y, b in g["baseline"].items()}
    write("impact.json", {
        "sizes": sizes, "years": sorted({int(k.split("|")[0]) for k in runs}),
        "variants": ["central", "binned", "structural"],
        "pdc_pct": g["pdc_points_pct"][keep], "baseline": baseline, "runs": runs,
        "upper_bound_note_mw": 500,
    })


def export_history() -> None:
    tables = {y: pd.read_parquet(P.PROCESSED_DIR / f"hourly_{y}.parquet") for y in HIST_YEARS}
    months, mix = [], {k: [] for k in ["consumption", *MIX]}
    for y, t in tables.items():
        m = t[["consumption", *MIX]].resample("MS").sum(min_count=1) / 1000  # MWh -> GWh
        months += [f"{d:%Y-%m}" for d in m.index]
        for k in mix:
            mix[k] += r(m[k], 0)

    rng = np.random.default_rng(0)
    scatter, medians, stats = {}, {}, {}
    for y, t in tables.items():
        ok = t["price_eur_mwh"].notna() & t["residual_load"].notna() & (t["flag_price"].isin(P.FIT_FLAGS))
        d = t.loc[ok, ["residual_load", "price_eur_mwh"]]
        idx = rng.choice(len(d), size=min(SCATTER_PER_YEAR, len(d)), replace=False)
        s = d.iloc[np.sort(idx)]
        scatter[y] = [[r(a / 1000, 2), r(b, 1)] for a, b in zip(s["residual_load"], s["price_eur_mwh"])]
        b = (np.floor(d["residual_load"] / 500) + 0.5) * 500
        med = d.groupby(b)["price_eur_mwh"].agg(["median", "size"])
        med = med[med["size"] >= 30]
        medians[y] = [[r(x / 1000, 2), r(v, 1)] for x, v in zip(med.index, med["median"])]
        p = t["price_eur_mwh"]
        stats[y] = {"mean": r(p.mean()), "median": r(p.median()), "negative_hours": int((p < 0).sum()),
                    "max": r(p.max(), 0), "min": r(p.min(), 0)}

    cfg = K.load_cfg()
    ta = K.prepare(HIST_YEARS, cfg)
    share = pd.crosstab(ta["year"], ta["regime"], normalize="index").reindex(columns=K.REGIMES, fill_value=0) * 100
    regimes = {str(y): {k: r(v, 1) for k, v in row.items() if k != "unknown"} for y, row in share.iterrows()}
    write("history.json", {
        "months": months, "mix_gwh": mix, "mix_order": MIX,
        "scatter_rl_gw_price": {str(k): v for k, v in scatter.items()},
        "median_by_rl": {str(k): v for k, v in medians.items()},
        "price_stats": {str(k): v for k, v in stats.items()},
        "regime_share_pct": regimes,
        "regime_order": ["coupled", "coupled_se3", "import_constrained", "separated_other", "export_constrained"],
    })


def export_renewables() -> None:
    p = json.loads(WIND_MODEL.read_text())["logistic_pooled"]
    tables = {y: pd.read_parquet(P.PROCESSED_DIR / f"hourly_{y}.parquet") for y in HIST_YEARS}
    fit = pd.concat([tables[y] for y in (2023, 2024, 2025)])
    from src.models.renewables import logistic
    from src.fetch import weather
    m = (fit["flag_wind"].isin(P.FIT_FLAGS) & ~fit["curtailment"] & (fit["price_eur_mwh"] > 0)
         & fit["wind_cf"].notna() & fit["ws100_mean"].notna())
    d = fit.loc[m, ["ws100_mean", "wind_cf"]]
    idx = np.random.default_rng(0).choice(len(d), size=min(WIND_SCATTER, len(d)), replace=False)
    s = d.iloc[np.sort(idx)]
    v = np.round(np.arange(0, 16.01, 0.25), 2)
    monthly = {}
    for y, t in tables.items():
        g = t.groupby(t["month_local"])
        monthly[str(y)] = {"wind_cf": r(g["wind_cf"].mean(), 3), "solar_cf": r(g["solar_cf_fingrid"].mean(), 3)}
    last_cap = tables[2025]["wind_capacity"].dropna().iloc[-1]
    write("renewables.json", {
        "wind_scatter_ws_cf": [[r(a, 2), r(b, 3)] for a, b in zip(s["ws100_mean"], s["wind_cf"])],
        "wind_curve": {"params": {k: r(p[k], 4) for k in ("L", "k", "v0", "c")},
                       "ws": v.tolist(), "cf": r(logistic(v, p["L"], p["k"], p["v0"], p["c"]), 4),
                       "fit_years": [2023, 2024, 2025]},
        "wind_points": list(weather.load_locations()["wind"]),
        "monthly_cf": monthly,
        "calculator": {"default_capacity_mw": r(last_cap, 0), "capacity_note": "Fingrid forecast capacity (dataset 268), end of 2025"},
    })


def main() -> None:
    export_impact()
    export_history()
    export_renewables()


if __name__ == "__main__":
    main()

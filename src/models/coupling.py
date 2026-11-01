"""Coupling-aware price effect of extra Finnish load.

Finland is often price-coupled with Sweden. In those hours the price is set Nordic-wide, so the
plain slope (price.py) mixes Finnish load with Nordic conditions and overstates what extra
Finnish load does. This module:

1. Classifies every hour (thresholds in config/coupling.yaml):
     coupled             |FI - SE1| <= tolerance
     coupled_se3         FI > SE1 + tol but |FI - SE3| <= tol: the SE1 link is congested, the SE3
                         link is not, so extra load can still be imported (from SE3)
     import_constrained  FI > SE1 + tol, FI not below SE3 - tol, and imports from SE1 + SE3 at
                         >= near_capacity_share of their import capacity
     separated_other     FI > SE1 + tol otherwise: imports not near the nominal capacity (flow-based
                         limits, outages, physical vs commercial flows) or FI coupled only with EE
                         (FI + EE form a small area cut off from the Nordics). Price separation
                         itself shows a binding constraint, so these hours use the Finnish curve.
     export_constrained  FI < SE1 - tol
   Import capacity: Fingrid official day-ahead capacity; where none is published (SE1/SE3 after
   the 2024-10-29 flow-based go-live) the fixed fallback values in config/coupling.yaml.
2. Re-estimates the slope (price change per +100 MW residual load) with models fitted on each
   regime separately, and in coupled hours also with the SE1 price as a feature (controls for
   Nordic-wide conditions; holds SE1 fixed, so it is the low end for coupled hours).
3. Coupling-aware +300 MW estimate for one year (default 2024), per hour:
     coupled and coupled_se3 hours: the part that fits in the spare import capacity on borders whose price equals
       FI's (headroom) is met by imports and priced with the coupled-hour model; the rest uses the
       Finnish (import-constrained) model. Hours where 300 MW exceeds the headroom "switch".
     import_constrained / separated_other: the whole 300 MW on the Finnish model.
     export_constrained: the Finnish (export-constrained) model, capped at the gap to SE1
       (once FI reaches SE1's price it couples).
     unknown (a neighbour price missing): the full-sample model.
   Variants: binned and GBM model families with the coupled-hour slope for the imported part
   (central: GBM), and GBM with the SE1 price for the imported part (low end: with SE1 held
   fixed the coupled price barely moves, partly by construction, since coupled means FI ~ SE1).
   The old estimate (full-sample models, +300 MW everywhere) is the upper bound.

Usage:
    python -m src.models.coupling
    python -m src.models.coupling --check-thresholds 0.5 2 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.models import price as P

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "coupling.yaml"
NEIGHBOURS = ["se1", "se3", "ee"]
REGIMES = ["coupled", "coupled_se3", "import_constrained", "separated_other", "export_constrained", "unknown"]
COUPLED = ["coupled", "coupled_se3"]  # hours where extra load can be met by imports at the coupled price


def load_cfg(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_capacity(t: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """cap_imp_<nb>: official capacity up to official_until, else the configured fixed steps."""
    t = t.copy()
    for nb in NEIGHBOURS:
        fb = cfg["capacity_fallback"].get(nb.upper()) or {}
        official = t[f"cap_{nb}_fi"]
        if fb.get("official_until"):
            official = official.where(t.index <= pd.Timestamp(fb["official_until"]))
        fixed = pd.Series(np.nan, index=t.index)
        for step in fb.get("steps", []):
            fixed[t.index >= pd.Timestamp(step["from"])] = step["mw"]
        t[f"cap_imp_{nb}"] = official.where(official.notna(), fixed)
        t[f"cap_source_{nb}"] = np.where(official.notna(), "official", np.where(fixed.notna(), "fixed", "none"))
    return t


def spare(t: pd.DataFrame, nb: str) -> pd.Series:
    """Spare import capacity on the border with nb (MW, >= 0; 0 where capacity is unknown)."""
    return (t[f"cap_imp_{nb}"] - t[f"imp_{nb}"]).clip(lower=0).fillna(0)


def coupled_with(t: pd.DataFrame, cfg: dict, nb: str) -> pd.Series:
    """FI price coupled with nb's.

    Always: |FI - nb| <= price_tolerance_eur. In the flow-based period, if configured
    (flow_based_rule), also: |FI - nb| <= its tolerance_eur AND spare import capacity on that
    border >= its min_spare_mw (flow-based coupling leaves small spreads between coupled zones).
    """
    spread = (t["price_eur_mwh"] - t[f"price_{nb}"]).abs()
    out = spread <= cfg["price_tolerance_eur"]
    fb = cfg.get("flow_based_rule")
    if fb and fb.get("enabled"):
        period = t.index >= pd.Timestamp(fb["from"])
        out |= period & (spread <= fb["tolerance_eur"]) & (spare(t, nb) >= fb["min_spare_mw"])
    return out


def classify(t: pd.DataFrame, cfg: dict) -> pd.Series:
    """coupled (with SE1), coupled_se3 (with SE3 but not SE1), import_constrained,
    separated_other, export_constrained, unknown. See the module doc."""
    near = cfg["near_capacity_share"]
    d1 = t["price_eur_mwh"] - t["price_se1"]
    c1, c3 = coupled_with(t, cfg, "se1"), coupled_with(t, cfg, "se3")
    d3 = t["price_eur_mwh"] - t["price_se3"]
    near_cap = (t["imp_se1"] + t["imp_se3"]) >= near * (t["cap_imp_se1"] + t["cap_imp_se3"])
    r = pd.Series("unknown", index=t.index)
    above, below = (d1 > 0) & ~c1, (d1 < 0) & ~c1
    r[above] = "separated_other"  # incl. FI between SE1 and SE3 with imports near capacity
    r[above & ((d3 > 0) | c3) & near_cap] = "import_constrained"
    r[above & c3] = "coupled_se3"  # SE1 link congested, SE3 link not
    r[c1] = "coupled"
    r[below] = "export_constrained"
    r[d1.isna()] = "unknown"
    return r


def headroom(t: pd.DataFrame, cfg: dict) -> pd.Series:
    """Spare import capacity (MW) on borders whose price is coupled with FI's."""
    h = pd.Series(0.0, index=t.index)
    for nb in NEIGHBOURS:
        h += spare(t, nb).where(coupled_with(t, cfg, nb), 0.0)
    return h


def fit_regime_models(t: pd.DataFrame, mask: pd.Series, regime: pd.Series) -> dict:
    se1 = {"price_se1": 1}
    m_c = mask & (regime == "coupled")
    m_i = mask & (regime == "import_constrained")
    m_e = mask & (regime == "export_constrained")
    m_c1 = m_c & t["price_se1"].notna()
    return {
        "coupled": {"binned": P.BinnedCurve().fit(t[m_c]), "gbm": P.GBMPrice().fit(t[m_c]),
                    "gbm_se1": P.GBMPrice(se1).fit(t[m_c1])},
        "import_constrained": {"binned": P.BinnedCurve().fit(t[m_i]), "gbm": P.GBMPrice().fit(t[m_i])},
        "export_constrained": {"binned": P.BinnedCurve().fit(t[m_e]), "gbm": P.GBMPrice().fit(t[m_e])},
        "full": {"binned": P.BinnedCurve().fit(t[mask]), "gbm": P.GBMPrice().fit(t[mask])},
    }


def lomo_mae(make, t: pd.DataFrame, mask: pd.Series) -> float:
    """Leave-one-month-out MAE over the hours in mask."""
    err = []
    for y, m in t.loc[mask, ["year", "month_local"]].drop_duplicates().itertuples(index=False):
        test = mask & (t["year"] == y) & (t["month_local"] == m)
        model = make().fit(t[mask & ~test])
        err.append(np.abs(t.loc[test, "price_eur_mwh"].to_numpy() - model.predict(t[test])))
    return float(np.mean(np.concatenate(err)))


def delta(model, d: pd.DataFrame, mw) -> np.ndarray:
    return model.predict(d, mw) - model.predict(d)


def coupling_aware(d: pd.DataFrame, regime: pd.Series, head: pd.Series, coupled_model, imp_model, exp_model,
                   full_model, mw: float) -> pd.DataFrame:
    """Per-hour price change for +mw, split by regime (see module doc)."""
    out = pd.Series(np.nan, index=d.index)
    cov = pd.Series(0.0, index=d.index)
    c = regime.isin(COUPLED)
    if c.any():
        cov[c] = head[c].clip(upper=mw)
        rest = mw - cov[c]
        out[c] = delta(coupled_model, d[c], cov[c].to_numpy()) + delta(imp_model, d[c], rest.to_numpy())
    i = regime.isin(["import_constrained", "separated_other"])
    if i.any():
        out[i] = delta(imp_model, d[i], mw)
    e = regime == "export_constrained"
    if e.any():
        gap = (d.loc[e, "price_se1"] - d.loc[e, "price_eur_mwh"]).clip(lower=0)
        out[e] = np.minimum(delta(exp_model, d[e], mw), gap.to_numpy())
    u = regime == "unknown"
    if u.any():
        out[u] = delta(full_model, d[u], mw)
    return pd.DataFrame({"delta": out, "covered_by_imports": cov})


VARIANTS = {  # name: (model for the imported part in coupled hours, Finnish-curve model family)
    "binned": ("binned", "binned"),
    "gbm": ("gbm", "gbm"),  # central
    "structural": ("structural", "gbm"),  # imported part: FI load share x Finnish curve (low end)
}
CENTRAL = "gbm"


def estimate(d: pd.DataFrame, models: dict, mw: float, variants: dict = VARIANTS) -> tuple[pd.DataFrame, pd.Series]:
    """Per-hour price change for +mw: coupling-aware variants ("new ...") and full-sample ("old ...")."""
    est, covered = {}, None
    for name, (cm, fm) in variants.items():
        coupled_model = (ScaledDelta(models["import_constrained"][fm], d["fi_load_share"]) if cm == "structural"
                         else models["coupled"][cm])
        res = coupling_aware(d, d["regime"], d["headroom"], coupled_model, models["import_constrained"][fm],
                             models["export_constrained"][fm], models["full"][fm], mw)
        est[f"new {name}"] = res["delta"]
        covered = res["covered_by_imports"]
    for k, m in models["full"].items():
        est[f"old {k}"] = pd.Series(delta(m, d, mw), index=d.index)
    return pd.DataFrame(est), covered


def prepare(years: list[int], cfg: dict) -> pd.DataFrame:
    t = add_capacity(P.load(years), cfg)
    t["regime"] = classify(t, cfg)
    t["headroom"] = headroom(t, cfg)
    try:
        t["fi_load_share"] = load_share(years, t.index)
    except FileNotFoundError:  # ENTSO-E load not fetched for some year: structural variant unavailable
        t["fi_load_share"] = np.nan
    return t


class ScaledDelta:
    """Price change = share * (change on base model): the coupled-area version of the Finnish curve."""

    def __init__(self, base, share: pd.Series):
        self.base, self.share = base, share

    def predict(self, d: pd.DataFrame, extra_mw=0.0) -> np.ndarray:
        p0 = self.base.predict(d)
        if np.all(np.asarray(extra_mw) == 0):
            return p0
        return p0 + self.share.reindex(d.index).to_numpy() * (self.base.predict(d, extra_mw) - p0)


def load_share(years: list[int], index: pd.DatetimeIndex) -> pd.Series:
    """Finland's share of FI + SE1 + SE3 actual load (ENTSO-E), per hour."""
    from src.fetch import entsoe as E
    load = {}
    for a in ("FI", "SE1", "SE3"):
        parts = [pd.read_parquet(E.hourly_path(f"{y}-01-01", f"{y + 1}-01-01", a, "load"))["value"] for y in years]
        load[a] = pd.concat(parts).reindex(index)
    return load["FI"] / (load["FI"] + load["SE1"] + load["SE3"])


def structural_check(fit_years: list[int], years: list[int]) -> tuple[pd.DataFrame, pd.Series]:
    """Compare the coupled-hour slope with FI load share x Finnish (import-constrained) slope."""
    cfg = load_cfg()
    t = prepare(sorted(set(fit_years) | set(years)), cfg)
    t["fi_load_share"] = load_share(sorted(set(fit_years) | set(years)), t.index)
    fit = t[t["year"].isin(fit_years)]
    mask = P.fit_mask(fit)
    models = fit_regime_models(fit, mask, fit["regime"])
    rows = {}
    c = mask & fit["regime"].isin(COUPLED) & fit["fi_load_share"].notna()
    ic = fit[mask & (fit["regime"] == "import_constrained")]
    # (a) the import-constrained slope as measured in import-constrained hours (regime average)
    ic_avg = {fam: float(delta(models["import_constrained"][fam], ic, P.SLOPE_MW).mean()) for fam in ("binned", "gbm")}
    for label, sel in [("all fit years", c)] + [(y, c & (fit["year"] == y)) for y in fit_years]:
        d = fit[sel]
        r = {"coupled hours": int(sel.sum()), "FI load share": float(d["fi_load_share"].mean())}
        for fam in ("binned", "gbm"):
            r[f"{fam}: coupled-hour slope"] = float(delta(models["coupled"][fam], d, P.SLOPE_MW).mean())
            r[f"{fam}: share x import-constr. slope"] = float(d["fi_load_share"].mean() * ic_avg[fam])
            # (b) the Finnish curve's slope at the coupled hour's own conditions (extrapolates the
            # import-constrained model to lower loads)
            fin = delta(models["import_constrained"][fam], d, P.SLOPE_MW)
            r[f"{fam}: share x Finnish curve at the hour"] = float((d["fi_load_share"].to_numpy() * fin).mean())
        rows[label] = r
    print("import-constrained slope (regime average, per +100 MW):", {k: round(v, 3) for k, v in ic_avg.items()})
    return pd.DataFrame(rows).T, t["fi_load_share"]


def threshold_check(fit_years: list[int], preview_years: list[int], tolerances: list[float], mw: float) -> pd.DataFrame:
    """Regime shares and the +mw estimate for each plain coupling threshold (no flow-based rule)."""
    rows = {}
    for tol in tolerances:
        cfg = load_cfg()
        cfg["price_tolerance_eur"] = tol
        cfg["flow_based_rule"] = {"enabled": False}
        t = prepare(sorted(set(fit_years) | set(preview_years)), cfg)
        fit = t[t["year"].isin(fit_years)]
        models = fit_regime_models(fit, P.fit_mask(fit), fit["regime"])
        for y in preview_years:
            d = t[(t["year"] == y) & t[P.FEATURES].notna().all(axis=1)]
            est, _ = estimate(d, models, mw)
            c = d["regime"].isin(COUPLED)
            share = d["regime"].value_counts(normalize=True) * 100
            rows[(tol, y)] = {**{f"% {r}": share.get(r, 0.0) for r in REGIMES[:-1]},
                              "coupled h switching": int((c & (d["headroom"] < mw)).sum()),
                              **est.mean().to_dict()}
    return pd.DataFrame(rows).T.rename_axis(["tolerance", "year"])


# ---------------------------------------------------------------- report

def run(years: list[int], preview_years: list[int], mw: float) -> None:
    pd.set_option("display.width", 200, "display.max_columns", 40)
    cfg = load_cfg()
    tag = f"{years[0]}_{years[-1]}"
    t = prepare(years, cfg)

    # --- regime shares (all hours of all built years, for context)
    all_years = sorted({*years, *preview_years, *cfg.get("report_years", [])})
    ta = prepare(all_years, cfg)
    share = pd.crosstab(ta["year"], ta["regime"], normalize="index").reindex(columns=REGIMES, fill_value=0) * 100
    print("\n=== Regime share per year (% of hours) ===")
    print(share.round(1).to_string())
    src = {nb: ta.groupby("year")[f"cap_source_{nb}"].agg(lambda s: dict(s.value_counts())) for nb in NEIGHBOURS}
    print("\nImport capacity source (hours): ", {nb: v.to_dict() for nb, v in src.items()})
    ic = ta["regime"] == "import_constrained"
    sep = ta["regime"].isin(["import_constrained", "separated_other", "coupled_se3"])
    print(f"FI above SE1 (not coupled): {int(sep.sum())} h, of which imports near capacity: {int(ic.sum())} h "
          f"({ic.sum() / max(sep.sum(), 1) * 100:.0f}%)")
    over = {nb: int((ta[f"imp_{nb}"] > ta[f"cap_imp_{nb}"] + 50).sum()) for nb in NEIGHBOURS}
    print("hours with measured import > capacity + 50 MW:", over)

    # --- models
    mask = P.fit_mask(t)
    models = fit_regime_models(t, mask, t["regime"])
    print(f"\n=== Fit hours by regime ({tag}, clean hours) ===")
    print(t.loc[mask, "regime"].value_counts().to_dict())

    m_c = mask & (t["regime"] == "coupled") & t["price_se1"].notna()
    cv = {"gbm (no SE1 price)": lomo_mae(P.GBMPrice, t, m_c),
          "gbm + SE1 price": lomo_mae(lambda: P.GBMPrice({"price_se1": 1}), t, m_c)}
    print("\nCoupled hours, leave-one-month-out MAE (EUR/MWh):", {k: round(v, 2) for k, v in cv.items()},
          f"| mean price {t.loc[m_c, 'price_eur_mwh'].mean():.2f}")

    print(f"\n=== Slope: price change per +{P.SLOPE_MW:.0f} MW residual load (EUR/MWh), mean within regime ===")
    rows = {}
    for reg in ("coupled", "coupled_se3", "import_constrained", "export_constrained", "separated_other"):
        sel = mask & (t["regime"] == reg)
        if not sel.any():
            continue
        r = {f"full-sample {k}": float(delta(m, t[sel], P.SLOPE_MW).mean()) for k, m in models["full"].items()}
        own = models.get(reg, {})
        r |= {f"own-regime {k}": float(delta(m, t[sel], P.SLOPE_MW).mean()) for k, m in own.items()}
        r["hours"] = int(sel.sum())
        r["mean price"] = float(t.loc[sel, "price_eur_mwh"].mean())
        rows[reg] = r
    slope_tab = pd.DataFrame(rows).T
    print(slope_tab.round(3).to_string())

    # --- structural check on the coupled slope
    struct, _ = structural_check(years, years)
    print(f"\n=== Structural check: slope per +{P.SLOPE_MW:.0f} MW in coupled hours (EUR/MWh) ===")
    print(struct.round(3).to_string())

    # --- +mw estimate per preview year
    results, compare = {}, {}
    for y in preview_years:
        d = t[(t["year"] == y) & t[P.FEATURES].notna().all(axis=1)].copy()
        if d.empty:
            continue
        reg, head = d["regime"], d["headroom"]
        est, covered = estimate(d, models, mw)
        new_cols = [c for c in est if c.startswith("new")]
        c = reg.isin(COUPLED)
        switched = c & (head < mw)
        summary = est.mean()
        old_ub = float(summary[["old binned", "old gbm"]].max())
        central = float(summary[f"new {CENTRAL}"])
        lo, hi = float(summary[new_cols].min()), float(summary[new_cols].max())
        compare[y] = {"actual mean price": float(d["price_eur_mwh"].mean()), "old upper bound": old_ub,
                      "central": central, "range low": lo, "range high": hi,
                      "% coupled (incl. SE3-only)": float(c.mean() * 100),
                      "coupled h switching": int(switched.sum()),
                      "share of +MW met by imports in coupled h %": float(covered[c].mean() / mw * 100),
                      "median headroom in coupled h (MW)": float(head[c].median()),
                      "mean SE1+SE3+EE import cap (MW)": float((d["cap_imp_se1"].fillna(0) + d["cap_imp_se3"].fillna(0)
                                                               + d["cap_imp_ee"].fillna(0)).mean()),
                      **{f"mean {k}": float(v) for k, v in summary.items()}}
        month, hour = d["month_local"], d["hour_local"]
        cols = ["old binned", "old gbm", *new_cols]
        by_month, by_hour = est.groupby(month)[cols].mean(), est.groupby(hour)[cols].mean()
        for df, key in ((by_month, month), (by_hour, hour)):  # range = min/max of the variant means
            df["new min"], df["new max"] = df[new_cols].min(axis=1), df[new_cols].max(axis=1)
            df["coupled %"] = c.groupby(key).mean() * 100
        results[y] = {"by_month": by_month, "by_hour": by_hour, "by_regime": est.groupby(reg).mean()}
        print(f"\n=== +{mw:.0f} MW in every hour of {y}: central (GBM) +{central:.2f} EUR/MWh, "
              f"range +{lo:.2f} to +{hi:.2f}, old upper bound +{old_ub:.2f} ===")
        print(f"coupled hours {int(c.sum())} ({c.mean() * 100:.0f}%), switching {int(switched.sum())}; "
              f"regimes {reg.value_counts().to_dict()}")
        print("By regime:\n" + results[y]["by_regime"].round(2).to_string())
        print("By month:\n" + by_month.round(2).T.to_string())
        print("By hour of day (local):\n" + by_hour.round(2).T.to_string())
        print(f"Wrote {plot(by_month, by_hour, y, mw, tag)}")

    comp = pd.DataFrame(compare).T
    print(f"\n=== Year comparison: +{mw:.0f} MW flat (EUR/MWh) ===")
    print(comp.round(2).to_string())

    P.MODELS_DIR.mkdir(exist_ok=True)
    path = P.MODELS_DIR / f"coupling_{tag}.json"
    path.write_text(json.dumps({
        "fit_years": years, "preview_years": preview_years, "added_mw": mw, "config": cfg,
        "central_variant": CENTRAL, "variants": VARIANTS,
        "regime_share_pct": share.round(2).to_dict(orient="index"),
        "coupled_cv_mae": cv, "slope_per_100mw_by_regime": slope_tab.round(4).to_dict(orient="index"),
        "structural_check": struct.round(4).to_dict(orient="index"),
        "year_comparison": comp.round(4).to_dict(orient="index"),
        "by_year": {str(y): {k: v.round(3).to_dict(orient="index") for k, v in r.items()} for y, r in results.items()},
    }, indent=2, default=str))
    print(f"\nWrote {path}")


def plot(by_month: pd.DataFrame, by_hour: pd.DataFrame, year: int, mw: float, tag: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True, facecolor=P.C["surface"])
    for ax, df, xlabel, ticks in ((axes[0], by_month, "Month", list(range(1, 13))),
                                  (axes[1], by_hour, "Hour of day (local time)", list(range(0, 24, 3)))):
        P._style(ax)
        x = df.index.to_numpy()
        ax.fill_between(x, df["new min"], df["new max"], color=P.C["binned"], alpha=0.18, lw=0,
                        label="Coupling-aware, range between models")
        ax.plot(x, df["new gbm"], color=P.C["binned"], lw=2.2, marker="o", ms=4,
                label="Coupling-aware estimate (central, GBM)")
        ax.plot(x, df[["old binned", "old gbm"]].max(axis=1), color=P.C["gbm"], lw=2, ls="--",
                label="Old upper bound (no coupling; max of binned and GBM)")
        ax.set_xticks(ticks)
        if xlabel == "Month":
            ax.set_xticklabels(pd.to_datetime([f"2024-{m:02d}-01" for m in ticks]).strftime("%b"))
        ax.set_xlabel(xlabel, color=P.C["muted"])
    top = max(by_month[["old binned", "old gbm", "new max"]].to_numpy().max(),
              by_hour[["old binned", "old gbm", "new max"]].to_numpy().max())
    axes[0].set_ylim(0, top * 1.15)
    axes[0].set_ylabel(f"Price change from +{mw:.0f} MW (EUR/MWh)", color=P.C["muted"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=3,
               labelcolor=P.C["text"])
    fig.suptitle(f"Finland {year}: price effect of +{mw:.0f} MW flat load, with and without price coupling",
                 x=0.01, ha="left", fontsize=12.5, color=P.C["text"])
    fig.text(0.01, 0.005, "Coupled hours: the share that fits in spare import capacity is met by imports. "
             "Statistical estimate: imports, hydro and nuclear otherwise held fixed. Sources: ENTSO-E; Fingrid (CC BY 4.0).",
             fontsize=8, color=P.C["muted"])
    fig.tight_layout(rect=(0, 0.03, 1, 0.88))
    return P._save(fig, f"coupling_plus{int(mw)}mw_{year}.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", type=int, nargs="+", default=P.DEFAULT_YEARS, help="fit years")
    p.add_argument("--preview-years", type=int, nargs="+", default=[2023, 2024, 2025])
    p.add_argument("--mw", type=float, default=P.PREVIEW_MW)
    p.add_argument("--check-thresholds", type=float, nargs="+",
                   help="instead of the report: regime shares and +MW estimate per plain threshold, e.g. 0.5 2 5")
    p.add_argument("--check-years", type=int, nargs="+", default=[2023, 2024, 2025])
    args = p.parse_args()
    if args.check_thresholds:
        pd.set_option("display.width", 220, "display.max_columns", 40)
        res = threshold_check(sorted(args.years), args.check_years, args.check_thresholds, args.mw)
        print(f"\n=== Threshold check: regime shares (% of hours) and +{args.mw:.0f} MW mean price change (EUR/MWh) ===")
        print(res.round(2).to_string())
        return
    run(sorted(args.years), sorted(args.preview_years), args.mw)

if __name__ == "__main__":
    main()

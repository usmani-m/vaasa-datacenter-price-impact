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


def classify(t: pd.DataFrame, cfg: dict) -> pd.Series:
    """coupled (with SE1), coupled_se3 (with SE3 but not SE1), import_constrained,
    separated_other, export_constrained, unknown. See the module doc."""
    tol, near = cfg["price_tolerance_eur"], cfg["near_capacity_share"]
    d1 = t["price_eur_mwh"] - t["price_se1"]
    d3 = t["price_eur_mwh"] - t["price_se3"]
    cap_sw = t["cap_imp_se1"] + t["cap_imp_se3"]
    imp_sw = t["imp_se1"] + t["imp_se3"]
    near_cap = imp_sw >= near * cap_sw
    r = pd.Series("unknown", index=t.index)
    above = d1 > tol
    r[above] = "separated_other"  # incl. FI between SE1 and SE3 with imports near capacity
    r[above & (d3 > -tol) & near_cap] = "import_constrained"
    r[above & (d3.abs() <= tol)] = "coupled_se3"  # SE1 link congested, SE3 link not
    r[d1.abs() <= tol] = "coupled"
    r[d1 < -tol] = "export_constrained"
    r[d1.isna()] = "unknown"
    return r


def headroom(t: pd.DataFrame, cfg: dict) -> pd.Series:
    """Spare import capacity (MW) on borders whose price is within tolerance of FI's."""
    tol = cfg["price_tolerance_eur"]
    h = pd.Series(0.0, index=t.index)
    for nb in NEIGHBOURS:
        same = (t["price_eur_mwh"] - t[f"price_{nb}"]).abs() <= tol
        spare = (t[f"cap_imp_{nb}"] - t[f"imp_{nb}"]).clip(lower=0).fillna(0)
        h += spare.where(same, 0.0)
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


# ---------------------------------------------------------------- report

def run(years: list[int], preview_year: int, mw: float) -> None:
    pd.set_option("display.width", 200, "display.max_columns", 40)
    cfg = load_cfg()
    tag = f"{years[0]}_{years[-1]}"
    t = add_capacity(P.load(years), cfg)
    t["regime"] = classify(t, cfg)
    t["headroom"] = headroom(t, cfg)

    # --- regime shares (all hours of all built years, for context)
    all_years = sorted({*years, preview_year, *cfg.get("report_years", [])})
    ta = add_capacity(P.load(all_years), cfg)
    ta["regime"] = classify(ta, cfg)
    share = pd.crosstab(ta["year"], ta["regime"], normalize="index").reindex(columns=REGIMES, fill_value=0) * 100
    print("\n=== Regime share per year (% of hours) ===")
    print(share.round(1).to_string())
    src = {nb: ta.groupby("year")[f"cap_source_{nb}"].agg(lambda s: dict(s.value_counts())) for nb in NEIGHBOURS}
    print("\nImport capacity source (hours): ", {nb: v.to_dict() for nb, v in src.items()})
    ic = ta["regime"] == "import_constrained"
    sep = (ta["price_eur_mwh"] - ta["price_se1"]) > cfg["price_tolerance_eur"]
    print(f"FI above SE1 (> tol): {int(sep.sum())} h, of which imports near capacity: {int(ic.sum())} h "
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

    # --- +mw preview
    d = t[(t["year"] == preview_year) & t[P.FEATURES].notna().all(axis=1)].copy()
    reg, head = d["regime"], d["headroom"]
    variants = {
        "binned": ("binned", "binned"),
        "gbm": ("gbm", "gbm"),
        "gbm_se1": ("gbm_se1", "gbm"),
    }
    est = {}
    for name, (cm, fm) in variants.items():
        res = coupling_aware(d, reg, head, models["coupled"][cm], models["import_constrained"][fm],
                             models["export_constrained"][fm], models["full"][fm], mw)
        est[f"new {name}"] = res["delta"]
        covered = res["covered_by_imports"]
    for k, m in models["full"].items():
        est[f"old {k}"] = pd.Series(delta(m, d, mw), index=d.index)
    est = pd.DataFrame(est)
    new_cols = [c for c in est if c.startswith("new")]

    c = reg.isin(COUPLED)
    switched = c & (head < mw)
    print(f"\n=== +{mw:.0f} MW in every hour of {preview_year} ===")
    print(f"regimes: {reg.value_counts().to_dict()}")
    print(f"coupled hours: {int(c.sum())}; spare import capacity >= {mw:.0f} MW in {int((c & (head >= mw)).sum())}, "
          f"switch to constrained (headroom < {mw:.0f} MW): {int(switched.sum())} "
          f"({switched.sum() / max(c.sum(), 1) * 100:.1f}% of coupled hours); "
          f"mean share of the {mw:.0f} MW met by imports in coupled hours: {covered[c].mean() / mw * 100:.0f}%")
    c1 = reg == "coupled"
    print(f"  of which coupled with SE1 (original definition): {int(c1.sum())} h, switching {int((c1 & (head < mw)).sum())}; "
          f"coupled with SE3 only: {int((reg == 'coupled_se3').sum())} h, switching "
          f"{int(((reg == 'coupled_se3') & (head < mw)).sum())}")
    print(f"coupled-hour headroom (MW): median {head[c].median():.0f}, p10 {head[c].quantile(0.1):.0f}, "
          f"p90 {head[c].quantile(0.9):.0f}")
    summary = est.mean()
    print("\nMean price change (EUR/MWh):")
    print(summary.round(2).to_string())
    actual = d["price_eur_mwh"].mean()
    old_ub = float(summary[["old binned", "old gbm"]].max())
    central = float(summary["new gbm"])
    lo, hi = float(summary[new_cols].min()), float(summary[new_cols].max())
    print(f"\nThree numbers for {preview_year} (actual mean price {actual:.2f} EUR/MWh):")
    print(f"  old upper bound (full-sample models, max of binned/GBM): +{old_ub:.2f} EUR/MWh")
    print(f"  new coupling-aware estimate (GBM, coupled-hour slope for imported part): +{central:.2f} EUR/MWh")
    print(f"  range between coupling-aware models: +{lo:.2f} to +{hi:.2f} EUR/MWh")
    by_reg = est.groupby(reg).mean()[["old gbm", "old binned", *new_cols]]
    print("\nBy regime:")
    print(by_reg.round(2).to_string())

    month = d["month_local"]
    hour = d["hour_local"]
    cols = ["old binned", "old gbm", *new_cols]
    by_month, by_hour = est.groupby(month)[cols].mean(), est.groupby(hour)[cols].mean()
    for df in (by_month, by_hour):  # range between models = min/max of the variant means
        df["new min"], df["new max"] = df[new_cols].min(axis=1), df[new_cols].max(axis=1)
    by_month["coupled %"] = c.groupby(month).mean() * 100
    by_hour["coupled %"] = c.groupby(hour).mean() * 100
    print("\nBy month (EUR/MWh):")
    print(by_month.round(2).T.to_string())
    print("\nBy hour of day, local (EUR/MWh):")
    print(by_hour.round(2).T.to_string())

    P.MODELS_DIR.mkdir(exist_ok=True)
    path = P.MODELS_DIR / f"coupling_{tag}.json"
    path.write_text(json.dumps({
        "years": years, "preview_year": preview_year, "added_mw": mw, "config": cfg,
        "regime_share_pct": share.round(2).to_dict(orient="index"),
        "coupled_cv_mae": cv, "slope_per_100mw_by_regime": slope_tab.round(4).to_dict(orient="index"),
        "preview": {"actual_mean_price": actual, "old_upper_bound": old_ub, "new_central": central,
                    "new_range": [lo, hi], "means": summary.round(4).to_dict(),
                    "coupled_hours": int(c.sum()), "switched_hours": int(switched.sum()),
                    "by_month": by_month.round(3).to_dict(orient="index"),
                    "by_hour_local": by_hour.round(3).to_dict(orient="index")},
    }, indent=2, default=str))
    print(f"\nWrote {path}")
    print(f"Wrote {plot(by_month, by_hour, preview_year, mw, tag)}")


def plot(by_month: pd.DataFrame, by_hour: pd.DataFrame, year: int, mw: float, tag: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True, facecolor=P.C["surface"])
    for ax, df, xlabel, ticks in ((axes[0], by_month, "Month", list(range(1, 13))),
                                  (axes[1], by_hour, "Hour of day (local time)", list(range(0, 24, 3)))):
        P._style(ax)
        x = df.index.to_numpy()
        ax.fill_between(x, df["new min"], df["new max"], color=P.C["binned"], alpha=0.18, lw=0,
                        label="Coupling-aware, range between models")
        ax.plot(x, df["new gbm"], color=P.C["binned"], lw=2.2, marker="o", ms=4,
                label="Coupling-aware estimate (GBM)")
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
    p.add_argument("--preview-year", type=int, default=P.PREVIEW_YEAR)
    p.add_argument("--mw", type=float, default=P.PREVIEW_MW)
    args = p.parse_args()
    run(sorted(args.years), args.preview_year, args.mw)


if __name__ == "__main__":
    main()

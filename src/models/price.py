"""Price model: FI day-ahead price as a function of residual load (and other drivers).

Two models, fitted on 2023-2025 by default (2022 is excluded through crisis_2022 unless
--include-crisis; other years can be added with --years):

  binned : supply curve per year = median price per residual-load bin (BIN_MW wide, bins with
           fewer than MIN_BIN_HOURS dropped), made non-decreasing with isotonic regression and
           linearly interpolated (flat beyond the observed range). A level factor per year and
           month scales it: factor = sum(price) / sum(curve) over that month's hours.
  gbm    : HistGradientBoosting on residual load, net import, nuclear, hydro, temperature,
           hour, weekday, month and year, with price constrained to be non-decreasing in
           residual load (monotonic_cst), so adding load can never lower the price.

Fitting excludes hours where price or any feature is flagged suspect or missing.

Validation: leave-one-month-out over the year-months in the fit (train on all other months,
predict the held-out one). For the binned model the held-out month's level factor is unknown,
so it is the mean factor of the same calendar month in the other years (1.0 if none).

Main output: the slope, i.e. the price change per +100 MW of residual load, by month and by
hour of day, and a preview of adding a flat 300 MW to every hour of 2024. Everything else is
held fixed (net import, nuclear, hydro, ...): in reality imports and hydro respond to a higher
price, so this is a partial, statistical effect, not a market simulation.

Usage:
    python -m src.models.price
    python -m src.models.price --years 2019 2020 2021 2023 2024 2025
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"

DEFAULT_YEARS = [2023, 2024, 2025]
FIT_FLAGS = ["ok", "filled"]
FLAG_COLS = ["flag_price", "flag_residual_load", "flag_net_import", "flag_nuclear", "flag_hydro"]
FEATURES = ["residual_load", "net_import", "nuclear", "hydro", "temp_mean",
            "hour_local", "weekday_local", "month_local", "year"]
MONOTONIC = [1 if f == "residual_load" else 0 for f in FEATURES]
BIN_MW = 250.0
MIN_BIN_HOURS = 24
SLOPE_MW = 100.0
PREVIEW_MW = 300.0
PREVIEW_YEAR = 2024

C = {"actual": "#0b0b0b", "binned": "#2a78d6", "gbm": "#eb6834", "text": "#0b0b0b", "muted": "#52514e",
     "grid": "#e4e3df", "surface": "#fcfcfb"}


# ---------------------------------------------------------------- data

def load(years: list[int]) -> pd.DataFrame:
    t = pd.concat([pd.read_parquet(PROCESSED_DIR / f"hourly_{y}.parquet") for y in years])
    t["year"] = t.index.year
    return t


def fit_mask(t: pd.DataFrame, include_crisis: bool = False) -> pd.Series:
    m = t[FLAG_COLS].isin(FIT_FLAGS).all(axis=1) & t[FEATURES + ["price_eur_mwh"]].notna().all(axis=1)
    if not include_crisis:
        m &= ~t["crisis_2022"]
    return m


# ---------------------------------------------------------------- models

class BinnedCurve:
    name = "binned"

    def fit(self, d: pd.DataFrame):
        self.curves, self.factors = {}, {}
        for y, g in d.groupby("year"):
            b = (np.floor(g["residual_load"] / BIN_MW) + 0.5) * BIN_MW
            agg = g.groupby(b)["price_eur_mwh"].agg(["median", "size"])
            agg = agg[agg["size"] >= MIN_BIN_HOURS]
            if len(agg) < 2:  # too few hours for a curve; this year falls back to the pooled curve
                continue
            iso = IsotonicRegression(increasing=True).fit(agg.index.to_numpy(), agg["median"].to_numpy(),
                                                          sample_weight=agg["size"].to_numpy())
            self.curves[int(y)] = (agg.index.to_numpy(), iso.predict(agg.index.to_numpy()))
        base = self._curve(d["year"], d["residual_load"])
        for (y, m), idx in d.groupby(["year", "month_local"]).groups.items():
            den = base.loc[idx].sum()
            self.factors[(int(y), int(m))] = float(d.loc[idx, "price_eur_mwh"].sum() / den) if den > 1 else 1.0
        return self

    def _curve(self, year: pd.Series, rl: pd.Series) -> pd.Series:
        out = pd.Series(np.nan, index=rl.index)
        for y in year.unique():
            x, v = self.curves.get(int(y), self._pooled())
            sel = year == y
            out[sel] = np.interp(rl[sel].to_numpy(), x, v)
        return out

    def _pooled(self):
        """Fallback for a year without its own curve: mean of the fitted curves on a common grid."""
        grid = np.arange(0, 16000 + BIN_MW, BIN_MW)
        return grid, np.mean([np.interp(grid, x, v) for x, v in self.curves.values()], axis=0)

    def _factor(self, year: int, month: int) -> float:
        if (year, month) in self.factors:
            return self.factors[(year, month)]
        same = [f for (y, m), f in self.factors.items() if m == month]
        return float(np.mean(same)) if same else 1.0

    def predict(self, d: pd.DataFrame, extra_mw: float | np.ndarray = 0.0) -> np.ndarray:
        f = np.array([self._factor(int(y), int(m)) for y, m in zip(d["year"], d["month_local"])])
        return f * self._curve(d["year"], d["residual_load"] + extra_mw).to_numpy()

    def to_dict(self) -> dict:
        return {
            "bin_mw": BIN_MW, "min_bin_hours": MIN_BIN_HOURS,
            "curves": {str(y): {"residual_load_mw": x.round(1).tolist(), "price_eur_mwh": v.round(3).tolist()}
                       for y, (x, v) in self.curves.items()},
            "monthly_factors": {f"{y}-{m:02d}": round(f, 4) for (y, m), f in sorted(self.factors.items())},
        }


class GBMPrice:
    """extra_features: added to FEATURES with the given monotonic constraint each (e.g. SE1 price, +1)."""
    name = "gbm"

    def __init__(self, extra_features: dict[str, int] | None = None):
        self.features = FEATURES + list(extra_features or {})
        self.monotonic = MONOTONIC + list((extra_features or {}).values())

    def fit(self, d: pd.DataFrame):
        self.model = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
            l2_regularization=1.0, monotonic_cst=self.monotonic, random_state=0)
        self.model.fit(d[self.features], d["price_eur_mwh"])
        return self

    def predict(self, d: pd.DataFrame, extra_mw: float | np.ndarray = 0.0) -> np.ndarray:
        x = d[self.features].copy()
        x["residual_load"] = x["residual_load"] + extra_mw
        return self.model.predict(x)


MODELS = {"binned": BinnedCurve, "gbm": GBMPrice}


def leave_one_month_out(t: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Out-of-fold predictions for every fit hour, one fold per (year, month)."""
    oof = pd.DataFrame(np.nan, index=t.index, columns=list(MODELS))
    folds = t.loc[mask, ["year", "month_local"]].drop_duplicates().itertuples(index=False)
    for y, m in folds:
        test = mask & (t["year"] == y) & (t["month_local"] == m)
        train = mask & ~test
        for name, cls in MODELS.items():
            oof.loc[test, name] = cls().fit(t[train]).predict(t[test])
    return oof


# ---------------------------------------------------------------- outputs

def mae(a: pd.Series, b: pd.Series) -> float:
    return float(np.mean(np.abs(a - b)))


def slopes(models: dict, d: pd.DataFrame, mw: float = SLOPE_MW) -> pd.DataFrame:
    return pd.DataFrame({k: m.predict(d, mw) - m.predict(d) for k, m in models.items()}, index=d.index)


def _style(ax):
    ax.set_facecolor(C["surface"])
    ax.grid(axis="y", color=C["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["grid"])
    ax.tick_params(colors=C["muted"], labelsize=9)


def _save(fig, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, facecolor=C["surface"])
    plt.close(fig)
    return path


def plot_weeks(t: pd.DataFrame, oof: pd.DataFrame, weeks: list[tuple[str, str]], tag: str) -> Path:
    fig, axes = plt.subplots(len(weeks), 1, figsize=(11, 3.8 * len(weeks) + 0.6), facecolor=C["surface"])
    for ax, (label, start) in zip(np.atleast_1d(axes), weeks):
        _style(ax)
        a = pd.Timestamp(start, tz="Europe/Helsinki").tz_convert("UTC")
        w = (t.index >= a) & (t.index < a + pd.Timedelta(days=7))
        x = t.index[w].tz_convert("Europe/Helsinki")
        ax.plot(x, t.loc[w, "price_eur_mwh"], color=C["actual"], lw=2, label="Actual day-ahead price")
        ax.plot(x, oof.loc[w, "binned"], color=C["binned"], lw=1.6, label="Binned supply curve")
        ax.plot(x, oof.loc[w, "gbm"], color=C["gbm"], lw=1.6, label="Gradient boosting")
        e = {k: mae(t.loc[w, "price_eur_mwh"], oof.loc[w, k]) for k in MODELS}
        ax.set_title(f"{label}: {a.tz_convert('Europe/Helsinki'):%a %d %b} - "
                     f"{(a + pd.Timedelta(days=6)).tz_convert('Europe/Helsinki'):%a %d %b %Y}   "
                     f"(MAE binned {e['binned']:.1f}, GBM {e['gbm']:.1f} EUR/MWh)",
                     loc="left", fontsize=10.5, color=C["text"])
        ax.axhline(0, color=C["muted"], lw=0.8)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a %d", tz="Europe/Helsinki"))
        ax.set_ylabel("EUR/MWh", color=C["muted"])
    np.atleast_1d(axes)[0].legend(loc="upper left", bbox_to_anchor=(0, 1.25), ncol=3, frameon=False, fontsize=9,
                                  labelcolor=C["text"])
    fig.suptitle("FI day-ahead price: actual vs out-of-sample predictions (local time)", x=0.06, ha="left",
                 fontsize=13, color=C["text"], y=0.995)
    fig.text(0.06, 0.005, "Leave-one-month-out: each prediction comes from a model that did not see that month. "
             "Gaps: hours excluded as suspect or missing. Sources: ENTSO-E; Fingrid (CC BY 4.0); Open-Meteo ERA5 "
             "(CC BY 4.0).", fontsize=8, color=C["muted"])
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return _save(fig, f"price_weeks_{tag}.png")


def plot_slopes(by_month: pd.DataFrame, by_hour: pd.DataFrame, tag: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, facecolor=C["surface"])
    for ax, df, xlabel, ticks in ((axes[0], by_month, "Month", list(range(1, 13))),
                                  (axes[1], by_hour, "Hour of day (local time)", list(range(0, 24, 3)))):
        _style(ax)
        for k, label in (("binned", "Binned supply curve"), ("gbm", "Gradient boosting")):
            ax.plot(df.index, df[k], color=C[k], lw=2, marker="o", ms=4, label=label)
        ax.set_xticks(ticks)
        if xlabel == "Month":
            ax.set_xticklabels(pd.to_datetime([f"2024-{m:02d}-01" for m in ticks]).strftime("%b"))
        ax.set_xlabel(xlabel, color=C["muted"])
    top = max(by_month.to_numpy().max(), by_hour.to_numpy().max())
    axes[0].set_ylim(0, top * 1.12)
    axes[0].set_ylabel(f"Price change per +{SLOPE_MW:.0f} MW (EUR/MWh)", color=C["muted"])
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", labelcolor=C["text"])
    fig.suptitle(f"How much the FI price rises per +{SLOPE_MW:.0f} MW of residual load ({tag.replace('_', '-')} fit)",
                 x=0.01, ha="left", fontsize=12.5, color=C["text"])
    fig.text(0.01, 0.005, "Mean over all fit hours. Other drivers (imports, hydro, nuclear) held fixed: a partial "
             "statistical effect, not a market simulation.", fontsize=8, color=C["muted"])
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return _save(fig, f"price_slope_{tag}.png")


# ---------------------------------------------------------------- main

def run(years: list[int], include_crisis: bool) -> None:
    pd.set_option("display.width", 180)
    tag = f"{years[0]}_{years[-1]}"
    t = load(years)
    mask = fit_mask(t, include_crisis)
    print(f"\n=== Price model, years {years}{' incl. 2022' if include_crisis else ''}: "
          f"{int(mask.sum())} fit hours, {int((~mask).sum())} excluded ===")

    oof = leave_one_month_out(t, mask)
    y = t.loc[mask, "price_eur_mwh"]
    res = {k: {"MAE": mae(y, oof.loc[mask, k]), "RMSE": float(np.sqrt(np.mean((y - oof.loc[mask, k]) ** 2))),
               "median abs error": float(np.median(np.abs(y - oof.loc[mask, k])))} for k in MODELS}
    res["mean price"] = {"MAE": float(y.mean())}
    print("\nLeave-one-month-out (EUR/MWh):")
    print(pd.DataFrame(res).T.round(2).to_string())

    g = pd.DataFrame({"y": y, **{k: oof.loc[mask, k] for k in MODELS}, "month": t.loc[mask, "month_local"],
                      "year": t.loc[mask, "year"]})
    by_m = pd.DataFrame({f"MAE {k}": g.groupby("month").apply(lambda d, k=k: mae(d["y"], d[k])) for k in MODELS})
    by_m["mean price"] = g.groupby("month")["y"].mean()
    by_y = pd.DataFrame({f"MAE {k}": g.groupby("year").apply(lambda d, k=k: mae(d["y"], d[k])) for k in MODELS})
    by_y["mean price"] = g.groupby("year")["y"].mean()
    print("\nMAE by month (all years pooled):")
    print(by_m.round(2).T.to_string())
    print("\nMAE by year:")
    print(by_y.round(2).to_string())

    # full fits for slopes and the preview
    models = {k: cls().fit(t[mask]) for k, cls in MODELS.items()}
    sl = slopes(models, t[mask])
    hour = t.loc[mask, "hour_local"]
    slope_month = sl.groupby(t.loc[mask, "month_local"]).mean()
    slope_hour = sl.groupby(hour).mean()
    print(f"\n=== Slope: price change per +{SLOPE_MW:.0f} MW residual load (EUR/MWh), mean over fit hours ===")
    print("overall:", {k: round(float(v), 3) for k, v in sl.mean().items()},
          "| by year:", sl.groupby(t.loc[mask, "year"]).mean().round(3).to_dict(orient="index"))
    print("by month:\n" + slope_month.round(3).T.to_string())
    print("by hour of day (local):\n" + slope_hour.round(3).T.to_string())
    print("share of hours with slope exactly 0:", (sl == 0).mean().round(3).to_dict())

    preview = {}
    if PREVIEW_YEAR in years:
        p = t[(t["year"] == PREVIEW_YEAR) & t[FEATURES].notna().all(axis=1)]
        w = p["consumption"] / p["consumption"].sum()
        for k, m in models.items():
            base, plus = m.predict(p), m.predict(p, PREVIEW_MW)
            d = plus - base
            preview[k] = {"hours": len(p), "mean price modelled": float(base.mean()),
                          "mean change": float(d.mean()), "consumption-weighted change": float((d * w).sum()),
                          "change % of modelled mean": float(d.mean() / base.mean() * 100),
                          "p10 change": float(np.percentile(d, 10)), "p90 change": float(np.percentile(d, 90))}
        print(f"\n=== Preview: +{PREVIEW_MW:.0f} MW flat in every hour of {PREVIEW_YEAR} (EUR/MWh) ===")
        print(f"actual mean price {PREVIEW_YEAR}: {t.loc[t['year'] == PREVIEW_YEAR, 'price_eur_mwh'].mean():.2f}")
        print(pd.DataFrame(preview).T.round(2).to_string())

    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"price_{tag}.json"
    path.write_text(json.dumps({
        "years": years, "include_crisis": include_crisis, "fit_hours": int(mask.sum()),
        "features": FEATURES, "monotonic_cst": MONOTONIC,
        "binned": models["binned"].to_dict(),
        "gbm": {k: v for k, v in models["gbm"].model.get_params().items() if k != "monotonic_cst"}
               | {"note": "not saved; refit with src.models.price (deterministic, random_state=0)"},
        "cv_leave_one_month_out": res,
        "mae_by_month": by_m.round(3).to_dict(orient="index"), "mae_by_year": by_y.round(3).to_dict(orient="index"),
        "slope_per_100mw": {"overall": sl.mean().round(4).to_dict(),
                            "by_month": slope_month.round(4).to_dict(orient="index"),
                            "by_hour_local": slope_hour.round(4).to_dict(orient="index")},
        "preview_plus_300mw": {"year": PREVIEW_YEAR, **preview},
    }, indent=2, default=float))
    print(f"\nWrote {path}")

    weeks = [("Winter week", f"{PREVIEW_YEAR}-01-08"), ("Summer week", f"{PREVIEW_YEAR}-07-08")]
    if PREVIEW_YEAR not in years:
        weeks = [("Winter week", f"{years[-1]}-01-13"), ("Summer week", f"{years[-1]}-07-07")]
    print(f"Wrote {plot_weeks(t, oof, weeks, tag)}")
    print(f"Wrote {plot_slopes(slope_month, slope_hour, tag)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS)
    p.add_argument("--include-crisis", action="store_true", help="keep 2022 hours (crisis_2022) in the fit")
    args = p.parse_args()
    run(sorted(args.years), args.include_crisis)


if __name__ == "__main__":
    main()

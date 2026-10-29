"""Checks on FI day-ahead prices in the hourly tables, plus cross-checks with earlier findings.

- Per year: rows, missing hours, mean / median / min / max, negative-price hours, correlation of
  price with consumption and with residual load; mean price by month.
- curtailment hours (config/data_issues.yaml): price at those hours.
- unexplained wind ramp jumps (config/data_issues.yaml): prices around those hours.
- Figure: price vs residual load, one panel per year (outputs/figures/price_vs_residual_load.png).

Usage:
    python -m src.process.check_prices --years 2019 2020 2021 2022 2023 2024 2025
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"
ISSUES_PATH = REPO_ROOT / "config" / "data_issues.yaml"

# Reference palette (dataviz skill), light mode. Years are identified by panel, not by colour.
C = {"year": "#2a78d6", "crisis": "#eb6834", "context": "#d6d5d0", "text": "#0b0b0b", "muted": "#52514e",
     "grid": "#e4e3df", "surface": "#fcfcfb"}


def per_year(tables: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = {}
    for y, t in tables.items():
        p = t["price_eur_mwh"]
        rows[y] = {
            "rows": len(t), "missing h": int(p.isna().sum()), "filled h": int((t["flag_price"] == "filled").sum()),
            "mean": p.mean(), "median": p.median(), "min": p.min(), "max": p.max(),
            "negative h": int((p < 0).sum()), "zero h": int((p == 0).sum()),
            "corr consumption": p.corr(t["consumption"]), "corr residual load": p.corr(t["residual_load"]),
        }
    return pd.DataFrame(rows).T


def monthly(tables: dict[int, pd.DataFrame]) -> pd.DataFrame:
    m = {y: t["price_eur_mwh"].groupby(t["month_local"]).mean() for y, t in tables.items()}
    out = pd.DataFrame(m).T
    out.columns = pd.to_datetime(out.columns, format="%m").strftime("%b")
    return out


def curtailment_check(tables: dict[int, pd.DataFrame]) -> pd.DataFrame:
    t = pd.concat(tables.values())
    c = t[t["curtailment"]]
    return c[["price_eur_mwh", "wind", "wind_cf", "ws100_mean", "solar", "consumption"]]


def ramp_check(tables: dict[int, pd.DataFrame], issues: list[dict], window_h: int = 2) -> dict[str, pd.DataFrame]:
    t = pd.concat(tables.values())
    out = {}
    for item in issues:
        if "time" not in item:
            continue
        ts = pd.Timestamp(item["time"])
        w = t.loc[ts - pd.Timedelta(hours=window_h): ts + pd.Timedelta(hours=window_h),
                  ["price_eur_mwh", "wind", "wind_cf", "ws100_mean", "consumption", "net_import"]]
        if len(w):
            out[str(ts)] = w
    return out


def plot(tables: dict[int, pd.DataFrame]) -> Path:
    years = list(tables)
    t = pd.concat(tables.values())
    ok = t["price_eur_mwh"].notna() & t["residual_load"].notna()
    x_all, y_all = t.loc[ok, "residual_load"] / 1000, t.loc[ok, "price_eur_mwh"]
    ncols = 4
    nrows = int(np.ceil(len(years) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.6 * nrows + 0.8), sharex=True, sharey=True,
                             facecolor=C["surface"])
    axes = axes.ravel()
    ylim = (max(-60, y_all.min() - 5), min(800, y_all.quantile(0.9995) + 20))
    for ax, y in zip(axes, years):
        d = tables[y]
        m = d["price_eur_mwh"].notna() & d["residual_load"].notna()
        ax.set_facecolor(C["surface"])
        ax.scatter(x_all, y_all, s=2, color=C["context"], lw=0, rasterized=True)
        color = C["crisis"] if y == 2022 else C["year"]
        ax.scatter(d.loc[m, "residual_load"] / 1000, d.loc[m, "price_eur_mwh"], s=3, alpha=0.35, color=color, lw=0,
                   rasterized=True)
        med = d.loc[m, "price_eur_mwh"].median()
        label = f"{y}" + ("  (energy crisis)" if y == 2022 else "")
        ax.set_title(f"{label}   median {med:.0f} EUR/MWh", loc="left", fontsize=10, color=C["text"])
        ax.axhline(0, color=C["muted"], lw=0.8)
        ax.grid(color=C["grid"], lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(C["grid"])
        ax.tick_params(colors=C["muted"], labelsize=8.5)
        ax.set_ylim(*ylim)
    for ax in axes[len(years):]:
        ax.set_visible(False)
    for ax in axes[::ncols]:
        ax.set_ylabel("Day-ahead price (EUR/MWh)", color=C["muted"], fontsize=9)
    for ax in axes[max(0, len(years) - ncols):len(years)]:
        ax.set_xlabel("Residual load = consumption - wind - solar (GW)", color=C["muted"], fontsize=9)
        ax.xaxis.set_tick_params(labelbottom=True)
    above, below = int((y_all > ylim[1]).sum()), int((y_all < ylim[0]).sum())
    fig.suptitle("Finland: hourly day-ahead price vs residual load, by year", x=0.01, ha="left", fontsize=13,
                 color=C["text"])
    fig.text(0.01, 0.005, f"Grey: all other years. y-axis shown from {ylim[0]:.0f} to {ylim[1]:.0f} EUR/MWh "
             f"({above} hours above and {below} below not shown). "
             "Sources: ENTSO-E Transparency Platform; Fingrid / data.fingrid.fi (CC BY 4.0).",
             fontsize=8, color=C["muted"])
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "price_vs_residual_load.png"
    fig.savefig(path, dpi=150, facecolor=C["surface"])
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", type=int, nargs="+", required=True)
    years = sorted(p.parse_args().years)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    tables = {y: pd.read_parquet(PROCESSED_DIR / f"hourly_{y}.parquet") for y in years}

    print("\n=== FI day-ahead price per year (EUR/MWh) ===")
    print(per_year(tables).round(2).to_string())
    print("\n=== Mean price by month (EUR/MWh, local month) ===")
    print(monthly(tables).round(1).to_string())

    c = curtailment_check(tables)
    print(f"\n=== curtailment hours ({len(c)}) ===")
    print(c.round(3).to_string())
    print(f"price <= 0 in {int((c['price_eur_mwh'] <= 0).sum())} of {len(c)} hours; "
          f"max {c['price_eur_mwh'].max():.2f} EUR/MWh")

    with open(ISSUES_PATH, encoding="utf-8") as f:
        unexplained = (yaml.safe_load(f) or {}).get("unexplained", [])
    print("\n=== Unexplained wind ramp jumps: prices +- 2 h ===")
    for ts, w in ramp_check(tables, unexplained).items():
        print(f"\n-- {ts}")
        print(w.round(3).to_string())
    print(f"\nWrote {plot(tables)}")


if __name__ == "__main__":
    main()

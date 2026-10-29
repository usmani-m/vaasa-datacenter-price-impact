"""Quantify the 09-12 UTC artefact in Fingrid solar (248) and preview a correction.

REPORT ONLY: nothing here changes the hourly table. The correction is a proposal until the
user confirms it.

Evidence that the artefact is not physical (2024): in May-Aug, Fingrid's day-ahead forecast
(247), the metered small-scale surplus (362, Datahub) and ERA5 irradiance all peak smoothly
around 10-11 UTC, while 248 peaks at 09 UTC and dips at 10-11 UTC. Fingrid's own day-ahead
series has no dip, so the artefact sits in how the 15-min-updated series (248) is assembled.

Proposed correction (correct_248): for each day, keep 248's total over the window 09-12 UTC and
redistribute it over those four hours in proportion to 247's values for the same hours. Daily
totals are unchanged exactly; hours outside the window are untouched; within the window the
shape follows Fingrid's own day-ahead forecast for that day.

Usage:
    python -m src.process.solar_artefact --year 2024
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.fetch import fingrid

WINDOW_UTC = [9, 10, 11, 12]
SUMMER = [5, 6, 7, 8]
REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4",
     "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df", "surface": "#fcfcfb"}


def correct_248(solar: pd.Series, dayahead: pd.Series, window: list[int] = WINDOW_UTC) -> pd.Series:
    """Redistribute each day's 248 window total using 247's within-window shares (see module doc).

    A day is corrected only if both series have all window hours and 247's window total is > 0;
    otherwise 248 is left unchanged for that day.
    """
    out = solar.copy()
    win = solar.index.hour.isin(window)
    a, b = solar[win], dayahead.reindex(solar.index)[win]
    day = a.index.date
    a_tot, b_tot = a.groupby(day).transform("sum"), b.groupby(day).transform("sum")
    complete = (a.notna().groupby(day).transform("sum") == len(window)) & \
               (b.notna().groupby(day).transform("sum") == len(window)) & (b_tot > 0)
    out[win] = (a_tot * b / b_tot).where(complete, a)
    return out


def has_artefact(solar: pd.Series, dayahead: pd.Series) -> bool:
    """True if the year's 248 shows the timing artefact.

    Rule: over May-Aug hours where both series exist, the mean daily shape of 248 peaks at an
    earlier UTC hour than 247's. 2019-2025 check: present in 2019, 2020, 2021 and 2024 (248 peaks
    at 09, 247 at 10 UTC); absent in 2022, 2023 and 2025 (both peak at 10 UTC).
    """
    both = solar.notna() & dayahead.reindex(solar.index).notna()
    summer = both & solar.index.month.isin(SUMMER)
    if summer.sum() < 24 * 30:
        return False
    a = daily_shape(solar[both], SUMMER)
    b = daily_shape(dayahead.reindex(solar.index)[both], SUMMER)
    return int(a.idxmax()) < int(b.idxmax())


def daily_shape(s: pd.Series, months: list[int]) -> pd.Series:
    """Mean share of the daily total per UTC hour (%), over days in the given months."""
    x = s[s.index.month.isin(months)]
    share = x / x.groupby(x.index.date).transform("sum")
    return share.groupby(share.index.hour).mean() * 100


def run(year: int) -> None:
    pd.set_option("display.width", 160)
    start, end = f"{year}-01-01", f"{year + 1}-01-01"
    fg = lambda n: pd.read_parquet(fingrid.hourly_path(n, start, end))["value"]
    t = pd.read_parquet(REPO_ROOT / "data" / "processed" / f"hourly_{year}.parquet")
    s248, s247, s362 = fg("solar"), fg("solar_dayahead"), fg("solar_surplus_small")
    corrected = correct_248(s248, s247)

    print(f"\n=== Annual totals {year} (TWh) ===")
    print(f"248 (15-min-updated forecast): {s248.sum() / 1e6:.3f} | 247 (day-ahead forecast): {s247.sum() / 1e6:.3f} | "
          f"362 (metered small-scale SURPLUS only): {s362.sum() / 1e6:.3f}")

    shapes = pd.DataFrame({"248": daily_shape(s248, SUMMER), "247": daily_shape(s247, SUMMER),
                           "362 surplus": daily_shape(s362, SUMMER), "ERA5 GHI": daily_shape(t["ghi_mean"], SUMMER),
                           "248 corrected": daily_shape(corrected, SUMMER)})
    print("\n=== May-Aug mean daily shape, % of daily total per UTC hour ===")
    print(shapes.loc[5:15].round(2).T.to_string())
    peak = shapes.idxmax()
    print("peak hour (UTC):", peak.to_dict())

    delta = corrected - s248
    m = delta.groupby([delta.index.month, delta.index.hour]).mean().unstack()[WINDOW_UTC].round(1)
    m.columns = [f"{h:02d} UTC" for h in WINDOW_UTC]
    moved = delta.clip(lower=0).groupby(delta.index.month).sum()
    m["energy_moved_MWh"] = moved.round(0)
    m["pct_of_month"] = (moved / s248.groupby(s248.index.month).sum() * 100).round(2)
    m["max_abs_hour_MW"] = delta.abs().groupby(delta.index.month).max().round(0)
    print("\n=== Artefact size: proposed correction minus 248, mean MW per hour by month ===")
    print("(positive = 248 too low at that hour; negative = too high)")
    print(m.to_string())
    day_diff = (corrected.groupby(corrected.index.date).sum() - s248.groupby(s248.index.date).sum()).abs().max()
    print(f"daily totals preserved: max |difference| = {day_diff:.2e} MWh")
    print(f"Wrote {plot_shapes(shapes, year)}")


def plot_shapes(shapes: pd.DataFrame, year: int) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.8), facecolor=C["surface"])
    ax.set_facecolor(C["surface"])
    hours = list(range(2, 20))
    s = shapes.loc[hours]
    styles = [("248", "Fingrid 248, 15-min-updated (baseline)", C["blue"], "-", 2.4),
              ("247", "Fingrid 247, day-ahead forecast", C["orange"], "-", 2),
              ("362 surplus", "Datahub 362, metered small-scale surplus", C["aqua"], "-", 2),
              ("ERA5 GHI", "ERA5 shortwave irradiance, city points", C["yellow"], "-", 2),
              ("248 corrected", "248 with proposed correction (not applied)", C["blue"], "--", 1.8)]
    for col, label, color, ls, lw in styles:
        ax.plot(hours, s[col], color=color, ls=ls, lw=lw, label=label, marker="o" if col == "248" else None, ms=4)
    ax.axvspan(min(WINDOW_UTC) - 0.5, max(WINDOW_UTC) + 0.5, color=C["grid"], alpha=0.5, lw=0)
    ax.text(min(WINDOW_UTC) - 0.4, ax.get_ylim()[1] * 0.98, "correction window", va="top", fontsize=8.5, color=C["muted"])
    ax.grid(axis="y", color=C["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(C["grid"])
    ax.tick_params(colors=C["muted"], labelsize=9)
    ax.set_xticks(hours[::2])
    ax.set_xlabel("Hour (UTC, hour start)", color=C["muted"])
    ax.set_ylabel("Share of daily total (%)", color=C["muted"])
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=C["text"])
    ax.set_title(f"Solar: average daily shape, May-Aug {year}", loc="left", fontsize=12, color=C["text"])
    fig.text(0.01, 0.005, "Sources: Fingrid / data.fingrid.fi (CC BY 4.0); Open-Meteo ERA5 (CC BY 4.0).",
             fontsize=8, color=C["muted"])
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"solar_daily_shape_{year}.png"
    fig.savefig(path, dpi=150, facecolor=C["surface"])
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, required=True)
    run(p.parse_args().year)


if __name__ == "__main__":
    main()

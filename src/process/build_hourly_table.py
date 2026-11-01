"""Merge Fingrid and weather series into one hourly UTC table.

Input:  data/hourly/fingrid/<name>_<start>_<end>.parquet  (src.fetch.fingrid)
        data/hourly/weather/<point>_<start>_<end>.parquet (src.fetch.weather)
Output: data/processed/hourly_<year>.parquet
        outputs/figures/sanity_weeks_<year>.png

Columns (power in MW = MWh/h, index = hour start, UTC):
  Fingrid      consumption, production, wind, solar (Fingrid FORECAST 248), nuclear, hydro,
               net_import (+ = import), other (= production - wind - solar - nuclear - hydro),
               solar_capacity (MW assumed by Fingrid, forward-filled),
               wind_capacity (MW; only if a source is configured, see WIND_CAPACITY_NAME)
  flags        flag_<series>: "ok" | "filled" (gap <= MAX_GAP_H, linear interpolation)
               | "suspect" (known bad data in config/data_issues.yaml, or FLAT_RUN_H+ identical
               hourly values = frozen data; keep but exclude from model fitting) | "missing".
               Derived series take the worst flag of their inputs.
  solar        solar = Fingrid 248 with the 09-12 UTC artefact correction if
               config/scenarios.yaml solar.artefact_correction is true, only in years where the
               artefact is detected (solar_artefact.has_artefact) and days where 247 exists;
               solar_raw = uncorrected 248; solar_corrected = True where the correction changed it.
  weather      ws100_<point> per wind point, ws100_mean (wind points),
               temp_mean (city points, C), ghi_mean (shortwave, city points, W/m2)
  derived      residual_load = consumption - wind - solar, wind_cf = wind / wind_capacity,
               solar_cf_fingrid = solar / solar_capacity (baseline solar profile),
               solar_cf_fingrid_raw = solar_raw / solar_capacity
  issues       curtailment: True in likely curtailment hours (config/data_issues.yaml);
               excluded from wind fitting only. crisis_2022: True for all hours of 2022 (UTC).
  calendar     hour_local, weekday_local (0 = Mon), month_local, is_holiday
               (Europe/Helsinki local time; holidays from the `holidays` package, incl.
               Midsummer Eve and Christmas Eve)
  price        price_eur_mwh: ENTSO-E day-ahead price, bidding zone FI (EUR/MWh), hourly mean of
               15-min prices from 2025-10-01; flag_price as for the Fingrid series
  neighbours   price_se1/se3/ee (ENTSO-E day-ahead), imp_se1/se3/ee (MW, + = import into Finland;
               Fingrid measured flows), cap_se1_fi/cap_se3_fi/cap_ee_fi (Fingrid official day-ahead
               import capacity; NaN where not published)

Usage:
    python -m src.process.build_hourly_table --year 2019 2020 2021 2022 2023 2024 2025
"""

from __future__ import annotations

import argparse
from pathlib import Path

import holidays
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.fetch import entsoe, fingrid, weather
from src.process.solar_artefact import correct_248, has_artefact

MAX_GAP_H = 3
FLAT_RUN_H = 6  # runs of >= this many identical hourly values are marked suspect (frozen data)
LOCAL_TZ = "Europe/Helsinki"
FINGRID_SERIES = ["consumption", "production", "wind", "solar", "nuclear", "hydro", "net_import"]
FLAT_IGNORE_ZERO = {"solar"}  # zero at night is real, not frozen data
NEIGHBOURS = ["se1", "se3", "ee"]  # bidding zones used for price coupling
# Short name in config/fingrid_datasets.yaml for wind capacity (Fingrid 268: forecast
# capacity, i.e. capacity actually producing; see the config notes).
WIND_CAPACITY_NAME: str | None = "wind_capacity"

REPO_ROOT = Path(__file__).resolve().parents[2]
ISSUES_PATH = REPO_ROOT / "config" / "data_issues.yaml"
SCENARIOS_PATH = REPO_ROOT / "config" / "scenarios.yaml"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"

FLAG_ORDER = ["ok", "filled", "suspect", "missing"]  # increasing severity


def fill_short_gaps(s: pd.Series, max_gap: int = MAX_GAP_H) -> tuple[pd.Series, pd.Series]:
    """Linearly interpolate interior NaN runs of <= max_gap hours; flag every hour."""
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("size")
    interp = s.interpolate(method="linear", limit_area="inside")
    fill = isna & (run_len <= max_gap) & interp.notna()
    out = s.where(~fill, interp)
    flag = pd.Series(np.where(fill, "filled", np.where(out.isna(), "missing", "ok")), index=s.index)
    return out, flag


def worst_flag(*flags: pd.Series) -> pd.Series:
    rank = pd.concat([f.map(FLAG_ORDER.index) for f in flags], axis=1).max(axis=1)
    return rank.map(dict(enumerate(FLAG_ORDER)))


def load_issues(kind: str, path: Path = ISSUES_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get(kind, [])


def in_periods(index: pd.DatetimeIndex, periods: list[dict]) -> np.ndarray:
    """True for hours inside any {start, end} period (both inclusive, hour-start labels)."""
    mask = np.zeros(len(index), dtype=bool)
    for p in periods:
        mask |= (index >= pd.Timestamp(p["start"])) & (index <= pd.Timestamp(p["end"]))
    return mask


def flat_runs(s: pd.Series, min_len: int = FLAT_RUN_H, ignore_zero: bool = False) -> tuple[np.ndarray, list[dict]]:
    """Hours in runs of >= min_len identical consecutive non-missing values (frozen data)."""
    run_id = (s != s.shift()).cumsum()
    size = s.groupby(run_id).transform("size")
    mask = s.notna() & (size >= min_len)
    if ignore_zero:
        mask &= s != 0
    runs = []
    for _, r in s[mask].groupby(run_id[mask]):
        runs.append({"start": r.index[0], "end": r.index[-1], "hours": len(r), "value": float(r.iloc[0])})
    return mask.to_numpy(), runs


def seeded_ffill(s: pd.Series, name: str, year: int) -> tuple[pd.Series, np.ndarray]:
    """Forward-fill a step series; if the year starts with gaps, seed from the previous year's file.

    Non-positive values are treated as missing (a capacity of 0 MW is a data error, e.g. Fingrid
    268 on 2022-03-22 12:00).
    """
    s = s.where(s > 0)
    if pd.isna(s.iloc[0]):
        prev = fingrid.hourly_path(name, f"{year - 1}-01-01", f"{year}-01-01")
        if prev.exists():
            last = pd.read_parquet(prev)["value"].dropna()
            if len(last):
                s = s.copy()
                s.iloc[0] = last.iloc[-1]
                filled = s.ffill()
                flag = np.where(s.notna() & (s.index != s.index[0]), "ok", np.where(filled.notna(), "filled", "missing"))
                return filled, flag
    filled = s.ffill()
    return filled, np.where(s.notna(), "ok", np.where(filled.notna(), "filled", "missing"))


def load_solar_settings(path: Path = SCENARIOS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("solar", {})


def mark_suspect(t: pd.DataFrame, suspects: list[dict], columns: list[str]) -> None:
    """Set flag_<col> to "suspect" (unless already missing) for listed hours, in place."""
    for s in suspects:
        hours = in_periods(t.index, [s])
        for col in set(s["series"]) & set(columns):
            flag = t[f"flag_{col}"]
            t[f"flag_{col}"] = flag.where(~hours | (flag == "missing"), "suspect")


def build(year: int) -> pd.DataFrame:
    start, end = f"{year}-01-01", f"{year + 1}-01-01"
    idx = pd.date_range(start, end, freq="1h", inclusive="left", tz="UTC", name="time_utc")
    t = pd.DataFrame(index=idx)

    def fg(name: str) -> pd.Series:
        return pd.read_parquet(fingrid.hourly_path(name, start, end))["value"].reindex(idx)

    # Fingrid series with short-gap filling; frozen runs (flat for FLAT_RUN_H+ hours) -> suspect
    runs_report = []
    for name in FINGRID_SERIES:
        raw = fg(name)
        flat, runs = flat_runs(raw, ignore_zero=name in FLAT_IGNORE_ZERO)
        runs_report += [{"series": name, **r} for r in runs]
        t[name], flag = fill_short_gaps(raw)
        t[f"flag_{name}"] = flag.where(~flat, "suspect")
    t.attrs["flat_runs"] = [{**r, "start": str(r["start"]), "end": str(r["end"])} for r in runs_report]
    # known bad data: base series first so derived series inherit the flag
    suspects = load_issues("suspect")
    mark_suspect(t, suspects, FINGRID_SERIES)

    # Solar 09-12 UTC artefact correction (config/scenarios.yaml: solar.artefact_correction)
    # applied only in years where the artefact is detected (solar_artefact.has_artefact)
    t["solar_raw"] = t["solar"]
    dayahead = fg("solar_dayahead")
    t.attrs["solar_artefact_detected"] = has_artefact(t["solar_raw"], dayahead)
    if load_solar_settings().get("artefact_correction", False) and t.attrs["solar_artefact_detected"]:
        t["solar"] = correct_248(t["solar_raw"], dayahead)
    t["solar_corrected"] = (t["solar"] - t["solar_raw"]).abs().fillna(0) > 1e-9

    parts = ["production", "wind", "solar", "nuclear", "hydro"]
    t["other"] = t["production"] - t["wind"] - t["solar"] - t["nuclear"] - t["hydro"]
    t["flag_other"] = worst_flag(*(t[f"flag_{p}"] for p in parts))

    # Capacities: step series, forward-filled (seeded from the previous year if needed)
    t["solar_capacity"], t["flag_solar_capacity"] = seeded_ffill(fg("solar_capacity"), "solar_capacity", year)
    if WIND_CAPACITY_NAME:
        t["wind_capacity"], t["flag_wind_capacity"] = seeded_ffill(fg(WIND_CAPACITY_NAME), WIND_CAPACITY_NAME, year)
    else:
        t["wind_capacity"] = np.nan
        t["flag_wind_capacity"] = "missing"

    # Weather
    groups = weather.load_locations()
    wx = {n: pd.read_parquet(weather.hourly_path(n, start, end)).reindex(idx) for g in groups.values() for n in g}
    for n in groups["wind"]:
        t[f"ws100_{n}"] = wx[n]["wind_speed_100m"]
    t["ws100_mean"] = t[[f"ws100_{n}" for n in groups["wind"]]].mean(axis=1)
    t["temp_mean"] = pd.concat([wx[n]["temperature_2m"] for n in groups["solar"]], axis=1).mean(axis=1)
    t["ghi_mean"] = pd.concat([wx[n]["shortwave_radiation"] for n in groups["solar"]], axis=1).mean(axis=1)

    # Derived
    t["residual_load"] = t["consumption"] - t["wind"] - t["solar"]
    t["flag_residual_load"] = worst_flag(t["flag_consumption"], t["flag_wind"], t["flag_solar"])
    mark_suspect(t, suspects, ["other", "residual_load"])
    t["wind_cf"] = t["wind"] / t["wind_capacity"]
    t["solar_cf_fingrid"] = t["solar"] / t["solar_capacity"]
    t["solar_cf_fingrid_raw"] = t["solar_raw"] / t["solar_capacity"]

    # Calendar (local time)
    local = idx.tz_convert(LOCAL_TZ)
    fi_holidays = holidays.Finland(years=[year - 1, year, year + 1])
    t["hour_local"] = local.hour
    t["weekday_local"] = local.weekday
    t["month_local"] = local.month
    t["is_holiday"] = [d in fi_holidays for d in local.date]

    # likely curtailment: data kept and flags unchanged, excluded from wind fitting only
    t["curtailment"] = in_periods(idx, load_issues("curtailment"))
    # energy crisis year, so models can include or exclude it
    t["crisis_2022"] = idx.year == 2022

    # ENTSO-E day-ahead price (FI), hourly mean (15-min prices from 2025-10-01 are averaged)
    price_path = entsoe.hourly_path(start, end)
    if price_path.exists():
        t["price_eur_mwh"], t["flag_price"] = fill_short_gaps(pd.read_parquet(price_path)["value"].reindex(idx))
    else:
        t["price_eur_mwh"], t["flag_price"] = np.nan, "missing"
    mark_suspect(t, suspects, ["price"])  # flag_price (series name "price" in data_issues.yaml)

    # Neighbours, for price coupling: day-ahead prices, imports per border (+ = into Finland,
    # i.e. minus Fingrid's flow sign) and Fingrid's official day-ahead import capacity
    # (NaN where none is published, e.g. SE1/SE3 after 2024-10-29; see config/coupling.yaml).
    for nb in NEIGHBOURS:
        pp = entsoe.hourly_path(start, end, nb.upper())
        t[f"price_{nb}"] = pd.read_parquet(pp)["value"].reindex(idx) if pp.exists() else np.nan
        t[f"imp_{nb}"], _ = fill_short_gaps(-fg(f"flow_{nb}"))
        cp = fingrid.hourly_path(f"cap_{nb}_fi", start, end)
        t[f"cap_{nb}_fi"] = pd.read_parquet(cp)["value"].reindex(idx) if cp.exists() else np.nan

    for c in t.columns:
        if c.startswith("flag_"):
            t[c] = pd.Categorical(t[c], categories=FLAG_ORDER)
    return t


def summarize(t: pd.DataFrame) -> None:
    pd.set_option("display.width", 160)
    print(f"\n=== Hourly table: {t.shape[0]} rows x {t.shape[1]} columns, "
          f"{t.index.min():%Y-%m-%d %H:%M} .. {t.index.max():%Y-%m-%d %H:%M} UTC ===")
    print("Columns:", ", ".join(t.columns))

    flags = {c.removeprefix("flag_"): t[c].value_counts().reindex(FLAG_ORDER, fill_value=0) for c in t if c.startswith("flag_")}
    print("\nHours per flag:")
    print(pd.DataFrame(flags).T.to_string())

    data_cols = [c for c in t.columns if not c.startswith("flag_")]
    nan = t[data_cols].isna().sum()
    print("\nRemaining NaN hours (non-flag columns):", nan[nan > 0].to_dict() or "none")

    runs = t.attrs.get("flat_runs", [])
    print(f"\nFlat runs (>= {FLAT_RUN_H} identical hourly values, marked suspect): {len(runs)}")
    if runs:
        print(pd.DataFrame(runs).to_string(index=False))
    print(f"Solar artefact detected: {t.attrs.get('solar_artefact_detected')}; hours changed by the correction: {int(t['solar_corrected'].sum())}")

    m = t[["residual_load", "consumption", "wind", "solar"]].resample("MS").mean().round(0)
    m.index = m.index.strftime("%Y-%m")
    print("\nMonthly mean (MW):")
    print(m.to_string())


def plot_weeks(t: pd.DataFrame, year: int) -> Path:
    # Reference palette (dataviz skill), categorical slots 1-4, light mode
    series = [
        ("consumption", "Consumption", "#2a78d6"),
        ("residual_load", "Residual load", "#eb6834"),
        ("nuclear", "Nuclear", "#1baf7a"),
        ("wind", "Wind", "#eda100"),
    ]
    weeks = [("January", f"{year}-01-08", f"{year}-01-15"), ("July", f"{year}-07-08", f"{year}-07-15")]
    text, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharey=True, facecolor=surface)
    for ax, (label, a, b) in zip(axes, weeks):
        w = t[(t.index >= pd.Timestamp(a, tz="UTC")) & (t.index < pd.Timestamp(b, tz="UTC"))]
        ax.set_facecolor(surface)
        for col, name, color in series:
            ax.plot(w.index, w[col], color=color, lw=2, label=name)
        # direct labels at the line ends, pushed apart so they don't collide
        ends = sorted(((w[col].dropna().iloc[-1], name) for col, name, _ in series), reverse=True)
        min_gap, y_prev = 700.0, np.inf
        for y, name in ends:
            y = min(y, y_prev - min_gap)
            ax.annotate(name, (w.index[-1], y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=text)
            y_prev = y
        ax.set_title(f"{label}: {pd.Timestamp(a):%a %d %b} - {pd.Timestamp(b) - pd.Timedelta('1D'):%a %d %b %Y}",
                     loc="left", fontsize=11, color=text)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
        ax.tick_params(colors=muted, labelsize=9)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a %d"))
        ax.set_ylabel("MW", color=muted)
        ax.set_xlim(w.index.min(), w.index.max() + pd.Timedelta("18h"))
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.2), ncol=4, frameon=False, fontsize=9, labelcolor=text)
    fig.suptitle(f"Finland hourly power, one winter and one summer week ({year}, UTC)", x=0.06, ha="left",
                 fontsize=13, color=text, y=0.995)
    fig.text(0.06, 0.005, "Residual load = consumption - wind - solar. Source Fingrid / data.fingrid.fi, license CC 4.0 BY.",
             fontsize=8, color=muted)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"sanity_weeks_{year}.png"
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, nargs="+", required=True, help="one or more years, e.g. 2019 2020")
    args = p.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for year in args.year:
        t = build(year)
        out = PROCESSED_DIR / f"hourly_{year}.parquet"
        t.to_parquet(out)
        summarize(t)
        print(f"\nWrote {out}")
        print(f"Wrote {plot_weeks(t, year)}")


if __name__ == "__main__":
    main()

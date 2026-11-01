"""Sanity checks on the hourly weather written by src.fetch.weather.

- Per location: rows, gaps / missing values, ERA5 grid cell, mean 100 m wind speed.
- Points that snap to the same ERA5 grid cell (duplicates add no information).
- Correlation of the mean 100 m wind speed over the wind points with Fingrid wind (75),
  directly and cubed.
- Radiation timing: mean diurnal profile of shortwave radiation vs Fingrid solar (248).
- Monthly mean temperature vs monthly mean consumption (124).

Usage:
    python -m src.process.check_weather --start 2024-01-01 --end 2025-01-01
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.fetch import fingrid, weather


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    args = p.parse_args()
    pd.set_option("display.width", 160)

    idx = pd.date_range(args.start, args.end, freq="1h", inclusive="left", tz="UTC")
    groups = weather.load_locations()
    data = {
        name: pd.read_parquet(weather.hourly_path(name, args.start, args.end))
        for points in groups.values()
        for name in points
    }

    rows = {}
    for group, points in groups.items():
        for name in points:
            df = data[name]
            rows[name] = {
                "group": group,
                "rows": len(df),
                "missing_h": len(idx.difference(df.index)),
                "nan_values": int(df.isna().sum().sum()),
                "grid_cell": f"{df.attrs['grid_lat']:.2f}, {df.attrs['grid_lon']:.2f}",
                "mean_ws100": round(df["wind_speed_100m"].mean(), 2),
                "mean_ws10": round(df["wind_speed_10m"].mean(), 2),
                "mean_T2m": round(df["temperature_2m"].mean(), 1),
            }
    summary = pd.DataFrame(rows).T
    print(f"\n=== Weather points {args.start} .. {args.end} (expected rows: {len(idx)}) ===")
    print(summary.to_string())
    dup = summary[summary.duplicated("grid_cell", keep=False)]
    print("\nPoints sharing an ERA5 grid cell:", "none" if dup.empty else dup["grid_cell"].to_dict())

    def fg(name: str) -> pd.Series:
        return pd.read_parquet(fingrid.hourly_path(name, args.start, args.end))["value"]

    wind_pts = list(groups["wind"])
    ws = pd.concat({n: data[n]["wind_speed_100m"] for n in wind_pts}, axis=1)
    wind = fg("wind")
    cand = {
        "mean ws100": ws.mean(axis=1),
        "mean ws100 ^3": ws.mean(axis=1) ** 3,
        "mean of (ws100 ^3)": (ws**3).mean(axis=1),
    }
    print(f"\n=== Correlation with Fingrid wind (75), {len(wind_pts)} wind points ===")
    for label, s in cand.items():
        both = pd.concat([s, wind], axis=1).dropna()
        print(f"  {label:<20} Pearson r = {both.corr().iloc[0, 1]:.3f}   Spearman = {both.corr('spearman').iloc[0, 1]:.3f}")

    sw = pd.concat({n: data[n]["shortwave_radiation"] for n in groups["solar"]}, axis=1).mean(axis=1)
    solar = fg("solar")
    summer = (sw.index.month >= 5) & (sw.index.month <= 8)
    prof = pd.DataFrame(
        {
            "shortwave_W/m2": sw[summer].groupby(sw.index[summer].hour).mean(),
            "fingrid_solar_MW": solar[summer].groupby(solar.index[summer].hour).mean(),
        }
    )
    print("\n=== Radiation timing, May-Aug mean by hour (UTC, hour-start label) ===")
    print(f"  peak hour: shortwave {prof.iloc[:, 0].idxmax()}:00, Fingrid solar {prof.iloc[:, 1].idxmax()}:00")
    print(f"  hourly correlation shortwave vs Fingrid solar: {pd.concat([sw, solar], axis=1).corr().iloc[0, 1]:.3f}")
    print(prof.loc[3:18].round(0).T.to_string())

    temp = pd.concat({n: d["temperature_2m"] for n, d in data.items()}, axis=1).mean(axis=1)
    temp_pop = pd.concat({n: data[n]["temperature_2m"] for n in groups["solar"]}, axis=1).mean(axis=1)
    cons = fg("consumption")
    monthly = pd.DataFrame(
        {
            "T2m all points (C)": temp.resample("MS").mean(),
            "T2m solar/city points (C)": temp_pop.resample("MS").mean(),
            "consumption (MW)": cons.resample("MS").mean(),
        }
    )
    monthly.index = monthly.index.strftime("%Y-%m")
    print("\n=== Monthly mean temperature vs consumption ===")
    print(monthly.round(1).to_string())
    print(f"  monthly correlation (city points): {monthly.iloc[:, 1].corr(monthly.iloc[:, 2]):.3f}")
    daily = pd.concat([temp_pop.resample("D").mean(), cons.resample("D").mean()], axis=1)
    print(f"  daily correlation (city points):   {daily.corr().iloc[0, 1]:.3f}")


if __name__ == "__main__":
    main()

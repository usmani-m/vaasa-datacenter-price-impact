"""Sanity checks on the hourly Fingrid series written by src.fetch.fingrid.

- Per series: rows, missing hours, incomplete hours, mean MW, annual TWh.
- Derived "other" = production - wind - solar - nuclear - hydro (CHP, biomass, industry, ...).
- Balance: consumption vs production + net_import, with and without the Aland link.
- Cross-check: net_import (summed borders) vs the real-time dataset 194 (sign flipped),
  over whatever period 194 was fetched for.

Usage:
    python -m src.process.check_fingrid --start 2024-01-01 --end 2025-01-01
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.fetch.fingrid import HOURLY_DIR, hourly_path, load_config

# Series where an energy total makes no sense
NO_ENERGY = {"solar_capacity"}


def load(name: str, start: str, end: str) -> pd.DataFrame | None:
    path = hourly_path(name, start, end)
    return pd.read_parquet(path) if path.exists() else None


def series_summary(series: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {}
    for name, df in series.items():
        v = df["value"]
        mean = v.mean()
        rows[name] = {
            "rows": len(df),
            "missing_h": int(v.isna().sum()),
            "incomplete_h": int((~df["complete"]).sum()),
            "mean_MW": round(mean, 1),
            # mean x hours: missing hours are filled with the mean
            "TWh": None if name in NO_ENERGY else round(mean * len(df) / 1e6, 2),
        }
    return pd.DataFrame(rows).T


def mismatch_stats(diff: pd.Series) -> dict:
    d = diff.dropna()
    worst = d.abs().idxmax()
    return {
        "hours": len(d),
        "mean_MW": round(d.mean(), 1),
        "mean_abs_MW": round(d.abs().mean(), 1),
        "p99_abs_MW": round(d.abs().quantile(0.99), 1),
        "max_abs_MW": round(d.abs().max(), 1),
        "max_at": worst,
        "max_value": round(d[worst], 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    args = p.parse_args()
    pd.set_option("display.width", 160)

    cfg = load_config()
    names = list(cfg["datasets"]) + list(cfg.get("derived", {}))
    s = {n: df for n in names if (df := load(n, args.start, args.end)) is not None}
    v = {n: df["value"] for n, df in s.items()}

    needed = ["production", "wind", "solar", "nuclear", "hydro"]
    if all(n in v for n in needed):
        other = v["production"] - v["wind"] - v["solar"] - v["nuclear"] - v["hydro"]
        complete = pd.concat([s[n]["complete"] for n in needed], axis=1).all(axis=1)
        s["other (derived)"] = pd.DataFrame({"value": other, "complete": complete})

    print(f"\n=== Hourly series {args.start} .. {args.end} (complete hours expected: all rows) ===")
    print(series_summary({n: df for n, df in s.items() if n != "net_export_rt"}).to_string())

    if all(n in v for n in ("consumption", "production", "net_import")):
        print("\n=== Balance: consumption - (production + net_import) ===")
        res = {"net_import as configured": v["consumption"] - (v["production"] + v["net_import"])}
        if "flow_ax" in v:
            in_cfg = "flow_ax" in cfg["derived"]["net_import"]["sum_of"]
            alt = v["net_import"] + v["flow_ax"] if in_cfg else v["net_import"] - v["flow_ax"]
            label = "without Aland" if in_cfg else "with Aland (- flow_ax)"
            res[f"net_import {label}"] = v["consumption"] - (v["production"] + alt)
        print(pd.DataFrame({k: mismatch_stats(d) for k, d in res.items()}).T.to_string())

    # 194 is fetched for a shorter period than the rest: use whatever file exists
    rt_files = sorted(HOURLY_DIR.glob("net_export_rt_*.parquet"))
    if rt_files and "net_import" in v:
        rt = -pd.concat([pd.read_parquet(f)["value"] for f in rt_files])
        rt = rt[~rt.index.duplicated()]
        both = pd.concat({"borders": v["net_import"], "rt194": rt}, axis=1).dropna()
        if len(both):
            print(f"\n=== Cross-check: net_import (borders) - (-194 real-time), {both.index.min():%Y-%m-%d} .. {both.index.max():%Y-%m-%d} ===")
            st = mismatch_stats(both["borders"] - both["rt194"])
            st["corr"] = round(both.corr().iloc[0, 1], 4)
            print(pd.Series(st).to_string())


if __name__ == "__main__":
    main()

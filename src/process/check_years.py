"""Multi-year checks on the hourly tables and Fingrid series.

- Per year: TWh by series, balance error (consumption - production - net_import), missing and
  suspect hours, flat runs.
- Resolution switch: interval lengths per dataset and year (hourly before 2023-06-13, 15 min
  after, 3 min for real-time series).
- Russia: import volume and share, and the balance with and without the FI-RU link.
- Solar availability: coverage of 248, 247, 267, 362 per year.
- 248 artefact by year: May-Aug share of the daily total at 09 and 11 UTC, 248 vs 247.

Usage:
    python -m src.process.check_years --years 2019 2020 2021 2022 2023 2024 2025
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.fetch import fingrid
from src.process.solar_artefact import daily_shape

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ENERGY = ["consumption", "production", "wind", "solar", "nuclear", "hydro", "other", "net_import"]
BASE = ["consumption", "production", "wind", "solar", "nuclear", "hydro", "net_import"]


def fg(name: str, year: int) -> pd.Series:
    path = fingrid.hourly_path(name, f"{year}-01-01", f"{year + 1}-01-01")
    return pd.read_parquet(path)["value"] if path.exists() else pd.Series(dtype=float)


def read_raw_cache(dataset_id: int, year: int) -> pd.DataFrame:
    """Raw records from the local monthly cache only (never calls the API)."""
    files = sorted((fingrid.RAW_DIR / str(dataset_id)).glob(f"{year}-*.parquet"))
    if not files:
        return pd.DataFrame({"end_time": pd.Series(dtype="datetime64[ns, UTC]")})
    return pd.concat([pd.read_parquet(f) for f in files])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", type=int, nargs="+", required=True)
    years = sorted(p.parse_args().years)
    pd.set_option("display.width", 200, "display.max_columns", 40)
    tables = {y: pd.read_parquet(PROCESSED_DIR / f"hourly_{y}.parquet") for y in years}

    # --- per-year table
    rows = {}
    for y, t in tables.items():
        r = {f"{c} TWh": t[c].sum() / 1e6 for c in ENERGY}
        bal = t["consumption"] - t["production"] - t["net_import"]
        clean = t[[f"flag_{c}" for c in ("consumption", "production", "net_import")]].isin(["ok", "filled"]).all(axis=1)
        r |= {"bal mean MW": bal.mean(), "bal |mean| MW": bal.abs().mean(), "bal max MW": bal.abs().max(),
              "bal |mean| clean h": bal[clean].abs().mean(),
              "hours": len(t),
              "missing h (any base)": int((t[[f"flag_{c}" for c in BASE]] == "missing").any(axis=1).sum()),
              "suspect h (any base)": int((t[[f"flag_{c}" for c in BASE]] == "suspect").any(axis=1).sum()),
              "solar corrected h": int(t["solar_corrected"].sum())}
        rows[y] = r
    print("\n=== Per-year check ===")
    print(pd.DataFrame(rows).T.round(2).to_string())

    print("\n=== Missing / suspect hours per series ===")
    ms = {y: {f"{c} {k}": int((t[f"flag_{c}"] == k).sum()) for c in BASE for k in ("missing", "suspect")} for y, t in tables.items()}
    ms = pd.DataFrame(ms).T
    print(ms.loc[:, (ms != 0).any()].to_string())

    # --- resolution switch
    print("\n=== Interval lengths in raw records (minutes: count), per dataset and year ===")
    cfg = fingrid.load_config()["datasets"]
    res = {}
    for name in ["consumption", "production", "wind", "solar", "solar_dayahead", "nuclear", "hydro",
                 "flow_ee", "flow_no", "flow_ru", "flow_se1", "flow_se3"]:
        res[name] = {}
        for y in years:
            raw = read_raw_cache(cfg[name]["id"], y)
            iv = ((raw["end_time"] - raw.index).dt.total_seconds() / 60).round().astype(int)
            res[name][y] = ", ".join(f"{k}:{v}" for k, v in iv.value_counts().sort_index().items())
    print(pd.DataFrame(res).to_string())

    # --- Russia
    print("\n=== Russia (FI-RU, dataset 58; + = import here) ===")
    ru = {}
    for y, t in tables.items():
        imp = -fg("flow_ru", y)
        bal = t["consumption"] - t["production"] - t["net_import"]
        bal_wo = bal + imp.reindex(t.index)  # as if net_import excluded Russia
        ru[y] = {"RU import TWh": imp.sum() / 1e6, "share of consumption %": imp.sum() / t["consumption"].sum() * 100,
                 "RU missing h": int(imp.isna().sum()), "last nonzero": str(imp[imp.abs() > 1].index.max()),
                 "bal |mean| with RU": bal.abs().mean(), "bal |mean| without RU": bal_wo.abs().mean()}
    print(pd.DataFrame(ru).T.round(2).to_string())

    # --- solar availability
    print("\n=== Solar series availability (hours with data / first / last) ===")
    av = {}
    for y in years:
        for name in ("solar", "solar_dayahead", "solar_capacity", "solar_surplus_small"):
            s = fg(name, y).dropna()
            av[(y, name)] = {"hours": len(s), "first": str(s.index.min()) if len(s) else "-",
                             "last": str(s.index.max()) if len(s) else "-"}
    print(pd.DataFrame(av).T.to_string())

    # --- 248 artefact by year
    print("\n=== 248 artefact by year: May-Aug share of daily total (%), 248 minus 247 ===")
    art = {}
    for y in years:
        a, b = fg("solar", y), fg("solar_dayahead", y)
        if a.dropna().empty or b.dropna().empty:
            continue
        both = a.notna() & b.notna()
        sa, sb = daily_shape(a[both], [5, 6, 7, 8]), daily_shape(b[both], [5, 6, 7, 8])
        art[y] = {f"{h:02d} UTC": sa[h] - sb[h] for h in (8, 9, 10, 11, 12)} | {"248 peak": int(sa.idxmax()), "247 peak": int(sb.idxmax())}
    print(pd.DataFrame(art).T.round(2).to_string())


if __name__ == "__main__":
    main()

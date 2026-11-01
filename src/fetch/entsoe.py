"""Fetch day-ahead prices and actual load (FI and neighbouring zones) from ENTSO-E Transparency.

Data: ENTSO-E Transparency Platform (transparency.entsoe.eu).

- Bidding zones FI, SE1, SE3, EE (see AREAS), EUR/MWh, day-ahead (SDAC) prices.
- Resolution: hourly until 2025-09-30; 15 min from delivery day 2025-10-01. The raw monthly
  cache keeps the native resolution; the hourly output is the mean of the intervals in each
  UTC hour (an hour of 15-min prices is complete with 4 values).
- Token in .env as ENTSOE_API_KEY. One request per month, polite pause, retries.

Usage:
    python -m src.fetch.entsoe --start 2019-01-01 --end 2026-01-01 --areas FI SE1 SE3 EE
    python -m src.fetch.entsoe --start 2023-01-01 --end 2026-01-01 --areas FI SE1 SE3 --kind load
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

AREAS = {"FI": "FI", "SE1": "SE_1", "SE3": "SE_3", "EE": "EE"}  # label -> entsoe-py area code
# kind -> (raw cache suffix, value column, hourly file prefix)
KINDS = {"price": ("day_ahead", "price_eur_mwh", "price"), "load": ("load", "load_mw", "load")}
MIN_INTERVAL_S = 2.0
MAX_RETRIES = 5

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = REPO_ROOT / "data" / "raw" / "entsoe"
HOURLY_DIR = REPO_ROOT / "data" / "hourly" / "entsoe"

log = logging.getLogger(__name__)


class EntsoeFetcher:
    def __init__(self, api_key: str | None = None):
        load_dotenv(REPO_ROOT / ".env")
        api_key = api_key or os.environ.get("ENTSOE_API_KEY")
        if not api_key:
            raise RuntimeError("ENTSOE_API_KEY not set (see .env.example)")
        # entsoe-py's own retries cover connection errors; HTTP errors are retried below
        self.client = EntsoePandasClient(api_key=api_key, retry_count=2, retry_delay=10, timeout=120)
        self._last_request = 0.0

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < MIN_INTERVAL_S:
            time.sleep(MIN_INTERVAL_S - elapsed)
        self._last_request = time.monotonic()

    def query(self, start: pd.Timestamp, end: pd.Timestamp, area: str = "FI", kind: str = "price") -> pd.Series:
        """Day-ahead prices (kind="price") or actual total load (kind="load") with
        start <= timestamp < end at native resolution, UTC index."""
        for attempt in range(1, MAX_RETRIES + 1):
            self._wait()
            try:
                if kind == "price":
                    s = self.client.query_day_ahead_prices(AREAS[area], start=start, end=end)
                else:
                    s = self.client.query_load(AREAS[area], start=start, end=end).iloc[:, 0]
            except NoMatchingDataError:
                log.warning("No data for %s .. %s", start, end)
                return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
            except requests.HTTPError as e:
                code = e.response.status_code if e.response is not None else None
                if code is not None and code != 429 and code < 500:
                    raise  # bad token / bad request: retrying won't help
                delay = min(10 * 2 ** (attempt - 1), 300)
                log.warning("HTTP %s, retry %d/%d in %ds", code, attempt, MAX_RETRIES, delay)
                time.sleep(delay)
                continue
            except requests.RequestException as e:
                delay = min(10 * 2 ** (attempt - 1), 300)
                log.warning("Request error (%s), retry %d/%d in %ds", e, attempt, MAX_RETRIES, delay)
                time.sleep(delay)
                continue
            s = s.tz_convert("UTC").sort_index()
            s = s[~s.index.duplicated(keep="last")]
            return s[(s.index >= start) & (s.index < end)].astype(float)
        raise RuntimeError(f"ENTSO-E request failed after {MAX_RETRIES} attempts: {start} .. {end}")


def raw_dir(area: str = "FI", kind: str = "price") -> Path:
    return RAW_ROOT / f"{area}_{KINDS[kind][0]}"


def fetch_prices(start: str | pd.Timestamp, end: str | pd.Timestamp, refresh: bool = False,
                 area: str = "FI", kind: str = "price") -> pd.Series:
    """[start, end) at native resolution; one cached Parquet file per completed month."""
    start, end = _utc(start), _utc(end)
    cache = raw_dir(area, kind)
    col = KINDS[kind][1]
    cache.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    fetcher = None
    parts = []
    for m_start in pd.date_range(start.floor("D").replace(day=1), end, freq="MS", inclusive="left"):
        m_end = m_start + pd.offsets.MonthBegin(1)
        path = cache / f"{m_start:%Y-%m}.parquet"
        if path.exists() and not refresh:
            s = pd.read_parquet(path)[col]
        else:
            fetcher = fetcher or EntsoeFetcher()
            log.info("Fetching %s %s for %s", area, kind, f"{m_start:%Y-%m}")
            try:
                s = fetcher.query(m_start, min(m_end, now), area, kind)
            except requests.HTTPError as e:
                if e.response is None or e.response.status_code != 400:
                    raise
                # ENTSO-E sometimes rejects a month that it serves fine in smaller pieces
                # (e.g. SE1 load 2025-05): fall back to weekly requests for this month
                log.warning("HTTP 400 for %s %s %s; retrying in weekly chunks", area, kind, f"{m_start:%Y-%m}")
                edges = list(pd.date_range(m_start, min(m_end, now), freq="7D")) + [min(m_end, now)]
                s = pd.concat([fetcher.query(a, b, area, kind) for a, b in zip(edges[:-1], edges[1:]) if a < b])
                s = s[~s.index.duplicated(keep="last")].sort_index()
            # day-ahead prices for tomorrow exist, so a month is complete once it has ended
            if m_end <= now:
                s.rename(col).to_frame().to_parquet(path)
        parts.append(s)
    out = pd.concat(parts) if parts else pd.Series(dtype=float)
    out.index.name = "start_time"
    return out[(out.index >= start) & (out.index < end)]


def to_hourly(s: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Hourly mean on the full [start, end) UTC index, with interval counts and completeness."""
    idx = pd.date_range(start, end, freq="1h", inclusive="left", name="time_utc")
    g = s.resample("1h")
    hourly = pd.DataFrame({"value": g.mean(), "n_intervals": g.count()}).reindex(idx)
    # an hour holding any interval that does not start on the hour is a 15-min hour
    sub_hourly = pd.Series((s.index != s.index.floor("1h")).astype(int), index=s.index).resample("1h").max()
    expected = (sub_hourly.reindex(idx).fillna(0) > 0).map({True: 4, False: 1})
    hourly["n_intervals"] = hourly["n_intervals"].fillna(0).astype(int)
    hourly["complete"] = hourly["n_intervals"] >= expected
    hourly["resolution_min"] = expected.map({4: 15, 1: 60})
    return hourly


def hourly_path(start: str | pd.Timestamp, end: str | pd.Timestamp, area: str = "FI", kind: str = "price") -> Path:
    return HOURLY_DIR / f"{KINDS[kind][2]}_{area.lower()}_{_utc(start):%Y%m%d}_{_utc(end):%Y%m%d}.parquet"


def _utc(t: str | pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True, help="UTC start (inclusive), e.g. 2019-01-01")
    p.add_argument("--end", required=True, help="UTC end (exclusive), e.g. 2026-01-01")
    p.add_argument("--areas", nargs="+", default=["FI"], choices=list(AREAS), help="bidding zones, e.g. FI SE1 SE3 EE")
    p.add_argument("--kind", default="price", choices=list(KINDS), help="price (day-ahead) or load (actual total load)")
    p.add_argument("--refresh", action="store_true", help="ignore cached months")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    start, end = _utc(args.start), _utc(args.end)
    HOURLY_DIR.mkdir(parents=True, exist_ok=True)
    for area in args.areas:
        raw = fetch_prices(start, end, refresh=args.refresh, area=area, kind=args.kind)
        # one hourly file per calendar year, matching the other hourly series
        for year in range(start.year, (end - pd.Timedelta("1h")).year + 1):
            ys, ye = max(start, _utc(f"{year}-01-01")), min(end, _utc(f"{year + 1}-01-01"))
            hourly = to_hourly(raw[(raw.index >= ys) & (raw.index < ye)], ys, ye)
            path = hourly_path(ys, ye, area, args.kind)
            hourly.to_parquet(path)
            log.info("Wrote %d hourly rows to %s (%d missing, %d incomplete)", len(hourly), path,
                     int(hourly["value"].isna().sum()), int((~hourly["complete"]).sum()))

if __name__ == "__main__":
    main()

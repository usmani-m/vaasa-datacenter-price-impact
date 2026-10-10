"""Fetch time series from Fingrid Open Data (data.fingrid.fi).

Data: Source Fingrid / data.fingrid.fi, license CC 4.0 BY.

Datasets are listed by short name in config/fingrid_datasets.yaml.

API facts (verified against live responses):
- GET https://data.fingrid.fi/api/datasets/{id}/data
  params: startTime, endTime (ISO 8601 UTC), format=json, pageSize, page
- Response: {"data": [{"datasetId", "startTime", "endTime", "value"}, ...],
             "pagination": {"total", "lastPage", "nextPage", "currentPage", "perPage", ...}}
- Records are returned newest first; endTime of the query is exclusive.
- Values are average power (MW) over each interval. Measured series are hourly before
  2023-06-13 and 15-min after; real-time series are 3-min samples. The hourly value is
  the mean over the hour (= MWh/h).
- Rate limit: 1 request / 2 s, 10,000 / day.

Usage:
    python -m src.fetch.fingrid --all --start 2024-01-01 --end 2025-01-01
    python -m src.fetch.fingrid --name net_export_rt --start 2024-01-01 --end 2024-02-01
    python -m src.fetch.fingrid --dataset 124 --start 2024-01-01 --end 2025-01-01
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

BASE_URL = "https://data.fingrid.fi/api/datasets/{dataset_id}/data"
MIN_INTERVAL_S = 2.5  # API allows 1 request / 2 s; keep a margin
PAGE_SIZE = 20000
MAX_RETRIES = 6
TIMEOUT_S = 60
SAMPLE_PERIOD_MIN = 3  # Fingrid real-time series
SAMPLE_PERIOD = f"{SAMPLE_PERIOD_MIN}min"

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "fingrid_datasets.yaml"
RAW_DIR = REPO_ROOT / "data" / "raw" / "fingrid"
HOURLY_DIR = REPO_ROOT / "data" / "hourly" / "fingrid"

log = logging.getLogger(__name__)


class FingridClient:
    """Thin HTTP client with rate limiting and retries."""

    def __init__(self, api_key: str | None = None):
        load_dotenv(REPO_ROOT / ".env")
        api_key = api_key or os.environ.get("FINGRID_API_KEY")
        if not api_key:
            raise RuntimeError("FINGRID_API_KEY not set (see .env.example)")
        self.session = requests.Session()
        self.session.headers["x-api-key"] = api_key
        self._last_request = 0.0

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < MIN_INTERVAL_S:
            time.sleep(MIN_INTERVAL_S - elapsed)
        self._last_request = time.monotonic()

    def get_page(self, dataset_id: int, start: pd.Timestamp, end: pd.Timestamp, page: int) -> dict:
        params = {
            "startTime": _iso(start),
            "endTime": _iso(end),
            "format": "json",
            "pageSize": PAGE_SIZE,
            "page": page,
        }
        url = BASE_URL.format(dataset_id=dataset_id)
        for attempt in range(1, MAX_RETRIES + 1):
            self._wait()
            try:
                r = self.session.get(url, params=params, timeout=TIMEOUT_S)
            except requests.RequestException as e:
                delay = _backoff(attempt)
                log.warning("Request error (%s), retry %d/%d in %.0fs", e, attempt, MAX_RETRIES, delay)
                time.sleep(delay)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 or r.status_code >= 500:
                delay = _retry_after(r) or _backoff(attempt)
                log.warning("HTTP %d, retry %d/%d in %.0fs", r.status_code, attempt, MAX_RETRIES, delay)
                time.sleep(delay)
                continue
            # Other 4xx: bad key, bad params — retrying won't help
            raise RuntimeError(f"Fingrid HTTP {r.status_code}: {r.text[:300]}")
        raise RuntimeError(f"Fingrid request failed after {MAX_RETRIES} attempts: {params}")

    def fetch_range(
        self, dataset_id: int, start: pd.Timestamp, end: pd.Timestamp, record_filter: dict | None = None
    ) -> pd.DataFrame:
        """All records with start <= startTime < end, following pagination.

        record_filter: keep only records whose additionalJson matches every key/value
        (for datasets with several series per timestamp, e.g. 362 by ProductionType).
        """
        records: list[dict] = []
        page = 1
        while True:
            body = self.get_page(dataset_id, start, end, page)
            records.extend(body.get("data", []))
            next_page = (body.get("pagination") or {}).get("nextPage")
            if not next_page:
                break
            page = next_page
        if record_filter:
            records = [r for r in records
                       if all(str((r.get("additionalJson") or {}).get(k)) == str(v) for k, v in record_filter.items())]
        return _records_to_frame(records)


def fetch_dataset(
    dataset_id: int,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    client: FingridClient | None = None,
    refresh: bool = False,
    record_filter: dict | None = None,
) -> pd.DataFrame:
    """Fetch [start, end) at native resolution, one cached Parquet file per month.

    Completed months are cached under data/raw/fingrid/<id>[_<filter>]/YYYY-MM.parquet and
    skipped on later runs. Months that are not yet over are never cached.
    """
    start, end = _utc(start), _utc(end)
    suffix = "".join(f"_{k}-{v}" for k, v in sorted((record_filter or {}).items()))
    cache_dir = RAW_DIR / f"{dataset_id}{suffix}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")

    frames = []
    for m_start in pd.date_range(start.floor("D").replace(day=1), end, freq="MS", inclusive="left"):
        m_end = m_start + pd.offsets.MonthBegin(1)
        path = cache_dir / f"{m_start:%Y-%m}.parquet"
        if path.exists() and not refresh:
            df = pd.read_parquet(path)
        else:
            client = client or FingridClient()
            log.info("Fetching dataset %d for %s", dataset_id, f"{m_start:%Y-%m}")
            df = client.fetch_range(dataset_id, m_start, min(m_end, now), record_filter)
            if m_end <= now:
                df.to_parquet(path)
        frames.append(df)

    if not frames:
        return _records_to_frame([])
    out = pd.concat(frames)
    return out[(out.index >= start) & (out.index < end)]


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Hourly mean of interval values on a complete UTC hourly index.

    Three kinds of record are handled, in increasing priority per hour:
    - real-time samples (interval < 15 min): Fingrid's 3-min series publish each
      sample twice (a 1-min interval and a zero-length record), so values are first
      averaged per 3-min slot, then per hour; 20 slots make a complete hour.
    - 15-min intervals: 4 make a complete hour.
    - full-hour records: win over the others (Fingrid publishes both hourly and
      15-min records at the 2023-06-13 resolution switch).
    `n_intervals` counts the intervals (or slots) used; `complete` is False where an
    hour has fewer than its resolution implies (or none at all).
    """
    if df.empty:
        return pd.DataFrame(columns=["value", "n_intervals", "complete"])
    interval_min = (df["end_time"] - df.index).dt.total_seconds() / 60
    is_sub = (interval_min >= 15) & (interval_min < 60)
    rt, sub, full = df[interval_min < 15], df[is_sub], df[interval_min >= 60]

    parts = []
    if len(rt):
        slots = rt["value"].resample(SAMPLE_PERIOD).mean().dropna()
        g = slots.resample("1h")
        parts.append(pd.DataFrame({"value": g.mean(), "n_intervals": g.count(), "expected": 60 // SAMPLE_PERIOD_MIN}))
    if len(sub):
        g = sub["value"].resample("1h")
        expected = (60 / interval_min[is_sub]).resample("1h").max()
        parts.append(pd.DataFrame({"value": g.mean(), "n_intervals": g.count(), "expected": expected}))
    if len(full):
        full_h = full["value"][full.index == full.index.floor("1h")]
        parts.append(pd.DataFrame({"value": full_h, "n_intervals": 1, "expected": 1}))

    hourly = pd.concat([p[p["n_intervals"] > 0] for p in parts])
    hourly = hourly[~hourly.index.duplicated(keep="last")].sort_index()
    hourly["complete"] = hourly["n_intervals"] >= hourly.pop("expected")
    hourly = hourly.reindex(pd.date_range(hourly.index.min(), hourly.index.max(), freq="1h"))
    hourly["n_intervals"] = hourly["n_intervals"].fillna(0).astype(int)
    hourly["complete"] = hourly["complete"].fillna(False).astype(bool)
    hourly.index.name = "time_utc"
    return hourly


def _records_to_frame(records: list[dict]) -> pd.DataFrame:
    if not records:
        idx = pd.DatetimeIndex([], tz="UTC", name="start_time")
        return pd.DataFrame({"end_time": pd.Series(dtype="datetime64[ns, UTC]"), "value": pd.Series(dtype=float)}, index=idx)
    df = pd.DataFrame.from_records(records)
    df = pd.DataFrame(
        {
            "start_time": pd.to_datetime(df["startTime"], utc=True),
            "end_time": pd.to_datetime(df["endTime"], utc=True),
            "value": pd.to_numeric(df["value"], errors="coerce").astype(float),
        }
    ).set_index("start_time")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _utc(t: str | pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _iso(t: pd.Timestamp) -> str:
    return t.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _backoff(attempt: int) -> float:
    return min(5 * 2 ** (attempt - 1), 300)


def _retry_after(r: requests.Response) -> float | None:
    try:
        return float(r.headers["Retry-After"])
    except (KeyError, ValueError):
        return None


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def hourly_path(key: str | int, start: str | pd.Timestamp, end: str | pd.Timestamp) -> Path:
    return HOURLY_DIR / f"{key}_{_utc(start):%Y%m%d}_{_utc(end):%Y%m%d}.parquet"


def build_hourly(
    dataset_id: int,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    client: FingridClient | None = None,
    refresh: bool = False,
    missing_as_zero: bool = False,
    record_filter: dict | None = None,
    scale: float = 1.0,
) -> pd.DataFrame:
    """Fetch a dataset and return it hourly on the full [start, end) UTC index.

    missing_as_zero: hours without data get value 0 and count as complete (for
    discontinued links such as FI-RU).
    record_filter: see FingridClient.fetch_range.
    scale: multiply values (e.g. 0.001 for kWh per hour -> MW).
    """
    start, end = _utc(start), _utc(end)
    raw = fetch_dataset(dataset_id, start, end, client=client, refresh=refresh, record_filter=record_filter)
    hourly = to_hourly(raw)
    hourly["value"] = hourly["value"] * scale
    idx = pd.date_range(start, end, freq="1h", inclusive="left", name="time_utc")
    hourly = hourly.reindex(idx)
    hourly["n_intervals"] = hourly["n_intervals"].fillna(0).astype(int)
    hourly["complete"] = hourly["complete"].fillna(False).astype(bool)
    if missing_as_zero:
        empty = hourly["value"].isna()
        hourly.loc[empty, "value"] = 0.0
        hourly.loc[empty, "complete"] = True
    return hourly


def build_derived(spec: dict, parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """sign * sum of component series; an hour is complete only if every component is."""
    values = pd.concat({k: parts[k]["value"] for k in spec["sum_of"]}, axis=1)
    complete = pd.concat({k: parts[k]["complete"] for k in spec["sum_of"]}, axis=1)
    return pd.DataFrame(
        {
            "value": spec.get("sign", 1) * values.sum(axis=1, min_count=len(spec["sum_of"])),
            "n_intervals": complete.sum(axis=1),
            "complete": complete.all(axis=1),
        }
    )


def fetch_named(names: list[str], start: str, end: str, refresh: bool = False) -> None:
    """Fetch config datasets (and any derived series they make up) and write hourly Parquet."""
    cfg = load_config()
    datasets, derived = cfg["datasets"], cfg.get("derived", {})
    client = FingridClient()
    HOURLY_DIR.mkdir(parents=True, exist_ok=True)

    built: dict[str, pd.DataFrame] = {}
    for name in names:
        ds = datasets[name]
        log.info("== %s (dataset %d)", name, ds["id"])
        hourly = build_hourly(ds["id"], start, end, client, refresh, ds.get("missing_as_zero", False),
                              ds.get("record_filter"), ds.get("scale", 1.0))
        _write(hourly, hourly_path(name, start, end))
        built[name] = hourly

    for name, spec in derived.items():
        if all(k in built for k in spec["sum_of"]):
            _write(build_derived(spec, built), hourly_path(name, start, end))


def _write(hourly: pd.DataFrame, path: Path) -> None:
    hourly.to_parquet(path)
    log.info("Wrote %d hourly rows to %s (%d incomplete hours)", len(hourly), path, int((~hourly["complete"]).sum()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--all", action="store_true", help="every dataset in config/fingrid_datasets.yaml (except fetch_all: false)")
    which.add_argument("--name", nargs="+", help="short name(s) from the config, e.g. wind solar")
    which.add_argument("--dataset", type=int, help="any Fingrid dataset ID, e.g. 124")
    p.add_argument("--start", required=True, help="UTC start (inclusive), e.g. 2024-01-01")
    p.add_argument("--end", required=True, help="UTC end (exclusive), e.g. 2025-01-01")
    p.add_argument("--refresh", action="store_true", help="ignore cached months")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.dataset is not None:
        HOURLY_DIR.mkdir(parents=True, exist_ok=True)
        hourly = build_hourly(args.dataset, args.start, args.end, refresh=args.refresh)
        _write(hourly, hourly_path(args.dataset, args.start, args.end))
        return
    datasets = load_config()["datasets"]
    names = [n for n, d in datasets.items() if d.get("fetch_all", True)] if args.all else args.name
    unknown = set(names) - set(datasets)
    if unknown:
        p.error(f"unknown dataset name(s): {', '.join(sorted(unknown))}")
    fetch_named(names, args.start, args.end, refresh=args.refresh)


if __name__ == "__main__":
    main()

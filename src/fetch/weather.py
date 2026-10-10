"""Fetch hourly weather from the Open-Meteo Historical Weather API (ERA5).

Data: Open-Meteo.com, CC BY 4.0; ERA5 reanalysis (Copernicus Climate Change Service).

API facts (checked against open-meteo.com/en/docs/historical-weather-api):
- GET https://archive-api.open-meteo.com/v1/archive, no key for non-commercial use
  params: latitude, longitude, start_date, end_date (dates, inclusive), hourly=<vars>,
          wind_speed_unit=ms, timezone=GMT (default), models
- models=era5 is set explicitly: the default "best_match" blends in IFS HRES for recent
  years, which would make the source change over time. ERA5 is 0.25 deg (~25 km); the
  coordinates are snapped to the nearest grid cell (returned in the response).
- Wind and temperature are instantaneous at the timestamp. Radiation is the mean over
  the PRECEDING hour. Here radiation is shifted back one hour so every row is labelled by
  the start of its hour, like the Fingrid series (row 00:00 = 00:00-01:00 mean).
- ERA5 has a ~5 day delay.

Usage:
    python -m src.fetch.weather --start 2024-01-01 --end 2025-01-01
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
MODEL = "era5"
INSTANT_VARS = ["wind_speed_100m", "wind_direction_100m", "wind_speed_10m", "temperature_2m"]
RADIATION_VARS = ["shortwave_radiation", "direct_radiation", "diffuse_radiation", "direct_normal_irradiance"]
MIN_INTERVAL_S = 2.0  # polite; no hard limit for single-point year requests
MAX_RETRIES = 6
TIMEOUT_S = 120
DATA_DELAY = pd.Timedelta(days=7)  # ERA5 ~5 days, with margin

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "locations.yaml"
RAW_DIR = REPO_ROOT / "data" / "raw" / "weather"
HOURLY_DIR = REPO_ROOT / "data" / "hourly" / "weather"

log = logging.getLogger(__name__)


class OpenMeteoClient:
    def __init__(self):
        self.session = requests.Session()
        self._last_request = 0.0

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < MIN_INTERVAL_S:
            time.sleep(MIN_INTERVAL_S - elapsed)
        self._last_request = time.monotonic()

    def get(self, params: dict) -> dict:
        for attempt in range(1, MAX_RETRIES + 1):
            self._wait()
            try:
                r = self.session.get(BASE_URL, params=params, timeout=TIMEOUT_S)
            except requests.RequestException as e:
                delay = _backoff(attempt)
                log.warning("Request error (%s), retry %d/%d in %.0fs", e, attempt, MAX_RETRIES, delay)
                time.sleep(delay)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 or r.status_code >= 500:
                # 429 here means the minute/hour/day fair-use limit was hit
                delay = 60.0 if r.status_code == 429 else _backoff(attempt)
                log.warning("HTTP %d, retry %d/%d in %.0fs", r.status_code, attempt, MAX_RETRIES, delay)
                time.sleep(delay)
                continue
            raise RuntimeError(f"Open-Meteo HTTP {r.status_code}: {r.text[:300]}")
        raise RuntimeError(f"Open-Meteo request failed after {MAX_RETRIES} attempts: {params}")

    def fetch_year(self, lat: float, lon: float, year: int) -> pd.DataFrame:
        """One calendar year (UTC), radiation shifted to hour-start labels."""
        # one extra day so the shifted radiation covers the last hour of the year
        body = self.get(
            {
                "latitude": lat,
                "longitude": lon,
                "start_date": f"{year}-01-01",
                "end_date": f"{year + 1}-01-01",
                "hourly": ",".join(INSTANT_VARS + RADIATION_VARS),
                "wind_speed_unit": "ms",
                "timezone": "GMT",
                "models": MODEL,
            }
        )
        h = body["hourly"]
        df = pd.DataFrame({v: h[v] for v in INSTANT_VARS + RADIATION_VARS}, dtype=float)
        df.index = pd.DatetimeIndex(pd.to_datetime(h["time"]).tz_localize("UTC"), name="time_utc")
        df[RADIATION_VARS] = df[RADIATION_VARS].shift(-1)
        df = df[df.index.year == year]
        df.attrs = {
            "grid_lat": body["latitude"],
            "grid_lon": body["longitude"],
            "elevation_m": body.get("elevation"),
            "units": {v: body["hourly_units"][v] for v in INSTANT_VARS + RADIATION_VARS},
            "model": MODEL,
        }
        return df


def load_locations(path: Path = CONFIG_PATH) -> dict[str, dict[str, dict]]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_location(
    name: str,
    lat: float,
    lon: float,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    client: OpenMeteoClient | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """[start, end) hourly UTC for one point; one cached Parquet file per year.

    Years whose data may still be inside the ERA5 delay are never cached.
    """
    start, end = _utc(start), _utc(end)
    cache_dir = RAW_DIR / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")

    frames, attrs = [], {}
    for year in range(start.year, (end - pd.Timedelta("1h")).year + 1):
        path = cache_dir / f"{year}.parquet"
        if path.exists() and not refresh:
            df = pd.read_parquet(path)
        else:
            client = client or OpenMeteoClient()
            log.info("Fetching %s (%.2f, %.2f) for %d", name, lat, lon, year)
            df = client.fetch_year(lat, lon, year)
            if pd.Timestamp(f"{year + 1}-01-01", tz="UTC") + DATA_DELAY <= now:
                df.to_parquet(path)
        attrs = attrs or df.attrs
        frames.append(df)

    out = pd.concat(frames)
    out = out[(out.index >= start) & (out.index < end)]
    out.attrs = {**attrs, "location": name, "lat": lat, "lon": lon}
    return out


def hourly_path(name: str, start: str | pd.Timestamp, end: str | pd.Timestamp) -> Path:
    return HOURLY_DIR / f"{name}_{_utc(start):%Y%m%d}_{_utc(end):%Y%m%d}.parquet"


def fetch_all(start: str, end: str, refresh: bool = False) -> None:
    client = OpenMeteoClient()
    HOURLY_DIR.mkdir(parents=True, exist_ok=True)
    for group, points in load_locations().items():
        for name, p in points.items():
            df = fetch_location(name, p["lat"], p["lon"], start, end, client, refresh)
            df.attrs["group"] = group
            path = hourly_path(name, start, end)
            df.to_parquet(path)
            log.info("Wrote %d rows to %s (grid cell %.2f, %.2f)", len(df), path, df.attrs["grid_lat"], df.attrs["grid_lon"])


def _utc(t: str | pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _backoff(attempt: int) -> float:
    return min(5 * 2 ** (attempt - 1), 300)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True, help="UTC start (inclusive), e.g. 2024-01-01")
    p.add_argument("--end", required=True, help="UTC end (exclusive), e.g. 2025-01-01")
    p.add_argument("--refresh", action="store_true", help="ignore cached years")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fetch_all(args.start, args.end, refresh=args.refresh)


if __name__ == "__main__":
    main()

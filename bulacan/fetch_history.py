"""Download daily reanalysis weather and GloFAS river discharge for Bulacan."""

import argparse
import hashlib
import json
import time
from datetime import date, timedelta

import pandas as pd
import truststore

truststore.inject_into_ssl()
import requests  # noqa: E402

from common import CACHE_DIR, DAILY_FILE, RAIN_POINTS, RIVER_POINTS, START_DATE

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FLOOD_URL = "https://flood-api.open-meteo.com/v1/flood"
TIMEZONE = "Asia/Manila"
# ERA5 is published about five days behind real time.
ERA5_LAG_DAYS = 6
FORECAST_ARCHIVE_START = "2022-01-01"


def cached_get(url: str, params: dict) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps([url, params], sort_keys=True).encode()).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    for attempt in range(6):
        response = requests.get(url, params=params, timeout=180)
        if response.status_code != 429:
            break
        time.sleep(65 * (attempt + 1))
    response.raise_for_status()
    payload = response.json()
    path.write_text(json.dumps(payload), encoding="utf-8")
    time.sleep(1.0)
    return payload


def weather_series(url: str, name: str, lat: float, lon: float, start: str, end: str,
                   with_soil: bool, model: str | None) -> pd.DataFrame:
    params = {
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "daily": "precipitation_sum", "timezone": TIMEZONE,
    }
    if model:
        params["models"] = model
    if with_soil:
        params["hourly"] = "soil_moisture_0_to_7cm"

    payload = cached_get(url, params)
    daily = payload["daily"]
    frame = pd.DataFrame(
        {f"rain_{name}": daily["precipitation_sum"]},
        index=pd.to_datetime(daily["time"]),
    )
    if with_soil:
        hourly = payload["hourly"]
        soil = pd.Series(hourly["soil_moisture_0_to_7cm"],
                         index=pd.to_datetime(hourly["time"]), dtype=float)
        frame[f"soil_moisture_{name}"] = soil.resample("D").mean()
    return frame


def discharge_series(name: str, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "daily": "river_discharge", "timezone": TIMEZONE,
    }
    daily = cached_get(FLOOD_URL, params)["daily"]
    return pd.DataFrame(
        {f"discharge_{name}": daily["river_discharge"]},
        index=pd.to_datetime(daily["time"]),
    )


def collect(url: str, start: str, end: str, model: str | None) -> pd.DataFrame:
    frames = []
    for name, (lat, lon) in RAIN_POINTS.items():
        print(f"  weather {name}")
        frames.append(weather_series(url, name, lat, lon, start, end,
                                     with_soil=name == "calumpit", model=model))
    for name, (lat, lon) in RIVER_POINTS.items():
        print(f"  river {name}")
        frames.append(discharge_series(name, lat, lon, start, end))
    daily = pd.concat(frames, axis=1).sort_index()
    daily.index.name = "date"
    return daily


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default=str(date.today() - timedelta(days=ERA5_LAG_DAYS)))
    args = parser.parse_args()

    print(f"ERA5 and GloFAS: {START_DATE} to {args.end}")
    history = collect(ARCHIVE_URL, START_DATE, args.end, "era5")
    DAILY_FILE.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(DAILY_FILE)
    print(f"Wrote {DAILY_FILE} with {len(history):,} days")
    print(history.isna().sum()[lambda s: s > 0].to_string() or "No missing values")

    # This archive resembles what a live system can read, so it tests whether
    # the ERA5-trained model survives a switch of input source.
    print(f"Forecast-model archive: {FORECAST_ARCHIVE_START} to {args.end}")
    live_like = collect(HISTORICAL_FORECAST_URL, FORECAST_ARCHIVE_START, args.end, None)
    live_file = DAILY_FILE.with_name("bulacan_daily_forecast_archive.csv")
    live_like.to_csv(live_file)
    print(f"Wrote {live_file} with {len(live_like):,} days")


if __name__ == "__main__":
    main()

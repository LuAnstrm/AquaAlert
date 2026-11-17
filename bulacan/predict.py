"""Produce today's internal Bulacan flood signal and barangay review list."""

import argparse
from datetime import date

import joblib
import truststore

truststore.inject_into_ssl()
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from collect_gauges import PDRRMO_URL, USER_AGENT, parse_stations  # noqa: E402
from common import (HAZARD_FILE, MODEL_FILE, RAIN_POINTS, RIVER_POINTS,  # noqa: E402
                    build_features)
from round2 import FORECAST_CANDIDATE_FILE, proxy_forecast  # noqa: E402

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FLOOD_URL = "https://flood-api.open-meteo.com/v1/flood"
PAST_DAYS = 20
FORECAST_DAYS = 3
TIMEZONE = "Asia/Manila"


def fetch_recent() -> pd.DataFrame:
    frames = []
    for name, (lat, lon) in RAIN_POINTS.items():
        params = {"latitude": lat, "longitude": lon, "daily": "precipitation_sum",
                  "past_days": PAST_DAYS, "forecast_days": FORECAST_DAYS, "timezone": TIMEZONE}
        if name == "calumpit":
            params["hourly"] = "soil_moisture_0_to_7cm"
        response = requests.get(FORECAST_URL, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        frame = pd.DataFrame({f"rain_{name}": payload["daily"]["precipitation_sum"]},
                             index=pd.to_datetime(payload["daily"]["time"]))
        if name == "calumpit":
            soil = pd.Series(payload["hourly"]["soil_moisture_0_to_7cm"],
                             index=pd.to_datetime(payload["hourly"]["time"]), dtype=float)
            frame["soil_moisture_calumpit"] = soil.resample("D").mean()
        frames.append(frame)
    for name, (lat, lon) in RIVER_POINTS.items():
        params = {"latitude": lat, "longitude": lon, "daily": "river_discharge",
                  "past_days": PAST_DAYS, "forecast_days": FORECAST_DAYS, "timezone": TIMEZONE}
        response = requests.get(FLOOD_URL, params=params, timeout=60)
        response.raise_for_status()
        daily = response.json()["daily"]
        frames.append(pd.DataFrame({f"discharge_{name}": daily["river_discharge"]},
                                   index=pd.to_datetime(daily["time"])))
    return pd.concat(frames, axis=1).sort_index()


def official_gauge_status() -> pd.DataFrame:
    response = requests.get(PDRRMO_URL, timeout=60, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    gauges = pd.DataFrame(parse_stations(response.text))
    if gauges.empty:
        raise RuntimeError("No gauge rows found. The PDRRMO page layout may have changed.")
    gauges["status"] = "Below alert"
    gauges.loc[gauges["level_m"] >= gauges["alert_m"], "status"] = "ALERT"
    gauges.loc[gauges["level_m"] >= gauges["alarm_m"], "status"] = "ALARM"
    gauges.loc[gauges["level_m"] >= gauges["critical_m"], "status"] = "CRITICAL"
    return gauges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=15, help="Barangays to list when elevated.")
    args = parser.parse_args()

    bundle = joblib.load(MODEL_FILE)
    recent = fetch_recent()
    today = pd.Timestamp(date.today())
    issued = recent.loc[:today]
    features = build_features(issued).dropna()[bundle["features"]]
    if features.empty:
        raise RuntimeError("Recent data has gaps, so the model cannot run today.")
    latest = features.iloc[[-1]]
    if latest.index[0].date() != date.today():
        print(f"WARNING: latest complete inputs are from {latest.index[0].date()}.")
    score = float(bundle["model"].predict_proba(latest)[0, 1])
    elevated = score >= bundle["threshold"]

    print("INTERNAL TEST OUTPUT. NOT AN OFFICIAL WARNING. Follow PAGASA and Bulacan PDRRMO.")
    print("This model passed 0 of 5 development gates. Do not use it for decisions.")
    print(f"Issue date: {latest.index[0].date()}")
    print(f"Model signal: {'ELEVATED' if elevated else 'NORMAL'} "
          f"(score {score:.3f}, threshold {bundle['threshold']:.3f})")
    print("The score ranks risk and is not a calibrated probability.")

    if FORECAST_CANDIDATE_FILE.exists():
        candidate = joblib.load(FORECAST_CANDIDATE_FILE)
        # The fetched rows after today are the current forecast for the next two days.
        with_forecast = build_features(issued).join(pd.DataFrame(proxy_forecast(recent)))
        row = with_forecast.loc[[today], candidate["features"]]
        if row.notna().all(axis=None):
            candidate_score = float(candidate["model"].predict_proba(row)[0, 1])
            state = "ELEVATED" if candidate_score >= candidate["threshold"] else "NORMAL"
            print(f"Shadow signal with rain forecast: {state} (score {candidate_score:.3f}, "
                  f"threshold {candidate['threshold']:.3f}); also failed the gates, logged for testing only.")
            print(f"Forecast rain next 2 days: {row['fc_rain_local_next2d'].iloc[0]:.1f} mm local, "
                  f"{row['fc_rain_upstream_next2d'].iloc[0]:.1f} mm upstream.")

    print("\nOfficial PDRRMO gauges:")
    print(official_gauge_status()[["station", "level_m", "alert_m", "status", "reported_date"]]
          .to_string(index=False))

    if elevated and HAZARD_FILE.exists():
        hazard = pd.read_csv(HAZARD_FILE)
        columns = ["adm3_name", "adm4_name", "share_medium_25yr", "share_high_25yr", "exposure_score"]
        print(f"\nBarangays for human review, by NOAH 25-year hazard area (top {args.top}):")
        print(hazard.nlargest(args.top, "exposure_score")[columns].to_string(index=False))
        print("This list shows mapped exposure. It does not predict flooding in each barangay.")


if __name__ == "__main__":
    main()

"""
live_check.py
=============
AquaAlert – Test the trained models on REAL gauge data
-------------------------------------------------------
Pulls observed rainfall and river levels from PAGASA's Pasig-Marikina-Tullahan
Flood Forecasting and Warning System (FFWS) and soil moisture from Open-Meteo,
builds the same 19 features the models were trained on, predicts Low /
Moderate / High, and compares each prediction with what the river gauge
actually did against PAGASA's own alert / alarm levels.

DATA SOURCES
------------
  PAGASA FFWS  https://pasig-marikina-tullahanffws.pagasa.dost.gov.ph
               10-minute rainfall and water-level readings (the JSON feed the
               public website itself loads).  No key needed.  Requests are
               cached in data/live_cache/ so each day is downloaded only once.
  Open-Meteo   https://open-meteo.com  – daily soil moisture (0–7 cm).
               Free for non-commercial use, no key needed.

REFERENCE ("what really happened") PER CITY-DAY
-----------------------------------------------
  High     = the city's river gauge reached PAGASA's ALARM level that day
  Moderate = it reached the ALERT level
  Low      = it stayed below alert
This is an independent, official threshold – not the rule the training labels
were built from – so it is a fair real-world check.

PROTOTYPE ASSUMPTIONS (read before quoting results)
---------------------------------------------------
  * River level: FFWS reports elevation in metres above datum (e.g. Sto. Niño
    ≈ 12 m normal, 20 m+ in a typhoon).  The training data's WaterLevel_m
    (0.5–6.4 m) has no documented datum.  We feed the model the rise above the
    station's dry-season level (median on BASELINE_DATE) plus 0.5 m, so 0.5 m
    means "normal", like the training minimum.
  * Rainfall: each city uses one PAGASA rain gauge (see STATIONS).  Days the
    gauge was offline, or read 0 mm while Open-Meteo shows ≥ 10 mm, use
    Open-Meteo's modelled rainfall instead (column rain_source).
  * Only the four training cities are supported.

HOW TO RUN
----------
    python live_check.py                       # default real-event windows
    python live_check.py --start 2026-10-22 --end 2026-10-29
OUTPUT : reports/live_check_results.csv
"""

import argparse
import contextlib
import io
import json
import time
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import truststore
from sklearn.metrics import confusion_matrix, recall_score

# PAGASA's server does not send its intermediate certificate, so Python's
# bundled CA list cannot verify it.  truststore uses the operating system's
# certificate store instead (like a browser), keeping TLS verification ON.
truststore.inject_into_ssl()
import requests  # noqa: E402  (must be imported after inject_into_ssl)

import preprocess
from build_real_dataset import CITY_CENTRES

FFWS_URL      = "https://pasig-marikina-tullahanffws.pagasa.dost.gov.ph"
ARCHIVE_URL   = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL  = "https://api.open-meteo.com/v1/forecast"
FEATURES_FILE = Path("data/processed/aquaalert_features.csv")
MODEL_FILES   = {
    "Random Forest":       Path("models/rf_pipeline_final.pkl"),
    "Logistic Regression": Path("models/baseline_lr_pipeline.pkl"),
}
CACHE_DIR     = Path("data/live_cache")
OUT_CSV       = Path("reports/live_check_results.csv")

CLASS_ORDER   = ["Low", "Moderate", "High"]
HISTORY_DAYS  = 7
REQUEST_PAUSE = 0.3             # seconds between live requests (be polite)
BASELINE_DATE = date(2025, 4, 15)   # dry-season day used as "normal" river level
LEVEL_FLOOR_M = 0.5             # training data's minimum WaterLevel_m

# One PAGASA FFWS rain gauge and one water-level gauge per training city.
STATIONS = {
    "Manila":      {"rain": "11204301", "water": "11204201"},  # Fort Santiago / Pandacan
    "Quezon City": {"rain": "11203101", "water": "11203201"},  # Science Garden / Mindanao
    "Marikina":    {"rain": "11103107", "water": "11104204"},  # Youth Camp / Sto Nino (MMDA)
    "Pasig":       {"rain": "11201101", "water": "11202201"},  # Napindan / Napindan-2
}

# Real events to check by default (inclusive date ranges).
DEFAULT_WINDOWS = [
    ("Typhoon Carina + habagat", date(2024, 7, 20), date(2024, 7, 27)),
    ("Dry season",               date(2025, 3, 1),  date(2025, 3, 7)),
    ("Habagat, July 2025",       date(2025, 7, 18), date(2025, 7, 25)),
]


# =============================================================================
# DATA FETCHING
# =============================================================================

def _to_float(value) -> float:
    """FFWS values look like '12.34' or '12.34(*)' (flagged); '0.00' = no data."""
    if value in (None, "", "-"):
        return np.nan
    v = float(str(value).replace("(*)", ""))
    return np.nan if v == 0.0 else v


def ffws_day(kind: str, obscd: str, day: date) -> list[dict]:
    """All 10-minute readings for one station on one calendar day (cached)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{kind}_{obscd}_{day:%Y%m%d}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    resp = requests.post(
        f"{FFWS_URL}/{kind}/detail_list.do",
        data={"obscd": obscd, "ymdhm": f"{day:%Y%m%d}2350"},
        timeout=30,
    )
    resp.raise_for_status()
    rows = [r for r in resp.json() if r["timestr"].startswith(f"{day:%Y-%m-%d}")]
    if day < date.today():          # today's data is still growing – don't cache
        cache.write_text(json.dumps(rows), encoding="utf-8")
    time.sleep(REQUEST_PAUSE)
    return rows


def daily_rain_mm(obscd: str, day: date) -> float:
    """Sum of 10-minute rainfall for the day; NaN if the gauge was mostly offline."""
    rows = ffws_day("rainfall", obscd, day)
    vals = [float(str(r["rf"]).replace("(*)", "")) for r in rows
            if r.get("rf") not in (None, "", "-")]
    return float(np.sum(vals)) if len(vals) >= 72 else np.nan   # ≥ 12 h of data


def daily_max_level(obscd: str, day: date, alert: float) -> float:
    """
    Highest level of the day; NaN if mostly offline (< 12 h of readings) or
    stuck: a river at or above its alert level never stays perfectly still for
    a whole day, so a flat line there means a frozen sensor.  (A flat line
    below alert is normal in the dry season and is kept.)
    """
    rows = ffws_day("water", obscd, day)
    vals = [_to_float(r["wl"]) for r in rows]
    vals = [v for v in vals if not np.isnan(v)]
    if len(vals) < 72 or (max(vals) == min(vals) and max(vals) >= alert):
        return np.nan
    return max(vals)


def baseline_level(obscd: str) -> float:
    rows = ffws_day("water", obscd, BASELINE_DATE)
    vals = [_to_float(r["wl"]) for r in rows]
    return float(np.nanmedian(vals))


def station_thresholds() -> dict[str, dict]:
    """Current alert / alarm / critical levels published by FFWS."""
    resp = requests.get(f"{FFWS_URL}/water/main_list.do", timeout=30)
    resp.raise_for_status()
    return {s["obscd"]: {k: _to_float(s[k]) for k in ("alertwl", "alarmwl", "criticalwl")}
            for s in resp.json()}


def open_meteo_daily(city: str, start: date, end: date) -> pd.DataFrame:
    """
    Daily mean 0–7 cm soil moisture (%) and modelled precipitation (mm) from
    Open-Meteo.  The precipitation is only used to spot a rain gauge that
    reports 0 mm while it is clearly raining (gauge offline).
    """
    lat, lon = CITY_CENTRES[city]
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": f"{start}", "end_date": f"{end}",
        "hourly": "soil_moisture_0_to_7cm", "daily": "precipitation_sum",
        "timezone": "Asia/Manila",
    }
    # The archive lags real time by about a week; recent days use the forecast API.
    url = ARCHIVE_URL if end < date.today() - timedelta(days=7) else FORECAST_URL
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    js = resp.json()
    soil = pd.Series(js["hourly"]["soil_moisture_0_to_7cm"],
                     index=pd.to_datetime(js["hourly"]["time"]), dtype=float)
    out = pd.DataFrame({
        "soil_moisture_pct": soil.resample("D").mean() * 100,
        "om_precip_mm": pd.Series(js["daily"]["precipitation_sum"],
                                  index=pd.to_datetime(js["daily"]["time"]), dtype=float),
    })
    out.index = out.index.date
    return out


# =============================================================================
# FEATURES + PREDICTION
# =============================================================================

def city_constants() -> pd.DataFrame:
    """Per-city static columns exactly as used in training."""
    cols = ["municipality", "elevation_m", "aegis_mean_flood_height_code",
            "aegis_share_deep_flood", "aegis_mean_elevation_m", "flood_hazard_level"]
    return (pd.read_csv(FEATURES_FILE, usecols=cols)
            .drop_duplicates("municipality").set_index("municipality"))


def build_rows(city: str, start: date, end: date, thresholds: dict) -> list[dict]:
    st    = STATIONS[city]
    base  = baseline_level(st["water"])
    days  = pd.date_range(start - timedelta(days=HISTORY_DAYS - 1), end).date
    om    = open_meteo_daily(city, days[0], end)
    rain  = pd.Series({d: daily_rain_mm(st["rain"], d) for d in days})
    # A gauge that is offline (NaN) or reads exactly 0 mm while Open-Meteo
    # shows ≥ 10 mm is replaced by Open-Meteo's modelled rainfall and flagged.
    om_rain  = om["om_precip_mm"].reindex(rain.index)
    fallback = rain.isna() | ((rain == 0) & (om_rain >= 10))
    rain[fallback] = om_rain[fallback]
    thr   = thresholds.get(st["water"], {})
    alert = thr.get("alertwl", np.inf)
    level = pd.Series({d: daily_max_level(st["water"], d, alert)
                       for d in days if d >= start - timedelta(days=1)})

    def model_level(raw: float) -> float:
        if np.isnan(raw):
            return np.nan
        return max(LEVEL_FLOOR_M, raw - base + LEVEL_FLOOR_M)

    rows = []
    for d in pd.date_range(start, end).date:
        hist = rain[[x for x in days if x <= d]].tail(HISTORY_DAYS)
        today_lvl, prev_lvl = level.get(d, np.nan), level.get(d - timedelta(days=1), np.nan)

        reference = None                     # unknown when the gauge has no data
        if not np.isnan(today_lvl):
            reference = "Low"
            if today_lvl >= thr.get("alarmwl", np.inf):
                reference = "High"
            elif today_lvl >= thr.get("alertwl", np.inf):
                reference = "Moderate"

        rows.append({
            "timestamp": pd.Timestamp(d), "municipality": city,
            "rainfall_last_24h_mm": hist.iloc[-1],
            "rainfall_prev_day_mm": hist.iloc[-2],
            "rainfall_last_3d_mm":  hist.tail(3).sum(min_count=3),
            "rainfall_last_7d_mm":  hist.sum(min_count=HISTORY_DAYS),
            "river_water_level_m":  model_level(today_lvl),
            "water_level_change_1d_m": model_level(today_lvl) - model_level(prev_lvl),
            "soil_moisture_pct":    om["soil_moisture_pct"].get(d, np.nan),
            "rain_source":          "open-meteo" if fallback[d] else "pagasa-gauge",
            "gauge_level_elm":      today_lvl,
            "gauge_alert_elm":      thr.get("alertwl"),
            "gauge_alarm_elm":      thr.get("alarmwl"),
            "reference_risk":       reference,
        })
    return rows


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same encoding / feature engineering as preprocess.py, then predict."""
    consts = city_constants()
    df = df.join(consts, on="municipality")
    with contextlib.redirect_stdout(io.StringIO()):      # silence step banners
        df = preprocess.parse_timestamps(df)
        df = preprocess.encode_categoricals(df)
        df = preprocess.engineer_features(df)

    for name, path in MODEL_FILES.items():
        model = joblib.load(path)
        cols  = list(model.feature_names_in_)
        ok    = df[cols].notna().all(axis=1)
        short = "rf" if "Forest" in name else "lr"
        df[f"{short}_pred"] = pd.Series(pd.NA, index=df.index, dtype="object")
        df.loc[ok, f"{short}_pred"] = model.predict(df.loc[ok, cols])
        if "High" in model.classes_:
            hi = list(model.classes_).index("High")
            df.loc[ok, f"{short}_p_high"] = model.predict_proba(df.loc[ok, cols])[:, hi].round(3)
    return df


def summarise(df: pd.DataFrame) -> None:
    usable = df.dropna(subset=["rf_pred", "reference_risk"])
    print(f"\n  Usable city-days: {len(usable)} of {len(df)} "
          f"(others had river- or rain-gauge gaps)")
    for short, name in [("rf", "Random Forest"), ("lr", "Logistic Regression")]:
        y, p = usable["reference_risk"], usable[f"{short}_pred"].astype(str)
        print(f"\n  {name}: agreement with gauge reference = {(y == p).mean():.2f}")
        rec = recall_score(y, p, labels=CLASS_ORDER, average=None, zero_division=0)
        for lbl, r in zip(CLASS_ORDER, rec):
            print(f"    {lbl:<9} recall {r:.2f}  (n={int((y == lbl).sum())})")
        print(pd.DataFrame(confusion_matrix(y, p, labels=CLASS_ORDER),
                           index=[f"true {c}" for c in CLASS_ORDER],
                           columns=[f"pred {c}" for c in CLASS_ORDER]).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--start", type=date.fromisoformat)
    ap.add_argument("--end",   type=date.fromisoformat)
    args = ap.parse_args()
    windows = ([("Custom", args.start, args.end)] if args.start and args.end
               else DEFAULT_WINDOWS)

    thresholds = station_thresholds()
    rows = []
    for label, start, end in windows:
        print(f"Fetching {label}: {start} → {end}")
        for city in STATIONS:
            for r in build_rows(city, start, end, thresholds):
                rows.append({"window": label, **r})

    df = predict(pd.DataFrame(rows))
    keep = ["window", "timestamp", "municipality", "rain_source", "rainfall_last_24h_mm",
            "rainfall_last_7d_mm", "soil_moisture_pct", "gauge_level_elm",
            "gauge_alert_elm", "gauge_alarm_elm", "river_water_level_m",
            "reference_risk", "rf_pred", "rf_p_high", "lr_pred", "lr_p_high"]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[keep].to_csv(OUT_CSV, index=False)
    print(f"\n  Results → {OUT_CSV}")
    summarise(df)


if __name__ == "__main__":
    main()

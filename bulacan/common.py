"""Shared settings and feature engineering for the Bulacan flood model."""

from pathlib import Path

import numpy as np
import pandas as pd

import tide

ROOT = Path(__file__).resolve().parents[1]
EVENTS_FILE = ROOT / "data/external/bulacan/flood_events.csv"
NOAH_DIR = ROOT / "data/external/noah"
BOUNDARY_ZIP = ROOT / "data/external/boundaries/phl_admin_boundaries.gdb.zip"
CACHE_DIR = ROOT / "data/external/open_meteo_cache"
DAILY_FILE = ROOT / "data/processed/bulacan_daily.csv"
HAZARD_FILE = ROOT / "data/processed/bulacan_barangay_hazard.csv"
REPORT_DIR = ROOT / "reports/bulacan"
MODEL_FILE = ROOT / "models/bulacan_flood_model.pkl"
GAUGE_LOG = ROOT / "data/live_gauges/bulacan_pdrrmo.csv"

START_DATE = "2011-01-01"
HOLDOUT_YEAR = 2026
# A model issued on day t scores flooding reported on t+1 or t+2.
HORIZON_DAYS = 2

# Rain points cover the local area and the upstream catchments that send water
# into Bulacan. PRFFWC's 2020 post-flood report describes this routing.
RAIN_POINTS = {
    "calumpit": (14.91, 120.77),
    "malolos": (14.85, 120.81),
    "san_miguel": (15.15, 120.98),
    "sjdm": (14.81, 121.05),
    "angat_watershed": (14.95, 121.25),
    "upper_pampanga": (15.60, 121.05),
}

# GloFAS only resolves large channels, so each cell was picked by scanning nearby
# cells for peak 2020 discharge. Nearby coordinates can land on a tiny tributary.
RIVER_POINTS = {
    "pampanga_main": (15.075, 120.775),  # peak about 2,990 m3/s
    "angat_lower": (14.925, 120.925),  # peak about 640 m3/s
    "san_miguel": (15.175, 120.925),  # peak about 180 m3/s
    "marilao": (14.775, 120.875),  # peak about 135 m3/s
}

SYNODIC_MONTH_DAYS = 29.530588
KNOWN_NEW_MOON = pd.Timestamp("2000-01-06 18:14")


def load_catalog() -> pd.DataFrame:
    catalog = pd.read_csv(EVENTS_FILE, parse_dates=["start", "end"], encoding="utf-8")
    if (catalog["end"] < catalog["start"]).any():
        raise ValueError("Catalog has a row whose end is before its start.")
    return catalog


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Build features for each issue date using only values up to that date."""
    df = daily.sort_index().copy()
    out = pd.DataFrame(index=df.index)

    for point in RAIN_POINTS:
        rain = df[f"rain_{point}"]
        out[f"rain_1d_{point}"] = rain
        out[f"rain_3d_{point}"] = rain.rolling(3, min_periods=3).sum()
        out[f"rain_7d_{point}"] = rain.rolling(7, min_periods=7).sum()
        out[f"rain_14d_{point}"] = rain.rolling(14, min_periods=14).sum()

    for point in RIVER_POINTS:
        flow = df[f"discharge_{point}"]
        out[f"discharge_{point}"] = np.log1p(flow)
        out[f"discharge_rise_3d_{point}"] = np.log1p(flow) - np.log1p(flow.shift(3))

    # Backfloods can arrive days after upstream rain, so keep recent peaks.
    for point in RIVER_POINTS:
        out[f"discharge_peak_7d_{point}"] = np.log1p(df[f"discharge_{point}"]).rolling(7, min_periods=7).max()
    local = ["calumpit", "malolos", "sjdm"]
    upstream = ["san_miguel", "angat_watershed", "upper_pampanga"]
    out["rain_1d_max_all"] = df[[f"rain_{p}" for p in RAIN_POINTS]].max(axis=1)
    out["rain_3d_local_mean"] = df[[f"rain_{p}" for p in local]].mean(axis=1).rolling(3, min_periods=3).sum()
    out["rain_7d_local_mean"] = df[[f"rain_{p}" for p in local]].mean(axis=1).rolling(7, min_periods=7).sum()
    out["rain_3d_upstream_mean"] = df[[f"rain_{p}" for p in upstream]].mean(axis=1).rolling(3, min_periods=3).sum()
    out["rain_7d_upstream_mean"] = df[[f"rain_{p}" for p in upstream]].mean(axis=1).rolling(7, min_periods=7).sum()
    out["rain_14d_upstream_mean"] = df[[f"rain_{p}" for p in upstream]].mean(axis=1).rolling(14, min_periods=14).sum()

    out["soil_moisture_calumpit"] = df["soil_moisture_calumpit"]

    # Spring tides follow the moon, so this feature needs no tide gauge.
    days = (df.index - KNOWN_NEW_MOON) / pd.Timedelta(days=1)
    lunar_angle = 2 * np.pi * (np.asarray(days) % SYNODIC_MONTH_DAYS) / SYNODIC_MONTH_DAYS
    out["spring_tide_index"] = np.cos(2 * lunar_angle)
    # Astronomical tides are known in advance, so this uses no future observations.
    out["tide_high_next2d"] = tide.next_days_high(df.index, HORIZON_DAYS)

    day_of_year = df.index.day_of_year.to_numpy()
    out["season_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    out["season_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return out


def build_labels(index: pd.DatetimeIndex, catalog: pd.DataFrame) -> pd.DataFrame:
    """Mark reported flood days and dates whose outcome cannot be trusted."""
    flood_day = pd.Series(False, index=index)
    masked_day = pd.Series(False, index=index)
    for row in catalog.itertuples():
        days = index[(index >= row.start) & (index <= row.end)]
        if row.kind == "event":
            flood_day.loc[days] = True
        elif row.kind == "mask":
            masked_day.loc[days] = True

    future_flood = pd.concat(
        [flood_day.shift(-h, fill_value=False) for h in range(1, HORIZON_DAYS + 1)],
        axis=1,
    ).any(axis=1)
    future_masked = pd.concat(
        [masked_day.shift(-h, fill_value=False) for h in range(1, HORIZON_DAYS + 1)],
        axis=1,
    ).any(axis=1)

    # A positive outcome is still informative when one horizon day is masked.
    usable = future_flood | ~future_masked
    return pd.DataFrame({"flood_day": flood_day, "target": future_flood, "usable": usable})

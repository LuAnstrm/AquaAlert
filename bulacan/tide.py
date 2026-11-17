"""Fit and predict Manila tides from the UHSLC gauge using harmonic analysis."""

import json

import numpy as np
import pandas as pd

from common_paths import TIDE_CSV, TIDE_MODEL

MISSING_VALUE = -30000
FIT_START = "2005-01-01"
FIT_END = "2025-12-31"
EPOCH = pd.Timestamp("2000-01-01")
MANILA_UTC_OFFSET = pd.Timedelta(hours=8)
NODAL_SPEED = 360 / (18.613 * 365.25 * 24)

# Constituent speeds are in degrees per hour. K1 and O1 dominate Manila's tide,
# so they also get sidebands for the 18.6-year nodal cycle.
SPEEDS = {
    "M2": 28.9841042, "S2": 30.0, "N2": 28.4397295, "K2": 30.0821373,
    "K1": 15.0410686, "O1": 13.9430356, "P1": 14.9589314, "Q1": 13.3986609,
    "Mf": 1.0980331, "Mm": 0.5443747, "Ssa": 0.0821373, "Sa": 0.0410686,
    "K1_nodal_up": 15.0410686 + NODAL_SPEED, "K1_nodal_down": 15.0410686 - NODAL_SPEED,
    "O1_nodal_up": 13.9430356 + NODAL_SPEED, "O1_nodal_down": 13.9430356 - NODAL_SPEED,
}


def load_observed() -> pd.Series:
    """Return hourly sea level in metres, indexed by UTC."""
    raw = pd.read_csv(TIDE_CSV, header=None, names=["year", "month", "day", "hour", "mm"])
    index = pd.to_datetime(raw[["year", "month", "day", "hour"]])
    level = pd.Series(raw["mm"].to_numpy(dtype=float), index=index, name="sea_level_m")
    return level[level > MISSING_VALUE] / 1000.0


def design_matrix(times: pd.DatetimeIndex) -> np.ndarray:
    hours = np.asarray((times - EPOCH) / pd.Timedelta(hours=1), dtype=float)
    columns = [np.ones(len(hours)), hours / (24 * 365.25)]
    for speed in SPEEDS.values():
        angle = np.radians(speed) * hours
        columns += [np.cos(angle), np.sin(angle)]
    return np.column_stack(columns)


def fit() -> dict:
    observed = load_observed().loc[FIT_START:FIT_END]
    coefficients, *_ = np.linalg.lstsq(design_matrix(observed.index), observed.to_numpy(), rcond=None)
    model = {"coefficients": coefficients.tolist(), "constituents": list(SPEEDS),
             "fit_start": FIT_START, "fit_end": FIT_END,
             "source": "UHSLC Manila station 370 fast-delivery hourly sea level"}
    TIDE_MODEL.parent.mkdir(parents=True, exist_ok=True)
    TIDE_MODEL.write_text(json.dumps(model), encoding="utf-8")
    return model


def predict_hourly(times_utc: pd.DatetimeIndex, model: dict | None = None) -> pd.Series:
    model = model or json.loads(TIDE_MODEL.read_text(encoding="utf-8"))
    return pd.Series(design_matrix(times_utc) @ np.asarray(model["coefficients"]), index=times_utc)


def predict_periodic(times_utc: pd.DatetimeIndex, model: dict | None = None) -> pd.Series:
    """Predict the tide with the long-term sea-level trend removed."""
    model = model or json.loads(TIDE_MODEL.read_text(encoding="utf-8"))
    coefficients = np.asarray(model["coefficients"]).copy()
    coefficients[1] = 0.0
    return pd.Series(design_matrix(times_utc) @ coefficients, index=times_utc)


def next_days_high(dates: pd.DatetimeIndex, horizon_days: int) -> pd.Series:
    """Return the predicted highest tide in the local days after each issue date."""
    first = dates.min() - MANILA_UTC_OFFSET
    last = dates.max() + pd.Timedelta(days=horizon_days + 1) - MANILA_UTC_OFFSET
    hourly = predict_hourly(pd.date_range(first, last, freq="h"))
    daily_high = hourly.groupby((hourly.index + MANILA_UTC_OFFSET).normalize()).max()
    future = pd.concat([daily_high.shift(-h) for h in range(1, horizon_days + 1)], axis=1).max(axis=1)
    return future.reindex(dates)

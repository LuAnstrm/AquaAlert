"""Turn AGOS ultrasonic readings into cleaned depth, alerts and likely causes."""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

INCH_M = 0.0254
# MMDA flood gauge in inches. Confirm these bands with the pilot LGU's DRRMO.
DEPTH_LEVELS = [
    (8, "gutter", "passable to all vehicles"),
    (10, "half-knee", "passable to all vehicles"),
    (13, "half-tire", "not passable to light vehicles"),
    (19, "knee", "not passable to light vehicles"),
    (26, "tire", "not passable to any vehicle"),
    (37, "waist", "not passable to any vehicle"),
    (45, "chest", "not passable to any vehicle"),
]
REFERENCE_TEMP_C = 20.0
ALERT_LEVELS = ["NORMAL", "WATCH", "WARNING", "DANGER"]


@dataclass
class NodeConfig:
    """Site settings to measure during installation. The defaults need field testing."""

    node_id: str
    mount_height_m: float
    max_step_m: float = 0.30
    measurement_noise_m: float = 0.03
    process_noise: float = 0.002
    stale_after_minutes: float = 15.0
    stuck_after_readings: int = 12
    watch_depth_m: float = 8 * INCH_M
    warning_depth_m: float = 13 * INCH_M
    danger_depth_m: float = 26 * INCH_M
    rapid_rise_m_per_h: float = 0.15
    projection_hours: float = 1.0


def temperature_corrected_distance(distance_m, air_temp_c, reference_c=REFERENCE_TEMP_C):
    """Correct distance when a sensor assumes sound speed at the reference temperature."""
    ratio = np.sqrt((np.asarray(air_temp_c, dtype=float) + 273.15) / (reference_c + 273.15))
    return np.asarray(distance_m, dtype=float) * ratio


def describe_depth(depth_m: float) -> tuple[str, str]:
    inches = depth_m / INCH_M
    if inches < DEPTH_LEVELS[0][0]:
        return "below gutter", "passable to all vehicles"
    label, passability = DEPTH_LEVELS[0][1:]
    for limit, name, passable in DEPTH_LEVELS:
        if inches >= limit:
            label, passability = name, passable
    return label, passability


def clean_readings(readings: pd.DataFrame, config: NodeConfig) -> pd.DataFrame:
    """Convert distance to depth and flag impossible, spiking or stuck readings."""
    frame = readings.sort_values("timestamp").reset_index(drop=True).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    distance = frame["distance_m"].to_numpy(dtype=float)
    if "air_temp_c" in frame:
        temperatures = frame["air_temp_c"].fillna(REFERENCE_TEMP_C)
        distance = temperature_corrected_distance(distance, temperatures)
    frame["raw_depth_m"] = config.mount_height_m - distance

    invalid = (frame["raw_depth_m"] < -0.05) | (frame["raw_depth_m"] > config.mount_height_m)
    median = frame["raw_depth_m"].where(~invalid).rolling(5, center=True, min_periods=2).median()
    spike = (frame["raw_depth_m"] - median).abs() > config.max_step_m
    frame["flag_invalid"] = invalid | frame["raw_depth_m"].isna()
    frame["flag_spike"] = spike.fillna(False)

    unchanged = frame["raw_depth_m"].diff().abs().fillna(1.0) < 1e-6
    run_id = (~unchanged).cumsum()
    run_length = unchanged.groupby(run_id).cumsum() + 1
    wet = frame["raw_depth_m"] > 0.02
    frame["flag_stuck"] = wet & (run_length >= config.stuck_after_readings)
    return frame


def kalman_depth(frame: pd.DataFrame, config: NodeConfig) -> pd.DataFrame:
    """Estimate depth and rise rate with a constant-velocity Kalman filter."""
    state, covariance = None, None
    last_time = None
    depths, rates = [], []
    reject = frame["flag_invalid"] | frame["flag_spike"] | frame["flag_stuck"]
    measurement_variance = config.measurement_noise_m ** 2

    for time, value, skip in zip(frame["timestamp"], frame["raw_depth_m"], reject):
        if state is None:
            if skip:
                depths.append(np.nan)
                rates.append(np.nan)
                continue
            state = np.array([max(float(value), 0.0), 0.0])
            covariance = np.array([[measurement_variance, 0.0], [0.0, 1.0]])
        else:
            hours = (time - last_time) / pd.Timedelta(hours=1)
            transition = np.array([[1.0, hours], [0.0, 1.0]])
            state = transition @ state
            q = config.process_noise
            noise = q * np.array([[hours ** 3 / 3, hours ** 2 / 2], [hours ** 2 / 2, hours]])
            covariance = transition @ covariance @ transition.T + noise
            if not skip:
                innovation = float(value) - state[0]
                denominator = covariance[0, 0] + measurement_variance
                gain = covariance[:, 0] / denominator
                state = state + gain * innovation
                covariance = covariance - np.outer(gain, covariance[0, :])
        last_time = time
        state[0] = max(state[0], 0.0)
        depths.append(state[0])
        rates.append(state[1])

    output = frame.copy()
    output["depth_m"] = depths
    output["rise_m_per_h"] = rates
    return output


def hours_until(depth_m: float, rate_m_per_h: float, target_m: float) -> float:
    if depth_m >= target_m:
        return 0.0
    if rate_m_per_h <= 0:
        return math.inf
    return (target_m - depth_m) / rate_m_per_h


def assess_latest(frame: pd.DataFrame, config: NodeConfig, now=None) -> dict:
    """Apply transparent rules to the latest reading, which also serve as the AI fallback."""
    usable = frame.dropna(subset=["depth_m"])
    if usable.empty:
        return {"node_id": config.node_id, "status": "NO_DATA", "alert": "UNKNOWN",
                "reason": "No valid reading yet"}
    latest = usable.iloc[-1]
    now = pd.Timestamp(now) if now is not None else frame["timestamp"].iloc[-1]
    age_minutes = (now - latest["timestamp"]) / pd.Timedelta(minutes=1)
    tail_flags = frame.tail(config.stuck_after_readings)
    stuck = bool(tail_flags["flag_stuck"].any())

    depth, rate = float(latest["depth_m"]), float(latest["rise_m_per_h"])
    level, passability = describe_depth(depth)
    alert, reasons = "NORMAL", []
    if depth >= config.danger_depth_m:
        alert = "DANGER"
        reasons.append("depth at or above danger level")
    elif depth >= config.warning_depth_m:
        alert = "WARNING"
        reasons.append("depth at or above warning level")
    elif depth >= config.watch_depth_m:
        alert = "WATCH"
        reasons.append("depth at or above watch level")

    to_warning = hours_until(depth, rate, config.warning_depth_m)
    if rate >= config.rapid_rise_m_per_h and to_warning <= config.projection_hours and alert in ("NORMAL", "WATCH"):
        alert = "WARNING"
        reasons.append(f"rapid rise projects warning depth in about {to_warning * 60:.0f} minutes")
    elif rate >= config.rapid_rise_m_per_h and alert == "NORMAL":
        alert = "WATCH"
        reasons.append("water is rising rapidly")

    status = "OK"
    if age_minutes > config.stale_after_minutes:
        status = "STALE"
        reasons.append(f"latest valid reading is {age_minutes:.0f} minutes old")
    if stuck:
        status = "CHECK_SENSOR"
        reasons.append("reading looks stuck; inspect for debris or a sensor fault")

    return {
        "node_id": config.node_id, "time": latest["timestamp"], "status": status, "alert": alert,
        "depth_m": round(depth, 3), "rise_m_per_h": round(rate, 3),
        "level": level, "passability": passability,
        "minutes_to_warning": None if math.isinf(to_warning) else round(to_warning * 60, 0),
        "reason": "; ".join(reasons) or "below watch level",
    }


def likely_causes(depth_m: float, rain_last_3h_mm: float, tide_height_percentile: float,
                  upstream_alert: bool, drain_slowdown_ratio: float | None = None) -> list[str]:
    """Suggest causes for operators to check. Field labels must validate these rules."""
    if depth_m < 0.02:
        return []
    causes = []
    if rain_last_3h_mm >= 10:
        causes.append("recent local rain")
    if tide_height_percentile >= 0.85:
        causes.append("high tide")
    if upstream_alert:
        causes.append("upstream or river backflow")
    if drain_slowdown_ratio is not None and drain_slowdown_ratio >= 2.0:
        causes.append("possible blocked drain; send camera or field check")
    return causes or ["unknown; review camera and citizen reports"]


def recession_rate(frame: pd.DataFrame) -> float | None:
    """Return median fall rate in metres per hour during falling periods."""
    falling = frame.loc[(frame["rise_m_per_h"] < -0.01) & (frame["depth_m"] > 0.02), "rise_m_per_h"]
    return None if falling.empty else float(-falling.median())


def drain_slowdown_ratio(recent: pd.DataFrame, node_history: pd.DataFrame) -> float | None:
    """Compare recent drainage with this node's history. Above 1 means slower than usual."""
    usual, current = recession_rate(node_history), recession_rate(recent)
    if usual is None or current is None or current <= 0:
        return None
    return usual / current


def process_node(readings: pd.DataFrame, config: NodeConfig, now=None) -> tuple[pd.DataFrame, dict]:
    filtered = kalman_depth(clean_readings(readings, config), config)
    return filtered, assess_latest(filtered, config, now)

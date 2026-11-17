"""Unit tests for street_node.py using made-up readings. Do not use these for training."""

import unittest

import numpy as np
import pandas as pd

from street_node import (INCH_M, NodeConfig, describe_depth, drain_slowdown_ratio, kalman_depth,
                         clean_readings, likely_causes, process_node,
                         temperature_corrected_distance)

MOUNT_M = 2.0


def readings_from_depths(depths, minutes=5):
    times = pd.date_range("2026-07-01 08:00", periods=len(depths), freq=f"{minutes}min")
    return pd.DataFrame({"timestamp": times, "distance_m": MOUNT_M - np.asarray(depths, dtype=float)})


class StreetNodeTests(unittest.TestCase):
    def setUp(self):
        self.config = NodeConfig("test-node", mount_height_m=MOUNT_M)

    def test_dry_road_is_normal(self):
        _, result = process_node(readings_from_depths(np.zeros(30)), self.config)
        self.assertEqual(result["alert"], "NORMAL")
        self.assertEqual(result["status"], "OK")

    def test_depth_scale_matches_mmda_labels(self):
        self.assertEqual(describe_depth(19 * INCH_M)[0], "knee")
        self.assertEqual(describe_depth(26 * INCH_M)[1], "not passable to any vehicle")

    def test_deep_water_is_danger(self):
        depths = np.linspace(0.70, 0.75, 30)
        _, result = process_node(readings_from_depths(depths), self.config)
        self.assertEqual(result["alert"], "DANGER")

    def test_rapid_rise_warns_before_warning_depth(self):
        depths = np.arange(30) * 0.02
        depths = depths[depths < 0.26]
        _, result = process_node(readings_from_depths(depths), self.config)
        self.assertLess(result["depth_m"], self.config.warning_depth_m)
        self.assertEqual(result["alert"], "WARNING")
        self.assertIsNotNone(result["minutes_to_warning"])

    def test_single_debris_spike_is_rejected(self):
        depths = np.full(30, 0.05)
        depths[15] = 1.20
        filtered, result = process_node(readings_from_depths(depths), self.config)
        self.assertTrue(filtered.loc[15, "flag_spike"])
        self.assertLess(filtered["depth_m"].max(), 0.15)
        self.assertEqual(result["alert"], "NORMAL")

    def test_stuck_wet_sensor_is_flagged(self):
        depths = np.full(30, 0.40)
        _, result = process_node(readings_from_depths(depths), self.config)
        self.assertEqual(result["status"], "CHECK_SENSOR")

    def test_old_reading_is_stale(self):
        frame = readings_from_depths(np.zeros(10))
        _, result = process_node(frame, self.config, now=frame["timestamp"].iloc[-1] + pd.Timedelta(hours=1))
        self.assertEqual(result["status"], "STALE")

    def test_missing_data_reports_unknown(self):
        frame = readings_from_depths(np.zeros(5))
        frame["distance_m"] = np.nan
        _, result = process_node(frame, self.config)
        self.assertEqual(result["alert"], "UNKNOWN")

    def test_temperature_correction_is_neutral_at_reference(self):
        self.assertAlmostEqual(float(temperature_corrected_distance(1.5, 20.0)), 1.5)
        self.assertGreater(float(temperature_corrected_distance(1.5, 35.0)), 1.5)

    def test_flood_without_rain_can_point_to_tide(self):
        causes = likely_causes(0.2, rain_last_3h_mm=0, tide_height_percentile=0.95, upstream_alert=False)
        self.assertEqual(causes, ["high tide"])

    def test_slow_drain_points_to_field_check(self):
        usual = readings_from_depths(np.linspace(0.5, 0.0, 12))
        slow = readings_from_depths(np.linspace(0.5, 0.25, 12))
        usual_filtered = kalman_depth(clean_readings(usual, self.config), self.config)
        slow_filtered = kalman_depth(clean_readings(slow, self.config), self.config)
        ratio = drain_slowdown_ratio(slow_filtered, usual_filtered)
        self.assertGreater(ratio, 1.5)
        causes = likely_causes(0.2, 0, 0.1, False, drain_slowdown_ratio=2.5)
        self.assertIn("possible blocked drain; send camera or field check", causes)


if __name__ == "__main__":
    unittest.main()

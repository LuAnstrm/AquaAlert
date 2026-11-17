"""Check the Bulacan data, features, labels, tide model and saved flood model."""

import unittest

import joblib
import numpy as np
import pandas as pd

import tide
from common import (DAILY_FILE, HAZARD_FILE, HOLDOUT_YEAR, MODEL_FILE, RAIN_POINTS, RIVER_POINTS,
                    build_features, build_labels, load_catalog)
from round2 import real_forecast, town_catalog
from train import wet_season_equivalents

ALLOWED_KINDS = {"event", "mask", "test_event", "tidal_event"}
ALLOWED_BASIS = {"documented", "report_date", "coverage_gap", "undated_report"}
PHYSICAL_MAX_RAIN_MM = 1500


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.daily = pd.read_csv(DAILY_FILE, parse_dates=["date"], index_col="date")
        cls.catalog = load_catalog()

    def test_dates_are_unique_and_continuous(self):
        self.assertTrue(self.daily.index.is_unique)
        gaps = self.daily.index.to_series().diff().dropna()
        self.assertTrue((gaps == pd.Timedelta(days=1)).all())

    def test_expected_columns_have_no_gaps(self):
        columns = ([f"rain_{p}" for p in RAIN_POINTS] + [f"discharge_{p}" for p in RIVER_POINTS]
                   + ["soil_moisture_calumpit"])
        self.assertEqual(int(self.daily[columns].isna().sum().sum()), 0)

    def test_values_are_physically_possible(self):
        rain = self.daily.filter(like="rain_")
        self.assertGreaterEqual(float(rain.min().min()), 0)
        self.assertLess(float(rain.max().max()), PHYSICAL_MAX_RAIN_MM)
        self.assertGreaterEqual(float(self.daily.filter(like="discharge_").min().min()), 0)
        soil = self.daily["soil_moisture_calumpit"]
        self.assertTrue(soil.between(0, 1).all())

    def test_catalog_is_valid(self):
        self.assertTrue(set(self.catalog["kind"]) <= ALLOWED_KINDS)
        self.assertTrue(set(self.catalog["date_basis"]) <= ALLOWED_BASIS)
        for kind in ["event", "test_event", "tidal_event"]:
            rows = self.catalog[self.catalog["kind"] == kind]
            self.assertTrue(rows["source_url"].notna().all(), kind)

    def test_events_do_not_overlap_each_other(self):
        events = self.catalog[self.catalog["kind"] == "event"].sort_values("start")
        self.assertTrue((events["start"].iloc[1:].to_numpy() > events["end"].iloc[:-1].to_numpy()).all())

    def test_hazard_table_is_complete(self):
        hazard = pd.read_csv(HAZARD_FILE)
        self.assertEqual(hazard["adm4_pcode"].nunique(), len(hazard))
        self.assertEqual(hazard["adm3_name"].nunique(), 24)
        shares = hazard.filter(like="share_")
        self.assertEqual(int(shares.isna().sum().sum()), 0)
        self.assertTrue(shares.stack().between(0, 1).all())
        for period in ["5yr", "25yr", "100yr"]:
            total = hazard[[f"share_{level}_{period}" for level in ["low", "medium", "high"]]].sum(axis=1)
            self.assertLessEqual(float(total.max()), 1.01, period)


class LeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.daily = pd.read_csv(DAILY_FILE, parse_dates=["date"], index_col="date")
        cls.catalog = load_catalog()

    def test_changing_the_future_does_not_change_past_features(self):
        cut = pd.Timestamp("2020-07-01")
        original = build_features(self.daily)
        changed = self.daily.copy()
        changed.loc[changed.index > cut] = changed.loc[changed.index > cut] * 5 + 100
        recomputed = build_features(changed)
        common = [c for c in original.columns if c != "tide_high_next2d"]
        pd.testing.assert_frame_equal(original.loc[:cut, common], recomputed.loc[:cut, common])

    def test_label_uses_only_the_next_two_days(self):
        index = pd.date_range("2020-07-01", "2020-07-20")
        catalog = pd.DataFrame({"kind": ["event"], "start": [pd.Timestamp("2020-07-10")],
                                "end": [pd.Timestamp("2020-07-10")]})
        labels = build_labels(index, catalog)
        expected = {pd.Timestamp("2020-07-08"), pd.Timestamp("2020-07-09")}
        self.assertEqual(set(labels.index[labels["target"]]), expected)

    def test_masked_days_are_excluded_unless_flood_is_known(self):
        index = pd.date_range("2020-07-01", "2020-07-20")
        catalog = pd.DataFrame({"kind": ["mask"], "start": [pd.Timestamp("2020-07-10")],
                                "end": [pd.Timestamp("2020-07-12")]})
        labels = build_labels(index, catalog)
        self.assertFalse(labels.loc["2020-07-09", "usable"])
        self.assertTrue(labels.loc["2020-07-13", "usable"])

    def test_saved_model_has_no_label_or_identity_features(self):
        features = joblib.load(MODEL_FILE)["features"]
        for word in ["target", "flood_day", "usable", "year", "date"]:
            self.assertFalse(any(word in name for name in features), word)


class EvaluationTests(unittest.TestCase):
    def test_masked_wet_seasons_do_not_dilute_yearly_rates(self):
        scored_wet = pd.date_range("2018-06-01", "2018-11-30")
        dry_only = pd.date_range("2019-01-01", "2019-05-31")
        seasons = wet_season_equivalents(scored_wet.append(dry_only))
        self.assertAlmostEqual(seasons, 1.0)


class ForecastTests(unittest.TestCase):
    def test_real_forecast_uses_only_day_one_and_two_runs(self):
        index = pd.date_range("2025-07-01", periods=6)
        data = {f"fc{lead}_{p}": np.full(6, 0.0) for lead in [1, 2] for p in RAIN_POINTS}
        frame = pd.DataFrame(data, index=index)
        frame.loc["2025-07-03", [f"fc1_{p}" for p in RAIN_POINTS]] = 10.0
        frame.loc["2025-07-04", [f"fc2_{p}" for p in RAIN_POINTS]] = 20.0
        result = real_forecast(frame)["fc_rain_local_next2d"]
        self.assertEqual(float(result.loc["2025-07-02"]), 30.0)
        self.assertEqual(float(result.loc["2025-07-01"]), 0.0)

    def test_town_catalog_masks_floods_not_naming_the_town(self):
        catalog = pd.DataFrame({"kind": ["event", "event", "tidal_event"],
                                "start": pd.to_datetime(["2020-01-05", "2020-02-05", "2020-03-05"]),
                                "end": pd.to_datetime(["2020-01-05", "2020-02-05", "2020-03-05"]),
                                "towns": ["Hagonoy", "Calumpit", "Hagonoy"]})
        result = town_catalog(catalog, "Hagonoy")
        self.assertEqual(result["kind"].tolist(), ["event", "mask", "event"])


class TideTests(unittest.TestCase):
    def test_fit_period_ends_before_holdout(self):
        self.assertLess(pd.Timestamp(tide.FIT_END).year, HOLDOUT_YEAR)

    def test_prediction_covers_future_dates(self):
        dates = pd.date_range("2027-07-01", "2027-07-10")
        self.assertEqual(int(tide.next_days_high(dates, 2).isna().sum()), 0)

    def test_rejects_missing_gauge_values(self):
        self.assertGreater(float(tide.load_observed().min()), 0)


class SavedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = joblib.load(MODEL_FILE)
        daily = pd.read_csv(DAILY_FILE, parse_dates=["date"], index_col="date")
        cls.features = build_features(daily).dropna()[cls.bundle["features"]]

    def test_outputs_are_valid_probabilities(self):
        score = self.bundle["model"].predict_proba(self.features)[:, 1]
        self.assertTrue(np.isfinite(score).all())
        self.assertTrue(((score >= 0) & (score <= 1)).all())

    def test_predictions_are_repeatable(self):
        sample = self.features.iloc[:200]
        first = self.bundle["model"].predict_proba(sample)
        np.testing.assert_array_equal(first, self.bundle["model"].predict_proba(sample))

    def test_more_rain_and_flow_do_not_lower_a_flood_day_score(self):
        row = self.features.loc[["2024-07-24"]].copy()
        base = self.bundle["model"].predict_proba(row)[0, 1]
        wetter = row.copy()
        for column in ["rain_3d_local_mean", "rain_7d_upstream_mean", "discharge_peak_7d_pampanga_main"]:
            wetter[column] = wetter[column] * 1.5
        self.assertGreaterEqual(self.bundle["model"].predict_proba(wetter)[0, 1], base)

    def test_dry_season_is_mostly_quiet(self):
        dry = self.features[self.features.index.month.isin([2, 3, 4])]
        alerts = self.bundle["model"].predict_proba(dry)[:, 1] >= self.bundle["threshold"]
        self.assertLess(float(alerts.mean()), 0.01)


if __name__ == "__main__":
    unittest.main()

"""Recheck the locked gates after correcting the yearly-rate denominator."""

import pandas as pd

from common import DAILY_FILE, HOLDOUT_YEAR, REPORT_DIR, build_features, build_labels, load_catalog
from improve import FEATURE_SETS, logistic
from train import evaluate, fit_with_threshold, leave_one_year_out, wet_season_equivalents

OLD_SEASONS = 15
LIVE_FILE = DAILY_FILE.with_name("bulacan_daily_forecast_archive.csv")


def main() -> None:
    catalog = load_catalog()
    daily = pd.read_csv(DAILY_FILE, parse_dates=["date"], index_col="date")
    features = build_features(daily).dropna()
    labels = build_labels(features.index, catalog)
    usable = labels["usable"]
    X_all, y_all = features[usable], labels.loc[usable, "target"].astype(int)
    dev = X_all.index.year < HOLDOUT_YEAR
    X, y = X_all[dev], y_all[dev]
    columns = FEATURE_SETS["compact"]

    seasons = wet_season_equivalents(X.index)
    print(f"Scored wet seasons: {seasons:.1f}, not {OLD_SEASONS}")

    previous = pd.read_csv(REPORT_DIR / "improvement_round.csv")
    scale = OLD_SEASONS / seasons
    previous["false_alert_episodes_per_season"] = (previous["false_alert_episodes_per_year"] * scale).round(1)
    previous["alert_days_per_season"] = (previous["alert_days_per_year"] * scale).round(1)
    previous["D3_corrected"] = previous["false_alert_episodes_per_season"] <= 3
    previous["D4_corrected"] = previous["alert_days_per_season"] <= 25
    candidates = previous[previous["gates_passed"].notna()].copy()
    candidates["gates_corrected"] = (candidates[["D1_detection", "D2_early", "D5_beats_rules"]].astype(bool).sum(axis=1)
                                     + candidates[["D3_corrected", "D4_corrected"]].sum(axis=1))
    candidates.to_csv(REPORT_DIR / "improvement_round_corrected.csv", index=False)
    print(candidates[["model", "pr_auc", "events_detected", "events", "false_alert_episodes_per_season",
                      "alert_days_per_season", "gates_corrected"]].to_string(index=False))

    factory = logistic(0.05)
    score, alerts = leave_one_year_out(factory, X[columns], y)
    row, _ = evaluate("lr_c005|compact", y, score.loc[y.index], alerts, catalog, labels, "dev_corrected")
    print({key: row[key] for key in ["false_alert_episodes_per_year", "alert_days_per_year"]})

    live = build_features(pd.read_csv(LIVE_FILE, parse_dates=["date"], index_col="date")).dropna()
    live_labels = build_labels(live.index, catalog)
    live_X = live[live_labels["usable"] & live.index.year.isin([2023, 2024, 2025, 2026])]
    scores, flags = [], []
    for year in sorted(set(live_X.index.year)):
        train = X_all.index.year != year
        model, threshold = fit_with_threshold(factory, X_all.loc[train, columns], y_all[train])
        test = live_X.loc[live_X.index.year == year, columns]
        result = pd.Series(model.predict_proba(test)[:, 1], index=test.index)
        scores.append(result)
        flags.append(result >= threshold)
    t1, _ = evaluate("lr_c005|compact", live_labels.loc[live_X.index, "target"].astype(int),
                     pd.concat(scores), pd.concat(flags), catalog, live_labels, "T1_corrected")
    print({key: t1[key] for key in ["events_detected", "events", "pr_auc",
                                    "false_alert_episodes_per_year", "alert_days_per_year"]})
    pd.DataFrame([row, t1]).to_csv(REPORT_DIR / "acceptance_corrected.csv", index=False)


if __name__ == "__main__":
    main()

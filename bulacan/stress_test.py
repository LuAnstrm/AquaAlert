"""Test whether the locked Bulacan model results hold up. This never saves a model."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import tide
from common import DAILY_FILE, HOLDOUT_YEAR, REPORT_DIR, build_features, build_labels, load_catalog
from improve import FEATURE_SETS, logistic
from train import SEED, leave_one_year_out, score_events

COLUMNS = FEATURE_SETS["compact"]
FACTORY = logistic(0.05)
SHUFFLES = 20
OUTPUT_FILE = REPORT_DIR / "stress_test_results.csv"


def prepare(catalog, features):
    labels = build_labels(features.index, catalog)
    usable = labels["usable"] & (features.index.year < HOLDOUT_YEAR)
    return features.loc[usable, COLUMNS], labels.loc[usable, "target"].astype(int), labels


def summarise(name, catalog, features) -> dict:
    X, y, _ = prepare(catalog, features)
    score, alerts = leave_one_year_out(FACTORY, X, y)
    score = score.loc[y.index]
    events = score_events(alerts, catalog)
    return {"test": name, "days": len(y), "target_days": int(y.sum()),
            "pr_auc": round(average_precision_score(y, score), 3),
            "roc_auc": round(roc_auc_score(y, score), 3),
            "events_detected": int(events["alert_by_end"].sum()), "events": len(events),
            "alert_days": int(alerts.sum())}


def shift_events(catalog, days) -> pd.DataFrame:
    shifted = catalog.copy()
    rows = shifted["kind"] == "event"
    shifted.loc[rows, "start"] += pd.Timedelta(days=days)
    shifted.loc[rows, "end"] += pd.Timedelta(days=days)
    return shifted


def per_year(catalog, features) -> pd.DataFrame:
    X, y, _ = prepare(catalog, features)
    score, alerts = leave_one_year_out(FACTORY, X, y)
    events = score_events(alerts, catalog)
    events["year"] = pd.to_datetime(events["start"]).dt.year
    table = pd.DataFrame({
        "days": y.groupby(y.index.year).size(),
        "target_days": y.groupby(y.index.year).sum(),
        "alert_days": alerts.groupby(alerts.index.year).sum().astype(int),
        "hits": (alerts & y.astype(bool)).groupby(y.index.year).sum().astype(int),
    })
    table["events"] = events.groupby("year").size().reindex(table.index).fillna(0).astype(int)
    table["detected"] = events.groupby("year")["alert_by_end"].sum().reindex(table.index).fillna(0).astype(int)
    return table


def shuffled_year_check(catalog, features) -> dict:
    """Assign each year another year's labels. Real signal should disappear."""
    X, y, _ = prepare(catalog, features)
    rng = np.random.default_rng(SEED)
    years = sorted(set(y.index.year))
    real_score, _ = leave_one_year_out(FACTORY, X, y)
    real = average_precision_score(y, real_score.loc[y.index])
    values = []
    for _ in range(SHUFFLES):
        mapping = dict(zip(years, rng.permutation(years)))
        labels = pd.Series(0, index=y.index, dtype=int)
        for source, target in mapping.items():
            take = y[y.index.year == source]
            day_of_year = take.index.day_of_year
            positives = day_of_year[take.to_numpy() == 1]
            mask = (y.index.year == target) & np.isin(y.index.day_of_year, positives)
            labels[mask] = 1
        if labels.sum() == 0:
            continue
        score, _ = leave_one_year_out(FACTORY, X, labels)
        values.append(average_precision_score(labels, score.loc[labels.index]))
    return {"real_pr_auc": round(real, 3), "shuffled_mean": round(float(np.mean(values)), 3),
            "shuffled_max": round(float(np.max(values)), 3), "runs": len(values),
            "share_at_or_above_real": round(float(np.mean(np.array(values) >= real)), 3)}


def rolling_tide_backtest() -> pd.DataFrame:
    """Fit tides on past years only, then predict each later year."""
    observed = tide.load_observed()
    rows = []
    for year in range(2018, 2027):
        train = observed.loc["2005-01-01":f"{year - 1}-12-31"]
        test = observed.loc[str(year)]
        coefficients, *_ = np.linalg.lstsq(tide.design_matrix(train.index), train.to_numpy(), rcond=None)
        predicted = pd.Series(tide.design_matrix(test.index) @ coefficients, index=test.index)
        residual = test - predicted
        peak_error = test.resample("D").max() - predicted.resample("D").max()
        rows.append({"year": year, "hours": len(test), "rmse_m": round(float(np.sqrt((residual ** 2).mean())), 3),
                     "bias_m": round(float(residual.mean()), 3),
                     "daily_peak_mae_m": round(float(peak_error.abs().mean()), 3),
                     "correlation": round(float(np.corrcoef(test, predicted)[0, 1]), 3)})
    return pd.DataFrame(rows)


def main() -> None:
    catalog = load_catalog()
    daily = pd.read_csv(DAILY_FILE, parse_dates=["date"], index_col="date")
    features = build_features(daily).dropna()

    rows = [summarise("locked finalist", catalog, features)]
    for days in [-2, -1, 1, 2]:
        rows.append(summarise(f"flood dates moved {days:+d} days", shift_events(catalog, days), features))

    # Drop each event in turn and see how far the score moves.
    events = catalog[catalog["kind"] == "event"]
    events = events[events["start"].dt.year < HOLDOUT_YEAR]
    leave_one = [summarise(f"without {row.name}", catalog.drop(index=index), features)
                 for index, row in zip(events.index, events.itertuples())]

    for row in rows:
        print(row)
    table = pd.DataFrame(rows + leave_one)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_FILE, index=False)
    drop_table = pd.DataFrame(leave_one)
    print(f"\nDrop one flood: PR-AUC {drop_table['pr_auc'].min()}-{drop_table['pr_auc'].max()}, "
          f"median {drop_table['pr_auc'].median()}")
    print(drop_table.nsmallest(2, "pr_auc")[["test", "pr_auc"]].to_string(index=False))
    print(drop_table.nlargest(2, "pr_auc")[["test", "pr_auc"]].to_string(index=False))

    print("\nBy year:")
    yearly = per_year(catalog, features)
    yearly.to_csv(REPORT_DIR / "stress_test_by_year.csv")
    print(yearly.to_string())

    print("\nShuffled years:")
    print(shuffled_year_check(catalog, features))

    print("\nRolling tide backtest:")
    backtest = rolling_tide_backtest()
    backtest.to_csv(REPORT_DIR / "tide_backtest.csv", index=False)
    print(backtest.to_string(index=False))


if __name__ == "__main__":
    main()

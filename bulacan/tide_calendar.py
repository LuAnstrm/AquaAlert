"""List the highest predicted Manila tides for scheduling Hagonoy field checks."""

import argparse
from datetime import date, timedelta

import pandas as pd

import tide
from common import REPORT_DIR

MANILA_OFFSET = pd.Timedelta(hours=8)
OUTPUT_FILE = REPORT_DIR / "hagonoy_tide_calendar_manila_gauge.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date.today())
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    end = args.start + timedelta(days=args.days)
    times_utc = pd.date_range(pd.Timestamp(args.start) - MANILA_OFFSET,
                              pd.Timestamp(end) - MANILA_OFFSET, freq="10min")
    predicted = tide.predict_hourly(times_utc)
    predicted.index = predicted.index + MANILA_OFFSET

    days = predicted.groupby(predicted.index.normalize())
    peak = pd.DataFrame({"predicted_high_m": days.max().round(2), "local_time": days.idxmax()})
    peak["local_time"] = peak["local_time"].dt.strftime("%H:%M")
    peak.index.name = "date"
    top = peak.nlargest(args.top, "predicted_high_m").sort_index()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    top.to_csv(OUTPUT_FILE)
    print(f"Wrote {OUTPUT_FILE}")
    print("Heights use the Manila gauge datum. Hagonoy height and timing must be measured.")
    print(top.to_string())
    print(peak.groupby(peak.index.month)["predicted_high_m"].max().to_string())


if __name__ == "__main__":
    main()

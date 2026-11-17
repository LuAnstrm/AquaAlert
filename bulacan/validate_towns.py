"""Compare NOAH mapped exposure with how often flood reports name each town."""

import re

import pandas as pd
from scipy.stats import spearmanr

from common import HAZARD_FILE, REPORT_DIR, load_catalog


def normalise(name: str) -> str:
    text = re.sub(r"^city of ", "", name.strip().lower())
    text = re.sub(r"\s*\(.*\)$", "", text)
    aliases = {"baliuag": "baliwag", "sta. maria": "santa maria", "san jose del monte": "sjdm",
               "bulakan": "bulacan"}
    return aliases.get(text, text)


def main() -> None:
    catalog = load_catalog()
    events = catalog[(catalog["kind"] == "event") & catalog["towns"].notna()]
    named = events["towns"].str.split(";").explode().map(normalise)
    counts = named.value_counts()

    hazard = pd.read_csv(HAZARD_FILE)
    hazard["town"] = hazard["adm3_name"].map(normalise)
    hazard["area_medium"] = hazard["area_m2"] * hazard["share_medium_25yr"]
    hazard["area_high"] = hazard["area_m2"] * hazard["share_high_25yr"]
    towns = hazard.groupby("town").agg(
        area_m2=("area_m2", "sum"), area_medium=("area_medium", "sum"),
        area_high=("area_high", "sum"), barangays=("town", "size"),
    )
    towns["share_medium_high_25yr"] = ((towns["area_medium"] + towns["area_high"])
                                       / towns["area_m2"]).round(3)
    towns["events_named"] = counts.reindex(towns.index).fillna(0).astype(int)

    unmatched = sorted(set(counts.index) - set(towns.index))
    if unmatched:
        raise ValueError(f"Catalog towns are missing from the boundaries: {unmatched}")

    rho, p_value = spearmanr(towns["share_medium_high_25yr"], towns["events_named"])
    output = towns[["barangays", "share_medium_high_25yr", "events_named"]].sort_values(
        "events_named", ascending=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(REPORT_DIR / "town_exposure_vs_reports.csv")
    print(output.to_string())
    print(f"\n{len(events)} events with named towns across {len(towns)} towns")
    print(f"Spearman correlation {rho:.2f}, p = {p_value:.3f}")


if __name__ == "__main__":
    main()

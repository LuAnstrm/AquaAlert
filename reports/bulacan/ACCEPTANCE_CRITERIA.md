# Bulacan Model Acceptance Criteria

Locked on 2026-11-05, before the improvement round. Do not change these numbers after seeing results. Any later change needs a new, dated section and a fresh test.

## Gates on development years (leave one year out, 2011-2025)

| Gate | Requirement |
|---|---|
| D1 Flood detection | At least 80% of reported floods have an alert from 3 days before the start through the end |
| D2 Early warning | At least 75% of floods with documented start dates have an alert 1-3 days before the start |
| D3 False alarms | At most 3 alert episodes per year with no reported flood nearby |
| D4 Alert burden | At most 25 alert days per year |
| D5 Better than simple rules | Year-block bootstrap 95% interval for PR-AUC gain over the best simple baseline stays above 0 |

## Gates on data not used for selection

| Gate | Requirement |
|---|---|
| T1 Live-like inputs, 2023-2026 | At least 75% flood detection and at most 4 false alert episodes per year |
| T2 Unseen 2009 floods | Both 2009 flood reports detected |

## Limits on the improvement round

- Try at most 15 model and feature combinations, and report that number.
- Choose the finalist using development years only.
- Test the finalist on T1 and T2 once. If it fails, report the failure and do not tune against those results.

## Limits of a pass

These gates can only clear a model for silent shadow testing beside PDRRMO operations. They cannot clear it for public warnings, because:

- The labels are about two dozen floods taken from news and agency reports.
- 2026 was already viewed once during the first training run.
- Barangay-level flooding has no labels.

Public alerts need validation against PAGASA gauge records, a full wet season in shadow mode, and sign-off from PDRRMO and PAGASA.

## Correction, 2026-11-05: yearly rates

Testing found a calculation bug. D3 and D4 divided by 15 years, including years whose wet seasons were masked. These gates now use scored wet seasons: scored June-November days divided by 183, or about 8.3 seasons. Thresholds did not change. The finalist now passes 0 of 5 development gates, and results are in `acceptance_corrected.csv` and `improvement_round_corrected.csv`.

## Addendum, 2026-11-05: tide test

The original gates are unchanged. Before running this test, one new comparison was declared: the finalist with and without `tide_high_next2d`, using the same model settings and development years. T1, T2 and 2026 were already viewed, so they cannot provide a fresh pass for this variant. The deployed model stays unchanged unless the variant passes all five development gates, and even then it needs new unseen data.

## Addendum, 2026-11-05: round 2 factor test

Declared before running round 2. The five development gates and their thresholds do not change.

- **Records:** round 2 uses the updated flood records. Round 1 results stay as reported, based on the earlier records.
- **Model:** logistic regression with C = 0.05, the round 1 finalist's settings.
- **Comparisons:** the base model, one test for each factor group, and all available factors together.
- **Factor groups:** rain forecast, tide, wind and pressure, tropical cyclones, El Nino, and rain intensity with deep soil moisture.
- **Rain forecasts:** real forecasts start around 2024. Tests over 2011-2025 use later observed rain as a best-case stand-in, so they are an upper bound and cannot pass a gate. Real forecast skill is measured only on 2024-2026.
- **Towns:** Hagonoy and Calumpit get their own targets. A day counts as positive only if a report names the town. Other flood days are excluded for that town because town lists can be incomplete.
- **Saved model:** `bulacan_flood_model.pkl` changes only if a model using inputs available at issue time passes all five development gates.
- **Limit:** 2026 and earlier tests have been viewed, so no result here is a fresh test. The first fresh test is the 2027 wet season.


# Bulacan Pilot: Data and Model Status

**Status: not ready for warnings.** The model failed its locked development gates. It can log results for internal testing, but no one should use it for decisions.

## Current result

Pass criteria were locked before the improvement round: [ACCEPTANCE_CRITERIA.md](../reports/bulacan/ACCEPTANCE_CRITERIA.md). The finalist was logistic regression on 15 rain, river, soil and season features, chosen from 12 combinations.

| Gate | Requirement | Result | Outcome |
|---|---|---|---|
| D1 Flood detection | At least 80% | 17 of 22 (77%) | Fail |
| D2 Early warning | At least 75% | 4 of 7 (57%) | Fail |
| D3 False alarms | At most 3 episodes per wet season | 4.7 | Fail |
| D4 Alert burden | At most 25 days per wet season | 36.2 | Fail |
| D5 Beats simple rules | PR-AUC gain interval above 0 | -0.126 to 0.107 | Fail |
| T1 Live-like inputs, 2023-2026 | At least 75% detected, at most 4 false episodes per wet season | 6 of 8, 2.4 per season, 40.6 alert days | Pass |
| T2 Unseen 2009 floods | Both detected | 2 of 2 | Pass (weak test) |

- **Correction, 5 November 2026:** D3 and D4 first passed because rates were averaged over 15 years, even though only about 8.3 wet seasons of data were scored. The thresholds did not change. The model now passes 0 of 5 development gates.
- **Finalist not changed:** if selection had used the corrected rates, a different model could have been chosen, but none passes more than 1 of 5 gates. Those candidates are in [improvement_round_corrected.csv](../reports/bulacan/improvement_round_corrected.csv).
- **Warning timing:** PR-AUC falls from 0.202 to 0.126 when flood dates are moved 2 days earlier, and rises to 0.267 when moved 2 days later. This suggests the model mostly reacts as rain and river flow arrive, and it has no rain forecast input yet.
- **Tests:** leakage and data checks pass. Results stay in a similar range when any one flood is removed (PR-AUC 0.176-0.214). Shuffled labels fall to 0.04, so the model has a real signal, though not enough for warnings.
- **T2 is weak:** the model raised alerts on 25 of the scored days in the wet-season window around Ondoy, so catching two reports there proves little.
- **No proven gain:** the model does not clearly beat a simple river-flow rule (PR-AUC 0.202 vs 0.221).
- **False alarms may be overstated:** a flood that no report covered counts as a false alarm.
- **Tuning has stopped:** further tuning would fit these same floods and make the scores less trustworthy.

## Round 2: all available factors

No model passed all five development gates, so the saved model did not change. Round 2 used updated flood records, so its numbers don't compare exactly with round 1.

**Development years, 2011-2025**

| Factors added to the base model | PR-AUC | Floods detected | False alarms per wet season | Gates passed |
|---|---|---|---|---|
| None | 0.207 | 19 of 23 | 4.6 | 1 of 5 |
| Tide | 0.187 | 18 of 23 | 4.7 | 0 of 5 |
| Wind and pressure | 0.456 | 19 of 23 | 3.1 | 2 of 5 |
| Cyclone distance | 0.208 | 19 of 23 | 4.8 | 1 of 5 |
| El Nino | 0.195 | 18 of 23 | 4.6 | 0 of 5 |
| Rain intensity and deep soil moisture | 0.239 | 18 of 23 | 5.1 | 0 of 5 |
| All factors known at issue time | 0.443 | 19 of 23 | 3.6 | 2 of 5 |
| Perfect 2-day rain forecast (best case only) | 0.235 | 17 of 23 | 3.0 | 2 of 5 |

**Real issued forecasts and live-style inputs, 2024-2026 (6 floods)**

| Model | PR-AUC | Floods detected | Warned before start | False alarms per wet season | Alert days per wet season |
|---|---|---|---|---|---|
| Base | 0.648 | 5 of 6 | 3 of 3 | 2.9 | 41.3 |
| Base + rain forecast | 0.766 | 5 of 6 | 2 of 3 | 2.0 | 29.8 |
| Base + wind and pressure | 0.535 | 5 of 6 | 3 of 3 | 4.9 | 53.4 |
| All factors | 0.599 | 5 of 6 | 3 of 3 | 2.9 | 37.1 |

- **Rain forecast:** this was the only group that clearly improved live-style results, but it rests on 6 floods and lost one early warning. On heavy-rain days, forecasts issued 1-2 days ahead averaged less than half of the later reanalysis rain at the lowland Bulacan points.
- **Wind and pressure:** the historical jump does not look like a data leak, but it did not carry over. Live wind data differs from ERA5, and news-based labels may favour large storms, so don't rely on this result yet.
- **Tide-only floods:** the Hagonoy model missed both scored tide-only reports. Daily town-level models are the wrong tool for these, so local sensors are needed.
- **Shadow signal:** `predict.py` shows the rain-forecast model for logging only. It has no authority over warnings.
- **Fresh test:** all results here have been viewed. The first fresh test is the 2027 wet season.

Factors not modelled because no usable history was found: dam releases, floodgates and pumps, garbage and silt, and a land-sinking grid. `collect_gauges.py` now logs dam levels, but scheduled collection still needs setting up and PDRRMO's approval.

Pilot choice: [PILOT_SITE_COMPARISON.md](PILOT_SITE_COMPARISON.md).

## Data in use

| Data | Source | Used for | License or access |
|---|---|---|---|
| Flood events | [flood_events.csv](../data/external/bulacan/flood_events.csv), from news and PRFFWC reports | Labels | Links to each source; dates are often report dates |
| Rainfall and soil moisture | [Open-Meteo](https://open-meteo.com/) ERA5 and forecast archive | Model inputs | Free for non-commercial use; check terms before deployment |
| River discharge | [GloFAS through Open-Meteo](https://open-meteo.com/en/docs/flood-api) | Model inputs | Modelled, not measured; 5 km cells miss small rivers |
| Flood hazard | [Project NOAH mirror](https://huggingface.co/datasets/bettergovph/project-noah-hazard-maps) | Barangay exposure | ODC-ODbL with attribution and share-alike; compare with official NOAH before use |
| Barangay boundaries | [HDX COD-AB](https://data.humdata.org/dataset/cod-ab-phl) | Barangay exposure | CC BY-IGO |
| Rain forecasts | [Open-Meteo previous runs](https://open-meteo.com/en/docs/previous-runs-api), from January 2024 | Round 2 tests and shadow signal | Check non-commercial terms |
| Wind, pressure, rain intensity, deep soil | Open-Meteo ERA5 and forecast archive | Round 2 tests | Check non-commercial terms |
| Cyclone tracks | [NOAA IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive) | Round 2 tests | Recent tracks are provisional |
| El Nino index | [NOAA ONI](https://psl.noaa.gov/data/correlation/oni.data) | Round 2 tests | Public data |
| Elevation | [Copernicus DEM GLO-30](https://registry.opendata.aws/copernicus-dem/) | Site comparison | Surface includes buildings and trees; check license before redistribution |
| Land cover | [ESA WorldCover 2021](https://worldcover2021.esa.int/download) | Site comparison | CC BY 4.0 |
| Waterways | [OpenStreetMap](https://www.openstreetmap.org/) | Site comparison | ODbL; coverage varies |
| Live gauges | [Bulacan PDRRMO](https://pdrrmo.bulacan.gov.ph/) | Official status and new history | Public page; ask PDRRMO before scheduled collection |

## Barangay and factor checks

- **Barangay exposure:** [bulacan_barangay_hazard.csv](../data/processed/bulacan_barangay_hazard.csv) covers 572 barangays in 24 towns for 5-, 25- and 100-year NOAH scenarios. It measures mapped hazard and doesn't predict when a barangay floods.
- **Check against reports:** towns with more mapped hazard were named more often in flood reports (Spearman 0.51, p = 0.012, 17 events). Some towns differ, such as Pulilan with high mapped exposure and one report, and news coverage may also affect the count.
- **Tide:** a tide model fitted through 2025 tracked 2026 observations with 0.98 correlation and 0.14 m error. Rolling tests show daily peak errors of about 5 cm in 2018-2022 and 10-13 cm in 2023-2026, when predictions ran too high. Use measured Hagonoy levels to correct this. Adding predicted tide did not make the province-wide model pass, so the saved model does not use it.
- **Tide-only floods:** these are logged as `tidal_event` rows in the flood records. The province-wide model does not use them, so its results stay reproducible, but it still treats those days as ordinary days.
- **Next stage:** street sensors, garbage, and other non-rain factors are covered in [AGOS_STREET_LEVEL_ML.md](AGOS_STREET_LEVEL_ML.md).

## Data needed to pass

In priority order:

1. **River levels and rainfall:** hourly or 10-minute history from PAGASA and PRFFWC for Sulipan, Arayat, Calumpit and San Rafael. Request it through the [PAGASA data portal](https://ihdrms.app/) or at hmdas.hmd@gmail.com.
2. **Flood records:** dated PDRRMO situation reports by barangay, with depths, for 2011-2026. These would replace news-based labels and allow barangay-level validation.
3. **Dam releases:** hourly Angat, Ipo and Bustos release records. Reports repeatedly link these to Bulacan flooding.
4. **Tide observations:** Manila Bay or Navotas records for Hagonoy, Paombong and Calumpit. Tidal flooding is a likely reason for missed floods.

## Before public use

1. Retrain and re-test against official gauge alert levels using the new data.
2. Run silently for a full wet season beside PDRRMO operations.
3. Have PDRRMO and PAGASA review the thresholds and message wording.
4. Agree who issues alerts. The system should support the official warning chain, not compete with it.

## How to run

Run these from `bulacan/` with the project's virtual environment:

```bash
python fetch_history.py
python build_hazard.py
python validate_towns.py
python train.py
python improve.py
python evaluate_factors.py
python fetch_factors.py
python round2.py
python collect_gauges.py
python predict.py
```

Before running, download these files into `data/external/`:

- `noah/Bulacan_5yr.zip`, `Bulacan_25yr.zip` and `Bulacan_100yr.zip` from the NOAH mirror
- `boundaries/phl_admin_boundaries.gdb.zip` from HDX
- `tide/h370.csv` from [UHSLC](https://uhslc.soest.hawaii.edu/data/csv/fast/hourly/h370.csv)

`train.py` writes comparison reports, while `improve.py` selects and saves the model.

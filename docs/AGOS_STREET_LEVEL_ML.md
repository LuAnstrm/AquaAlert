# AGOS Street-Level ML Plan

AGOS sensors fill the biggest data gap in this project: measured flood depth at street level with exact times. The current model has about two dozen floods from news and agency reports, and no measured street depth.

## What sensors can and cannot do

| Warning horizon | Main source | Expected use |
|---|---|---|
| Now to 1 hour | Street sensors, rate of rise, nearby nodes | Best first use, because it follows water already arriving |
| 1 to 6 hours | Sensors, rain, tide, upstream gauges | Possible after enough local data, especially for slow river and tide floods |
| 6 to 24 hours | Forecast rain, predicted tide, upstream river and dam data | Mostly depends on forecasts and upstream data, not street sensors alone |

The proposal's "6-24 hours additional warning" should stay a target until a pilot measures it. Flash floods may give only minutes of warning. Tide peaks can be predicted days ahead, but flood depth at a street still has to be validated.

## Flood factors and available data

| Factor | Why it matters | Data found | Status |
|---|---|---|---|
| Tide | Can flood low areas and block drainage | [UHSLC Manila gauge](https://uhslc.soest.hawaii.edu/datainfo/), hourly 1984-September 2026 | Added and predictable; correlation 0.98 and error 0.14 m on 2026 |
| Upstream rain | Water can arrive after local rain ends | Open-Meteo at upstream points | Added |
| River backflow | Pampanga and Angat water can back up into towns | GloFAS modelled flow; PDRRMO live gauges | Added; measured history still needed |
| Dam releases | Reports link Ipo and Bustos releases to floods | PDRRMO live dam levels; release history not found | Request from PAGASA, NIA or NPC |
| Place and terrain | Low land floods more | NOAH maps and barangay boundaries; elevation API available | Barangay exposure added |
| Land subsidence | Makes flooding and tides worse over time | Published estimates conflict; no open Bulacan grid found | Treat as slow change; check bench marks and InSAR sources |
| Garbage and silt | Blocks drains and channels | No Bulacan drain or waste time series found | Collect through AGOS |
| Fishponds, dikes and land use | Change where water can go | [ESA WorldCover](https://worldcover2021.esa.int/download), OpenStreetMap | Not added; verify local dikes and floodgates in the field |
| Pumps and floodgates | Their operation can change depth quickly | None found | Ask LGUs for logs |
| Flood-control structures | Unfinished or faulty works may trap water or fail to block it | News and government statements; no as-built or status dataset found | Ask DPWH and the LGU; measure both sides of key structures |

## Hagonoy: tide-first pilot case

Hagonoy is a useful first site because tide floods can be predicted and measured separately from storm floods.

- **Tide without storms:** a [GMA report](https://www.gmanetwork.com/news/topstories/regions/964110/hagonoy-public-cemetery-remains-flooded-despite-town-s-43-flood-projects/story) says high tide floods the public cemetery and nearby roads even without a typhoon. On [3 July 2023](https://mb.com.ph/2023/7/4/high-tide-causes-deep-flood-in-12-hagonoy-bulacan-vilages), high tide flooded 12 villages. The Manila gauge that day was in its top 5% of days since 2011, with only 2.5 mm of modelled local rain.
- **Not every flood is tidal:** Hagonoy's disaster office [said](https://www.dzrh.com.ph/post/hagonoy-bulacan-submerged-due-to-back-to-back-storms) the August 2026 flooding was not caused by high tide alone. Storms and upstream water also played a part.
- **Flood-control works:** [residents blame](https://tribune.net.ph/2026/08/23/dpwh-checks-ghost-flood-control-projects-in-paombong-and-hagonoy) an unfinished Barangay Iba floodgate for trapping water, and Hagonoy's municipal engineer [says it blocks the waterway](https://www.gmanetwork.com/news/balitambayan/trending/998860/dating-magandang-subdivision-sa-hagonoy-bulacan-iniwan-ng-ilang-dating-nakatira-dahil-nalubog-na/story/). The Independent Commission for Infrastructure has also [recommended charges](https://www.pna.gov.ph/articles/1263226) over a P74.1-million Hagonoy project that a government audit found was never implemented. These reports do not measure how much each structure changes flooding.

Suggested Hagonoy measurements:

1. **Local tide:** install a node on open water, because Manila tide height and timing can differ from Hagonoy.
2. **Street depth:** install nodes in low streets in barangays that flood often. Confirm the sites with the MDRRMO.
3. **Blocked flow:** install a pair on opposite sides of the Iba floodgate, if the LGU allows it. Water-level differences can test the trapping claim.

Separate "tide only" flooding from "storm plus tide" in the labels. The first Hagonoy model should estimate street depth from predicted tide, local rain and upstream flow. The province-wide model is not built for this, and its training data still treats tide-only flood days as ordinary days.

## How AGOS can measure garbage and blockages

No dataset can predict blockage on a specific street today, so AGOS should collect its own evidence:

1. **Compare drainage speed:** record how quickly each node normally drains. If water falls at half its usual speed, flag a possible blockage. `street_node.py` includes this check, but the 2x rule still needs field testing.
2. **Place sensors in pairs:** install one before and one after a drain or culvert. A growing water-level difference can point to a blockage.
3. **Use camera checks:** at first, ask people to label images as clear, partial blockage or heavy blockage. Train on those labels later.
4. **Add reports and logs:** link citizen reports and LGU cleanup dates to each node. Cleanup dates can show whether drainage improved.

Two public datasets may help with camera work, but their contents still need checking:

- [Urban Flood Water Depth Estimation](https://www.hydroshare.org/resource/9017a306f36945b18e64b5fffa41919e/) pairs camera images with pressure-sensor depth (CC BY 4.0).
- [Riverine Plastic Litter Dataset](https://data.mendeley.com/datasets/7yvrhkxz22/1) covers floating litter (CC BY 4.0). It isn't from Bulacan, and the page doesn't confirm its annotations.

## What the current records show

These results use 23 flood reports and modelled weather, so treat them as clues:

| Condition in the 4 days up to a flood report | Flood reports | Ordinary wet-season periods |
|---|---|---|
| High upstream rain | 61% | 15% |
| High Pampanga River flow | 52% | 17% |
| High observed tide | 36% | 19% |

- **Upstream water matters:** reports often follow high upstream rain or river flow.
- **Tide may matter:** high tides appear more often before floods, but the sample is small. Adding predicted tide to the province-wide daily model didn't produce a clear improvement: early warnings changed from 4 to 5 of 7, while PR-AUC fell from 0.202 to 0.187.
- **Rain-free floods are not proven:** 5 of 23 reports had under 50 mm of modelled local rain in 3 days, but the model can miss local storms. A report from 3 August 2018 says local gauges recorded nearly 200 mm, while the modelled 3-day figure was about 37 mm.
- **Why local data matters:** a tide effect may still appear at street level in places like Hagonoy, which a province-wide daily model can't see.

## Starter pipeline

[street_node.py](../bulacan/street_node.py) is ready for testing with real sensor files. Its [tests](../bulacan/test_street_node.py) use made-up readings only, and those readings must not be used for training.

Each node needs these fields:

| Field | Meaning |
|---|---|
| `timestamp` | Reading time, preferably in UTC |
| `distance_m` | Ultrasonic distance from sensor to surface |
| `air_temp_c` | Optional air temperature for sound-speed correction |

The pipeline:

1. Converts distance to depth using the measured mounting height.
2. Rejects impossible readings and debris spikes.
3. Flags stuck or stale sensors.
4. Estimates depth and rise rate with a Kalman filter.
5. Maps depth to the [MMDA flood gauge](https://interaksyon.philstar.com/trends-spotlights/2024/09/04/282826/mmda-flood-gauge-system-travelers-motorists/) terms, such as knee or waist deep.
6. Raises rule-based alerts, including projected time to warning depth.
7. Suggests likely causes for an operator to check.

All thresholds are starting values for DRRMO review and field testing. Bulacan's local flood bands may differ from the MMDA's Metro Manila gauge.

## Model roadmap

| Phase | Needed | Model | Pass check |
|---|---|---|---|
| 1. Measure | Installed nodes, surveyed mounting heights, staff gauge in camera view | Filter and rules only | Sensor depth matches staff gauge readings |
| 2. Silent wet season | At least one wet season from every pilot node, plus DRRMO flood logs | No public AI alert; compare rules and simple models | Measured lead time and false-alarm rate by node |
| 3. Short-range prediction | Several flood events per node | Gradient boosting for 30-minute to 6-hour depth | Beats persistence and rate-of-rise on later events |
| 4. Sequence models | Many nodes and seasons | LSTM, only if it beats simpler models | Beats phase 3 on unseen events and locations |
| 5. Public support | DRRMO and PAGASA review | Advisory support with human authority | Agreed thresholds, message wording and fallback plan |

An LSTM should wait for this data. With only a few flood events per sensor, it would likely memorize them, and simpler models are easier for DRRMO staff to check.

## Pilot installation

- **Cover different flood causes:** use a tidal site such as Hagonoy or Paombong, a river-backflow site such as Calumpit, and a drainage-prone site such as Meycauayan or Marilao.
- **Prevent bad readings:** mount over a clear surface away from walls, vehicles and hanging objects. Debris and waves can cause false echoes.
- **Keep ground truth in view:** place a painted staff gauge in camera view. Each photo can then check sensor depth and become a labelled image.
- **Record site details:** save GPS position, road elevation, mounting height, nearby drain, tide gate and pump.
- **Protect privacy:** aim cameras at the road surface and gauge, and avoid faces, plates and home interiors. Although sensor readings are environmental, camera images can capture personal data.

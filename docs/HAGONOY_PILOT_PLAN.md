# AGOS Hagonoy Pilot Plan

Draft, 5 November 2026. Sites, thresholds and targets need review by the Hagonoy MDRRMO before use.

**Goal:** run a silent wet-season test in 2027 that measures whether AGOS can predict tide flooding at street level. The system stays internal until the MDRRMO and PDRRMO review the results.

## Why Hagonoy

- **Tide flooding happens without heavy rain.** The provincial disaster office [says](https://www.gmanetwork.com/news/topstories/regions/997873/habagat-high-tide-cause-water-levels-to-rise-in-bulacan-coastal-areas/story/) coastal water runs high "with or without ulan." [High tide flooded 12 villages](https://mb.com.ph/2023/07/04/high-tide-causes-deep-flood-in-12-hagonoy-bulacan-vilages) on 3 July 2023.
- **Tide peaks can be predicted days ahead.** Our Manila tide model tracked 2026 well, with daily peaks off by about 0.13 m.
- **The flood causes overlap.** [August 2026 flooding](https://www.dzrh.com.ph/post/hagonoy-bulacan-submerged-due-to-back-to-back-storms) involved storms, upstream water and tide, so the pilot must measure each one.

## Questions to answer

1. How much lower or later is each street's flooding than the Manila tide prediction suggests?
2. At what local tide level does each street start to flood?
3. How far ahead can we predict peak depth on tide-only days?
4. How much do rain and upstream water add on top of the tide?
5. Does water level differ across the unfinished Barangay Iba floodgate?

## Proposed sites

Use 8 nodes, plus 2 spares. These are starting points, and the MDRRMO should pick exact streets.

| Node | Barangay | Purpose | Why this barangay |
|---|---|---|---|
| 1 | Pugad or San Miguel | Local tide reference over open water | Both are [reported tide-flood areas](https://mb.com.ph/2023/07/04/high-tide-causes-deep-flood-in-12-hagonoy-bulacan-vilages); choose the site with safe, clear water access |
| 2 | San Miguel | Coastal street depth | [Reported](https://www.gmanetwork.com/news/topstories/regions/997873/habagat-high-tide-cause-water-levels-to-rise-in-bulacan-coastal-areas/story/) at 2-3 ft during high tide |
| 3 | Sagrada Familia or Tibaguin | Second coastal street | Named in the 2023 tide flood |
| 4 | Santa Elena or Santo Niño (Pob.) | Town-centre street | Confirm with the MDRRMO which floods more and serves more people |
| 5 | San Sebastian | Long-term standing water | The Mary the Queen subdivision is [reported flooded for seven years](https://www.dzrh.com.ph/post/hagonoy-bulacan-submerged-due-to-back-to-back-storms) |
| 6-7 | Iba | Pair on opposite sides of the floodgate | [Residents blame it for trapping water](https://tribune.net.ph/2026/08/23/dpwh-checks-ghost-flood-control-projects-in-paombong-and-hagonoy); needs LGU and DPWH permission |
| 8 | Tampok or Santa Monica | Inland storm and backflow comparison | Highest NOAH mapped rain-flood hazard in Hagonoy, about 70% high-hazard area |

**Don't pick tide sites from NOAH alone.** NOAH's flood maps here are built for rainfall scenarios. Tampok ranks highest on those maps, while reported tide-flood areas such as Tibaguin rank much lower.

**Name to confirm:** a [news report](https://www.gmanetwork.com/news/topstories/regions/997873/habagat-high-tide-cause-water-levels-to-rise-in-bulacan-coastal-areas/story/) names "Santa Ana," but it isn't in our list of 26 Hagonoy barangays.

## What each node records

| Data | How | Frequency |
|---|---|---|
| Water surface distance | Ultrasonic sensor; [street_node.py](../bulacan/street_node.py) converts it to depth | Every 5 minutes, then every minute when rising fast |
| Air temperature | Sensor on the node | Every 5 minutes |
| Rain | One gauge at the town centre and one on the coast | Every 5 minutes |
| Ground truth | Painted staff gauge in camera view | Hourly photo, then every 15 minutes when wet |
| Site details | GPS, surveyed road height, mounting height, drains, gates and pumps | Install, then yearly |

**Install notes:**

- **Mounting height:** keep electronics above the worst expected flood. Reports for Hagonoy reach about 5 ft, and Bulacan has reached 8 ft. Then check that the sensor can read the full range.
- **Road height:** resurvey yearly. Land sinking and sea-level change can shift flood levels, so old thresholds may stop working.
- **Harsh conditions:** test corrosion, wave noise and floating debris in saltwater before buying all the hardware.
- **Privacy:** aim cameras at the staff gauge and road surface. Agree with the LGU who can view images and how long they are kept.

## Labels

Sensors show how deep the water got, but people must record why. Ask a barangay contact to fill in a simple log for each flood, using these labels:

| Label | Meaning |
|---|---|
| `tide_only` | High tide with no meaningful local rain or upstream flood |
| `storm_plus_tide` | Rain and tide together |
| `upstream_backflow` | River or dam water arriving, with or without local rain |
| `blocked_or_trapped` | Water stays after the tide falls; add a photo or field note |
| `unknown` | Cause isn't clear |

Record start time, peak time, end time, estimated depth, affected streets and notes. Keep cleanup, pump and floodgate work in the same log, because they can change how a street floods.

## Models, simplest first

Each model must beat the one before it on later, unseen data:

1. **Persistence:** assume the current depth continues.
2. **Tide threshold:** fit one flood start level and depth rule per node, using predicted tide with the measured local offset.
3. **Tide plus context:** add local rain, upstream flow and recent water level with gradient boosting.
4. **LSTM:** try this only after several flood events per node, and keep it only if it wins.

Predict at 1, 6 and 24 hours. The Hagonoy tide calendar and depth rules can stay useful even when the AI layer fails.

## Proposed targets

Lock these with the MDRRMO before the wet season, and report the results even if they fail:

| Measure | Target |
|---|---|
| Sensor depth vs staff gauge | Typical error within 3 cm when water is present |
| Node data availability | At least 95% during flood periods |
| Tide-only floods, 24 hours ahead | Catch at least 90% of days reaching gutter depth (8 in) |
| Peak depth, 24 hours ahead | Typical error within 10 cm |
| False alerts | At most 1 in 10 predicted flood days |
| Better than simple rules | Beats the persistence and tide-threshold models |

Fit models on earlier months, then test on later months and storm events kept aside. One wet season may not include enough storms to judge the storm-plus-tide model.

## Timeline

Dates use our tide calendar, based on the Manila gauge. Hagonoy's timing may differ.

| When | Work |
|---|---|
| November 2026 | Meet the Hagonoy MDRRMO and Bulacan PDRRMO; agree on roles, sites, data sharing and targets |
| 26-29 November 2026 | First high-tide walk with no hardware: photograph marked points and log depth by hand |
| 24-25 December 2026 | Repeat at more sites, and check the local tide timing |
| December 2026-February 2027 | Build and bench-test 2 nodes; install the tide reference and one street node with permission |
| March-April 2027 | Compare sensors with staff gauges, fix hardware, and install the remaining nodes |
| 10 May 2027 | Run the full network through the first big predicted tide |
| June-August 2027 | Main tide test; the highest predicted days are around 5-8 June, 3-7 July and 1-4 August |
| June-November 2027 | Silent storm season with no public AI alerts |
| December 2027 | Publish results and decide with the LGU whether advisories can move to testing |

The proposal's Q3-Q4 2026 pilot date has passed without built hardware, so this plan moves it to 2027. The hand-measured tide walks need no funding and can start now.

## Budget

| Item | Estimate |
|---|---|
| 10 nodes at the proposal's PHP 5,875 each | PHP 58,750 |
| Staff gauges, cameras, rain gauges, LoRaWAN gateway, mounting, road survey, data plans, maintenance | Not yet costed; get quotes |

## Who does what

| Role | Responsibility |
|---|---|
| AGOS team | Hardware, data, models, reports and maintenance |
| Hagonoy MDRRMO | Site choices, flood-log review and threshold review; keeps authority over warnings |
| Barangay contacts | Flood log, and local checks during high tides |
| Bulacan PDRRMO | Provincial gauges and reports, and reviews any move to advisories |
| PAGASA | Gauge history and official flood guidance |
| DPWH | Permission and information for flood-control structures |

## Data requests to send now

- **Hagonoy MDRRMO and Bulacan PDRRMO:** tide advisories, flood reports by barangay, and evacuation records
- **PAGASA:** river, rain and tide records through the [data portal](https://ihdrms.app/)
- **DPWH:** status and design details for the Iba floodgate and Hagonoy projects
- **NAMRIA or PAGASA:** official tide predictions and vertical reference points

## Main risks

| Risk | Response |
|---|---|
| Funding delays hardware | Start hand-measured tide walks now; begin with 2 nodes |
| Salt, debris or vandalism breaks sensors | Keep spares, flag bad readings, and agree on a local inspection routine |
| Network fails during storms | Store readings on the node and test all fallback links |
| Few storms during the pilot | Report tide results separately, and extend the storm test to 2028 |
| A flood-control project changes the site | Log the change date and re-fit that node |
| People treat test results as warnings | Keep AI outputs internal and label them clearly; the MDRRMO issues warnings |

## How to refresh the tide calendar

Run this from `bulacan/`:

```bash
python tide_calendar.py --start 2026-11-05 --days 365 --top 30
```

The output is [hagonoy_tide_calendar_manila_gauge.csv](../reports/bulacan/hagonoy_tide_calendar_manila_gauge.csv). Its heights use the Manila gauge's reference level, so they can't be compared directly with PDRRMO tide schedules or street depth.

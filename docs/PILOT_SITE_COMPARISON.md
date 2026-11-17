# Hagonoy vs Calumpit Pilot Comparison

Draft, 5 November 2026. **Recommendation: start with Hagonoy, unless the Calumpit MDRRMO is much readier to partner.** Current models can't decide this alone, so the choice depends mainly on how quickly AGOS can collect useful data.

## Side-by-side evidence

Report counts come from our news-based records. Town areas and shares are area-weighted, using boundary, NOAH, Copernicus DEM, ESA WorldCover and OpenStreetMap data.

| Measure | Hagonoy | Calumpit |
|---|---|---|
| Barangays | 26 | 29 |
| Flood reports in records | 15 | 15 |
| Tide-flood reports in records | 4 | 0 |
| Reports with high river flow nearby | 53% | 73% |
| Reports with high tide nearby | 36% | 27% |
| NOAH 25-year medium or high hazard | 65% | 63% |
| Land below 2 m | 86% | 52% |
| Permanent-water land cover | 85% | 16% |
| Mapped waterways | 2.36 km/km² | 1.13 km/km² |
| PDRRMO river gauges | None found in Hagonoy | Caniogan River and Meyto Bridge |

Ordinary wet-season periods show high river flow or high tide about 17-19% of the time. The elevation data includes buildings and trees, while land-cover water may include fishponds, so confirm these figures in the field.

## What each site offers

| Question | Hagonoy | Calumpit |
|---|---|---|
| Main flood cause | Tide plus storms and upstream water | River backflow, dam releases and rain, plus tide |
| How much data one season gives | Likely more, because [reports](https://newscorebulacan.net/hagonoy-schools-float-in-high-tide-waters/) describe frequent tide flooding | Likely fewer, larger floods |
| Warning lead | Tide peaks are known days ahead | Water from upstream may give time, which needs measuring |
| Existing ground truth | Almost none, so AGOS adds more new data | PDRRMO gauges can help check sensors |
| Flood-control questions | [Unfinished Iba floodgate](https://tribune.net.ph/2026/08/23/dpwh-checks-ghost-flood-control-projects-in-paombong-and-hagonoy) | [Unfinished Bulusan dike](https://www.sunstar.com.ph/pampanga/new-dpwh-bulacan-execsordered-to-complete-projects) and [Calizon dike replacement](https://www.gmanetwork.com/news/topstories/regions/987372/dpwh-begins-replacing-brittle-dike-in-calumpit-bulacan/story/) |
| Site changes during the pilot | DPWH has reported plans for a Hagonoy dike from 2027; confirm the location and schedule | Calizon work is targeted for March 2027, so nearby measurements may shift |

## Why Hagonoy first

- **Faster learning:** frequent tide floods could give many events in one 2027 season. A river-flood site may have only a few.
- **Biggest data gap:** every current model missed the scored Hagonoy tide-only floods, and no PDRRMO gauge was found there.
- **Visible value:** predicted tide plus street depth can be checked repeatedly with the MDRRMO.

**Choose Calumpit instead if** its MDRRMO is much readier, river floods are your main goal, or using existing gauges as a check matters most.

## If you choose Calumpit

Use 8 nodes plus 2 spares, and let the MDRRMO choose exact streets:

| Node | Barangay | Purpose |
|---|---|---|
| 1 | Beside Caniogan River gauge | Check AGOS against an existing gauge |
| 2 | Beside Meyto Bridge gauge | Second check and river timing |
| 3 | Meysulao | Reported [deepest flooding](https://www.pna.gov.ph/articles/1254943) |
| 4 | Gugo | Reported [deep flooding in 2026](https://newsinfo.inquirer.net/2304318/bulacan-logs-5-leptospirosis-deaths-as-cases-rise-to-nearly-400) |
| 5 | Sapang Bayan | [Backflood reports](https://www.gmanetwork.com/news/topstories/regions/586158/floods-in-calumpit-bulacan-still-waist-deep-after-typhoon-lawin/story) |
| 6-7 | Bulusan | Pair across the dike, if permitted |
| 8 | Poblacion | Town-centre roads and services |

Put Calizon on the watch list, and log the dates of any construction there. The same labels, targets and model steps from the [Hagonoy plan](HAGONOY_PILOT_PLAN.md) apply, using `upstream_backflow` more often.

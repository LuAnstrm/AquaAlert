# AquaAlert: Bulacan Flood ML for AGOS

Flood data and machine-learning work for [AGOS](docs/AGOS_STREET_LEVEL_ML.md), with Bulacan as the first pilot province.

> **Not an official warning system.** The current model failed its acceptance gates and is for internal testing only. Follow PAGASA, Bulacan PDRRMO and your local DRRMO. In an emergency, call 911.

## Status

| Part | Status |
|---|---|
| Daily rain, soil moisture and modelled river flow | Ready, 2011 to October 2026 |
| Manila Bay tide observations and predictions | Ready; a Hagonoy gauge is still needed |
| Barangay flood exposure | Ready for 572 barangays; shows mapped hazard only |
| Province-wide daily flood model | Trained, but failed: [results](docs/BULACAN_PILOT.md) |
| Street sensor pipeline | Starter code; tested only on made-up readings |
| Rain forecasts and extra factors | Downloaded and tested; shown only as an internal shadow signal |
| Calumpit | Added to town models, records and site comparison |
| Tide-only flooding in Hagonoy | Early records only; no model yet |

## Start here

- [Hagonoy vs Calumpit](docs/PILOT_SITE_COMPARISON.md): evidence for choosing the first pilot site
- [Hagonoy pilot plan](docs/HAGONOY_PILOT_PLAN.md): sites, data collection, timeline and targets
- [Bulacan pilot status](docs/BULACAN_PILOT.md): data, results and the data still needed
- [AGOS street-level plan](docs/AGOS_STREET_LEVEL_ML.md): sensors, tides, garbage and the model roadmap
- [Acceptance criteria](reports/bulacan/ACCEPTANCE_CRITERIA.md): the locked pass mark
- [Flood records](data/external/bulacan/flood_events.csv): dated reports with source links

## Setup

Install Python 3.10 or newer, then run this from the repository root:

```bash
python -m venv .venv
```

Activate the environment, then install the packages:

```bash
pip install -r requirements.txt
```

Download the source files listed in the [pilot status](docs/BULACAN_PILOT.md), then run the pipeline from `bulacan/`:

```bash
python fetch_history.py
python build_hazard.py
python validate_towns.py
python train.py
python improve.py
python evaluate_factors.py
python fetch_factors.py
python round2.py
python predict.py
```

Run the sensor and data tests from `bulacan/`:

```bash
python -m unittest test_street_node test_ml
```

Run the slower model checks from `bulacan/`. They never save a model:

```bash
python stress_test.py
python recheck_gates.py
```

## Repository layout

```text
bulacan/   Data, model, tide and street-sensor code
data/      Flood records and processed data; large downloads stay out of git
docs/      Pilot status and AGOS plan
models/    Saved Bulacan model
reports/   Bulacan evaluation results
archive/   Original Metro Manila capstone work
```

The original capstone can still run from [archive/ncr_capstone](archive/ncr_capstone/README.md).

## Team

- Francisco, Eliesha Mae L.
- Gemina, Jamaica
- Labaco, Charina Mae
- Viray, Rodelyn U.
- Viri, Aira Kristel

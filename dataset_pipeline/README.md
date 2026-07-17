# India Regional Crop-Recommendation Dataset — Reproducible Pipeline

This is a **pipeline**, not a pre-built dataset. Read this section first —
it explains a real tension in the original brief and how this pipeline
resolves it honestly.

## Why there's no ready-made "hundreds of thousands of rows" CSV in this folder

Two of the original requirements pull in opposite directions:

1. *"Do NOT fabricate or randomly generate values. Use only real public data."*
2. *"50–100 samples per district... hundreds of thousands of rows."*

Real Indian agricultural statistics (Soil Health Card, Ministry of
Agriculture crop production, IMD) are published at **district/season**
granularity — India has ~766 districts × 3 seasons ≈ **2,300 base rows**.
There is no genuine, unique source that hands you 50–100 *independent*
real samples per district without either resampling (i.e. synthesizing
variation — the exact thing rule #1 forbids) or adding a real second
dimension.

**The real second dimension this pipeline uses is time.** NASA POWER
has genuine daily meteorology back to 1981. Pulling ~20 years of real
weather per district-season turns ~2,300 base rows into **~46,000
genuinely real rows** (2,300 × 20), each backed by actual observed/
reanalysis data for that specific year — not invented variation. Push
the window to 30+ years and you're past 60,000+ real rows. That is the
honest way to hit a large row count from real sources.

## Why I'm not running this pipeline for you right now

My code sandbox in this conversation has no outbound internet access,
so I can't execute a multi-hour, thousands-of-requests download job
from here (it would also blow past ISRIC SoilGrids' 5-requests/minute
fair-use limit if rushed). What's in this folder is a **complete,
tested pipeline** you run locally where you have network access. I
unit-tested the core logic (season aggregation, texture classification,
outlier flagging) against synthetic inputs to confirm the code itself
is correct — see `src/utils.py` and the script docstrings for what was
verified.

## Pipeline stages

| # | Script | What it does | Requires manual download first? |
|---|---|---|---|
| 1 | `01_get_districts.py` | Builds `state, district, latitude, longitude` via OpenStreetMap Nominatim geocoding | Optional: Census 2011 district list from data.gov.in for the full 766-district run (a 15-district real seed list is included so the pipeline runs end-to-end out of the box) |
| 2 | `02_fetch_climate_nasa_power.py` | Real multi-year seasonal temperature/humidity/rainfall per district | No — public API, no key |
| 3 | `03_fetch_soil_isric.py` | Real soil pH + texture-derived soil type per district | No — public API, no key |
| 4 | `04_prepare_soil_health_card.py` | Real available N/P/K (kg/ha) per district | **Yes** — soilhealth.dac.gov.in / data.gov.in has no stable bulk-download URL; script tells you exactly what to download and where to put it |
| 5 | `05_prepare_crop_season_stats.py` | Real major/secondary crop per district-season, ranked by actual production | **Yes** — same reason as above, for Ministry of Agriculture crop production statistics |
| 6 | `06_merge_and_validate.py` | Joins everything into the final schema; runs all requested validation (duplicates, missing values, outliers, impossible combinations); writes `output/india_crop_dataset.csv` and `output/cleaning_log.txt` | No |

If you skip steps 4/5 (no manual download), the pipeline still runs end
to end — `nitrogen_kg_ha`, `phosphorus_kg_ha`, `potassium_kg_ha`,
`major_crop`, `secondary_crops` are simply left `null` for every row,
exactly per the "leave missing rather than invent" instruction.

## How to run it

```bash
pip install -r requirements.txt

cd src
python 01_get_districts.py          # ~1 request/sec to Nominatim — respects their fair-use policy
python 02_fetch_climate_nasa_power.py   # the long one: districts × years of API calls
python 03_fetch_soil_isric.py       # respects SoilGrids' 5 req/min limit
python 04_prepare_soil_health_card.py   # only useful after you've done the manual download — see its docstring
python 05_prepare_crop_season_stats.py  # same
python 06_merge_and_validate.py
```

Final outputs land in `output/`:
- `india_crop_dataset.csv` — the dataset
- `cleaning_log.txt` — every validation/cleaning decision made, in order
- (see `../DATA_DICTIONARY.md` for column-by-column source attribution)

`02` and `03` are checkpointed / resume-safe respectively so a long
national run surviving an interruption doesn't have to restart from
zero — re-running just picks up where it left off.

## What's still manual or approximate, and why

- **Soil Health Card NPK and crop-production stats (steps 4–5)** need a
  human to download from data.gov.in — their bulk-export links aren't
  stable enough to hardcode without risking a silent wrong-download.
- **`climate_zone` (Köppen) and `irrigation_availability`** are in the
  final schema but left null by default — they require joining a
  Köppen-Geiger raster and an Irrigation Census table respectively,
  which I've documented in `DATA_DICTIONARY.md` rather than guessed at.
- **Rabi's Oct–Mar season crosses a calendar-year boundary.** The
  merge script uses a documented simplification (keeps Rabi keyed to
  the October-start "agricultural year," the standard convention in
  Indian statistics) rather than a full day-level two-file stitch —
  flagged explicitly in `06_merge_and_validate.py`.
- **Genetic soil types (Alluvial/Black/Red/Laterite)** can't be derived
  from SoilGrids texture alone — see the caveat in
  `src/utils.classify_soil_texture` and `DATA_DICTIONARY.md`. Getting
  these right needs NBSS&LUP soil survey maps, a genuine extra
  data-acquisition step.

## Using this alongside your existing model (recommendation from the brief)

Your instinct in the brief — approach 2, keep the current
`recommendation_engine.py` model and use this regional dataset as a
second-stage lookup — lines up well with what Part 2 of your project
(regional crop intelligence / `data/regional_crops.json`) already does.
Once you've run this pipeline for real, `major_crop` /
`secondary_crops` per `(state, district, season)` is a drop-in
richer replacement for the static `regional_crops.json` lookup, and
`soil_type` / seasonal `soil_ph` / `nitrogen_kg_ha` etc. give you a
second, regionally-aware sanity check to compare against the ML
model's N/P/K/temperature/humidity/rainfall-based prediction — exactly
the "second-stage recommendation engine" pattern the brief suggests,
without touching the trained model at all.

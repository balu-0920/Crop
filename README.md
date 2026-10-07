# Smart Crop Recommendation System

An end-to-end ML-powered crop recommendation system: trains a model on soil +
climate data, serves it through a Flask API, and connects it to a
weather-aware frontend that recommends a crop, explains why, and layers on
**regional crop intelligence** so the final suggestion reflects what actually
grows well *in your area* - not just what the model alone would pick.

## Phase 7 summary - what changed

**Part 1 - Prediction pipeline audit (highest priority).**
The full training -> saving -> loading -> inference chain was inspected end
to end: dataset, `StandardScaler`, `LabelEncoder`, feature order, and the
saved `RandomForestClassifier`. The pipeline itself (scaling, feature order,
label decoding) was already correct. **The actual root cause of the
"always predicts Muskmelon" bug was upstream of the model**: rainfall was
being read from OpenWeatherMap's *current weather* endpoint, which only
reports a `rain` value when it is raining at that exact moment - otherwise
it's silently absent, and the old code treated that as `0`. Muskmelon (and
watermelon) are the crops in this dataset with the lowest average rainfall
(~25mm), so a hard `0` reliably pulled predictions toward them regardless of
every other input. Fix: rainfall is now computed from the **5-day/3-hour
forecast endpoint**, summing the next 24 hours of forecast rainfall. See
`fetchRainfallForecast()` in `weather/main.js` for the full explanation and
`scoring_utils.py` / `app.py` for the verified scaling + feature-order +
label-decoding chain. Debug logging (`DEBUG = True` in `app.py` and
`main.js`) prints every stage of a request - incoming JSON, feature vector,
scaled vector, raw model output, decoded prediction - and can be turned off
by setting `DEBUG = False` in both files once you've confirmed everything.

**Part 2 - Regional Crop Intelligence** (new, application-layer only - the ML
model is untouched). `geocoding.py` reverse-geocodes the searched city's
coordinates into Country / State / District using OpenStreetMap Nominatim
(free, no API key). `data/regional_crops.json` maps regions to their
commonly-grown crops, with automatic fallback: **district -> state ->
country -> global default**, exactly matching the brief. `recommendation_engine.py`
loads and queries this file.

**Part 3 - Smart Recommendation Engine** (new). Instead of showing only the
model's single top pick, `recommendation_engine.py` combines three
independent, explainable signals into one ranked list:

| Signal | Default weight | Source |
|---|---|---|
| ML confidence | 50% | the model's own `predict_proba()`, for every crop it knows |
| Regional popularity | 30% | rank position in `data/regional_crops.json` |
| Weather suitability | 20% | rule-based comparison against `crop_info.json` ideal ranges |

Weights live in `RECOMMENDATION_WEIGHTS` at the top of `recommendation_engine.py`
- change the three numbers there (they must sum to 1.0) and every score
recalculates automatically; nothing else needs to change. Each ranked crop
gets a 1-5 star rating and three ✔/⚠/✖ explanation badges (regional, weather,
soil/ML) - the same "Why Recommended" bullets shown in the UI.

## Folder structure

```
CropRecommendation_final/
│
├── app.py                      # Flask backend - /predict and /region routes
├── scoring_utils.py             # shared rule-based weather-suitability scoring
├── geocoding.py                 # Nominatim reverse geocoding (Part 2)
├── recommendation_engine.py     # regional lookup + ranked scoring (Part 2 + 3)
├── regional_data.py             # reads dataset_pipeline output (season-aware crops, soil)
├── soil_fit.py                  # crop soil-fit from training-data ranges
├── dataset_pipeline/            # offline NASA POWER / SoilGrids / SHC data pipeline
├── tests/                       # unit tests
│
├── data/
│   ├── Crop_recommendation.csv # training dataset (2200 rows, 22 crops)
│   ├── crop_info.json          # crop knowledge base (ideal ranges, tips, etc.)
│   └── regional_crops.json     # NEW - regional crop popularity database
│
├── models/
│   ├── crop_model.pkl          # trained Random Forest model (unchanged)
│   └── preprocessing.pkl       # scaler + label encoder + feature order
│
├── static/
│   └── images/                 # one SVG icon per crop, served at /static/images/<crop>.svg
│
├── weather/
│   ├── index.html              # frontend UI (weather + region + soil input + recommendations)
│   ├── style.css
│   └── main.js                 # fetch() calls to Flask, renders all result cards
│
├── requirements.txt
└── README.md
```

## How to run it

**1. Start the backend:**
```bash
cd CropRecommendation_final
pip install -r requirements.txt
python app.py
```
This loads the model, scaler, label encoder, crop knowledge base, and the
regional crop database once at startup (with a verification check on
feature order), then serves `POST /predict` and `POST /region` on
`http://127.0.0.1:5000`.

**2. Serve the frontend** (don't just double-click the HTML file - some
browsers block `fetch()` calls from `file://` origins):
```bash
cd weather
python -m http.server 8080
```
Then open `http://localhost:8080/index.html`.

**3. Use it:** search a city (or use your location) to load weather - this
also resolves your region and shows the **Current Region** and **Common
Crops in Your Region** cards right away. Fill in Nitrogen, Phosphorus,
Potassium, and Soil pH, then click **Recommend Crop**. Temperature,
humidity, and rainfall are read automatically from the weather/forecast data.

## API reference

### `POST /region`
Body: `{"lat": number, "lon": number}`.
Returns `{success, location: {country, state, district}, common_crops, matched_level}`.

### `POST /predict`
Body: `{N, P, K, temperature, humidity, ph, rainfall}` plus optional
`weather_condition`, `cloud_cover`, and (to avoid re-geocoding) `country`,
`state`, `district`, or `lat`/`lon`.

Returns:
- `prediction`, `confidence`, `top_predictions`, `crop_info`, `suitability`,
  `suggestions`, `why_this_crop`, `image_url`, `prediction_time` - the
  original ML-prediction fields, unchanged.
- `region` - the resolved `{country, state, district, matched_level}`.
- `common_crops` - the regional crop list used for ranking.
- `recommendations` - the Part 3 ranked list: each entry has `crop`,
  `overall_score` (0-100), `stars` (1-5), `score_breakdown`
  (ml_confidence / regional_popularity / weather_suitability), and `reasons`
  (the ✔/⚠/✖ explanation badges).
- `recommendation_weights` - the current `RECOMMENDATION_WEIGHTS`, so the UI
  can display exactly how the score was built.

## Notes

- **Rainfall** is sourced from OpenWeatherMap's 5-day/3-hour forecast API
  (summed over the next 24h), not the current-weather endpoint - see the
  Phase 7 summary above for why that was the actual cause of the muskmelon
  bug. The UI labels the rainfall source (forecast / extrapolated /
  estimated) so it's never mistaken for a live reading.
- **Geocoding** uses OpenStreetMap Nominatim, which is free and needs no API
  key but is soft-rate-limited (~1 req/sec) - results are cached in memory
  per session. If a lookup fails (offline, quota, unrecognized location),
  regional intelligence gracefully falls back to the global default crop
  list rather than breaking the prediction.
- `regional_crops.json` currently covers a representative set of Indian
  states/districts plus a couple of other countries as examples - extend it
  with more entries anytime; no code changes are needed, the fallback chain
  (district -> state -> country -> default) handles any gaps automatically.
- The crop images are simple generated SVG icons, not real photography.
- Ideal temperature/humidity/rainfall ranges in `crop_info.json`, and the
  regional popularity rankings in `regional_crops.json`, are reasonable
  general-knowledge approximations for a learning project, not authoritative
  agronomic data.
- CORS is enabled with `Access-Control-Allow-Origin: *` for local
  development. Restrict this to your actual frontend's domain before
  deploying anywhere public.

## Phase 8 summary - frontend redesign + regional dataset pipeline

**Frontend redesign (`weather/index.html`, `weather/style.css`, `weather/main.js`).**
The UI was rebuilt from a weather-app look into a two-column SaaS-dashboard
layout (search bar, region/weather/soil-input panels on the left, a large
prediction hero + detail cards + overall recommendations on the right).
Every element ID and class the JavaScript depends on was preserved, so this
was a visual-layer-only change: `app.py`, `recommendation_engine.py`,
`scoring_utils.py`, `geocoding.py`, the trained model, and all data files are
byte-for-byte unchanged. The only JS addition is a client-side **Prediction
History** panel (search/filter/sort/delete/export/pagination) backed by
`localStorage`, since the existing backend has no endpoint to read
predictions back out - see the comment block at the top of the history
section in `weather/main.js` for the full reasoning, and let me know if
you'd like a real `/history` Flask + MongoDB endpoint added instead (that
would be a backend change, done separately from this redesign).

**Regional dataset pipeline (`dataset_pipeline/`).** A reproducible Python
pipeline (not a pre-built dataset) that builds a real, India-wide regional
crop dataset from NASA POWER (climate), ISRIC SoilGrids (soil pH/texture),
OpenStreetMap Nominatim (district geocoding), and - via one manual download
each - India's Soil Health Card and Ministry of Agriculture crop-production
statistics. See `dataset_pipeline/README.md` for how to run it and the
honest discussion of why "no fabricated values" and "hundreds of thousands
of rows" pull in different directions, and how the pipeline resolves that
using genuine multi-year climate history rather than invented variation.
This folder is a standalone data-engineering asset - nothing in `app.py` or
`recommendation_engine.py` currently reads from it; `dataset_pipeline/README.md`
explains how its output could plug in as the second-stage regional engine
described in Phase 2/3 above, once you've run it.

## Phase 9 summary - Nitrogen/Potassium scale correction (real data)

`data/Crop_recommendation.csv`'s Nitrogen and Potassium columns were
capped at 0-140 and 5-205 kg/ha respectively for every crop - far below
what a real Indian Soil Health Card reports (Low/Medium/High bands of
<280/280-560/>560 for N and <108/108-280/>280 for K). Every record was
silently out-of-distribution for real farmer input. N and K were
rescaled with a documented, reproducible linear transform that
preserves each record's relative standing exactly (rank correlation
1.0) while matching the real SHC scale; the model was retrained on the
corrected data with identical hyperparameters. Phosphorus, temperature,
humidity, pH and rainfall were spot-checked and left unchanged - see
`CHANGELOG_NPK_CORRECTION.md` for full sources, the exact method, and
what's still worth auditing further. The frontend's soil-input hints
were updated to match.

## Phase 10 summary - regional data pipeline connected to the recommendation engine

### Data flow

```
Browser: city search -> OpenWeatherMap (current + 24h forecast rain)   [unchanged, frontend-side]
   |  lat/lon, weather, user soil inputs (N, P, K, pH)
   v
POST /predict
   1. Location -> district/region   geocoding.py (Nominatim; successful lookups cached on disk, 30 days)
   2. Season                        current month -> Kharif / Rabi / Zaid (ICAR definitions), or "season" in request
   3. Regional data                 regional_data.py reads the pipeline CSV, if it exists:
                                      district + season crops -> state + season crops (needs >= 3 districts)
                                      -> otherwise static data/regional_crops.json (district -> state -> country -> default)
   4. Soil information              district pH / N / P / K / texture from the pipeline CSV (if present)
   5. Weather                       live weather + 24h forecast rainfall sent by the frontend
   6. ML prediction                 Random Forest probabilities for all 22 crops (unchanged)
   7. Ranking                       recommendation_engine.build_recommendations()
```

### What is used from where

| Signal | Source | Used when |
|---|---|---|
| ML suitability | `models/crop_model.pkl` | always |
| Regional crop suitability | pipeline `major_crop` / `secondary_crops` for the current season, else static JSON | always (source reported in the response as `regional_data.source`) |
| Season | month of year; crops grown locally only in another season score 25 instead of 0 | pipeline data has season info for the district/state |
| Soil | district soil (pH, N, P, K) compared with the 10th-90th percentile band each crop shows in the training dataset (`soil_fit.py`) | the pipeline CSV has at least one soil value for the district |
| Weather | existing rule-based suitability (`scoring_utils.py`) | always |

Weights: ML 50% / Regional 30% / Weather 20% as before. When district soil exists, Soil takes 15%, carved proportionally out of the other three (ML 42.5 / Regional 25.5 / Weather 17 / Soil 15). With no soil data the scoring is exactly what it was before. Edit `RECOMMENDATION_WEIGHTS` / `SOIL_WEIGHT` in `recommendation_engine.py`.

### Important: the pipeline has to be run for any of this to be active

`dataset_pipeline/` is a set of scripts, not a dataset - no generated CSV ships with this project. Until you run it, the app reports `regional_data.source = "static_json"` and behaves as before. The loader looks for `dataset_pipeline/output/india_crop_dataset.csv` (or `dataset_pipeline/src/output/...`, which is where it lands when you run the scripts from `src/` as the pipeline README says), or the path in `REGIONAL_DATASET_PATH`. The file is re-read automatically when it changes; no restart needed.

What the pipeline can fill in, honestly:
- Climate (NASA POWER) and soil pH / texture (ISRIC SoilGrids): automatic, public APIs, no key.
- N / P / K and `major_crop` / `secondary_crops`: need the manual Soil Health Card and crop-production downloads (pipeline steps 4-5). Skip them and those columns stay empty; the engine then keeps using the static crop list, and only uses soil that exists (pH, probably).
- The seed list covers 15 districts; for others, the engine falls back to static data. Nothing is guessed or interpolated.

District matching is by normalised name (case/accents/"District" suffix ignored) against what OpenStreetMap returns; spelling differences (e.g. Mysore/Mysuru) will simply fall back. Government crop names are mapped to the model's names where they are plain synonyms (Paddy -> Rice, Arhar/Tur -> Pigeonpeas, ...) - see `CROP_ALIASES` in `regional_data.py`.

### Caching

- Reverse geocoding: in-memory + `cache/geocode_cache.json` (30-day TTL, survives restarts); also rate-limited to Nominatim's 1 request/second; failures are only remembered for 5 minutes and never written to disk.
- Regional dataset: parsed once, re-parsed only if the file's modification time changes.
- Weather is still fetched by the browser directly from OpenWeatherMap (unchanged).

### Configuration (.env)

Copy `.env.example` to `.env` (git-ignored). Optional keys: `NOMINATIM_USER_AGENT`, `REGIONAL_DATASET_PATH`, `CACHE_DIR`.

**Security note:** the OpenWeatherMap key is hard-coded in `weather/main.js` (pre-existing, left untouched to preserve the frontend). A key in browser JavaScript is visible to anyone who loads the page. Moving it into `.env` needs a small server-side weather proxy plus a one-line change in `main.js` - recommended as a follow-up, and consider regenerating that key if the code has been shared.

### New response fields (additive; the existing frontend ignores what it doesn't use)

`/region` adds `data_source`, `season`, `regional_soil`. `/predict` adds `regional_data {source, season, soil, seasonal_climate}`, `score_breakdown.soil_fit`, and `recommendation_weights` now reflects the weights actually used.

### Tests

`python -m unittest discover -s tests -v` (uses a throw-away CSV fixture written at test time; not real data).

### Files changed

New: `regional_data.py`, `soil_fit.py`, `tests/test_regional_data.py`, `.env.example`. Modified: `recommendation_engine.py`, `app.py`, `geocoding.py`, `requirements.txt`, `.gitignore`, `README.md`.

## Phase 11 summary - profitability and risk decision layer

### What data exists (audit result)

| Needed | In the project? | Status |
|---|---|---|
| Crop price | No | **Unavailable** - add `data/economics/market_prices.csv` |
| Cultivation cost | No | **Unavailable** - add `data/economics/cultivation_costs.csv` |
| Expected yield | No (no yield labels anywhere; no yield model) | **Unavailable** - add `data/economics/historical_yields.csv` |
| Regional production | Only via the pipeline, not yet generated | Used as crop lists when present (Phase 10) |
| Weather risk | Yes - current temperature, humidity, 24h rain vs crop ideal ranges | **Available** |

The three economics CSVs ship as header-only templates (`data/economics/README.md` has the columns, units and suggested public sources). Nothing is pre-filled: no price, cost or yield in this project is invented. Until you add real rows, yield / revenue / profit are returned as `{"available": false, "reason": ...}` and shown as "Unavailable" in the UI. `GET /data_status` shows what is loaded.

### Formulas

```
Revenue (INR/ha) = expected yield (t/ha) x price (INR/quintal) x 10
Profit  (INR/ha) = revenue - cultivation cost (INR/ha)
```
Expected yield = median of the latest 5 years of `historical_yields.csv` at the most specific level (district, then state, then national); range = 10th-90th percentile (min-max if < 5 values). Price/cost use the newest row (district -> state -> national); prices older than 1 year are flagged `stale`. Ranges cover yield variability only - price volatility is not modelled.

**Risk score (0-100, `risk.py`)** - weighted mean of the components that can be measured:

| Component | Weight | Definition |
|---|---|---|
| Temperature stress | 0.35 | `min(100, 100 x distance outside the crop's ideal range / range width)` for today's temperature (heat or cold) |
| Rainfall stress | 0.40 | same formula for rain: excess (waterlogging) or deficit (drought indicator) |
| Rainfall variability | 0.25 | district's year-to-year CV% of seasonal rainfall x 2, capped at 100; only with >= 5 years from the regional pipeline |

Levels: < 20 Low, < 45 Medium, else High. Components without data are dropped and the weights re-normalised. **Unavailable and reported as such:** a multi-week drought index (needs rainfall history) and forecast uncertainty (the weather input has no spread). Temperature and rain are a snapshot of current conditions, not a season outlook.

**Final ranking score (`decision.py`)** - every component is 0-100, higher is better:

```
final = sum(w_i x component_i) / sum(w_i)    over the components that are available

suitability  w=0.45   existing recommendation score (ML + regional + weather + soil)
yield        w=0.15   expected yield / best yield ever recorded for that crop in historical_yields.csv
profit       w=0.25   profit / highest profit among the candidates (a loss scores 0)
safety       w=0.15   100 - risk score
```
These weights are a documented judgement call, **not fitted to outcome data**: suitability is the only component backed by a trained model; profit outweighs yield because it already contains yield; risk is a moderate modifier. Edit `DECISION_WEIGHTS` (`decision.py`) and `RISK_WEIGHTS` (`risk.py`). With no economics data, `final = 0.75 x suitability + 0.25 x (100 - risk)`; the response lists `components_used` and `components_unavailable` for every crop.

Known limitations: weather already feeds suitability, so risk partly overlaps with it (risk measures severity outside the ideal range, suitability measures overall fit); a crop with economics data and one without are scored on different component sets, so fill the data for all candidate crops to compare fairly; profit is an estimate from historical figures, not a forecast of farm results.

### API

`POST /predict` -> each item in `recommendations` now has a `decision` object: `final_score`, `suitability_pct`, `expected_yield`, `market_price`, `cultivation_cost`, `revenue`, `profit`, `risk {score, level, components, drivers, unavailable}`, `components`, `components_used`, `components_unavailable`, `main_reasons`. The list is sorted by `final_score`; top-level `decision_weights` is returned too. `GET /data_status` reports economics/regional data availability.

### Frontend

`weather/index.html`, `main.js`, `style.css`: a "Recommended crop" card (suitability, expected yield, revenue, profit, risk level, final score, main reasons) above the existing ranked list; unavailable fields are shown as "Unavailable" with the reason on hover. Each list row also shows a risk badge.

### Tests

`python -m unittest discover -s tests -v` (22 tests; CSV fixtures are test-only placeholders written to a temp dir).

## Phase 12 summary - explainable AI

`explain.py` uses SHAP (`TreeExplainer`) on the Random Forest to explain each prediction. For the ML-predicted crop (and the top-ranked crop, if different) `/predict` returns `explanations[crop_key]`: per-feature `contribution` in percentage points of the model's probability (base rate + contributions = model probability), sorted by influence. If SHAP is unavailable it falls back to the model's global feature importance (`method: "feature_importance"`). The frontend shows the bars under the recommended-crop card.

Honesty note (also returned in the response): this shows how the trained model weighs the inputs for one prediction. It is not causal proof that those factors drive real crop success, and inputs that move together in the training data share credit. Tests: `tests/test_explain.py`.

## Phase 13 summary - agricultural RAG assistant (independent of the crop ML model)

```
PDFs (rag/documents/) -> pypdf text per page -> sentence-packed chunks (~900 chars, overlap, never spanning pages)
  -> local embeddings (fastembed, BAAI/bge-small-en-v1.5, ONNX, no PyTorch) -> vector store (rag/index/: vectors.npy + chunks.json + manifest.json)
Question -> embed -> top-5 cosine matches -> drop chunks below RAG_MIN_SCORE -> LLM (Anthropic) sees ONLY those numbered passages
  -> answer with [n] citations + the documents/pages cited + the retrieved passages (with scores)
```

- **Grounding:** if no chunk is relevant enough, the API says the documents don't contain enough information *without calling the LLM*; the LLM is also told to answer `INSUFFICIENT_CONTEXT` if the passages don't cover the question, and to ignore any instructions inside passages.
- **Vector store:** a NumPy matrix with exact cosine search (vectors are pre-normalised). It is the simplest dependable option at this scale (a few thousand chunks) and adds no database dependency; `rag/store.py` is the only file to change if you move to Chroma/FAISS.
- **Metadata:** each chunk keeps document filename, PDF title, page number, chunk id; the manifest keeps SHA-256, page/chunk counts, ingest time and the embedding model. Re-running ingest only re-embeds new/changed PDFs and drops deleted ones.
- **Keys:** `ANTHROPIC_API_KEY` (and optional `RAG_LLM_MODEL`) are read from `.env` only. Without a key, `/ask` returns 503 plus the retrieved passages.
- **No documents are bundled.** Add authoritative PDFs (ICAR/SAU package of practices, FAO, government advisories) to `rag/documents/`. Scanned PDFs (no text layer) are skipped - OCR is not supported.
- **Tuning:** `RAG_MIN_SCORE` (default 0.45) is a similarity floor that depends on the embedding model; check it against your own documents. The assistant is single-turn (no chat memory).

Endpoints: `POST /api/agriculture/ask` `{"question": "..."}` -> `{success, question, answer, answerable, sources[{document,title,page,passage}], retrieved_context[{document,page,score,excerpt,used_in_answer,...}]}`; `GET /api/agriculture/status`. UI: `weather/assistant.html` (linked from the top bar). Tests: `tests/test_rag.py` (uses a hashing embedder and stub LLM so it runs offline).

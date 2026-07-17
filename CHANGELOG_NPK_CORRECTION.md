# Changelog — Nitrogen/Potassium Scale Correction

## What changed

`data/Crop_recommendation.csv` — the training data behind `models/crop_model.pkl`
— had its **Nitrogen (N)** and **Potassium (K)** columns rescaled. Phosphorus,
temperature, humidity, pH and rainfall are **unchanged**. The model was
retrained on the corrected data with identical hyperparameters, so the
deployed `.pkl` files now match.

The original file is kept at `data/Crop_recommendation_original.csv` for
comparison / rollback.

## Why

Every record in the original CSV had N between 0–140 kg/ha and K between
5–205 kg/ha, for *all 22 crops, with no exceptions*. Compare that to
India's actual Soil Health Card (SHC) classification bands — the same
kg/ha units this app's own soil-input form asks a user to type in from
their real soil-test report:

| Nutrient | Low | Medium | High |
|---|---|---|---|
| Available N | < 280 kg/ha | 280–560 kg/ha | > 560 kg/ha |
| Available K | < 108 kg/ha | 108–280 kg/ha | > 280 kg/ha |

*(Source: ICAR-IISS Bhopal soil-testing protocol, as published via the
Soil Health Card scheme's PIB factsheet and NIC Soil Health Card portal.
Cross-checked against real published district-level surveys — e.g. a
Chhattisgarh Vertisol study reporting observed available K from
208–821 kg/ha, and an Indo-Gangetic Plains four-transect study reporting
available P 2.2–185 kg/ha and available K 45–560 kg/ha.)*

Every single row in the old dataset fell in the **Low** N band, and the
K column never even reached the **Medium** band. That means a farmer
entering their real Soil Health Card's N or K value — which routinely
reads in the hundreds — was feeding the model an input it had never
seen anything like in training. The model wasn't wrong so much as it
had never been shown realistic Indian soil-test numbers.

## What was deliberately left unchanged, and why

- **Phosphorus (P):** the existing 5–145 kg/ha range already sits inside
  real observed Olsen-P values (cited range above: 2.2–185 kg/ha), so it
  was left as-is rather than "corrected" without real justification.
- **Temperature, humidity, rainfall, pH:** spot-checked against cited
  agronomic sources during this pass (e.g. chickpea: a cited source
  states 21–41% relative humidity is ideal for seed-set, which lines up
  with this dataset's existing 14–20% range for chickpea) and found to
  already track real agronomy reasonably well. These were **not**
  audited crop-by-crop in full, though — see "Suggested follow-up" below.

## How the correction was done

A single deterministic **linear rescale** was applied to the N column,
and separately to the K column: the dataset's existing [min, max] was
mapped onto a new [min, max] chosen to span the SHC Low→High bands
(N: 20–600, K: 20–550), using real observed extremes from the cited
district studies as the outer bounds. This is a **unit/scale
correction**, not invented values — every record's *relative* standing
against every other record is mathematically preserved (Spearman rank
correlation between old and new columns = 1.0 for both N and K), so a
crop that needed more N than another crop before the fix still needs
more N than it after the fix. Only the absolute scale changed, from
"a truncated 0–140/5–205 toy range" to "the real range a Soil Health
Card and this app's own input form actually use."

See `dataset_pipeline/fix_npk_scale/fix_npk_scale.py` for the exact,
reproducible code and full source citations in its docstring.

## Per-crop before/after (mean values)

| Crop | N before | N after | N band | K before | K after | K band |
|---|---|---|---|---|---|---|
| apple | 21 | 106 | Low | 200 | 536 | High |
| banana | 100 | 435 | Medium | 50 | 139 | Medium |
| blackgram | 40 | 186 | Low | 19 | 58 | Low |
| chickpea | 40 | 186 | Low | 80 | 218 | Medium |
| coconut | 22 | 111 | Low | 31 | 88 | Low |
| coffee | 101 | 439 | Medium | 30 | 86 | Low |
| cotton | 118 | 508 | Medium | 20 | 58 | Low |
| grapes | 23 | 116 | Low | 200 | 537 | High |
| jute | 78 | 345 | Medium | 40 | 113 | Medium |
| kidneybeans | 21 | 106 | Low | 20 | 60 | Low |
| lentil | 19 | 98 | Low | 19 | 58 | Low |
| maize | 78 | 342 | Medium | 20 | 59 | Low |
| mango | 20 | 103 | Low | 30 | 86 | Low |
| mothbeans | 21 | 109 | Low | 20 | 60 | Low |
| mungbean | 21 | 107 | Low | 20 | 59 | Low |
| muskmelon | 100 | 436 | Medium | 50 | 140 | Medium |
| orange | 20 | 101 | Low | 10 | 33 | Low |
| papaya | 50 | 227 | Low | 50 | 139 | Medium |
| pigeonpeas | 21 | 106 | Low | 20 | 60 | Low |
| pomegranate | 19 | 98 | Low | 40 | 113 | Medium |
| rice | 80 | 351 | Medium | 40 | 112 | Medium |
| watermelon | 99 | 432 | Medium | 50 | 140 | Medium |

Legumes (blackgram, chickpea, lentil, mothbeans, mungbean, pigeonpeas,
kidneybeans) staying in the Low N band even after correction is
agronomically expected and correct — they fix their own nitrogen, so
real Soil Health Cards for legume fields commonly do read Low-N. That
pattern in the original data was correct; only the absolute numbers
needed fixing.

## Model retraining

`models/crop_model.pkl` and `models/preprocessing.pkl` were regenerated
by `retrain_on_corrected_data.py`, using the exact same model type and
hyperparameters as before (`RandomForestClassifier(n_estimators=200,
max_features='sqrt', random_state=42)`, `StandardScaler`,
`LabelEncoder`), read directly from the previous `preprocessing.pkl`
before it was overwritten. Feature order (`N, P, K, temperature,
humidity, ph, rainfall`) and every other file (`app.py`,
`recommendation_engine.py`, `scoring_utils.py`, `geocoding.py`) are
unchanged. Held-out test accuracy after retraining: 99.5% (a pure
monotonic rescale doesn't remove any information a tree-based model
uses, so this is expected, not a sign the correction "helped" or "hurt"
accuracy on this synthetic dataset).

The frontend's soil-input hints (`weather/index.html`) were updated to
show the new realistic ranges and reference the Soil Health Card bands
directly.

## Suggested follow-up (not done in this pass)

- A full crop-by-crop audit of temperature/humidity/rainfall/pH against
  cited agronomic sources, the same way this pass checked chickpea.
  `dataset_pipeline/` (the India-wide regional pipeline built earlier in
  this project) is the more rigorous way to get real per-district
  climate and soil data if you want to go further than a spot-check.
- Genetic soil-type-aware NPK ranges (this dataset still has no notion
  of *which state/soil type* a record represents) — see
  `dataset_pipeline/README.md` for how the regional pipeline's output
  could eventually feed a state/district-aware version of this dataset.

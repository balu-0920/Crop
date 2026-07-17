# recommendation_engine.py
#
# Part 2 (Regional Crop Intelligence) + Part 3 (Smart Recommendation Engine).
#
# This module NEVER touches the ML model, the scaler, or the label
# encoder - it only ever *consumes* things the model already produced
# (a probability for every crop it knows about) and combines that with
# two other, non-ML signals:
#   - regional popularity  (data/regional_crops.json)
#   - weather suitability  (scoring_utils.analyze_suitability, same rule-based
#                            logic already used to explain the single ML pick)
#
# Everything here is plain Python + JSON. If you want to change how much
# each signal matters, edit RECOMMENDATION_WEIGHTS below - nothing else
# needs to change.

import json
import os

from scoring_utils import analyze_suitability

REGIONAL_CROPS_PATH = os.path.join("data", "regional_crops.json")

# ---------------------------------------------------------------------------
# Configurable weights (Part 3 requirement: "these weights should be
# configurable"). Must sum to 1.0. Change these three numbers to change how
# the Overall Recommendation Score is calculated - nothing else in this
# file needs to be touched.
# ---------------------------------------------------------------------------
RECOMMENDATION_WEIGHTS = {
    "ml": 0.50,
    "regional": 0.30,
    "weather": 0.20,
}

assert abs(sum(RECOMMENDATION_WEIGHTS.values()) - 1.0) < 1e-6, "RECOMMENDATION_WEIGHTS must sum to 1.0"

# How many candidate crops to show in the final ranked list.
MAX_RECOMMENDATIONS = 6

# Regional popularity is rank-based: the 1st crop listed for a
# region scores 100, the 2nd scores 85, and so on down to a floor of 40
# for anything further down the list. A crop that isn't in the regional
# list at all scores 0 ("rarely cultivated locally").
_RANK_SCORES = [100, 85, 70, 55, 40]
_RANK_FLOOR = 40


# ---------------------------------------------------------------------------
# Regional Crop Database loading (Part 2)
#
# Loaded ONCE at import time - same reasoning as the ML model in app.py:
# this JSON file never changes while the server runs, so re-reading it on
# every request would be pure waste.
# ---------------------------------------------------------------------------
with open(REGIONAL_CROPS_PATH, encoding="utf-8") as f:
    _REGIONAL_DB = json.load(f)

print(f"Regional crop database loaded: {len(_REGIONAL_DB)} top-level regions.")

_COMMON_KEY = "_common"
_DEFAULT_KEY = "_default"


def get_regional_crops(country=None, state=None, district=None):
    """
    Look up common crops for a location, falling back from district ->
    state -> country -> global default, exactly as required:
    "If district is unavailable, fall back to state. If state is
    unavailable, fall back to country."

    Returns (crop_list, matched_level) where matched_level is one of
    "district", "state", "country", "default" - used by the UI to say
    *which* level of the fallback chain actually matched, e.g.
    "Common Crops in Hyderabad, Telangana" vs a more generic
    "Common Crops in India" when only the country matched.
    """
    country_node = _REGIONAL_DB.get(country) if country else None

    if country_node:
        state_node = country_node.get(state) if state else None

        if state_node:
            # District is a leaf list in some regions and a dict (with its
            # own _common) in others - handle both shapes.
            if district and district in state_node:
                district_value = state_node[district]
                if isinstance(district_value, list):
                    return district_value, "district"

            if _COMMON_KEY in state_node:
                return state_node[_COMMON_KEY], "state"

        if _COMMON_KEY in country_node:
            return country_node[_COMMON_KEY], "country"

    default_node = _REGIONAL_DB.get(_DEFAULT_KEY, {})
    return default_node.get(_COMMON_KEY, []), "default"


# ---------------------------------------------------------------------------
# Crop name normalization
#
# regional_crops.json uses human-readable display names ("Kidney Beans",
# "Blackgram"). The ML model's label encoder uses lowercase, no-space keys
# ("kidneybeans", "blackgram") - crop_info.json is keyed the same way. This
# is the ONE place that maps between the two, so every other function can
# just use normalized keys without worrying about spacing/casing.
# ---------------------------------------------------------------------------
def normalize_crop_key(display_name):
    return display_name.strip().lower().replace(" ", "").replace("-", "")


# ---------------------------------------------------------------------------
# Regional popularity score for one candidate crop.
# ---------------------------------------------------------------------------
def _regional_score(crop_key, regional_crops_normalized):
    if crop_key not in regional_crops_normalized:
        return 0
    rank = regional_crops_normalized.index(crop_key)
    if rank < len(_RANK_SCORES):
        return _RANK_SCORES[rank]
    return _RANK_FLOOR


# ---------------------------------------------------------------------------
# Weather suitability score for one candidate crop.
# Reuses the exact same rule-based logic already used to explain the
# single ML-predicted crop (scoring_utils.analyze_suitability) - a crop
# that isn't in crop_info.json (no known ideal ranges) gets a neutral 50
# rather than being unfairly punished for missing data.
# ---------------------------------------------------------------------------
def _weather_score(crop_key, crop_info, weather):
    details = crop_info.get(crop_key)
    if not details:
        return None  # caller decides how to treat "unknown"
    return analyze_suitability(details, weather)["overall_percentage"]


def _reason_bullets(regional_rank_score, weather_score, ml_score, in_ml_vocabulary):
    """Builds the ✔/⚠/✖ explanation lines shown under each recommendation."""
    reasons = []

    if regional_rank_score >= 70:
        reasons.append({"ok": True, "text": "Common in your region"})
    elif regional_rank_score > 0:
        reasons.append({"ok": True, "text": "Grown in your region, though less widely"})
    else:
        reasons.append({"ok": False, "text": "Rarely cultivated locally"})

    if weather_score is None:
        reasons.append({"ok": None, "text": "Weather ideal-range data unavailable for this crop"})
    elif weather_score >= 75:
        reasons.append({"ok": True, "text": "Weather suitable"})
    elif weather_score >= 50:
        reasons.append({"ok": None, "text": "Weather partially suitable"})
    else:
        reasons.append({"ok": False, "text": "Weather not ideal right now"})

    if not in_ml_vocabulary:
        reasons.append({"ok": None, "text": "Not in the ML model's trained crop list"})
    elif ml_score >= 40:
        reasons.append({"ok": True, "text": "Soil & climate model favors this crop"})
    elif ml_score >= 10:
        reasons.append({"ok": None, "text": "Soil & climate model shows partial support"})
    else:
        reasons.append({"ok": False, "text": "Not indicated by soil/climate model"})

    return reasons


def _stars(overall_score):
    """Convert a 0-100 score into a 1-5 star rating for display."""
    return max(1, min(5, round(overall_score / 20)))


# ---------------------------------------------------------------------------
# Main entry point (Part 3).
#
# ml_probabilities: {crop_key -> 0-100 probability} for EVERY class the
#     model knows, straight from model.predict_proba() - not just the
#     top-3. This lets a crop that's #2 or #3 in the region's list still
#     get its real ML score even if it wasn't the model's top pick.
# common_crops: display-name list returned by get_regional_crops().
# crop_info: the full data/crop_info.json dict (for ideal ranges + display names).
# weather: {"temperature", "humidity", "rainfall"} - same values sent to the model.
# ---------------------------------------------------------------------------
def build_recommendations(ml_probabilities, common_crops, crop_info, weather):
    regional_normalized = [normalize_crop_key(c) for c in common_crops]

    # Candidate pool = every regional crop + the model's top predictions.
    # A dict keyed by normalized crop key keeps the union de-duplicated
    # while preserving one display name per crop.
    candidates = {}

    for display_name in common_crops:
        candidates[normalize_crop_key(display_name)] = display_name

    top_ml_crops = sorted(ml_probabilities.items(), key=lambda kv: kv[1], reverse=True)[:5]
    for crop_key, _ in top_ml_crops:
        details = crop_info.get(crop_key)
        display_name = details["name"] if details else crop_key.capitalize()
        candidates.setdefault(crop_key, display_name)

    scored = []
    for crop_key, display_name in candidates.items():
        in_ml_vocabulary = crop_key in ml_probabilities
        ml_score = ml_probabilities.get(crop_key, 0)

        regional_score = _regional_score(crop_key, regional_normalized)
        weather_score = _weather_score(crop_key, crop_info, weather)
        weather_score_for_math = weather_score if weather_score is not None else 50  # neutral, not punitive

        overall = (
            ml_score * RECOMMENDATION_WEIGHTS["ml"]
            + regional_score * RECOMMENDATION_WEIGHTS["regional"]
            + weather_score_for_math * RECOMMENDATION_WEIGHTS["weather"]
        )

        scored.append({
            "crop": display_name,
            "crop_key": crop_key,
            "overall_score": round(overall, 1),
            "stars": _stars(overall),
            "score_breakdown": {
                "ml_confidence": round(ml_score, 1),
                "regional_popularity": regional_score,
                "weather_suitability": weather_score,  # may be None -> shown as "N/A" in UI
            },
            "reasons": _reason_bullets(regional_score, weather_score, ml_score, in_ml_vocabulary),
            "in_regional_list": regional_score > 0,
            "in_ml_vocabulary": in_ml_vocabulary,
        })

    scored.sort(key=lambda r: r["overall_score"], reverse=True)
    return scored[:MAX_RECOMMENDATIONS]

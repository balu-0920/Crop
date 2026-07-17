# scoring_utils.py
#
# Small, dependency-free helper functions for turning a raw weather value
# into an explainable "how suitable is this?" score.
#
# IMPORTANT: nothing in this file is machine learning. It's plain
# if/else rule-based comparison against a crop's known ideal ranges
# (from data/crop_info.json). It is used in two places:
#   1. app.py            - suitability of the SINGLE crop the ML model predicted
#   2. recommendation_engine.py - suitability of EVERY candidate crop being ranked
# Keeping it here (instead of copy-pasted in both files) means there is
# exactly one place that defines what "suitable" means.


def score_factor(value, ideal_range, tolerance_ratio=0.2):
    """
    Classify a single value (e.g. today's temperature) against a crop's
    ideal [low, high] range, with a tolerance band around that range so a
    value just outside it isn't unfairly called "unsuitable".

    Returns a dict with a 0-100 score, a short status key, a display
    symbol, and a human-readable label.
    """
    low, high = ideal_range
    tolerance = (high - low) * tolerance_ratio

    if low <= value <= high:
        return {"score": 100, "status": "suitable", "symbol": "✔", "label": "Suitable"}
    if low - tolerance <= value < low:
        return {"score": 65, "status": "slightly_low", "symbol": "⚠", "label": "Slightly Low"}
    if high < value <= high + tolerance:
        return {"score": 65, "status": "slightly_high", "symbol": "⚠", "label": "Slightly High"}
    if value < low - tolerance:
        return {"score": 30, "status": "too_low", "symbol": "✖", "label": "Too Low"}
    return {"score": 30, "status": "too_high", "symbol": "✖", "label": "Too High"}


def analyze_suitability(crop_details, weather):
    """
    Compares live weather (temperature, humidity, rainfall) against one
    crop's ideal ranges. Returns per-factor breakdowns plus one overall
    percentage (a simple average of the three factor scores).

    `crop_details` is a single entry from data/crop_info.json (must have
    ideal_temperature / ideal_humidity / ideal_rainfall). `weather` is a
    dict with temperature / humidity / rainfall keys.
    """
    temperature_result = score_factor(weather["temperature"], crop_details["ideal_temperature"])
    humidity_result = score_factor(weather["humidity"], crop_details["ideal_humidity"])
    rainfall_result = score_factor(weather["rainfall"], crop_details["ideal_rainfall"])

    overall = round(
        (temperature_result["score"] + humidity_result["score"] + rainfall_result["score"]) / 3
    )

    return {
        "temperature": temperature_result,
        "humidity": humidity_result,
        "rainfall": rainfall_result,
        "overall_percentage": overall
    }

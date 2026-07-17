# app.py
#
# Flask backend for the Smart Crop Recommendation System.
#
#   Part 1 - ML prediction pipeline (unchanged model, verified end-to-end)
#   Part 2 - Regional Crop Intelligence (geocoding.py + regional_crops.json)
#   Part 3 - Smart Recommendation Engine (recommendation_engine.py)
#
# Parts 2 and 3 are deliberately kept in their own modules and never touch
# the model, scaler, or label encoder - they only ever consume the model's
# own output (a probability per crop). See recommendation_engine.py.

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify

from scoring_utils import analyze_suitability
from geocoding import reverse_geocode
from recommendation_engine import get_regional_crops, build_recommendations, RECOMMENDATION_WEIGHTS

# ---------------------------------------------------------------------------
# Load the trained model and preprocessing objects ONCE, when the server starts.
#
# Why "once" matters: joblib.load() reads a file from disk and rebuilds a
# Python object from it - this is relatively slow (tens to hundreds of
# milliseconds). If we loaded the model INSIDE the /predict route function,
# we'd reload it from disk on every single request, which would make every
# prediction slower than it needs to be, for no benefit (the model file
# never changes while the server is running).
#
# Loading it here, at module level (outside any function), means it runs
# exactly once - when Python first imports/runs this file - and the loaded
# objects then stay in memory, ready to be reused by every request.
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join("models", "crop_model.pkl")
PREPROCESSING_PATH = os.path.join("models", "preprocessing.pkl")

model = joblib.load(MODEL_PATH)
preprocessing = joblib.load(PREPROCESSING_PATH)

# preprocessing.pkl was saved in the notebook as a dictionary containing
# three related objects - we pull each one out here under a clear name.
scaler = preprocessing["scaler"]
label_encoder = preprocessing["label_encoder"]
feature_names = preprocessing["feature_names"]

print("Model and preprocessing objects loaded successfully.")

# ---------------------------------------------------------------------------
# Model verification: confirm the feature order this preprocessing.pkl
# expects EXACTLY matches what the model was trained on. This is a cheap
# safety check that runs once at startup - if a preprocessing.pkl from a
# different training run ever gets swapped in by mistake, the server
# refuses to start rather than silently making predictions with columns
# in the wrong order (which produces confident-looking but WRONG output,
# since the model has no way to know the columns got shuffled).
# ---------------------------------------------------------------------------
EXPECTED_FEATURE_ORDER = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

assert feature_names == EXPECTED_FEATURE_ORDER, (
    f"Feature order mismatch! preprocessing.pkl expects {feature_names}, "
    f"but the pipeline requires {EXPECTED_FEATURE_ORDER}. "
    "Do not proceed - this would cause silently wrong predictions."
)
print(f"Feature order verified: {feature_names}")

# ---------------------------------------------------------------------------
# DEBUG logging.
#
# Set DEBUG = False to silence this once you've confirmed everything works -
# it's intentionally verbose (every stage of one prediction, end to end) so
# you can see exactly where a bad value would enter the pipeline.
# ---------------------------------------------------------------------------
DEBUG = True


def debug_log(label, value):
    if DEBUG:
        print(f"[DEBUG] {label}: {value}")

# ---------------------------------------------------------------------------
# Load the crop knowledge base ONCE at startup too - same reasoning as the
# model above. This is a plain dictionary (loaded from JSON), keyed by the
# exact crop name strings the model itself outputs (e.g. "rice", "maize"),
# so we can look a prediction straight up with no extra mapping step.
# ---------------------------------------------------------------------------
CROP_INFO_PATH = os.path.join("data", "crop_info.json")

with open(CROP_INFO_PATH) as f:
    crop_info = json.load(f)

print(f"Crop knowledge base loaded: {len(crop_info)} crops.")

# Create the Flask application object.
# __name__ is a built-in Python variable that holds the name of the current module
# ("app" if this file is run directly). Flask uses it internally to figure out
# the root path of your project, so it knows where to look for things like the
# templates/ and static/ folders later on.
app = Flask(__name__)


# ---------------------------------------------------------------------------
# Helper: get the model's confidence and top-N predictions.
#
# model.predict() only ever gives you the single best answer. But most
# scikit-learn classifiers (including our RandomForest) can also return
# predict_proba() - the probability it assigned to EVERY class, not just
# the winner. That's a genuine measure of how confident the model was,
# not something we're computing after the fact - it comes straight from
# how the model itself weighed the evidence (e.g. for a Random Forest,
# roughly "what fraction of the 200 trees voted for this crop").
#
# Not every model type supports this (e.g. a plain SVC without
# probability=True does not) - hasattr() lets us check gracefully instead
# of assuming it's always available.
# ---------------------------------------------------------------------------
def get_top_predictions(input_scaled, top_n=3):
    if not hasattr(model, "predict_proba"):
        return None  # caller will handle "confidence unavailable"

    probabilities = model.predict_proba(input_scaled)[0]  # one probability per class

    # argsort gives indices that would sort ascending; [::-1] reverses to
    # descending, so the first index is the model's most confident class.
    top_indices = np.argsort(probabilities)[::-1][:top_n]

    results = []
    for idx in top_indices:
        crop_name = label_encoder.classes_[idx]
        results.append({
            "crop": crop_name,
            "probability": round(float(probabilities[idx]) * 100, 1)
        })
    return results


# ---------------------------------------------------------------------------
# Helper: get the model's probability for EVERY class it knows about, not
# just the top few. This is what the Smart Recommendation Engine (Part 3)
# needs - it has to score crops that might not be the model's #1 pick
# (e.g. the region's most common crop, ranked #2 or #3 by the model).
# Still a single predict_proba() call, just returned in full instead of
# truncated to top-N.
# ---------------------------------------------------------------------------
def get_all_probabilities(input_scaled):
    if not hasattr(model, "predict_proba"):
        return {}
    probabilities = model.predict_proba(input_scaled)[0]
    return {
        label_encoder.classes_[idx]: round(float(prob) * 100, 1)
        for idx, prob in enumerate(probabilities)
    }


# ---------------------------------------------------------------------------
# Rule-based weather suitability scoring (score_factor / analyze_suitability)
# now lives in scoring_utils.py, imported above - it's shared with
# recommendation_engine.py so both the single ML prediction and the ranked
# recommendation list explain "suitable weather" using the exact same logic.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helper: turn the suitability breakdown into plain-language suggestions.
# Still rule-based - if/else logic keyed off the "status" buckets above,
# not a model. Kept separate from analyze_suitability() so each function
# has one clear job (easier to read, test, and reuse).
# ---------------------------------------------------------------------------
def generate_suggestions(suitability, weather):
    suggestions = []

    humidity_status = suitability["humidity"]["status"]
    if humidity_status in ("too_low", "slightly_low"):
        suggestions.append("Humidity is lower than ideal. Increase irrigation frequency.")
    elif humidity_status in ("too_high", "slightly_high"):
        suggestions.append("Humidity is higher than ideal. Ensure good field drainage and airflow.")

    rainfall_status = suitability["rainfall"]["status"]
    if rainfall_status in ("too_high", "slightly_high"):
        suggestions.append("Heavy rainfall expected. Avoid additional irrigation today.")
    elif rainfall_status in ("too_low", "slightly_low"):
        suggestions.append("Rainfall is below ideal levels. Plan for supplemental irrigation.")

    temperature_status = suitability["temperature"]["status"]
    if temperature_status in ("too_high", "slightly_high"):
        suggestions.append("Temperature is higher than ideal. Consider shade netting or extra watering.")
    elif temperature_status in ("too_low", "slightly_low"):
        suggestions.append("Temperature is lower than ideal. Consider protective covering or delaying sowing.")

    # Optional weather condition text (e.g. "Rain", "Thunderstorm") passed
    # from the frontend, if the weather API provided one.
    condition = (weather.get("condition") or "").lower()
    if "rain" in condition or "storm" in condition or "drizzle" in condition:
        suggestions.append("Current conditions show active rainfall. Hold off on manual irrigation.")

    if not suggestions:
        suggestions.append("Current weather conditions are well-matched to this crop - no adjustments needed.")

    return suggestions


# ---------------------------------------------------------------------------
# Helper: build the "Why this crop?" explanation bullets.
# Also rule-based, reusing the same suitability breakdown so we never
# duplicate the comparison logic - it's computed once in analyze_suitability()
# and just rendered differently here.
# ---------------------------------------------------------------------------
def build_why_this_crop(suitability):
    reasons = []

    for factor_name, factor_key in (("Temperature", "temperature"), ("Humidity", "humidity"), ("Rainfall", "rainfall")):
        factor = suitability[factor_key]
        reasons.append(f"{factor['symbol']} {factor_name} is {factor['label'].lower()}.")

    # We don't have numeric ideal N/P/K ranges per crop (out of scope for
    # this dataset's crop_info), but the model itself already used the
    # user's N/P/K values as direct inputs to reach this prediction - so we
    # can truthfully say soil nutrients were factored in, without inventing
    # numeric thresholds we don't actually have.
    reasons.append("✔ Soil nutrient levels (N, P, K) were used by the model to reach this prediction.")

    return reasons


# ---------------------------------------------------------------------------
# CORS (Cross-Origin Resource Sharing)
#
# Browsers enforce a "same-origin policy": JavaScript running on one origin
# (e.g. a file opened directly, or a page served from a different port) is
# blocked by the BROWSER ITSELF from reading responses from a different
# origin (e.g. http://127.0.0.1:5000), unless that server explicitly says
# "this origin is allowed" via a response header.
#
# This is not a Flask limitation or a network issue - the request actually
# reaches our server fine either way. It's the browser that refuses to hand
# the response back to your JavaScript code unless it sees this header.
#
# @app.after_request registers a function that Flask runs on EVERY response,
# right before sending it back to the client - a good place to add a header
# that should apply to all routes, without repeating it in every view function.
# ---------------------------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# @app.route(...) is a DECORATOR. It tells Flask:
# "whenever a request comes in for this URL path, call the function below it."
# This is how Flask connects a URL to a piece of Python code - this connection
# is called a "route".
@app.route("/")
def home():
    # Whatever this function returns becomes the HTTP response body.
    # Flask automatically wraps a plain string into a valid HTTP response for us.
    return "Crop Recommendation backend is running."


# ---------------------------------------------------------------------------
# Regional Crop Intelligence endpoint (Part 2).
#
# Separate from /predict on purpose: the frontend calls this as soon as
# weather loads for a city (so the "Current Region" and "Common Crops"
# cards can appear immediately), and then reuses the resolved
# country/state/district when it later calls /predict - avoiding a
# second, redundant geocoding round trip when the user clicks
# "Get Recommendation".
# ---------------------------------------------------------------------------
@app.route("/region", methods=["POST"])
def region():
    data = request.get_json(silent=True) or {}

    lat, lon = data.get("lat"), data.get("lon")
    if lat is None or lon is None:
        return jsonify({"success": False, "message": "lat and lon are required."}), 400

    location = reverse_geocode(lat, lon)
    debug_log("Reverse geocode result", location)

    common_crops, matched_level = get_regional_crops(
        country=location["country"], state=location["state"], district=location["district"]
    )

    return jsonify({
        "success": True,
        "location": location,
        "common_crops": common_crops,
        "matched_level": matched_level,  # "district" | "state" | "country" | "default"
    })


# ---------------------------------------------------------------------------
# The prediction endpoint.
#
# methods=["POST"] means this route ONLY responds to POST requests.
# A GET request to the same URL will get an automatic 405 Method Not Allowed
# from Flask - we don't have to write that check ourselves.
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    # request.get_json() parses the incoming request body as JSON and gives
    # us back a normal Python dict. silent=True means: if parsing fails
    # (e.g. the body wasn't valid JSON at all), return None instead of
    # raising an exception - we then handle that "None" case ourselves below,
    # so we can return a clean error message instead of a server crash.
    data = request.get_json(silent=True)

    debug_log("Incoming JSON", data)

    if data is None:
        return jsonify({
            "success": False,
            "message": "Request body must be valid JSON."
        }), 400  # 400 Bad Request: the client sent something we can't process

    # --- Validation: check every required field is present ---
    # feature_names came from preprocessing.pkl, so this list is guaranteed
    # to match exactly what the model was trained on - no hardcoding here.
    for field in feature_names:
        if field not in data:
            return jsonify({
                "success": False,
                "message": f"Missing field: {field}"
            }), 400

    # --- Validation: check every value is numeric ---
    for field in feature_names:
        value = data[field]
        if not isinstance(value, (int, float)):
            return jsonify({
                "success": False,
                "message": f"Field '{field}' must be a number, got: {type(value).__name__}"
            }), 400

    # --- Build model input ---
    # The model expects a 2D table (rows x columns), even for a single
    # prediction - so we build a one-row DataFrame, with columns in the
    # EXACT same order the model was trained on.
    input_df = pd.DataFrame([[data[field] for field in feature_names]], columns=feature_names)

    debug_log("Processed feature vector (unscaled)", input_df.to_dict(orient="records")[0])

    # Everything from here on can theoretically fail in ways outside our
    # direct control (e.g. a corrupted model file, an unexpected value that
    # slips past validation) - wrapping it means a genuine prediction
    # failure returns a clean JSON error instead of a raw Python traceback
    # leaking back to the client.
    try:
        # Apply the SAME scaling used during training. We only ever call
        # .transform() here, never .fit() - fitting again would recompute
        # the scaling based on this one input row, which is not what we want.
        input_scaled = scaler.transform(input_df)

        debug_log("Scaled feature vector", input_scaled.tolist())

        # model.predict() always returns an array (it's built to handle
        # batches of rows), so even for our single row we get back
        # something like array([15]) - we pull out that one value with [0].
        predicted_encoded = model.predict(input_scaled)[0]

        debug_log("Raw model prediction (encoded)", predicted_encoded)

        # Convert the encoded integer back into the actual crop name string.
        predicted_crop = label_encoder.inverse_transform([predicted_encoded])[0]

        debug_log("Decoded prediction", predicted_crop)

        # --- Confidence & top-3 (Phase 5) ---
        # See get_top_predictions() above: this comes directly from the
        # model's own predict_proba(), not a separate calculation.
        top_predictions = get_top_predictions(input_scaled, top_n=3)
        confidence = top_predictions[0]["probability"] if top_predictions else None

        debug_log("Top-3 probability scores", top_predictions)

        # --- Crop details lookup (Phase 5) ---
        crop_details = crop_info.get(predicted_crop)

        # --- Weather suitability analysis (Phase 4) ---
        # Rule-based only - see analyze_suitability(). Requires the crop's
        # ideal ranges, so we only run it if we actually found crop_details.
        weather = {
            "temperature": data["temperature"],
            "humidity": data["humidity"],
            "rainfall": data["rainfall"],
            "condition": data.get("weather_condition"),   # optional, may be absent
            "cloud_cover": data.get("cloud_cover"),        # optional, may be absent
        }

        suitability = None
        suggestions = None
        why_this_crop = None

        if crop_details:
            suitability = analyze_suitability(crop_details, weather)
            suggestions = generate_suggestions(suitability, weather)
            why_this_crop = build_why_this_crop(suitability)

        # -----------------------------------------------------------------
        # Part 2 - Regional Crop Intelligence
        #
        # The frontend already resolved the location via /region and sends
        # the result along (country/state/district) so we don't geocode
        # twice per click. If those fields are missing (e.g. an older
        # client, or geocoding failed client-side) but lat/lon were sent
        # instead, we resolve it here as a fallback.
        # -----------------------------------------------------------------
        country = data.get("country")
        state = data.get("state")
        district = data.get("district")

        if not any([country, state, district]) and data.get("lat") is not None and data.get("lon") is not None:
            location = reverse_geocode(data["lat"], data["lon"])
            country, state, district = location["country"], location["state"], location["district"]

        common_crops, matched_level = get_regional_crops(country=country, state=state, district=district)
        debug_log("Regional crops", {"level": matched_level, "crops": common_crops})

        # -----------------------------------------------------------------
        # Part 3 - Smart Recommendation Engine
        #
        # Uses the model's FULL probability distribution (every crop, not
        # just the top-3) so regionally-common crops that weren't the
        # model's #1 pick still get scored fairly. The ML model itself is
        # never called again here - this is one extra read of the same
        # predict_proba() output already computed above.
        # -----------------------------------------------------------------
        all_probabilities = get_all_probabilities(input_scaled)
        recommendations = build_recommendations(
            ml_probabilities=all_probabilities,
            common_crops=common_crops,
            crop_info=crop_info,
            weather=weather,
        )
        debug_log("Ranked recommendations", recommendations)

        # jsonify() builds a proper Flask Response object with the correct
        # Content-Type header (application/json), from our Python dict.
        return jsonify({
            "success": True,
            "prediction": predicted_crop,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "crop_info": crop_details,
            "weather_used": weather,
            "suitability": suitability,
            "suggestions": suggestions,
            "why_this_crop": why_this_crop,
            "image_url": f"/static/images/{predicted_crop}.svg",
            "prediction_time": datetime.now().strftime("%I:%M:%S %p"),

            # --- Regional Crop Intelligence (Part 2) ---
            "region": {"country": country, "state": state, "district": district, "matched_level": matched_level},
            "common_crops": common_crops,

            # --- Smart Recommendation Engine (Part 3) ---
            "recommendations": recommendations,
            "recommendation_weights": RECOMMENDATION_WEIGHTS,
        })  # no explicit status code -> Flask defaults to 200 OK

    except Exception as exc:
        # Catch-all for anything unexpected during prediction itself
        # (as opposed to bad input, which we already rejected above with
        # a specific 400 message). 500 = Internal Server Error: the
        # request was valid, but our server failed to fulfill it.
        debug_log("Prediction pipeline raised an exception", repr(exc))
        return jsonify({
            "success": False,
            "message": "Prediction failed unexpectedly. Please try again."
        }), 500


# This guard means: "only run the development server if this file is executed
# directly (python app.py), not if it's imported by something else."
if __name__ == "__main__":
    # debug=True enables:
    #  1. Auto-reload: the server restarts itself whenever you save a code change.
    #  2. Detailed error pages in the browser if something crashes (great while learning).
    # We will turn this off before deploying anywhere real - it's a development-only setting.
    app.run(debug=True)

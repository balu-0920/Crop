

import os
import json

# Load settings from .env (see .env.example) BEFORE importing modules that read them.
from dotenv import load_dotenv
load_dotenv()

import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify

from scoring_utils import analyze_suitability
from geocoding import reverse_geocode
from recommendation_engine import get_regional_context, build_recommendations
from regional_data import store as regional_store
from decision import apply_decision_layer, DECISION_WEIGHTS
from explain import explain_prediction
from economics import data_status as economics_data_status


MODEL_PATH = os.path.join("models", "crop_model.pkl")
PREPROCESSING_PATH = os.path.join("models", "preprocessing.pkl")

model = joblib.load(MODEL_PATH)
preprocessing = joblib.load(PREPROCESSING_PATH)


scaler = preprocessing["scaler"]
 
label_encoder = preprocessing["label_encoder"]
feature_names = preprocessing["feature_names"]

print("Model and preprocessing objects loaded successfully.")


EXPECTED_FEATURE_ORDER = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

assert feature_names == EXPECTED_FEATURE_ORDER, (
    f"Feature order mismatch! preprocessing.pkl expects {feature_names}, "
    f"but the pipeline requires {EXPECTED_FEATURE_ORDER}. "
    "Do not proceed - this would cause silently wrong predictions."
)
print(f"Feature order verified: {feature_names}")


DEBUG = True


def debug_log(label, value):
    if DEBUG:
        print(f"[DEBUG] {label}: {value}")


CROP_INFO_PATH = os.path.join("data", "crop_info.json")

with open(CROP_INFO_PATH) as f:
    crop_info = json.load(f)

print(f"Crop knowledge base loaded: {len(crop_info)} crops.")

if regional_store.available():
    print("Regional crops: using pipeline dataset where it has the district/state, static JSON otherwise.")
else:
    print("Regional crops: no pipeline dataset found - using static data/regional_crops.json "
          "(run dataset_pipeline to generate real regional data; see README).")


app = Flask(__name__)


def get_top_predictions(input_scaled, top_n=3):
    if not hasattr(model, "predict_proba"):
        return None  # caller will handle "confidence unavailable"

    probabilities = model.predict_proba(input_scaled)[0]  # one probability per class

    top_indices = np.argsort(probabilities)[::-1][:top_n]

    results = []
    for idx in top_indices:
        crop_name = label_encoder.classes_[idx]
        results.append({
            "crop": crop_name,
            "probability": round(float(probabilities[idx]) * 100, 1)
        })
    return results

def get_all_probabilities(input_scaled):
    if not hasattr(model, "predict_proba"):
        return {}
    probabilities = model.predict_proba(input_scaled)[0]
    return {
        label_encoder.classes_[idx]: round(float(prob) * 100, 1)
        for idx, prob in enumerate(probabilities)
    }

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

    condition = (weather.get("condition") or "").lower()
    if "rain" in condition or "storm" in condition or "drizzle" in condition:
        suggestions.append("Current conditions show active rainfall. Hold off on manual irrigation.")

    if not suggestions:
        suggestions.append("Current weather conditions are well-matched to this crop - no adjustments needed.")

    return suggestions



def build_why_this_crop(suitability):
    reasons = []

    for factor_name, factor_key in (("Temperature", "temperature"), ("Humidity", "humidity"), ("Rainfall", "rainfall")):
        factor = suitability[factor_key]
        reasons.append(f"{factor['symbol']} {factor_name} is {factor['label'].lower()}.")


    reasons.append("✔ Soil nutrient levels (N, P, K) were used by the model to reach this prediction.")

    return reasons


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response



@app.route("/")
def home():
  
    return "Crop Recommendation backend is running."


@app.route("/data_status")
def data_status():
    """Which economics inputs are loaded (prices / costs / yields) vs unavailable."""
    return jsonify({"success": True, "economics": economics_data_status(),
                    "regional_dataset": regional_store.info(), "decision_weights": DECISION_WEIGHTS})


@app.route("/region", methods=["POST"])
def region():
    data = request.get_json(silent=True) or {}

    lat, lon = data.get("lat"), data.get("lon")
    if lat is None or lon is None:
        return jsonify({"success": False, "message": "lat and lon are required."}), 400

    location = reverse_geocode(lat, lon)
    debug_log("Reverse geocode result", location)

    context = get_regional_context(
        country=location["country"], state=location["state"], district=location["district"],
        season=data.get("season"),
    )

    return jsonify({
        "success": True,
        "location": location,
        "common_crops": context["crops"],
        "matched_level": context["matched_level"],  # "district" | "state" | "country" | "default"
        "data_source": context["source"],           # "pipeline_dataset" | "static_json" | "default"
        "season": context["season"],
        "regional_soil": context["soil"],
    })


@app.route("/predict", methods=["POST"])
def predict():
   
    data = request.get_json(silent=True)

    debug_log("Incoming JSON", data)

    if data is None:
        return jsonify({
            "success": False,
            "message": "Request body must be valid JSON."
        }), 400  # 400 Bad Request: the client sent something we can't process

   
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

    
    input_df = pd.DataFrame([[data[field] for field in feature_names]], columns=feature_names)

    debug_log("Processed feature vector (unscaled)", input_df.to_dict(orient="records")[0])

    
    try:
        
        input_scaled = scaler.transform(input_df)

        debug_log("Scaled feature vector", input_scaled.tolist())

        
        predicted_encoded = model.predict(input_scaled)[0]

        debug_log("Raw model prediction (encoded)", predicted_encoded)

        # Convert the encoded integer back into the actual crop name string.
        predicted_crop = label_encoder.inverse_transform([predicted_encoded])[0]

        debug_log("Decoded prediction", predicted_crop)

        
        top_predictions = get_top_predictions(input_scaled, top_n=3)
        confidence = top_predictions[0]["probability"] if top_predictions else None

        debug_log("Top-3 probability scores", top_predictions)

        # --- Crop details lookup (Phase 5) ---
        crop_details = crop_info.get(predicted_crop)

     
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

     
        country = data.get("country")
        state = data.get("state")
        district = data.get("district")

        if not any([country, state, district]) and data.get("lat") is not None and data.get("lon") is not None:
            location = reverse_geocode(data["lat"], data["lon"])
            country, state, district = location["country"], location["state"], location["district"]

        context = get_regional_context(
            country=country, state=state, district=district, season=data.get("season")
        )
        common_crops, matched_level = context["crops"], context["matched_level"]
        debug_log("Regional context", {k: context[k] for k in ("matched_level", "source", "season", "soil")})

       
        all_probabilities = get_all_probabilities(input_scaled)
        recommendations, weights_used = build_recommendations(
            ml_probabilities=all_probabilities,
            common_crops=common_crops,
            crop_info=crop_info,
            weather=weather,
            other_season_crops=context["other_season_crops"],
            regional_soil=context["soil"],
        )
        # Profitability + risk layer: re-ranks by the documented final score (decision.py).
        recommendations, _ = apply_decision_layer(
            recommendations, crop_info, weather, state=state, district=district,
            seasonal_climate=context["seasonal_climate"],
        )
        debug_log("Ranked recommendations", recommendations)

       
        # --- Explainable AI: why the model favours the predicted crop (and the
        # top-ranked crop, if different). Failure here must never break /predict.
        explanations = {}
        try:
            crops_to_explain = [predicted_crop]
            if recommendations and recommendations[0]["crop_key"] != predicted_crop:
                crops_to_explain.append(recommendations[0]["crop_key"])
            explanations = explain_prediction(
                model, label_encoder, feature_names,
                {f: data[f] for f in feature_names}, input_scaled, crops_to_explain,
            )
        except Exception as exc:
            debug_log("Explanation failed", repr(exc))

        return jsonify({
            "success": True,
            "prediction": predicted_crop,
            "explanations": explanations,  # {crop_key: {method, base_value, features[...], note}}
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
            "recommendation_weights": weights_used,
            "decision_weights": DECISION_WEIGHTS,

            # --- Regional data pipeline connection ---
            "regional_data": {
                "source": context["source"],  # "pipeline_dataset" | "static_json" | "default"
                "season": context["season"],  # None when the static list (no season info) was used
                "soil": context["soil"],
                "seasonal_climate": context["seasonal_climate"],
            },
        })  # no explicit status code -> Flask defaults to 200 OK

    except Exception as exc:
      
        debug_log("Prediction pipeline raised an exception", repr(exc))
        return jsonify({
            "success": False,
            "message": "Prediction failed unexpectedly. Please try again."
        }), 500



if __name__ == "__main__":
    app.run(debug=True)

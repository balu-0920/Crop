"""
The agent's four tools - and nothing else. Each is a thin wrapper over something that already exists:

  get_weather            OpenWeatherMap current + forecast (same endpoints/logic as weather/main.js)
  predict_crop           the existing /predict pipeline (ML model + recommendation + decision layer)
  estimate_yield_profit  existing economics.py (price/cost/yield data) + risk.py (weather risk)
  search_agri_documents  the existing RAG pipeline (rag/)

Rules every tool follows:
  * Returns a dict. On failure or missing data it returns {"available": False, "reason": ...} -
    it never substitutes a guessed number.
  * These outputs are the ONLY source of numbers the agent may state.
  * Secrets stay in the environment (.env); error messages never echo URLs/keys.
"""
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import requests

OWM = "https://api.openweathermap.org/data/2.5/"
WEATHER_TTL_SECONDS = 600
HTTP_TIMEOUT = 8

_cache = {}
_cache_lock = threading.Lock()


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable


def _unavailable(reason, **extra):
    return {"available": False, "reason": reason, **extra}


# --------------------------------------------------------------------------- weather
def get_weather(location=None, lat=None, lon=None):
    key = os.environ.get("OPENWEATHER_API_KEY")
    if not key:
        return _unavailable("OPENWEATHER_API_KEY is not set in .env, so weather cannot be fetched.")
    if not location and (lat is None or lon is None):
        return _unavailable("No location given (need a place name or lat/lon).")

    cache_key = ("q", location.strip().lower()) if location else ("c", round(float(lat), 2), round(float(lon), 2))
    with _cache_lock:
        hit = _cache.get(cache_key)
        if hit and time.time() - hit[0] < WEATHER_TTL_SECONDS:
            return {**hit[1], "cached": True}

    try:
        params = {"q": location} if location else {"lat": lat, "lon": lon}
        cur = requests.get(OWM + "weather", params={**params, "units": "metric", "appid": key}, timeout=HTTP_TIMEOUT)
        cur.raise_for_status()
        cj = cur.json()
        coord = cj["coord"]
        result = {
            "available": True, "source": "OpenWeatherMap",
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "cached": False,
            "location": {"name": cj.get("name"), "country": (cj.get("sys") or {}).get("country"),
                         "lat": coord["lat"], "lon": coord["lon"]},
            "current": {
                "temperature_c": cj["main"]["temp"], "humidity_pct": cj["main"]["humidity"],
                "condition": ((cj.get("weather") or [{}])[0]).get("main"),
            },
            "forecast_24h": None, "forecast_5d": None,
        }
    except (requests.RequestException, KeyError, ValueError, TypeError) as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return _unavailable(f"Weather service request failed ({type(exc).__name__}{f', HTTP {status}' if status else ''}).")

    # Forecast is a separate call; if it fails we keep current weather but leave the forecast
    # explicitly unavailable - we do NOT fall back to an estimated rainfall.
    try:
        fc = requests.get(OWM + "forecast", params={"lat": coord["lat"], "lon": coord["lon"], "units": "metric",
                                                    "appid": key}, timeout=HTTP_TIMEOUT)
        fc.raise_for_status()
        slots = fc.json()["list"]
        next24 = slots[:8]  # 8 x 3-hour slots; a missing "rain" key means no rain forecast for that slot
        result["forecast_24h"] = {"rainfall_mm": round(sum((s.get("rain") or {}).get("3h", 0) for s in next24), 2)}
        temps = [s["main"]["temp"] for s in slots]
        result["forecast_5d"] = {
            "temp_min_c": min(temps), "temp_max_c": max(temps),
            "total_rain_mm": round(sum((s.get("rain") or {}).get("3h", 0) for s in slots), 2),
            "hours_covered": len(slots) * 3,
        }
    except (requests.RequestException, KeyError, ValueError, TypeError):
        result["forecast_note"] = "Forecast unavailable (request failed); rainfall is therefore unknown."

    with _cache_lock:
        _cache[cache_key] = (time.time(), result)
    return result


# ---------------------------------------------------------------- crop prediction
def make_predict_crop(predict_fn):
    def predict_crop(weather, soil, season=None, target_crop_key=None):
        """weather: output of get_weather; soil: {"N","P","K","ph"} supplied by the user."""
        if not weather or not weather.get("available"):
            return _unavailable("Crop prediction needs weather (temperature, humidity, rainfall) and it is unavailable.")
        rain = (weather.get("forecast_24h") or {}).get("rainfall_mm")
        if rain is None:
            return _unavailable("Crop prediction needs the 24h rainfall forecast, which is unavailable.")
        cur, loc = weather["current"], weather["location"]
        payload = {
            "N": soil["N"], "P": soil["P"], "K": soil["K"], "ph": soil["ph"],
            "temperature": cur["temperature_c"], "humidity": cur["humidity_pct"], "rainfall": rain,
            "weather_condition": cur.get("condition"), "lat": loc["lat"], "lon": loc["lon"],
        }
        if season:
            payload["season"] = season
        body, status = predict_fn(payload)
        if status != 200 or not body.get("success"):
            return _unavailable(f"Prediction pipeline error: {body.get('message', status)}")

        recs = body["recommendations"]
        summary = [{
            "rank": i, "crop": r["crop"], "crop_key": r["crop_key"],
            "suitability_pct": r["overall_score"], "final_score_pct": r["decision"]["final_score"],
            "risk_level": r["decision"]["risk"]["level"], "in_ml_vocabulary": r["in_ml_vocabulary"],
            "ml_probability_pct": r["score_breakdown"]["ml_confidence"] if r["in_ml_vocabulary"] else None,
        } for i, r in enumerate(recs, start=1)]

        out = {
            "available": True, "source": "trained Random Forest + scoring layer (existing /predict pipeline)",
            "ml_top_prediction": {"crop": body["prediction"], "probability_pct": body["confidence"]},
            "recommendations": summary, "region": body["region"], "regional_data_source": body["regional_data"]["source"],
            "season": body["regional_data"]["season"], "seasonal_climate": body["regional_data"]["seasonal_climate"],
            "weather_used": {
                k: body["weather_used"][k] for k in ("temperature", "humidity", "rainfall")},
        }
        top_expl = (body.get("explanations") or {}).get(body["prediction"])
        if top_expl:
            out["top_prediction_drivers"] = [
                {"feature": f["label"], "value": f["value"], "contribution_pts": f["contribution"], "direction": f["direction"]}
                for f in top_expl["features"][:3]]
        if target_crop_key:
            hit = next((s for s in summary if s["crop_key"] == target_crop_key), None)
            out["target_crop"] = {
                "key": target_crop_key,
                "in_model_vocabulary": hit["in_ml_vocabulary"] if hit else None,
                "rank_in_recommendations": hit["rank"] if hit else None,
                "note": None if hit else "Not among the top-ranked candidates (or not a crop the ML model knows).",
            }
        return out
    return predict_crop


# ------------------------------------------------------------------ yield / profit
def make_estimate_yield_profit(crop_info, economics_mod, risk_mod):
    def estimate_yield_profit(crop_key, weather, state=None, district=None, seasonal_climate=None):
        econ = economics_mod.estimate_economics(crop_key, state, district)
        weather_ok = weather and weather.get("available") and (weather.get("forecast_24h") or {}).get("rainfall_mm") is not None
        details = crop_info.get(crop_key)
        if weather_ok and details:
            w = {"temperature": weather["current"]["temperature_c"], "humidity": weather["current"]["humidity_pct"],
                 "rainfall": weather["forecast_24h"]["rainfall_mm"]}
            risk = risk_mod.assess_risk(details, w, seasonal_climate)
        elif not details:
            risk = _unavailable("No ideal temperature/rainfall ranges are stored for this crop, so weather risk cannot be measured.")
        else:
            risk = _unavailable("Weather risk needs current weather and the rainfall forecast, which are unavailable.")
        return {
            "available": True, "crop_key": crop_key,
            "source": "economics.py (price/cost/yield data files you supply) and risk.py (rule-based weather risk)",
            "expected_yield": econ["expected_yield"], "market_price": econ["market_price"],
            "cultivation_cost": econ["cultivation_cost"], "revenue": econ["revenue"], "profit": econ["profit"],
            "risk": risk,
        }
    return estimate_yield_profit


# ---------------------------------------------------------------------------- RAG
def make_search_agri_documents(get_pipeline):
    def search_agri_documents(query):
        from rag.pipeline import RagError
        pipeline = get_pipeline()
        try:
            res = pipeline.ask(query)
        except RagError as exc:
            return _unavailable(str(exc), retrieved_context=exc.retrieved)
        except Exception as exc:
            return _unavailable(f"Document search failed ({type(exc).__name__}).")
        texts = {c["chunk_id"]: c["text"] for c in pipeline.store.chunks}
        passages = [{**m, "text": texts.get(m["chunk_id"], m["excerpt"])}
                    for m in res["retrieved_context"] if m["used_in_answer"]]
        return {
            "available": bool(res["answerable"]), "reason": None if res["answerable"] else res["answer"],
            "source": "agricultural document index (rag/)", "answer": res["answer"] if res["answerable"] else None,
            "sources": res["sources"], "passages": passages,
        }
    return search_agri_documents


def build_tools(predict_fn, crop_info, economics_mod, risk_mod, get_rag_pipeline):
    tools = [
        Tool("get_weather", "Current weather and forecast for a place.", get_weather),
        Tool("predict_crop", "Existing ML crop recommendation + scoring for given weather and soil.",
             make_predict_crop(predict_fn)),
        Tool("estimate_yield_profit", "Expected yield, revenue, profit and weather risk for a crop.",
             make_estimate_yield_profit(crop_info, economics_mod, risk_mod)),
        Tool("search_agri_documents", "Retrieve and summarise passages from the agricultural document index.",
             make_search_agri_documents(get_rag_pipeline)),
    ]
    return {t.name: t for t in tools}

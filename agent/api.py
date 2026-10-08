"""POST /api/agent/advice - runs the LangGraph decision agent. Existing endpoints are untouched."""
import threading

from flask import Blueprint, jsonify, request

import economics
import risk
from rag.api import get_pipeline
from rag.llm import AnthropicLLM

from .graph import build_graph
from .tools import build_tools

bp = Blueprint("agent", __name__, url_prefix="/api/agent")

MAX_QUESTION_CHARS = 1000
SEASONS = ("Kharif", "Rabi", "Zaid")
_SOIL_LIMITS = {"N": (0, 2000), "P": (0, 2000), "K": (0, 2000), "ph": (0, 14)}

_graph = None
_lock = threading.Lock()


def configure_agent(predict_fn, crop_info, llm=None, region_lookup=None, rag_pipeline_getter=None):
    """Called once by app.py (after predict_core exists). Tests can inject a stub llm / rag getter."""
    global _graph
    if region_lookup is None:
        from geocoding import reverse_geocode
        region_lookup = reverse_geocode
    tools = build_tools(predict_fn, crop_info, economics, risk, rag_pipeline_getter or get_pipeline)
    with _lock:
        _graph = build_graph(tools, llm or AnthropicLLM(), crop_info, region_lookup)


def _validate(data):
    """Returns (clean_request, error_message)."""
    if not isinstance(data, dict):
        return None, "Request body must be a JSON object."
    q = data.get("question")
    if not isinstance(q, str) or len(q.strip()) < 3:
        return None, "Provide a 'question' string (at least 3 characters)."
    if len(q.strip()) > MAX_QUESTION_CHARS:
        return None, f"Question is too long (max {MAX_QUESTION_CHARS} characters)."
    req = {"question": q.strip()}

    loc = data.get("location")
    if loc is not None:
        if not isinstance(loc, str) or len(loc) > 100:
            return None, "'location' must be a string of at most 100 characters."
        req["location"] = loc.strip() or None
    lat, lon = data.get("lat"), data.get("lon")
    if (lat is None) != (lon is None):
        return None, "Provide both 'lat' and 'lon', or neither."
    if lat is not None:
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (lat, lon)) \
                or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return None, "'lat'/'lon' must be valid coordinates."
        req["lat"], req["lon"] = lat, lon

    soil = data.get("soil")
    if soil is not None:
        if not isinstance(soil, dict):
            return None, "'soil' must be an object with N, P, K, ph."
        clean_soil = {}
        for k, (lo, hi) in _SOIL_LIMITS.items():
            v = soil.get(k)
            if v is None:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not (lo <= v <= hi):
                return None, f"soil.{k} must be a number between {lo} and {hi}."
            clean_soil[k] = v
        req["soil"] = clean_soil
    season = data.get("season")
    if season is not None:
        if season not in SEASONS:
            return None, f"'season' must be one of {list(SEASONS)}."
        req["season"] = season
    return req, None


def assemble(question, state):
    final = dict(state["final"])
    facts, passages = final.pop("facts", []), final.pop("passages", [])
    plan = state["plan"]
    return {
        "question": question,
        "plan": {k: plan[k] for k in ("is_agricultural", "crop", "steps", "understanding_method")},
        "missing_information": plan["missing_information"],
        # 1) What the trained ML model + scoring layer say (statistical estimates, not facts about your farm)
        "model_prediction": {
            "kind": "Trained Random Forest + scoring layer. Statistical suitability estimates from a historical dataset.",
            "result": state.get("prediction")},
        # 2) Data fetched from outside the model: weather service, your price/cost/yield files, rule-based risk calculation
        "external_data": {
            "kind": "Weather service data and price/cost/yield data you supplied, plus a rule-based weather-risk calculation.",
            "weather": state.get("weather"), "economics_and_risk": state.get("economics")},
        # 3) Passages and a grounded answer from the indexed agricultural documents
        "retrieved_knowledge": {
            "kind": "Passages from the indexed agricultural documents, with sources.",
            "result": state.get("knowledge")},
        # 4) The AI's own reasoning over (1)-(3); numbers in it are validated against the tool outputs
        "final_reasoning": {
            "kind": "AI-written reasoning over the evidence above. It can be wrong; numbers are checked against the tool outputs.",
            **final},
        "evidence": {"facts": facts, "passages": passages},
        "tool_trace": state["trace"],
    }


@bp.route("/advice", methods=["POST"])
def advice():
    req, error = _validate(request.get_json(silent=True))
    if error:
        return jsonify({"success": False, "message": error}), 400
    if _graph is None:
        return jsonify({"success": False, "message": "Agent is not configured."}), 500
    try:
        state = _graph.invoke({"request": req, "trace": []})
    except Exception as exc:
        print(f"[agent] unexpected error: {exc!r}")
        return jsonify({"success": False, "message": "The agent failed unexpectedly. Check the server log."}), 500
    body = assemble(req["question"], state)
    status = body["final_reasoning"].get("status")
    if status in ("llm_not_configured", "llm_error"):
        return jsonify({"success": False, "message": body["final_reasoning"]["caveats"][0], **body}), \
            503 if status == "llm_not_configured" else 502
    return jsonify({"success": True, **body})

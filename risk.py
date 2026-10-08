# risk.py
#
# Weather / agronomic risk from MEASURABLE factors only. Every component is
# 0 (no risk) - 100 (severe). A component whose input isn't available is
# listed under "unavailable" and left out - never filled with a default.
#
# Stress of a value against a crop's ideal [low, high] range (crop_info.json):
#     stress = min(100, 100 x distance_outside_range / (high - low))
# i.e. 0 inside the range, 100 once the value is a full range-width outside.
#
# Components and their weights inside the risk score (documented heuristics,
# not fitted to outcomes):
#   temperature_stress   0.35   today's temperature vs ideal (heat or cold)
#   rainfall_stress      0.40   rain vs ideal: excess (flood/waterlogging) OR
#                               deficit (drought indicator) - only one can apply
#   rainfall_variability 0.25   year-to-year CV of the district's seasonal
#                               rainfall (NASA POWER via the regional pipeline);
#                               CV% x 2, capped at 100 (so CV 50%+ = max risk).
#                               Only when the pipeline dataset has >= 5 years.
# Risk = weighted mean over the available components.
#
# Levels: < 20 Low, < 45 Medium, otherwise High.
#
# NOT available (and reported as such): a true multi-week drought index
# (e.g. SPI needs rainfall history), and forecast uncertainty (the frontend
# sends one current reading + a 24h rain total, with no spread/ensemble).
#
# Caveat: temperature and rain here are a snapshot of current conditions,
# not a season-long outlook.

RISK_WEIGHTS = {"temperature_stress": 0.35, "rainfall_stress": 0.40, "rainfall_variability": 0.25}
LOW_BELOW, MEDIUM_BELOW = 20, 45
CV_TO_RISK = 2.0

ALWAYS_UNAVAILABLE = {
    "drought_index": "Needs multi-week rainfall history (e.g. SPI); only a 24h rain total is available.",
    "forecast_uncertainty": "The weather input has no forecast spread/ensemble to measure uncertainty.",
}


def _stress(value, ideal):
    low, high = ideal
    width = max(high - low, 1e-9)
    if value < low:
        return min(100.0, 100 * (low - value) / width), "low"
    if value > high:
        return min(100.0, 100 * (value - high) / width), "high"
    return 0.0, "ok"


def assess_risk(crop_details, weather, seasonal_climate=None):
    unavailable = dict(ALWAYS_UNAVAILABLE)
    components, drivers = {}, []

    if not crop_details:
        return {"score": None, "level": "Unknown", "components": {}, "drivers": [],
                "unavailable": {"all": "No ideal ranges for this crop, so weather risk can't be measured."}}

    t_score, t_dir = _stress(weather["temperature"], crop_details["ideal_temperature"])
    components["temperature_stress"] = round(t_score, 1)
    if t_score >= 1:
        drivers.append((t_score, "Heat stress: temperature above this crop's ideal range" if t_dir == "high"
                        else "Cold stress: temperature below this crop's ideal range"))

    r_score, r_dir = _stress(weather["rainfall"], crop_details["ideal_rainfall"])
    components["rainfall_stress"] = round(r_score, 1)
    if r_score >= 1:
        drivers.append((r_score, "Excess rainfall (waterlogging risk)" if r_dir == "high"
                        else "Rainfall deficit (drought indicator)"))

    cv = (seasonal_climate or {}).get("rainfall_cv_pct")
    if cv is not None:
        v_score = min(100.0, cv * CV_TO_RISK)
        components["rainfall_variability"] = round(v_score, 1)
        if v_score >= 40:
            drivers.append((v_score, f"Rainfall in this district varies a lot between years (CV {cv}%)"))
    else:
        unavailable["rainfall_variability"] = "Needs >= 5 years of district seasonal rainfall from the regional pipeline dataset."

    used = {k: RISK_WEIGHTS[k] for k in components}
    total = sum(used.values())
    score = round(sum(components[k] * w for k, w in used.items()) / total, 1)
    level = "Low" if score < LOW_BELOW else "Medium" if score < MEDIUM_BELOW else "High"
    drivers.sort(key=lambda d: d[0], reverse=True)
    return {"score": score, "level": level, "components": components,
            "weights_used": {k: round(w / total, 3) for k, w in used.items()},
            "drivers": [d[1] for d in drivers], "unavailable": unavailable}

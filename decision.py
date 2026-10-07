# decision.py
#
# Final ranking = suitability + expected yield + profitability + (low) risk.
#
# Every component is a 0-100 "higher is better" number:
#   suitability  the existing recommendation score (ML + regional + weather
#                + soil, see recommendation_engine.py)
#   yield        expected yield / best historical yield seen for that crop in
#                data/economics/historical_yields.csv (how close the expected
#                yield is to what the crop has achieved; comparable across crops)
#   profit       estimated profit / the highest profit among the candidates,
#                floored at 0 (a loss scores 0); needs >= 1 profitable candidate
#   safety       100 - risk score (risk.py)
#
#   final = sum(w_i x component_i) / sum(w_i)   over the AVAILABLE components
#
# Default weights (DECISION_WEIGHTS) are a judgement call, not fitted to data:
# suitability carries most weight because it is the only component backed by a
# trained model; profit outweighs yield because it already includes yield and
# price; risk is a moderate modifier. If yield/profit data is not loaded, those
# components are dropped and the rest are re-normalised (the response lists
# `components_used`), so with no economics data final = 0.75 x suitability +
# 0.25 x safety. Change the numbers below to change the ranking.

from economics import estimate_economics
from risk import assess_risk

DECISION_WEIGHTS = {"suitability": 0.45, "yield": 0.15, "profit": 0.25, "safety": 0.15}
assert abs(sum(DECISION_WEIGHTS.values()) - 1.0) < 1e-9


def _main_reasons(rec, econ, risk):
    reasons = [r["text"] for r in rec["reasons"] if r["ok"] is True][:2]
    if risk["level"] != "Unknown":
        if risk["drivers"]:
            reasons.append(f"{risk['level']} weather risk (main factor: {risk['drivers'][0][0].lower() + risk['drivers'][0][1:]})")
        else:
            reasons.append("Low weather risk: current temperature and rainfall are within this crop's ideal ranges")
    if econ["profit"]["available"]:
        reasons.append(f"Estimated profit INR {econ['profit']['value']:,}/ha")
    else:
        reasons.append("Profit not estimated: " + econ["profit"]["reason"].replace("Profit unavailable: ", ""))
    return reasons


def apply_decision_layer(recommendations, crop_info, weather, state=None, district=None, seasonal_climate=None):
    entries = []
    for rec in recommendations:
        econ = estimate_economics(rec["crop_key"], state, district)
        risk = assess_risk(crop_info.get(rec["crop_key"]), weather, seasonal_climate)
        entries.append((rec, econ, risk))

    profits = [e["profit"]["value"] for _, e, _ in entries if e["profit"]["available"]]
    best_profit = max(profits) if profits and max(profits) > 0 else None

    for rec, econ, risk in entries:
        comps = {"suitability": rec["overall_score"]}
        y = econ["expected_yield"]
        if y["available"] and y["reference_best"] > 0:
            comps["yield"] = min(100.0, 100 * y["value"] / y["reference_best"])
        if econ["profit"]["available"] and best_profit:
            comps["profit"] = max(0.0, 100 * econ["profit"]["value"] / best_profit)
        if risk["score"] is not None:
            comps["safety"] = 100 - risk["score"]

        total_w = sum(DECISION_WEIGHTS[k] for k in comps)
        final = sum(DECISION_WEIGHTS[k] * v for k, v in comps.items()) / total_w
        rec["decision"] = {
            "final_score": round(final, 1),
            "suitability_pct": rec["overall_score"],
            "expected_yield": econ["expected_yield"],
            "market_price": econ["market_price"],
            "cultivation_cost": econ["cultivation_cost"],
            "revenue": econ["revenue"],
            "profit": econ["profit"],
            "risk": risk,
            "components": {k: round(v, 1) for k, v in comps.items()},
            "components_used": {k: round(DECISION_WEIGHTS[k] / total_w, 3) for k in comps},
            "components_unavailable": [k for k in DECISION_WEIGHTS if k not in comps],
            "main_reasons": _main_reasons(rec, econ, risk),
        }

    recommendations.sort(key=lambda r: r["decision"]["final_score"], reverse=True)
    return recommendations, DECISION_WEIGHTS

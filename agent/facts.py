"""
Turns tool outputs into labelled FACTS (F1, F2, ...) and retrieved PASSAGES (D1, D2, ...), and
enforces the rule that the LLM may not introduce numbers of its own.

Numeric guard (strict, no rounding allowed): every number in the LLM's text must appear verbatim in
the evidence it relies on. For a reasoning point that means the facts/passages the point CITES (plus
the user's question and fact labels such as "24 h"); for the summary and caveats, which cite nothing,
any evidence. A sentence containing any other number is dropped and reported.
Limitations: only digits are checked, not numbers spelled out as words; and a number that exists in
the cited evidence but is attached to the wrong quantity cannot be detected by a number check.
"""
import re

_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
_CITE = re.compile(r"\[(?:F|D)\d+\]")


def numbers_in(text):
    text = _CITE.sub(" ", str(text))
    out = []
    for tok in _NUM.findall(text):
        try:
            out.append((tok.rstrip(",."), float(tok.replace(",", "").rstrip("."))))
        except ValueError:
            continue
    return out


def allowed_numbers(facts, passages, question, only_ids=None):
    """Numbers permitted in text. only_ids restricts to those evidence items (still plus the question)."""
    allowed = []
    for f in facts:
        if only_ids is not None and f["id"] not in only_ids:
            continue
        allowed += [v for _, v in numbers_in(f["label"])]
        if f["value"] is not None:
            allowed += [v for _, v in numbers_in(f["value"])]
    for p in passages:
        if only_ids is not None and p["id"] not in only_ids:
            continue
        allowed += [v for _, v in numbers_in(p["text"])]
    allowed += [v for _, v in numbers_in(question)]
    return allowed


def unsupported_numbers(text, allowed):
    bad = []
    for raw, val in numbers_in(text):
        if not any(abs(val - a) < 0.005 for a in allowed):
            bad.append(raw)
    return bad


class _Facts:
    def __init__(self):
        self.items = []

    def add(self, category, label, value=None, reason=None):
        fid = f"F{len(self.items) + 1}"
        self.items.append({"id": fid, "category": category, "label": label,
                           "value": value, "available": value is not None, "reason": reason})


def _field(f, label, cat, facts, fmt):
    if f and f.get("available"):
        facts.add(cat, label, fmt(f))
    else:
        facts.add(cat, label, None, (f or {}).get("reason", "unavailable"))


def build_facts(plan, weather, prediction, economics, knowledge):
    """Returns (facts, passages). Categories: external_data | model_prediction."""
    F = _Facts()

    # ---- external data: weather -------------------------------------------------
    if weather is not None:
        if weather.get("available"):
            loc, cur = weather["location"], weather["current"]
            F.add("external_data", "Weather location", f"{loc['name']}, {loc['country']}")
            F.add("external_data", "Current temperature", f"{cur['temperature_c']} C")
            F.add("external_data", "Current humidity", f"{cur['humidity_pct']} %")
            if cur.get("condition"):
                F.add("external_data", "Current conditions", cur["condition"])
            f24 = weather.get("forecast_24h")
            F.add("external_data", "Forecast rainfall, next 24 h",
                  f"{f24['rainfall_mm']} mm" if f24 else None, None if f24 else weather.get("forecast_note"))
            f5 = weather.get("forecast_5d")
            if f5:
                F.add("external_data", "5-day forecast",
                      f"temperature {f5['temp_min_c']} to {f5['temp_max_c']} C, total rain {f5['total_rain_mm']} mm")
        else:
            F.add("external_data", "Weather", None, weather.get("reason"))

    # ---- model prediction -------------------------------------------------------
    if prediction is not None:
        if prediction.get("available"):
            top = prediction["ml_top_prediction"]
            F.add("model_prediction", "ML model's most likely crop",
                  f"{top['crop']} ({top['probability_pct']} % model probability)")
            for r in prediction["recommendations"][:3]:
                F.add("model_prediction", f"Ranked recommendation {r['rank']}",
                      f"{r['crop']}: suitability {r['suitability_pct']} %, final score {r['final_score_pct']} %, "
                      f"{r['risk_level']} weather risk")
            if prediction.get("top_prediction_drivers"):
                F.add("model_prediction", "Main model drivers for its top crop (SHAP, percentage points)",
                      "; ".join(f"{d['feature']} {d['contribution_pts']:+} ({d['direction']})"
                                for d in prediction["top_prediction_drivers"]))
            t = prediction.get("target_crop")
            if t:
                if t["in_model_vocabulary"] is None:
                    F.add("model_prediction", f"Asked crop '{plan['crop']}' in ranked candidates", "no",
                          t["note"])
                elif t["in_model_vocabulary"] is False:
                    F.add("model_prediction", f"Asked crop '{plan['crop']}' known to the ML model", "no",
                          "The ML model was not trained on this crop, so it gives no ML suitability for it.")
                else:
                    F.add("model_prediction", f"Asked crop '{plan['crop']}' rank in recommendations",
                          str(t["rank_in_recommendations"]))
        else:
            F.add("model_prediction", "ML crop recommendation", None, prediction.get("reason"))

    # ---- external data: economics + weather risk (calculated from user-supplied data / weather) ----
    if economics is not None:
        name = plan["crop"]
        _field(economics["expected_yield"], f"Expected yield, {name}", "external_data", F,
               lambda f: f"{f['value']} t/ha (range {f['range'][0]} to {f['range'][1]}, "
                         f"{f['n_observations']} observations, {f['range_basis']})")
        _field(economics["market_price"], f"Market price, {name}", "external_data", F,
               lambda f: f"{f['value']} INR/quintal (as of {f['price_date']}{', STALE' if f.get('stale') else ''})")
        _field(economics["cultivation_cost"], f"Cultivation cost, {name}", "external_data", F,
               lambda f: f"{f['value']} INR/ha")
        _field(economics["revenue"], f"Estimated revenue, {name}", "external_data", F,
               lambda f: f"{f['value']} INR/ha (range {f['range'][0]} to {f['range'][1]})")
        _field(economics["profit"], f"Estimated profit, {name}", "external_data", F,
               lambda f: f"{f['value']} INR/ha (range {f['range'][0]} to {f['range'][1]})")
        risk = economics["risk"]
        if risk.get("score") is not None:
            drivers = "; ".join(risk["drivers"]) or "no stress factors detected"
            F.add("external_data", f"Weather risk for {name} (rule-based calculation)",
                  f"{risk['level']} ({risk['score']} out of 100); main factors: {drivers}")
        else:
            F.add("external_data", f"Weather risk for {name}", None,
                  risk.get("reason") or "; ".join(risk.get("unavailable", {}).values()) or "unavailable")

    # ---- retrieved knowledge ------------------------------------------------------
    passages = []
    if knowledge is not None and knowledge.get("available"):
        for i, p in enumerate(knowledge["passages"], start=1):
            passages.append({"id": f"D{i}", "document": p["document"], "title": p.get("title"),
                             "page": p["page"], "score": p["score"], "text": p["text"]})
    return F.items, passages


def render_for_llm(facts, passages):
    lines = []
    for f in facts:
        body = f"{f['value']}" if f["available"] else f"UNAVAILABLE - {f['reason']}"
        lines.append(f"[{f['id']}] ({f['category']}) {f['label']}: {body}")
    plines = [f"[{p['id']}] (document: {p['document']}, page {p['page']})\n{p['text']}" for p in passages]
    return "\n".join(lines) or "(no facts)", "\n\n".join(plines) or "(no passages retrieved)"

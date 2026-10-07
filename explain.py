# explain.py
#
# Per-prediction explanation of the ML model using SHAP (TreeExplainer),
# with a fallback to the model's global feature importance if SHAP is not
# installed or fails.
#
# What the numbers mean (SHAP, per prediction, for ONE crop):
#   model probability for that crop = base rate + sum of the 7 contributions
#   contribution > 0  -> this input pushed the model's probability for the crop UP
#   contribution < 0  -> it pushed the probability DOWN
# Units are percentage points of model probability.
#
# What they do NOT mean: this describes how the trained model uses its inputs
# for this one prediction. It is not causal evidence that changing the input
# would change real-world crop success, and contributions are specific to this
# model and to the training data's correlations (features that move together
# can share credit).

FEATURE_LABELS = {
    "N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)",
    "temperature": "Temperature", "humidity": "Humidity", "ph": "Soil pH", "rainfall": "Rainfall",
}

_explainer = None
_explainer_failed = False


def _get_explainer(model):
    global _explainer, _explainer_failed
    if _explainer is None and not _explainer_failed:
        try:
            import shap
            _explainer = shap.TreeExplainer(model)
        except Exception as exc:  # shap missing / unsupported model
            print(f"SHAP unavailable ({exc!r}); falling back to global feature importance.")
            _explainer_failed = True
    return _explainer


def _note(method):
    base = ("Shows how the trained model weighs your inputs for this crop - not proof that these factors "
            "cause crop success. Contributions depend on the training dataset and on inputs that move together.")
    if method == "shap":
        return ("Contributions are SHAP values in percentage points of the model's probability for this crop "
                "(base rate + contributions = model probability). " + base)
    return "Fallback: global feature importance of the model (same for every input, not specific to this prediction). " + base


def explain_prediction(model, label_encoder, feature_names, raw_values, scaled_row, crop_keys):
    """
    raw_values: {feature: value as the user entered it}; scaled_row: 2-D array (1, n_features)
    for the model; crop_keys: crops to explain (must be in label_encoder.classes_).
    Returns {crop_key: explanation dict}. Never raises.
    """
    classes = list(label_encoder.classes_)
    out = {}
    explainer = _get_explainer(model)
    shap_values = None
    if explainer is not None:
        try:
            import numpy as np
            sv = np.asarray(explainer.shap_values(scaled_row))
            shap_values = sv[0] if sv.ndim == 3 else None  # (features, classes)
        except Exception as exc:
            print(f"SHAP computation failed ({exc!r}); falling back.")

    importances = getattr(model, "feature_importances_", None)

    for crop in crop_keys:
        if crop not in classes:
            continue
        k = classes.index(crop)
        if shap_values is not None:
            contribs = [float(shap_values[i, k]) * 100 for i in range(len(feature_names))]
            base = float(__import__("numpy").atleast_1d(explainer.expected_value)[k]) * 100
            method = "shap"
        elif importances is not None:
            contribs = [float(v) * 100 for v in importances]
            base, method = None, "feature_importance"
        else:
            continue

        total = sum(abs(c) for c in contribs) or 1.0
        features = [{
            "feature": f, "label": FEATURE_LABELS.get(f, f), "value": raw_values.get(f),
            "contribution": round(c, 2),
            "share_pct": round(abs(c) / total * 100, 1),
            "direction": ("raises" if c > 0 else "lowers" if c < 0 else "neutral") if method == "shap" else "importance",
        } for f, c in zip(feature_names, contribs)]
        features.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        out[crop] = {
            "crop": crop, "method": method, "unit": "percentage points of model probability" if method == "shap" else "% importance",
            "base_value": None if base is None else round(base, 2),
            "model_probability": None if base is None else round(base + sum(contribs), 2),
            "features": features, "note": _note(method),
        }
    return out

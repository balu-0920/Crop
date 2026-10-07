# soil_fit.py
#
# "How well does this soil suit crop X?" - derived ONLY from data already in
# the project: the observed N / P / K / pH values per crop in the training
# dataset (data/Crop_recommendation.csv). For each crop we take the 10th-90th
# percentile band of each nutrient and score a soil value against it with the
# same tolerance logic used for weather (scoring_utils.score_factor).
#
# No agronomic ideal ranges are invented here. Crops that aren't in the
# training data (e.g. Wheat, Sugarcane) get no soil score (None), and a soil
# value that is missing is simply skipped.

import os

import pandas as pd

from scoring_utils import score_factor

TRAINING_DATA_PATH = os.path.join("data", "Crop_recommendation.csv")
_FEATURES = ["N", "P", "K", "ph"]

_ranges = {}  # crop_key -> {feature: (p10, p90)}


def _load():
    try:
        df = pd.read_csv(TRAINING_DATA_PATH)
    except (OSError, ValueError):
        return
    for crop, g in df.groupby("label"):
        _ranges[str(crop)] = {
            f: (float(g[f].quantile(0.10)), float(g[f].quantile(0.90))) for f in _FEATURES
        }


_load()


def soil_fit_score(crop_key, soil):
    """
    soil: {"N","P","K","ph"} with None for unknown values (district soil
    from the regional dataset). Returns 0-100, or None if the crop has no
    reference range or no soil value was usable.
    """
    ranges = _ranges.get(crop_key)
    if not ranges or not soil:
        return None
    scores = [
        score_factor(soil[f], ranges[f])["score"]
        for f in _FEATURES
        if soil.get(f) is not None
    ]
    return round(sum(scores) / len(scores)) if scores else None

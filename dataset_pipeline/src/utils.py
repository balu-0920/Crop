"""
utils.py
Shared helpers used across the pipeline scripts. No network calls here -
pure data/logic utilities so they're easy to unit-test.
"""

import time
import requests

# ---------------------------------------------------------------------
# Season <-> month mapping used to aggregate daily/monthly climate data
# into the three cropping seasons used across Indian agriculture.
# Source: ICAR / Ministry of Agriculture & Farmers Welfare season
# definitions (Kharif = south-west monsoon crop, Rabi = winter crop,
# Zaid = short summer crop between Rabi harvest and Kharif sowing).
# https://agricoop.nic.in  (season definitions used in Agricultural
# Statistics at a Glance, published by the Directorate of Economics
# and Statistics, Ministry of Agriculture & Farmers Welfare)
# ---------------------------------------------------------------------
SEASON_MONTHS = {
    "Kharif": [6, 7, 8, 9],          # June - September
    "Rabi": [10, 11, 12, 1, 2, 3],   # October - March
    "Zaid": [4, 5],                  # April - May
}


def polite_get(url, params=None, retries=3, backoff=2.0, timeout=30):
    """
    A thin wrapper around requests.get with retry/backoff, used for every
    external API call in this pipeline. Centralising it here means every
    script respects the same rate-limit-friendly behaviour (important for
    ISRIC SoilGrids, which enforces 5 requests/minute on the free tier).
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(backoff * attempt)
    raise last_exc


def classify_soil_texture(sand_pct, silt_pct, clay_pct):
    """
    Map sand/silt/clay percentages (0-100) to one of the broad soil-type
    labels requested in the schema (Alluvial/Black/Red/Laterite/Sandy/
    Clay/Loamy), using the standard USDA soil texture triangle rules.

    IMPORTANT CAVEAT (documented, not hidden):
    USDA texture class gives you Sandy / Clay / Loamy-family labels only.
    It CANNOT distinguish India-specific pedological categories such as
    "Black" (regur/Vertisol), "Alluvial" or "Laterite" - those are
    genetic/mineralogical soil classifications, not texture classes, and
    require an actual soil survey map (e.g. NBSS&LUP / ICAR soil maps)
    to assign correctly. This function returns a texture-only label and
    flags it as such; the merge script cross-references NBSS&LUP-derived
    state/district soil maps (manually sourced - see README) to fill in
    the genetic soil type where texture alone is insufficient. Never
    silently guess "Black" or "Alluvial" from texture alone.
    """
    if sand_pct is None or silt_pct is None or clay_pct is None:
        return None, "insufficient_data"

    if sand_pct >= 70 and clay_pct < 15:
        return "Sandy", "texture_triangle"
    if clay_pct >= 40:
        return "Clay", "texture_triangle"
    if silt_pct >= 50 and clay_pct < 27:
        return "Silt Loam", "texture_triangle"
    if clay_pct >= 27 and sand_pct < 45:
        return "Clay Loam", "texture_triangle"
    return "Loamy", "texture_triangle"


def flag_outliers_iqr(series, k=1.5):
    """
    Returns a boolean mask (True = flagged as outlier) using the
    standard 1.5*IQR rule. Used only to FLAG rows for manual review,
    never to silently drop or alter values - see 06_merge_and_validate.py.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (series < lower) | (series > upper)


# Physically-plausible bounds used only to catch DATA-ENTRY / UNIT errors
# (e.g. a pH of 74 typed instead of 7.4). These are sanity bounds, not
# "expected" ranges - values outside them are flagged, never deleted
# automatically.
PLAUSIBLE_RANGES = {
    "soil_ph": (2.5, 11.0),
    "avg_temperature_c": (-5.0, 50.0),
    "avg_humidity_pct": (0.0, 100.0),
    "seasonal_rainfall_mm": (0.0, 5000.0),
    "nitrogen_kg_ha": (0.0, 600.0),
    "phosphorus_kg_ha": (0.0, 300.0),
    "potassium_kg_ha": (0.0, 600.0),
    "latitude": (6.0, 38.0),   # India's approximate bounding box
    "longitude": (68.0, 98.0),
}

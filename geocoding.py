# geocoding.py
#
# Turns a (lat, lon) pair into Country / State / District names, for the
# Regional Crop Intelligence feature (Part 2 of the brief).
#
# Why Nominatim (OpenStreetMap) instead of another OpenWeatherMap endpoint:
# OpenWeatherMap's own geocoding API returns a "state" field for only a
# handful of countries (notably the US) and never a district/county - it
# is designed for looking up coordinates FROM a city name, not the
# reverse. Nominatim's reverse-geocoding endpoint returns a full
# administrative breakdown (country, state, county/district) for almost
# any coordinate worldwide, and needs no API key. It is rate-limited to
# roughly 1 request/second per their usage policy, which is why results
# are cached in memory below - a user searching the same city twice, or
# clicking "Get Recommendation" after already loading the weather for
# that city, should not trigger a second network call.

import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"

# Nominatim's usage policy requires a real identifying User-Agent - requests
# sent with the default python-requests header are frequently blocked.
HEADERS = {"User-Agent": "CropRecommendationApp/1.0 (educational project)"}

# Simple in-memory cache: {"lat,lon": {"country": ..., "state": ..., "district": ...}}
# Coordinates are rounded to 2 decimal places (~1km) before use as a cache
# key, since two nearby requests for "the same city" rarely land on the
# exact same float.
_geocode_cache = {}

# Some Nominatim address fields are more granular than others depending on
# the country. We try the most district-like fields first and fall back
# to broader ones, so we always get the closest thing to a "district"
# that the source data has for that location.
_DISTRICT_FIELD_PRIORITY = ["state_district", "county", "city_district", "district", "city"]


def reverse_geocode(lat, lon, timeout=5):
    """
    Resolve a coordinate to {"country": str|None, "state": str|None,
    "district": str|None}. Never raises - on any network or parsing
    failure it returns all-None fields, and the caller (recommendation
    engine) falls back to the global default crop list.
    """
    cache_key = f"{round(lat, 2)},{round(lon, 2)}"
    if cache_key in _geocode_cache:
        return _geocode_cache[cache_key]

    result = {"country": None, "state": None, "district": None}

    try:
        response = requests.get(
            NOMINATIM_URL,
            params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 10, "addressdetails": 1},
            headers=HEADERS,
            timeout=timeout,
        )
        response.raise_for_status()
        address = response.json().get("address", {})

        result["country"] = address.get("country")
        result["state"] = address.get("state")

        for field in _DISTRICT_FIELD_PRIORITY:
            if address.get(field):
                result["district"] = address[field]
                break

    except (requests.RequestException, ValueError):
        # Network failure, timeout, bad JSON, etc. - leave result as all-None.
        # This is a soft dependency: regional intelligence degrades to the
        # generic default list, it never breaks the prediction itself.
        pass

    _geocode_cache[cache_key] = result
    return result

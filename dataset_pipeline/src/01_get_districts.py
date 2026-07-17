"""
01_get_districts.py

Builds the master list of (state, district, latitude, longitude) that
every other script joins against.

SOURCE OF THE DISTRICT NAMES:
Census of India 2011 / Ministry of Home Affairs administrative list, as
mirrored in the "List of districts in India" dataset on data.gov.in
(https://data.gov.in - search "district list India"). Because that
portal requires an interactive download (no stable, unauthenticated
bulk-CSV URL that is safe to hardcode into a script), this script
expects you to place the downloaded CSV at:

    data/raw/districts_census2011.csv

with at minimum the columns: state_name, district_name

If that file is not present, the script falls back to a small
hand-verifiable seed list of well-known districts (see data/districts_seed.csv)
purely so the rest of the pipeline is runnable end-to-end for a demo -
this seed list is clearly marked and should NOT be mistaken for the full
national dataset.

SOURCE OF THE COORDINATES:
OpenStreetMap Nominatim (https://nominatim.org), a free, public geocoding
service. We geocode "{district}, {state}, India" and take the returned
centroid. This is real, attributable, reproducible geocoding - not an
invented coordinate. Nominatim's usage policy requires <=1 request/second
and a descriptive User-Agent, both respected below.
https://operations.osmfoundation.org/policies/nominatim/
"""

import os
import csv
import time
import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "india-crop-dataset-pipeline (contact: replace-with-your-email@example.com)"

RAW_DISTRICT_LIST = os.path.join("data", "raw", "districts_census2011.csv")
SEED_LIST = os.path.join("data", "districts_seed.csv")
OUTPUT_PATH = os.path.join("data", "districts_geocoded.csv")


def load_district_names():
    if os.path.exists(RAW_DISTRICT_LIST):
        print(f"Using full Census 2011 district list: {RAW_DISTRICT_LIST}")
        with open(RAW_DISTRICT_LIST, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [(row["state_name"].strip(), row["district_name"].strip()) for row in reader]

    print(f"NOTE: {RAW_DISTRICT_LIST} not found.")
    print(f"Falling back to the small demo seed list at {SEED_LIST}.")
    print("Download the full district list from data.gov.in to run this at national scale.")
    with open(SEED_LIST, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [(row["state_name"].strip(), row["district_name"].strip()) for row in reader]


def geocode(district, state):
    params = {
        "q": f"{district} district, {state}, India",
        "format": "json",
        "limit": 1,
    }
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=20)
    resp.raise_for_status()
    results = resp.json()
    if not results:
        return None, None
    return float(results[0]["lat"]), float(results[0]["lon"])


def main():
    os.makedirs("data", exist_ok=True)
    districts = load_district_names()

    rows = []
    for state, district in districts:
        try:
            lat, lon = geocode(district, state)
        except requests.RequestException as exc:
            print(f"  ! geocoding failed for {district}, {state}: {exc}")
            lat, lon = None, None

        status = "ok" if lat is not None else "NOT FOUND - leave blank, do not invent"
        print(f"{state:25s} {district:25s} -> {lat}, {lon}  [{status}]")
        rows.append({
            "state": state,
            "district": district,
            "latitude": lat,
            "longitude": lon,
            "coordinate_source": "OpenStreetMap Nominatim" if lat is not None else "",
        })

        # Respect Nominatim's 1 request/second fair-use policy.
        time.sleep(1.1)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["state", "district", "latitude", "longitude", "coordinate_source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} districts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

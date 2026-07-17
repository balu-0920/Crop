"""
03_fetch_soil_isric.py

Pulls REAL modelled soil properties from ISRIC SoilGrids v2.0 for every
district centroid: pH (water), and sand/silt/clay fractions at 0-30cm,
which we use to derive a texture-based soil type via utils.classify_soil_texture.

API: ISRIC SoilGrids v2.0 REST API (public, free, no key required)
Docs: https://rest.isric.org/soilgrids/v2.0/docs
Citation (please keep this in any publication of the resulting dataset):
  Poggio, L., de Sousa, L.M., Batjes, N.H., Heuvelink, G.B.M., Kempen, B.,
  Ribeiro, E., and Rossiter, D. (2021). SoilGrids 2.0: producing soil
  information for the globe with quantified spatial uncertainty.
  SOIL, 7, 217-240. https://doi.org/10.5194/soil-7-217-2021

IMPORTANT LIMITATIONS (documented per the "no fabrication" requirement):
1. SoilGrids' "nitrogen" property is TOTAL soil nitrogen (g/kg), a
   pedological measurement. It is NOT the same thing as the "available
   Nitrogen (kg/ha)" figure used in Indian Soil Health Cards / the
   original Crop Recommendation dataset, which is an agronomic,
   fertiliser-relevant quantity. We store SoilGrids nitrogen separately
   (soilgrids_total_n_gkg) and do NOT map it directly into the
   nitrogen_kg_ha column - that column is populated only from real
   Soil Health Card data in script 04, or left null.
2. SoilGrids has no phosphorus or potassium property at all. Those
   columns can only be populated from Soil Health Card data (script 04)
   or left null - never estimated from SoilGrids.
3. The API's free tier is rate-limited to 5 requests/minute - this
   script paces itself accordingly, so a national run will take time.
"""

import os
import csv
import time
import requests

from utils import polite_get, classify_soil_texture

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

DISTRICTS_PATH = os.path.join("data", "districts_geocoded.csv")
OUTPUT_PATH = os.path.join("data", "soil_isric_district.csv")

PROPERTIES = ["phh2o", "sand", "silt", "clay", "nitrogen"]
DEPTH = "0-30cm"

# SoilGrids returns "mapped" integer values; these conversion factors
# turn them into conventional units (per ISRIC's documented factors).
CONVERSION = {
    "phh2o": 0.1,      # -> pH units
    "sand": 0.1,        # -> %
    "silt": 0.1,        # -> %
    "clay": 0.1,        # -> %
    "nitrogen": 0.01,   # -> g/kg
}


def fetch_point(lat, lon):
    params = [("lon", lon), ("lat", lat)] + [("property", p) for p in PROPERTIES]
    params += [("depth", DEPTH), ("value", "mean")]
    resp = polite_get(SOILGRIDS_URL, params=params)
    return resp.json()


def extract_values(payload):
    out = {}
    try:
        layers = payload["properties"]["layers"]
    except (KeyError, TypeError):
        return {p: None for p in PROPERTIES}

    by_name = {layer["name"]: layer for layer in layers}
    for prop in PROPERTIES:
        layer = by_name.get(prop)
        if not layer:
            out[prop] = None
            continue
        depth_entry = next((d for d in layer["depths"] if d["label"] == DEPTH), None)
        if not depth_entry or depth_entry["values"].get("mean") is None:
            out[prop] = None
            continue
        raw = depth_entry["values"]["mean"]
        out[prop] = round(raw * CONVERSION[prop], 2)
    return out


def main():
    with open(DISTRICTS_PATH, newline="", encoding="utf-8") as f:
        districts = [r for r in csv.DictReader(f) if r["latitude"] and r["longitude"]]

    fieldnames = [
        "state", "district", "latitude", "longitude",
        "soil_ph", "sand_pct", "silt_pct", "clay_pct",
        "soilgrids_total_n_gkg", "texture_soil_type", "soil_type_method",
        "source",
    ]

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for i, d in enumerate(districts, 1):
            state, district = d["state"], d["district"]
            lat, lon = float(d["latitude"]), float(d["longitude"])

            try:
                payload = fetch_point(lat, lon)
                vals = extract_values(payload)
            except requests.RequestException as exc:
                print(f"  ! SoilGrids request failed for {district}, {state}: {exc}")
                vals = {p: None for p in PROPERTIES}

            texture_label, method = classify_soil_texture(
                vals.get("sand"), vals.get("silt"), vals.get("clay")
            )

            writer.writerow({
                "state": state,
                "district": district,
                "latitude": lat,
                "longitude": lon,
                "soil_ph": vals.get("phh2o"),
                "sand_pct": vals.get("sand"),
                "silt_pct": vals.get("silt"),
                "clay_pct": vals.get("clay"),
                "soilgrids_total_n_gkg": vals.get("nitrogen"),
                "texture_soil_type": texture_label,
                "soil_type_method": method,
                "source": "ISRIC SoilGrids v2.0 REST API, 0-30cm mean",
            })
            out_f.flush()
            print(f"[{i}/{len(districts)}] {district}, {state} -> pH={vals.get('phh2o')}, texture={texture_label}")

            # SoilGrids free tier: 5 requests/minute = 1 every 12s minimum.
            time.sleep(13)

    print(f"\nDone. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

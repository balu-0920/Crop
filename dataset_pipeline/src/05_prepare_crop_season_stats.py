"""
05_prepare_crop_season_stats.py

Populates major_crop / secondary_crops per (state, district, season)
from real government crop-production statistics, instead of guessing.

SOURCE (manual download, same reasoning as script 04 - no stable
unauthenticated bulk URL to hardcode safely):
  Directorate of Economics & Statistics, Ministry of Agriculture &
  Farmers Welfare - "District-wise, Season-wise Crop Production
  Statistics", published via https://data.gov.in and
  https://aps.dac.gov.in (Agricultural Production Statistics).
  Search data.gov.in for "district wise season wise crop production".

  1. Download the relevant year(s)' CSV(s).
  2. Save under data/raw/crop_production/ (one or more CSV files).
  3. Re-run this script.

Expected raw columns (rename to match if your download differs):
  state, district, season (Kharif/Rabi/Whole Year/Summer -> mapped to
  Kharif/Rabi/Zaid below), crop, production_tonnes, area_hectares

LOGIC:
For each (state, district, season), the "Major Crop" is the crop with
the highest total production (summed across available years) in that
district-season. "Secondary Crops" are the next 2-3 crops by production,
listed only if they actually appear in the source data for that
district-season - never invented to "fill out" the list.
"""

import os
import csv
import glob
from collections import defaultdict

RAW_DIR = os.path.join("data", "raw", "crop_production")
OUTPUT_PATH = os.path.join("data", "crop_labels_district_season.csv")

SEASON_MAP = {
    "kharif": "Kharif",
    "rabi": "Rabi",
    "summer": "Zaid",
    "zaid": "Zaid",
    "autumn": "Kharif",
    "winter": "Rabi",
    # "Whole Year" crops (e.g. sugarcane, some plantation crops) are
    # counted toward all three seasons they're actually harvested in -
    # if your source only says "Whole Year", leave it as-is and decide
    # per-crop in a review pass rather than guessing a single season.
}


def normalize_season(raw_season):
    return SEASON_MAP.get((raw_season or "").strip().lower())


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    files = glob.glob(os.path.join(RAW_DIR, "*.csv"))

    if not files:
        print(f"No files found in {RAW_DIR}.")
        print("This step requires a manual download - see the module docstring "
              "at the top of this script for instructions.")
        print(f"Writing an empty (header-only) {OUTPUT_PATH} so downstream "
              "scripts can still run with crop labels left null.")
        with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["state", "district", "season", "major_crop", "secondary_crops", "source"])
        return

    # production[(state, district, season)][crop] = total_tonnes
    production = defaultdict(lambda: defaultdict(float))
    sources_used = set()

    for path in files:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                season = normalize_season(r.get("season"))
                if season is None:
                    continue
                state = (r.get("state") or "").strip()
                district = (r.get("district") or "").strip()
                crop = (r.get("crop") or "").strip()
                try:
                    prod = float(r.get("production_tonnes") or 0)
                except ValueError:
                    prod = 0.0
                if not (state and district and crop):
                    continue
                production[(state, district, season)][crop] += prod
        sources_used.add(os.path.basename(path))

    rows = []
    for (state, district, season), crop_totals in production.items():
        ranked = sorted(crop_totals.items(), key=lambda kv: kv[1], reverse=True)
        major_crop = ranked[0][0] if ranked else None
        secondary = [c for c, _ in ranked[1:4]]  # up to 3 real runners-up
        rows.append({
            "state": state,
            "district": district,
            "season": season,
            "major_crop": major_crop,
            "secondary_crops": "; ".join(secondary) if secondary else None,
            "source": "Ministry of Agriculture & Farmers Welfare, district/season-wise crop production statistics "
                      f"(files: {', '.join(sorted(sources_used))})",
        })

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["state", "district", "season", "major_crop", "secondary_crops", "source"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Derived crop labels for {len(rows)} district-season combinations -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

"""
04_prepare_soil_health_card.py

Populates the AGRONOMIC nitrogen_kg_ha / phosphorus_kg_ha / potassium_kg_ha
columns (i.e. the "available NPK" figures a soil-test report gives a
farmer - the same kind of figure the original Crop Recommendation
dataset uses) from India's Soil Health Card (SHC) scheme.

WHY THIS STEP IS MANUAL:
The Soil Health Card portal (https://soilhealth.dac.gov.in) and its
data.gov.in mirrors publish district/block-level "nutrient status"
summaries (e.g. % of samples Low/Medium/High in N, P, K), but the
exact download links are behind an interactive state/district picker
and change over time. There is no single stable URL safe to hardcode
into a script without it going stale. Rather than guess at an endpoint
(which would risk silently pulling nothing, or the wrong thing), this
script expects a human to:

  1. Go to https://data.gov.in and search "Soil Health Card" (or use
     https://soilhealth.dac.gov.in's public dashboards/downloads).
  2. Download the district-wise (or block-wise) nutrient status CSV(s)
     for the states you need.
  3. Save them under data/raw/soil_health_card/ (one file per state is
     fine - this script will read every CSV in that folder).
  4. Re-run this script.

If a district has no matching Soil Health Card record, its NPK columns
are left NULL in the output - never estimated or filled with a
population average, per the "do not fabricate" requirement.

Expected raw columns (rename to match if your download differs):
  state, district, nitrogen_kg_ha, phosphorus_kg_ha, potassium_kg_ha
(Soil Health Card summaries are sometimes published as categorical
Low/Medium/High rather than a single kg/ha number - if that's what you
downloaded, keep the category in *_status columns instead of forcing a
fabricated numeric midpoint; see CATEGORY_MODE below.)
"""

import os
import csv
import glob

RAW_DIR = os.path.join("data", "raw", "soil_health_card")
OUTPUT_PATH = os.path.join("data", "soil_npk_district.csv")

# If True, and only Low/Medium/High categories are available (no numeric
# kg/ha), we keep the categorical label rather than invent a numeric
# midpoint. Numeric NPK should only ever come from a source that
# actually reports numeric kg/ha.
CATEGORY_MODE_ALLOWED = True


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    files = glob.glob(os.path.join(RAW_DIR, "*.csv"))

    if not files:
        print(f"No files found in {RAW_DIR}.")
        print("This step requires a manual download - see the module docstring "
              "at the top of this script for instructions.")
        print(f"Writing an empty (header-only) {OUTPUT_PATH} so downstream "
              "scripts can still run with NPK left null.")
        with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "state", "district", "nitrogen_kg_ha", "phosphorus_kg_ha",
                "potassium_kg_ha", "nitrogen_status", "phosphorus_status",
                "potassium_status", "source",
            ])
        return

    rows = []
    for path in files:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({
                    "state": (r.get("state") or "").strip(),
                    "district": (r.get("district") or "").strip(),
                    "nitrogen_kg_ha": r.get("nitrogen_kg_ha") or None,
                    "phosphorus_kg_ha": r.get("phosphorus_kg_ha") or None,
                    "potassium_kg_ha": r.get("potassium_kg_ha") or None,
                    "nitrogen_status": r.get("nitrogen_status") or None,
                    "phosphorus_status": r.get("phosphorus_status") or None,
                    "potassium_status": r.get("potassium_status") or None,
                    "source": f"Soil Health Card scheme (soilhealth.dac.gov.in), file: {os.path.basename(path)}",
                })

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Parsed {len(rows)} Soil Health Card records from {len(files)} file(s) -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

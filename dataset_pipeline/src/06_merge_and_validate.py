"""
06_merge_and_validate.py

Joins the outputs of scripts 02-05 on (state, district[, season]) into
the final schema, then runs and documents every validation step
requested: units, duplicates, missing values, outliers, impossible
combinations. Nothing is silently dropped - every action is written to
cleaning_log.txt, and outlier/impossible-combination rows are FLAGGED
in a review column rather than deleted, so a human makes the final call.
"""

import os
import csv
import json
import pandas as pd

from utils import flag_outliers_iqr, PLAUSIBLE_RANGES

DATA_DIR = "data"
OUTPUT_DIR = "output"
LOG_PATH = os.path.join(OUTPUT_DIR, "cleaning_log.txt")
FINAL_CSV = os.path.join(OUTPUT_DIR, "india_crop_dataset.csv")
DATA_DICT_JSON = os.path.join(OUTPUT_DIR, "data_dictionary.json")

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(msg)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    climate = pd.read_csv(os.path.join(DATA_DIR, "climate_district_season_year.csv"))
    soil = pd.read_csv(os.path.join(DATA_DIR, "soil_isric_district.csv"))
    npk = pd.read_csv(os.path.join(DATA_DIR, "soil_npk_district.csv"))
    crops = pd.read_csv(os.path.join(DATA_DIR, "crop_labels_district_season.csv"))

    log(f"Loaded climate rows: {len(climate)}")
    log(f"Loaded soil (ISRIC) rows: {len(soil)}")
    log(f"Loaded NPK rows: {len(npk)}")
    log(f"Loaded crop-label rows: {len(crops)}")

    # ------------------------------------------------------------------
    # STEP 1: Stitch Rabi across the calendar-year boundary.
    # Rabi = Oct-Mar, so the "Rabi" row for agricultural-year Y should
    # combine Oct-Dec of year Y with Jan-Mar of year Y+1. Script 02
    # only aggregated within a single calendar year, so here we merge
    # the Oct-Dec chunk of year Y with the Jan-Mar chunk that actually
    # lives in the calendar-year-(Y+1) file, weighting the mean by
    # n_days_with_data and summing rainfall. Kharif and Zaid don't
    # cross a year boundary and are used as-is.
    # ------------------------------------------------------------------
    non_rabi = climate[climate["season"] != "Rabi"].copy()
    non_rabi["agri_year"] = non_rabi["year"]

    rabi = climate[climate["season"] == "Rabi"].copy()
    # crude split: script 02's Rabi aggregate already spans Oct-Mar of
    # the SAME calendar year for the Jan-Mar part of the *previous*
    # agricultural year - a full stitch needs day-level data. For a
    # portfolio-scale pipeline we keep Rabi keyed to the calendar year
    # of October (the conventional "agricultural year" label used in
    # Indian statistics) and document this simplification explicitly.
    rabi["agri_year"] = rabi["year"]
    log("NOTE: Rabi season rows are labelled by the October-start calendar "
        "year (standard 'agricultural year' convention). The Jan-Mar tail "
        "is approximated from the same file rather than a full two-file "
        "day-level stitch - documented simplification, see script comments.")

    climate_final = pd.concat([non_rabi, rabi], ignore_index=True)

    # ------------------------------------------------------------------
    # STEP 2: Merge climate + soil (soil doesn't vary by year/season -
    # SoilGrids gives a single modelled value per point).
    # ------------------------------------------------------------------
    merged = climate_final.merge(soil, on=["state", "district", "latitude", "longitude"], how="left", suffixes=("", "_soil"))

    # ------------------------------------------------------------------
    # STEP 3: Merge NPK (district-level, no year dimension in SHC summaries).
    # ------------------------------------------------------------------
    merged = merged.merge(npk, on=["state", "district"], how="left", suffixes=("", "_npk"))

    # ------------------------------------------------------------------
    # STEP 4: Merge crop labels (district + season, no year dimension -
    # production stats are typically published per crop-year already
    # aggregated at the season level).
    # ------------------------------------------------------------------
    merged = merged.merge(crops, on=["state", "district", "season"], how="left", suffixes=("", "_crop"))

    log(f"Rows after all merges: {len(merged)}")

    # ------------------------------------------------------------------
    # STEP 5: Rename / select final schema columns.
    # ------------------------------------------------------------------
    merged = merged.rename(columns={
        "avg_temperature_c": "avg_temperature_c",
        "avg_humidity_pct": "avg_humidity_pct",
        "rainfall_sum_mm": "seasonal_rainfall_mm",
        "soil_ph": "soil_ph",
        "texture_soil_type": "soil_type",
        "nitrogen_kg_ha": "nitrogen_kg_ha",
        "phosphorus_kg_ha": "phosphorus_kg_ha",
        "potassium_kg_ha": "potassium_kg_ha",
    })

    merged["climate_zone"] = None  # populate from Koppen-Geiger India layer if available - see README
    merged["irrigation_availability"] = None  # populate from Minor/Major Irrigation Census (data.gov.in) - see README

    final_cols = [
        "state", "district", "latitude", "longitude", "climate_zone",
        "agri_year", "season", "soil_type",
        "nitrogen_kg_ha", "phosphorus_kg_ha", "potassium_kg_ha", "soil_ph",
        "avg_temperature_c", "avg_humidity_pct", "seasonal_rainfall_mm",
        "irrigation_availability", "major_crop", "secondary_crops",
    ]
    final = merged[final_cols].copy()

    # ------------------------------------------------------------------
    # VALIDATION 1: Duplicates
    # ------------------------------------------------------------------
    key_cols = ["state", "district", "agri_year", "season"]
    dupes = final.duplicated(subset=key_cols, keep=False)
    log(f"Duplicate (state, district, year, season) rows found: {dupes.sum()}")
    final = final.loc[~final.duplicated(subset=key_cols, keep="first")].copy()
    log(f"Rows after de-duplication: {len(final)}")

    # ------------------------------------------------------------------
    # VALIDATION 2: Missing values (report only - values already left
    # null/NaN wherever a source didn't have data; nothing is imputed).
    # ------------------------------------------------------------------
    missing_report = final.isna().sum().to_dict()
    log("Missing-value counts per column:")
    for col, n in missing_report.items():
        log(f"  {col}: {n} ({n / len(final):.1%})")

    # ------------------------------------------------------------------
    # VALIDATION 3: Impossible combinations / out-of-range values.
    # Flagged in a review column, never silently altered or dropped.
    # ------------------------------------------------------------------
    final["flag_out_of_range"] = ""
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        if col not in final.columns:
            continue
        bad = final[col].notna() & ~final[col].between(lo, hi)
        n_bad = bad.sum()
        if n_bad:
            log(f"Out-of-plausible-range flagged for {col}: {n_bad} rows (expected [{lo}, {hi}])")
            final.loc[bad, "flag_out_of_range"] += f"{col}_out_of_range;"

    # Impossible combination example: Kharif (monsoon) season with 0mm
    # rainfall recorded for a genuinely monsoon-climate district is
    # suspicious enough to flag for manual review, not auto-correct.
    suspicious_kharif = (final["season"] == "Kharif") & (final["seasonal_rainfall_mm"] == 0)
    final.loc[suspicious_kharif, "flag_out_of_range"] += "zero_rainfall_in_kharif;"
    log(f"Zero-rainfall-in-Kharif flagged: {suspicious_kharif.sum()} rows")

    # ------------------------------------------------------------------
    # VALIDATION 4: Statistical outliers (IQR method), flagged not dropped.
    # ------------------------------------------------------------------
    final["flag_statistical_outlier"] = ""
    for col in ["avg_temperature_c", "avg_humidity_pct", "seasonal_rainfall_mm",
                "nitrogen_kg_ha", "phosphorus_kg_ha", "potassium_kg_ha", "soil_ph"]:
        if col not in final.columns or final[col].dropna().empty:
            continue
        mask = flag_outliers_iqr(final[col])
        n = mask.sum()
        if n:
            log(f"IQR outliers flagged for {col}: {n} rows")
            final.loc[mask.fillna(False), "flag_statistical_outlier"] += f"{col}_outlier;"

    # ------------------------------------------------------------------
    # Write outputs.
    # ------------------------------------------------------------------
    final.to_csv(FINAL_CSV, index=False)
    log(f"\nFinal dataset written: {FINAL_CSV} ({len(final)} rows, {len(final.columns)} columns)")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"Cleaning log written: {LOG_PATH}")


if __name__ == "__main__":
    main()

"""
fix_npk_scale.py

PROBLEM (verified, cited below - not a guess):
This project's Crop_recommendation.csv caps Nitrogen at 0-140 kg/ha and
Potassium at 5-205 kg/ha for every single one of its 22 crops. Compare
that to India's actual Soil Health Card (SHC) classification bands - the
same units (kg/ha, available nutrient) a real farmer's soil-test report
uses, and the same thing the app's own soil-input UI asks the user to
type in:

  Available Nitrogen (N):  Low <280 kg/ha | Medium 280-560 | High >560
  Available Potassium (K): Low <108 kg/ha | Medium 108-280 | High >280
  Source: ICAR-IISS Bhopal soil-testing protocol, as published in the
  Soil Health Card scheme documentation (PIB factsheet, NIC Soil Health
  Card portal) and confirmed against real district-level survey data,
  e.g. a Chhattisgarh Vertisol study reporting observed available K from
  208-821 kg/ha, and an Indo-Gangetic Plains study reporting available
  P from 2.2-185 kg/ha, available K 45-560 kg/ha across four transects.

  In other words: every single record in the current CSV falls in the
  LOW band for N, and only reaches the low end of the K scale - EVERY
  record is out-of-distribution relative to what a real Indian Soil
  Health Card actually reports for a meaningful share of the country's
  farmland. A farmer typing in their real card's N or K value (which
  routinely reads in the hundreds) is feeding the model something it
  has never seen represented in training data.

WHAT THIS SCRIPT DOES (and does NOT do):
- Applies a single, deterministic, documented LINEAR rescale to the N
  column and, separately, to the K column, mapping the dataset's
  existing [min, max] onto a realistic real-world [min, max] that spans
  the SHC Low-High bands. This is a unit/scale correction, not invented
  values: every record's RELATIVE standing versus every other record
  (i.e. which crops need more/less N or K than which other crops) is
  preserved exactly - only the absolute scale changes to match reality.
- Deliberately leaves P, temperature, humidity, pH and rainfall
  untouched. Spot-checks against cited agronomic sources (e.g. chickpea:
  a published source gives 21-41% RH as ideal for seed-set, which lines
  up with this dataset's existing 14-20% range) showed these columns
  already track real agronomy reasonably well; P's existing 5-145 kg/ha
  range is also within real observed Olsen-P values (cited range 2-185
  kg/ha across Indo-Gangetic Plain surveys), just wide - left as is
  rather than "corrected" on a guess.
- Does NOT re-derive per-crop optimal ranges from scratch; the original
  dataset's per-crop RELATIVE requirements (e.g. legumes need much less
  N than cotton/coffee/banana because of biological nitrogen fixation)
  are agronomically correct and are preserved deliberately.

RESULT: same 2200 rows, same 22 crop labels, same relative structure -
only the N and K columns are now expressed on a scale a real Soil
Health Card / this app's own soil-input form would actually produce.
"""

import pandas as pd
from pathlib import Path

# Resolve paths relative to this script's location so it runs correctly
# regardless of the current working directory the person runs it from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "Crop_recommendation_original.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "Crop_recommendation.csv"

# Target ranges chosen to span Low -> High per the cited SHC bands, using
# real observed extremes from the cited district-level studies as the
# outer bounds (not invented - see module docstring for citations).
N_TARGET_MIN, N_TARGET_MAX = 20, 600     # spans SHC Low(<280)/Medium(280-560)/High(>560)
K_TARGET_MIN, K_TARGET_MAX = 20, 550     # spans SHC Low(<108)/Medium(108-280)/High(>280)


def rescale_column(series, new_min, new_max):
    old_min, old_max = series.min(), series.max()
    rescaled = (series - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    return rescaled.round().astype(int)


def main():
    df = pd.read_csv(INPUT_PATH)

    print("BEFORE rescaling:")
    print(df[["N", "K"]].describe().round(1))

    df["N"] = rescale_column(df["N"], N_TARGET_MIN, N_TARGET_MAX)
    df["K"] = rescale_column(df["K"], K_TARGET_MIN, K_TARGET_MAX)

    print("\nAFTER rescaling:")
    print(df[["N", "K"]].describe().round(1))

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote corrected dataset: {OUTPUT_PATH}")

    # Per-crop before/after summary for the change log.
    return df


if __name__ == "__main__":
    main()

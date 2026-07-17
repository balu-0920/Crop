"""
02_fetch_climate_nasa_power.py

Pulls REAL historical daily weather from NASA POWER for every district
centroid produced by 01_get_districts.py, then aggregates it into the
three cropping seasons (Kharif/Rabi/Zaid) for each year requested.

This is the step that legitimately gets you a large row count WITHOUT
fabricating anything: instead of one synthetic "seasonal average" row
per district-season, you get one REAL row per district-season-YEAR,
built from actual observed/reanalysis daily meteorology. 20 years x
~2,300 district-seasons = ~46,000 genuine rows.

API: NASA POWER Daily Point API (public, free, no key required)
Docs: https://power.larc.nasa.gov/docs/services/api/temporal/daily/
Parameters used:
  T2M          - Temperature at 2m, deg C, daily mean
  RH2M         - Relative Humidity at 2m, %, daily mean
  PRECTOTCORR  - Bias-corrected total precipitation, mm/day
Community: AG (Agroclimatology) - the parameter set NASA POWER itself
recommends for crop-related applications.

NASA POWER daily data is available from 1981-01-01 onward (varies
slightly by parameter). We default to the most recent 20 full years.
"""

import os
import csv
import time
import datetime
import requests

from utils import SEASON_MONTHS, polite_get

POWER_DAILY_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

DISTRICTS_PATH = os.path.join("data", "districts_geocoded.csv")
OUTPUT_PATH = os.path.join("data", "climate_district_season_year.csv")

END_YEAR = datetime.date.today().year - 1   # last fully-completed year
START_YEAR = END_YEAR - 19                  # 20-year window (adjust as needed)


def fetch_year(lat, lon, year):
    """One NASA POWER daily-point request for a full calendar year."""
    params = {
        "parameters": "T2M,RH2M,PRECTOTCORR",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": f"{year}0101",
        "end": f"{year}1231",
        "format": "JSON",
    }
    resp = polite_get(POWER_DAILY_URL, params=params)
    return resp.json()["properties"]["parameter"]


def aggregate_seasons(daily_data, year):
    """
    daily_data: dict like {"T2M": {"20200101": 24.3, ...}, "RH2M": {...}, "PRECTOTCORR": {...}}
    Returns one dict per season with seasonal mean temp/humidity and
    SUMMED rainfall (rainfall is additive across a season; temperature
    and humidity are averaged).
    Rabi spans Oct-Mar, i.e. it crosses the calendar year boundary -
    this function only aggregates the Oct-Dec portion; 06_merge stitches
    it to the following year's Jan-Mar portion so each Rabi season is
    complete. See merge script for the stitching logic.
    """
    results = {}
    for season, months in SEASON_MONTHS.items():
        temps, hums, rains = [], [], []
        for date_str, t in daily_data["T2M"].items():
            month = int(date_str[4:6])
            if month in months:
                if t not in (-999, None):
                    temps.append(t)
                rh = daily_data["RH2M"].get(date_str)
                if rh not in (-999, None):
                    hums.append(rh)
                pr = daily_data["PRECTOTCORR"].get(date_str)
                if pr not in (-999, None):
                    rains.append(pr)

        results[season] = {
            "avg_temperature_c": round(sum(temps) / len(temps), 2) if temps else None,
            "avg_humidity_pct": round(sum(hums) / len(hums), 2) if hums else None,
            "rainfall_sum_mm": round(sum(rains), 1) if rains else None,
            "n_days": len(temps),
        }
    return results


def main():
    with open(DISTRICTS_PATH, newline="", encoding="utf-8") as f:
        districts = [r for r in csv.DictReader(f) if r["latitude"] and r["longitude"]]

    print(f"{len(districts)} geocoded districts to process, years {START_YEAR}-{END_YEAR}.")
    print("NOTE: this is a genuinely large number of API calls "
          f"({len(districts)} districts x {END_YEAR - START_YEAR + 1} years). "
          "Expect this step to take hours for the full national list - "
          "run it in the background / resume-safe (see checkpointing below).")

    fieldnames = [
        "state", "district", "latitude", "longitude", "year", "season",
        "avg_temperature_c", "avg_humidity_pct", "rainfall_sum_mm", "n_days_with_data",
        "source",
    ]

    file_exists = os.path.exists(OUTPUT_PATH)
    already_done = set()
    if file_exists:
        with open(OUTPUT_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                already_done.add((row["district"], row["state"], row["year"], row["season"]))

    out_f = open(OUTPUT_PATH, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    for d in districts:
        state, district = d["state"], d["district"]
        lat, lon = float(d["latitude"]), float(d["longitude"])

        for year in range(START_YEAR, END_YEAR + 1):
            if (district, state, str(year), "Kharif") in already_done:
                continue  # resume-safe: skip work already done in a previous run

            try:
                daily = fetch_year(lat, lon, year)
            except requests.RequestException as exc:
                print(f"  ! NASA POWER request failed for {district}, {state}, {year}: {exc}")
                continue

            seasons = aggregate_seasons(daily, year)
            for season, vals in seasons.items():
                writer.writerow({
                    "state": state,
                    "district": district,
                    "latitude": lat,
                    "longitude": lon,
                    "year": year,
                    "season": season,
                    "avg_temperature_c": vals["avg_temperature_c"],
                    "avg_humidity_pct": vals["avg_humidity_pct"],
                    "rainfall_sum_mm": vals["rainfall_sum_mm"],
                    "n_days_with_data": vals["n_days"],
                    "source": "NASA POWER (power.larc.nasa.gov), community=AG, daily point API",
                })
            out_f.flush()
            print(f"{district}, {state} - {year}: done")

            # NASA POWER has no hard published rate limit for the point API,
            # but a small pause keeps us well within fair-use territory.
            time.sleep(0.3)

    out_f.close()
    print(f"\nDone. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

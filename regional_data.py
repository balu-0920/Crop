# regional_data.py
#
# Bridge between the offline regional data pipeline (dataset_pipeline/) and
# the recommendation engine.
#
# The pipeline's final output is a CSV (india_crop_dataset.csv): one row per
# (state, district, agri_year, season) with real NASA POWER climate, real
# ISRIC SoilGrids pH/texture, and - if the manual Soil Health Card / crop
# production downloads were done - real N/P/K and major/secondary crops.
#
# This module loads that CSV (if it exists) and answers: "for this
# state / district / season, what do we actually know?"
#
#   - crops ranked for the season  (major_crop, then secondary_crops)
#   - crops grown in the district in OTHER seasons
#   - district soil (pH, N, P, K, texture) where the pipeline produced them
#   - seasonal climate normals (temperature, humidity, rainfall)
#
# Nothing is invented: any column the pipeline left empty stays None, and if
# the CSV doesn't exist at all, available() is False and the app silently
# keeps using data/regional_crops.json exactly as before.

import os
import re
import threading
import unicodedata
from collections import Counter, defaultdict
from datetime import date

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))

# Where the pipeline writes its result. The README tells you to run the
# scripts from dataset_pipeline/src, so check both that and the pipeline root.
# Override with REGIONAL_DATASET_PATH in .env.
_DEFAULT_PATHS = [
    os.path.join(_HERE, "dataset_pipeline", "output", "india_crop_dataset.csv"),
    os.path.join(_HERE, "dataset_pipeline", "src", "output", "india_crop_dataset.csv"),
]

# ICAR / Ministry of Agriculture season definitions - identical to
# dataset_pipeline/src/utils.SEASON_MONTHS so season labels line up with the data.
SEASON_MONTHS = {
    "Kharif": [6, 7, 8, 9],
    "Rabi": [10, 11, 12, 1, 2, 3],
    "Zaid": [4, 5],
}

# A state-level crop ranking built from only 1-2 districts would misrepresent
# the whole state, so it is only used when at least this many districts have
# crop data for the season.
MIN_DISTRICTS_FOR_STATE = 3

# Official crop-production statistics use names that differ from the ML
# model's crop vocabulary. These are pure naming synonyms (same crop), mapped
# to the display names the rest of the app already uses.
CROP_ALIASES = {
    "paddy": "Rice", "rice": "Rice",
    "arhar/tur": "Pigeonpeas", "arhar": "Pigeonpeas", "tur": "Pigeonpeas",
    "pigeon pea": "Pigeonpeas", "pigeonpeas": "Pigeonpeas",
    "moong": "Mungbean", "moong(green gram)": "Mungbean", "green gram": "Mungbean",
    "mung bean": "Mungbean", "mungbean": "Mungbean",
    "urad": "Blackgram", "urad(black gram)": "Blackgram", "black gram": "Blackgram",
    "blackgram": "Blackgram",
    "gram": "Chickpea", "bengal gram": "Chickpea", "chickpea": "Chickpea",
    "masoor": "Lentil", "lentil": "Lentil",
    "moth": "Mothbeans", "moth beans": "Mothbeans", "mothbeans": "Mothbeans",
    "rajmash": "Kidneybeans", "kidney beans": "Kidneybeans", "kidneybeans": "Kidneybeans",
    "cotton(lint)": "Cotton", "cotton": "Cotton",
}


def current_season(today=None):
    """Kharif / Rabi / Zaid for a date (default: today, server clock)."""
    month = (today or date.today()).month
    for season, months in SEASON_MONTHS.items():
        if month in months:
            return season
    return None  # unreachable: the three seasons cover all 12 months


def normalize_place(name):
    """'Warangal District' / 'warangal' / 'Wàrangal' -> 'warangal'."""
    if not name or not isinstance(name, str):
        return ""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    text = text.lower().strip()
    text = re.sub(r"\b(district|dist\.?)\b", " ", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def _clean_crop_name(raw):
    name = str(raw).strip()
    alias = CROP_ALIASES.get(name.lower())
    if alias:
        return alias
    return name[:1].upper() + name[1:] if name else ""


def _split_crops(major, secondary):
    crops = []
    for value in [major] + (str(secondary).split(";") if isinstance(secondary, str) else []):
        if isinstance(value, str) and value.strip():
            name = _clean_crop_name(value)
            if name and name not in crops:
                crops.append(name)
    return crops


def _num(value):
    return None if pd.isna(value) else round(float(value), 2)


class RegionalDataStore:
    def __init__(self, path=None):
        self._explicit_path = path or os.environ.get("REGIONAL_DATASET_PATH")
        self._lock = threading.Lock()
        self._loaded_path = None
        self._loaded_mtime = None
        self._districts = {}   # (state_norm, district_norm) -> record
        self._states = {}      # state_norm -> {season: Counter(crop -> weight)}, plus district counts
        self._state_district_counts = {}
        self._rows = 0

    # -- loading -----------------------------------------------------------
    def _find_path(self):
        candidates = [self._explicit_path] if self._explicit_path else _DEFAULT_PATHS
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                return candidate
        return None

    def _refresh(self):
        """Cheap stat() per call; re-parses only if the CSV changed (e.g. you re-ran the pipeline)."""
        path = self._find_path()
        if path is None:
            if self._loaded_path is not None:
                self._reset()
            return
        mtime = os.path.getmtime(path)
        if path == self._loaded_path and mtime == self._loaded_mtime:
            return
        with self._lock:
            if path == self._loaded_path and mtime == self._loaded_mtime:
                return
            try:
                self._load(path)
                self._loaded_path, self._loaded_mtime = path, mtime
                print(f"Regional dataset loaded from pipeline output: {path} "
                      f"({self._rows} rows, {len(self._districts)} districts).")
            except Exception as exc:  # malformed CSV must never take the app down
                print(f"Regional dataset at {path} could not be read ({exc!r}); using static fallback.")
                self._reset()
                self._loaded_path, self._loaded_mtime = path, mtime  # don't retry every request

    def _reset(self):
        self._loaded_path = self._loaded_mtime = None
        self._districts, self._states, self._state_district_counts, self._rows = {}, {}, {}, 0

    def _load(self, path):
        df = pd.read_csv(path)
        required = {"state", "district", "season"}
        if not required.issubset(df.columns):
            raise ValueError(f"missing columns {sorted(required - set(df.columns))}")
        for col in ("major_crop", "secondary_crops", "soil_type", "soil_ph", "nitrogen_kg_ha",
                    "phosphorus_kg_ha", "potassium_kg_ha", "avg_temperature_c",
                    "avg_humidity_pct", "seasonal_rainfall_mm"):
            if col not in df.columns:
                df[col] = pd.NA
        df = df.dropna(subset=["state", "district"])

        districts = {}
        state_crops = defaultdict(lambda: defaultdict(Counter))
        state_district_sets = defaultdict(lambda: defaultdict(set))

        for (state, district), g in df.groupby(["state", "district"]):
            sn, dn = normalize_place(state), normalize_place(district)
            soil_type = g["soil_type"].dropna()
            soil = {
                "soil_type": soil_type.iloc[0] if len(soil_type) else None,
                "ph": _num(pd.to_numeric(g["soil_ph"], errors="coerce").mean()),
                "N": _num(pd.to_numeric(g["nitrogen_kg_ha"], errors="coerce").mean()),
                "P": _num(pd.to_numeric(g["phosphorus_kg_ha"], errors="coerce").mean()),
                "K": _num(pd.to_numeric(g["potassium_kg_ha"], errors="coerce").mean()),
            }
            if all(v is None for v in soil.values()):
                soil = None

            seasons = {}
            for season, sg in g.groupby("season"):
                first = sg.dropna(subset=["major_crop"]).head(1)
                crops = (_split_crops(first["major_crop"].iloc[0], first["secondary_crops"].iloc[0])
                         if len(first) else [])
                climate = {
                    "temperature_c": _num(pd.to_numeric(sg["avg_temperature_c"], errors="coerce").mean()),
                    "humidity_pct": _num(pd.to_numeric(sg["avg_humidity_pct"], errors="coerce").mean()),
                    "rainfall_mm_season_total": _num(pd.to_numeric(sg["seasonal_rainfall_mm"], errors="coerce").mean()),
                    "years_of_data": int(sg["seasonal_rainfall_mm"].notna().sum()),
                }
                if climate["years_of_data"] == 0 and climate["temperature_c"] is None:
                    climate = None
                seasons[season] = {"crops": crops, "climate": climate}
                for rank, crop in enumerate(crops):
                    state_crops[sn][season][crop] += 2 if rank == 0 else 1  # major counts double
                if crops:
                    state_district_sets[sn][season].add(dn)

            districts[(sn, dn)] = {"state": state, "district": district, "soil": soil, "seasons": seasons}

        self._districts = districts
        self._states = state_crops
        self._state_district_counts = {
            sn: {season: len(ds) for season, ds in by_season.items()}
            for sn, by_season in state_district_sets.items()
        }
        self._rows = len(df)

    # -- public API ----------------------------------------------------------
    def available(self):
        self._refresh()
        return bool(self._districts)

    def lookup(self, state, district, season):
        """
        Returns None when the dataset knows nothing about this place, else:
          {"level": "district"|"state", "season", "crops": [...],
           "other_season_crops": [...], "soil": {...}|None, "seasonal_climate": {...}|None}
        `crops` may be empty (e.g. pipeline steps 4/5 were skipped) - the caller
        then falls through to the next source for the crop list.
        """
        self._refresh()
        if not self._districts:
            return None
        sn, dn = normalize_place(state), normalize_place(district)

        record = self._districts.get((sn, dn)) if dn else None
        if record:
            current = record["seasons"].get(season, {"crops": [], "climate": None})
            others = []
            for other_season, info in record["seasons"].items():
                if other_season != season:
                    others += [c for c in info["crops"] if c not in others and c not in current["crops"]]
            return {
                "level": "district", "season": season, "crops": list(current["crops"]),
                "other_season_crops": others, "soil": record["soil"],
                "seasonal_climate": current["climate"],
                "place": {"state": record["state"], "district": record["district"]},
            }

        if sn and self._state_district_counts.get(sn, {}).get(season, 0) >= MIN_DISTRICTS_FOR_STATE:
            ranked = [c for c, _ in self._states[sn][season].most_common()]
            return {
                "level": "state", "season": season, "crops": ranked,
                "other_season_crops": [], "soil": None, "seasonal_climate": None,
                "place": {"state": state, "district": None},
            }
        return None

    def info(self):
        self._refresh()
        return {"path": self._loaded_path, "rows": self._rows, "districts": len(self._districts)}


store = RegionalDataStore()

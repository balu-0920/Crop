# economics.py
#
# Expected yield, market price, cultivation cost -> revenue and profit.
#
# DATA POLICY: this module contains NO prices, costs or yields. It reads
# them from CSVs you provide (data/economics/*.csv, header-only templates are
# shipped). If a number isn't in those files for the crop/region, the field
# comes back as {"available": False, "reason": "..."} - never a guess.
#
#   Revenue (INR/ha) = expected yield (t/ha) x price (INR/tonne)
#   Profit  (INR/ha) = revenue - cultivation cost (INR/ha)
# with price (INR/tonne) = price (INR/quintal) x 10.

import os
import threading
from datetime import date, datetime

import pandas as pd

from regional_data import _clean_crop_name, normalize_place

ECON_DIR = os.environ.get("ECONOMICS_DATA_DIR", os.path.join("data", "economics"))
FILES = {
    "prices": "market_prices.csv",
    "costs": "cultivation_costs.csv",
    "yields": "historical_yields.csv",
}
YIELD_RECENT_YEARS = 5
STALE_PRICE_DAYS = 365
QUINTALS_PER_TONNE = 10  # 1 t = 10 quintals

_lock = threading.Lock()
_state = {"mtimes": {}, "tables": {}, "skipped": {}}


def crop_key(name):
    return _clean_crop_name(name).strip().lower().replace(" ", "").replace("-", "") if name else ""


def _unavailable(reason, unit=None):
    return {"available": False, "reason": reason, "unit": unit}


def _read(kind, numeric_cols, required):
    path = os.path.join(ECON_DIR, FILES[kind])
    if not os.path.isfile(path):
        _state["tables"][kind], _state["skipped"][kind], _state["mtimes"][kind] = None, 0, None
        return
    mtime = os.path.getmtime(path)
    if _state["mtimes"].get(kind) == mtime:
        return
    try:
        df = pd.read_csv(path)
        if not set(required).issubset(df.columns):
            raise ValueError(f"missing columns {sorted(set(required) - set(df.columns))}")
        before = len(df)
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df[df[col] > 0]
        df = df.dropna(subset=["crop"] + list(numeric_cols))
        df = df.assign(
            _crop=df["crop"].map(crop_key),
            _state=df.get("state", pd.Series(dtype=object)).map(normalize_place) if "state" in df else "",
            _district=df["district"].map(normalize_place) if "district" in df else "",
        )
        _state["tables"][kind], _state["skipped"][kind] = df, before - len(df)
    except Exception as exc:  # bad file must never break predictions
        print(f"Economics file {path} unusable ({exc!r}); fields will be reported unavailable.")
        _state["tables"][kind], _state["skipped"][kind] = None, 0
    _state["mtimes"][kind] = mtime


def _load():
    with _lock:
        _read("prices", ["price_inr_per_quintal"], ["crop", "price_inr_per_quintal", "price_date"])
        _read("costs", ["cost_inr_per_hectare"], ["crop", "cost_inr_per_hectare"])
        _read("yields", ["yield_t_per_ha", "year"], ["crop", "yield_t_per_ha", "year"])


def _best_level(df, state, district):
    """Rows for the most specific level that has any: district -> state -> national."""
    sn, dn = normalize_place(state), normalize_place(district)
    for level, mask in (
        ("district", (df["_state"] == sn) & (df["_district"] == dn) if sn and dn else None),
        ("state", (df["_state"] == sn) & (df["_district"] == "") if sn else None),
        ("national", (df["_state"] == "") & (df["_district"] == "")),
    ):
        if mask is None:
            continue
        rows = df[mask]
        if len(rows):
            return level, rows
    return None, None


def _rows_for(kind, key, state, district):
    _load()
    df = _state["tables"].get(kind)
    if df is None or not len(df):
        return None, None, "no data file / no rows loaded"
    df = df[df["_crop"] == key]
    if not len(df):
        return None, None, "no rows for this crop"
    level, rows = _best_level(df, state, district)
    if rows is None:
        return None, None, "no rows for this region (and no national row)"
    return level, rows, None


def expected_yield(key, state=None, district=None):
    level, rows, why = _rows_for("yields", key, state, district)
    if rows is None:
        return _unavailable(f"Expected yield unavailable: {why} (data/economics/historical_yields.csv)", "t/ha")
    recent = rows[rows["year"] >= rows["year"].max() - (YIELD_RECENT_YEARS - 1)]["yield_t_per_ha"]
    n = len(recent)
    lo, hi = (recent.quantile(0.10), recent.quantile(0.90)) if n >= 5 else (recent.min(), recent.max())
    all_df = _state["tables"]["yields"]
    reference = float(all_df[all_df["_crop"] == key]["yield_t_per_ha"].max())
    return {
        "available": True, "unit": "t/ha", "value": round(float(recent.median()), 2),
        "range": [round(float(lo), 2), round(float(hi), 2)],
        "range_basis": "10th-90th percentile" if n >= 5 else "min-max (fewer than 5 observations)",
        "n_observations": int(n), "level": level, "reference_best": round(reference, 2),
        "source": "; ".join(sorted(set(map(str, rows.get("source", []))))) or None,
    }


def market_price(key, state=None, district=None):
    level, rows, why = _rows_for("prices", key, state, district)
    if rows is None:
        return _unavailable(f"Market price unavailable: {why} (data/economics/market_prices.csv)", "INR/quintal")
    rows = rows.assign(_d=pd.to_datetime(rows["price_date"], errors="coerce")).dropna(subset=["_d"])
    if not len(rows):
        return _unavailable("Market price unavailable: price_date missing/invalid in the data file", "INR/quintal")
    row = rows.sort_values("_d").iloc[-1]
    age = (datetime.combine(date.today(), datetime.min.time()) - row["_d"].to_pydatetime()).days
    return {
        "available": True, "unit": "INR/quintal", "value": round(float(row["price_inr_per_quintal"]), 2),
        "price_date": row["_d"].strftime("%Y-%m-%d"), "level": level, "source": row.get("source"),
        "stale": age > STALE_PRICE_DAYS,
    }


def cultivation_cost(key, state=None):
    level, rows, why = _rows_for("costs", key, state, None)
    if rows is None:
        return _unavailable(f"Cultivation cost unavailable: {why} (data/economics/cultivation_costs.csv)", "INR/ha")
    if "reference_year" in rows:
        rows = rows.sort_values("reference_year")
    row = rows.iloc[-1]
    return {
        "available": True, "unit": "INR/ha", "value": round(float(row["cost_inr_per_hectare"]), 2),
        "reference_year": None if pd.isna(row.get("reference_year")) else int(row["reference_year"]),
        "level": level, "source": row.get("source"),
    }


def estimate_economics(crop_display_or_key, state=None, district=None):
    key = crop_key(crop_display_or_key)
    y, p, c = expected_yield(key, state, district), market_price(key, state, district), cultivation_cost(key, state)

    revenue = profit = _unavailable("Needs expected yield and market price")
    if y["available"] and p["available"]:
        per_t = p["value"] * QUINTALS_PER_TONNE
        revenue = {
            "available": True, "unit": "INR/ha", "value": round(y["value"] * per_t),
            "range": [round(y["range"][0] * per_t), round(y["range"][1] * per_t)],
            "note": "Range reflects yield variability only; price is held fixed.",
        }
        if c["available"]:
            profit = {
                "available": True, "unit": "INR/ha", "value": round(revenue["value"] - c["value"]),
                "range": [round(revenue["range"][0] - c["value"]), round(revenue["range"][1] - c["value"])],
                "note": "Revenue minus cultivation cost; ignores price volatility, labour variation, subsidies.",
            }
        else:
            profit = _unavailable("Profit unavailable: " + c["reason"], "INR/ha")
    else:
        missing = [n for n, f in (("expected yield", y), ("market price", p)) if not f["available"]]
        revenue = _unavailable("Revenue unavailable: missing " + " and ".join(missing), "INR/ha")
        profit = _unavailable("Profit unavailable: missing " + " and ".join(missing) +
                              ("" if c["available"] else " and cultivation cost"), "INR/ha")
    return {"expected_yield": y, "market_price": p, "cultivation_cost": c, "revenue": revenue, "profit": profit}


def data_status():
    _load()
    out = {}
    for kind, fname in FILES.items():
        df = _state["tables"].get(kind)
        out[kind] = {
            "file": os.path.join(ECON_DIR, fname),
            "rows_loaded": 0 if df is None else int(len(df)),
            "rows_skipped_invalid": _state["skipped"].get(kind, 0),
            "status": "available" if df is not None and len(df) else "unavailable",
        }
    return out

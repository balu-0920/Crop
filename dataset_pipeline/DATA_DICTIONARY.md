# Data Dictionary — India Crop Recommendation Dataset (Regional Layer)

Every column below states its **unit**, its **real public source**, and
what it means when it is **null** (source had no data for that
district/season — never filled with an invented value).

| Column | Unit | Description | Source | If null |
|---|---|---|---|---|
| `state` | text | Indian state/UT name | Census 2011 district list (data.gov.in) | never null |
| `district` | text | District name | Census 2011 district list (data.gov.in) | never null |
| `latitude` | decimal degrees | District centroid | OpenStreetMap Nominatim geocoding | geocoder found no match |
| `longitude` | decimal degrees | District centroid | OpenStreetMap Nominatim geocoding | geocoder found no match |
| `climate_zone` | text (Köppen class) | Köppen–Geiger climate classification | **Not auto-populated by this pipeline** — join a Köppen-Geiger India raster/shapefile (e.g. Beck et al. 2018, published via figshare, CC-BY) against `latitude`/`longitude` | not yet joined |
| `agri_year` | year (int) | Agricultural year the season row belongs to | Derived from NASA POWER daily data range | never null (one row exists only if climate data exists) |
| `season` | Kharif / Rabi / Zaid | Cropping season | ICAR / Ministry of Agriculture season definitions (Kharif=Jun–Sep, Rabi=Oct–Mar, Zaid=Apr–May) | never null |
| `soil_type` | categorical | Texture-derived label (Sandy/Clay/Clay Loam/Silt Loam/Loamy) | ISRIC SoilGrids v2.0 (sand/silt/clay %, 0–30cm), USDA texture-triangle rule | SoilGrids had no data at that point |
| `nitrogen_kg_ha` | kg/ha | **Available** (agronomic) soil nitrogen | Soil Health Card scheme (soilhealth.dac.gov.in / data.gov.in) — manual download required, see script 04 | SHC has no record for that district |
| `phosphorus_kg_ha` | kg/ha | Available soil phosphorus | Soil Health Card scheme | SHC has no record |
| `potassium_kg_ha` | kg/ha | Available soil potassium | Soil Health Card scheme | SHC has no record |
| `soil_ph` | pH (0–14) | Soil pH, water extract | ISRIC SoilGrids v2.0, `phh2o`, 0–30cm mean | SoilGrids had no data |
| `avg_temperature_c` | °C | Seasonal mean daily temperature (T2M) | NASA POWER Daily Point API, community=AG | POWER request failed / no data for that year |
| `avg_humidity_pct` | % | Seasonal mean relative humidity (RH2M) | NASA POWER Daily Point API | as above |
| `seasonal_rainfall_mm` | mm | **Summed** precipitation over the season (PRECTOTCORR) | NASA POWER Daily Point API | as above |
| `irrigation_availability` | text/% | Irrigated area share | **Not auto-populated** — join Minor Irrigation Census / Agriculture Census tables (data.gov.in) by district | not yet joined |
| `major_crop` | text | Highest-production crop in that district-season | Ministry of Agriculture & Farmers Welfare, district/season-wise crop production statistics (data.gov.in) — manual download, see script 05 | no production record for that district-season |
| `secondary_crops` | text (`;`-separated) | Next 2–3 crops by production, only if present in source data | same as `major_crop` | no runner-up crops in source data |
| `flag_out_of_range` | text (semicolon flags) | Values outside physically plausible bounds (see `utils.PLAUSIBLE_RANGES`) — **for manual review, not auto-corrected** | derived during validation | empty = no flag |
| `flag_statistical_outlier` | text (semicolon flags) | 1.5×IQR outliers per column — **for manual review, not auto-corrected** | derived during validation | empty = no flag |

## Notes on things this pipeline deliberately does NOT do

- **It does not compute "total N" from SoilGrids and call it `nitrogen_kg_ha`.** SoilGrids' `nitrogen` property is total pedological nitrogen (g/kg), a different quantity from the plant-available, fertiliser-relevant kg/ha figure your model was trained on. Conflating the two would be a subtle but real fabrication of meaning, even with genuine numbers behind it.
- **It does not assign "Alluvial"/"Black"/"Red"/"Laterite" from texture alone.** Those are genetic/pedological soil classifications tied to parent material and geological history, not something texture percentages can determine. Getting real values for these requires the NBSS&LUP (National Bureau of Soil Survey & Land Use Planning) state soil maps — a genuine additional data-acquisition step, documented in the README, not shortcut here.
- **It does not compute Köppen climate zone or irrigation share automatically** — both require a raster/shapefile join this pipeline doesn't ship with by default, so the columns exist in the schema but are left null with instructions on how to fill them for real.

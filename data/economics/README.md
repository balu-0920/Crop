# Economics data (you supply it - nothing here is pre-filled)

The three CSVs in this folder are **header-only templates**. The app never
invents prices, costs or yields: until you add real rows, the profit /
revenue / expected-yield fields are reported as "unavailable".

Leave `state` / `district` blank for a national figure. Lookup order is
district -> state -> national. Crop names may be the model's names
(`rice`, `blackgram`) or common Indian names (`Paddy`, `Arhar/Tur`).

| File | Columns | Notes |
|---|---|---|
| `market_prices.csv` | `crop,state,district,price_inr_per_quintal,price_date,source` | 1 quintal = 100 kg. Newest `price_date` (YYYY-MM-DD) wins. Prices older than 1 year are flagged stale. Suggested sources: Agmarknet (mandi modal prices), MSP notifications (CACP / DAC&FW). |
| `cultivation_costs.csv` | `crop,state,cost_inr_per_hectare,reference_year,source` | Total cost per hectare. Suggested source: CACP "Cost of Cultivation" reports (Cost A2+FL or C2 - say which in `source`). Newest `reference_year` wins. |
| `historical_yields.csv` | `crop,state,district,year,yield_t_per_ha,source` | Tonnes/hectare = production / area. Suggested sources: DES&MoA&FW district-wise crop production statistics (data.gov.in), ICRISAT district database. Expected yield = median of the latest 5 years; range = 10th-90th percentile (min-max if < 5 years). |

Rows with non-numeric or non-positive values are skipped and counted
(see `GET /data_status`). Files are re-read automatically when changed.

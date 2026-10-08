# How to Run the Smart Crop Recommendation Project

The project has four parts that run from one Flask backend plus three static web pages:

| Part | Page | Needs |
|---|---|---|
| Crop recommendation (ML + regional + weather + yield/profit/risk + explanations) | `index.html` | nothing extra (works out of the box) |
| Agriculture documents Q&A (RAG) | `assistant.html` | `ANTHROPIC_API_KEY`, your PDFs, `python -m rag.ingest` |
| Farm decision agent (LangGraph) | `agent.html` | `ANTHROPIC_API_KEY`, `OPENWEATHER_API_KEY` (optionally PDFs) |

You can run just the first part and add the others later.

---

## 1. Requirements

- Python 3.10 or newer (developed and tested on 3.13)
- `pip`
- Internet access for: installing packages, weather, place lookup (OpenStreetMap), and the one-time download of the document-embedding model (about 130 MB, only if you use the Documents Q&A)

## 2. Install

Open a terminal in the project folder (the one that contains `app.py`):

```bash
cd combined_project

# recommended: a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **Always run commands from inside `combined_project/`.** The app uses relative paths (`data/`, `models/`) and will fail if started from another folder.

## 3. Configure (only for the optional parts)

```bash
cp .env.example .env              # Windows: copy .env.example .env
```

Open `.env` and fill in what you need (lines start with `#`, so remove the `#` to enable a line):

```
ANTHROPIC_API_KEY=your-key-here           # Documents Q&A and Decision Agent
OPENWEATHER_API_KEY=your-key-here         # Decision Agent weather tool
NOMINATIM_USER_AGENT=CropApp/1.0 (you@example.com)   # recommended: identify yourself to OpenStreetMap
```

Never commit `.env` (it is already in `.gitignore`).

The main crop page (`index.html`) fetches weather directly from the browser using the key already inside `weather/main.js`, so it does not use `.env`. (That key is visible to anyone who opens the page; consider replacing it with your own.)

## 4. Start the backend

```bash
python app.py
```

You should see lines like `Model and preprocessing objects loaded successfully.` and the server on **http://127.0.0.1:5000**. Leave this terminal running.

Quick check: open http://127.0.0.1:5000 in a browser; it should say *"Crop Recommendation backend is running."*

## 5. Start the frontend

In a **second terminal**:

```bash
cd combined_project/weather
python -m http.server 8080
```

Then open:

- **Crop recommendation:** http://localhost:8080/index.html
- **Documents Q&A:** http://localhost:8080/assistant.html
- **Decision agent:** http://localhost:8080/agent.html

(Do not just double-click the HTML files; browsers block the backend calls from `file://` pages.) The backend address `127.0.0.1:5000` is fixed in the page scripts, so keep the backend on port 5000.

---

## 6. Using each page

### Crop recommendation (`index.html`)
1. Search a city (or use the location button) to load weather. The page also resolves your region.
2. Enter soil values: Nitrogen, Phosphorus, Potassium (kg/ha) and soil pH.
3. Click **Recommend Crop**. You get the ML prediction, ranked recommendations, a risk level, a feature-contribution chart (SHAP) explaining the model's output, and a yield/revenue/profit card.

Yield, revenue and profit show **"Unavailable"** until you add real data (see section 8). They are never guessed.

### Documents Q&A (`assistant.html`)
1. Put agricultural PDFs (text-based, not scanned) into `combined_project/rag/documents/`.
2. Build the index (first run downloads the embedding model):
   ```bash
   python -m rag.ingest
   ```
   Re-run it whenever you add, change or remove PDFs; unchanged files are skipped.
3. With the backend running, ask a question on the page. Answers come only from your documents and list their sources; if the documents don't cover it, the assistant says so.

Test from the command line:
```bash
curl -X POST http://127.0.0.1:5000/api/agriculture/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the recommended seed rate for rice?"}'
```

### Decision agent (`agent.html`)
Fill in your question (e.g. *"Should I grow soybean in my farm this season?"*), a location, and your soil test values, then click **Get advice**. The agent fetches weather, runs the ML recommendation, looks up yield/profit/risk data, searches your documents, and writes advice. The result keeps four things separate: model prediction, external data, retrieved document knowledge, and the AI's reasoning (with numbers checked against the tools' outputs).

If location or soil values are missing, the page tells you what to add and skips the steps that need them.

Test from the command line:
```bash
curl -X POST http://127.0.0.1:5000/api/agent/advice \
  -H "Content-Type: application/json" \
  -d '{"question": "Should I grow soybean in my farm this season?", "location": "Hyderabad", "soil": {"N": 90, "P": 42, "K": 43, "ph": 6.5}}'
```

---

## 7. Run the tests

```bash
cd combined_project
python -m unittest discover -s tests -v
```

All tests run offline (they use stand-ins for weather, the LLM and the embedding model).

## 8. Optional: add your own data

**Yield / price / cost (turns on revenue and profit estimates).** Add real rows to the CSVs in `data/economics/` (`historical_yields.csv`, `market_prices.csv`, `cultivation_costs.csv`). Column formats and suggested public sources are in `data/economics/README.md`. Files reload automatically; check what loaded at http://127.0.0.1:5000/data_status.

**Regional crop and soil data.** The folder `dataset_pipeline/` contains scripts that build a real regional dataset (NASA POWER climate, SoilGrids soil, plus manual Soil Health Card and crop-statistics downloads). Run on a machine with internet access:
```bash
cd dataset_pipeline && pip install -r requirements.txt && cd src
python 01_get_districts.py && python 02_fetch_climate_nasa_power.py && python 03_fetch_soil_isric.py
python 04_prepare_soil_health_card.py && python 05_prepare_crop_season_stats.py   # need manual downloads first
python 06_merge_and_validate.py
```
The app picks up `dataset_pipeline/src/output/india_crop_dataset.csv` automatically. Without it, the app uses the built-in static regional list.

**Retrain the ML model** (compares Random Forest, XGBoost and Logistic Regression and saves the best):
```bash
python train_and_evaluate.py
```
Results are written to `reports/` and described in `MODEL_EVALUATION.md`.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| Page says it cannot reach the backend | Make sure `python app.py` is running and the page was opened via `http://localhost:8080` |
| `Address already in use` on port 5000 | Another program uses it (on macOS, AirPlay Receiver). Turn that off, or stop the other program; the pages expect port 5000 |
| `FileNotFoundError: data/...` or `models/...` | You are not inside `combined_project/`; `cd` into it first |
| `InconsistentVersionWarning` from scikit-learn | Harmless if predictions work; to remove it, run `python train_and_evaluate.py` to re-save the model with your installed version |
| Documents Q&A says "No documents are indexed" | Add PDFs to `rag/documents/`, then run `python -m rag.ingest` |
| Documents Q&A / agent returns 503 | `ANTHROPIC_API_KEY` is missing in `.env` (restart `python app.py` after editing it). Retrieved passages/evidence are still shown |
| Agent says weather unavailable | `OPENWEATHER_API_KEY` is missing in `.env` or invalid |
| Ingest fails to download the embedding model | Needs internet access to Hugging Face on first run; retry on an open network |
| A PDF is skipped during ingest | It is scanned/image-only (no text layer) or password-protected; OCR is not supported |
| "Unavailable" for yield, revenue, profit | Expected until you add data to `data/economics/` |

## 10. What to expect from the results

- Model accuracy figures describe agreement with the training dataset, not real farm outcomes.
- Feature-contribution charts show how the model weighs its inputs; they are not proof of cause and effect.
- The decision agent is decision support, not agronomic advice; check important decisions with a local extension officer.

# ForecastIQ — Setup Instructions (for teammates)

You've been sent this project as a raw folder/zip, not a git clone, so a few things need
manual setup on your machine before it'll run. Follow these in order.

## 0. Before you unzip — two things to know

1. **`.env` contains a real Gemini API key.** It's included so the app works out of the
   box, but don't commit this folder to a public GitHub repo or share it further without
   removing/rotating the key first.
2. **Don't reuse `venv/` or `frontend/node_modules/` if they got included in the zip.**
   They're both machine-specific (the Python venv has Pavan's exact folder path baked
   into its activate scripts, and `node_modules` is OS/architecture-specific). If you see
   these folders after unzipping, delete them and reinstall fresh per the steps below —
   trying to reuse them will fail in confusing ways.

## 1. Prerequisites

Install these first if you don't have them:

- **Python 3.11** (the project was built/tested on 3.11.9 — other 3.11.x should be fine,
  avoid 3.12+ since some pinned ML library versions may not have wheels for it yet)
- **Node.js 18.17+ or 20+** (required by Next.js 14)
- **Git** (optional, only needed if you want to commit changes)

## 2. Backend setup (Python / FastAPI)

Open a terminal in the project root (the folder with `README.md`, `run.sh`, `backend/`, `src/`).

```bash
# Create a fresh virtual environment
python -m venv venv

# Activate it
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (cmd):
venv\Scripts\activate.bat
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

This installs pandas, xgboost, lightgbm, catboost, prophet, optuna, shap, fastapi,
uvicorn, and everything else the pipeline needs. Prophet in particular can take a few
minutes to install (it builds a Stan model backend).

### Model file

`pickle/model.pkl` is already included and committed — you do **not** need to retrain
anything to run the app. It's the trained ensemble (XGBoost + LightGBM + CatBoost +
Prophet) as of the last training run.

If you ever do want to retrain from scratch (e.g. after changing `data/`), run:

```bash
python retrain_and_pickle.py
```

from the project root, with the venv activated.

## 3. Frontend setup (Next.js)

Open a **second** terminal (keep this separate from the backend one).

```bash
cd frontend
npm install
```

This will take a minute — it needs to reinstall all of `node_modules` from scratch.

`frontend/.env.local` should already be present and points the frontend at
`http://localhost:8000` (the backend's default port). You don't need to change this
unless you're running the backend on a different port.

## 4. Running the app

You need **both** the backend and frontend running at the same time, in two separate
terminals.

**Terminal 1 — backend** (from the project root, venv activated):
```bash
python backend/src/main.py
```
This starts the FastAPI server on `http://localhost:8000`. First request after startup
will take a few seconds while it trains/loads the ensemble and warms the cache — this is
normal, subsequent requests are fast.

**Terminal 2 — frontend** (from `frontend/`):
```bash
npm run dev
```
This starts Next.js on `http://localhost:3000`.

Then open **http://localhost:3000** in your browser. You should see the ForecastIQ
dashboard (Overview, Data Validation, Model Validation, Forecasts, Scenarios, Budget
Optimizer, Monte Carlo, Explainability, Risk & Insights, Ask ForecastIQ).

## 5. Sanity checks

- Backend health check: open `http://localhost:8000/api/status` in a browser — should
  return a small JSON blob confirming the system is online.
- If the frontend loads but shows no data / infinite loading, the backend probably isn't
  running yet, or is still on its first (slow) request warming the cache — wait ~10-20
  seconds and refresh.
- If `pip install -r requirements.txt` fails on `prophet`, make sure you're on Python
  3.11 (not 3.12+) — this is the most common cause.

## 6. Other useful scripts (optional, not needed just to run the app)

- `python -m src.evaluation --folds 3 --min-train-days 120 --max-horizon 90 --output-dir output`
  — reruns the full rolling-origin backtest and regenerates
  `output/backtest_summary.json` / `output/backtest_scorecard.csv` (what powers the Model
  Validation tab). Takes about a minute.
- `./run.sh <DATA_DIR> <MODEL_PATH> <OUTPUT_PATH>` — the core pipeline evaluation
  entrypoint (Linux/Mac/WSL/Git Bash; not a plain Windows PowerShell script).

## 7. Project structure (quick orientation)

```
data/                    Raw channel CSVs (Google Ads, Meta Ads, Bing Ads)
src/                      Core Python pipeline (models, validation, scenarios,
                          monte carlo, budget optimizer, risk engine, chat engine, etc.)
backend/src/main.py       FastAPI server — wires src/ into REST endpoints for the frontend
frontend/src/app/page.tsx The entire dashboard UI (single-page Next.js app)
pickle/model.pkl          Trained ensemble artifact (already included, no retrain needed)
output/                   Backtest scorecard + summary (feeds Model Validation tab)
retrain_and_pickle.py     Full retrain script — only needed if you change data/
run.sh                    Core pipeline entrypoint
```

If anything doesn't come up, ping Pavan rather than debugging blind — a couple of the
fixes made recently (interval calibration, hierarchical reconciliation) depend on the
exact `pickle/model.pkl` included in this repo, so a from-scratch retrain on a different
machine could produce slightly different numbers than what's shown in the demo.

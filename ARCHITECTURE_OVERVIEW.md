# ForecastIQ — Architecture Overview

## System Architecture

```
data/ (CSV inputs)
    │
    ▼
[ValidationEngine]          ← ingestion, schema enforcement, quality scoring
    │
    ▼
[FeatureEngineer]           ← time, performance, marketing, channel features
    │
    ▼
[EnsembleForecaster]        ← Prophet + XGBoost + LightGBM + CatBoost
    │         │
    │    pickle/model.pkl   ← pre-trained, committed to repo
    │
    ▼
output/predictions.csv      ← scored deliverable (P10/P50/P90 from residual-based
    │                          interval scaling -- see Forecasting Pipeline below,
    │                          NOT from Monte Carlo; that's a dashboard-only feature)
    ▼
[FastAPI Backend]           ← 17 REST endpoints serving live analytics
    │         │
    │    [MonteCarloSimulator]  ← 10,000-iteration portfolio simulation, powers
    │                              only the dashboard's /api/simulations view --
    │                              not part of the run.sh / predictions.csv path
    ▼
[Next.js Frontend]          ← executive dashboard UI
```

---

## Frontend Stack

| Component | Technology |
|---|---|
| Framework | Next.js 14 (App Router) |
| Language | TypeScript |
| Styling | Tailwind CSS |
| Charts | Recharts |
| State | React useState / useEffect |
| API communication | fetch() with `NEXT_PUBLIC_API_BASE` env var |

The frontend is a single-page executive dashboard with nine views: Overview, Data Validation, Forecasts, Scenarios, Budget Optimizer, Monte Carlo, Explainability, Risk & Insights, and Ask ForecastIQ (chat). All data is fetched from the FastAPI backend at startup and cached client-side.

---

## Backend Stack

| Component | Technology |
|---|---|
| Framework | FastAPI |
| Server | Uvicorn |
| Language | Python 3.11+ |
| Data layer | SQLite via SQLAlchemy (models defined in `src/database.py`, actively wired into `/api/runs` and a persist-on-every-computed-run audit trail — see note below) |
| Serialization | Pydantic v2 |
| PDF generation | ReportLab |

The backend exposes 17 REST endpoints under `/api/`. All heavy computation (forecasting, SHAP, Monte Carlo, optimization) runs once on startup and is cached in-memory (behind a lock, so concurrent requests during a cold cache trigger exactly one rebuild, not one per request) for sub-millisecond subsequent API responses. Cache invalidation is triggered by file modification timestamps on the `data/` CSVs.

**Note on `src/database.py`:** a full SQLAlchemy persistence layer (`User`, `UploadedDataset`, `ForecastRun`, `Scenario`, `Report` models) is wired into the live request path - every time the analytics cache is (re)computed, the resulting run is persisted via `save_forecast_run()`, and `/api/runs` reads that history back out as an audit trail. `User`/`UploadedDataset`/`Scenario`/`Report` remain scaffolding for a future multi-user "save/revisit past runs" feature beyond the current single-tenant dashboard.

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | GET | System health, LLM provider, data quality score |
| `/api/overview` | GET | Executive KPIs, 90-day trajectory, channel attribution |
| `/api/trajectory` | GET | 90-day daily revenue trajectory for area chart |
| `/api/validation` | GET | Full data quality audit report |
| `/api/model-validation` | GET | Backtest/model validation scorecard |
| `/api/forecasts` | GET | P10/P50/P90 by dimension (Overall/Channel/CampaignType/Campaign) |
| `/api/dimensions` | GET | Real set of channels and campaign types present in the current data |
| `/api/simulations` | GET | Monte Carlo portfolio simulation results |
| `/api/simulate-budget` | POST | What-if budget change simulation (any channel present in the data) |
| `/api/optimize-budget` | POST | Optuna budget allocation optimizer (any channel present in the data) |
| `/api/scenarios` | GET | Bull/Base/Bear scenario projections |
| `/api/explainability` | GET | SHAP feature importance for revenue and ROAS |
| `/api/risk` | GET | Risk profile with factor scores and mitigations |
| `/api/insights` | GET | Rule-based executive insights and recommendations |
| `/api/runs` | GET | Recent persisted forecast runs (SQLite audit trail) |
| `/api/chat` | POST | Conversational AI query answering |
| `/api/report/pdf` | GET | Download executive PDF report |

---

## Forecasting Pipeline

```
run.sh
  ├── src/generate_features.py
  │     ├── ValidationEngine.run_full_ingestion()   ← reads data/*.csv
  │     ├── FeatureEngineer.generate_all_features()
  │     └── writes features.pkl
  │
  └── src/predict.py
        ├── loads features.pkl → determines forecast start_date
        ├── EnsembleForecaster.load_models()         ← loads pickle/model.pkl
        ├── EnsembleForecaster.produce_full_predictions_table()
        └── writes output/predictions.csv
```

The pipeline is fully offline and requires no network access. All model weights are pre-trained and committed to the repository under `pickle/model.pkl`.

Forecast generation includes a recent-reality calibration layer: learned ensemble outputs are blended with a trailing 30-day observed revenue and spend baseline (matching the same window `evaluation.py`'s naive-baseline benchmark uses), plus a damped-trend momentum term derived from a 30-vs-30-day comparison (see `_recent_baseline`/`_blend_with_recent_baseline` in `src/models.py`). P10/P90 bands then use an empirical `3.0x` interval scale to improve holdout coverage. This keeps forecasts anchored to the latest business level while preserving seasonality, nonlinear model behavior, and dimension-specific forecasts.

---

## Model Evaluation Workflow

```
src/evaluation.py
  - ValidationEngine.run_full_ingestion()
  - selects rolling forecast origins with enough train and holdout history
  - retrains EnsembleForecaster on data before each origin
  - forecasts 30/60/90-day P10/P50/P90 windows
  - compares predictions with actual holdout Revenue and ROAS
  - writes output/backtest_scorecard.csv and output/backtest_summary.json
```

The evaluator is intentionally separate from `run.sh` so the core pipeline command remains fast and contract-compliant, while the project still has a reviewer-facing reliability story with WAPE, SMAPE, MAE, and interval coverage.

---

## LLM Integration Workflow

```
User Question / Insight Request
        │
        ▼
  GEMINI_API_KEY set?
   YES ──────────────────────────────────────────────────────┐
        │                                                     │
        NO                                              GeminiProvider
        │                                               (gemini-2.5-flash)
        ▼                                                     │
  MockLLMProvider                                             │
  (fully data-driven,                                         │
   no network calls)                                          │
        │                                                     │
        └───────────────────┬─────────────────────────────────┘
                            │
                            ▼
              Structured analytics context payload:
              - total spend / revenue / ROAS
              - channel breakdown (per-channel stats)
              - 30d vs prior 30d trend deltas
              - 90-day P50 forecast values
              - risk classification
                            │
                            ▼
              Natural language causal narrative response
              (grounded in actual computed dataset statistics)
```

The dual-provider architecture ensures the system delivers high-quality AI-assisted insights in production (Gemini) while remaining fully functional in offline/automated evaluation environments (MockLLMProvider).

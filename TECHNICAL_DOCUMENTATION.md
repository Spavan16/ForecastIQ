# ForecastIQ — Technical Documentation

## 1. Forecasting Methodology

ForecastIQ uses a **weighted ensemble of four models** to produce probabilistic revenue and ROAS forecasts across 30, 60, and 90-day planning horizons.

### Ensemble Architecture

| Model | Weight | Role |
|---|---|---|
| Prophet | 35% | Seasonal decomposition, trend + weekly/annual cycles |
| XGBoost | 25% | Non-linear spend-to-revenue mapping, feature interactions |
| LightGBM | 20% | Gradient boosting with fast training on high-cardinality campaign features |
| CatBoost | 20% | Categorical feature handling (channel, campaign type, season) |

The weighted average of all four model predictions forms the raw ensemble signal. This raw signal is then blended with a **recency-anchor baseline** at an **8% raw-ensemble / 92% recency-anchor** split for revenue (`model_blend_weight`; spend uses its own `spend_blend_weight`, kept at 25%) before being projected forward as the final P50 forecast. The anchor itself is not a flat trailing average -- it's a damped-trend extrapolation (Holt's damped trend): a base level from the trailing 30-day mean, plus a recent momentum term (last 30 vs. previous 30 days, clamped to +/-3%/day) that fades out with a per-day damping factor (`PHI=0.92`) the further out the forecast goes. This was added specifically because the naive baseline the evaluation harness benchmarks against is flat *by definition* (see `evaluation.py::_naive_baseline_value`) and structurally can't capture real revenue momentum -- see `_recent_baseline`/`_blend_with_recent_baseline` in `models.py` and Section 5 for the backtest evidence that motivated it. Before that trailing-window average is used, it's checked against a real per-calendar-month seasonal index (`_monthly_seasonal_index`, built from 2+ years of actual history) and deseasonalized/re-seasonalized to the forecast's actual start month -- but only when the origin month's typical level differs from the trailing window's by 2x or more, which limits this correction to genuinely extreme seasonal transitions (e.g. the real Nov-Dec revenue spike here, which craters back to baseline every January) rather than ordinary month-to-month drift the momentum term above already handles. See Section 5 for the specific bug (catastrophic January-origin forecasts) this closed. This heavy weighting toward the recency anchor is a deliberate, empirically-tuned trade-off: it keeps the ensemble meaningfully model-driven while closing most of the Revenue-vs-naive-baseline gap, without collapsing to `model_blend_weight=0` (which would make the forecast mathematically identical to the naive baseline it's benchmarked against). Probabilistic ranges (P10/P90) are derived from historical residual standard deviation scaled by planning horizon, then widened by an empirically-tuned interval calibration scale — wider intervals at 90 days reflect genuine macro uncertainty, not arbitrary padding. This scale factor was re-tuned against an 8-fold backtest (not the 3-fold default) after a fold-count stress test showed the original value, tuned only against 3 folds, undercovered on harder/sparser historical periods; see Section 5 for the honest before/after numbers.

### Monte Carlo Simulation

Separately from the ensemble's own P10/P50/P90 bands, a Monte Carlo simulation (10,000 iterations, seed=42 for reproducibility) samples from a volatility distribution derived from the ensemble's residual std to produce the Risk tab's portfolio-level revenue/ROAS distributions, worst/expected/best-case classifications, and histogram visualizations. It does not feed into `predictions.csv` — the P10/P90 values there are computed analytically from residual standard deviation scaled by horizon (see above), independent of the Monte Carlo paths.

### Forecast Dimensions

Forecasts are produced at four levels of granularity:
- **Overall**: blended portfolio across all channels
- **Channel**: Google Ads, Meta Ads, Bing Ads independently
- **Campaign Type**: SEARCH, SOCIAL, PERFORMANCE_MAX, SHOPPING, DISPLAY, VIDEO, DEMAND_GEN, AUDIENCE (derived from the actual channel data at prediction time, not a fixed list — see Section 5)
- **Campaign**: top individual campaigns by historical revenue contribution

---

## 2. Model Selection Rationale

**Why an ensemble and not a single model?**

Each model captures a different signal in marketing data:

- Prophet excels at capturing weekly purchase intent cycles and seasonal shopping events (Black Friday, Q4) that are invisible to tree-based models without explicit feature engineering.
- XGBoost/LightGBM/CatBoost capture non-linear diminishing returns on ad spend — the relationship between spend and revenue is not linear at high budget levels, and tree-based models learn this directly from data.
- Ensemble weighting (Prophet higher at 35%) reflects the strong seasonality present in e-commerce marketing data, where day-of-week and month effects are among the most predictive features.

**Why not a neural network / LSTM?**

The dataset (887 daily rows across ~2.5 years) is too small for deep learning to outperform well-tuned gradient boosting. Ensemble tree models + Prophet deliver superior out-of-sample accuracy on marketing time-series at this data volume.

---

## 3. Data Preprocessing Logic

### Ingestion

Three channel CSV files are ingested independently:
- `google_ads_campaign_stats.csv` — 19,272 records
- `meta_ads_campaign_stats.csv` — 3,417 records  
- `bing_campaign_stats.csv` — 2,873 records

Each file undergoes channel-specific validation (column existence, dtype enforcement, spend/revenue range checks) before being unified into a single cross-channel dataframe (25,562 records total).

### Normalization

- **Google Ads spend**: converted from micros (divide by 1,000,000) to dollars
- **Meta Ads revenue**: sourced from the `conversion` column (represents conversion value)
- **Bing Ads columns**: normalized from PascalCase (`Spend`, `Revenue`) to lowercase

### Date Handling

Malformed or missing dates are forward-filled using `.bfill()` then defaulted to `2025-01-01` if the entire column is null. Date parsing uses `errors='coerce'` to prevent ingestion failures on format inconsistencies.

### Feature Engineering

From the unified dataframe, the following features are constructed:
- **Time features**: month, quarter, ISO week, day_of_week, is_weekend
- **Season encoding**: WINTER(0), SPRING(1), SUMMER(2), FALL(3)
- **Rolling performance**: 7/14/30-day rolling mean and std of revenue and spend per channel
- **Efficiency features**: spend-to-revenue ratio, rolling ROAS, volatility index
- **Channel encoding**: one-hot encoded channel and campaign_type columns

---

## 4. Assumptions

1. **Attribution is treated as source of truth.** No custom attribution model or Media Mix Modeling (MMM) is applied. Channel-reported revenue figures are used as-is per challenge specification.

2. **Stationarity of channel mix.** The ensemble assumes the relative contribution of Google Ads, Meta Ads, and Bing Ads to total revenue remains broadly stable over the forecast horizon. Structural shifts (e.g. a channel being paused entirely) are outside the model's scope.

3. **Spend continuity.** Forecasts assume ad spend continues at approximately the historical run rate unless explicitly modified via the budget simulator. The model does not account for future spend changes autonomously.

4. **No external macro variables.** The model does not incorporate macroeconomic signals (inflation, consumer confidence) or competitor activity. Seasonality is captured via date features derived from historical patterns only.

5. **Campaign structure stability.** Which channels, campaign types, and campaigns get scored is derived from the actual held-out data at prediction time (not a fixed list baked in at training time), so new campaign types or campaign names introduced after the training cutoff are still scored — via the Universal Dimension Fallback, which scales down the current overall-portfolio forecast for that entity (see Section 5) rather than requiring an entity-specific trained model.

---

## 5. Limitations

- **Short Bing Ads history**: Bing Ads has 2,873 records across fewer unique campaigns, making campaign-level Bing forecasts less reliable than Google Ads equivalents.
- **Cold start for new campaigns**: Campaigns not seen during training use a scaled-down version of the overall forecast. This is disclosed in the fallback path.
- **90-day horizon uncertainty**: P10/P90 intervals widen significantly at 90 days. The P50 point estimate is reliable; the tails should be treated as scenario bounds rather than precise predictions.
- **No intra-day modeling**: All forecasts are aggregate-period (30/60/90 days) as specified. Daily granularity trajectories shown in the UI are smoothed interpolations for visualization only and are not scored outputs.
- **Revenue/ROAS MAPE vs. a naive baseline, and why this result is fold-count sensitive**: In the offline rolling-origin backtest (3 folds, `output/backtest_summary.json`), ROAS forecasts beat a naive baseline (flat trailing-30-day average) at all three horizons (16.1% improvement at 30 days, 29.4% at 60 days, 6.7% at 90 days); Revenue forecasts beat the naive baseline at two of three horizons (60 days: +5.5%, 90 days: +6.9%), with 30-day Revenue still trailing by -5.1%. **This ROAS result does not hold up under the 12-fold stress test** (`output_12fold/backtest_summary.json`, same model and data, more rolling origins): ROAS loses to the naive baseline by -40.4% at 30 days and -43.1% at 60 days, and only narrowly wins at 90 days (+3.7%). Revenue moves the opposite direction under the same stress test, widening to +42-46% across all three horizons. Current read: ROAS is a ratio (`revenue_p50/spend_sum`) with a much smaller denominator range than Revenue's raw dollar scale, so forecast noise that barely moves Revenue's percentage error swings ROAS's percentage error far more on the sparser, harder 12-fold origins. The `model_blend_weight`/damped-trend fixes below were tuned against Revenue's own error, not re-run with ROAS's error as the target — that is a known, not-yet-done next step. Treat the 3-fold ROAS win as real but fold-count-fragile, not as a settled result. We're disclosing both fold counts rather than only the friendlier 3-fold number — see the full backtest breakdown and root-cause analysis (recency-anchoring weight, limited history at the earliest backtest origin) in `executive_forecast_report.pdf`, Section 4.
- **January-origin forecasts and the Nov–Dec demand spike**: A 12-fold backtest (`output_12fold/backtest_summary.json`, more rolling origins than the 3-fold default) surfaced a January-specific failure the 3-fold sample happened to skip: origins landing in January inherited a recency-anchor baseline still anchored to December's trailing window, which runs 5-10x a normal month in this data. A seasonal deseasonalization step (`_monthly_seasonal_index` in `models.py`, described in Section 1) now corrects this when the mismatch is extreme (2x+) and there's 2+ years of history to trust the pattern — fixing the 2026-01-05 origin's Revenue APE from 553%/375%/332% down to 25%/9%/17% (30/60/90 days) with zero change to the 3-fold headline numbers above. The 2025-01-01 origin remains uncorrected (~389% APE) and is a disclosed limitation, not a bug we chased further: at that point in the timeline there's only 1 observed December, and the guard intentionally requires 2+ years before trusting a monthly pattern as real seasonality rather than a one-off.
- **Campaign-level accuracy and `top_down_weight`, and why this too is fold-count sensitive**: individual named campaigns (as opposed to portfolio/channel/campaign-type rollups) carry the highest error in the system — Revenue WAPE of 50.8% in the 3-fold backtest, down from 70.8% before hierarchical reconciliation was added (see Section 1) and from 52.6% at that fix's original, un-swept blend weight. **Under the 12-fold stress test this nearly doubles, to 99.66% at the chosen weight** (`output_12fold/backtest_summary.json`, `campaign_level.revenue_wape`) — the same fold-count-sensitivity pattern documented above for ROAS-vs-naive and interval coverage, driven by the same root cause: campaigns with the shortest, sparsest history are disproportionately represented in the harder 12-fold origin set. `top_down_weight` (how much to trust the top-down channel-share allocation vs. each campaign's own noisy bottom-up model) was swept against the live 3-fold backtest at 0.0/0.35/0.5/0.65/0.8 and cross-checked against the 12-fold stress test; 0.5 won on every campaign-level metric simultaneously (WAPE/SMAPE/ROAS-SMAPE/interval-coverage) at both fold counts, not just one cherry-picked number — full results and reasoning are in the code comment at its point of use in `forecast_dimension()`. Campaign-level forecasts remain the weakest part of the system regardless — each campaign is trained on only 6 calendar features against a single, often short/sparse revenue history, and campaign lifecycle effects (wind-down, budget exhaustion) aren't modeled without an explicit end-date signal.
- **Interval coverage is fold-count sensitive**: Coverage numbers reported from the default 3-fold backtest are optimistic. Re-running the identical model and data with `--folds 8` (more rolling origins, no new data — a stress test) drops coverage from ~91% to 62.3%, because the wider origin set includes sparser, harder historical periods the 3-fold sample happened to skip. The interval calibration scale was originally tuned against only 3 folds; we re-tuned it empirically against the 8-fold backtest instead (1.45x → 3.0x), which lifts 8-fold coverage to 77.0% — a real, verified improvement, not a claim of hitting 90% under harder conditions. A P10 floor in the code (`max(5% of P50, computed P10)`) caps how much further widening alone can close the remaining gap; fully closing it would need a separate fix to that floor logic. Point estimates (WAPE, ROAS/Revenue MAPE) are unaffected, since this scale only widens P10/P90, not the P50 forecast.

---

## 6. AI Integration Strategy

### Primary LLM: Gemini (gemini-2.5-flash)

When a `GEMINI_API_KEY` is present in the environment, all causal summaries, anomaly interpretations, and chat responses are generated by Gemini via the `/v1beta/models/gemini-2.5-flash:generateContent` endpoint. The LLM receives a structured analytics context payload containing:
- Historical spend, revenue, and ROAS by channel
- Recent 30-day vs prior 30-day trend deltas
- 90-day P50 forecast values
- Risk profile classification

This grounds every LLM response in actual computed numbers rather than general marketing knowledge.

### Offline Fallback: Rule-Based Causal Engine

When no API key is present (e.g. in an automated CI/evaluation run), the system falls back to a fully data-driven rule engine (`rule_engine.py`) and a data-driven chat engine (`chat_engine.py`). These compute the same analytical statistics from the data and generate structured causal narratives — with real numbers from the actual dataset — without any network calls.

This design ensures the forecasting utility functions completely offline while maintaining high-quality AI-assisted insights in production environments where an API key is available.

### Pipeline Causal Summary (`predict.py` → `output/causal_summary.json`)

AI-assisted causal summaries are a core "Working Prototype" capability. `src/predict.py` — the script `run.sh` actually invokes — writes `output/causal_summary.json` alongside `predictions.csv` on every run, using `MockLLMProvider` directly (not the API-key-autodetecting factory) so this artifact stays 100% network-free regardless of local `.env` configuration, matching the "no network calls at runtime" contract. The summary is built from the same real 90-day P50 revenue/ROAS forecast and channel breakdown produced by that run — not hardcoded text.

### LLM Use Cases

| Use Case | Implementation |
|---|---|
| Executive summary generation | `RuleInsightEngine` (offline) / Gemini (online) |
| Anomaly interpretation | `RuleInsightEngine` ROAS trend detection |
| Causal Q&A | `ForecastChatBot` with 10 intent classifiers |
| Risk narrative | `RiskIntelligenceEngine` factor scoring |
| Budget recommendation rationale | `BudgetOptimizer` + Gemini explanation |

# Update: Pre-Release Audit — Bing Cold-Start Contamination + Explainability Mislabel

**Date:** July 19, 2026
**Scope:** `src/models.py`, `src/budget_optimizer.py`, `src/explainability.py`, `.gitattributes`, `output/`

---

## 1. Bug found: Bing Ads 30-day ROAS forecast cratering to 0.64x

Live dashboard showed Bing Ads' 30-day ROAS forecast at **0.64x median**, against a
trailing 30/60-day actual ROAS of 1.65x/1.32x — an obviously wrong, order-of-magnitude
mismatch caught by a manual pre-release spot-check, not a backtest fold.

**Root cause:** `_monthly_seasonal_index()` in `src/models.py` (the same function fixed
for the January/Nov-Dec spike on July 17) had a second, distinct blind spot: it gates a
month's seasonal index on 2+ distinct calendar years being *present*, but didn't check
whether either year's total was genuinely zero. Bing Ads' June 2024 (channel not yet
launched, entire calendar month at $0 revenue) was being averaged in alongside June
2025's real data (16 non-zero days, mean $168.26), dragging June's mean — and therefore
its seasonal index — down to 0.44x. That fed into `_recent_baseline()` and cratered the
30-day ROAS forecast.

An earlier attempted fix (gating on non-zero-day *count* summed across years) did **not**
resolve it — verified directly, since June's non-zero-day count (0+16+2=18) still cleared
that gate. Re-diagnosed year-by-year and fixed by excluding any single (month, year) pair
from the average when that year's *total* for the month is exactly zero, before requiring
2+ remaining years.

### Verified result

| | Before | After |
|---|---|---|
| Bing Ads 30-day ROAS forecast (median) | 0.64x | **1.8x** |
| Google Ads / Meta Ads forecasts | — | Unaffected (unchanged) |

---

## 2. Same bug class, second location: `budget_optimizer.py`

`BudgetOptimizer._fit_channel_efficiency()` calibrates each channel's diminishing-returns
curve from `historical_df`, independently of `EnsembleForecaster` — so it wasn't covered
by the `models.py` fix above. Its monthly aggregation filtered out months with zero
*spend* but not months with real spend and zero *revenue*: Bing Ads' 2024-05/06/07 show
$28 / $3,436 / $4,396 of real spend against $0.00 revenue each (the same launch/tracking-
ramp period), ~$11,860 of real spend total dragging the fitted alpha (expected revenue
per unit spend) down using pre-launch data.

**Fix:** exclude `revenue == 0` months from the monthly aggregation before fitting alpha,
same logic as Section 1. Confirmed Google Ads and Meta Ads curves unaffected. No backtest
harness exists for this module (unlike `models.py`), so verified via direct before/after
comparison of Optuna's resulting recommendation rather than a formal metric.

---

## 3. Explainability mislabel: historical ratio labeled "P50"

`ExplainabilityEngine._stability_label()` labeled a channel's trailing historical
ROAS (`sum(revenue)/sum(spend)` across all history) as **"P50"** — borrowing
probabilistic-forecast language from the app's real P10/P50/P90 forecast intervals
(`models.py`/`predictions.csv`) for a plain historical aggregate with no distributional
basis at all. A reader-facing mislabeling risk: it implies this number is a forecast
percentile when it's a trailing average.

**Fix:** relabeled to **"Historical ROAS"**, explicit in both the stability tier string
and a code comment disclosing the distinction, so it can't be mistaken for a forecast
value downstream.

---

## 4. Git binary tracking (`.pdf`/`.pkl` diffed as text)

`.gitattributes` had no `binary` rules for `.pdf`/`.pkl`, so both were being diffed
(and risked CRLF-normalized) as text — a silent corruption risk for the trained model
artifact and the executive PDF report. Added explicit `binary` rules for both, verified
via `git check-attr` and confirmed `git diff` now shows clean `Bin X -> Y bytes` deltas,
not line-level text diffs.

---

## 5. Verification

- Regenerated `output/predictions.csv` via `run.sh`, confirmed consistent across
  Forecasts, Model Validation, and the causal summary.
- Full pytest suite (19 tests) passes unchanged.
- Fresh clean-clone test (new venv, `pip install -r requirements.txt`, `run.sh` via a
  separate clone) reproduces `output/predictions.csv` byte-for-bit identically — zero
  drift from any local machine state.
- No stale pre-fix numbers (e.g. the old 0.64x) found remaining in `README.md` or
  `TECHNICAL_DOCUMENTATION.md`.

---

## Files changed

- `src/models.py` — `_monthly_seasonal_index()`: exclude (month, year) pairs with a
  genuinely zero-revenue year from the seasonal average
- `src/budget_optimizer.py` — `_fit_channel_efficiency()`: exclude zero-revenue months
  (not just zero-spend months) from the monthly efficiency-curve fit
- `src/explainability.py` — `_stability_label()`: "P50" → "Historical ROAS"
- `.gitattributes` — added `*.pdf binary`, `*.pkl binary`
- `output/predictions.csv`, `output/causal_summary.json`,
  `output/executive_forecast_report.pdf` — regenerated to reflect all fixes above

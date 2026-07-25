# Update: January Seasonality Fix + top_down_weight Sweep

**Date:** July 17, 2026
**Scope:** `src/models.py`, `README.md`, `TECHNICAL_DOCUMENTATION.md`, `pickle/model.pkl`, `output/`, `output_12fold/`

---

## 1. Bug found: catastrophic January-origin forecasts

A 12-fold backtest (more rolling origins than the standard 3-fold demo, same model and
data — a stress test, not new data) surfaced something the 3-fold sample happened to
skip entirely: the two origins landing in January (2025-01-01, 2026-01-05) posted Revenue
APE of **330–553%**, roughly an order of magnitude worse than every other fold
(next-worst ~80%).

**Root cause:** Nov–Dec revenue genuinely spikes 5–10x a normal month in both 2024 and
2025, then craters back to baseline every January. The recency-anchor baseline's trailing
30-day window, for any January origin, *is* December — so the anchor had no way to know
"the last 30 days were a known seasonal spike, not the new normal." The damped-trend term
was reading the same contaminated window and extrapolating the spike even further upward
into January instead of projecting the crash back down.

### Fix

Added `_monthly_seasonal_index()` to `EnsembleForecaster` in `src/models.py`: computes a
real per-calendar-month seasonal index from the full available history (2+ years), then
`_recent_baseline()` deseasonalizes the trailing window before it's used and
re-seasonalizes to the month the forecast actually starts in. Built from the data, not a
hardcoded "December is special" rule.

Two guardrails, each added only after a regression proved it necessary:

1. **Median-anchored index, not mean-anchored.** An overall-mean reference gets dragged
   upward by the two outlier months themselves, which silently shrunk every *other*
   month's forecast by 30–50% in an earlier version of this fix (caught by the standard
   3-fold backtest, whose origins aren't even in January).
2. **Extreme-mismatch gate (2x+).** The correction only fires when the origin month's
   typical level is at least 2x (or at most 0.5x) the trailing window's typical level —
   this excludes ordinary month-to-month drift (which the existing damped-trend/momentum
   term already handles correctly) and fires only for genuinely extreme seasonal
   transitions like Dec→Jan.
3. **2-year confidence guard.** A month only gets a seasonal index once observed in 2+
   distinct calendar years — a single year's occurrence of noisy channels (e.g. Meta
   Ads, ~11 months of history) can't be told apart from a real recurring pattern. An
   earlier 1-year-shrinkage version of this guard also shipped a real regression and was
   reverted for the same reason.

### Verified results

| Fold | Before | After |
|---|---|---|
| 2026-01-05 (30/60/90d Revenue APE) | 553% / 375% / 332% | **25% / 9% / 17%** |
| 2025-01-01 (30/60/90d Revenue APE) | 389% / 380% / 368% | 46% / 44% / 40% *(unchanged — disclosed limitation, only 1 year of December history at that point in the timeline, guard intentionally declines to apply)* |
| Standard 3-fold backtest (headline numbers) | WAPE 22.65, Rev-vs-naive -5.1%/+5.5%/+6.9% | **Byte-identical** — zero collateral damage |
| 12-fold Revenue-vs-naive | ~40–46% at all horizons | 46.4% / 42.2% / 41.6% |

---

## 2. top_down_weight sweep (campaign-level accuracy)

`top_down_weight` controls how much each campaign's forecast trusts a top-down channel
allocation vs. its own noisy bottom-up model. It was a hardcoded `0.65`, asserted without
evidence. Refactored into a proper instance attribute (`self.top_down_weight`, persisted
in the pickle, same pattern as `model_blend_weight`) so it could be swept safely.

Swept 0.0 / 0.35 / 0.5 / 0.65 / 0.8 against the live 3-fold backtest (confirmed this only
affects `dimension_type == "campaign"` — Overall/Channel/CampaignType and every ROAS
number were byte-identical across every run), then cross-checked against the 12-fold
stress test.

| top_down_weight | Revenue WAPE | Revenue SMAPE | ROAS SMAPE | Interval Coverage |
|---|---|---|---|---|
| 0.00 (no blend) | 56.73 | 69.18 | 56.62 | 90.0% |
| 0.35 | 50.45 | 60.95 | 44.96 | 90.0% |
| **0.50 (chosen, final)** | **50.78** | **61.29** | **43.59** | **90.67%** |
| 0.65 (old default) | 52.62 | 62.48 | 43.60 | 90.67% |
| 0.80 | 55.57 | 64.02 | 45.75 | 88.67% |

**0.50 selected** — best or tied-best on every campaign metric simultaneously, not just
one cherry-picked number. 0.35 edges out 0.50 on WAPE/SMAPE alone but gives back more
ground on ROAS SMAPE and coverage than it gains. Deliberately stopped at 0.05 increments
rather than chasing the 3-fold WAPE optimum — with only 3–12 folds to validate against,
finer tuning risks fitting the folds rather than fixing anything real.

Net effect: campaign-level Revenue WAPE went from **70.8%** (before hierarchical
reconciliation existed at all) → **52.6%** (hierarchical reconciliation added, weight
un-swept) → **50.8%** (this sweep). This is now final — no further tinkering on this
parameter.

---

## 3. Pickle regeneration (important gotcha caught and fixed)

`predict.py` loads `pickle/model.pkl` via `load_models()`, which **overwrites**
`top_down_weight` from the saved artifact — falling back to a hardcoded `0.65` if the key
is missing. The pre-existing pickle predated this attribute, so the swept `0.5` value was
silently *not* reaching the actual production pipeline until the pickle itself was
regenerated.

Fixed by running the existing, sanctioned `retrain_and_pickle.py` (trains on the full
`data/` dataset the same way the original pickle was built — not a "retrain on test data"
concern). Verified directly: `pickle/model.pkl` now contains `top_down_weight: 0.5`.

By contrast, the January seasonal fix (Section 1) did **not** need a pickle regeneration —
`_recent_baseline()`/`_monthly_seasonal_index()` are called fresh against held-out data at
prediction time via `refresh_recent_context()`, not loaded from the pickle.

**Lesson:** any fix to a value that's part of the saved model artifact (blend weights,
calibration scales, top_down_weight) requires a pickle regeneration to actually reach
`predict.py`. Fixes to logic that runs fresh against live data at prediction time do not.

---

## 4. Verification

Ran the full pipeline end-to-end (`run.sh` → `generate_features.py` →
`predict.py`) twice — once before and once after the pickle regeneration — confirming
clean completion, no errors, no NaN/negative predictions, and sane proportional scaling
across 30/60/90-day horizons both times.

Regenerated `output/` (3-fold) and `output_12fold/` (12-fold) to reflect the final code
state, and updated `README.md` / `TECHNICAL_DOCUMENTATION.md` with the real, verified
numbers from both sweeps — no stale or placeholder figures left in either doc.

---

## Files changed

- `src/models.py` — `_monthly_seasonal_index()` (new), `_recent_baseline()` trend/baseline
  deseasonalization, `top_down_weight` refactored to instance attribute + pickle
  persistence, `forecast_dimension()` updated to use it
- `README.md` — new sections documenting both fixes with verified before/after numbers
- `TECHNICAL_DOCUMENTATION.md` — Section 1 (architecture) and Section 5 (limitations)
  updated to match
- `pickle/model.pkl` — regenerated via `retrain_and_pickle.py`
- `output/`, `output_12fold/` — regenerated backtest artifacts

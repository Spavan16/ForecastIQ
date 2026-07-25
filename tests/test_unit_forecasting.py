"""
Unit tests for the core forecasting invariants, as opposed to the end-to-end smoke test in
test_run_pipeline.py.

Code audit (High Issue #6): the only test coverage in the repo was a single end-to-end
smoke test exercising the run.sh contract. That catches "does the pipeline crash," but not
"is the math inside it actually correct in isolation" -- e.g. a P10 > P50 violation could be
masked in the full pipeline by real data happening to avoid the edge case, and wouldn't be
caught until unseen holdout data hits it differently. These tests exercise the two
functions most load-bearing for the project's own headline claims (the P10/P50/P90 interval
math models.py uses in every forecast, and the naive-baseline calc evaluation.py uses as the
project's own "prove the ensemble earns its complexity" bar) directly, with synthetic inputs
where the correct answer is known exactly -- not inferred from real data.

Same no-pytest, plain-assert convention as test_run_pipeline.py (requirements.txt is locked
this close to the deadline; not worth a new pinned dependency for two test files). Run
directly:

    python tests/test_unit_forecasting.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import EnsembleForecaster
from src.evaluation import _naive_baseline_value


# ---------------------------------------------------------------------------
# _aggregate_probabilistic_sums (src/models.py) -- the P10/P50/P90 interval math
# ---------------------------------------------------------------------------

def test_p10_p50_p90_ordering_holds_across_scales_and_horizons():
    """P10 <= P50 <= P90 must hold for every horizon and every interval_calibration_scale,
    not just the values that happen to appear in real backtest data."""
    forecaster = EnsembleForecaster()
    rng = np.random.default_rng(42)
    for scale in [0.5, 1.0, 3.0, 10.0]:
        forecaster.interval_calibration_scale = scale
        for periods in [30, 60, 90]:
            for _ in range(20):
                daily_preds = rng.uniform(0, 50000, size=periods)
                std_daily = float(rng.uniform(0, 20000))
                p10, p50, p90 = forecaster._aggregate_probabilistic_sums(daily_preds, std_daily, periods)
                assert p10 <= p50 + 1e-6, f"p10 > p50 at scale={scale}, periods={periods}: {p10} > {p50}"
                assert p50 <= p90 + 1e-6, f"p50 > p90 at scale={scale}, periods={periods}: {p50} > {p90}"
                assert p10 >= 0.0, f"p10 went negative: {p10}"
    print("PASS: P10 <= P50 <= P90 holds across 4 scales x 3 horizons x 20 random draws each.")


def test_p10_floor_engages_when_interval_is_wide():
    """When horizon_std is large relative to p50, p10 should be floored at 5% of p50 rather
    than going to zero or negative -- this is the documented, intentional floor (see the
    Rolling-Origin Backtesting section of README.md), not a bug. Verifying it actually
    engages, since the P10 floor is also the disclosed reason 8-fold interval coverage
    can't be pushed further without a separate fix."""
    forecaster = EnsembleForecaster()
    forecaster.interval_calibration_scale = 3.0
    p50_expected = 100_000.0
    daily_preds = np.full(30, p50_expected / 30.0)
    huge_std = 50_000.0  # deliberately large enough that the raw P10 would go negative
    p10, p50, p90 = forecaster._aggregate_probabilistic_sums(daily_preds, huge_std, 30)
    assert abs(p50 - p50_expected) < 1.0, f"p50 sum doesn't match input: {p50} vs {p50_expected}"
    assert abs(p10 - p50_expected * 0.05) < 1.0, (
        f"P10 floor didn't engage as documented: expected {p50_expected * 0.05}, got {p10}"
    )
    print("PASS: P10 floor (5% of P50) engages correctly under a wide interval.")


def test_zero_std_collapses_interval_to_point_estimate():
    """A zero-uncertainty input (std_daily=0) should produce p10 == p50 == p90 -- no interval
    math should manufacture spread that isn't in the input."""
    forecaster = EnsembleForecaster()
    daily_preds = np.full(30, 1000.0)
    p10, p50, p90 = forecaster._aggregate_probabilistic_sums(daily_preds, 0.0, 30)
    assert abs(p10 - p50) < 1e-6 and abs(p90 - p50) < 1e-6, (
        f"Zero std should collapse the interval, got p10={p10}, p50={p50}, p90={p90}"
    )
    print("PASS: zero std collapses P10/P50/P90 to a single point estimate.")


# ---------------------------------------------------------------------------
# _naive_baseline_value (src/evaluation.py) -- the project's own honesty benchmark
# ---------------------------------------------------------------------------

def _make_daily_df(dates, revenue, spend):
    return pd.DataFrame({"date": pd.to_datetime(dates), "revenue": revenue, "spend": spend})


def test_naive_baseline_is_a_pure_flat_trailing_mean():
    """The naive baseline's own docstring promises 'no model, no seasonality, just the
    trailing window average projected flat.' Verifying that promise with a synthetic
    dataset where the correct flat-projection answer is known exactly by construction,
    not inferred from real data where the correct answer isn't independently known."""
    origin = pd.Timestamp("2026-04-01")
    dates = pd.date_range(origin - pd.Timedelta(days=30), periods=30, freq="D")
    # Constant $1000/day revenue, $200/day spend for the full trailing window.
    df = _make_daily_df(dates, [1000.0] * 30, [200.0] * 30)

    rev_30 = _naive_baseline_value(df, origin, horizon_days=30, metric="Revenue")
    rev_90 = _naive_baseline_value(df, origin, horizon_days=90, metric="Revenue")
    roas = _naive_baseline_value(df, origin, horizon_days=30, metric="ROAS")

    assert rev_30 is not None and abs(rev_30 - 30_000.0) < 1e-6, f"30-day naive revenue wrong: {rev_30}"
    assert rev_90 is not None and abs(rev_90 - 90_000.0) < 1e-6, (
        f"90-day naive revenue should scale linearly with horizon_days (flat projection): {rev_90}"
    )
    assert roas is not None and abs(roas - 5.0) < 1e-6, f"naive ROAS wrong: {roas} (expected 1000/200=5.0)"
    print("PASS: naive baseline is a pure flat trailing-30-day projection, exactly as documented.")


def test_naive_baseline_ignores_data_outside_the_trailing_window():
    """A spike sitting just before the trailing window (e.g. day 31-40 before origin) must
    NOT leak into the naive baseline -- if it did, the naive baseline would no longer be the
    honest, dumb comparison point the project's own methodology relies on."""
    origin = pd.Timestamp("2026-04-01")
    old_spike_dates = pd.date_range(origin - pd.Timedelta(days=40), periods=10, freq="D")
    recent_dates = pd.date_range(origin - pd.Timedelta(days=30), periods=30, freq="D")
    df = pd.concat([
        _make_daily_df(old_spike_dates, [1_000_000.0] * 10, [1.0] * 10),
        _make_daily_df(recent_dates, [500.0] * 30, [100.0] * 30),
    ], ignore_index=True)

    rev_30 = _naive_baseline_value(df, origin, horizon_days=30, metric="Revenue")
    assert rev_30 is not None and abs(rev_30 - 15_000.0) < 1e-6, (
        f"Old spike leaked into the trailing-30-day baseline: got {rev_30}, expected 15000.0 "
        f"(500/day x 30, ignoring the spike outside the window)"
    )
    print("PASS: naive baseline correctly ignores data outside its trailing window.")


def test_naive_baseline_returns_none_on_empty_trailing_window():
    """No history before the origin -> None, not a crash or a silent zero that could be
    mistaken for a real (if unlikely) zero-revenue baseline."""
    origin = pd.Timestamp("2026-04-01")
    df = _make_daily_df(pd.date_range("2020-01-01", periods=5, freq="D"), [100.0] * 5, [10.0] * 5)
    result = _naive_baseline_value(df, origin, horizon_days=30, metric="Revenue")
    assert result is None, f"Expected None for an empty trailing window, got {result}"
    print("PASS: naive baseline returns None (not a crash or a silent zero) when there's no trailing history.")


if __name__ == "__main__":
    test_p10_p50_p90_ordering_holds_across_scales_and_horizons()
    test_p10_floor_engages_when_interval_is_wide()
    test_zero_std_collapses_interval_to_point_estimate()
    test_naive_baseline_is_a_pure_flat_trailing_mean()
    test_naive_baseline_ignores_data_outside_the_trailing_window()
    test_naive_baseline_returns_none_on_empty_trailing_window()
    print("\nAll unit tests passed.")

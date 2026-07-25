"""
Smoke test for the core run.sh pipeline (generate_features.py -> predict.py).

Code audit (High Issue #6) flagged zero automated tests anywhere in the repo, despite
a documented history of bugs found via manual forensic audit -- exactly the failure mode
a cheap smoke test exists to catch.

No pytest dependency added on purpose (requirements.txt is pinned and kept minimal --
not worth the risk of a version mismatch on the evaluation environment for one
test file). Plain asserts, run directly:

    python tests/test_run_pipeline.py

Exits non-zero on any failure, same "fail loudly" contract as run.sh itself.
"""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "output" / "predictions.csv"
CAUSAL_PATH = ROOT / "output" / "causal_summary.json"

EXPECTED_COLUMNS = ["forecast_period", "dimension_type", "dimension_value", "metric", "p10", "p50", "p90"]
EXPECTED_PERIODS = {"30_days", "60_days", "90_days"}
EXPECTED_METRICS = {"Revenue", "ROAS"}

# BUG fix (found while writing this test): passing env={"PYTHONPATH": ...} to subprocess.run
# REPLACES the entire environment instead of extending it, wiping PATH/SYSTEMROOT/etc. On
# Windows this breaks native DLL loading for xgboost/sklearn (import crashes deep inside
# compiled extensions). Copy the real environment and only add PYTHONPATH on top of it.
RUN_ENV = os.environ.copy()
RUN_ENV["PYTHONPATH"] = str(ROOT)


def run(cmd):
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT, env=RUN_ENV)
    assert result.returncode == 0, f"Command failed: {' '.join(cmd)}"


def test_pipeline_runs_and_produces_valid_output():
    run([sys.executable, "src/generate_features.py", "--data-dir", "./data", "--out", "features.pkl"])
    run([sys.executable, "src/predict.py", "--features", "features.pkl", "--model", "./pickle/model.pkl",
         "--output", "./output/predictions.csv"])

    assert OUTPUT_PATH.exists(), "predictions.csv was not created"
    df = pd.read_csv(OUTPUT_PATH)

    assert not df.empty, "predictions.csv is empty"
    assert list(df.columns) == EXPECTED_COLUMNS, f"Schema mismatch: {list(df.columns)}"
    assert set(df["forecast_period"].unique()) == EXPECTED_PERIODS, "Missing a forecast window"
    assert set(df["metric"].unique()) == EXPECTED_METRICS, "Missing a metric"
    assert df["dimension_type"].isin(["Overall", "Channel", "CampaignType", "Campaign"]).all()

    # P10 <= P50 <= P90 must hold for every row -- a violation here means the interval
    # math is broken, not just imprecise.
    assert (df["p10"] <= df["p50"] + 1e-6).all(), "Found a row where p10 > p50"
    assert (df["p50"] <= df["p90"] + 1e-6).all(), "Found a row where p50 > p90"

    assert CAUSAL_PATH.exists(), "causal_summary.json was not created (required AI deliverable)"
    print("PASS: predictions.csv and causal_summary.json both valid.")


def test_fails_loudly_on_missing_model():
    result = subprocess.run(
        [sys.executable, "src/predict.py", "--features", "features.pkl",
         "--model", "./pickle/does_not_exist.pkl", "--output", "./output/should_not_exist.csv"],
        cwd=ROOT, env=RUN_ENV,
    )
    assert result.returncode != 0, "predict.py should exit non-zero on a missing model pickle"
    assert not (ROOT / "output" / "should_not_exist.csv").exists(), (
        "predict.py silently produced output despite a missing model -- contract violation"
    )
    print("PASS: missing model.pkl fails loudly, no bad output written.")


def test_fails_loudly_on_corrupt_model():
    # Code audit (High Issue #5, follow-up): the missing-path case above doesn't exercise a
    # model.pkl that EXISTS but fails to unpickle -- a version mismatch between the environment
    # it was pickled in and the evaluation environment, which the project spec calls "the single
    # most common reason submissions fail." This writes garbage bytes to a real file at the
    # expected path so load_models()'s pickle.load() itself throws, not just a missing-file check.
    corrupt_path = ROOT / "pickle" / "corrupt_test.pkl"
    corrupt_path.write_bytes(b"not a real pickle file")
    try:
        result = subprocess.run(
            [sys.executable, "src/predict.py", "--features", "features.pkl",
             "--model", str(corrupt_path), "--output", "./output/should_not_exist2.csv"],
            cwd=ROOT, env=RUN_ENV,
        )
        assert result.returncode != 0, "predict.py should exit non-zero on a corrupt model pickle"
        assert not (ROOT / "output" / "should_not_exist2.csv").exists(), (
            "predict.py silently produced output despite a corrupt/unloadable model -- contract violation"
        )
        print("PASS: corrupt model.pkl fails loudly, no bad output written.")
    finally:
        corrupt_path.unlink(missing_ok=True)


if __name__ == "__main__":
    test_pipeline_runs_and_produces_valid_output()
    test_fails_loudly_on_missing_model()
    test_fails_loudly_on_corrupt_model()
    print("\nAll smoke tests passed.")

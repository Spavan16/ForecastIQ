#!/usr/bin/env bash
# ==============================================================================
# ForecastIQ — Marketing Revenue Forecasting & Budget Optimization Pipeline
# Author: Pavan Kumar S
# Contract: ./run.sh <DATA_DIR> <MODEL_PATH> <OUTPUT_PATH>
# ==============================================================================

set -euo pipefail

# BUG 8 fix (CRITICAL): PYTHONPATH="." resolves relative to the shell's CWD at invocation
# time, not this script's location. Confirmed failure mode: running this script via an
# absolute path from a different CWD throws `ModuleNotFoundError: No module named 'src'`.
# Anchor everything to the script's own directory instead, and cd into it so the relative
# defaults below (./data, ./pickle, ./output, features.pkl) resolve correctly regardless
# of where the caller invokes run.sh from.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Mute warnings and set Python path
export PYTHONWARNINGS="ignore"
export PYTHONPATH="$SCRIPT_DIR"

# Detect the correct Python interpreter. Linux/Mac venvs (and the python.org installer on
# Windows) typically provide a `python3` binary, but Windows' built-in `python -m venv`
# only ever creates python.exe — python3.exe does not exist there at all, so a hardcoded
# `python3` call fails outright on a stock Windows venv (confirmed: this venv's Scripts/
# folder has python.exe but no python3.exe). Prefer python3 where it exists (keeps Linux/Mac
# behavior unchanged), fall back to python otherwise.
if command -v python3 >/dev/null 2>&1 && python3 -c "import xgboost, lightgbm, catboost, prophet, optuna, shap, reportlab" 2>/dev/null; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1 && python -c "import xgboost, lightgbm, catboost, prophet, optuna, shap, reportlab" 2>/dev/null; then
    PYTHON_BIN="python"
else
    echo "ERROR: No Python interpreter with required packages found on PATH (checked python3, python)."
    echo "Run: pip install -r requirements.txt"
    exit 1
fi

# Verify required dependencies are installed — fail loudly if not (no runtime internet calls)
if ! "$PYTHON_BIN" -c "import xgboost, lightgbm, catboost, prophet, optuna, shap, reportlab" 2>/dev/null; then
    echo "ERROR: One or more required dependencies are missing."
    echo "Run: pip install -r requirements.txt"
    exit 1
fi

# Accept arguments with highly robust defaults for standalone local runs
DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

echo "=============================================================================="
echo "ForecastIQ Evaluation Pipeline"
echo "DATA_DIR:    $DATA_DIR"
echo "MODEL_PATH:  $MODEL_PATH"
echo "OUTPUT_PATH: $OUTPUT_PATH"
echo "=============================================================================="

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT_PATH")"

# 1. Generate the advanced time, performance, marketing, and channel features
echo "[1/2] Ingesting analytics datasets & generating multi-dimensional features..."
"$PYTHON_BIN" src/generate_features.py \
    --data-dir "$DATA_DIR" \
    --out features.pkl

# 2. Load the pickled weighted ensemble and produce probabilistic P10-P50-P90 forecasts
echo "[2/2] Loading pickled ensemble models & computing probabilistic predictions..."
"$PYTHON_BIN" src/predict.py \
    --features features.pkl \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_PATH"

echo "=============================================================================="
echo "Done. Predictions successfully written to $OUTPUT_PATH"
echo "=============================================================================="

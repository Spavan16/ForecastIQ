"""
ForecastIQ — Retrain & Re-Pickle Script
Run from project root: python retrain_and_pickle.py
Retrains the full ensemble on data/ and saves a fresh pickle/model.pkl
with the new top_campaign_names field baked in.
"""
import sys
import os
# See src/predict.py for the full explanation: skips a joblib/loky Windows subprocess
# probe that can print a non-fatal but console-alarming traceback on restricted systems.
# This script trains the full ensemble (heaviest joblib usage in the codebase), so it's
# the most likely place this would otherwise fire.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["PYTHONWARNINGS"] = "ignore"

print("=" * 60)
print("ForecastIQ — Full Retrain & Re-Pickle")
print("=" * 60)

# Step 1: Ingest & validate data
print("\n[1/4] Ingesting and validating data...")
from src.validation import ValidationEngine
val_engine = ValidationEngine(data_dir=ROOT / "data")
df, summary = val_engine.run_full_ingestion()
print(f"      Records: {len(df)} | Quality Score: {summary['data_quality_score']}/100")

# Step 2: Build daily aggregate for overall models
print("\n[2/4] Building feature matrix...")
import pandas as pd
from src.features import FeatureEngineer

# BUG 1 fix: previously rebuilt only 6 basic time features here inline, bypassing
# FeatureEngineer entirely — the rolling windows / ROAS / CPC / CTR / volatility / trend
# features it computes were never fed into training. Fix: use
# FeatureEngineer.create_daily_aggregate_features() as the single source of truth for the
# overall model's feature matrix (already includes month/quarter/week/day_of_week/
# is_weekend/season_encoded plus the full engineered set), instead of redundantly
# rederiving a subset of it by hand.
feat_engine = FeatureEngineer()
daily = feat_engine.create_daily_aggregate_features(df)
print(f"      Daily rows: {len(daily)} | Date range: {daily['date'].min().date()} -> {daily['date'].max().date()}")
print(f"      Feature columns: {len(daily.columns) - 1} (incl. rolling windows, ROAS, CPC, CTR, volatility, trend)")

# Step 3: Train full ensemble
print("\n[3/4] Training full ensemble (Prophet + XGBoost + LightGBM + CatBoost)...")
from src.models import EnsembleForecaster
forecaster = EnsembleForecaster(model_path=ROOT / "pickle" / "model.pkl")
forecaster.train_overall_models(daily)
forecaster.train_dimension_models(df)

# Confirm top_campaign_names populated
print(f"      top_campaign_names ({len(forecaster.top_campaign_names)}): {forecaster.top_campaign_names[:5]}")

# Step 4: Save pickle
print("\n[4/4] Saving fresh model.pkl...")
forecaster.save_models()

# Verify
import pickle
with open(ROOT / "pickle" / "model.pkl", "rb") as f:
    artifact = pickle.load(f)

has_campaigns = "top_campaign_names" in artifact
campaign_count = len(artifact.get("top_campaign_names", []))
model_keys = [k for k in artifact.get("models", {}).keys() if not k.startswith("_")]

print("\n" + "=" * 60)
print("RETRAIN COMPLETE")
print(f"  top_campaign_names in artifact : {has_campaigns} ({campaign_count} campaigns)")
print(f"  Model keys count               : {len(model_keys)}")
print(f"  Pickle path                    : {ROOT / 'pickle' / 'model.pkl'}")
print("=" * 60)

# Quick sanity — run a forecast to confirm nothing is broken
print("\nRunning quick sanity forecast...")
forecaster2 = EnsembleForecaster(model_path=ROOT / "pickle" / "model.pkl")
forecaster2.load_models()
start = daily['date'].max() + pd.Timedelta(days=1)
result = forecaster2.forecast_overall(start)
p50_30 = result['30_days']['Revenue_P50']
p50_90 = result['90_days']['Revenue_P50']
print(f"  30-day P50 Revenue : ${p50_30:,.0f}")
print(f"  90-day P50 Revenue : ${p50_90:,.0f}")
print("\nAll good. Commit pickle/model.pkl to git before committing.")

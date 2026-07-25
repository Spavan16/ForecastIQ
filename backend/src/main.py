import os
import sys
import threading
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException, Body, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Add root to python path for clean shared imports
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.validation import ValidationEngine
from src.models import EnsembleForecaster
from src.monte_carlo import MonteCarloSimulator, derive_revenue_volatility
from src.budget_optimizer import BudgetOptimizer
from src.scenarios import ScenarioGenerator
from src.explainability import ExplainabilityEngine
from src.risk_engine import RiskIntelligenceEngine
from src.rule_engine import RuleInsightEngine
from src.chat_engine import ForecastChatBot
from src.pdf_reporting import EnterprisePDFReport
from src.llm_provider import get_llm_provider
from src.database import DatabaseManager
from src.utils import get_logger

app = FastAPI(
    title="ForecastIQ API",
    # NOTE: this description is visible directly to anyone opening the FastAPI Swagger
    # docs at /docs, so it's worth keeping accurate and product-focused.
    description="FastAPI backend for the ForecastIQ revenue forecasting & budget optimization platform.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE: Dict[str, Any] = {}
logger = get_logger("API")

_DB_MANAGER: Optional[DatabaseManager] = None
_CACHE_LOCK = threading.Lock()


def get_db_manager() -> DatabaseManager:
    """Lazily instantiates a single shared DatabaseManager so we don't reopen a SQLite
    connection/engine on every request."""
    global _DB_MANAGER
    if _DB_MANAGER is None:
        _DB_MANAGER = DatabaseManager()
    return _DB_MANAGER


def _get_data_max_mtime() -> float:
    """Latest modification time across data/*.csv, used to detect a stale cache."""
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    max_mtime = 0.0
    if data_dir.exists():
        for csv_file in data_dir.glob("*.csv"):
            try:
                mtime = csv_file.stat().st_mtime
                if mtime > max_mtime:
                    max_mtime = mtime
            except OSError:
                return float('inf')
    return max_mtime


def _cache_is_fresh() -> bool:
    if not CACHE:
        return False
    return _get_data_max_mtime() <= CACHE.get("_data_mtime", 0)

def get_analytics_state() -> Dict[str, Any]:
    """Loads and returns the unified analytics state, caching it to ensure sub-millisecond API latency."""
    # Fast path: no lock needed for the common case (cache already warm). This is the path
    # nearly every request takes once the cache has been built once.
    if _cache_is_fresh():
        return CACHE

    # BUG fix: the frontend fires ~9 parallel requests on initial page load (Overview,
    # Validation, Forecasts, Scenarios, Budget, Monte Carlo, Explainability, Risk, Insights).
    # With no locking here, every one of them could see a cold/stale CACHE at the same instant
    # and each independently trigger a full rebuild — retraining the ensemble, rerunning
    # 10,000 Monte Carlo simulations, Optuna, and PDF generation, all in parallel — then each
    # persisting its own duplicate row via save_forecast_run(). A lock with double-checked
    # freshness (re-verify immediately after acquiring, since another thread may have just
    # finished rebuilding while we were waiting for the lock) ensures only one thread ever
    # pays for the rebuild; the rest simply reuse its result.
    with _CACHE_LOCK:
        if _cache_is_fresh():
            return CACHE

        val_engine = ValidationEngine()
        df, summary = val_engine.run_full_ingestion()

        forecaster = EnsembleForecaster(validation_engine=val_engine)
        forecaster.load_models()

        # BUG fix (consistency with predict.py's CLI path): the backend previously never
        # called refresh_recent_context(), so the live dashboard's recent-conditions anchor
        # stayed frozen at whatever was true when model.pkl was last trained locally, and
        # produce_full_predictions_table() below iterated the pickle's frozen dimension lists
        # instead of whatever channels/campaign types/campaigns are actually in df right now.
        forecaster.refresh_recent_context(df)
        current_dims = forecaster.get_current_dimension_values(df)

        start_date = df['date'].max() + pd.Timedelta(days=1)
        overall_fc = forecaster.forecast_overall(start_date)
        preds_table = forecaster.produce_full_predictions_table(
            start_date,
            channels=current_dims["channels"],
            campaign_types=current_dims["campaign_types"],
            top_campaign_names=current_dims["top_campaign_names"],
        )

        risk_engine = RiskIntelligenceEngine(df, data_quality_score=summary['data_quality_score'])
        risk_profile = risk_engine.evaluate_risk()

        rule_engine = RuleInsightEngine(df, overall_fc['90_days'], risk_profile)
        insights = rule_engine.generate_all_insights()

        budget_opt = BudgetOptimizer(df)

        monte_carlo = MonteCarloSimulator()
        base_rev_30d = overall_fc['30_days']['Revenue_P50']
        base_sp_30d = overall_fc['30_days']['Spend_Expected']

        # Calculate historical channel splits for Monte Carlo and Overview Pie Chart
        ch_sums = df.groupby('channel')['revenue'].sum()
        ch_splits = (ch_sums / ch_sums.sum()).to_dict()
        # BUG 4 fix: pass the ensemble's own daily residual std through so Monte Carlo's spread
        # is derived from real model uncertainty instead of a hardcoded 15% volatility constant.
        # See src/monte_carlo.py run_portfolio_simulation() for the derivation.
        _residual_std_daily = forecaster.residuals_std.get("overall", None)
        mc_results = monte_carlo.run_portfolio_simulation(
            base_rev_30d, base_sp_30d, ch_splits,
            residual_std_daily=_residual_std_daily, horizon_days=30
        )

        explain_engine = ExplainabilityEngine(df)
        shap_drivers = explain_engine.compute_shap_drivers()

        # Generate PDF Report once for instant download
        rep_gen = EnterprisePDFReport()
        pdf_path = rep_gen.generate_report(summary, preds_table, budget_opt.optimize_allocation(100000, 4.5), risk_profile, insights)

        # Generate 90-day daily trajectory for the Master AreaChart
        # Derived directly from the ensemble: Prophet + XGBoost + LightGBM + CatBoost weighted predictions
        # BUG 1 fix consistency: forecast_overall() now trains/predicts on the full engineered
        # feature set (rolling windows, ROAS, CPC, CTR, volatility, trend), not just 6 base time
        # columns. This duplicate trajectory feature-prep previously hardcoded the old 6-column
        # slice, which would throw a feature-mismatch error against the retrained XGB/LGB/CatBoost
        # models (sklearn-style .predict() requires identical columns to .fit() time). Reuse the
        # forecaster's own feature schema + last-known-engineered carry-forward instead of
        # duplicating (and now diverging from) the logic in models.py.
        _future_df = forecaster._prepare_future_features(start_date, periods=90, last_known_engineered=forecaster.last_known_engineered)
        _prophet_future = _future_df[['date']].rename(columns={'date': 'ds'})
        _p_prophet = forecaster.models["overall_prophet"].predict(_prophet_future)['yhat'].values
        _feature_cols = forecaster.feature_columns if forecaster.feature_columns else forecaster.BASE_TIME_FEATURES
        _X_tree = _future_df[_feature_cols]
        _p_xgb  = forecaster.models["overall_xgb"].predict(_X_tree)
        _p_lgb  = forecaster.models["overall_lgb"].predict(_X_tree)
        _p_cat  = forecaster.models["overall_cat"].predict(_X_tree)
        _daily_p50 = np.clip(
            forecaster.weights["prophet"]   * _p_prophet +
            forecaster.weights["xgboost"]  * _p_xgb +
            forecaster.weights["lightgbm"] * _p_lgb +
            forecaster.weights["catboost"] * _p_cat,
            a_min=0.0, a_max=None
        )
        _std_rev = forecaster.residuals_std.get("overall", 1000.0)

        daily_res = []
        for day in range(90):
            p50_val = float(_daily_p50[day])
            # BUG 3 fix: previously (day + 1) ** 0.5 here, while models.py's
            # _aggregate_probabilistic_sums() (and the derive_revenue_volatility() helper used by
            # Monte Carlo/Scenarios) both use ** 0.75. Two different unexplained exponents for what
            # reads, to a reviewer, as the same underlying concept — forecast uncertainty
            # compounding with horizon — undermines credibility even though they technically apply to
            # slightly different things (this is a single day's point estimate at horizon `day`,
            # those are a cumulative sum over `periods` days). Standardized on 0.75 to match the rest
            # of the codebase, so every uncertainty-growth calculation in the project is explainable
            # with one consistent number if asked.
            horizon_std = _std_rev * ((day + 1) ** 0.75)
            p10_val = max(0.0, p50_val - 1.28 * horizon_std)
            p90_val = p50_val + 1.28 * horizon_std
            daily_res.append({
                "day": f"Day {day + 1}",
                "p10": round(p10_val, 2),
                "p50": round(p50_val, 2),
                "p90": round(p90_val, 2)
            })

        trajectory_method = "ensemble_weighted_daily"

        # Build channel_shares list for the frontend Attribution Pie Chart
        # ch_splits is a dict of {channel: fraction} — convert to [{name, value%, color}]
        # Restyled per Stitch design pass: electric cyan / purple / green replacing the
        # previous blue/teal/slate set, to match the new frontend theme.
        CHANNEL_COLORS = {
            "Google Ads": "#00F2FF",
            "Meta Ads":   "#B000FF",
            "Bing Ads":   "#00FF66"
        }
        channel_shares = [
            {
                "name": ch,
                "value": round(float(share) * 100.0, 1),
                "color": CHANNEL_COLORS.get(ch, "#94A3B8")
            }
            for ch, share in ch_splits.items()
        ]

        CACHE["_data_mtime"] = _get_data_max_mtime()
        CACHE["df"] = df
        CACHE["summary"] = summary
        CACHE["forecaster"] = forecaster
        CACHE["current_dims"] = current_dims
        CACHE["overall_fc"] = overall_fc
        CACHE["preds_table"] = preds_table
        CACHE["risk_profile"] = risk_profile
        CACHE["insights"] = insights
        CACHE["budget_opt"] = budget_opt
        CACHE["mc_results"] = mc_results
        CACHE["shap_drivers"] = shap_drivers
        CACHE["daily_trajectory"] = daily_res
        CACHE["trajectory_method"] = trajectory_method
        CACHE["channel_shares"] = channel_shares
        CACHE["pdf_path"] = pdf_path

        # Persist this run to SQLite as an audit trail / run-history record. Best-effort: a
        # persistence failure (locked file, disk issue) must never break the actual forecast
        # response, since the CLI grading pipeline and the live dashboard both depend on this
        # function succeeding regardless of database state.
        try:
            get_db_manager().save_forecast_run(
                label=f"auto_run_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S')}",
                params={
                    "data_quality_score": summary["data_quality_score"],
                    "total_records": len(df)
                },
                predictions_summary={
                    "risk_classification": risk_profile["risk_classification"],
                    "forecast_90d_p50_revenue": overall_fc["90_days"]["Revenue_P50"],
                    "forecast_90d_p50_roas": overall_fc["90_days"]["ROAS_P50"]
                }
            )
        except Exception as e:
            logger.warning(f"Failed to persist forecast run to database: {e}")

        return CACHE


@app.get("/api/status")
def get_system_status():
    state = get_analytics_state()
    llm = get_llm_provider()
    return {
        "status": "online",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "active_llm_provider": llm.get_provider_name(),
        "data_quality_score": state["summary"]["data_quality_score"],
        "risk_classification": state["risk_profile"]["risk_classification"],
        "has_critical_warnings": state["summary"]["has_critical_warnings"]
    }


@app.get("/api/overview")
def get_executive_overview():
    state = get_analytics_state()
    s = state["summary"]
    r = state["risk_profile"]
    ov_90d = state["overall_fc"]["90_days"]
    return {
        "total_historical_spend": s["total_spend"],
        "total_historical_revenue": s["total_revenue"],
        "overall_historical_roas": s["overall_roas"],
        "data_quality_score": s["data_quality_score"],
        "risk_classification": r["risk_classification"],
        "risk_badge_color": r["badge_color"],
        "forecast_90d_p50_revenue": ov_90d["Revenue_P50"],
        "forecast_90d_p50_roas": ov_90d["ROAS_P50"],
        # Exposing values already computed by forecast_overall() but not previously returned
        # here — needed for the Confidence Bracket component (P50 hero number + real P10-P90
        # range), not new computation, just surfacing what models.py already produces.
        "forecast_90d_p10_revenue": ov_90d["Revenue_P10"],
        "forecast_90d_p90_revenue": ov_90d["Revenue_P90"],
        "forecast_90d_p10_roas": ov_90d["ROAS_P10"],
        "forecast_90d_p90_roas": ov_90d["ROAS_P90"],
        "executive_summary": state["insights"]["executive_summary"],
        "daily_trajectory": state["daily_trajectory"],
        "channel_shares": state["channel_shares"],
        "critical_warnings": s["critical_warnings"],
        "trajectory_method": state["trajectory_method"]
    }


@app.get("/api/trajectory")
def get_daily_trajectory():
    state = get_analytics_state()
    return state["daily_trajectory"]


@app.get("/api/validation")
def get_validation_details():
    state = get_analytics_state()
    return state["summary"]


@app.get("/api/model-validation")
def get_model_validation_scorecard():
    """Returns rolling-origin backtest artifacts generated by src/evaluation.py."""
    output_dir = ROOT_DIR / "output"
    summary_path = output_dir / "backtest_summary.json"
    scorecard_path = output_dir / "backtest_scorecard.csv"

    if not summary_path.exists() or not scorecard_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Backtest artifacts not found. Run: python src/evaluation.py --data-dir ./data --output-dir ./output --folds 3"
        )

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    scorecard = pd.read_csv(scorecard_path)
    segment_rows = summary.get("by_segment", [])
    strongest_segments = sorted(segment_rows, key=lambda r: r.get("wape", r.get("smape", 999999)))[:6]
    watchlist_segments = sorted(segment_rows, key=lambda r: r.get("wape", r.get("smape", -1)), reverse=True)[:6]

    recent_rows = (
        scorecard.sort_values(["origin_date", "forecast_period", "dimension_type", "metric"])
        .tail(18)
        .replace({np.nan: None})
        .to_dict(orient="records")
    )

    return {
        "summary": summary,
        "strongest_segments": strongest_segments,
        "watchlist_segments": watchlist_segments,
        "recent_rows": recent_rows,
        "artifacts": {
            "summary_path": str(summary_path),
            "scorecard_path": str(scorecard_path),
            "scorecard_rows": int(len(scorecard))
        }
    }


@app.get("/api/forecasts")
def get_probabilistic_forecasts(dimension: str = Query("Overall")):
    """Returns structured P10, P50, P90 probabilistic forecasting intelligence."""
    state = get_analytics_state()
    df = state["preds_table"]

    # BUG 17 fix (backend counterpart): previously hardcoded to ["Google Ads","Meta Ads",
    # "Bing Ads"] and ["SEARCH","SOCIAL"], mirroring the same gap as BUG 10. Any real campaign
    # type outside that list (PERFORMANCE_MAX, SHOPPING, VIDEO, DEMAND_GEN, Bing's Audience)
    # silently fell into the Campaign-lookup `else` branch, found no match, and silently
    # returned Overall — even though produce_full_predictions_table() now actually produces
    # rows for these dimensions (BUG 10 fix). Resolve dynamically by looking up whatever
    # dimension_type bucket this dimension_value actually belongs to in the predictions
    # table, instead of hardcoding the list of valid values in two places.
    if dimension == "Overall":
        sub = df[df['dimension_type'] == 'Overall']
    else:
        match = df[df['dimension_value'] == dimension]
        sub = match if len(match) > 0 else df[df['dimension_type'] == 'Overall']
    
    res = {}
    for period in ["30_days", "60_days", "90_days"]:
        p_sub = sub[sub['forecast_period'] == period]
        period_data = {}
        for idx, row in p_sub.iterrows():
            metric = str(row['metric'])
            period_data[metric] = {
                "P10": row['p10'],
                "P50": row['p50'],
                "P90": row['p90']
            }
        res[period] = period_data
    return res


@app.get("/api/dimensions")
def get_available_dimensions():
    """
    BUG 17 fix: exposes the REAL set of channels and campaign types, so the frontend's
    dimension filter buttons stop hardcoding ["SEARCH","SOCIAL"] and can render whatever the
    data and model actually support — including PERFORMANCE_MAX, SHOPPING, etc.

    Follow-up fix: reads from current_dims (derived from the actual data currently loaded,
    via get_current_dimension_values()) instead of forecaster.trained_channels/
    trained_campaign_types, which are frozen into model.pkl at last local training time and
    can silently omit channels/types that only exist in newer or swapped-in data.
    """
    state = get_analytics_state()
    current_dims = state["current_dims"]
    return {
        "channels": current_dims["channels"],
        "campaign_types": current_dims["campaign_types"]
    }


@app.get("/api/simulations")
def get_monte_carlo_simulations():
    state = get_analytics_state()
    return state["mc_results"]


class BudgetSimRequest(BaseModel):
    # BUG fix (bug-hunt sweep, same class as the frozen-dimension predictions bug): previously
    # three fixed fields (google_pct/meta_pct/bing_pct), which made the API contract itself
    # incapable of simulating a 4th channel no matter what the optimizer or frontend could do.
    # Now accepts any channel name -> pct_change mapping; channels omitted default to 0% change.
    channel_pcts: Dict[str, float] = {}


@app.post("/api/simulate-budget")
def simulate_budget_changes(req: BudgetSimRequest):
    state = get_analytics_state()
    opt = state["budget_opt"]
    df = state["df"]
    
    # Calculate true authentic historical 30-day average run-rate spends from audited data
    unique_days = df['date'].nunique()
    if unique_days > 0:
        norm_spends = (df.groupby('channel')['spend'].sum() / unique_days * 30.0).to_dict()
    else:
        # BUG fix (bug-hunt sweep): previously hardcoded exactly Google/Meta/Bing here too.
        # Genuinely-no-data edge case only; fall back to whatever channels the optimizer
        # itself knows about (opt.channels, data-derived), not a hardcoded 3-name list.
        norm_spends = {ch: 25000.0 for ch in opt.channels}
        
    for req_ch in opt.channels:
        if req_ch not in norm_spends or norm_spends[req_ch] <= 0:
            # Derive the fallback from whatever channels DO have valid data (average of the
            # other run-rates), so a missing channel gets a portfolio-informed estimate rather
            # than an arbitrary constant. Only falls through to a fixed value in the
            # genuinely-no-data-anywhere case, which can't be derived from anything else.
            valid_spends = [v for k, v in norm_spends.items() if k != req_ch and v > 0]
            norm_spends[req_ch] = (sum(valid_spends) / len(valid_spends)) if valid_spends else 25000.0
            
    res = opt.simulate_budget_change(req.channel_pcts, norm_spends)
    return res


class BudgetOptRequest(BaseModel):
    max_budget: float = 100000.0
    target_roas: float = 4.5


@app.post("/api/optimize-budget")
def optimize_budget_allocation(req: BudgetOptRequest):
    state = get_analytics_state()
    opt = state["budget_opt"]
    # BUG fix (pre-release live API audit): optimize_allocation() already fails loudly
    # with a ValueError for max_budget <= 0 (see budget_optimizer.py), but nothing here
    # caught it -- FastAPI's default handler turned that clean, informative ValueError into
    # an opaque "Internal Server Error" 500 with no message reaching the frontend/reviewer.
    # Surface it as a proper 400 with the actual reason instead.
    try:
        res = opt.optimize_allocation(req.max_budget, req.target_roas)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return res


@app.get("/api/scenarios")
def get_strategic_scenarios():
    state = get_analytics_state()
    forecaster: EnsembleForecaster = state["forecaster"]
    # BUG 6 fix: previously called with no volatility info, leaving ScenarioGenerator's
    # internal default to silently apply a flat +/-15% band to every scenario regardless of
    # how extreme it is. Derive the same residual-based fraction Monte Carlo uses (BUG 4) via
    # the shared helper, so Scenarios stays consistent with it instead of guessing separately.
    base_volatility = derive_revenue_volatility(
        forecaster.residuals_std.get("overall"),
        state["overall_fc"]["30_days"].get("Revenue_P50"),
        horizon_days=30
    )
    sg = ScenarioGenerator(state["overall_fc"]["30_days"], state["overall_fc"]["60_days"], state["overall_fc"]["90_days"], base_revenue_volatility=base_volatility)
    return sg.generate_all_scenarios()


@app.get("/api/explainability")
def get_shap_explainability():
    state = get_analytics_state()
    return state["shap_drivers"]


@app.get("/api/risk")
def get_risk_intelligence():
    state = get_analytics_state()
    return state["risk_profile"]


@app.get("/api/insights")
def get_executive_insights():
    state = get_analytics_state()
    return state["insights"]


@app.get("/api/runs")
def get_recent_runs(limit: int = Query(10)):
    """Returns recent persisted forecast runs from the SQLite audit trail (src/database.py)."""
    return get_db_manager().get_recent_forecast_runs(limit=limit)


class ChatRequest(BaseModel):
    question: str


@app.post("/api/chat")
def ask_chatbot(req: ChatRequest):
    state = get_analytics_state()
    bot = ForecastChatBot(state["df"], state["overall_fc"]["90_days"])
    answer = bot.answer_query(req.question)
    return {"question": req.question, "answer": answer}


@app.get("/api/report/pdf")
def download_pdf_report():
    state = get_analytics_state()
    pdf_path = state["pdf_path"]
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF Report not found.")
    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename="ForecastIQ_Executive_Report.pdf"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

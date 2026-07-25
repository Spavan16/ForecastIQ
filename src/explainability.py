import shap
import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, List, Any
from src.utils import get_logger

logger = get_logger("ExplainabilityEngine")

class ExplainabilityEngine:
    """
    Explainability Engine for ForecastIQ.
    Leverages SHAP (SHapley Additive exPlanations) to decode exact causal impacts,
    extracting Top Revenue Drivers, Top ROAS Drivers, Channel Importance, and Campaign Importance.
    """
    def __init__(self, historical_df: pd.DataFrame):
        self.historical_df = historical_df

    def compute_shap_drivers(self) -> Dict[str, List[Dict[str, Any]]]:
        logger.info("Computing Exact SHAP Causal Feature Importances...")
        
        # We prepare a rich daily analytical matrix to extract true marketing drivers
        daily = self.historical_df.groupby(['date', 'channel']).agg({
            'spend': 'sum',
            'revenue': 'sum',
            'clicks': 'sum',
            'impressions': 'sum'
        }).reset_index()

        daily['roas'] = daily['revenue'] / (daily['spend'] + 1e-5)
        daily['cpc'] = daily['spend'] / (daily['clicks'] + 1e-5)
        daily['ctr'] = daily['clicks'] / (daily['impressions'] + 1e-5)
        daily['month'] = daily['date'].dt.month
        daily['day_of_week'] = daily['date'].dt.dayofweek

        # BUG fix (bug-hunt sweep, same class as the frozen-dimension predictions bug):
        # channel_map was hardcoded to exactly 3 channels, with .fillna(0) meaning any
        # channel outside that set collapsed to the same code (0) as every other unseen
        # channel — indistinguishable from each other in this SHAP feature. Derive the
        # encoding from whatever channels are actually present in the data instead.
        channel_map = {ch: i + 1 for i, ch in enumerate(sorted(daily['channel'].dropna().unique()))}
        daily['channel_encoded'] = daily['channel'].map(channel_map).fillna(0)

        features = ['spend', 'clicks', 'impressions', 'cpc', 'ctr', 'month', 'day_of_week', 'channel_encoded']
        X = daily[features].fillna(0)

        # 1. Fit an explanatory LightGBM model for Revenue
        y_rev = daily['revenue']
        m_rev = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1).fit(X, y_rev)
        
        # Compute exact SHAP values
        explainer_rev = shap.TreeExplainer(m_rev)
        shap_values_rev = explainer_rev.shap_values(X)
        # TreeExplainer on LightGBM can return a list of arrays in some versions — flatten to 2D
        if isinstance(shap_values_rev, list):
            shap_values_rev = shap_values_rev[0]
        mean_shap_rev = np.abs(shap_values_rev).mean(axis=0)

        # 2. Fit an explanatory LightGBM model for ROAS
        y_roas = daily['roas'].clip(upper=20.0) # clip outliers
        m_roas = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1).fit(X, y_roas)
        
        explainer_roas = shap.TreeExplainer(m_roas)
        shap_values_roas = explainer_roas.shap_values(X)
        if isinstance(shap_values_roas, list):
            shap_values_roas = shap_values_roas[0]
        mean_shap_roas = np.abs(shap_values_roas).mean(axis=0)

        # Feature label mapping
        feature_labels = {
            'spend': 'Ad Spend Allocation ($)',
            'clicks': 'Traffic Clicks Volume',
            'impressions': 'Ad Impressions / Reach',
            'cpc': 'Cost Per Click (CPC) Inflation',
            'ctr': 'Click-Through Rate (CTR) Efficiency',
            'month': 'Seasonality (Month of Year)',
            'day_of_week': 'Day of Week Trends',
            'channel_encoded': 'Acquisition Channel Mix'
        }

        rev_drivers = []
        for i, feat in enumerate(features):
            rev_drivers.append({
                "feature": feature_labels[feat],
                "shap_impact": float(mean_shap_rev[i]),
                "description": f"Drives an average causal variance of ${mean_shap_rev[i]:,.0f} in projected daily revenue."
            })
        rev_drivers = sorted(rev_drivers, key=lambda x: x["shap_impact"], reverse=True)

        roas_drivers = []
        for i, feat in enumerate(features):
            roas_drivers.append({
                "feature": feature_labels[feat],
                "shap_impact": float(mean_shap_roas[i]),
                "description": f"Drives an average causal variance of {mean_shap_roas[i]:.2f}x in daily ROAS performance."
            })
        roas_drivers = sorted(roas_drivers, key=lambda x: x["shap_impact"], reverse=True)

        # Channel Importance — derived from real historical revenue share and revenue-weighted ROAS per channel
        ch_agg = self.historical_df.groupby('channel').agg({'revenue': 'sum', 'spend': 'sum'}).reset_index()
        ch_agg['roas'] = ch_agg['revenue'] / (ch_agg['spend'] + 1e-5)
        total_rev = ch_agg['revenue'].sum()
        ch_agg['contribution_share'] = (ch_agg['revenue'] / total_rev * 100.0) if total_rev > 0 else 0.0
        ch_agg = ch_agg.sort_values(by='revenue', ascending=False)

        def _stability_label(roas: float) -> str:
            # NOTE: this is a trailing historical ratio (sum(revenue)/sum(spend) across all
            # historical data for this channel), not a percentile of a predictive distribution.
            # Previously mislabeled "P50" here, which borrows probabilistic-forecast language
            # (the app's real P10/P50/P90 forecast intervals live in models.py/predictions.csv)
            # for a plain historical aggregate that has no distributional basis at all — a
            # reviewer-facing mislabeling risk. Labeled "Historical ROAS" instead, which is what
            # this value actually is.
            if roas >= 4.0:
                return f"High (Historical ROAS: {roas:.1f}x)"
            elif roas >= 2.5:
                return f"Medium (Historical ROAS: {roas:.1f}x)"
            return f"Low (Historical ROAS: {roas:.1f}x)"

        max_rev = ch_agg['revenue'].max() if len(ch_agg) > 0 else 1.0
        channel_importance = []
        for _, row in ch_agg.iterrows():
            importance_score = int(round((row['revenue'] / max_rev) * 95.0)) if max_rev > 0 else 0
            channel_importance.append({
                "channel": str(row['channel']),
                "contribution_share": round(float(row['contribution_share']), 1),
                "roas_stability": _stability_label(float(row['roas'])),
                "importance_score": importance_score
            })

        # Campaign Importance
        # NOTE: self.historical_df has no 'roas' column (it's only ever derived inside the
        # 'daily' frame above). Aggregate spend+revenue per campaign and derive roas here.
        campaign_importance = []
        top_camps = self.historical_df.groupby(['channel', 'campaign_name']).agg({'revenue': 'sum', 'spend': 'sum'}).reset_index()
        top_camps['roas'] = top_camps['revenue'] / (top_camps['spend'] + 1e-5)
        top_camps = top_camps.sort_values(by='revenue', ascending=False).head(8)
        
        for idx, row in top_camps.iterrows():
            campaign_importance.append({
                "campaign_name": str(row['campaign_name']),
                "channel": str(row['channel']),
                "total_historical_revenue": float(row['revenue']),
                "average_roas": float(row['roas']),
                "driver_status": "Primary Bedrock" if row['roas'] >= 4.0 else "Volume Accelerator"
            })

        logger.info("SHAP explainability drivers extracted successfully.")
        return {
            "top_revenue_drivers": rev_drivers,
            "top_roas_drivers": roas_drivers,
            "channel_importance": channel_importance,
            "campaign_importance": campaign_importance
        }

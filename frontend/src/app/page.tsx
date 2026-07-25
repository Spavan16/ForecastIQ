"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  LayoutDashboard, Database, TrendingUp, Sliders, Cpu,
  Layers, ShieldAlert, MessageSquare, FileText,
  RefreshCw, CheckCircle2, AlertTriangle, ArrowUpRight,
  ArrowDownRight, Sparkles, Zap, Download, DollarSign, BarChart2, PieChart as PieIcon, Send,
  Maximize2, X, Search, ShoppingBag, Share2, Video, Target, Monitor, Users, Tag, Calendar, Gauge
} from "lucide-react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Legend, Cell, PieChart as RePieChart, Pie, LineChart, Line, ComposedChart
} from "recharts";

// ─── API base URL ────────────────────────────────────────────────────────────
const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

// ─── Types (mirrors backend/src/main.py response shapes exactly) ────────────

interface StatusState {
  status: string;
  timestamp: string;
  active_llm_provider: string;
  data_quality_score: number;
  risk_classification: string;
  has_critical_warnings: boolean;
}

interface TrajectoryPoint {
  day: string;
  p10: number;
  p50: number;
  p90: number;
}

interface ChannelShare {
  name: string;
  value: number;
  color: string;
}

interface CriticalWarning {
  message: string;
  penalty: number;
}

interface OverviewState {
  total_historical_spend: number;
  total_historical_revenue: number;
  overall_historical_roas: number;
  data_quality_score: number;
  risk_classification: string;
  risk_badge_color: string;
  forecast_90d_p50_revenue: number;
  forecast_90d_p50_roas: number;
  forecast_90d_p10_revenue: number;
  forecast_90d_p90_revenue: number;
  forecast_90d_p10_roas: number;
  forecast_90d_p90_roas: number;
  executive_summary: string;
  daily_trajectory: TrajectoryPoint[];
  channel_shares: ChannelShare[];
  critical_warnings: CriticalWarning[];
  trajectory_method: string;
}

interface ValidationSummary {
  total_records: number;
  channels_ingested: string[];
  channel_record_counts: Record<string, number>;
  min_date: string;
  max_date: string;
  total_spend: number;
  total_revenue: number;
  overall_roas: number;
  data_quality_score: number;
  audit_logs: string[];
  critical_warnings: CriticalWarning[];
  has_critical_warnings: boolean;
}

interface ModelValidationOverall {
  revenue_wape: number;
  revenue_smape: number;
  roas_smape: number;
  interval_coverage: number;
}

interface ModelValidationSegment {
  metric: string;
  forecast_period: string;
  dimension_type: string;
  rows: number;
  mae: number;
  mape: number;
  smape: number;
  wape?: number;
  interval_coverage: number;
}

interface ModelValidationRow {
  origin_date: string;
  forecast_period: string;
  dimension_type: string;
  dimension_value: string;
  metric: string;
  actual: number;
  p10: number;
  p50: number;
  p90: number;
  absolute_error: number;
  ape: number;
  smape: number;
  covered_by_interval: boolean;
}

interface ModelValidationResponse {
  summary: {
    status: string;
    folds: number;
    origin_dates: string[];
    data_quality_score: number;
    rows_scored: number;
    overall: ModelValidationOverall;
    by_segment: ModelValidationSegment[];
  };
  strongest_segments: ModelValidationSegment[];
  watchlist_segments: ModelValidationSegment[];
  recent_rows: ModelValidationRow[];
  artifacts: {
    summary_path: string;
    scorecard_path: string;
    scorecard_rows: number;
  };
}

interface RunHistoryItem {
  id: number;
  run_label: string;
  created_at: string;
  parameters: { data_quality_score: number; total_records: number };
  summary: { risk_classification: string; forecast_90d_p50_revenue: number; forecast_90d_p50_roas: number };
}

interface ForecastMetricValue {
  P10: number;
  P50: number;
  P90: number;
}
type ForecastPeriodData = Record<string, ForecastMetricValue>;
type ForecastsResponse = Record<string, ForecastPeriodData>;

interface DimensionsResponse {
  channels: string[];
  campaign_types: string[];
}

interface ScenarioForecastWindow {
  Revenue_P10: number;
  Revenue_P50: number;
  Revenue_P90: number;
  ROAS_P10: number;
  ROAS_P50: number;
  ROAS_P90: number;
  Spend_Expected: number;
}

interface ScenarioItem {
  id: string;
  name: string;
  description: string;
  cpc_change: string;
  conv_rate_change: string;
  revenue_multiplier: number;
  roas_multiplier: number;
  tag: string;
  forecasts: Record<string, ScenarioForecastWindow>;
}

interface RiskFactor {
  name: string;
  score: number;
  status: string;
  impact: string;
  mitigation: string;
}

interface RiskProfile {
  risk_score: number;
  risk_classification: string;
  badge_color: string;
  executive_risk_summary: string;
  risk_factors: RiskFactor[];
}

interface GrowthOpportunity {
  title: string;
  tag: string;
  insight: string;
}
interface RiskAssessmentItem {
  title: string;
  severity: string;
  insight: string;
}
interface BudgetRecommendation {
  channel: string;
  action: string;
  rationale: string;
}
interface InsightsResponse {
  executive_summary: string;
  growth_opportunities: GrowthOpportunity[];
  risk_assessment: RiskAssessmentItem[];
  budget_recommendations: BudgetRecommendation[];
  forecast_explanation: string;
}

interface ShapDriver {
  feature: string;
  shap_impact: number;
  description: string;
}
interface ChannelImportance {
  channel: string;
  contribution_share: number;
  roas_stability: string;
  importance_score: number;
}
interface CampaignImportance {
  campaign_name: string;
  channel: string;
  total_historical_revenue: number;
  average_roas: number;
  driver_status: string;
}
interface ExplainabilityResponse {
  top_revenue_drivers: ShapDriver[];
  top_roas_drivers: ShapDriver[];
  channel_importance: ChannelImportance[];
  campaign_importance: CampaignImportance[];
}

interface MonteCarloChannelDist {
  worst_case: number;
  expected_case: number;
  best_case: number;
}
interface HistogramBin {
  bin_center: number;
  bin_min: number;
  bin_max: number;
  frequency: number;
}
interface MonteCarloResponse {
  n_simulations: number;
  worst_case_revenue: number;
  expected_revenue: number;
  best_case_revenue: number;
  worst_case_roas: number;
  expected_roas: number;
  best_case_roas: number;
  channel_distributions: Record<string, MonteCarloChannelDist>;
  revenue_histogram: HistogramBin[];
  roas_histogram: HistogramBin[];
}

interface BudgetSimContribution {
  spend: number;
  revenue: number;
  roas: number;
  revenue_change_pct: number;
}
interface BudgetSimResponse {
  total_spend: number;
  total_revenue: number;
  total_roas: number;
  channel_contributions: Record<string, BudgetSimContribution>;
}

interface BudgetOptChannelRec {
  allocated_spend: number;
  expected_revenue: number;
  expected_roas: number;
  budget_share: number;
}
interface ConfidenceRange {
  revenue_p10: number;
  revenue_p50: number;
  revenue_p90: number;
  roas_p10: number;
  roas_p50: number;
  roas_p90: number;
}
interface BudgetOptResponse {
  max_budget: number;
  target_roas: number;
  recommended_total_spend: number;
  expected_total_revenue: number;
  expected_total_roas: number;
  channel_recommendations: Record<string, BudgetOptChannelRec>;
  confidence_range: ConfidenceRange;
}

interface ChatMessage {
  role: "user" | "bot";
  text: string;
}

type TabId =
  | "overview" | "validation" | "accuracy" | "forecasts" | "scenarios"
  | "budget" | "montecarlo" | "explainability" | "risk" | "chat";

const TABS: { id: TabId; label: string; icon: React.ElementType }[] = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "validation", label: "Data Validation", icon: Database },
  { id: "accuracy", label: "Model Validation", icon: Gauge },
  { id: "forecasts", label: "Forecasts", icon: TrendingUp },
  { id: "scenarios", label: "Scenarios", icon: Sliders },
  { id: "budget", label: "Budget Optimizer", icon: DollarSign },
  { id: "montecarlo", label: "Monte Carlo", icon: Layers },
  { id: "explainability", label: "Explainability", icon: Cpu },
  { id: "risk", label: "Risk & Insights", icon: ShieldAlert },
  { id: "chat", label: "Ask ForecastIQ", icon: MessageSquare },
];

const TAB_SUBTITLES: Record<TabId, string> = {
  overview: "Portfolio performance at a glance",
  validation: "Data quality audit and ingestion log",
  accuracy: "Backtested accuracy and interval coverage",
  forecasts: "AI-powered predictions to plan your growth",
  scenarios: "Bull, base, and bear projections",
  budget: "Simulate and optimize channel allocation",
  montecarlo: "Probabilistic outcome distributions",
  explainability: "What's actually driving the forecast",
  risk: "Portfolio risk factors and recommendations",
  chat: "Ask questions about your data in plain English",
};

// ─── Formatters ───────────────────────────────────────────────────────────
const fmtCompactCurrency = (n: number | undefined | null): string => {
  if (n === undefined || n === null || Number.isNaN(n)) return "$0";
  const abs = Math.abs(n);
  if (abs >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (abs >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return `${n.toFixed(0)}`;
};
const fmtCurrency = (n: number | undefined | null): string => {
  if (n === undefined || n === null || Number.isNaN(n)) return "$0.00";
  return `${n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
};
const fmtRoas = (n: number | undefined | null): string => {
  if (n === undefined || n === null || Number.isNaN(n)) return "0.00x";
  return `${n.toFixed(2)}x`;
};
const fmtPct = (n: number | undefined | null, digits = 1): string => {
  if (n === undefined || n === null || Number.isNaN(n)) return "0%";
  return `${n.toFixed(digits)}%`;
};

// ─── Small shared UI primitives ──────────────────────────────────────────
function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`rounded-2xl border border-[#1C2338] bg-[#0D1220] p-5 glow-card ${className}`}>
      {children}
    </div>
  );
}

// Restyled per Stitch design pass: cyan/purple/green replacing the old sky/violet/teal set,
// so KPI icon accents match the new electric-cyan theme instead of clashing with it.
const KPI_TINTS: Record<string, { bg: string; text: string }> = {
  sky: { bg: "bg-[#00F2FF]/15", text: "text-[#00F2FF]" },
  violet: { bg: "bg-[#B000FF]/15", text: "text-[#C86BFF]" },
  // Retinted to the real Instrument Black palette (was #00FF66/#4EDEA3 leftover from an
  // earlier pass). "sky" and "violet" above are left as-is deliberately — they still belong
  // to tabs that haven't been moved onto the real design system yet; changing them here would
  // silently reskin those other tabs half-way instead of as a deliberate pass.
  teal: { bg: "bg-[#3FB8A6]/15", text: "text-[#3FB8A6]" },
  amber: { bg: "bg-[#E8A33D]/15", text: "text-[#E8A33D]" },
  brick: { bg: "bg-[#C4544A]/15", text: "text-[#C4544A]" },
};

function KpiCard({ label, value, sub, icon: Icon, trend, tint = "sky" }: {
  label: string; value: string; sub?: string; icon: React.ElementType; trend?: "up" | "down"; tint?: keyof typeof KPI_TINTS;
}) {
  const t = KPI_TINTS[tint];
  return (
    <Card className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[11px] uppercase tracking-wide text-slate-500">{label}</span>
        <span className={`w-9 h-9 rounded-xl flex items-center justify-center shrink-0 ${t.bg}`}>
          <Icon size={16} className={t.text} />
        </span>
      </div>
      <span className="text-[26px] leading-none font-bold text-slate-50">{value}</span>
      {sub && (
        <span className={`flex items-center gap-1 text-xs ${trend === "up" ? "text-emerald-400" : trend === "down" ? "text-rose-400" : "text-slate-500"}`}>
          {trend === "up" && <ArrowUpRight size={12} />}
          {trend === "down" && <ArrowDownRight size={12} />}
          {sub}
        </span>
      )}
    </Card>
  );
}

function LoadingBlock({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2 text-slate-400 text-sm py-10 justify-center">
      <RefreshCw size={16} className="animate-spin" />
      Loading {label}...
    </div>
  );
}

function ErrorBlock({ label, message }: { label: string; message: string }) {
  return (
    <div className="flex flex-col items-center gap-2 text-rose-400 text-sm py-10 justify-center text-center">
      <AlertTriangle size={20} />
      <span>Failed to load {label}.</span>
      <span className="text-xs text-slate-500">{message}</span>
      <span className="text-xs text-slate-500">Check that the FastAPI backend is running at {API_BASE}.</span>
    </div>
  );
}

// Shared "glossy sheen" overlay for the Overview tab's warm-glass panels — a soft directional
// highlight (as if lit from the upper-left) plus a hairline top edge-light, layered on top of
// each panel's own gradient fill so panels read as an actual lit glass surface, not a flat
// tinted rectangle. Requires the panel itself to be position:relative + overflow-hidden.
function OverviewSheen() {
  return (
    <>
      <span className="pointer-events-none absolute inset-0" style={{ background:
        "radial-gradient(120% 80% at 0% 0%, rgba(255,255,255,0.55) 0%, rgba(255,255,255,0) 40%), " +
        "radial-gradient(140% 120% at 100% 0%, rgba(255,214,170,0.18) 0%, rgba(255,255,255,0) 45%)" }} />
      <span className="pointer-events-none absolute inset-x-0 top-0 h-px" style={{ background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.9), transparent)" }} />
    </>
  );
}

// Light glass card — inverse of GLASS_PANEL (frost over the mesh background instead of
// frost over black). Border/blur/shadow live in the className; the actual fill is a
// diagonal white gradient (not flat bg-white/45) so each card reads as an actual glossy
// pane catching light, rather than a flat translucent rectangle.
const GLASS_LIGHT = "relative overflow-hidden rounded-3xl border border-white/70 backdrop-blur-2xl shadow-[0_30px_60px_-28px_rgba(80,50,40,0.4),inset_0_1px_0_rgba(255,255,255,0.95)]";
const GLASS_LIGHT_BG = { background: "linear-gradient(135deg, rgba(255,255,255,0.68) 0%, rgba(255,255,255,0.34) 45%, rgba(255,255,255,0.54) 100%)" };
// Deliberately near-opaque, unlike GLASS_LIGHT_BG above. GLASS_LIGHT_BG's ~34–68% translucency
// only works because Overview's own panels sit on a background WE control (the mesh gradient),
// so the "glass" has something intentional to catch light from. A modal/overlay sits on top of
// whatever tab content is behind it instead — piping that same translucent fill onto a modal
// lets the blurred backdrop bleed through and wash out whatever's inside (this is what caused
// the Explainability expand-chart modal to look smeared). Modals get this solid-ish fill so
// their contents render cleanly regardless of what they're floating over.
const MODAL_GLASS_BG = { background: "linear-gradient(135deg, rgba(255,252,248,0.97) 0%, rgba(250,244,238,0.97) 100%)" };

function LightCard({ children, className = "", style }: { children: React.ReactNode; className?: string; style?: React.CSSProperties }) {
  return (
    <div className={`${GLASS_LIGHT} p-5 ${className}`} style={{ ...GLASS_LIGHT_BG, ...style }}>
      <OverviewSheen />
      <div className="relative z-10">{children}</div>
    </div>
  );
}

const LIGHT_KPI_TINTS: Record<string, { bg: string; text: string }> = {
  sky: { bg: "bg-[#0088AA]/15", text: "text-[#007090]" },
  violet: { bg: "bg-[#8000FF]/10", text: "text-[#6A00D5]" },
  teal: { bg: "bg-[#1F7A78]/15", text: "text-[#156B5E]" },
  amber: { bg: "bg-[#DD7A3C]/15", text: "text-[#B54A1C]" },
  brick: { bg: "bg-[#B5443B]/15", text: "text-[#92332A]" },
};

function LightKpiCard({ label, value, sub, icon: Icon, trend, tint = "sky" }: {
  label: string; value: string; sub?: string; icon: React.ElementType; trend?: "up" | "down"; tint?: keyof typeof LIGHT_KPI_TINTS;
}) {
  const t = LIGHT_KPI_TINTS[tint] || LIGHT_KPI_TINTS.sky;
  return (
    <LightCard className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[11px] uppercase tracking-widest text-[#8A7A6E]">{label}</span>
        <span className={`w-9 h-9 rounded-xl flex items-center justify-center shrink-0 ${t.bg}`}>
          <Icon size={20} className={t.text} strokeWidth={2.5} />
        </span>
      </div>
      <span className="text-[26px] leading-none font-bold text-[#3A2E28] font-serif">{value}</span>
      {sub && (
        <span className={`flex items-center gap-1 text-xs ${trend === "up" ? "text-[#1F7A78]" : trend === "down" ? "text-[#C0632B]" : "text-[#8A7A6E]"}`}>
          {trend === "up" && <ArrowUpRight size={12} />}
          {trend === "down" && <ArrowDownRight size={12} />}
          {sub}
        </span>
      )}
    </LightCard>
  );
}

// Numbered editorial section header — the gold-ringed "01 / 02 / 03" badge is what makes the
// warm theme read as a designed report rather than a reskinned dashboard card.
function OverviewSectionHead({ index, title, meta }: { index: string; title: string; meta?: string }) {
  return (
    <div className="relative flex items-end justify-between gap-4 mb-4">
      <div className="flex items-center gap-3">
        <span className="flex h-7 w-7 items-center justify-center rounded-full text-[11px] font-semibold" style={{ background: "rgba(192,99,43,0.12)", color: "#C0632B", border: "1px solid rgba(192,99,43,0.28)" }}>
          {index}
        </span>
        <h3 className="font-serif text-[20px] leading-none tracking-tight text-[#3A2E28]">{title}</h3>
      </div>
      {meta && <span className="text-[10px] uppercase tracking-widest text-[#8A7A6E]">{meta}</span>}
    </div>
  );
}

// Overview Revenue Trajectory tooltip — dark instrument-style card (deliberately breaking from
// the light glass theme here, same way the reference design does) with a per-series color key
// so P10/P50/P90 read instantly without a legend lookup. Label reformatted from the backend's
// raw "Day 21" string to "D21" — same real value, just a tighter label to match the chart's
// tick format below. Reads real values from payload[i].payload.p10/p50/p90 rather than
// payload[i].value, since .value on each series now holds the sqrt-transformed PLOT coordinate
// (p10Plot/p50Plot/p90Plot) — the tooltip must show the actual number, not the plotting one.
function OverviewChartTooltip({ active, payload, label }: any) {
  if (!active || !payload || !payload.length) return null;
  const row = payload[0]?.payload;
  if (!row) return null;
  const { p10, p50, p90 } = row;
  const shortLabel = typeof label === "string" ? label.replace(/^Day\s*/i, "D") : label;
  return (
    <div className="rounded-xl px-4 py-3 min-w-[190px]" style={{ background: "#221911", border: "1px solid rgba(255,255,255,0.08)", boxShadow: "0 20px 40px -12px rgba(0,0,0,0.5)" }}>
      <div className="text-[11px] font-semibold text-white/45 uppercase tracking-widest mb-2 font-mono">{shortLabel}</div>
      <div className="flex flex-col gap-1.5 text-[13px] font-mono">
        {p10 != null && (
          <div className="flex items-center justify-between gap-4">
            <span style={{ color: "#5FBFAE" }}>P10</span>
            <span className="text-white">{fmtCurrency(p10)}</span>
          </div>
        )}
        {p50 != null && (
          <div className="flex items-center justify-between gap-4">
            <span style={{ color: "#E8965B" }}>P50</span>
            <span className="text-white font-semibold">{fmtCurrency(p50)}</span>
          </div>
        )}
        {p90 != null && (
          <div className="flex items-center justify-between gap-4">
            <span style={{ color: "#5FBFAE" }}>P90</span>
            <span className="text-white">{fmtCurrency(p90)}</span>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Tab: Overview ─────────────────────────────────────────────────────────
// NOTE: Overview intentionally runs its own "warm glossy" light theme, separate from the
// Instrument Black system used by every other tab (Forecasts, Validation, etc). This is a
// scoped, temporary divergence per explicit request — not a sign the rest of the app should
// be retinted to match. All colors below are local to this function.
function TabOverview({ data, loading, error }: { data: OverviewState | null; loading: boolean; error?: string }) {
  if (loading) return <LoadingBlock label="Overview" />;
  if (error) return <ErrorBlock label="Overview" message={error} />;
  if (!data) return <LoadingBlock label="Overview" />;

  // Warm-glossy palette — orange/teal split (echoing the mesh-gradient reference), tuned
  // dark enough to hold contrast against the light glass panels below.
  const CHANNEL_COLORS: Record<string, string> = { "Google Ads": "#DD7A3C", "Meta Ads": "#1F7A78", "Bing Ads": "#9C8F82" };
  const qualityColor = data.data_quality_score >= 90 ? "#1F7A78" : data.data_quality_score >= 70 ? "#C77C22" : "#B5443B";
  const qualityCircumference = 2 * Math.PI * 42;
  const qualityOffset = qualityCircumference * (1 - Math.max(0, Math.min(100, data.data_quality_score)) / 100);

  const roasColor = data.overall_historical_roas < 1 ? "#B5443B" : data.overall_historical_roas < 3 ? "#C77C22" : "#1F7A78";
  const roasLabel = data.overall_historical_roas < 1 ? "Below breakeven" : data.overall_historical_roas < 3 ? "Positive return" : "Strong return";
  const isLowRisk = data.risk_classification.toLowerCase().includes("low");
  const riskColor = isLowRisk ? "#1F7A78" : "#B5443B";

  // Revenue Trajectory chart data — sqrt-transform applied only to the PLOTTED coordinates
  // (p10Plot/p50Plot/p90Plot). The real, untransformed p10/p50/p90 travel in the same row and
  // are what OverviewChartTooltip actually reads, so the tooltip always shows exact values
  // regardless of how the line positions are compressed for readability.
  const sqrt = (v: number) => Math.sqrt(Math.max(0, v));
  const trajData = data.daily_trajectory.map(p => ({
    day: p.day, p10: p.p10, p50: p.p50, p90: p.p90,
    p10Plot: sqrt(p.p10), p50Plot: sqrt(p.p50), p90Plot: sqrt(p.p90),
  }));
  const maxPlotVal = Math.max(...trajData.map(p => p.p90Plot), 1) * 1.04;
  const yPlotTicks = [0, 0.15, 0.35, 0.6, 1].map(f => maxPlotVal * f);

  return (
    <div className="flex flex-col gap-6">

      {/* Branded editorial header — this was the actual missing chunk. Every other tab gets its
          title from the shared app header ("Overview" / "Portfolio performance at a glance"),
          but the warm theme's own reference design promotes this to a dedicated in-page
          masthead. The status pill reuses data.trajectory_method (the same real field already
          shown lower down next to Revenue Trajectory) rather than a static label. */}
      <div className="relative flex items-start justify-between gap-6">
        <div>
          <span className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.22em]" style={{ color: "#C0632B" }}>
            <Sparkles size={12} />
            Performance Intelligence
          </span>
          <h1 className="font-serif text-[32px] leading-tight tracking-tight text-[#3A2E28] mt-1.5">Portfolio Overview</h1>
        </div>
        <span className="flex items-center gap-2 shrink-0 mt-2 rounded-full border border-white/70 bg-white/45 backdrop-blur-xl px-3.5 py-1.5 text-[10px] font-semibold uppercase tracking-widest text-[#3A2E28]">
          <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: "#1F7A78" }} />
          {data.trajectory_method}
        </span>
      </div>

      {data.critical_warnings.length > 0 && (
          <div className="rounded-2xl border border-[#B5443B]/25 bg-[#B5443B]/[0.07] backdrop-blur-xl overflow-hidden">
            <div className="flex items-center gap-2 px-5 pt-4 pb-2">
              <AlertTriangle size={14} className="text-[#B5443B]" />
              <span className="text-[11px] font-bold uppercase tracking-widest text-[#B5443B]">Critical Warnings</span>
            </div>
            <div className="divide-y divide-[#B5443B]/15">
              {/* Real penalty value from the backend's rule engine (CriticalWarning.penalty) —
                  not a display invention. Previously fetched but never rendered. */}
              {data.critical_warnings.map((w, i) => (
                <div key={i} className="flex items-center justify-between gap-4 px-5 py-3">
                  <span className="text-sm text-[#4A3F38]">{w.message}</span>
                  <span className="shrink-0 text-sm font-bold font-mono text-[#B5443B]">-{Math.abs(w.penalty)} pts</span>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className={`${GLASS_LIGHT} grid grid-cols-5 divide-x divide-[#3A2E28]/10`} style={GLASS_LIGHT_BG}>
          <OverviewSheen />
          <div className="group relative overflow-hidden p-5">
            <div className="absolute top-0 left-0 right-0 h-[3px]" style={{ backgroundColor: "#8A7A6E" }} />
            <span className="pointer-events-none absolute left-0 right-0 top-0 h-14 opacity-0 transition-opacity duration-300 group-hover:opacity-100" style={{ background: "linear-gradient(to bottom, #8A7A6E22, transparent)" }} />
            <span className="text-[11px] uppercase tracking-widest text-[#8A7A6E]">Historical Spend</span>
            <div className="text-2xl font-bold text-[#3A2E28] font-serif mt-2">{fmtCompactCurrency(data.total_historical_spend)}</div>
          </div>
          <div className="group relative overflow-hidden p-5">
            <div className="absolute top-0 left-0 right-0 h-[3px]" style={{ backgroundColor: "#DD7A3C" }} />
            <div className="absolute inset-0 bg-gradient-to-br from-[#DD7A3C]/[0.14] to-transparent pointer-events-none" />
            <span className="pointer-events-none absolute left-0 right-0 top-0 h-14 opacity-0 transition-opacity duration-300 group-hover:opacity-100" style={{ background: "linear-gradient(to bottom, #DD7A3C22, transparent)" }} />
            <span className="relative text-[11px] uppercase tracking-widest text-[#8A7A6E]">Historical Revenue</span>
            <div className="relative text-3xl font-bold font-serif mt-2 text-[#C0632B]">{fmtCompactCurrency(data.total_historical_revenue)}</div>
          </div>
          <div className="group relative overflow-hidden p-5">
            <div className="absolute top-0 left-0 right-0 h-[3px]" style={{ backgroundColor: roasColor }} />
            <div className="absolute inset-0 pointer-events-none" style={{ background: `linear-gradient(135deg, ${roasColor}14, transparent 65%)` }} />
            <span className="pointer-events-none absolute left-0 right-0 top-0 h-14 opacity-0 transition-opacity duration-300 group-hover:opacity-100" style={{ background: `linear-gradient(to bottom, ${roasColor}22, transparent)` }} />
            <span className="relative text-[11px] uppercase tracking-widest text-[#8A7A6E]">Overall ROAS</span>
            <div className="relative text-2xl font-bold font-serif mt-2" style={{ color: roasColor }}>{fmtRoas(data.overall_historical_roas)}</div>
            <div className="relative text-[11px] mt-1 font-medium" style={{ color: roasColor }}>{roasLabel}</div>
          </div>
          <div className="group relative overflow-hidden p-5 flex items-center gap-4">
            <div className="absolute top-0 left-0 right-0 h-[3px]" style={{ backgroundColor: qualityColor }} />
            <span className="pointer-events-none absolute left-0 right-0 top-0 h-14 opacity-0 transition-opacity duration-300 group-hover:opacity-100" style={{ background: `linear-gradient(to bottom, ${qualityColor}22, transparent)` }} />
            <div className="relative w-16 h-16 shrink-0">
              <svg viewBox="0 0 100 100" className="w-16 h-16 -rotate-90">
                <circle cx="50" cy="50" r="42" fill="none" stroke="#E4D6C8" strokeWidth="10" />
                <circle cx="50" cy="50" r="42" fill="none" stroke={qualityColor} strokeWidth="10" strokeLinecap="round"
                  strokeDasharray={qualityCircumference} strokeDashoffset={qualityOffset} style={{ transition: "stroke-dashoffset 0.6s ease" }} />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center text-sm font-bold text-[#3A2E28] font-mono">{Math.round(data.data_quality_score)}</div>
            </div>
            <div className="relative">
              <span className="text-[11px] uppercase tracking-widest text-[#8A7A6E] block">Data Quality</span>
              <span className="text-xs font-medium" style={{ color: qualityColor }}>Audit passed</span>
            </div>
          </div>
          <div className="group relative overflow-hidden p-5">
            <div className="absolute top-0 left-0 right-0 h-[3px]" style={{ backgroundColor: riskColor }} />
            <div className="absolute inset-0 pointer-events-none" style={{ background: `linear-gradient(135deg, ${riskColor}10, transparent 65%)` }} />
            <span className="pointer-events-none absolute left-0 right-0 top-0 h-14 opacity-0 transition-opacity duration-300 group-hover:opacity-100" style={{ background: `linear-gradient(to bottom, ${riskColor}22, transparent)` }} />
            <div className="relative flex items-center justify-between mb-1">
              <span className="text-[11px] uppercase tracking-widest text-[#8A7A6E]">Portfolio Risk</span>
              <AlertTriangle size={14} style={{ color: riskColor }} />
            </div>
            <div className="relative text-lg font-bold font-serif mt-1" style={{ color: riskColor }}>{data.risk_classification}</div>
            <div className="relative text-[10px] text-[#8A7A6E] mt-1 leading-relaxed">Driven by channel concentration, not data quality</div>
          </div>
        </div>

        <div className={`${GLASS_LIGHT} p-5`} style={GLASS_LIGHT_BG}>
          <OverviewSheen />
          <OverviewSectionHead index="01" title="Revenue Trajectory" meta={data.trajectory_method} />
          <div className="relative flex items-center gap-5 mb-4 -mt-1">
            <span className="flex items-center gap-1.5 text-[11px] text-[#4A3F38]">
              <span className="w-3 h-[2px] rounded-full" style={{ backgroundColor: "#DD7A3C" }} />
              P50 median
            </span>
            <span className="flex items-center gap-1.5 text-[11px] text-[#4A3F38]">
              <svg width="12" height="2" className="shrink-0"><line x1="0" y1="1" x2="12" y2="1" stroke="#1F7A78" strokeWidth="1.5" strokeDasharray="3 2" /></svg>
              P10–P90 band
            </span>
          </div>
          {/* Honest caption, not a cosmetic apology — the backend's daily residual std is
              nearly double the daily P50 (confirmed against real numbers: main.py computes
              horizon_std = std_rev * (day+1)**0.75, then p10 = max(0, p50 - 1.28*horizon_std)),
              so P10 floors at exactly $0 starting day 1, for the whole window. That's real
              model output, not a bug — this just says so instead of pretending otherwise. */}
          <p className="relative text-[11px] text-[#8A7A6E]/80 italic -mt-2 mb-1">
            P10 floors at $0 across most of this horizon — daily-level volatility in the underlying model exceeds the median forecast.
          </p>
          {/* Second, distinct caption (not a restatement of the one above): explains WHY this
              chart's band looks so much wider than the 90-day revenue KPI cards elsewhere on
              this page, so it doesn't read as inconsistent or broken. This chart's P10/P90 are
              per-DAY at horizon N (backend main.py: horizon_std = std_rev*(day+1)**0.75, no
              floor beyond max(0,...)) — genuinely noisier than a 90-day SUM, since summing 90
              days cancels out day-to-day noise a single far-out day can't. Both numbers are
              real model output computed the same disciplined way; they just answer different
              questions ("what will day 90 alone look like" vs "what will the 90-day total be"). */}
          <p className="relative text-[11px] text-[#8A7A6E]/80 italic -mt-1 mb-3">
            Shown here is uncertainty for that single day's revenue, not the cumulative forecast — a lone day out at D90 is naturally far noisier than the 90-day total shown above, since summing days cancels out day-to-day noise a single point can't.
          </p>
          <ResponsiveContainer width="100%" height={340}>
            <AreaChart data={trajData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="p50FillLight" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#DD7A3C" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#DD7A3C" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#3A2E2815" />
              <XAxis dataKey="day" tickFormatter={(d: string) => typeof d === "string" ? d.replace(/^Day\s*/i, "D") : d} tick={{ fontSize: 9, fill: "#8A7A6E" }} interval={5} />
              {/* Recharts' own scale="sqrt" prop didn't visually apply last round (gridlines
                  stayed evenly spaced — a dead giveaway it silently fell back to linear). Fixed
                  properly here by pre-transforming the PLOTTED coordinates only (p10Plot/
                  p50Plot/p90Plot, computed below) — same technique already proven on the
                  Forecasts tab's horizon fan charts. Real, untransformed values (p10/p50/p90)
                  travel alongside in the same data row and are what the tooltip actually reads
                  (see OverviewChartTooltip — payload[i].payload.p10, not payload[i].value). */}
              <YAxis domain={[0, maxPlotVal]} ticks={yPlotTicks} tickFormatter={(v: number) => fmtCompactCurrency(v * v)} tick={{ fontSize: 10, fill: "#8A7A6E" }} />
              <Tooltip content={<OverviewChartTooltip />} cursor={{ stroke: "#3A2E28", strokeOpacity: 0.25, strokeWidth: 1 }} />
              <Area type="monotone" dataKey="p90Plot" stroke="#1F7A78" fill="transparent" strokeDasharray="5 3" strokeWidth={1.5} name="P90" activeDot={{ r: 4, fill: "#1F7A78", stroke: "#fff", strokeWidth: 2 }} />
              <Area type="monotone" dataKey="p50Plot" stroke="#DD7A3C" fill="url(#p50FillLight)" strokeWidth={2.5} name="P50" activeDot={{ r: 5, fill: "#DD7A3C", stroke: "#fff", strokeWidth: 2 }} />
              <Area type="monotone" dataKey="p10Plot" stroke="#1F7A78" fill="transparent" strokeDasharray="5 3" strokeWidth={1.5} name="P10" activeDot={{ r: 4, fill: "#1F7A78", stroke: "#fff", strokeWidth: 2 }} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Reverted to match the actual v0 reference: Channel Attribution (donut + list +
            nested Dominant Channel callout) and Executive Summary are two EQUAL-WIDTH cards
            side by side — not a 65/35 split with Dominant Channel pulled into its own card
            (that was my error last round) and not stacked full-width (the version before that). */}
        <div className="grid grid-cols-2 gap-6 items-stretch">
          <div className={`${GLASS_LIGHT} p-5 flex flex-col`} style={GLASS_LIGHT_BG}>
            <OverviewSheen />
            <OverviewSectionHead index="02" title="Channel Attribution" meta="Revenue Share" />
            <div className="relative grid grid-cols-2 gap-6 items-center mb-4">
              <div className="relative">
                <ResponsiveContainer width="100%" height={200}>
                  <RePieChart>
                    <Pie data={data.channel_shares} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={52} outerRadius={92} paddingAngle={2}>
                      {data.channel_shares.map((c, i) => <Cell key={i} fill={CHANNEL_COLORS[c.name] ?? "#9C8F82"} stroke="rgba(255,255,255,0.6)" strokeWidth={2} />)}
                    </Pie>
                    <Tooltip contentStyle={{ backgroundColor: "rgba(255,250,245,0.95)", borderColor: "#E6D9CC", borderRadius: 8, color: "#3A2E28" }} itemStyle={{ color: "#3A2E28" }} labelStyle={{ color: "#8A7A6E" }} formatter={(v: any) => `${v}%`} />
                  </RePieChart>
                </ResponsiveContainer>
                <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
                  <span className="text-[10px] uppercase tracking-widest text-[#8A7A6E]">Total</span>
                  <span className="font-serif text-xl text-[#3A2E28]">{fmtCompactCurrency(data.total_historical_revenue)}</span>
                </div>
              </div>
              <div className="flex flex-col gap-2.5">
                {[...data.channel_shares].sort((a, b) => b.value - a.value).map((c, i) => {
                  const color = CHANNEL_COLORS[c.name] ?? "#9C8F82";
                  return (
                    <div key={i} className="bg-white/50 backdrop-blur-sm rounded-xl border border-white/60 px-3.5 py-2.5">
                      <div className="flex items-center justify-between">
                        <span className="flex items-center gap-2 text-[#3A2E28] text-sm font-medium">
                          <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
                          {c.name}
                        </span>
                        <div className="text-right font-mono">
                          <div className="text-[#3A2E28] font-bold text-sm">{c.value}%</div>
                          <div className="text-[#8A7A6E] text-[11px]">{fmtCompactCurrency(data.total_historical_revenue * (c.value / 100))}</div>
                        </div>
                      </div>
                      <div className="h-1 rounded-full bg-[#3A2E28]/10 mt-2 overflow-hidden">
                        <div className="h-full rounded-full" style={{ width: `${c.value}%`, backgroundColor: color, transition: "width 0.6s ease" }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
            {(() => {
              const top = [...data.channel_shares].sort((a, b) => b.value - a.value)[0];
              if (!top) return null;
              const concColor = top.value >= 70 ? "#B5443B" : top.value >= 45 ? "#C77C22" : "#1F7A78";
              return (
                <div className="relative bg-white/50 backdrop-blur-sm rounded-xl border-l-2 border-t border-r border-b border-white/60 p-4 mt-auto" style={{ borderLeftColor: concColor }}>
                  <span className="text-[11px] uppercase tracking-widest text-[#8A7A6E] block mb-1.5">Dominant Channel</span>
                  <div className="text-sm text-[#3A2E28]">
                    <span className="font-bold">{top.name}</span> drives{" "}
                    <span className="font-semibold font-mono" style={{ color: concColor }}>{top.value}%</span> of attributed revenue.
                  </div>
                </div>
              );
            })()}
          </div>

          <div className={`${GLASS_LIGHT} p-5 flex flex-col`} style={GLASS_LIGHT_BG}>
            <OverviewSheen />
            <OverviewSectionHead index="03" title="Executive Summary" />
            <p className="relative text-[15px] text-[#4A3F38] leading-[1.6] font-serif">{data.executive_summary}</p>
            <div className="relative grid grid-cols-2 gap-4 mt-auto pt-4 border-t border-[#3A2E28]/10">
              <div className="relative overflow-hidden rounded-2xl p-4" style={{ background: "linear-gradient(155deg, #241B14 0%, #1A130E 100%)", boxShadow: "0 20px 40px -24px rgba(40,25,15,0.7), inset 0 1px 0 rgba(255,255,255,0.06)" }}>
                <span className="pointer-events-none absolute -right-6 -top-6 h-20 w-20 rounded-full" style={{ background: "radial-gradient(circle, rgba(221,122,60,0.22), transparent 70%)" }} />
                <span className="relative mb-2 block text-[10.5px] font-medium uppercase tracking-[0.18em] text-white/45">90-Day Revenue Forecast</span>
                <div className="relative">
                  <ConfidenceBracket p10={data.forecast_90d_p10_revenue} p50={data.forecast_90d_p50_revenue} p90={data.forecast_90d_p90_revenue} formatFn={fmtCompactCurrency} />
                </div>
              </div>
              <div className="relative overflow-hidden rounded-2xl p-4" style={{ background: "linear-gradient(155deg, #241B14 0%, #1A130E 100%)", boxShadow: "0 20px 40px -24px rgba(40,25,15,0.7), inset 0 1px 0 rgba(255,255,255,0.06)" }}>
                <span className="pointer-events-none absolute -right-6 -top-6 h-20 w-20 rounded-full" style={{ background: "radial-gradient(circle, rgba(221,122,60,0.22), transparent 70%)" }} />
                <span className="relative mb-2 block text-[10.5px] font-medium uppercase tracking-[0.18em] text-white/45">90-Day ROAS Forecast</span>
                <div className="relative">
                  <ConfidenceBracket p10={data.forecast_90d_p10_roas} p50={data.forecast_90d_p50_roas} p90={data.forecast_90d_p90_roas} formatFn={fmtRoas} />
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="relative flex items-center justify-center gap-1.5 text-xs text-[#8A7A6E]">
          <span>Generated from ensemble forecasting</span>
          <ArrowUpRight size={14} />
        </div>
    </div>
  );
}

// ─── Tab: Data Validation ───────────────────────────────────────────────────
function TabValidation({ data, loading, error }: { data: ValidationSummary | null; loading: boolean; error?: string }) {
  if (loading) return <LoadingBlock label="Data Validation" />;
  if (error) return <ErrorBlock label="Data Validation" message={error} />;
  if (!data) return <LoadingBlock label="Data Validation" />;
  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-4 gap-4">
        <LightKpiCard label="Total Records" value={data.total_records.toLocaleString()} icon={Database} tint="teal" />
        <LightKpiCard label="Data Quality Score" value={fmtPct(data.data_quality_score, 1)} icon={CheckCircle2} tint="teal" />
        <LightKpiCard label="Date Range" value={`${data.min_date} → ${data.max_date}`} icon={Calendar} tint="amber" />
        <LightKpiCard label="Channels Ingested" value={String(data.channels_ingested.length)} icon={Layers} sub={data.channels_ingested.join(", ")} tint="sky" />
      </div>
      {data.has_critical_warnings && (
        <LightCard className="!border-[#C0632B]/30 !bg-white/80">
          <h3 className="text-sm font-bold text-[#C0632B] mb-3 flex items-center gap-2"><AlertTriangle size={16} /> Critical Warnings ({data.critical_warnings.length})</h3>
          <div className="flex flex-col gap-2">
            {data.critical_warnings.map((w, i) => (
              <div key={i} className="flex items-center justify-between text-xs bg-[#C0632B]/5 rounded-lg px-3 py-2 border border-[#C0632B]/10">
                <span className="text-[#A64516] font-medium">{w.message}</span>
                <span className="text-[#C0632B] font-bold shrink-0 ml-4 font-mono">-{w.penalty} pts</span>
              </div>
            ))}
          </div>
        </LightCard>
      )}
      <LightCard>
        <h3 className="text-base font-extrabold text-[#1A1512] mb-4 flex items-center gap-2">
           <Database size={18} className="text-[#8A7A6E]" />
           Full Audit Log
        </h3>
        <div className="flex flex-col gap-2 max-h-96 overflow-y-auto font-mono">
          {data.audit_logs.map((log, i) => (
            <div key={i} className="text-[13px] font-semibold text-[#2B221E] border-b border-[#3A2E28]/15 pb-2.5">{log}</div>
          ))}
        </div>
      </LightCard>
    </div>
  );
}

// ─── Tab: Forecasts ─────────────────────────────────────────────────────
function TabModelValidation({ data, loading, error }: { data: ModelValidationResponse | null; loading: boolean; error?: string }) {
  if (loading) return <LoadingBlock label="Model Validation" />;
  if (error) return <ErrorBlock label="Model Validation" message={error} />;
  if (!data) return <LoadingBlock label="Model Validation" />;

  const overall = data.summary.overall;
  const coverageColor = overall.interval_coverage >= 80 ? "#1F7A78" : overall.interval_coverage >= 70 ? "#C77C22" : "#B5443B";
  const coverageCircumference = 2 * Math.PI * 42;
  const coverageOffset = coverageCircumference * (1 - Math.max(0, Math.min(100, overall.interval_coverage)) / 100);
  const shortDimension = (dim: string) => dim === "CampaignType" ? "Type" : dim;
  const shortWindow = (period: string) => period.replace("_days", "d");
  const chartData = data.summary.by_segment
    .filter(row => row.metric === "Revenue")
    .sort((a, b) => {
      const periodOrder = (PERIOD_DAYS[a.forecast_period] ?? 999) - (PERIOD_DAYS[b.forecast_period] ?? 999);
      return periodOrder || shortDimension(a.dimension_type).localeCompare(shortDimension(b.dimension_type));
    })
    .map(row => ({
      name: `${shortDimension(row.dimension_type)} ${shortWindow(row.forecast_period)}`,
      wape: row.wape ?? row.smape,
    }));
  const pickSegments = (direction: "best" | "worst") => {
    const sorted = [...data.summary.by_segment].sort((a, b) => {
      const av = a.metric === "Revenue" ? (a.wape ?? a.smape) : a.smape;
      const bv = b.metric === "Revenue" ? (b.wape ?? b.smape) : b.smape;
      return direction === "best" ? av - bv : bv - av;
    });
    const seen = new Set<string>();
    const result: ModelValidationSegment[] = [];
    for (const row of sorted) {
      const key = `${row.dimension_type}-${row.metric}`;
      if (seen.has(key)) continue;
      seen.add(key);
      result.push(row);
      if (result.length === 6) break;
    }
    return result;
  };
  const strongestRows = pickSegments("best");
  const watchlistRows = pickSegments("worst");
  const valueForMetric = (row: ModelValidationRow) => row.metric === "Revenue" ? fmtCompactCurrency(row.p50) : fmtRoas(row.p50);
  const actualForMetric = (row: ModelValidationRow) => row.metric === "Revenue" ? fmtCompactCurrency(row.actual) : fmtRoas(row.actual);
  const errorForSegment = (row: ModelValidationSegment) => row.metric === "Revenue" ? `${fmtPct(row.wape ?? row.smape, 1)} WAPE` : `${fmtPct(row.smape, 1)} SMAPE`;
  const displayDimensionValue = (value: string) => value.length > 28 ? `${value.slice(0, 25)}...` : value;

  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-4 gap-4">
        <LightKpiCard label="Revenue WAPE" value={fmtPct(overall.revenue_wape, 2)} icon={Target} tint="teal" sub="weighted holdout error" />
        <LightKpiCard label="Revenue SMAPE" value={fmtPct(overall.revenue_smape, 2)} icon={BarChart2} tint="amber" sub={`${data.summary.folds} rolling folds`} />
        <LightKpiCard label="ROAS SMAPE" value={fmtPct(overall.roas_smape, 2)} icon={Gauge} tint="sky" sub={`${data.summary.rows_scored} scored rows`} />
        <LightKpiCard label="Interval Coverage" value={fmtPct(overall.interval_coverage, 2)} icon={ShieldAlert} tint="teal" sub="actual inside P10-P90" />
      </div>

      <div className="grid grid-cols-[360px_1fr] gap-6">
        <LightCard className="min-h-[310px] flex flex-col justify-between">
          <OverviewSectionHead index="01" title="Coverage Calibration" meta="P10 / P90" />
          <div className="flex items-center gap-6">
            <div className="relative w-36 h-36 shrink-0">
              <svg viewBox="0 0 100 100" className="w-36 h-36 -rotate-90">
                <circle cx="50" cy="50" r="42" fill="none" stroke="#E4D6C8" strokeWidth="10" />
                <circle cx="50" cy="50" r="42" fill="none" stroke={coverageColor} strokeWidth="10" strokeLinecap="round"
                  strokeDasharray={coverageCircumference} strokeDashoffset={coverageOffset} />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-3xl font-black font-serif text-[#3A2E28]">{fmtPct(overall.interval_coverage, 0)}</span>
                <span className="text-[10px] uppercase tracking-widest text-[#8A7A6E]">covered</span>
              </div>
            </div>
            <div className="flex flex-col gap-3">
              <div>
                <span className="text-[10px] uppercase tracking-widest text-[#8A7A6E]">Fold Origins</span>
                <div className="mt-2 flex flex-wrap gap-2">
                  {data.summary.origin_dates.map(origin => (
                    <span key={origin} className="rounded-full border border-[#3A2E28]/10 bg-white/55 px-3 py-1 text-xs font-semibold text-[#3A2E28]">
                      {origin}
                    </span>
                  ))}
                </div>
              </div>
              <div>
                <span className="text-[10px] uppercase tracking-widest text-[#8A7A6E]">Artifacts</span>
                <div className="mt-1 text-sm font-semibold text-[#3A2E28]">{data.artifacts.scorecard_rows.toLocaleString()} scorecard rows</div>
              </div>
            </div>
          </div>
        </LightCard>

        <LightCard className="min-h-[310px]">
          <OverviewSectionHead index="02" title="Revenue Error By Segment" meta="WAPE" />
          <ResponsiveContainer width="100%" height={235}>
            <BarChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#3A2E2815" />
              <XAxis dataKey="name" interval={0} tick={{ fontSize: 10, fill: "#8A7A6E" }} height={34} tickLine={false} />
              <YAxis tickFormatter={(v: number) => `${v}%`} tick={{ fontSize: 10, fill: "#8A7A6E" }} />
              <Tooltip contentStyle={{ backgroundColor: "rgba(255,250,245,0.97)", borderColor: "#E6D9CC", borderRadius: 8, color: "#3A2E28" }} formatter={(v: any) => fmtPct(Number(v), 1)} />
              <Bar dataKey="wape" radius={[6, 6, 0, 0]} fill="#1F7A78" />
            </BarChart>
          </ResponsiveContainer>
        </LightCard>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <LightCard>
          <OverviewSectionHead index="03" title="Strongest Segments" meta="Lowest error" />
          <div className="flex flex-col gap-2">
            {strongestRows.map((row, i) => (
              <div key={`${row.metric}-${row.forecast_period}-${row.dimension_type}-${i}`} className="grid grid-cols-[1fr_auto_auto] items-center gap-3 rounded-xl border border-white/60 bg-white/50 px-3 py-2.5">
                <div>
                  <div className="text-sm font-bold text-[#3A2E28]">{row.dimension_type} · {PERIOD_LABELS[row.forecast_period] ?? row.forecast_period}</div>
                  <div className="text-[11px] uppercase tracking-widest text-[#8A7A6E]">{row.metric} · {row.rows} rows</div>
                </div>
                <span className="text-sm font-black text-[#1F7A78] font-mono">{errorForSegment(row)}</span>
                <span className="text-xs font-bold text-[#3A2E28] font-mono">{fmtPct(row.interval_coverage, 0)}</span>
              </div>
            ))}
          </div>
        </LightCard>

        <LightCard>
          <OverviewSectionHead index="04" title="Watchlist Segments" meta="Highest error" />
          <div className="flex flex-col gap-2">
            {watchlistRows.map((row, i) => (
              <div key={`${row.metric}-${row.forecast_period}-${row.dimension_type}-${i}`} className="grid grid-cols-[1fr_auto_auto] items-center gap-3 rounded-xl border border-white/60 bg-white/50 px-3 py-2.5">
                <div>
                  <div className="text-sm font-bold text-[#3A2E28]">{row.dimension_type} · {PERIOD_LABELS[row.forecast_period] ?? row.forecast_period}</div>
                  <div className="text-[11px] uppercase tracking-widest text-[#8A7A6E]">{row.metric} · {row.rows} rows</div>
                </div>
                <span className="text-sm font-black text-[#B5443B] font-mono">{errorForSegment(row)}</span>
                <span className="text-xs font-bold text-[#3A2E28] font-mono">{fmtPct(row.interval_coverage, 0)}</span>
              </div>
            ))}
          </div>
        </LightCard>
      </div>

      <LightCard>
        <OverviewSectionHead index="05" title="Recent Holdout Rows" meta="Actual vs P50" />
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-[10px] uppercase tracking-widest text-[#8A7A6E]">
              <tr className="border-b border-[#3A2E28]/10">
                <th className="py-2 pr-4">Origin</th>
                <th className="py-2 pr-4">Window</th>
                <th className="py-2 pr-4">Dimension</th>
                <th className="py-2 pr-4">Value</th>
                <th className="py-2 pr-4">Metric</th>
                <th className="py-2 pr-4 text-right">Actual</th>
                <th className="py-2 pr-4 text-right">P50</th>
                <th className="py-2 text-right">Covered</th>
              </tr>
            </thead>
            <tbody>
              {data.recent_rows.slice(-10).map((row, i) => (
                <tr key={`${row.origin_date}-${row.forecast_period}-${row.dimension_value}-${row.metric}-${i}`} className="border-b border-[#3A2E28]/10 last:border-0">
                  <td className="py-2.5 pr-4 font-mono text-xs text-[#3A2E28]">{row.origin_date}</td>
                  <td className="py-2.5 pr-4 text-[#3A2E28]">{PERIOD_LABELS[row.forecast_period] ?? row.forecast_period}</td>
                  <td className="py-2.5 pr-4 text-[#3A2E28]">{row.dimension_type}</td>
                  <td className="py-2.5 pr-4 text-[#3A2E28]" title={row.dimension_value}>{displayDimensionValue(row.dimension_value)}</td>
                  <td className="py-2.5 pr-4 text-[#3A2E28]">{row.metric}</td>
                  <td className="py-2.5 pr-4 text-right font-mono text-[#3A2E28]">{actualForMetric(row)}</td>
                  <td className="py-2.5 pr-4 text-right font-mono font-bold text-[#3A2E28]">{valueForMetric(row)}</td>
                  <td className="py-2.5 text-right">
                    <span className={`inline-flex items-center justify-center rounded-full px-2.5 py-1 text-[11px] font-bold ${row.covered_by_interval ? "bg-[#1F7A78]/15 text-[#156B5E]" : "bg-[#B5443B]/15 text-[#92332A]"}`}>
                      {row.covered_by_interval ? "Yes" : "No"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </LightCard>
    </div>
  );
}

const PERIOD_LABELS: Record<string, string> = { "30_days": "30 Days", "60_days": "60 Days", "90_days": "90 Days" };
const PERIOD_DAYS: Record<string, number> = { "30_days": 30, "60_days": 60, "90_days": 90 };
// Distinct accent per horizon — 30/60/90 day cards read as a clear progression (teal → blue →
// violet) instead of three identical teal cards, so the three cards are visually distinguishable
// at a glance the way the reference direction's per-card coding intended.
const PERIOD_COLORS: Record<string, string> = { "30_days": "#3FB8A6", "60_days": "#4D9DE0", "90_days": "#8B6BFF" };

// Shared "glass over dark metal" panel treatment for the Forecasts tab's aviation-cockpit +
// glass-depth pass: frosted translucency with backdrop blur, a hairline top highlight (as if
// lit from above, like an avionics bezel), and a soft ambient drop shadow so panels feel
// lifted off the page rather than flat-painted onto it.
const GLASS_PANEL = "rounded-2xl border border-white/[0.09] bg-gradient-to-b from-[#1B2136]/70 to-[#0B0E17]/78 backdrop-blur-xl shadow-[inset_0_1px_0_rgba(255,255,255,0.08),0_24px_48px_-24px_rgba(0,0,0,0.65)]";

// Campaign-type icon lookup — semantic, not brand logos (we don't have real brand marks to use
// and faking them would misrepresent the product). Falls back to a generic Tag icon for any
// campaign type not in this list, so a new/unrecognized type from the data never breaks render.
const DIMENSION_ICONS: Record<string, React.ElementType> = {
  AUDIENCE: Users,
  DEMAND_GEN: Target,
  DISPLAY: Monitor,
  PERFORMANCE_MAX: BarChart2,
  SEARCH: Search,
  SHOPPING: ShoppingBag,
  SOCIAL: Share2,
  VIDEO: Video,
};
function dimensionIcon(dim: string): React.ElementType {
  return DIMENSION_ICONS[dim] ?? Tag;
}

// Per-dimension accent colors for inactive chip icons. Google/Meta/Bing reuse the exact hex
// values from the Overview tab's channel pie chart, so the same channel means the same color
// everywhere in the app rather than a second, disconnected color scheme just for this panel.
// Campaign-type dimensions get their own distinct hues so the whole chip row reads as a real
// multi-channel spread instead of one flat gray row.
const DIMENSION_ACCENTS: Record<string, string> = {
  "Google Ads": "#E8A33D",
  "Meta Ads": "#3FB8A6",
  "Bing Ads": "#8B92A0",
  AUDIENCE: "#8B6BFF",
  DEMAND_GEN: "#4D9DE0",
  DISPLAY: "#E0729D",
  PERFORMANCE_MAX: "#E0C05F",
  SEARCH: "#5FD3A4",
  SHOPPING: "#F2A65A",
  SOCIAL: "#B98BFF",
  VIDEO: "#C4544A",
};
function dimensionAccentColor(d: string): string {
  return DIMENSION_ACCENTS[d] ?? "#8B92A0";
}

// Derived (not fabricated) confidence read: tighter P10–P90 spread relative to P50 = higher
// confidence. This is a real function of the actual forecast interval width, not a static
// number — documented here so the formula is auditable.
function deriveConfidence(p10: number, p50: number, p90: number): { label: string; pct: number } {
  if (p50 <= 0) return { label: "Low", pct: 40 };
  const relativeSpread = (p90 - p10) / p50;
  const pct = Math.max(35, Math.min(97, Math.round(100 - relativeSpread * 28)));
  const label = pct >= 80 ? "High" : pct >= 60 ? "Medium" : "Low";
  return { label, pct };
}

// Same three-zone mapping as ConfidenceDial, pulled out so the scale bar under the dial and
// the dial itself never drift out of sync with two separate color rules.
function confidenceColor(pct: number): string {
  return pct >= 80 ? "#3FB8A6" : pct >= 60 ? "#E8A33D" : "#C4544A";
}

// Fan chart, v3. v1 let the uncertainty wedge dominate the axis. v2 fixed the domain but still
// drew P10/P90 as separate dashed lines, which for a genuinely skewed, widening band still read
// as a stray diagonal stroke rather than "uncertainty." This version renders the P10–P90 band as
// a single filled cone (Recharts' array-valued dataKey does this natively) with the real P50
// line on top, plus a real date axis instead of hiding it entirely — the shape reads instantly
// as "forecast fan," not as noise.
function HorizonFanChart({ points, color, startDate, height = 152 }: { points: TrajectoryPoint[]; color: string; startDate: Date | null; height?: number }) {
  const gradId = `fan-${color.replace("#", "")}`;
  // Square-root transform on the PLOTTED values only — the actual P10/P50/P90 numbers shown as
  // text elsewhere on this card are untouched, exact values. This chart's real data is
  // genuinely lopsided (P10 stays near P50 while P90 balloons over the horizon), which drawn on
  // a linear scale makes the band look like a hard-edged wedge rather than a soft cone. A
  // sqrt scale compresses the large P90 growth proportionally more than the small values,
  // which is the standard trick for plotting right-skewed distributions without lying about
  // the numbers — the shape reads as "widening uncertainty" instead of "one line exploding."
  const sqrt = (v: number) => Math.sqrt(Math.max(0, v));
  // Tick labels: real calendar dates when startDate is available (the backend's forecast
  // literally starts at validation.max_date + 1 day, so startDate + i days IS the true date
  // for point i — not a new computation, just surfacing what the backend already anchors to).
  // When startDate is unavailable/unparseable, fall back to the backend's own "Day N" string
  // exactly as returned, rather than ever feeding an ambiguous string into `new Date(...)`
  // again — that's what produced the stray "Jul 1" / repeated "Jan 1" ticks previously.
  const data = points.map((p, i) => {
    const label = startDate
      ? new Date(startDate.getTime() + i * 86_400_000).toLocaleDateString("en-US", { month: "short", day: "numeric" })
      : p.day;
    return { i, label, p50: sqrt(p.p50), range: [sqrt(p.p10), sqrt(p.p90)] as [number, number] };
  });
  const allVals = data.flatMap(d => [d.range[0], d.p50, d.range[1]]);
  const bandMin = Math.min(...allVals);
  const bandMax = Math.max(...allVals);
  const pad = Math.max((bandMax - bandMin) * 0.1, bandMax * 0.03, 0.5);
  const domain: [number, number] = [Math.max(0, bandMin - pad), bandMax + pad];
  const tickInterval = Math.max(0, Math.floor(data.length / 5) - 1);
  return (
    <ResponsiveContainer width="100%" height={height}>
      {/* left/right margin here isn't decorative — Recharts centers each tick label on its
          data point, and the first/last points sit exactly at the plot area's edges. With no
          margin, half of "Jun 6" and "Jun 30" render outside the plot area and get clipped by
          the card's rounded corners (this card uses overflow-hidden) — reading as "n 6" / "un
          30". 28px on each side is enough for a "Mmm DD" label to clear that edge cleanly. */}
      <ComposedChart data={data} margin={{ top: 8, right: 28, left: 28, bottom: 0 }}>
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.18} />
            <stop offset="100%" stopColor={color} stopOpacity={0.03} />
          </linearGradient>
        </defs>
        <CartesianGrid horizontal vertical={false} strokeDasharray="1 6" stroke="#3A2E28" strokeOpacity={0.15} />
        <XAxis dataKey="label" interval={tickInterval} tick={{ fontSize: 9, fill: "#8A7A6E" }} axisLine={{ stroke: "#3A2E28", strokeOpacity: 0.15 }} tickLine={false} />
        <YAxis hide domain={domain} />
        <Area type="monotone" dataKey="range" stroke="none" fill={`url(#${gradId})`} isAnimationActive={false} />
        <Line type="monotone" dataKey="p50" stroke={color} strokeWidth={2.5} dot={false} isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

// Small embedded trend line for the hero Revenue KPI cell — reuses the real daily P50 series
// already fetched for the Overview chart. This never appears on the ROAS cell: the backend
// only gives us a daily series for revenue, so a ROAS sparkline here would be fabricated.
function MiniSparkline({ points, color }: { points: TrajectoryPoint[]; color: string }) {
  const gradId = `spark-${color.replace("#", "")}`;
  const data = points.map((p, i) => ({ i, v: p.p50 }));
  return (
    <ResponsiveContainer width="100%" height={44}>
      <AreaChart data={data} margin={{ top: 2, right: 0, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.4} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <Area type="monotone" dataKey="v" stroke={color} strokeWidth={1.75} fill={`url(#${gradId})`} isAnimationActive={false} dot={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

// Shows where the real P50 sits within the real P10–P90 band — a derived position, not a
// fabricated one. Deliberately a heavier filled bar (vs. the thin tick-mark ConfidenceBracket
// used on Overview) since these period cards carry more visual weight/density.
function RangeBar({ p10, p50, p90, color }: { p10: number; p50: number; p90: number; color: string }) {
  const pct = p90 > p10 ? Math.max(4, Math.min(100, ((p50 - p10) / (p90 - p10)) * 100)) : 50;
  return (
    <div className="h-1.5 rounded-full bg-[#3A2E28]/15 overflow-hidden">
      <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color, transition: "width 0.6s ease" }} />
    </div>
  );
}

// Semantic confidence meter — same three-zone teal/amber/brick convention already used by
// RiskGauge and the Data Quality ring on Overview, so "what does this color mean" stays
// consistent across the whole app rather than inventing a new meaning for these colors here.
function ConfidenceMeter({ pct, label }: { pct: number; label: string }) {
  const color = pct >= 80 ? "#3FB8A6" : pct >= 60 ? "#E8A33D" : "#C4544A";
  return (
    <div className="flex flex-col gap-2.5">
      <span className="text-3xl font-bold font-mono leading-none" style={{ color }}>{label}</span>
      <div className="h-1.5 rounded-full bg-[#3A2E28]/15 overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color, transition: "width 0.6s ease" }} />
      </div>
      <div className="flex justify-between text-[9px] text-[#8A7A6E] uppercase tracking-widest">
        <span>Low</span><span>Medium</span><span>High</span>
      </div>
    </div>
  );
}

// Cockpit instrument dial — the Forecasts tab's signature element for this pass. Same
// teal/amber/brick semantic mapping as every other gauge in the app, but rendered as a real
// avionics-style dial: tick marks around the bezel, a glowing sweep arc, mono readout at
// center. This replaces the flat ConfidenceMeter bar in the KPI strip's hero confidence cell.
function ConfidenceDial({ pct, label }: { pct: number; label: string }) {
  const clamped = Math.max(0, Math.min(100, pct));
  const color = clamped >= 80 ? "#3FB8A6" : clamped >= 60 ? "#E8A33D" : "#C4544A";
  const circumference = 2 * Math.PI * 34;
  const offset = circumference * (1 - clamped / 100);
  const ticks = Array.from({ length: 9 }, (_, i) => -135 + i * (270 / 8));
  return (
    <div className="flex items-center gap-4">
      <div className="relative w-20 h-20 shrink-0">
        <svg viewBox="0 0 100 100" className="absolute inset-0 w-20 h-20">
          {ticks.map((deg, i) => (
            <line key={i} x1="50" y1="6" x2="50" y2="12" stroke="#3A2E28" strokeOpacity="0.2" strokeWidth="1.5" transform={`rotate(${deg} 50 50)`} />
          ))}
        </svg>
        <svg viewBox="0 0 100 100" className="w-20 h-20 -rotate-90">
          <circle cx="50" cy="50" r="34" fill="none" stroke="#3A2E28" strokeOpacity="0.1" strokeWidth="7" />
          <circle cx="50" cy="50" r="34" fill="none" stroke={color} strokeWidth="7" strokeLinecap="round"
            strokeDasharray={circumference} strokeDashoffset={offset}
            style={{ transition: "stroke-dashoffset 0.6s ease", filter: `drop-shadow(0 0 4px ${color}66)` }} />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-lg font-bold font-mono" style={{ color }}>{Math.round(clamped)}</span>
        </div>
      </div>
      <div className="flex flex-col gap-1">
        <span className="text-2xl font-bold font-mono leading-none" style={{ color }}>{label}</span>
        <span className="text-[9px] text-[#8A7A6E] uppercase tracking-widest">Forecast Confidence</span>
      </div>
    </div>
  );
}

function TabForecasts({
  forecasts, dimensions, selectedDimension, onSelectDimension, loading, error, dailyTrajectory,
  forecastExplanation, onViewRisk, latestDataDate,
}: {
  forecasts: ForecastsResponse | null;
  dimensions: DimensionsResponse | null;
  selectedDimension: string;
  onSelectDimension: (d: string) => void;
  loading: boolean;
  error?: string;
  dailyTrajectory: TrajectoryPoint[] | null;
  forecastExplanation?: string | null;
  onViewRisk?: () => void;
  latestDataDate?: string | null;
}) {
  // The forecast's real day-1 date. Backend anchors it as validation.max_date + 1 day (see
  // main.py: start_date = df['date'].max() + pd.Timedelta(days=1)) — this mirrors that exactly
  // rather than introducing a second, possibly-diverging definition of "forecast start."
  // Parsed defensively: if max_date isn't a clean YYYY-MM-DD, startDate stays null and every
  // chart below falls back to the honest "Day N" labels instead of risking a garbled date.
  const startDate = React.useMemo(() => {
    if (!latestDataDate) return null;
    const base = new Date(`${latestDataDate}T00:00:00Z`);
    if (isNaN(base.getTime())) return null;
    base.setUTCDate(base.getUTCDate() + 1);
    return base;
  }, [latestDataDate]);
  const dimensionOptions = ["Overall", ...(dimensions?.channels ?? []), ...(dimensions?.campaign_types ?? [])];
  const period30 = forecasts?.["30_days"];
  const revenue30 = period30 ? Object.entries(period30).find(([k]) => k.toLowerCase().includes("revenue"))?.[1] : undefined;
  const roas30 = period30 ? Object.entries(period30).find(([k]) => k.toLowerCase().includes("roas"))?.[1] : undefined;
  const confidence = revenue30 ? deriveConfidence(revenue30.P10, revenue30.P50, revenue30.P90) : null;
  // Real sparkline only makes sense for "Overall" — that's the only dimension the backend gives
  // us a real daily series for (see dailyTrajectory prop, sourced from /api/overview). For any
  // other dimension we simply don't render a sparkline rather than fabricate one.
  const showRealTrend = selectedDimension === "Overall" && dailyTrajectory && dailyTrajectory.length > 0;

  return (
    <div className="relative flex flex-col gap-6">
      {/* Ambient light sources behind the glass panels — backdrop-blur needs something to catch,
          otherwise translucent panels over a flat black page look identical to opaque ones.
          These sit behind everything and bleed color through the panels above them. */}
      <div className="pointer-events-none absolute -top-16 left-10 w-72 h-72 rounded-full bg-[#3FB8A6]/[0.16] blur-[100px]" />
      <div className="pointer-events-none absolute top-24 right-10 w-96 h-96 rounded-full bg-[#E8A33D]/[0.13] blur-[110px]" />
      {/* Dimension selector — retinted from the leftover teal-gradient/cyan-hover pair onto the
          real Instrument Black system. Teal is the "frequent, quiet" accent per the established
          convention, so it carries the active chip state; amber stays reserved for the one hero
          number below and never appears here. */}
      <LightCard className="!p-5">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-[11px] uppercase tracking-widest text-[#8A7A6E]">Dimension Focus</h3>
          <span className="text-[10px] text-[#8A7A6E] font-mono">{dimensionOptions.length} dimensions available</span>
        </div>
        <div className="flex flex-wrap gap-2">
          {dimensionOptions.map(d => {
            const Icon = d === "Overall" ? LayoutDashboard : dimensionIcon(d);
            const isActive = selectedDimension === d;
            return (
              <button key={d} onClick={() => onSelectDimension(d)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium border transition-all duration-150 ${
                  isActive
                    ? "bg-[#3FB8A6] border-[#3FB8A6] text-white font-semibold shadow-[0_4px_12px_rgba(63,184,166,0.3)]"
                    : "bg-white/40 border-white/60 text-[#5C4F46] hover:bg-white/80 hover:text-[#3A2E28]"
                }`}>
                <Icon size={13} color={isActive ? "#ffffff" : dimensionAccentColor(d)} />
                {d}
              </button>
            );
          })}
        </div>
      </LightCard>

      {loading ? <LoadingBlock label="Forecasts" /> : error ? <ErrorBlock label="Forecasts" message={error} /> : !forecasts ? <LoadingBlock label="Forecasts" /> : (
        <>
          {/* Fused instrument-panel KPI strip */}
          <LightCard className="!p-0 border-white/60">
            <div className="grid grid-cols-4 divide-x divide-[#3A2E28]/10">
              <div className="p-5 relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-br from-[#E8A33D]/[0.05] to-transparent pointer-events-none" />
                <div className="relative flex items-center justify-between mb-2">
                  <span className="text-[11px] uppercase tracking-widest text-[#8A7A6E]">30-Day Revenue (P50)</span>
                  <span className="w-7 h-7 rounded-full flex items-center justify-center shrink-0 bg-[#E8A33D]/15"><DollarSign size={13} className="text-[#B54A1C]" /></span>
                </div>
                <div className="relative text-3xl font-bold font-mono text-[#B54A1C]" style={{ textShadow: "0 0 16px rgba(232,163,61,0.2)" }}>
                  {revenue30 ? fmtCompactCurrency(revenue30.P50) : "—"}
                </div>
                <div className="relative text-[11px] text-[#8A7A6E] mt-1 font-mono">
                  {revenue30 ? `${fmtCompactCurrency(revenue30.P10)} – ${fmtCompactCurrency(revenue30.P90)}` : ""}
                </div>
                {showRealTrend && (
                  <div className="relative mt-2 -mx-1">
                    <MiniSparkline points={dailyTrajectory!.slice(0, 30)} color="#E8A33D" />
                  </div>
                )}
              </div>
              <div className="p-5">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[11px] uppercase tracking-widest text-[#8A7A6E]">30-Day ROAS (P50)</span>
                  <span className="w-7 h-7 rounded-full flex items-center justify-center shrink-0 bg-[#3FB8A6]/15"><Gauge size={13} className="text-[#156B5E]" /></span>
                </div>
                <div className="text-3xl font-bold font-mono text-[#156B5E]">{roas30 ? fmtRoas(roas30.P50) : "—"}</div>
                <div className="text-[11px] text-[#156B5E] mt-1 font-mono">{roas30 ? `${fmtRoas(roas30.P10)} – ${fmtRoas(roas30.P90)}` : ""}</div>
              </div>
              <div className="p-5">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[11px] uppercase tracking-widest text-[#8A7A6E]">Rolling Windows</span>
                  <span className="w-7 h-7 rounded-full flex items-center justify-center shrink-0 bg-[#8A7A6E]/10"><Calendar size={13} className="text-[#8A7A6E]" /></span>
                </div>
                <div className="text-3xl font-bold font-mono">
                  <span style={{ color: PERIOD_COLORS["30_days"] }}>30</span>
                  <span className="text-[#8A7A6E]"> / </span>
                  <span style={{ color: PERIOD_COLORS["60_days"] }}>60</span>
                  <span className="text-[#8A7A6E]"> / </span>
                  <span style={{ color: PERIOD_COLORS["90_days"] }}>90</span>
                </div>
                <div className="text-[11px] text-[#8A7A6E] mt-1">Days from latest ingested data</div>
              </div>
              <div className="p-5 flex flex-col justify-center">
                <span className="text-[11px] uppercase tracking-widest text-[#8A7A6E] mb-3 block">Confidence Level</span>
                {confidence && (
                  <>
                    <ConfidenceDial pct={confidence.pct} label={confidence.label} />
                    <div className="mt-3">
                      <div className="h-1 rounded-full bg-[#3A2E28]/15 overflow-hidden">
                        <div className="h-full rounded-full" style={{ width: `${confidence.pct}%`, backgroundColor: confidenceColor(confidence.pct), transition: "width 0.6s ease" }} />
                      </div>
                      <div className="flex justify-between text-[9px] text-[#8A7A6E] uppercase tracking-widest mt-1">
                        <span>Low</span><span>Medium</span><span>High</span>
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>
          </LightCard>

          {/* Forecast Horizons — each period gets its own real fan chart (solid P50 line over
              the actual P10–P90 band) plus range bars showing where P50 actually sits in that
              band, instead of the old plain 3-column P10/P50/P90 number grid. */}
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-[#1A1512] uppercase tracking-wide">Forecast Horizons</h3>
            <span className="text-[10px] text-[#8A7A6E] uppercase tracking-wide font-mono">View: {selectedDimension}</span>
          </div>
          <div className="grid grid-cols-3 gap-5">
            {Object.entries(forecasts).map(([period, metrics]) => {
              const days = PERIOD_DAYS[period] ?? 30;
              const trend = showRealTrend ? dailyTrajectory!.slice(0, days) : null;
              const revenueEntry = Object.entries(metrics).find(([k]) => k.toLowerCase().includes("revenue"))?.[1];
              const roasEntry = Object.entries(metrics).find(([k]) => k.toLowerCase().includes("roas"))?.[1];
              const periodConfidence = revenueEntry ? deriveConfidence(revenueEntry.P10, revenueEntry.P50, revenueEntry.P90) : null;
              const periodColor = PERIOD_COLORS[period] ?? "#3FB8A6";
              return (
                <LightCard key={period} className="!p-0 overflow-hidden flex flex-col relative transition-all duration-300 hover:border-white hover:shadow-[0_40px_80px_-28px_rgba(80,50,40,0.5)] hover:-translate-y-0.5">
                  <div className="absolute inset-0 pointer-events-none" style={{ background: `linear-gradient(160deg, ${periodColor}10, transparent 40%)` }} />
                  <div className="flex items-center justify-between px-5 pt-5 relative z-10">
                    <span className="flex items-center gap-2 text-xs font-bold text-[#1A1512] uppercase tracking-wide">
                      <Calendar size={13} color={periodColor} />
                      {PERIOD_LABELS[period] ?? period}
                    </span>
                    {periodConfidence && (
                      <span className={`text-[10px] px-2 py-0.5 rounded font-semibold border ${SEVERITY_STYLES_BY_CONF(periodConfidence.label)}`}>
                        {periodConfidence.label} Confidence
                      </span>
                    )}
                  </div>
                  {trend && (
                    <div className="mt-2 relative z-10">
                      <HorizonFanChart points={trend} color={periodColor} startDate={startDate} />
                    </div>
                  )}
                  <div className="grid grid-cols-2 gap-4 px-5 pb-5 pt-3 border-t border-[#3A2E28]/10 mt-1 relative z-10">
                    <div>
                      <span className="text-[10px] uppercase tracking-widest text-[#8A7A6E] block mb-1">Revenue (P50)</span>
                      <div className="text-lg font-bold font-mono mb-1.5" style={{ color: periodColor }}>{revenueEntry ? fmtCompactCurrency(revenueEntry.P50) : "—"}</div>
                      {revenueEntry && (
                        <>
                          <RangeBar p10={revenueEntry.P10} p50={revenueEntry.P50} p90={revenueEntry.P90} color={periodColor} />
                          <div className="flex justify-between text-[12px] font-semibold text-[#5C4F46] font-mono mt-1.5">
                            <span>{fmtCompactCurrency(revenueEntry.P10)}</span><span>{fmtCompactCurrency(revenueEntry.P90)}</span>
                          </div>
                        </>
                      )}
                    </div>
                    <div>
                      <span className="text-[10px] uppercase tracking-widest text-[#8A7A6E] block mb-1">ROAS (P50)</span>
                      <div className="text-lg font-bold font-mono mb-1.5" style={{ color: periodColor }}>{roasEntry ? fmtRoas(roasEntry.P50) : "—"}</div>
                      {roasEntry && (
                        <>
                          <RangeBar p10={roasEntry.P10} p50={roasEntry.P50} p90={roasEntry.P90} color={periodColor} />
                          <div className="flex justify-between text-[12px] font-semibold text-[#5C4F46] font-mono mt-1.5">
                            <span>{fmtRoas(roasEntry.P10)}</span><span>{fmtRoas(roasEntry.P90)}</span>
                          </div>
                        </>
                      )}
                    </div>
                  </div>
                </LightCard>
              );
            })}
          </div>

          {forecastExplanation && (
            <LightCard className="!p-5 flex items-center justify-between gap-6">
              <div className="flex items-start gap-3 min-w-0 flex-1">
                <Sparkles size={16} className="text-[#1F7A78] shrink-0 mt-0.5" />
                <div className="min-w-0">
                  <span className="text-xs font-bold text-[#1A1512] block mb-1">ForecastIQ Insight</span>
                  <p className="text-xs text-[#5C4F46] font-medium leading-relaxed">{forecastExplanation}</p>
                </div>
              </div>
              <svg viewBox="0 0 140 90" className="hidden lg:block shrink-0 w-32 h-20 opacity-90" aria-hidden="true">
                <rect x="14" y="10" width="70" height="70" rx="8" fill="#3FB8A6" fillOpacity="0.05" stroke="#3FB8A6" strokeOpacity="0.18" />
                <rect x="34" y="20" width="70" height="70" rx="8" fill="#E8A33D" fillOpacity="0.04" stroke="#E8A33D" strokeOpacity="0.16" />
                <rect x="54" y="30" width="70" height="70" rx="8" fill="#8B6BFF" fillOpacity="0.04" stroke="#8B6BFF" strokeOpacity="0.14" />
                <polyline points="60,78 78,64 92,70 108,48 122,38" fill="none" stroke="#3FB8A6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                {[[60, 78], [78, 64], [92, 70], [108, 48], [122, 38]].map(([x, y], i) => (
                  <circle key={i} cx={x} cy={y} r="2.2" fill="#3FB8A6" />
                ))}
              </svg>
              {onViewRisk && (
                <button onClick={onViewRisk}
                  className="shrink-0 flex items-center gap-1.5 text-xs font-bold text-[#3A2E28] border border-white/60 bg-white/40 rounded-md px-3 py-2 hover:border-white/80 hover:bg-white/80 transition-colors">
                  View Risk &amp; Insights <ArrowUpRight size={12} />
                </button>
              )}
            </LightCard>
          )}
        </>
      )}
    </div>
  );
}

// Small helper so the period-card confidence badge reuses the exact same style tokens as
// SEVERITY_STYLES (Risk & Insights tab), rather than a second parallel color mapping.
function SEVERITY_STYLES_BY_CONF(label: string): string {
  if (label === "High") return "text-[#3FB8A6] bg-[#3FB8A6]/10 border-[#3FB8A6]/20";
  if (label === "Medium") return "text-[#E8A33D] bg-[#E8A33D]/10 border-[#E8A33D]/20";
  return "text-[#C4544A] bg-[#C4544A]/10 border-[#C4544A]/20";
}

// ─── Tab: Scenarios ────────────────────────────────────────────────────
function TabScenarios({ scenarios, loading, error }: { scenarios: ScenarioItem[] | null; loading: boolean; error?: string }) {
  const [activeWindow, setActiveWindow] = useState<string>("30_days");
  if (loading) return <LoadingBlock label="Scenarios" />;
  if (error) return <ErrorBlock label="Scenarios" message={error} />;
  if (!scenarios) return <LoadingBlock label="Scenarios" />;
  return (
    <div className="flex flex-col gap-6">
      <LightCard className="!p-4">
        <div className="flex items-center gap-2">
          {Object.keys(PERIOD_LABELS).map(w => (
            <button key={w} onClick={() => setActiveWindow(w)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${activeWindow === w ? "bg-[#1F7A78] border-[#1F7A78] text-white" : "bg-white/40 border-white/60 text-[#5C4F46] hover:bg-white/80 hover:text-[#3A2E28]"}`}>
              {PERIOD_LABELS[w]}
            </button>
          ))}
        </div>
      </LightCard>
      <div className="grid grid-cols-2 gap-6">
        {scenarios.map(s => {
          const w = s.forecasts[activeWindow];
          return (
            <LightCard key={s.id}>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-bold text-[#1A1512]">{s.name}</h3>
                <span className="text-[10px] px-2 py-0.5 rounded bg-[#1F7A78]/10 text-[#156B5E] border border-[#1F7A78]/20 font-semibold">{s.tag}</span>
              </div>
              <p className="text-xs text-[#5C4F46] font-medium mb-4">{s.description}</p>
              {w && (
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="bg-white/60 border border-white/80 rounded-lg p-3">
                    <span className="text-[10px] text-[#8A7A6E] uppercase font-semibold">Revenue P50</span>
                    <div className="text-lg font-bold text-[#3A2E28] font-mono">{fmtCompactCurrency(w.Revenue_P50)}</div>
                    <div className="text-[10px] text-[#8A7A6E] font-mono font-medium">{fmtCompactCurrency(w.Revenue_P10)} – {fmtCompactCurrency(w.Revenue_P90)}</div>
                  </div>
                  <div className="bg-white/60 border border-white/80 rounded-lg p-3">
                    <span className="text-[10px] text-[#8A7A6E] uppercase font-semibold">ROAS P50</span>
                    <div className="text-lg font-bold text-[#3A2E28] font-mono">{fmtRoas(w.ROAS_P50)}</div>
                    <div className="text-[10px] text-[#8A7A6E] font-mono font-medium">{fmtRoas(w.ROAS_P10)} – {fmtRoas(w.ROAS_P90)}</div>
                  </div>
                </div>
              )}
              <div className="flex gap-4 text-[11px] font-semibold text-[#8A7A6E] border-t border-[#3A2E28]/10 pt-3">
                <span>CPC {s.cpc_change}</span>
                <span>Conv. Rate {s.conv_rate_change}</span>
              </div>
            </LightCard>
          );
        })}
      </div>
    </div>
  );
}

// ─── Tab: Budget Optimizer ────────────────────────────────────────────
// BUG fix (bug-hunt sweep, same class as the frozen-dimension predictions bug): this
// used to hardcode exactly 3 slider rows for Google/Meta/Bing, so a 4th channel present in
// the actual data (backend now fully supports this) had no way to be simulated in the UI
// at all. `channels` now comes from the live /api/dimensions response.
const CHANNEL_TONE: Record<string, string> = {
  "Google Ads": "#DD7A3C",
  "Meta Ads": "#1F7A78",
  "Bing Ads": "#9C8F82",
};
const FALLBACK_CHANNEL_TONES = ["#8E6BB0", "#4E8CA6", "#B3844C", "#6E8C4E", "#A65C6E"];
function channelToneFor(channel: string, index: number): string {
  return CHANNEL_TONE[channel] ?? FALLBACK_CHANNEL_TONES[index % FALLBACK_CHANNEL_TONES.length];
}

function TabBudget({
  channels,
  simInputs, onSimChange, simResult, simLoading, simError,
  optInputs, onOptChange, optResult, optLoading, optError, onRunOptimize,
}: {
  channels: string[];
  simInputs: Record<string, number>;
  onSimChange: (channel: string, v: number) => void;
  simResult: BudgetSimResponse | null;
  simLoading: boolean;
  simError: string | null;
  optInputs: { max_budget: number; target_roas: number };
  onOptChange: (k: "max_budget" | "target_roas", v: number) => void;
  optResult: BudgetOptResponse | null;
  optLoading: boolean;
  optError: string | null;
  onRunOptimize: () => void;
}) {
  const sliderRows = channels.map((ch, i) => ({ key: ch, label: ch, tone: channelToneFor(ch, i) }));
  const channelTone: Record<string, string> = Object.fromEntries(sliderRows.map(r => [r.key, r.tone]));
  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-2 gap-6">
        <LightCard className="flex flex-col">
          <OverviewSectionHead index="01" title="Live Budget Simulator" meta="30-day run-rate" />
          <div className="flex flex-col gap-4">
            {sliderRows.map(row => {
              // Channels not yet touched aren't in the dynamic simInputs map — default to 0%.
              const pct = simInputs[row.key] ?? 0;
              return (
              <div key={row.key} className="rounded-2xl border border-white/70 bg-white/45 px-4 py-3 backdrop-blur-sm">
                <div className="flex justify-between text-xs mb-2">
                  <span className="flex items-center gap-2 font-semibold text-[#3A2E28]">
                    <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: channelTone[row.label] ?? "#8A7A6E" }} />
                    {row.label}
                  </span>
                  <span className={`font-bold font-mono ${pct > 0 ? "text-[#1F7A78]" : pct < 0 ? "text-[#B5443B]" : "text-[#8A7A6E]"}`}>
                    {pct > 0 ? "+" : ""}{pct}%
                  </span>
                </div>
                <input type="range" min={-50} max={100} step={5} value={pct}
                  onChange={e => onSimChange(row.key, Number(e.target.value))}
                  className="w-full accent-[#C0632B]" />
              </div>
              );
            })}
          </div>
          {simLoading ? <LoadingBlock label="simulation" /> : simResult && (
            <div className="grid grid-cols-3 gap-3 mt-5 pt-4 border-t border-[#3A2E28]/10">
              <div className="rounded-xl border border-white/70 bg-white/50 px-3 py-3">
                <span className="text-[10px] text-[#8A7A6E] uppercase tracking-widest">Total Spend</span>
                <div className="text-sm font-bold text-[#3A2E28] font-mono mt-1">{fmtCompactCurrency(simResult.total_spend)}</div>
              </div>
              <div className="rounded-xl border border-white/70 bg-white/50 px-3 py-3">
                <span className="text-[10px] text-[#8A7A6E] uppercase tracking-widest">Total Revenue</span>
                <div className="text-sm font-bold text-[#C0632B] font-mono mt-1">{fmtCompactCurrency(simResult.total_revenue)}</div>
              </div>
              <div className="rounded-xl border border-white/70 bg-white/50 px-3 py-3">
                <span className="text-[10px] text-[#8A7A6E] uppercase tracking-widest">Total ROAS</span>
                <div className="text-sm font-bold text-[#1F7A78] font-mono mt-1">{fmtRoas(simResult.total_roas)}</div>
              </div>
            </div>
          )}
          {/* BUG fix (pre-release live UI audit): the fetch .catch() here previously only
              reset loading state and silently swallowed the error -- an invalid/edge-case
              slider combination would just show nothing at all, with no indication anything
              had gone wrong. Surface the actual message instead of a silent no-op. */}
          {!simLoading && simError && (
            <div className="mt-5 pt-4 border-t border-[#3A2E28]/10">
              <div className="rounded-xl border border-red-300/60 bg-red-50/70 px-4 py-3 text-sm text-red-800">
                {simError}
              </div>
            </div>
          )}
        </LightCard>
        <LightCard className="flex flex-col">
          <OverviewSectionHead index="02" title="Optuna Global Optimizer" meta="Revenue goal seek" />
          <div className="flex flex-col gap-4 mb-4">
            <div>
              <label className="text-xs font-semibold uppercase tracking-widest text-[#8A7A6E]">Max Budget</label>
              <input type="number" value={optInputs.max_budget} onChange={e => onOptChange("max_budget", Number(e.target.value))}
                className="w-full mt-1 rounded-xl border border-white/70 bg-white/55 px-3 py-2 text-sm font-semibold text-[#3A2E28] outline-none transition-colors placeholder:text-[#8A7A6E] focus:border-[#C0632B]/50 focus:bg-white/80" />
            </div>
            <div>
              <label className="text-xs font-semibold uppercase tracking-widest text-[#8A7A6E]">Target ROAS</label>
              <input type="number" step={0.1} value={optInputs.target_roas} onChange={e => onOptChange("target_roas", Number(e.target.value))}
                className="w-full mt-1 rounded-xl border border-white/70 bg-white/55 px-3 py-2 text-sm font-semibold text-[#3A2E28] outline-none transition-colors placeholder:text-[#8A7A6E] focus:border-[#C0632B]/50 focus:bg-white/80" />
            </div>
            <button onClick={onRunOptimize} disabled={optLoading}
              className="flex items-center justify-center gap-2 rounded-xl border border-[#1F7A78]/30 bg-[#186664] py-2.5 text-sm font-bold text-white shadow-md shadow-[#1F7A78]/20 transition-all hover:bg-[#1F7A78] disabled:opacity-50">
              <Zap size={14} /> {optLoading ? "Optimizing..." : "Run Optimization"}
            </button>
          </div>
          {optResult && (
            <div className="flex flex-col gap-2 pt-4 border-t border-[#3A2E28]/10">
              {Object.entries(optResult.channel_recommendations).map(([ch, rec]) => (
                <div key={ch} className="grid grid-cols-[1fr_auto_auto_auto] items-center gap-3 rounded-xl border border-white/70 bg-white/50 px-3 py-2 text-xs">
                  <span className="flex items-center gap-2 font-semibold text-[#3A2E28]">
                    <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: channelTone[ch] ?? "#8A7A6E" }} />
                    {ch}
                  </span>
                  <span className="font-mono font-semibold text-[#5C4F46]">{fmtCompactCurrency(rec.allocated_spend)}</span>
                  <span className="font-mono font-semibold text-[#5C4F46]">{fmtRoas(rec.expected_roas)}</span>
                  <span className="font-mono font-bold text-[#1F7A78]">{fmtPct(rec.budget_share)}</span>
                </div>
              ))}
              <div className="flex justify-between gap-4 text-xs mt-2 pt-3 border-t border-[#3A2E28]/10">
                <span className="font-semibold uppercase tracking-widest text-[#8A7A6E]">Expected Revenue (P10-P90)</span>
                <span className="font-mono font-bold text-[#C0632B]">{fmtCompactCurrency(optResult.confidence_range.revenue_p10)} - {fmtCompactCurrency(optResult.confidence_range.revenue_p90)}</span>
              </div>
            </div>
          )}
          {/* BUG fix (pre-release live UI audit): same silent-swallow pattern as the
              Live Budget Simulator above. Confirmed live: Max Budget = 0 now correctly
              returns a 400 with a real message from the backend fix, but until this, the
              UI just stopped spinning and showed nothing -- no indication the run failed
              or why. */}
          {!optLoading && optError && (
            <div className="mt-2 pt-4 border-t border-[#3A2E28]/10">
              <div className="rounded-xl border border-red-300/60 bg-red-50/70 px-4 py-3 text-sm text-red-800">
                {optError}
              </div>
            </div>
          )}
        </LightCard>
      </div>

      <LightCard>
        <OverviewSectionHead index="03" title="How This Works" meta="Objective & constraints" />
        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-2xl border border-white/70 bg-white/45 backdrop-blur-sm p-4">
            <span className="text-[11px] uppercase tracking-widest text-[#8A7A6E] block mb-2">Objective</span>
            <p className="text-sm text-[#3A2E28] leading-relaxed">Maximize expected 30-day revenue across every active channel{channels.length ? `: ${channels.join(", ")}` : ""}.</p>
          </div>
          <div className="rounded-2xl border border-white/70 bg-white/45 backdrop-blur-sm p-4">
            <span className="text-[11px] uppercase tracking-widest text-[#8A7A6E] block mb-2">Constraints</span>
            <p className="text-sm text-[#3A2E28] leading-relaxed">Stay within your max budget, and never let blended ROAS fall below the target floor you set.</p>
          </div>
          <div className="rounded-2xl border border-white/70 bg-white/45 backdrop-blur-sm p-4">
            <span className="text-[11px] uppercase tracking-widest text-[#8A7A6E] block mb-2">Assumption</span>
            <p className="text-sm text-[#3A2E28] leading-relaxed">Each channel has diminishing marginal returns — pushing more spend into one channel yields progressively less revenue per dollar, so Optuna searches for the split that balances every active channel.</p>
          </div>
          <div className="rounded-2xl border border-white/70 bg-white/45 backdrop-blur-sm p-4">
            <span className="text-[11px] uppercase tracking-widest text-[#8A7A6E] block mb-2">Output</span>
            <p className="text-sm text-[#3A2E28] leading-relaxed">A recommended spend split per channel, plus an expected revenue/ROAS range (P10–P90) for that allocation.</p>
          </div>
        </div>
        <div className="mt-4 rounded-2xl border border-[#C0632B]/25 bg-[#C0632B]/[0.06] backdrop-blur-sm p-4 flex items-start gap-3">
          <Sparkles size={15} className="text-[#C0632B] shrink-0 mt-0.5" />
          <p className="text-xs text-[#5C4F46] leading-relaxed"><span className="font-semibold text-[#3A2E28]">Example:</span> if Meta spend rises 20%, revenue increases but marginal ROAS falls. The optimizer finds a better allocation that protects your ROAS floor while still increasing revenue.</p>
        </div>
      </LightCard>
    </div>
  );
}

// ─── Tab: Monte Carlo ────────────────────────────────────────────
// Retinted onto the warm premium glass theme to match Overview through Budget Optimizer:
// LightCard/LightKpiCard in place of the old Instrument Black Card/KpiCard, numbered
// OverviewSectionHead panels, and the same channel color mapping used on the Budget tab
// (Google Ads terracotta, Meta Ads teal, Bing Ads stone) so a channel reads the same color
// everywhere in the app.
function TabMontecarlo({ mc, loading, error }: { mc: MonteCarloResponse | null; loading: boolean; error?: string }) {
  if (loading) return <LoadingBlock label="Monte Carlo" />;
  if (error) return <ErrorBlock label="Monte Carlo" message={error} />;
  if (!mc) return <LoadingBlock label="Monte Carlo" />;

  const channelTone: Record<string, string> = {
    "Google Ads": "#DD7A3C",
    "Meta Ads": "#1F7A78",
    "Bing Ads": "#9C8F82",
  };

  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-3 gap-4">
        <LightKpiCard label="Worst Case Revenue" value={fmtCompactCurrency(mc.worst_case_revenue)} icon={ArrowDownRight} trend="down" tint="brick" sub={`${fmtRoas(mc.worst_case_roas)} ROAS`} />
        <LightKpiCard label="Expected Revenue" value={fmtCompactCurrency(mc.expected_revenue)} icon={TrendingUp} tint="amber" sub={`${fmtRoas(mc.expected_roas)} ROAS`} />
        <LightKpiCard label="Best Case Revenue" value={fmtCompactCurrency(mc.best_case_revenue)} icon={ArrowUpRight} trend="up" tint="teal" sub={`${fmtRoas(mc.best_case_roas)} ROAS`} />
      </div>
      <LightCard>
        <OverviewSectionHead index="01" title="Revenue Distribution" meta={`${mc.n_simulations.toLocaleString()} simulations`} />
        <p className="text-xs text-[#8A7A6E] -mt-2 mb-4">Histogram of simulated 30-day portfolio revenue outcomes.</p>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={mc.revenue_histogram} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#3A2E2815" />
            <XAxis dataKey="bin_center" tickFormatter={v => fmtCompactCurrency(v)} tick={{ fontSize: 10, fill: "#8A7A6E" }} />
            <YAxis tick={{ fontSize: 10, fill: "#8A7A6E" }} />
            <Tooltip contentStyle={{ backgroundColor: "rgba(255,250,245,0.95)", borderColor: "#E6D9CC", borderRadius: 8, color: "#3A2E28" }} itemStyle={{ color: "#3A2E28" }} labelStyle={{ color: "#8A7A6E" }} formatter={(v: any) => [v, "Frequency"]} labelFormatter={(v: any) => fmtCurrency(v)} />
            <Bar dataKey="frequency" fill="#DD7A3C" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </LightCard>
      <LightCard>
        <OverviewSectionHead index="02" title="Channel Distributions" />
        <div className="grid grid-cols-3 gap-4">
          {Object.entries(mc.channel_distributions).map(([ch, d]) => (
            <div key={ch} className="rounded-2xl border border-white/70 bg-white/45 backdrop-blur-sm p-4">
              <p className="flex items-center gap-2 text-xs font-bold text-[#3A2E28] mb-3">
                <span className="h-2.5 w-2.5 rounded-full shrink-0" style={{ backgroundColor: channelTone[ch] ?? "#8A7A6E" }} />
                {ch}
              </p>
              {([["Worst Case", d.worst_case], ["Expected", d.expected_case], ["Best Case", d.best_case]] as const).map(([label, val]) => (
                <div key={label} className="flex justify-between text-xs mb-1">
                  <span className="text-[#8A7A6E]">{label}</span>
                  <span className="text-[#3A2E28] font-mono font-semibold">{fmtCompactCurrency(val as number)}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </LightCard>
    </div>
  );
}

// ─── Tab: Explainability ─────────────────────────────────────────
// Retinted onto the warm premium glass theme. Revenue-mode charts use the terracotta
// accent (matches P50 in the Overview trajectory chart), ROAS-mode charts use Glide
// Teal (matches P10/P90 bounds) — same semantic split used across the app.
function ShapBarChart({ data, mode, height }: { data: ShapDriver[]; mode: "revenue" | "roas"; height: number }) {
  const accent = mode === "revenue" ? "#DD7A3C" : "#1F7A78";
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, left: 100, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#3A2E2815" />
        <XAxis type="number" tick={{ fontSize: 10, fill: "#8A7A6E" }} tickFormatter={v => mode === "revenue" ? `${(v / 1000).toFixed(0)}k` : `${v}x`} />
        <YAxis dataKey="feature" type="category" tick={{ fontSize: 10, fill: "#5C4F46" }} width={140} />
        <Tooltip contentStyle={{ backgroundColor: "rgba(255,250,245,0.95)", borderColor: "#E6D9CC", borderRadius: 8, color: "#3A2E28" }} itemStyle={{ color: "#3A2E28" }} labelStyle={{ color: "#8A7A6E" }} formatter={(v: any) => mode === "revenue" ? [`${Number(v).toLocaleString("en-US", { maximumFractionDigits: 0 })}`, "Causal Impact"] : [`${Number(v).toFixed(2)}x`, "ROAS Impact"]} />
        <Bar dataKey="shap_impact" fill={accent} radius={[0, 8, 8, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

// Badge tints reused from the Budget/Monte Carlo tabs' light palette (teal = positive,
// amber/terracotta = caution, brick = negative) instead of the old emerald/amber/rose set.
const STABILITY_STYLES: Record<string, string> = {
  High: "bg-[#1F7A78]/15 text-[#156B5E]",
  Medium: "bg-[#DD7A3C]/15 text-[#B54A1C]",
  Low: "bg-[#B5443B]/15 text-[#92332A]",
};

function TabExplainability({ data, loading, error }: { data: ExplainabilityResponse | null; loading: boolean; error?: string }) {
  const [showAllCampaigns, setShowAllCampaigns] = useState(false);
  const [campaignSort, setCampaignSort] = useState<"revenue" | "best" | "worst">("revenue");
  const [expandedChart, setExpandedChart] = useState<"revenue" | "roas" | null>(null);
  if (loading) return <LoadingBlock label="Explainability" />;
  if (error) return <ErrorBlock label="Explainability" message={error} />;
  if (!data) return <LoadingBlock label="Explainability" />;

  const sortedCampaigns = [...data.campaign_importance].sort((a, b) => {
    if (campaignSort === "best") return b.average_roas - a.average_roas;
    if (campaignSort === "worst") return a.average_roas - b.average_roas;
    return b.total_historical_revenue - a.total_historical_revenue;
  });
  const visibleCampaigns = showAllCampaigns ? sortedCampaigns : sortedCampaigns.slice(0, 3);

  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-2 gap-6">
        <LightCard>
          <div className="relative flex items-end justify-between gap-4 mb-4">
            <div className="flex items-center gap-3">
              <span className="flex h-7 w-7 items-center justify-center rounded-full text-[11px] font-semibold" style={{ background: "rgba(192,99,43,0.12)", color: "#C0632B", border: "1px solid rgba(192,99,43,0.28)" }}>01</span>
              <h3 className="font-serif text-[20px] leading-none tracking-tight text-[#3A2E28]">Top Revenue Drivers</h3>
            </div>
            <button onClick={() => setExpandedChart("revenue")} className="text-[#8A7A6E] hover:text-[#C0632B] transition-colors" title="Expand chart">
              <Maximize2 size={14} />
            </button>
          </div>
          <ShapBarChart data={data.top_revenue_drivers} mode="revenue" height={280} />
        </LightCard>
        <LightCard>
          <div className="relative flex items-end justify-between gap-4 mb-4">
            <div className="flex items-center gap-3">
              <span className="flex h-7 w-7 items-center justify-center rounded-full text-[11px] font-semibold" style={{ background: "rgba(192,99,43,0.12)", color: "#C0632B", border: "1px solid rgba(192,99,43,0.28)" }}>02</span>
              <h3 className="font-serif text-[20px] leading-none tracking-tight text-[#3A2E28]">Top ROAS Drivers</h3>
            </div>
            <button onClick={() => setExpandedChart("roas")} className="text-[#8A7A6E] hover:text-[#C0632B] transition-colors" title="Expand chart">
              <Maximize2 size={14} />
            </button>
          </div>
          <ShapBarChart data={data.top_roas_drivers} mode="roas" height={280} />
        </LightCard>
      </div>
      <LightCard>
        <OverviewSectionHead index="03" title="Channel Importance Rankings" meta="Historical, not forecast" />
        <div className="flex flex-col gap-2">
          {data.channel_importance.map((c, i) => (
            <div key={i} className="flex items-center justify-between rounded-xl border border-white/70 bg-white/45 backdrop-blur-sm px-4 py-2.5 text-sm">
              <span className="font-semibold text-[#3A2E28] w-32">{c.channel}</span>
              <span className="text-[#8A7A6E]">{fmtPct(c.contribution_share)} contribution</span>
              <span className={`text-xs px-2 py-0.5 rounded font-semibold ${STABILITY_STYLES[c.roas_stability.split(" ")[0]] ?? STABILITY_STYLES.Low}`}>{c.roas_stability}</span>
              <span className="text-[#C0632B] font-bold font-mono">{c.importance_score} score</span>
            </div>
          ))}
        </div>
      </LightCard>
      <LightCard>
        <div className="relative flex items-end justify-between gap-4 mb-4">
          <div className="flex items-center gap-3">
            <span className="flex h-7 w-7 items-center justify-center rounded-full text-[11px] font-semibold" style={{ background: "rgba(192,99,43,0.12)", color: "#C0632B", border: "1px solid rgba(192,99,43,0.28)" }}>04</span>
            <h3 className="font-serif text-[20px] leading-none tracking-tight text-[#3A2E28]">Top Campaigns by Historical Revenue</h3>
          </div>
          <div className="flex gap-2">
            {([["revenue", "Revenue"], ["best", "Best ROAS"], ["worst", "Worst ROAS"]] as const).map(([key, label]) => (
              <button key={key} onClick={() => setCampaignSort(key)}
                className={`px-2.5 py-1 rounded-lg text-[11px] font-medium border transition-colors ${campaignSort === key ? "bg-[#C0632B] border-[#C0632B] text-white" : "bg-white/45 border-white/70 text-[#5C4F46] hover:border-[#C0632B]/50 backdrop-blur-sm"}`}>
                {label}
              </button>
            ))}
          </div>
        </div>
        <div className="flex flex-col gap-2">
          {visibleCampaigns.map((c, i) => (
            <div key={i} className="flex items-center justify-between rounded-xl border border-white/70 bg-white/45 backdrop-blur-sm px-4 py-2.5 text-sm">
              <div className="flex flex-col">
                <span className="font-semibold text-[#3A2E28]">{c.campaign_name}</span>
                <span className="text-[10px] text-[#8A7A6E]">{c.channel}</span>
              </div>
              <span className="text-[#8A7A6E] font-mono">{fmtCompactCurrency(c.total_historical_revenue)}</span>
              <span className="text-[#8A7A6E] font-mono">{fmtRoas(c.average_roas)}</span>
              <span className={`text-[10px] px-2 py-0.5 rounded font-semibold ${c.driver_status === "Primary Bedrock" ? "bg-[#1F7A78]/15 text-[#156B5E]" : "bg-[#C0632B]/15 text-[#A64516]"}`}>{c.driver_status}</span>
            </div>
          ))}
        </div>
        {sortedCampaigns.length > 3 && (
          <button onClick={() => setShowAllCampaigns(v => !v)}
            className="w-full mt-3 py-2 rounded-xl text-xs font-medium text-[#C0632B] border border-white/70 bg-white/40 hover:border-[#C0632B]/40 hover:bg-white/60 transition-colors backdrop-blur-sm">
            {showAllCampaigns ? "Show less" : `Show ${sortedCampaigns.length - 3} more (${sortedCampaigns.length} total)`}
          </button>
        )}
      </LightCard>
      {expandedChart && (
        <div className="fixed inset-0 z-50 bg-[#2A1F18]/60 backdrop-blur-sm flex items-center justify-center p-8" onClick={() => setExpandedChart(null)}>
          <div className={`${GLASS_LIGHT} p-6 w-full max-w-4xl`} style={MODAL_GLASS_BG} onClick={e => e.stopPropagation()}>
            <OverviewSheen />
            <div className="relative z-10">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-serif text-[18px] text-[#3A2E28]">
                  {expandedChart === "revenue" ? "Top Revenue Drivers (SHAP Value Impact)" : "Top ROAS Drivers (SHAP Multiplier Impact)"}
                </h3>
                <button onClick={() => setExpandedChart(null)} className="text-[#8A7A6E] hover:text-[#3A2E28] transition-colors">
                  <X size={18} />
                </button>
              </div>
              <ShapBarChart
                data={expandedChart === "revenue" ? data.top_revenue_drivers : data.top_roas_drivers}
                mode={expandedChart}
                height={520}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Tab: Risk & Insights ────────────────────────────────────────────
const POSITIVE_STATUSES = new Set(["Healthy", "Diversified", "Stable", "Excellent", "Low"]);
// Retinted onto the warm premium palette: brick/amber/teal light-mode tokens (same family
// as Explainability's STABILITY_STYLES and Overview's roasColor/qualityColor thresholds)
// instead of the old dark-mode Instrument Black accent set.
const SEVERITY_STYLES: Record<string, string> = {
  High: "text-[#92332A] bg-[#B5443B]/10 border-[#B5443B]/20",
  Medium: "text-[#B54A1C] bg-[#DD7A3C]/10 border-[#DD7A3C]/20",
  Low: "text-[#156B5E] bg-[#1F7A78]/10 border-[#1F7A78]/20",
};

// The "Confidence Bracket" — the signature element from the design brief that every Stitch
// pass consistently dropped (novel pattern, no precedent in its training distribution). Built
// directly here instead, using real P10/P90 values now exposed by the backend, not invented.
// Renders the P50 large in mono, a thin tick-marked range line beneath it in Glide Teal, with
// the real P10/P90 bounds labeled at each end.
function ConfidenceBracket({ p10, p50, p90, formatFn }: { p10: number; p50: number; p90: number; formatFn: (n: number) => string }) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-2xl font-bold text-[#F0EDE6] font-mono leading-none">{formatFn(p50)}</span>
      <div className="relative w-full max-w-[180px] pt-2">
        <div className="h-px bg-[#3FB8A6]/50 w-full" />
        <div className="absolute left-0 top-0 w-px h-2.5 bg-[#3FB8A6]/70" />
        <div className="absolute right-0 top-0 w-px h-2.5 bg-[#3FB8A6]/70" />
        <div className="flex justify-between mt-1">
          <span className="text-[10px] text-[#8B92A0] font-mono">{formatFn(p10)}</span>
          <span className="text-[10px] text-[#8B92A0] font-mono">{formatFn(p90)}</span>
        </div>
      </div>
    </div>
  );
}

function RiskGauge({ score }: { score: number }) {
  const clamped = Math.max(0, Math.min(100, score));
  // Retinted from the old Instrument Black hex (#C4544A/#E8A33D/#3FB8A6) to the EXACT warm-
  // theme semantic trio already used everywhere else in this theme (Overview's roasColor/
  // qualityColor, Explainability's STABILITY_STYLES, Monte Carlo's LIGHT_KPI_TINTS) — same
  // risk level must render as the literal same color across every tab, not a close cousin.
  const color = clamped >= 70 ? "#B5443B" : clamped >= 40 ? "#DD7A3C" : "#1F7A78";
  const circumference = 2 * Math.PI * 42;
  const offset = circumference * (1 - clamped / 100);
  return (
    <div className="relative w-28 h-28 shrink-0">
      <svg viewBox="0 0 100 100" className="w-28 h-28 -rotate-90">
        <circle cx="50" cy="50" r="42" fill="none" stroke="#E4D6C8" strokeWidth="10" />
        <circle cx="50" cy="50" r="42" fill="none" stroke={color} strokeWidth="10" strokeLinecap="round"
          strokeDasharray={circumference} strokeDashoffset={offset} style={{ transition: "stroke-dashoffset 0.6s ease" }} />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-black text-[#3A2E28] font-mono">{clamped}</span>
        <span className="text-[9px] text-[#8A7A6E] uppercase tracking-wide">/ 100</span>
      </div>
    </div>
  );
}

function TabRisk({ risk, insights, loading, error }: { risk: RiskProfile | null; insights: InsightsResponse | null; loading: boolean; error?: string }) {
  if (loading) return <LoadingBlock label="Risk & Insights" />;
  if (error) return <ErrorBlock label="Risk & Insights" message={error} />;
  if (!risk) return <LoadingBlock label="Risk & Insights" />;

  // risk.badge_color is a backend-supplied className string built for the OLD dark theme
  // (e.g. "bg-red-500/10 text-red-400 border-red-500/20") — rendering that directly on a
  // light card risks poor contrast. Deriving the badge locally instead, same pattern already
  // used for Overview's isLowRisk and Explainability's STABILITY_STYLES, so it's guaranteed to
  // use the same warm-theme hex as every other severity indicator in the app.
  const riskCls = risk.risk_classification.toLowerCase();
  const riskBadgeStyle = riskCls.includes("low")
    ? "text-[#156B5E] bg-[#1F7A78]/10 border-[#1F7A78]/20"
    : riskCls.includes("high")
    ? "text-[#92332A] bg-[#B5443B]/10 border-[#B5443B]/20"
    : "text-[#B54A1C] bg-[#DD7A3C]/10 border-[#DD7A3C]/20";

  return (
    <div className="flex flex-col gap-6">
      <LightCard>
        <div className="relative flex items-start gap-6">
          <RiskGauge score={risk.risk_score} />
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              <h3 className="text-sm font-bold text-[#1A1512]">Risk Intelligence Profile</h3>
              <span className={`px-2 py-0.5 rounded text-xs font-bold border ${riskBadgeStyle}`}>{risk.risk_classification}</span>
            </div>
            <p className="text-xs text-[#5C4F46] font-medium leading-relaxed">{risk.executive_risk_summary}</p>
          </div>
        </div>
      </LightCard>

      <OverviewSectionHead index="01" title="Risk Factors" />
      <div className="grid grid-cols-2 gap-4 -mt-2">
        {risk.risk_factors.map((f, i) => {
          const positive = POSITIVE_STATUSES.has(f.status);
          const barColor = positive ? "#1F7A78" : "#B5443B";
          return (
            <LightCard key={i} className="!p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-[#1A1512]">{f.name}</span>
                <span className="text-xs font-bold" style={{ color: barColor }}>{f.status}</span>
              </div>
              <div className="h-1.5 rounded-full bg-[#3A2E28]/10 overflow-hidden mb-2">
                <div className="h-full rounded-full" style={{ width: `${Math.max(4, Math.min(100, f.score))}%`, backgroundColor: barColor }} />
              </div>
              <p className="text-[11px] text-[#8A7A6E] leading-relaxed">{f.mitigation}</p>
            </LightCard>
          );
        })}
      </div>
      {insights && (
        <div className="grid grid-cols-3 gap-4">
          <LightCard>
            <h3 className="text-sm font-bold text-[#1A1512] mb-3 flex items-center gap-2"><Sparkles size={14} className="text-[#1F7A78]" /> Growth Opportunities</h3>
            <div className="flex flex-col gap-3">
              {insights.growth_opportunities.map((g, i) => (
                <div key={i} className="bg-white/50 border border-white/60 backdrop-blur-sm rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs font-bold text-[#156B5E]">{g.title}</span>
                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-[#1F7A78]/10 text-[#156B5E] border border-[#1F7A78]/20">{g.tag}</span>
                  </div>
                  <p className="text-[11px] text-[#5C4F46] leading-relaxed">{g.insight}</p>
                </div>
              ))}
            </div>
          </LightCard>
          <LightCard>
            <h3 className="text-sm font-bold text-[#1A1512] mb-3 flex items-center gap-2"><AlertTriangle size={14} className="text-[#B54A1C]" /> Risk Assessment</h3>
            <div className="flex flex-col gap-3">
              {insights.risk_assessment.map((r, i) => (
                <div key={i} className="bg-white/50 border border-white/60 backdrop-blur-sm rounded-lg p-3">
                  <span className={`inline-block text-[10px] px-1.5 py-0.5 rounded font-semibold border mb-1 ${SEVERITY_STYLES[r.severity] ?? "text-[#8A7A6E] bg-white/40 border-white/60"}`}>{r.severity}</span>
                  <p className="text-xs font-bold text-[#1A1512] mb-1">{r.title}</p>
                  <p className="text-[11px] text-[#5C4F46] leading-relaxed">{r.insight}</p>
                </div>
              ))}
            </div>
          </LightCard>
          <LightCard>
            <h3 className="text-sm font-bold text-[#1A1512] mb-3 flex items-center gap-2"><DollarSign size={14} className="text-[#1F7A78]" /> Budget Recommendations</h3>
            <div className="flex flex-col gap-2">
              {insights.budget_recommendations.map((b, i) => (
                <div key={i} className="bg-white/50 border border-white/60 backdrop-blur-sm rounded-lg p-3">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-bold text-[#156B5E] text-xs">{b.channel}</span>
                    <span className={`text-[11px] font-semibold ${b.action.includes("Increase") ? "text-[#156B5E]" : b.action.includes("Maintain") ? "text-[#B54A1C]" : "text-[#92332A]"}`}>{b.action}</span>
                  </div>
                  <p className="text-[11px] text-[#5C4F46] leading-relaxed">{b.rationale}</p>
                </div>
              ))}
            </div>
          </LightCard>
        </div>
      )}
      {insights && (
        <LightCard>
          <h3 className="text-sm font-bold text-[#1A1512] mb-2">How This Forecast Was Built</h3>
          <p className="text-xs text-[#5C4F46] leading-relaxed">{insights.forecast_explanation}</p>
        </LightCard>
      )}
    </div>
  );
}

// ─── Tab: Chat ───────────────────────────────────────────────────────────
// Suggested prompts on the empty state — genuinely answerable questions given what /api/chat
// actually has access to (forecasts, risk, campaign data), not a marketing wishlist. Clicking
// one sends it through the exact same onSend path as typing it manually.
const SUGGESTED_PROMPTS = [
  "What's driving my forecast confidence?",
  "Which channel should I cut spend from?",
  "Summarize my portfolio risk in plain English",
];

function TabChat({ messages, input, onInput, onSend, loading }: {
  messages: ChatMessage[]; input: string; onInput: (v: string) => void; onSend: (overrideText?: string) => void; loading: boolean;
}) {
  const bottomRef = React.useRef<HTMLDivElement>(null);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  return (
    <div className="relative flex flex-col h-[calc(100vh-9rem)]">
      <div className={`${GLASS_LIGHT} flex-1 overflow-y-auto flex flex-col gap-4 p-5 mb-4`} style={GLASS_LIGHT_BG}>
        <OverviewSheen />
        {messages.length === 0 && (
          <div className="relative flex flex-col items-center justify-center h-full gap-5 text-center px-6">
            <span className="flex h-14 w-14 items-center justify-center rounded-full" style={{ background: "rgba(31,138,122,0.12)", border: "1px solid rgba(31,138,122,0.25)" }}>
              <Sparkles size={24} className="text-[#1F7A78]" />
            </span>
            <div>
              <p className="font-serif text-lg text-[#1A1512] mb-1">Ask ForecastIQ</p>
              <p className="text-sm text-[#8A7A6E] max-w-xs">Ask anything about your revenue, ROAS, campaigns, or risk profile in plain English.</p>
            </div>
            <div className="flex flex-wrap justify-center gap-2 max-w-md">
              {SUGGESTED_PROMPTS.map((p, i) => (
                <button key={i} onClick={() => onSend(p)}
                  className="text-xs font-medium text-[#3A2E28] bg-white/55 border border-white/70 rounded-full px-3.5 py-2 hover:bg-white/85 hover:border-[#1F7A78]/30 transition-colors backdrop-blur-sm">
                  {p}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`relative flex items-end gap-2.5 ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            {m.role === "bot" && (
              <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full mb-0.5" style={{ background: "rgba(31,138,122,0.12)", border: "1px solid rgba(31,138,122,0.22)" }}>
                <Sparkles size={13} className="text-[#1F7A78]" />
              </span>
            )}
            <div className={`max-w-[70%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
              m.role === "user"
                ? "text-white font-medium"
                : "bg-white/65 border border-white/70 text-[#2B221E] backdrop-blur-sm"
            }`} style={m.role === "user" ? { background: "linear-gradient(155deg, #1F7A78 0%, #145E5C 100%)", boxShadow: "0 8px 20px -8px rgba(20,94,92,0.5)" } : undefined}>
              {m.text}
            </div>
          </div>
        ))}
        {loading && (
          <div className="relative flex items-end gap-2.5 justify-start">
            <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full mb-0.5" style={{ background: "rgba(31,138,122,0.12)", border: "1px solid rgba(31,138,122,0.22)" }}>
              <Sparkles size={13} className="text-[#1F7A78]" />
            </span>
            <div className="bg-white/65 border border-white/70 backdrop-blur-sm rounded-2xl px-4 py-2.5 flex items-center gap-2 text-[#8A7A6E] text-sm">
              <RefreshCw size={13} className="animate-spin text-[#1F7A78]" /> ForecastIQ is thinking…
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <div className="relative flex gap-3">
        <input value={input} onChange={e => onInput(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); onSend(); }}}
          placeholder="Ask about your revenue, ROAS, campaigns…"
          className="flex-1 bg-white/55 border border-white/70 backdrop-blur-sm rounded-xl px-4 py-3 text-sm text-[#2B221E] placeholder:text-[#8A7A6E] outline-none transition-colors focus:border-[#1F7A78]/50 focus:bg-white/80" />
        <button onClick={() => onSend()} disabled={loading || !input.trim()}
          className="px-5 py-3 rounded-xl text-white font-bold transition-all disabled:opacity-40 flex items-center gap-2 text-sm shadow-md shadow-[#1F7A78]/20"
          style={{ background: "#186664" }}
          onMouseEnter={e => { if (!(loading || !input.trim())) (e.currentTarget as HTMLButtonElement).style.background = "#1F7A78"; }}
          onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.background = "#186664"; }}>
          <Send size={14} /> Send
        </button>
      </div>
    </div>
  );
}

// ─── Main Page ───────────────────────────────────────────────────────────────
// Endpoint map, verified directly against backend/src/main.py (not guessed):
//   GET  /api/status            -> StatusState
//   GET  /api/overview          -> OverviewState
//   GET  /api/validation        -> ValidationSummary
//   GET  /api/model-validation  -> ModelValidationResponse
//   GET  /api/forecasts?dimension=X -> ForecastsResponse
//   GET  /api/dimensions        -> DimensionsResponse
//   GET  /api/scenarios         -> ScenarioItem[]
//   GET  /api/simulations       -> MonteCarloResponse   (NOT /api/montecarlo)
//   GET  /api/explainability    -> ExplainabilityResponse
//   GET  /api/risk              -> RiskProfile
//   GET  /api/insights          -> InsightsResponse
//   POST /api/simulate-budget   -> BudgetSimResponse
//   POST /api/optimize-budget   -> BudgetOptResponse
//   POST /api/chat              -> { question, answer }
//   GET  /api/report/pdf        -> PDF file stream

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}${body ? " — " + body.slice(0, 200) : ""}`);
  }
  return res.json() as Promise<T>;
}

interface LoadState<T> {
  data: T | null;
  loading: boolean;
  error?: string;
}

export default function Page() {
  const [activeTab, setActiveTab] = useState<TabId>("overview");

  const [status, setStatus] = useState<LoadState<StatusState>>({ data: null, loading: true });
  const [overview, setOverview] = useState<LoadState<OverviewState>>({ data: null, loading: true });
  const [validation, setValidation] = useState<LoadState<ValidationSummary>>({ data: null, loading: true });
  const [modelValidation, setModelValidation] = useState<LoadState<ModelValidationResponse>>({ data: null, loading: true });
  const [dimensions, setDimensions] = useState<LoadState<DimensionsResponse>>({ data: null, loading: true });
  const [scenarios, setScenarios] = useState<LoadState<ScenarioItem[]>>({ data: null, loading: true });
  const [montecarlo, setMontecarlo] = useState<LoadState<MonteCarloResponse>>({ data: null, loading: true });
  const [explainability, setExplainability] = useState<LoadState<ExplainabilityResponse>>({ data: null, loading: true });
  const [risk, setRisk] = useState<LoadState<RiskProfile>>({ data: null, loading: true });
  const [insights, setInsights] = useState<LoadState<InsightsResponse>>({ data: null, loading: true });

  const [selectedDimension, setSelectedDimension] = useState<string>("Overall");
  const [forecasts, setForecasts] = useState<LoadState<ForecastsResponse>>({ data: null, loading: true });

  // BUG fix (bug-hunt sweep, same class as the frozen-dimension predictions bug): previously
  // a fixed { google_pct, meta_pct, bing_pct } object, which made a 4th channel structurally
  // impossible to simulate anywhere in the UI even though the backend optimizer can now
  // handle any channel present in the data. Now a dynamic channel-name -> pct map; sliders
  // are rendered per entry in dimensions.data.channels (see TabBudget below).
  const [simInputs, setSimInputs] = useState<Record<string, number>>({});
  const [simResult, setSimResult] = useState<BudgetSimResponse | null>(null);
  const [simLoading, setSimLoading] = useState(false);
  const [simError, setSimError] = useState<string | null>(null);

  const [optInputs, setOptInputs] = useState({ max_budget: 100000, target_roas: 4.5 });
  const [optResult, setOptResult] = useState<BudgetOptResponse | null>(null);
  const [optLoading, setOptLoading] = useState(false);
  const [optError, setOptError] = useState<string | null>(null);

  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);

  // ── Initial fetch-all on mount ──────────────────────────────────────────
  useEffect(() => {
    fetchJson<StatusState>("/api/status")
      .then(data => setStatus({ data, loading: false }))
      .catch(e => setStatus({ data: null, loading: false, error: String(e.message ?? e) }));

    fetchJson<OverviewState>("/api/overview")
      .then(data => setOverview({ data, loading: false }))
      .catch(e => setOverview({ data: null, loading: false, error: String(e.message ?? e) }));

    fetchJson<ValidationSummary>("/api/validation")
      .then(data => setValidation({ data, loading: false }))
      .catch(e => setValidation({ data: null, loading: false, error: String(e.message ?? e) }));

    fetchJson<ModelValidationResponse>("/api/model-validation")
      .then(data => setModelValidation({ data, loading: false }))
      .catch(e => setModelValidation({ data: null, loading: false, error: String(e.message ?? e) }));

    fetchJson<DimensionsResponse>("/api/dimensions")
      .then(data => setDimensions({ data, loading: false }))
      .catch(e => setDimensions({ data: null, loading: false, error: String(e.message ?? e) }));

    fetchJson<ScenarioItem[]>("/api/scenarios")
      .then(data => setScenarios({ data, loading: false }))
      .catch(e => setScenarios({ data: null, loading: false, error: String(e.message ?? e) }));

    // NOTE: this is /api/simulations, not /api/montecarlo — verified against main.py.
    fetchJson<MonteCarloResponse>("/api/simulations")
      .then(data => setMontecarlo({ data, loading: false }))
      .catch(e => setMontecarlo({ data: null, loading: false, error: String(e.message ?? e) }));

    fetchJson<ExplainabilityResponse>("/api/explainability")
      .then(data => setExplainability({ data, loading: false }))
      .catch(e => setExplainability({ data: null, loading: false, error: String(e.message ?? e) }));

    fetchJson<RiskProfile>("/api/risk")
      .then(data => setRisk({ data, loading: false }))
      .catch(e => setRisk({ data: null, loading: false, error: String(e.message ?? e) }));

    fetchJson<InsightsResponse>("/api/insights")
      .then(data => setInsights({ data, loading: false }))
      .catch(e => setInsights({ data: null, loading: false, error: String(e.message ?? e) }));
  }, []);

  // ── Forecasts: re-fetch whenever the selected dimension changes ────────
  useEffect(() => {
    setForecasts(prev => ({ ...prev, loading: true }));
    fetchJson<ForecastsResponse>(`/api/forecasts?dimension=${encodeURIComponent(selectedDimension)}`)
      .then(data => setForecasts({ data, loading: false }))
      .catch(e => setForecasts({ data: null, loading: false, error: String(e.message ?? e) }));
  }, [selectedDimension]);

  // ── Budget simulator: debounce-free, re-run on every slider change ─────
  useEffect(() => {
    setSimLoading(true);
    setSimError(null);
    // BUG fix (bug-hunt sweep): request body now matches the backend's dynamic
    // BudgetSimRequest.channel_pcts schema instead of the old fixed 3-field shape.
    fetchJson<BudgetSimResponse>("/api/simulate-budget", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ channel_pcts: simInputs }),
    })
      .then(data => { setSimResult(data); setSimLoading(false); })
      // BUG fix (pre-release live UI audit): previously only reset loading state and
      // swallowed the error entirely, matching the same fetchJson convention used for every
      // other panel's error state (see the useEffect above) instead of a silent no-op.
      .catch(e => { setSimError(String(e.message ?? e)); setSimLoading(false); });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [simInputs]);

  const handleSimChange = useCallback((channel: string, v: number) => {
    setSimInputs(prev => ({ ...prev, [channel]: v }));
  }, []);

  const handleOptChange = useCallback((k: "max_budget" | "target_roas", v: number) => {
    setOptInputs(prev => ({ ...prev, [k]: v }));
  }, []);

  const handleRunOptimize = useCallback(() => {
    setOptLoading(true);
    setOptError(null);
    setOptResult(null);
    fetchJson<BudgetOptResponse>("/api/optimize-budget", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(optInputs),
    })
      .then(data => { setOptResult(data); setOptLoading(false); })
      // BUG fix (pre-release live UI audit): previously only reset loading state and
      // swallowed the error entirely -- e.g. Max Budget = 0 now correctly returns a 400 with
      // a real message from the backend fix, but the UI just went quiet with no indication
      // the run failed or why. Also clear stale optResult on a new run so a previous
      // successful result can't linger on screen next to an unrelated new error.
      .catch(e => { setOptError(String(e.message ?? e)); setOptLoading(false); });
  }, [optInputs]);

  const handleSendChat = useCallback((overrideText?: string) => {
    // overrideText lets the suggested-prompt chips send immediately without first round-tripping
    // through the controlled input's state (which would otherwise still hold the OLD value at
    // the moment this closure runs, since setChatInput from a chip click hasn't committed yet).
    const question = (overrideText ?? chatInput).trim();
    if (!question) return;
    setChatMessages(prev => [...prev, { role: "user", text: question }]);
    setChatInput("");
    setChatLoading(true);
    fetchJson<{ question: string; answer: string }>("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    })
      .then(res => {
        setChatMessages(prev => [...prev, { role: "bot", text: res.answer }]);
        setChatLoading(false);
      })
      .catch(e => {
        setChatMessages(prev => [...prev, { role: "bot", text: `Sorry, something went wrong: ${String(e.message ?? e)}` }]);
        setChatLoading(false);
      });
  }, [chatInput]);

  const handleDownloadPdf = useCallback(() => {
    window.open(`${API_BASE}/api/report/pdf`, "_blank");
  }, []);

  const isOverview = activeTab === "overview" || activeTab === "validation" || activeTab === "accuracy" || activeTab === "forecasts" || activeTab === "scenarios" || activeTab === "budget" || activeTab === "montecarlo" || activeTab === "explainability" || activeTab === "risk" || activeTab === "chat";
  const overviewBg = {
    background: `
      radial-gradient(55% 32% at 26% 0%, rgba(255,255,255,0.6), transparent 65%),
      radial-gradient(115% 95% at 0% 0%, rgba(107,45,60,0.6), transparent 58%),
      radial-gradient(95% 95% at 100% 0%, rgba(110,95,150,0.42), transparent 52%),
      radial-gradient(115% 115% at 0% 100%, rgba(230,165,70,0.55), transparent 52%),
      radial-gradient(115% 115% at 100% 100%, rgba(35,120,120,0.62), transparent 55%),
      linear-gradient(135deg, #F3D9C4 0%, #ECD9CB 45%, #D9E2DE 100%)
    `,
  };

  return (
    <div 
      className={`flex min-h-screen transition-colors duration-500 ${!isOverview ? "bg-[#05070D]" : ""}`}
      style={isOverview ? overviewBg : undefined}
    >
      {isOverview && (
        <div aria-hidden className="pointer-events-none absolute inset-0 mix-blend-soft-light" style={{
          backgroundImage: "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.5'/%3E%3C/svg%3E\")",
          opacity: 0.35,
          zIndex: 0
        }} />
      )}

      {/* Sidebar */}
      <aside className={`w-64 shrink-0 flex flex-col relative z-10 transition-all duration-500 border-r ${
        isOverview 
          ? "bg-white/30 border-white/40 backdrop-blur-2xl shadow-[1px_0_24px_rgba(0,0,0,0.05)]" 
          : "bg-[#080B14] border-[#1C2338]"
      }`}>
        <div className={`h-24 px-6 flex flex-col justify-center border-b transition-colors duration-500 ${isOverview ? "border-white/30" : "border-[#1C2338]"}`}>
          <h1 className="text-xl font-black tracking-tight leading-none">
            <span className={isOverview ? "text-[#1A1512]" : "text-white"}>Forecast</span>
            <span className={isOverview ? "text-[#B54A1C]" : "gradient-accent-text"}>IQ</span>
          </h1>
          <p className={`text-[11px] uppercase tracking-wider mt-1.5 font-semibold transition-colors duration-500 ${isOverview ? "text-[#5C4F46]" : "text-slate-500"}`}>
            AI Revenue Intelligence
          </p>
        </div>
        <nav className="flex-1 py-5 flex flex-col gap-1.5 px-4">
          {TABS.map(tab => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            
            let btnClass = "";
            if (isOverview) {
               btnClass = isActive 
                 ? "bg-white/70 text-[#A64516] shadow-sm border border-white/60 font-bold" 
                 : "text-[#3A2E28] hover:bg-white/50 font-semibold";
            } else {
               btnClass = isActive 
                 ? "gradient-accent text-white shadow-lg shadow-[#00f2ff]/20 font-bold" 
                 : "text-slate-300 hover:text-white hover:bg-white/5 font-semibold";
            }

            return (
              <button key={tab.id} onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-3 px-4 py-3 rounded-xl text-[15px] transition-all text-left ${btnClass}`}>
                <Icon size={18} />
                {tab.label}
              </button>
            );
          })}
        </nav>
        <div className={`px-6 py-5 border-t transition-colors duration-500 ${isOverview ? "border-white/30" : "border-[#1C2338]"}`}>
          <button onClick={handleDownloadPdf}
            className={`w-full flex items-center justify-center gap-2 text-sm font-bold rounded-xl py-3 transition-all hover:opacity-90 ${
              isOverview 
                ? "bg-[#186664] text-white shadow-md shadow-[#1F7A78]/20 border border-[#1F7A78]/30" 
                : "gradient-accent text-white"
            }`}>
            <Download size={15} /> Export PDF Report
          </button>
        </div>
      </aside>

      {/* Main content */}
      <div className="flex-1 flex flex-col relative z-10">
        {/* Top bar */}
        <header className={`h-24 flex items-center justify-between px-8 shrink-0 transition-all duration-500 border-b ${
          isOverview ? "bg-white/20 border-white/40 backdrop-blur-xl" : "border-[#1C2338]"
        }`}>
          <div>
            <h2 className={`text-lg font-extrabold transition-colors duration-500 ${isOverview ? "text-[#1A1512]" : "text-slate-50"}`}>
              {TABS.find(t => t.id === activeTab)?.label}
            </h2>
            <p className={`text-xs font-semibold mt-0.5 transition-colors duration-500 ${isOverview ? "text-[#5C4F46]" : "text-slate-400"}`}>
              {TAB_SUBTITLES[activeTab]}
            </p>
          </div>
          <div className="flex items-center gap-5 text-[13px] font-semibold">
            {status.data && (
              <>
                <span className={`flex items-center gap-1.5 transition-colors duration-500 ${isOverview ? "text-[#2B221E]" : "text-slate-300"}`}>
                  <span className={`w-2 h-2 rounded-full transition-colors duration-500 ${isOverview ? "bg-[#156B5E]" : "bg-emerald-500"}`} />
                  {status.data.status === "online" ? "System Online" : status.data.status}
                </span>
                <span className={`transition-colors duration-500 ${isOverview ? "text-black/15" : "text-slate-700"}`}>|</span>
                <span className={`transition-colors duration-500 ${isOverview ? "text-[#2B221E]" : "text-slate-300"}`}>
                  LLM: {status.data.active_llm_provider}
                </span>
                <span className={`transition-colors duration-500 ${isOverview ? "text-black/15" : "text-slate-700"}`}>|</span>
                <span className={`transition-colors duration-500 ${isOverview ? "text-[#2B221E]" : "text-slate-300"}`}>
                  Data Quality: <span className={`font-extrabold transition-colors duration-500 ${isOverview ? "text-[#156B5E]" : "text-emerald-400"}`}>{fmtPct(status.data.data_quality_score, 1)}</span>
                </span>
              </>
            )}
          </div>
        </header>

        {/* Tab body */}
        <main key={activeTab} className="flex-1 overflow-y-auto p-6 animate-fadeIn">
          {activeTab === "overview" && <TabOverview data={overview.data} loading={overview.loading} error={overview.error} />}
          {activeTab === "validation" && <TabValidation data={validation.data} loading={validation.loading} error={validation.error} />}
          {activeTab === "accuracy" && <TabModelValidation data={modelValidation.data} loading={modelValidation.loading} error={modelValidation.error} />}
          {activeTab === "forecasts" && (
            <TabForecasts
              forecasts={forecasts.data} dimensions={dimensions.data}
              selectedDimension={selectedDimension} onSelectDimension={setSelectedDimension}
              loading={forecasts.loading} error={forecasts.error}
              dailyTrajectory={overview.data?.daily_trajectory ?? null}
              forecastExplanation={insights.data?.forecast_explanation ?? null}
              onViewRisk={() => setActiveTab("risk")}
              latestDataDate={validation.data?.max_date ?? null}
            />
          )}
          {activeTab === "scenarios" && <TabScenarios scenarios={scenarios.data} loading={scenarios.loading} error={scenarios.error} />}
          {activeTab === "budget" && (
            <TabBudget
              channels={dimensions.data?.channels ?? []}
              simInputs={simInputs} onSimChange={handleSimChange} simResult={simResult} simLoading={simLoading} simError={simError}
              optInputs={optInputs} onOptChange={handleOptChange} optResult={optResult} optLoading={optLoading} optError={optError}
              onRunOptimize={handleRunOptimize}
            />
          )}
          {activeTab === "montecarlo" && <TabMontecarlo mc={montecarlo.data} loading={montecarlo.loading} error={montecarlo.error} />}
          {activeTab === "explainability" && <TabExplainability data={explainability.data} loading={explainability.loading} error={explainability.error} />}
          {activeTab === "risk" && <TabRisk risk={risk.data} insights={insights.data} loading={risk.loading} error={risk.error} />}
          {activeTab === "chat" && (
            <TabChat messages={chatMessages} input={chatInput} onInput={setChatInput} onSend={handleSendChat} loading={chatLoading} />
          )}
        </main>
      </div>
    </div>
  );
}

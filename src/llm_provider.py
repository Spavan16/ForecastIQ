from abc import ABC, abstractmethod
import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import requests
import time
from src.utils import get_logger

logger = get_logger("LLMProvider")

# Load environment variables from .env
load_dotenv()


def _strip_markdown(text: str) -> str:
    """
    Gemini ignores the "plain prose, not markdown" instruction in the system prompt often
    enough that relying on prompting alone isn't reliable -- observed live output included
    **bold** spans and occasional bullet/header lines even with that instruction present.
    This is a deterministic post-process safety net applied to every live Gemini response
    (both generate_insight and ask_question) so the dashboard never renders raw markdown
    syntax to the user regardless of what the model actually returns. Does not touch
    MockLLMProvider output, which is already plain prose by construction.
    """
    import re
    # Bold/italic emphasis: **text**, __text__, *text*, _text_ -> text
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)", r"\1", text)
    # Markdown headers: "## Heading" -> "Heading"
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Bullet/numbered list markers at line start -> plain text, joined with a space
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    # Collapse the newlines those list items left behind into single spaces, then
    # normalize any leftover double-spacing from the substitutions above.
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

class BaseLLMProvider(ABC):
    """Abstract base class for AI/LLM functionality."""

    @abstractmethod
    def generate_insight(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate an executive business insight based on prompt and analytical context."""
        pass

    @abstractmethod
    def ask_question(self, question: str, analytics_context: Dict[str, Any]) -> str:
        """Answer a chat question from an executive marketing user."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the active provider."""
        pass


class MockLLMProvider(BaseLLMProvider):
    """
    Offline mock provider. Guarantees no crashes and works fully offline, returning
    template-based executive summaries when no LLM API key is configured.
    """
    def generate_insight(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        # Offline template responses stay generic on anything not actually passed into
        # `context` (only revenue_90d/roas_90d are guaranteed here — no per-channel
        # breakdown), pointing to the tabs that compute channel-specific numbers instead of
        # asserting an unsupported claim.
        total_rev = "$14.6M"
        overall_roas = "4.25x"
        if context and "revenue_90d" in context:
            rev_val = context["revenue_90d"]
            total_rev = f"${rev_val/1e6:.1f}M" if rev_val >= 1e6 else f"${rev_val:,.0f}"
        if context and "roas_90d" in context:
            overall_roas = f"{context['roas_90d']:.2f}x"

        if "risk" in prompt.lower() or "volatility" in prompt.lower():
            return (
                "[Mock Insight Mode Enabled] Risk Assessment: portfolio risk depends on recent "
                "ROAS volatility and channel concentration - see the Risk tab for the exact "
                "classification and driving factors. Recommendation: set CPA/CPC caps on "
                "whichever channel is currently showing the highest volatility."
            )
        elif "budget" in prompt.lower() or "allocate" in prompt.lower():
            return (
                f"[Mock Insight Mode Enabled] Strategic Budget Recommendations: to maximize total "
                f"revenue beyond {total_rev}, run the Optuna budget optimizer to find the exact "
                f"revenue-maximizing spend split across your active channels at the current "
                f"{overall_roas} blended ROAS."
            )
        elif "causal" in prompt.lower() or "summary" in prompt.lower():
            # BUG fix (code audit, "thin/templated causal summary"): this branch used to stay
            # fully generic ("your highest-volume channels", "whichever channel is showing the
            # highest CPM/CPC volatility") even when predict.py's run.sh-invoked call already
            # had real per-channel numbers available -- it just wasn't reading them. Now reads
            # context["channel_breakdown"] the same way ask_question() already does elsewhere
            # in this file, and falls back to the old generic phrasing only when that key is
            # genuinely absent (e.g. a caller with just revenue_90d/roas_90d and no channel
            # detail), so this still degrades safely rather than crashing.
            channel_ctx: Dict[str, Any] = (context or {}).get("channel_breakdown", {}) or {}
            ctype_ctx: Dict[str, Any] = (context or {}).get("campaign_type_breakdown", {}) or {}
            if channel_ctx:
                top_ch, top_row = max(channel_ctx.items(), key=lambda kv: kv[1].get("revenue", 0.0))
                volatile_ch, volatile_row = max(
                    channel_ctx.items(), key=lambda kv: kv[1].get("interval_width_pct", 0.0)
                )
                summary = (
                    f"[Mock Insight Mode Enabled] Causal Inference Summary: historical revenue growth "
                    f"({total_rev} projected over 90 days) is concentrated in {top_ch}, which accounts "
                    f"for {top_row.get('share_pct', 0.0):.1f}% of forecast channel revenue at "
                    f"{top_row.get('roas', 0.0):.2f}x ROAS - see the Explainability tab for the full "
                    f"SHAP-ranked channel and campaign contributions. Blended ROAS ({overall_roas}) is "
                    f"most sensitive to {volatile_ch}, which carries the widest 90-day ROAS forecast "
                    f"interval of the active channels ({volatile_row.get('interval_width_pct', 0.0):.0f}% "
                    f"of its own P50 value) - a swing there moves the blended number the most."
                )
                # Second synthesis layer: within that revenue mix, which campaign type is the
                # actual mechanism driving it. Grounded in the same real per-dimension numbers
                # already computed by produce_full_predictions_table(), not a new assumption.
                if ctype_ctx:
                    top_ctype, top_ctype_row = max(ctype_ctx.items(), key=lambda kv: kv[1].get("revenue", 0.0))
                    summary += (
                        f" Within that mix, {top_ctype} is the largest single campaign-type driver, "
                        f"accounting for {top_ctype_row.get('share_pct', 0.0):.1f}% of forecast revenue "
                        f"at {top_ctype_row.get('roas', 0.0):.2f}x ROAS - the Explainability tab's "
                        f"SHAP breakdown traces exactly how much of {top_ch}'s share above is carried "
                        f"by this campaign type versus the rest of its mix."
                    )
                return summary

            return (
                f"[Mock Insight Mode Enabled] Causal Inference Summary: historical revenue growth "
                f"({total_rev} projected over 90 days) is concentrated in your highest-volume "
                f"channels - see the Explainability tab for the exact SHAP-ranked channel and "
                f"campaign contributions. Blended ROAS ({overall_roas}) is most sensitive to "
                f"whichever channel currently carries the widest forecast interval."
            )
        else:
            return (
                f"[Mock Insight Mode Enabled] Executive AI Analyst Summary: the 90-day multi-channel "
                f"forecast projects total expected revenue reaching {total_rev} at an aggregate ROAS "
                f"of {overall_roas}. See the Channel Importance panel for which channel is currently "
                f"acting as the primary revenue bedrock."
            )

    def ask_question(self, question: str, analytics_context: Dict[str, Any]) -> str:
        # BUG fix (bug-hunt sweep): this used to completely ignore analytics_context and
        # return fixed canned numbers ($340,000, "3.8x to 3.55x", "over 50%"...) regardless
        # of the real data - directly contradicting this class's own docstring claim of
        # "no hardcoded numbers". This IS reachable in production: GeminiProvider.ask_question()
        # falls back to MockLLMProvider().ask_question() whenever the live Gemini call fails
        # (rate limit/outage) even though a GEMINI_API_KEY is configured - observed live
        # during this session's testing (429s on a real, currently-rate-limited key). Now
        # builds real prose from the context dict chat_engine.py actually passes (channel
        # breakdown is a plain dict, so this works for any number of channels, not just 3).
        ctx = analytics_context or {}
        q = question.lower()
        total_rev = float(ctx.get("total_historical_revenue", 0.0))
        total_spend = float(ctx.get("total_historical_spend", 0.0))
        overall_roas = float(ctx.get("overall_historical_roas", 0.0))
        fc_rev_90 = float(ctx.get("forecast_90d_p50_revenue", 0.0))
        fc_roas_90 = float(ctx.get("forecast_90d_p50_roas", overall_roas))
        roas_trend = float(ctx.get("roas_trend_30d_pct", 0.0))
        rev_trend = float(ctx.get("revenue_trend_30d_pct", 0.0))
        channels: Dict[str, Any] = ctx.get("channel_breakdown", {}) or {}

        def _top_channel(key: str):
            if not channels:
                return None, 0.0
            best = max(channels.items(), key=lambda kv: kv[1].get(key, 0.0))
            return best[0], float(best[1].get(key, 0.0))

        top_rev_ch, top_rev_val = _top_channel("revenue")
        top_roas_ch, top_roas_val = _top_channel("roas")

        def _channel_line() -> str:
            if not channels:
                return ""
            parts = [f"{ch} {row.get('roas', 0.0):.2f}x ROAS ({row.get('share_pct', 0.0):.1f}% of revenue)"
                     for ch, row in channels.items()]
            return " By channel: " + ", ".join(parts) + "."

        if "decrease" in q or "decreasing" in q or "drop" in q or "down" in q:
            direction = "declined" if rev_trend < 0 else "moderated"
            return (
                f"[Mock Insight Mode Enabled] Causal Analysis: revenue {direction} {abs(rev_trend):.1f}% "
                f"over the most recent 30-day window, with blended ROAS moving {roas_trend:+.1f}%."
                + (f" {top_roas_ch} remains the most efficient channel at {top_roas_val:.2f}x ROAS, "
                   f"suggesting the compression is concentrated elsewhere in the portfolio."
                   if top_roas_ch else "")
            )
        elif "channel" in q or "growth" in q or "best" in q:
            if top_rev_ch:
                return (
                    f"[Mock Insight Mode Enabled] Channel Intelligence: {top_rev_ch} is your dominant "
                    f"revenue driver at ${top_rev_val:,.0f} ({channels[top_rev_ch].get('share_pct', 0.0):.1f}% "
                    f"of total). {top_roas_ch} delivers the strongest efficiency at {top_roas_val:.2f}x ROAS."
                    + _channel_line()
                )
            return (
                f"[Mock Insight Mode Enabled] Channel Intelligence: total portfolio revenue is "
                f"${total_rev:,.0f} at {overall_roas:.2f}x blended ROAS."
            )
        elif "meta" in q or "increase" in q or "scale" in q or "budget" in q or "allocat" in q:
            return (
                f"[Mock Insight Mode Enabled] Budget Simulation: current blended ROAS is "
                f"{overall_roas:.2f}x on ${total_spend:,.0f} in spend, generating ${total_rev:,.0f} "
                f"in revenue."
                + (f" {top_roas_ch} shows the strongest efficiency at {top_roas_val:.2f}x ROAS and is "
                   f"the best candidate for incremental budget." if top_roas_ch else "")
                + " Run the Optuna budget optimizer in the navigation tab for an exact "
                  "revenue-maximizing split."
            )
        else:
            return (
                f"[Mock Insight Mode Enabled] Assistant Analyst: total portfolio spend is "
                f"${total_spend:,.0f} generating ${total_rev:,.0f} in revenue at {overall_roas:.2f}x "
                f"blended ROAS. The 90-day P50 forecast is ${fc_rev_90:,.0f} at {fc_roas_90:.2f}x ROAS."
                + _channel_line()
            )

    def get_provider_name(self) -> str:
        return "MockLLMProvider (offline fallback)"


class GeminiProvider(BaseLLMProvider):
    """Production provider connecting to Google Gemini API."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"

    def generate_insight(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        full_prompt = (
            "System: You are a marketing analytics assistant for ForecastIQ, a revenue "
            "forecasting and budget optimization platform. Write in a clear, professional, "
            "consultative tone \u2014 no hype, no exclamation points, no superlatives like "
            "'elite' or 'world-class'. Base every claim strictly on the analytics data "
            "provided below; do not invent numbers or channels that aren't present in it. "
            "Keep the response concise (3-5 sentences unless the question calls for more) "
            "and use plain prose, not markdown headers or bullet lists.\n"
            f"Analytics: {context}\nPrompt: {prompt}"
        )
        data = {
            "contents": [{"parts": [{"text": full_prompt}]}]
        }
        max_attempts = 3
        backoff = 1
        for attempt in range(max_attempts):
            try:
                res = requests.post(self.endpoint, json=data, timeout=10)
                if res.status_code == 200:
                    raw = res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    return _strip_markdown(raw)
                else:
                    logger.warning(f"Gemini API error {res.status_code}: {res.text} (attempt {attempt+1}/{max_attempts})")
            except Exception as e:
                logger.warning(f"Gemini connection exception: {str(e)} (attempt {attempt+1}/{max_attempts})")
            if attempt < max_attempts - 1:
                time.sleep(backoff * (2 ** attempt))
        logger.error("Gemini API failed after %d attempts. Falling back to Mock.", max_attempts)
        return MockLLMProvider().generate_insight(prompt, context)

    def ask_question(self, question: str, analytics_context: Dict[str, Any]) -> str:
        full_prompt = (
            "System: You are a marketing analytics assistant embedded in ForecastIQ, "
            "answering a question from the person viewing their own revenue/ROAS dashboard. "
            "Write in a clear, professional, consultative tone \u2014 no hype, no exclamation "
            "points, no superlatives like 'elite' or 'world-class'. Base every claim strictly "
            "on the analytics context provided below; do not invent numbers, channels, or "
            "campaigns that aren't present in it, and say so plainly if the data doesn't "
            "support a confident answer. Keep the response concise (3-5 sentences unless the "
            "question calls for more) and use plain prose, not markdown headers or bullet lists.\n"
            f"Analytics: {analytics_context}\nQuestion: {question}"
        )
        data = {
            "contents": [{"parts": [{"text": full_prompt}]}]
        }
        max_attempts = 3
        backoff = 1
        for attempt in range(max_attempts):
            try:
                res = requests.post(self.endpoint, json=data, timeout=10)
                if res.status_code == 200:
                    raw = res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    return _strip_markdown(raw)
                else:
                    logger.warning(f"Gemini Chat error {res.status_code} (attempt {attempt+1}/{max_attempts})")
            except Exception as e:
                logger.warning(f"Gemini Chat exception: {str(e)} (attempt {attempt+1}/{max_attempts})")
            if attempt < max_attempts - 1:
                time.sleep(backoff * (2 ** attempt))
        logger.error("Gemini Chat API failed after %d attempts. Falling back to Mock.", max_attempts)
        return MockLLMProvider().ask_question(question, analytics_context)

    def get_provider_name(self) -> str:
        return "Google Gemini (2.5 Flash)"


def get_llm_provider() -> BaseLLMProvider:
    """Factory function that auto-detects API keys from .env or returns MockLLMProvider."""
    if os.getenv("GEMINI_API_KEY"):
        logger.info("Auto-detected GEMINI_API_KEY in environment.")
        return GeminiProvider(os.getenv("GEMINI_API_KEY"))
    else:
        logger.info("No paid API keys detected. Falling back to MockLLMProvider (offline mode).")
        return MockLLMProvider()
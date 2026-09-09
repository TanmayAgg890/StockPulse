"""
ai_advisor.py — Grounded AI Market Advisor (RAG-Lite) for StockPulse.

Responsibilities:
- Build a compact, strictly factual context block from computed metrics and ML evaluations.
- Interface with the Groq API (Qwen-3.8 / Llama / GPT-OSS) using grounded system instructions.
- Provide defensive fallback synthesis when an API key is absent or network fails.
- Provide follow-up Q&A grounded exclusively in the active ticker's context.
"""

import os
from typing import Dict, List, Any, Optional
import streamlit as st

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Preferred models in order of availability on Groq
GROQ_MODEL_CANDIDATES = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
    "groq/compound-mini",
]


def build_context(
    ticker: str,
    stats: Dict[str, Any],
    predictions: Dict[str, float],
    eval_metrics: Dict[str, Dict[str, float]],
) -> str:
    """
    Build a compact, strictly factual context block for the AI advisor.
    The LLM does NOT calculate any metrics; it only synthesizes this provided data.
    """
    currency = stats.get("currency", "$")
    current_p = stats.get("current_price", 0.0)
    p_change_pct = stats.get("period_change_pct", 0.0)
    p_high = stats.get("period_high", 0.0)
    p_low = stats.get("period_low", 0.0)
    ma7 = stats.get("ma7", 0.0)
    ma30 = stats.get("ma30", 0.0)
    trend_rel = stats.get("trend_status", "N/A")

    lr_pred = predictions.get("Linear Regression", current_p)
    rf_pred = predictions.get("Random Forest", current_p)
    lr_delta_pct = ((lr_pred - current_p) / current_p) * 100 if current_p else 0.0
    rf_delta_pct = ((rf_pred - current_p) / current_p) * 100 if current_p else 0.0

    lr_rmse = eval_metrics.get("Linear Regression", {}).get("rmse", 0.0)
    lr_r2 = eval_metrics.get("Linear Regression", {}).get("r2", 0.0)
    rf_rmse = eval_metrics.get("Random Forest", {}).get("rmse", 0.0)
    rf_r2 = eval_metrics.get("Random Forest", {}).get("r2", 0.0)

    context = f"""--- FACTUAL RETRIEVED CONTEXT (STOCKPULSE PIPELINE) ---
Ticker Symbol: {ticker}
Current Close Price: {currency}{current_p:,.2f}
Historical Lookback Window: ~6 Months ({stats.get('data_points', 0)} trading days)
6-Month Period Change: {p_change_pct:+.2f}%
6-Month High: {currency}{p_high:,.2f}
6-Month Low: {currency}{p_low:,.2f}
Technical Moving Averages:
  - 7-Day Moving Average (MA7): {currency}{ma7:,.2f}
  - 30-Day Moving Average (MA30): {currency}{ma30:,.2f}
  - Momentum Relationship: {trend_rel}
Machine Learning Next-Day Models (Trained on 3-day Price Lags):
  - Linear Regression Estimate: {currency}{lr_pred:,.2f} (Delta: {lr_delta_pct:+.2f}%)
    * Chronological Test RMSE: {currency}{lr_rmse:,.2f}
    * Test R² Score: {lr_r2:.4f}
  - Random Forest Regressor Estimate: {currency}{rf_pred:,.2f} (Delta: {rf_delta_pct:+.2f}%)
    * Chronological Test RMSE: {currency}{rf_rmse:,.2f}
    * Test R² Score: {rf_r2:.4f}
Limitations: Exogenous news, earnings, macroeconomic indicators, and order-book depth are NOT included.
--------------------------------------------------------"""
    return context.strip()


def get_system_prompt() -> str:
    """Return strict system prompt enforcing factual grounding and safety."""
    return (
        "You are the StockPulse AI Advisor, an educational quantitative market-analysis assistant.\n"
        "Guidelines:\n"
        "1. Use ONLY the structured facts supplied in the context block.\n"
        "2. NEVER invent prices, percentages, news events, earnings estimates, or company developments.\n"
        "3. If requested information is absent from the context, state that it is unavailable.\n"
        "4. Do NOT provide personalized investment advice or recommend buying, selling, or holding.\n"
        "5. Structure your response concisely (approx. 90-120 words) with:\n"
        "   - **Trend Assessment**: Based strictly on MA7 vs MA30 and 6-month period return.\n"
        "   - **ML Projection Interpretation**: Explain the Linear Regression and Random Forest projections cautiously.\n"
        "   - **Key Caveats**: Note that price-lag models do not incorporate news or fundamentals.\n"
        "6. Conclude with: '*Educational analysis only — not financial advice.*'"
    )


def _generate_fallback_insight(context: str) -> str:
    """
    Deterministic rule-based factual synthesis used when no Groq API key is configured
    or when network requests fail. Guarantees zero downtime.
    """
    return (
        "### 🧠 Grounded Market Synthesis (Deterministic Mode)\n\n"
        "**1. Trend Assessment:**\n"
        "The short-term 7-day moving average reflects prevailing price momentum relative to the "
        "30-day baseline. Over the observed 6-month window, price action reflects past trading history "
        "without forward certainty.\n\n"
        "**2. Machine Learning Projections:**\n"
        "The Linear Regression and Random Forest models utilize 3-day autoregressive lags ($t-1, t-2, t-3$). "
        "Ensemble trees help mitigate linear extrapolation extremes, while test RMSE indicates typical "
        "discrepancy on un-shuffled historical data.\n\n"
        "**3. Risk & Engineering Considerations:**\n"
        "Autoregressive models rely solely on past price trajectories. They do not account for corporate earnings, "
        "interest rates, or breaking market news.\n\n"
        "*Educational analysis only — not financial advice.*"
    )


def resolve_api_key() -> Optional[str]:
    """Retrieve Groq API key from Streamlit secrets, environment variables, or session state."""
    # 1. Check Streamlit secrets
    try:
        if "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        pass

    # 2. Check environment variable
    env_key = os.environ.get("GROQ_API_KEY")
    if env_key:
        return env_key.strip()

    # 3. Check session state (user-entered via sidebar)
    if "user_groq_key" in st.session_state and st.session_state["user_groq_key"]:
        return str(st.session_state["user_groq_key"]).strip()

    return None


def get_ai_insight(context: str, custom_api_key: Optional[str] = None) -> str:
    """
    Generate an AI insight from the retrieved context using Groq,
    falling back gracefully if unavailable.
    """
    api_key = custom_api_key if custom_api_key is not None else resolve_api_key()

    if not GROQ_AVAILABLE or not api_key:
        return _generate_fallback_insight(context)

    client = Groq(api_key=api_key)
    last_err = None

    for model_name in GROQ_MODEL_CANDIDATES:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": get_system_prompt()},
                    {
                        "role": "user",
                        "content": f"Analyze the following retrieved market data:\n\n{context}",
                    },
                ],
                temperature=0.2,
                max_tokens=350,
            )
            return response.choices[0].message.content.strip()
        except Exception as err:
            last_err = err
            continue

    return (
        f"> ⚠️ *Note: Groq LLM unavailable ({last_err}). Displaying grounded synthesis fallback:*\n\n"
        + _generate_fallback_insight(context)
    )


def answer_followup(
    context: str,
    chat_history: List[Dict[str, str]],
    question: str,
    custom_api_key: Optional[str] = None,
) -> str:
    """
    Answer a follow-up question strictly grounded within the supplied context.
    """
    api_key = custom_api_key if custom_api_key is not None else resolve_api_key()

    if not GROQ_AVAILABLE or not api_key:
        return (
            "I am currently operating in Zero-Key Grounded Mode. "
            "To enable interactive conversational Q&A with Groq, "
            "please configure a free `GROQ_API_KEY` in the sidebar or `.streamlit/secrets.toml`."
        )

    client = Groq(api_key=api_key)
    messages = [
        {
            "role": "system",
            "content": (
                "You are the StockPulse AI Advisor answering questions about a specific stock.\n"
                "RULES:\n"
                "1. Rely ONLY on the retrieved context below.\n"
                "2. If the user asks something outside this data (e.g. predictions about next year, news, other stocks), "
                "explicitly say: 'I don't have enough information in the current StockPulse context to answer that reliably.'\n"
                "3. Never recommend buying or selling.\n"
                f"\nCURRENT RETRIEVED CONTEXT:\n{context}"
            ),
        }
    ]

    for msg in chat_history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": question})

    for model_name in GROQ_MODEL_CANDIDATES:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=250,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            continue

    return "Unable to process follow-up with current AI model configuration."

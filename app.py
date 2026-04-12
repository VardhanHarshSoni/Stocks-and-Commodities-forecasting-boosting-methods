"""
Stocks & Commodity Price Forecaster — Streamlit dashboard.
Run from project root: streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when launched via Streamlit
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import datetime

from config import (
    ASSETS,
    DEFAULT_FORECAST_DAYS,
    DEFAULT_INTERVAL,
    DEFAULT_PERIOD,
)
from src.data_loader import load_commodity_history
from src.model_ensemble import attach_forecast, train_and_backtest

st.set_page_config(
    page_title="Stocks & Commodity Price Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=Poppins:wght@400;600&display=swap');

    .title-wrapper {
        display: flex;
        justify-content: center;
        padding: 0 0 0.05rem;
        margin-bottom: 0.5rem;
    }
    .main-header {
        width: min(90vw, 900px);
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1;
        margin: 0;
        padding: 0.12rem 0.08rem 0.15rem;
        color: #2c1f14;
        background: linear-gradient(135deg, #fbf7f1 0%, #f4ede5 100%);
        border: 1px solid rgba(108, 88, 68, 0.18);
        border-radius: 2.4rem;
        box-shadow: 0 20px 36px rgba(69, 49, 30, 0.08);
        text-align: center;
        font-family: 'Poppins', sans-serif;
        position: relative;
        overflow: hidden;
    }
    .main-header span {
        display: block;
        font-family: 'Poppins', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        text-transform: none;
        margin-bottom: 0.05rem;
        color: #24180f;
        line-height: 1.02;
    }
    .main-header strong {
        display: block;
        font-family: 'Poppins', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #7d664d;
        line-height: 1.1;
        margin-bottom: 0;
    }
    @keyframes headerLift {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-1.8px); }
        100% { transform: translateY(0px); }
    }
    .subtle { color: #5E412F; font-size: 0.95rem; margin-bottom: 1.25rem; font-family: 'Georgia', serif; }
    
    /* All metric boxes and containers */
    div[data-testid="stMetric"] { 
        background: linear-gradient(135deg, #FFF3D6 0%, #FFE8B8 100%) !important;
        border: 2px solid #8B4513 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        color: #4A2E1A !important;
    }
    
    /* Card styling */
    .reportview-container .metric-container { 
        background: linear-gradient(135deg, #FFF4DD 0%, #FFE6C0 100%);
        border: 3px solid #8B4513;
        border-radius: 12px;
    }
    
    /* All section blocks */
    section[data-testid="stVerticalBlock"] div:has(> [data-testid="stMetric"]) {
        background: transparent !important;
    }
    
    body { background: linear-gradient(135deg, #FFFCFA 0%, #FFF9F4 60%, #FFFCFA 100%); font-family: 'Georgia', serif; margin: 0; }
    .stApp { background: linear-gradient(135deg, #FFFCFA 0%, #FFF9F4 60%, #FFFCFA 100%); font-family: 'Georgia', serif; }
    main > div.block-container { padding: 0.05rem 0.75rem 0.25rem; }
    section[data-testid="stVerticalBlock"] { padding: 0.05rem 0 0.25rem; margin: 0; }
    h1, h2, h3, h4, h5, h6 { color: #5B3C2B; font-family: 'Georgia', serif; }
    p, div, span, label { color: #4B2E2A; font-family: 'Georgia', serif; }
    
    .stSlider .rc-slider-handle,
    .stSlider .rc-slider-handle:focus,
    .stSlider .rc-slider-handle:hover,
    .stSlider .rc-slider-handle:active {
        box-shadow: none !important;
        outline: none !important;
    }
    
    /* Reduce header space but keep action buttons visible */
    header[data-testid="stHeader"] {
        min-height: 2rem !important;
        height: 2rem !important;
    }
    .stApp header {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        margin-bottom: 0 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] { background-color: transparent; border: none; }
    .stTabs [data-baseweb="tab"] { color: #654321; font-family: 'Georgia', serif; background-color: transparent; border: none; padding: 0.5rem 1rem; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { 
        background: linear-gradient(145deg, #FFD1B6 0%, #FFB179 100%);
        color: #654321;
        border: 2px solid #DAA520;
        border-radius: 8px;
    }
    
    .stSidebar { 
        background: linear-gradient(180deg, #FFFDF8 0%, #FFF6E8 100%);
        color: #3E2F1F;
        font-family: 'Poppins', sans-serif;
        border: 1px solid rgba(108, 88, 68, 0.16);
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.7);
    }
    
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar p, .stSidebar div, .stSidebar label, .stSidebar span { 
        color: #3E2F1F;
        font-family: 'Poppins', sans-serif;
        font-size: 0.95rem;
    }
    .stSidebar label {
        font-weight: 600;
        letter-spacing: 0.01em;
    }
    .stSidebar .stTextInput > div > label,
    .stSidebar .stNumberInput > div > label,
    .stSidebar .stSelectbox > div > label,
    .stSidebar .stSlider > label,
    .stSidebar .stCheckbox > label {
        color: #3E2F1F !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 0.95rem !important;
    }
    .stSidebar .css-1adrfps {
        background: transparent !important;
    }
    
    .stButton>button { 
        background: linear-gradient(145deg, #FF6347 0%, #FF7F50 100%);
        color: white;
        border: 2px solid #CD5C5C;
        border-radius: 8px;
        font-family: 'Georgia', serif;
        padding: 0.35rem 0.7rem !important;
        font-size: 0.95rem !important;
        min-height: auto !important;
    }
    
    .stButton>button:hover { 
        background: linear-gradient(145deg, #FF7F50 0%, #FF6347 100%);
    }
    
    .stSelectbox, .stSlider, .stCheckbox { 
        font-family: 'Georgia', serif;
        color: #4B2E2A;
    }
    
    .stSelectbox label, .stSlider label, .stCheckbox label { 
        color: #4B2E2A;
        font-family: 'Georgia', serif;
    }
    
    /* Dropdown styling with contrasting colors */
    .stSelectbox [data-baseweb="select"] {
        background: linear-gradient(135deg, #FF8C69 0%, #FF7F50 100%) !important;
        border: 2px solid #654321 !important;
        border-radius: 8px !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        color: white !important;
    }
    
    .stSelectbox [data-baseweb="select"] svg {
        fill: white !important;
    }
    
    /* Input fields styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: linear-gradient(135deg, #FF8C69 0%, #FF7F50 100%) !important;
        border: 2px solid #654321 !important;
        color: white !important;
    }

    /* Restore slider track while keeping the handle clean */
    .stSlider .rc-slider-rail {
        background: linear-gradient(90deg, #FFB6B6 0%, #FFA07A 100%) !important;
        opacity: 0.45 !important;
    }
    .stSlider .rc-slider-track {
        background: linear-gradient(90deg, #FF6347 0%, #FF7F50 100%) !important;
    }
    .stSlider .rc-slider-handle {
        background: #ffffff !important;
        border: 2px solid #654321 !important;
        box-shadow: none !important;
    }
</style>
""",
    unsafe_allow_html=True,
)





@st.cache_data(ttl=3600, show_spinner=False)
def _load_data(ticker: str, period: str, interval: str, use_cache: bool):
    return load_commodity_history(ticker, period=period, interval=interval, use_cache=use_cache)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_news(ticker: str) -> list[dict[str, str]]:
    try:
        ticker_obj = yf.Ticker(ticker)
        raw_news = ticker_obj.news or []
        headlines = []
        for item in raw_news[:8]:
            content = item.get("content", {})
            title = content.get("title", "")
            provider = content.get("provider", {}).get("displayName", "")
            pub_date = content.get("pubDate", "")
            published = ""
            if pub_date:
                try:
                    # Parse ISO format date
                    dt = datetime.datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    published = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    published = pub_date
            canonical_url = content.get("canonicalUrl", {}).get("url", "")
            headlines.append({"title": title, "provider": provider, "published": published, "link": canonical_url})
        return headlines
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []


def get_currency_symbol(asset_key: str) -> str:
    """Return currency symbol based on asset type.
    
    Args:
        asset_key: The asset key from ASSETS dictionary (e.g., "Indian: ...", "American: ...", "Commodity: ...")
    
    Returns:
        Currency symbol: "₹" for Indian stocks, "$" for others
    """
    if "Indian:" in asset_key:
        return "₹"
    return "$"


def _score_headlines_sentiment(headlines: list[dict[str, str]]) -> tuple[float, str]:
    positive = ["up", "rise", "gain", "beat", "strong", "bull", "positive", "surge", "jump", "record"]
    negative = ["down", "fall", "loss", "weak", "miss", "sell", "bear", "drop", "cut", "decline"]
    score = 0.0
    for item in headlines:
        text = item["title"].lower()
        for word in positive:
            if word in text:
                score += 1
        for word in negative:
            if word in text:
                score -= 1
    if not headlines:
        return 0.0, "Neutral"
    normalized = score / max(len(headlines), 1)
    label = "Neutral"
    if normalized >= 0.75:
        label = "Strong Positive"
    elif normalized >= 0.25:
        label = "Positive"
    elif normalized <= -0.75:
        label = "Strong Negative"
    elif normalized <= -0.25:
        label = "Negative"
    return normalized, label


def _compose_trade_signal(forecast_change_pct: float, sentiment_score: float, forecast_vol: float) -> tuple[str, str]:
    if forecast_change_pct > 1 and sentiment_score > 0.25:
        return "Strong Buy", "Forecast momentum and sentiment are aligned. Consider adding on dips."
    if forecast_change_pct > 0.5 and sentiment_score >= 0:
        return "Buy", "Moderate upside expected with neutral-to-positive news sentiment."
    if forecast_change_pct < -1 and sentiment_score < -0.25:
        return "Strong Sell", "Negative sentiment and forecast direction suggest significant downside risk."
    if forecast_change_pct < -0.5 and sentiment_score <= 0:
        return "Sell", "Forecast points down and headlines are cautious."
    if abs(forecast_change_pct) < 0.4 and abs(sentiment_score) < 0.25:
        return "Hold", "The model and news are mixed. Wait for clearer direction."
    if forecast_change_pct > 0:
        return "Watch / Hold", "Forecast is positive but sentiment is muted. Monitor the next data points."
    return "Watch / Hold", "Forecast is negative but sentiment is not strongly bearish. Avoid committing too soon."


def _fig_price_history(close: pd.Series, name: str, currency_symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=close.index,
            y=close.values,
            mode="lines",
            name="Close",
            line=dict(color="#FF7F50", width=1.8),
        )
    )
    fig.update_layout(
        title=dict(text=f"{name} — historical close", font=dict(size=16, color="#654321", family="Georgia, serif")),
        template="plotly_white",
        plot_bgcolor="#FAF0E6",
        paper_bgcolor="#FFFFF0",
        hovermode="x unified",
        margin=dict(l=40, r=60, t=50, b=40),
        height=380,
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_symbol})",
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="#654321", borderwidth=1),
    )
    return fig


def _fig_backtest(actual: pd.Series, predicted: pd.Series, name: str, currency_symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=actual.index,
            y=actual.values,
            mode="lines",
            name="Actual",
            line=dict(color="#8B4513", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=predicted.index,
            y=predicted.values,
            mode="lines",
            name="Predicted",
            line=dict(color="#FFA07A", width=2),
        )
    )
    fig.update_layout(
        title=dict(text=f"{name} — backtest actual vs predicted", font=dict(size=16, color="#654321", family="Georgia, serif")),
        template="plotly_white",
        plot_bgcolor="#FAF0E6",
        paper_bgcolor="#FFFFF0",
        hovermode="x unified",
        margin=dict(l=40, r=60, t=50, b=40),
        height=380,
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_symbol})",
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="#654321", borderwidth=1),
    )
    return fig


def _fig_forecast(
    history: pd.Series,
    fc_index: pd.DatetimeIndex,
    fc_values: np.ndarray,
    name: str,
    currency_symbol: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history.values,
            mode="lines",
            name="History",
            line=dict(color="#FF7F50", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc_index,
            y=fc_values,
            mode="lines",
            name="Forecast",
            line=dict(color="#8B4513", width=2),
        ),
    )
    fig.add_vline(
        x=history.index[-1],
        line_width=1,
        line_dash="dash",
        line_color="#8B4513",
    )
    fig.update_layout(
        title=dict(text=f"{name} — history + recursive multi-day forecast", font=dict(size=16, color="#654321", family="Georgia, serif")),
        template="plotly_white",
        plot_bgcolor="#FAF0E6",
        paper_bgcolor="#FFFFF0",
        hovermode="x unified",
        margin=dict(l=40, r=60, t=50, b=40),
        height=400,
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_symbol})",
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#3D9024"), tickfont=dict(color="#654321")),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
        legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="left", x=0.02, bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="#654321", borderwidth=1),
    )
    return fig


def _fig_importance(model, feature_names: list[str]) -> go.Figure:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        fig = go.Figure()
        fig.update_layout(title="Feature importance (n/a)", template="plotly_white", height=200)
        return fig
    order = np.argsort(imp)[::-1][:15]
    fig = go.Figure(
        go.Bar(
            x=imp[order],
            y=[feature_names[i] for i in order],
            orientation="h",
            marker_color="#FFA07A",
        )
    )
    fig.update_layout(
        title=dict(text="Top feature importances (ensemble method)", font=dict(size=14, color="#654321", family="Georgia, serif")),
        template="plotly_white",
        plot_bgcolor="#FAF0E6",
        paper_bgcolor="#FFFFF0",
        margin=dict(l=120, r=60, t=45, b=40),
        height=360,
        xaxis_title="Importance",
        yaxis=dict(autorange="reversed"),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
    )
    return fig


def main():
    st.markdown(
        '<div class="title-wrapper"><h1 class="main-header"><span>STOCKS & COMMODITIES</span><strong>PRICE FORECASTING ENGINE</strong></h1></div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Parameters")
        asset_labels = []
        asset_keys = list(ASSETS.keys())
        for k in asset_keys:
            currency_sym = get_currency_symbol(k)
            asset_labels.append(f"{currency_sym} {k}")
        # Default to first asset
        asset_label = st.selectbox("Asset", asset_labels, index=0)
        # Map back to the real asset key
        if asset_label.startswith("₹ "):
            asset = asset_label[2:]
        elif asset_label.startswith("$ "):
            asset = asset_label[2:]
        else:
            asset = asset_label
        ticker = ASSETS[asset]
        asset_name = asset.split(" - ", 1)[1] if " - " in asset else asset
        currency_symbol = get_currency_symbol(asset)
        period = st.selectbox("History window", ["2y", "3y", "4y"], index=0)
        forecast_days = st.slider("Forecast horizon (trading days)", 5, 15, DEFAULT_FORECAST_DAYS)
        refresh = st.checkbox("Refresh data", value=False)
        st.markdown(
            "**Note:** Forecasts are for demonstration only—not investment advice. "
            "Prices are driven by market dynamics, macro events, and risk premiums; "
            "any model will drift out-of-sample."
        )

    try:
        df = _load_data(ticker, period, DEFAULT_INTERVAL, use_cache=not refresh)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    close = df["close"].astype(float)
    last_update = close.index.max().strftime("%Y-%m-%d")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Latest Closing Price", f"{currency_symbol}{close.iloc[-1]:,.2f}")
    with c2:
        st.metric("Samples Observed", f"{len(close):,}")
    with c3:
        st.metric("Last Day Observed", last_update)
    with c4:
        ret_1y = close.pct_change(252).iloc[-1] if len(close) > 252 else float("nan")
        st.metric("Estimated 1yr Return", f"{ret_1y * 100:.1f}%" if np.isfinite(ret_1y) else "—")

    try:
        fit = train_and_backtest(close)
        fc = attach_forecast(fit, horizon_days=forecast_days)
    except Exception as e:
        st.error(f"Model error: {e}")
        st.stop()

    forecast_change = fc.forecast_close[-1] - close.iloc[-1]
    forecast_change_pct = (forecast_change / close.iloc[-1]) * 100
    avg_forecast = fc.forecast_close.mean()
    forecast_vol = np.std(fc.forecast_close)
    forecast_sentiment = "📈 BULLISH" if forecast_change > 0 else "📉 BEARISH"
    returns = close.pct_change() * 100
    sentiment_score = 0.0
    news_headlines = _fetch_news(ticker)
    sentiment_score, sentiment_label = _score_headlines_sentiment(news_headlines)

    # Compute additional metrics for report
    returns_pct = close.pct_change().dropna()
    var_95 = returns_pct.quantile(0.05)
    cvar_95 = returns_pct[returns_pct <= var_95].mean()
    max_dd = ((close.cummax() - close) / close.cummax()).max()
    sharpe = (returns_pct.mean() / returns_pct.std()) * np.sqrt(252) if returns_pct.std() > 0 else 0
    signal_text, signal_note = _compose_trade_signal(forecast_change_pct, sentiment_score, forecast_vol)

    tab_a, tab_b, tab_c, tab_d, tab_e, tab_f, tab_g, tab_h = st.tabs(
        [
            "Overview & forecast",
            "Backtest",
            "Insights",
            "Statistics",
            "News",
            "Trading Signals",
            "Technical Analysis",
            "Risk Metrics",
        ]
    )

    with tab_a:
        col_left, col_center, col_right = st.columns((0.15, 0.7, 0.15))
        with col_center:
            st.plotly_chart(
                _fig_forecast(fc.history_close, fc.forecast_index, fc.forecast_close, asset_name, currency_symbol),
                use_container_width=True,
            )
            st.markdown(
                '<p style="margin-top:0.25rem; margin-bottom:1rem; color:#3F2F1F; font-size:0.95rem;">Historical closing prices with the model&apos;s multi-day forecast shown beyond the latest observed value. Use this chart to compare recent price behavior against the projected trend.</p>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(_fig_importance(fc.model, fc.feature_names), use_container_width=True)
            st.markdown(
                '<p style="margin-top:0.25rem; margin-bottom:1rem; color:#3F2F1F; font-size:0.95rem;">Feature importance ranks the variables driving the forecast. Taller bars indicate features that had stronger influence on model predictions.</p>',
                unsafe_allow_html=True,
            )

    with tab_b:
        bt = fit.backtest
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE (holdout)", f"{bt.mae:,.4f}")
        m2.metric("RMSE (holdout)", f"{bt.rmse:,.4f}")
        m3.metric("MAPE (holdout)", f"{bt.mape:.2f}%")
        st.plotly_chart(_fig_backtest(fit.history_close, bt.y_pred, asset_name, currency_symbol), use_container_width=True)
        st.markdown(
            '<p style="margin-top:0.25rem; margin-bottom:1rem; color:#3F2F1F; font-size:0.95rem;">Backtest comparison of actual prices versus model predictions on the validation set. Smaller gaps and closer alignment show stronger historical fit.</p>',
            unsafe_allow_html=True,
        )

    with tab_c:
        st.subheader("Forecast Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #FFF4FB 0%, #F7E8F5 100%); color: #3A1D41; padding: 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 14px 30px rgba(141, 76, 131, 0.08); display: flex; flex-direction: column; justify-content: space-between; min-height: 220px;">
                    <h4 style="margin-top:0; color:#6B2B70;">1️⃣ Forecast Direction</h4>
                    <div>
                        <p style="margin:0.4rem 0 0.8rem 0; font-size:0.98rem; color:#5F3A72;">Sentiment: <strong>{forecast_sentiment}</strong></p>
                        <p style="margin:0.2rem 0 0.4rem 0; font-size:0.96rem; color:#4F2B5F;">The <strong>{forecast_days}-day forecast</strong> shows a {('bullish' if forecast_change > 0 else 'bearish')} bias with an expected move of <strong>{forecast_change:+.2f}</strong> ({forecast_change_pct:+.2f}%).</p>
                    </div>
                    <div style="margin-top:0.8rem; padding:0.85rem; background: rgba(255,255,255,0.88); border-radius: 12px; color:#3C1F44; font-weight:700;">Current price: {currency_symbol}{close.iloc[-1]:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #FFF6F9 0%, #F9E9F4 100%); color: #3D1D46; padding: 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 14px 30px rgba(161, 85, 143, 0.08); display: flex; flex-direction: column; justify-content: space-between; min-height: 220px;">
                    <h4 style="margin-top:0; color:#701F6B;">3️⃣ Forecast Range & Confidence</h4>
                    <div>
                        <p style="margin:0.4rem 0 0.85rem 0; color:#5B2C63;">Price Range: <strong>{currency_symbol}{fc.forecast_close.min():,.2f}</strong> → <strong>{currency_symbol}{fc.forecast_close.max():,.2f}</strong></p>
                        <p style="margin:0.2rem 0 0.8rem 0; color:#52315A;">Mean Forecast: <strong>{currency_symbol}{avg_forecast:,.2f}</strong> | Volatility: <strong>±{currency_symbol}{forecast_vol:.2f}</strong></p>
                    </div>
                    <div style="margin-top:0.8rem; padding:0.8rem; background: rgba(255,255,255,0.86); border-radius: 12px; color:#45225C; font-weight:700;">Tighter range = higher model confidence.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #FFF7FA 0%, #F6E8F4 100%); color: #3E1C42; padding: 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 14px 30px rgba(158, 86, 141, 0.08); display: flex; flex-direction: column; justify-content: space-between; min-height: 220px;">
                    <h4 style="margin-top:0; color:#6F1F6C;">4️⃣ Model Accuracy</h4>
                    <div>
                        <p style="margin:0.4rem 0 0.75rem 0; color:#5C2E69;">Backtest MAPE: <strong>{bt.mape:.2f}%</strong></p>
                        <p style="margin:0.2rem 0 0.85rem 0; color:#50315A;">This measures recent holdout fit, not future performance.</p>
                    </div>
                    <div style="margin-top:0.8rem; padding:0.85rem; background: rgba(255,255,255,0.86); border-radius: 12px; color:#40224A; font-weight:700;">Use as a signal guide, not a certainty.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with col2:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #FCE7F4 0%, #EFD2EE 100%); color: #3D1D41; padding: 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 14px 30px rgba(163, 94, 141, 0.08); display: flex; flex-direction: column; justify-content: space-between; min-height: 220px;">
                    <h4 style="margin-top:0; color:#772A71;">2️⃣ Short-term Momentum</h4>
                    <div>
                        <p style="margin:0.4rem 0 0.8rem 0; color:#5B2F61;">The model extends recent 1–7 day patterns forward.</p>
                        <p style="margin:0.2rem 0 0.8rem 0; color:#4F2A57;">Best during trending markets; may lag when conditions shift rapidly.</p>
                    </div>
                    <div style="margin-top:0.8rem; padding:0.85rem; background: rgba(255,255,255,0.86); border-radius: 12px; color:#3D1F42; font-weight:700;">Monitor short-term trend strength and momentum.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #FCE8F7 0%, #E9D4F5 100%); color: #3E1E44; padding: 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 14px 30px rgba(170, 96, 150, 0.08); display: flex; flex-direction: column; justify-content: space-between; min-height: 220px;">
                    <h4 style="margin-top:0; color:#752C6A;">5️⃣ Practical Use</h4>
                    <div>
                        <p style="margin:0.4rem 0 0.8rem 0; color:#5D2F64;">Use this forecast as a tactical baseline signal.</p>
                        <ul style="margin:0.4rem 0 0 1rem; padding:0; color:#4E2853;">
                            <li>Combine with fundamental analysis</li>
                            <li>Monitor inventory, demand, and macro risk</li>
                            <li>Always manage position risk</li>
                        </ul>
                    </div>
                    <div style="margin-top:0.8rem; padding:0.85rem; background: rgba(255,255,255,0.86); border-radius: 12px; color:#3F1D43; font-weight:700;">Treat this as a supporting analysis layer.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #F8E0F0 0%, #E9D4F7 100%); color: #3B1B43; padding: 1.2rem; border-radius: 18px; margin-top:0.5rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 34px rgba(175, 93, 150, 0.1);">
                <h4 style="margin-top:0; color:#7C2F70;">🔔 Reliability & Guidance</h4>
                <p style="margin:0.4rem 0 0.7rem 0; color:#5F2F67;">Peak forecast accuracy is strongest in days 1–5. The signal weakens on longer horizons.</p>
                <p style="margin:0.2rem 0 0.8rem 0; color:#4D2751;">This is a model-based guide, not investment advice. Use it only as part of broader research and risk controls.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown("---")

    with tab_d:
        st.subheader("Price Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Price", f"{currency_symbol}{close.mean():,.2f}")
        col2.metric("Median Price", f"{currency_symbol}{close.median():,.2f}")
        col3.metric("Standard Deviation", f"{currency_symbol}{close.std():,.2f}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Min Price", f"{currency_symbol}{close.min():,.2f}")
        col2.metric("Max Price", f"{currency_symbol}{close.max():,.2f}")
        col3.metric("Range", f"{currency_symbol}{close.max() - close.min():,.2f}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Skewness", f"{close.skew():.3f}")
        col2.metric("Kurtosis", f"{close.kurtosis():.3f}")
        col3.metric("Coefficient of Variation (%)", f"{(close.std() / close.mean()) * 100:.2f}%")
        
        st.markdown("---")
        st.subheader("Distribution")
        fig_dist = go.Figure()
        fig_dist.add_trace(
            go.Histogram(
                x=close.values,
                nbinsx=50,
                name="Price",
                marker_color="#FF7F50",
                opacity=0.7,
            )
        )
        fig_dist.update_layout(
            title=dict(text=f"{asset_name} — Price Distribution", font=dict(color="#654321", family="Georgia, serif")),
            xaxis_title="Price (USD)",
            yaxis_title="Frequency",
            template="plotly_white",
            plot_bgcolor="#FAF0E6",
            paper_bgcolor="#FFFFF0",
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
            height=400,
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        st.markdown(
            '<p style="margin-top:0.25rem; margin-bottom:1rem; color:#3F2F1F; font-size:0.95rem;">Histogram of historical closing prices for the selected asset. This highlights where the price has spent most of its time and how concentrated the levels are.</p>',
            unsafe_allow_html=True,
        )

    with tab_e:
        st.subheader("News")
        
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #FFF8F0 0%, #F5E6D3 100%); color: #3B1B43; padding: 1.2rem; border-radius: 18px; margin-bottom:1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 34px rgba(175, 93, 150, 0.1);">
                <h4 style="margin-top:0; color:#7C2F70;">📰 News Sentiment Analysis</h4>
                <p style="margin:0.4rem 0 0.7rem 0; color:#5F2F67;">We analyze recent news headlines to determine the overall market sentiment. This helps identify whether the news coverage is generally positive (bullish), neutral, or negative (bearish) for this asset.</p>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 2rem; font-weight: bold; color:#3C1D3F;">{sentiment_score:+.2f}</span>
                        <span style="font-size: 1rem; color:#5F2F67; margin-left: 0.5rem;">({sentiment_label})</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if news_headlines:
            for headline in news_headlines[:6]:
                st.markdown(
                    f"<div style='background: linear-gradient(135deg, #FFF8F0 0%, #F8E3D8 100%); border:3px solid #D99A6B; padding:0.72rem 0.9rem; border-radius:14px; margin-bottom:0.75rem; font-family: Poppins, sans-serif;'>"
                    f"<a href='{headline['link']}' target='_blank' style='text-decoration: none; color:#4F1D3A; font-size:1.18rem; font-weight:700; line-height:1.2;'><strong>{headline['title']}</strong></a><br>"
                    f"<span style='color:#7A4B2E; font-size:0.95rem; letter-spacing:0.01em;'>{headline['provider']} — {headline['published']}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.warning("No recent headlines found for this asset. Try a different ticker or wait for new market updates.")

    with tab_f:
        st.subheader("Trading Signals")
        signal_text, signal_note = _compose_trade_signal(forecast_change_pct, sentiment_score, forecast_vol)

        st.metric("Recommended Action", signal_text)
        st.markdown(
            f"<div style='background:#E8F6EF; border:1px solid #A6D4B8; padding:1rem; border-radius:16px; margin-bottom:1rem;'>"
            f"<p style='margin:0; color:#1F4D3D; font-size:1.1rem; font-weight:700;'><strong>Signal:</strong> {signal_text}</p>"
            f"<p style='margin:0.75rem 0 0 0; color:#2D5C48; font-size:1.05rem; font-weight:700;'>Why we prefer this action:</p>"
            f"<p style='margin:0.35rem 0 0 0; color:#2F3E46; font-size:0.98rem; line-height:1.55;'>"
            f"Current price is <strong>{currency_symbol}{close.iloc[-1]:,.2f}</strong>, and the model forecasts a <strong>{forecast_change_pct:+.2f}%</strong> move over the next <strong>{forecast_days}</strong> trading days." 
            f"News sentiment is <strong>{sentiment_label.lower()}</strong> ({sentiment_score:+.2f}), so this recommendation is built from the stock's current momentum and the latest headline bias." 
            "</p>"
            f"<p style='margin:0.8rem 0 0 0; color:#2F3E46; font-size:0.95rem; line-height:1.5;'>"
            f"Compared with alternative actions, this option is preferred because the stock's forecast move and sentiment signal are aligned and the implied volatility range is manageable at ±{currency_symbol}{forecast_vol:.2f}."
            f"</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
        
        st.info(
            "This signal is an informational overlay, not financial advice. Use it together with technical context and portfolio risk rules."
        )

    with tab_g:
        st.subheader("Technical Analysis")
        
        # Calculate technical indicators
        returns = close.pct_change() * 100
        sma_20 = close.rolling(window=20).mean()
        sma_50 = close.rolling(window=50).mean()
        rolling_vol = returns.rolling(window=20).std()
        avg_return = returns.mean()
        momentum_value = f"{((close.iloc[-1] / close.iloc[-5]) - 1) * 100:+.2f}%" if len(close) > 5 else "N/A"
        
        st.markdown("Technical Overview")
        row1, row2 = st.columns(2)
        
        with row1:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #F9E6F8 0%, #F1D9F4 100%); color: #3A1D41; padding: 1rem 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 32px rgba(148, 86, 157, 0.08);">
                    <h4 style="margin-top:0; color:#6A236D;">📊 Daily Volatility (20d)</h4>
                    <p style="margin:0.35rem 0 0.9rem 0; color:#5D2F64;">Standard deviation of returns over 20 days. Higher values indicate more uncertainty.</p>
                    <div style="padding:0.85rem 1rem; background: rgba(255,255,255,0.82); border-radius: 12px; color:#3C1D3F; font-size:1.5rem; font-weight:700;">{rolling_vol.iloc[-1]:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #FCE9F6 0%, #F4DAF1 100%); color: #3C1B42; padding: 1rem 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 32px rgba(170, 96, 155, 0.08);">
                    <h4 style="margin-top:0; color:#7A2F72;">📈 SMA 20</h4>
                    <p style="margin:0.35rem 0 0.9rem 0; color:#5E2F65;">Smoothed average price over 20 days. Crossovers with longer MAs indicate trend shifts.</p>
                    <div style="padding:0.85rem 1rem; background: rgba(255,255,255,0.82); border-radius: 12px; color:#3F1F43; font-size:1.5rem; font-weight:700;">{currency_symbol}{sma_20.iloc[-1]:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #F6DDF5 0%, #E9CCF2 100%); color: #3D1C40; padding: 1rem 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 32px rgba(164, 88, 146, 0.08);">
                    <h4 style="margin-top:0; color:#6B226A;">💹 Avg Daily Return</h4>
                    <p style="margin:0.35rem 0 0.9rem 0; color:#5C2E64;">Mean daily percent change. Reflects the average short-term momentum in the asset.</p>
                    <div style="padding:0.85rem 1rem; background: rgba(255,255,255,0.82); border-radius: 12px; color:#3E1C41; font-size:1.5rem; font-weight:700;">{avg_return:+.3f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with row2:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #F8E3F4 0%, #EAD5F3 100%); color: #3C1D43; padding: 1rem 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 32px rgba(162, 86, 143, 0.08);">
                    <h4 style="margin-top:0; color:#702F71;">📉 SMA 50</h4>
                    <p style="margin:0.35rem 0 0.9rem 0; color:#5B2E62;">Longer-term moving average. Prices above it often signal the persistence of an uptrend.</p>
                    <div style="padding:0.85rem 1rem; background: rgba(255,255,255,0.82); border-radius: 12px; color:#3F1F44; font-size:1.5rem; font-weight:700;">{currency_symbol}{sma_50.iloc[-1]:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #F9E7F3 0%, #EBD3F2 100%); color: #3D1C44; padding: 1rem 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 32px rgba(159, 88, 143, 0.08);">
                    <h4 style="margin-top:0; color:#722F6F;">📈 Annualized Return</h4>
                    <p style="margin:0.35rem 0 0.9rem 0; color:#5C2F65;">Projected yearly return if the average daily drift continues. Use this as a directional reference.</p>
                    <div style="padding:0.85rem 1rem; background: rgba(255,255,255,0.82); border-radius: 12px; color:#3D1D42; font-size:1.5rem; font-weight:700;">{avg_return * 252:+.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #F7DFF2 0%, #E7D2F3 100%); color: #3D1B42; padding: 1rem 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 32px rgba(161, 88, 146, 0.08);">
                    <h4 style="margin-top:0; color:#762E71;">🚀 Price Momentum (5d)</h4>
                    <p style="margin:0.35rem 0 0.9rem 0; color:#5C2E64;">Recent 5-day price change. Positive values indicate upward short-term pressure.</p>
                    <div style="padding:0.85rem 1rem; background: rgba(255,255,255,0.82); border-radius: 12px; color:#3F1D43; font-size:1.5rem; font-weight:700;">{momentum_value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        st.markdown("---")
        st.subheader("Moving Averages & Volatility")
        fig_ma = go.Figure()
        fig_ma.add_trace(
            go.Scatter(
                x=close.index,
                y=close.values,
                mode="lines",
                name="Close",
                line=dict(color="#FF7F50", width=1.5),
            )
        )
        fig_ma.add_trace(
            go.Scatter(
                x=sma_20.index,
                y=sma_20.values,
                mode="lines",
                name="SMA 20",
                line=dict(color="#8B4513", width=2, dash="dash"),
            )
        )
        fig_ma.add_trace(
            go.Scatter(
                x=sma_50.index,
                y=sma_50.values,
                mode="lines",
                name="SMA 50",
                line=dict(color="#FFA07A", width=2, dash="dash"),
            )
        )
        fig_ma.update_layout(
            title=dict(text=f"{asset_name} — Price with Moving Averages", font=dict(color="#654321", family="Georgia, serif")),
            template="plotly_white",
            plot_bgcolor="#FAF0E6",
            paper_bgcolor="#FFFFF0",
            hovermode="x unified",
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
            height=400,
            xaxis_title="Date",
            yaxis_title=f"Price ({currency_symbol})",
        )
        st.plotly_chart(fig_ma, use_container_width=True)
        st.markdown(
            '<p style="margin-top:0.25rem; margin-bottom:1rem; color:#3F2F1F; font-size:0.95rem;">Price chart with 20-day and 50-day moving averages. Watch for crossovers as potential trend signals and to see how the price interacts with shorter- and longer-term averages.</p>',
            unsafe_allow_html=True,
        )

    with tab_h:
        st.subheader("Risk Metrics")
        
        # Calculate risk metrics
        returns = close.pct_change().dropna()
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        max_dd = ((close.cummax() - close) / close.cummax()).max()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        row1, row2 = st.columns(2)
        
        with row1:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #F9E6F7 0%, #F2D9F2 100%); color: #3B1E41; padding: 1rem 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 30px rgba(148, 86, 157, 0.08);">
                    <h4 style="margin-top:0; color:#6F2670;">⚠️ Value at Risk (95%)</h4>
                    <p style="margin:0.35rem 0 0.85rem 0; color:#5A2D60;">Worst expected daily loss 95% of the time.</p>
                    <div style="padding:0.85rem 1rem; background: rgba(255,255,255,0.84); border-radius: 12px; color:#3D1C41; font-size:1.4rem; font-weight:700;">{var_95:.3f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #FCE8F7 0%, #E9D4F5 100%); color: #3E1E44; padding: 1rem 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 30px rgba(170, 96, 150, 0.08);">
                    <h4 style="margin-top:0; color:#742E72;">🔻 Max Drawdown</h4>
                    <p style="margin:0.35rem 0 0.9rem 0; color:#5A2D62;">Largest peak-to-trough decline in the series.</p>
                    <div style="padding:0.85rem 1rem; background: rgba(255,255,255,0.84); border-radius: 12px; color:#3F1D44; font-size:1.4rem; font-weight:700;">{max_dd * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with row2:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #FCE9F6 0%, #F4DAF1 100%); color: #3C1B42; padding: 1rem 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 30px rgba(170, 96, 155, 0.08);">
                    <h4 style="margin-top:0; color:#7A2F72;">📉 Conditional VaR</h4>
                    <p style="margin:0.35rem 0 0.9rem 0; color:#5E2F65;">Average loss on the worst 5% of days.</p>
                    <div style="padding:0.85rem 1rem; background: rgba(255,255,255,0.84); border-radius: 12px; color:#3F1F43; font-size:1.4rem; font-weight:700;">{cvar_95:.3f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #F7E2F3 0%, #E7D2F3 100%); color: #3D1B42; padding: 1rem 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 30px rgba(161, 88, 146, 0.08);">
                    <h4 style="margin-top:0; color:#752F6F;">⭐ Sharpe Ratio</h4>
                    <p style="margin:0.35rem 0 0.9rem 0; color:#5C2E64;">Risk-adjusted return relative to volatility.</p>
                    <div style="padding:0.85rem 1rem; background: rgba(255,255,255,0.84); border-radius: 12px; color:#3E1D43; font-size:1.4rem; font-weight:700;">{sharpe:.3f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #F8E0F0 0%, #E9D4F7 100%); color: #3B1B43; padding: 1rem 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 30px rgba(175, 93, 150, 0.1);">
                <h4 style="margin-top:0; color:#7C2F70;">📈 Volatility (Annual)</h4>
                <p style="margin:0.35rem 0 0.9rem 0; color:#5F2F67;">Annualized standard deviation of returns.</p>
                <div style="padding:0.85rem 1rem; background: rgba(255,255,255,0.84); border-radius: 12px; color:#3C1E42; font-size:1.4rem; font-weight:700;">{returns.std() * np.sqrt(252):.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #F8EAF3 0%, #E7DFF5 100%); color: #3B1C43; padding: 1rem 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 30px rgba(170, 92, 150, 0.1);">
                <h4 style="margin-top:0; color:#742F72;">📊 Days Up (%)</h4>
                <p style="margin:0.35rem 0 0.9rem 0; color:#5F2F67;">Percentage of trading days with positive returns.</p>
                <div style="padding:0.85rem 1rem; background: rgba(255,255,255,0.84); border-radius: 12px; color:#3C1D43; font-size:1.4rem; font-weight:700;">{(returns > 0).sum() / len(returns) * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown("---")
        st.subheader("Drawdown Analysis")
        fig_dd = go.Figure()
        cummax = close.cummax()
        drawdown = (close - cummax) / cummax * 100
        
        fig_dd.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                fill="tozeroy",
                name="Drawdown",
                line=dict(color="#8B4513"),
                fillcolor="rgba(139, 69, 19, 0.3)",
            )
        )
        fig_dd.update_layout(
            title=dict(text=f"{asset_name} — Drawdown from Peak", font=dict(color="#654321", family="Georgia, serif")),
            template="plotly_white",
            plot_bgcolor="#FAF0E6",
            paper_bgcolor="#FFFFF0",
            height=400,
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
        )
        st.plotly_chart(fig_dd, use_container_width=True)
        st.markdown(
            '<p style="margin-top:0.25rem; margin-bottom:1rem; color:#3F2F1F; font-size:0.95rem;">Drawdown chart showing how far the asset has fallen from its highest historical close. Useful for assessing downside risk and the depth of past sell-offs.</p>',
            unsafe_allow_html=True,
        )
        
        st.markdown("---")
        st.subheader("Returns Distribution")
        fig_ret = go.Figure()
        fig_ret.add_trace(
            go.Histogram(
                x=returns.values * 100,
                nbinsx=50,
                name="Return",
                marker_color="#FFA07A",
                opacity=0.7,
            )
        )
        fig_ret.update_layout(
            title=dict(text=f"{asset_name} — Daily Returns Distribution", font=dict(color="#654321", family="Georgia, serif")),
            xaxis_title="Return (%)",
            yaxis_title="Frequency",
            template="plotly_white",
            plot_bgcolor="#FAF0E6",
            paper_bgcolor="#FFFFF0",
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
            height=400,
        )
        st.plotly_chart(fig_ret, use_container_width=True)
        st.markdown(
            '<p style="margin-top:0.25rem; margin-bottom:1rem; color:#3F2F1F; font-size:0.95rem;">Histogram of daily returns showing the distribution of gains and losses. Use it to understand return volatility, skew, and the frequency of extreme moves.</p>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()

"""
Stocks & Commodity Price Forecaster — Streamlit dashboard.
Run from project root: streamlit run app_new.py
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
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sea Green, Blue, and Black Color Scheme
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
        color: #0F3460;
        background: linear-gradient(135deg, #E8F7F5 0%, #D4EFF0 100%);
        border: 2px solid #17A2B8;
        border-radius: 2.4rem;
        box-shadow: 0 20px 36px rgba(15, 52, 96, 0.1);
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
        color: #0F3460;
        line-height: 1.02;
    }
    .main-header strong {
        display: block;
        font-family: 'Poppins', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #17A2B8;
        line-height: 1.1;
        margin-bottom: 0;
    }
    
    .subtle { color: #138496; font-size: 0.95rem; margin-bottom: 1.25rem; font-family: 'Georgia', serif; }
    
    div[data-testid="stMetric"] { 
        background: linear-gradient(135deg, #D4F1F4 0%, #BFF0F8 100%) !important;
        border: 2px solid #17A2B8 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        color: #0F3460 !important;
    }
    
    body { background: linear-gradient(135deg, #F0F8F7 0%, #E3F2FD 60%, #F0F8F7 100%); font-family: 'Georgia', serif; margin: 0; }
    .stApp { background: linear-gradient(135deg, #F0F8F7 0%, #E3F2FD 60%, #F0F8F7 100%); font-family: 'Georgia', serif; }
    main > div.block-container { padding: 0.05rem 0.75rem 0.25rem; }
    section[data-testid="stVerticalBlock"] { padding: 0.05rem 0 0.25rem; margin: 0; }
    h1, h2, h3, h4, h5, h6 { color: #0F3460; font-family: 'Georgia', serif; font-weight: 700; }
    p, div, span, label { color: #1B5E75; font-family: 'Georgia', serif; }
    
    .stSlider .rc-slider-handle {
        box-shadow: none !important;
        outline: none !important;
    }
    
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
    .stTabs [data-baseweb="tab"] { color: #1B5E75; font-family: 'Georgia', serif; background-color: transparent; border: none; padding: 0.5rem 1rem; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { 
        background: linear-gradient(145deg, #A8E6E1 0%, #7DD3D0 100%);
        color: #0F3460;
        border: 2px solid #17A2B8;
        border-radius: 8px;
    }
    
    .stSidebar { 
        background: linear-gradient(180deg, #F5FCFB 0%, #E8F5F3 100%);
        color: #138496;
        font-family: 'Poppins', sans-serif;
        border: 2px solid #17A2B8;
        box-shadow: inset 0 0 0 1px rgba(23, 162, 184, 0.5);
    }
    
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar p, .stSidebar div, .stSidebar label, .stSidebar span { 
        color: #1B5E75;
        font-family: 'Poppins', sans-serif;
        font-size: 0.95rem;
    }
    .stSidebar label {
        font-weight: 600;
        letter-spacing: 0.01em;
        color: #0F3460 !important;
    }
    
    .stButton>button { 
        background: linear-gradient(145deg, #17A2B8 0%, #138496 100%);
        color: white;
        border: 2px solid #0F3460;
        border-radius: 8px;
        font-family: 'Georgia', serif;
        padding: 0.35rem 0.7rem !important;
        font-size: 0.95rem !important;
        min-height: auto !important;
    }
    
    .stButton>button:hover { 
        background: linear-gradient(145deg, #138496 0%, #0F3460 100%);
    }
    
    .stSelectbox [data-baseweb="select"] {
        background: linear-gradient(135deg, #17A2B8 0%, #138496 100%) !important;
        border: 2px solid #0F3460 !important;
        border-radius: 8px !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        color: white !important;
    }
    
    .stSelectbox [data-baseweb="select"] svg {
        fill: white !important;
    }
    
    .stSlider .rc-slider-rail {
        background: linear-gradient(90deg, #A8E6E1 0%, #7DD3D0 100%) !important;
        opacity: 0.45 !important;
    }
    .stSlider .rc-slider-track {
        background: linear-gradient(90deg, #17A2B8 0%, #138496 100%) !important;
    }
    .stSlider .rc-slider-handle {
        background: white !important;
        border: 2px solid #0F3460 !important;
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


def _get_currency_symbol(asset_key: str) -> str:
    """Return currency symbol based on asset type."""
    if "IND Stocks" in asset_key:
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
            line=dict(color="#17A2B8", width=1.8),
        )
    )
    fig.update_layout(
        title=dict(text=f"{name} — historical close", font=dict(size=16, color="#0F3460", family="Georgia, serif")),
        template="plotly_white",
        plot_bgcolor="#F0F8F7",
        paper_bgcolor="#FFFFFF",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=80, b=40),
        height=380,
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_symbol})",
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(23, 162, 184, 0.2)", title_font=dict(color="#0F3460"), tickfont=dict(color="#138496")),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(23, 162, 184, 0.2)", title_font=dict(color="#0F3460"), tickfont=dict(color="#138496")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(255, 255, 255, 0.95)", bordercolor="#17A2B8", borderwidth=2),
    )
    return fig


def _fig_backtest(y_test: pd.Series, y_pred: pd.Series, name: str, currency_symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_test.index,
            y=y_test.values,
            mode="lines",
            name="Actual",
            line=dict(color="#0F3460", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=y_pred.index,
            y=y_pred.values,
            mode="lines",
            name="Predicted (test)",
            line=dict(color="#20C997", width=1.5),
        )
    )
    fig.update_layout(
        title=dict(text=f"{name} — holdout backtest (one-step ahead)", font=dict(size=16, color="#0F3460", family="Georgia, serif")),
        template="plotly_white",
        plot_bgcolor="#F0F8F7",
        paper_bgcolor="#FFFFFF",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=80, b=40),
        height=380,
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_symbol})",
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(23, 162, 184, 0.2)", title_font=dict(color="#0F3460"), tickfont=dict(color="#138496")),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(23, 162, 184, 0.2)", title_font=dict(color="#0F3460"), tickfont=dict(color="#138496")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(255, 255, 255, 0.95)", bordercolor="#17A2B8", borderwidth=2),
    )
    return fig


def _fig_forecast(history: pd.Series, fc_index: pd.DatetimeIndex, fc_values: np.ndarray, name: str, currency_symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history.values,
            mode="lines",
            name="History",
            line=dict(color="#17A2B8", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc_index,
            y=fc_values,
            mode="lines",
            name="Forecast",
            line=dict(color="#0F3460", width=2),
        ),
    )
    fig.add_vline(
        x=history.index[-1],
        line_width=1,
        line_dash="dash",
        line_color="#138496",
    )
    fig.update_layout(
        title=dict(text=f"{name} — history + recursive multi-day forecast", font=dict(size=16, color="#0F3460", family="Georgia, serif")),
        template="plotly_white",
        plot_bgcolor="#F0F8F7",
        paper_bgcolor="#FFFFFF",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=100, b=40),
        height=400,
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_symbol})",
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(23, 162, 184, 0.2)", title_font=dict(color="#0F3460"), tickfont=dict(color="#138496")),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(23, 162, 184, 0.2)", title_font=dict(color="#0F3460"), tickfont=dict(color="#138496")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(255, 255, 255, 0.95)", bordercolor="#17A2B8", borderwidth=2),
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
            marker_color="#063E2D",
        )
    )
    fig.update_layout(
        title=dict(text="Top feature importances (ensemble model)", font=dict(size=14, color="#0F3460", family="Georgia, serif")),
        template="plotly_white",
        plot_bgcolor="#F0F8F7",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=120, r=20, t=45, b=40),
        height=360,
        xaxis_title="Importance",
        yaxis=dict(autorange="reversed"),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(23, 162, 184, 0.2)", title_font=dict(color="#0F3460"), tickfont=dict(color="#138496")),
    )
    return fig


def main():
    st.markdown(
        '<div class="title-wrapper"><h1 class="main-header"><span>Stocks & Commodity</span><strong>Price Forecaster</strong></h1></div>',
        unsafe_allow_html=True,
    )
    
    with st.sidebar:
        st.header("Parameters")
        # Show rupee symbol for Indian stocks in dropdown
        asset_labels = []
        asset_keys = list(ASSETS.keys())
        for k in asset_keys:
            if "IND Stocks" in k:
                asset_labels.append(f"₹ {k}")
            else:
                asset_labels.append(f"$ {k}")
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
        currency_symbol = _get_currency_symbol(asset)
        period = st.selectbox("History window", ["2y", "5y", "10y"], index=0)
        forecast_days = st.slider("Forecast horizon (trading days)", 5, 15, DEFAULT_FORECAST_DAYS)
        refresh = st.checkbox("Refresh data", value=False)
        st.divider()
        

    try:
        df = _load_data(ticker, period, DEFAULT_INTERVAL, use_cache=not refresh)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    close = df["close"].astype(float)
    last_update = close.index.max().strftime("%Y-%m-%d")
    # Always use rupee symbol for Indian stocks
    display_symbol = "₹" if _get_currency_symbol(asset) == "₹" else _get_currency_symbol(asset)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("💰 Latest Closing Price", f"{display_symbol}{close.iloc[-1]:,.2f}")
    with c2:
        st.metric("📊 Samples Observed", f"{len(close):,}")
    with c3:
        st.metric("📅 Last Day Observed", last_update)
    with c4:
        ret_1y = close.pct_change(252).iloc[-1] if len(close) > 252 else float("nan")
        st.metric("📈 1yr Return", f"{ret_1y * 100:.1f}%" if np.isfinite(ret_1y) else "—")

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
    forecast_sentiment = "BULLISH" if forecast_change > 0 else "BEARISH"
    returns = close.pct_change() * 100

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
                '<p style="margin-top:0.25rem; margin-bottom:1rem; color:#0F3460; font-size:0.95rem; font-weight: 500;">Historical closing prices with the model&apos;s multi-day forecast shown beyond the latest observed value. Use this chart to compare recent price behavior against the projected trend.</p>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(_fig_importance(fc.model, fc.feature_names), use_container_width=True)
            st.markdown(
                '<p style="margin-top:0.25rem; margin-bottom:1rem; color:#0F3460; font-size:0.95rem; font-weight: 500;">Feature importance ranks the variables driving the forecast. Taller bars indicate features that had stronger influence on model predictions.</p>',
                unsafe_allow_html=True,
            )

    with tab_b:
        bt = fit.backtest
        m1, m2, m3 = st.columns(3)
        m1.metric("🎯 MAE (holdout)", f"{display_symbol}{bt.mae:,.4f}")
        m2.metric("📏 RMSE (holdout)", f"{display_symbol}{bt.rmse:,.4f}")
        m3.metric("📊 MAPE (holdout)", f"{bt.mape:.2f}%")
        st.plotly_chart(_fig_backtest(bt.y_test, bt.y_pred, asset_name, display_symbol), use_container_width=True)
        st.markdown(
            '<p style="margin-top:0.25rem; margin-bottom:1rem; color:#7DAACB; font-size:0.95rem; font-weight: 500;">Backtest comparison of actual prices versus model predictions on the validation set. Smaller gaps and closer alignment show stronger historical fit.</p>',
            unsafe_allow_html=True,
        )

    with tab_c:
        st.subheader("Forecast Insights")
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #E6F7F5 0%, #D4F1F4 100%); color: #0F3460; padding: 1.2rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #17A2B8; font-family: 'Georgia', serif; box-shadow: 0 16px 34px rgba(23, 162, 184, 0.1);">
                <h4 style="margin-top:0; color:#138496;">Forecast Direction</h4>
                <p style="margin:0.4rem 0 0.7rem 0; color:#1B5E75;">Sentiment: <strong>{forecast_sentiment}</strong></p>
                <p style="margin:0.2rem 0 0.6rem 0; color:#138496;">The <strong>{forecast_days}-day forecast</strong> shows a {('bullish' if forecast_change > 0 else 'bearish')} bias with an expected move of <strong>{forecast_change:+.2f}</strong> ({forecast_change_pct:+.2f}%).</p>
                <p style="margin:0.2rem 0; color:#138496;">Current price: <strong>{currency_symbol}{close.iloc[-1]:,.2f}</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_d:
        st.subheader("Price Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("📊 Mean Price", f"{display_symbol}{close.mean():,.2f}")
        col2.metric("📈 Median Price", f"{display_symbol}{close.median():,.2f}")
        col3.metric("📉 Std Deviation", f"{display_symbol}{close.std():,.2f}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("⬇️ Min Price", f"{display_symbol}{close.min():,.2f}")
        col2.metric("⬆️ Max Price", f"{display_symbol}{close.max():,.2f}")
        col3.metric("📏 Range", f"{display_symbol}{close.max() - close.min():,.2f}")

    with tab_e:
        st.subheader("News Sentiment")
        news_headlines = _fetch_news(ticker)
        sentiment_score, sentiment_label = _score_headlines_sentiment(news_headlines)
        st.metric("Sentiment Score", f"{sentiment_score:+.2f}", sentiment_label)
        
        if news_headlines:
            for headline in news_headlines[:6]:
                st.markdown(
                    f"<div style='background: linear-gradient(135deg, #E8F7F5 0%, #D4F1F4 100%); border:2px solid #17A2B8; padding:0.72rem 0.9rem; border-radius:14px; margin-bottom:0.75rem;'>"
                    f"<a href='{headline['link']}' target='_blank' style='text-decoration: none; color:#0F3460; font-size:1rem; font-weight:700; line-height:1.2;'><strong>{headline['title']}</strong></a><br>"
                    f"<span style='color:#138496; font-size:0.9rem;'>{headline['provider']} — {headline['published']}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    with tab_f:
        st.subheader("Trading Signals")
        signal_text, signal_note = _compose_trade_signal(forecast_change_pct, sentiment_score, forecast_vol)
        st.metric("Recommended Action", signal_text)
        st.info(signal_note)

    with tab_g:
        st.subheader("Technical Analysis")
        returns = close.pct_change() * 100
        sma_20 = close.rolling(window=20).mean()
        sma_50 = close.rolling(window=50).mean()
        rolling_vol = returns.rolling(window=20).std()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("⚡ Volatility (20d)", f"{rolling_vol.iloc[-1]:.2f}%")
        col2.metric("📊 SMA 20", f"{display_symbol}{sma_20.iloc[-1]:,.2f}")
        col3.metric("📈 SMA 50", f"{display_symbol}{sma_50.iloc[-1]:,.2f}")
        
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=close.index, y=close.values, mode="lines", name="Close", line=dict(color="#7DAACB", width=2)))
        fig_ma.add_trace(go.Scatter(x=sma_20.index, y=sma_20.values, mode="lines", name="SMA 20", line=dict(color="#E8DBB3", width=2.5, dash="dash")))
        fig_ma.add_trace(go.Scatter(x=sma_50.index, y=sma_50.values, mode="lines", name="SMA 50", line=dict(color="#CE2626", width=2.5, dash="dash")))
        
        fig_ma.update_layout(
            title=dict(text=f"{asset_name} — Price with Moving Averages", font=dict(color="#7DAACB", family="Georgia, serif")),
            template="plotly_white", plot_bgcolor="#FFFDEB", paper_bgcolor="#FFFDEB", height=400,
            xaxis_title="Date", yaxis_title=f"Price ({display_symbol})",
            xaxis=dict(showgrid=True, gridcolor="rgba(125, 170, 203, 0.15)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(125, 170, 203, 0.15)"),
        )
        st.plotly_chart(fig_ma, use_container_width=True)

    with tab_h:
        st.subheader("Risk Metrics")
        returns = close.pct_change().dropna()
        var_95 = returns.quantile(0.05)
        max_dd = ((close.cummax() - close) / close.cummax()).max()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Value at Risk (95%)", f"{var_95:.3f}%")
        col2.metric("Max Drawdown", f"{max_dd * 100:.2f}%")
        col3.metric("Sharpe Ratio", f"{sharpe:.3f}")
        
        fig_dd = go.Figure()
        cummax = close.cummax()
        drawdown = (close - cummax) / cummax * 100
        fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, fill="tozeroy", name="Drawdown", 
                                    line=dict(color="#072A55"), fillcolor="rgba(23, 162, 184, 0.2)"))
        fig_dd.update_layout(title=dict(text=f"{asset_name} — Drawdown from Peak", font=dict(color="#000000")),
                            template="plotly_white", plot_bgcolor="#F0F8F7", paper_bgcolor="#FFFFFF", height=400,
                            xaxis_title="Date", yaxis_title="Drawdown (%)",
                            xaxis=dict(showgrid=True, gridcolor="rgba(23, 162, 184, 0.2)"),
                            yaxis=dict(showgrid=True, gridcolor="rgba(23, 162, 184, 0.2)"))
        st.plotly_chart(fig_dd, use_container_width=True)


if __name__ == "__main__":
    main()
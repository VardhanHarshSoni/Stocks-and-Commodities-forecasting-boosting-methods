"""Download and cache daily commodity OHLCV from Yahoo Finance."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import DATA_DIR, DEFAULT_INTERVAL, DEFAULT_PERIOD


def _cache_path(ticker: str, period: str, interval: str) -> Path:
    key = f"{ticker}_{period}_{interval}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / f"{h}_{ticker.replace('=', '_')}.parquet"


def load_commodity_history(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch daily bars for a Yahoo Finance ticker. Results are cached as Parquet
    to avoid repeated API calls during development and demos.
    """
    path = _cache_path(ticker, period, interval)
    if use_cache and path.exists():
        df = pd.read_parquet(path)
        if not df.empty:
            return df

    raw = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
    )
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}. Check symbol or network.")

    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = ("open", "high", "low", "close", "volume")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}; got {list(df.columns)}")
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    df = df.dropna(how="all")

    if use_cache:
        df.to_parquet(path)

    return df


def get_close_series(df: pd.DataFrame) -> pd.Series:
    return df["close"].astype(float)

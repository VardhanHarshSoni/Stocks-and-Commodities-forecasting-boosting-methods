"""Central configuration for tickers, paths, and model defaults."""

from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent

# Cached OHLCV under project root
DATA_DIR = ROOT / "data" / "cache"

# Yahoo Finance symbols for front-month futures (liquid, widely used proxies)
COMMODITIES = {
    "WTI Crude Oil": "CL=F",
    "Natural Gas": "NG=F",
    "Copper": "HG=F",
    "Coffee": "KC=F",
    "Sugar": "SB=F",
}

# Top 20 NASDAQ stocks by market cap
NASDAQ_STOCKS = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "NVIDIA (NVDA)": "NVDA",
    "Tesla (TSLA)": "TSLA",
    "Meta (META)": "META",
    "Alphabet/Google (GOOGL)": "GOOGL",
    "Broadcom (AVGO)": "AVGO",
    "Costco (COST)": "COST",
    "Invesco QQQ (QQQ)": "QQQ",
    "Adobe (ADBE)": "ADBE",
    "Netflix (NFLX)": "NFLX",
    "Paypal (PYPL)": "PYPL",
    "Cisco (CSCO)": "CSCO",
    "Asml (ASML)": "ASML",
    "AMD (AMD)": "AMD",
    "PepsiCo (PEP)": "PEP",
    "Walmart (WMT)": "WMT",
    "Intel (INTC)": "INTC",
    "Airbnb (ABNB)": "ABNB",
}

# All 30 BSE Sensex stocks
INDIAN_STOCKS = {
    "Tata Consultancy Services (TCS)": "TCS.NS",
    "Reliance Industries": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "Wipro": "WIPRO.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Axis Bank": "AXISBANK.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "State Bank of India": "SBIN.NS",
    "NTPC Limited": "NTPC.NS",
    "Coal India": "COALINDIA.NS",
    "Power Grid": "POWERGRID.NS",
    "Larsen & Toubro": "LT.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Titan Company": "TITAN.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Nestlé India": "NESTLEIND.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Sun Pharmaceutical": "SUNPHARMA.NS",
    "Bajaj Finance": "BAJAJFINSV.NS",
    "ITC Limited": "ITC.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Divi's Laboratories": "DIVISLAB.NS",
}

# Combined assets dictionary
ASSETS = {
    **{f"Commodity: {k}": v for k, v in COMMODITIES.items()},
    **{f"American: {k}": v for k, v in NASDAQ_STOCKS.items()},
    **{f"Indian: {k}": v for k, v in INDIAN_STOCKS.items()},
}

DEFAULT_PERIOD = "5y"
DEFAULT_INTERVAL = "1d"
DEFAULT_FORECAST_DAYS = 30

# Feature / model defaults
LAG_DAYS = list(range(1, 8))  # 1..7 day lags of close
ROLL_WINDOWS = (5, 10)
TEST_SIZE_FRAC = 0.2  # hold out last 20% for backtest metrics
RANDOM_STATE = 42

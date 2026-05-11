"""Central configuration for tickers, paths, and model defaults."""

from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent

# Cached OHLCV under project root
DATA_DIR = ROOT / "data" / "cache"

# Yahoo Finance symbols for front-month futures (liquid, widely used proxies)
COMMODITIES = {
    "WTI Crude Oil": "CL=F",
    "Brent Crude Oil": "BZ=F",
    "Natural Gas": "NG=F",
    "Copper": "HG=F",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Platinum": "PL=F",
    "Palladium": "PA=F",
    "Coffee": "KC=F",
    "Sugar": "SB=F",
    "Wheat": "ZWH=F",
    "Corn": "ZCH=F",
    "Soybeans": "ZSH=F",
}

# Top NASDAQ stocks by market cap and liquidity
NASDAQ_STOCKS = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "Tesla (TSLA)": "TSLA",
    "Meta (META)": "META",
    "Nvidia (NVDA)": "NVDA",
    "Alphabet/Google (GOOGL)": "GOOGL",
    "Broadcom (AVGO)": "AVGO",
    "Costco (COST)": "COST",
    "Invesco QQQ (QQQ)": "QQQ",
    "Netflix (NFLX)": "NFLX",
    "AMD (AMD)": "AMD",
    "Qualcomm (QCOM)": "QCOM",
    "PepsiCo (PEP)": "PEP",
    "Walmart (WMT)": "WMT",
    "Airbnb (ABNB)": "ABNB",
    "ASML (ASML)": "ASML",
    "Adobe (ADBE)": "ADBE",
    "Salesforce (CRM)": "CRM",
    "ServiceNow (NOW)": "NOW",
    "Accenture (ACN)": "ACN",
    "Cisco (CSCO)": "CSCO",
    "Intel (INTC)": "INTC",
}

# BSE Sensex and other major Indian stocks
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
    "Titan Company": "TITAN.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Nestlé India": "NESTLEIND.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Sun Pharmaceutical": "SUNPHARMA.NS",
    "Bajaj Finance": "BAJAJFINSV.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Divi's Laboratories": "DIVISLAB.NS",
    "Adani Green Energy": "ADANIGREEN.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "GAIL (India)": "GAIL.NS",
    "Indian Oil": "IOC.NS",
    "BPCL": "BPCL.NS",
    "Life Insurance Corporation": "LIC.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Bajaj Auto": "BAJAJGROUP.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "TVS Motor": "TVSMOTOR.NS",
    "Bosch India": "BOSCHIND.NS",
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

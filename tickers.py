"""
tickers.py — Centralized Stock & Index Ticker Directory for StockPulse.

Categorized lists of actively tracked symbols across Indian markets (NSE),
US markets (NYSE/NASDAQ), and key global benchmarks.
"""

from typing import Dict, List

# Indian Blue-chip & Trending Equities (National Stock Exchange)
INDIA_TRENDING: List[Dict[str, str]] = [
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "flag": "🇮🇳", "exchange": "NSE"},
    {"symbol": "TCS.NS", "name": "Tata Consultancy Services", "flag": "🇮🇳", "exchange": "NSE"},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank Ltd.", "flag": "🇮🇳", "exchange": "NSE"},
    {"symbol": "INFY.NS", "name": "Infosys Ltd.", "flag": "🇮🇳", "exchange": "NSE"},
    {"symbol": "ICICIBANK.NS", "name": "ICICI Bank Ltd.", "flag": "🇮🇳", "exchange": "NSE"},
    {"symbol": "SBIN.NS", "name": "State Bank of India", "flag": "🇮🇳", "exchange": "NSE"},
    {"symbol": "ITC.NS", "name": "ITC Ltd.", "flag": "🇮🇳", "exchange": "NSE"},
    {"symbol": "LT.NS", "name": "Larsen & Toubro", "flag": "🇮🇳", "exchange": "NSE"},
]

# US Popular Tech & Mega-caps
US_POPULAR: List[Dict[str, str]] = [
    {"symbol": "AAPL", "name": "Apple Inc.", "flag": "🇺🇸", "exchange": "NASDAQ"},
    {"symbol": "MSFT", "name": "Microsoft Corp.", "flag": "🇺🇸", "exchange": "NASDAQ"},
    {"symbol": "NVDA", "name": "NVIDIA Corp.", "flag": "🇺🇸", "exchange": "NASDAQ"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "flag": "🇺🇸", "exchange": "NASDAQ"},
    {"symbol": "AMZN", "name": "Amazon.com Inc.", "flag": "🇺🇸", "exchange": "NASDAQ"},
    {"symbol": "TSLA", "name": "Tesla Inc.", "flag": "🇺🇸", "exchange": "NASDAQ"},
    {"symbol": "META", "name": "Meta Platforms", "flag": "🇺🇸", "exchange": "NASDAQ"},
]

# Global & Domestic Market Indices
GLOBAL_INDICES: List[Dict[str, str]] = [
    {"symbol": "^NSEI", "name": "Nifty 50", "flag": "🇮🇳", "exchange": "NSE"},
    {"symbol": "^BSESN", "name": "BSE Sensex", "flag": "🇮🇳", "exchange": "BSE"},
    {"symbol": "^GSPC", "name": "S&P 500", "flag": "🇺🇸", "exchange": "INDEX"},
    {"symbol": "^IXIC", "name": "NASDAQ Composite", "flag": "🇺🇸", "exchange": "INDEX"},
]

DEFAULT_WATCHLIST: List[str] = ["AAPL", "NVDA", "TCS.NS", "RELIANCE.NS"]


def get_ticker_region(symbol: str) -> str:
    """Detect the market region: 'INDIA', 'US', or 'GLOBAL'."""
    clean = symbol.strip().upper()
    if clean.endswith(".NS") or clean.endswith(".BO") or clean in ["^NSEI", "^BSESN"]:
        return "INDIA"
    for item in INDIA_TRENDING:
        if item["symbol"] == clean:
            return "INDIA"
    for item in US_POPULAR:
        if item["symbol"] == clean:
            return "US"
    return "GLOBAL"


def get_ticker_display_name(symbol: str) -> str:
    """Return a human-readable company/index name for a ticker symbol."""
    clean = symbol.strip().upper()
    all_items = INDIA_TRENDING + US_POPULAR + GLOBAL_INDICES
    for item in all_items:
        if item["symbol"] == clean:
            return f"[{item['exchange']}] {item['name']} ({clean})"
    return clean

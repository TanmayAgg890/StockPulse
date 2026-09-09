"""
tickers.py — Centralized Stock & Index Ticker Directory for StockPulse.

Categorized lists of actively tracked symbols across Indian markets (NSE),
US markets (NYSE/NASDAQ), and key global benchmarks.
"""

from typing import Dict, List

# Indian Blue-chip & Trending Equities (National Stock Exchange)
INDIA_TRENDING: List[Dict[str, str]] = [
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "flag": "🇮🇳"},
    {"symbol": "TCS.NS", "name": "Tata Consultancy Services", "flag": "🇮🇳"},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank Ltd.", "flag": "🇮🇳"},
    {"symbol": "INFY.NS", "name": "Infosys Ltd.", "flag": "🇮🇳"},
    {"symbol": "ICICIBANK.NS", "name": "ICICI Bank Ltd.", "flag": "🇮🇳"},
    {"symbol": "SBIN.NS", "name": "State Bank of India", "flag": "🇮🇳"},
    {"symbol": "ITC.NS", "name": "ITC Ltd.", "flag": "🇮🇳"},
    {"symbol": "LT.NS", "name": "Larsen & Toubro", "flag": "🇮🇳"},
]

# US Popular Tech & Mega-caps
US_POPULAR: List[Dict[str, str]] = [
    {"symbol": "AAPL", "name": "Apple Inc.", "flag": "🇺🇸"},
    {"symbol": "MSFT", "name": "Microsoft Corp.", "flag": "🇺🇸"},
    {"symbol": "NVDA", "name": "NVIDIA Corp.", "flag": "🇺🇸"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "flag": "🇺🇸"},
    {"symbol": "AMZN", "name": "Amazon.com Inc.", "flag": "🇺🇸"},
    {"symbol": "TSLA", "name": "Tesla Inc.", "flag": "🇺🇸"},
    {"symbol": "META", "name": "Meta Platforms", "flag": "🇺🇸"},
]

# Global & Domestic Market Indices
GLOBAL_INDICES: List[Dict[str, str]] = [
    {"symbol": "^NSEI", "name": "Nifty 50 (India)", "flag": "🇮🇳"},
    {"symbol": "^BSESN", "name": "BSE Sensex (India)", "flag": "🇮🇳"},
    {"symbol": "^GSPC", "name": "S&P 500 (US)", "flag": "🇺🇸"},
    {"symbol": "^IXIC", "name": "NASDAQ Composite (US)", "flag": "🇺🇸"},
]

# Default starting watchlist for new sessions
DEFAULT_WATCHLIST: List[str] = ["AAPL", "NVDA", "TCS.NS", "RELIANCE.NS"]


def get_ticker_display_name(symbol: str) -> str:
    """Return a human-readable company/index name for a ticker symbol."""
    clean = symbol.strip().upper()
    all_items = INDIA_TRENDING + US_POPULAR + GLOBAL_INDICES
    for item in all_items:
        if item["symbol"] == clean:
            return f"{item['flag']} {item['name']} ({clean})"
    return clean

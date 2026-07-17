"""Verify Yahoo search with only a temporary directory writable."""
from __future__ import annotations

from fundlens.data.yahoo import get_yfinance


def main() -> None:
    yf = get_yfinance()
    cache_dir = yf.cache._CookieDBManager.get_location()
    quotes = yf.Search("GB00B41YBW71", max_results=5, news_count=0).quotes
    symbols = [quote.get("symbol") for quote in quotes]
    assert "0P0000RU81.L" in symbols, symbols
    print("YAHOO_CACHE_SMOKE_OK", cache_dir, "0P0000RU81.L")


if __name__ == "__main__":
    main()

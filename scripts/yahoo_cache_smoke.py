"""Verify FundLens/Yahoo caches with only a temporary directory writable."""
from __future__ import annotations

from fundlens.cache import DiskCache
from fundlens.config import get_settings
from fundlens.data.yahoo import get_yfinance


def main() -> None:
    fundlens_cache = DiskCache(get_settings().cache_dir).cache_dir
    yf = get_yfinance()
    yahoo_cache = yf.cache._CookieDBManager.get_location()
    quotes = yf.Search("GB00B41YBW71", max_results=5, news_count=0).quotes
    symbols = [quote.get("symbol") for quote in quotes]
    assert "0P0000RU81.L" in symbols, symbols
    print(
        "RUNTIME_CACHE_SMOKE_OK",
        fundlens_cache,
        yahoo_cache,
        "0P0000RU81.L",
    )


if __name__ == "__main__":
    main()

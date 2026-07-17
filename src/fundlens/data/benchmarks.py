"""Benchmark proxy resolution and benchmark/style-proxy return fetching.

Benchmark and style-proxy returns are fetched from yfinance (auto-adjusted
close -> decimal period returns) and cached on disk for 1 day.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from fundlens.cache import DiskCache
from fundlens.config import get_settings
from fundlens.data.resolver import FundMeta
from fundlens.data.yahoo import get_yfinance

BENCHMARK_MAP_PATH = str(Path(__file__).with_name("benchmark_map.yaml"))

_CACHE_TTL_DAYS = 1


@dataclass(frozen=True)
class BenchmarkProxyMatch:
    ticker: str
    source: str
    matched_text: str

# Style-proxy ETFs for return-based style analysis (RBSA). LSE-listed iShares
# factor ETFs give GBP-friendly, long-history exposures. Verified empirically
# in scripts/smoke_data.py; swap here if any ticker stops returning data.
_STYLE_PROXIES: dict[str, dict[str, str]] = {
    "developed": {
        "value": "IWVL.L",  # iShares Edge MSCI World Value Factor
        "momentum": "IWMO.L",  # iShares Edge MSCI World Momentum Factor
        "quality": "IWQU.L",  # iShares Edge MSCI World Quality Factor
        "small_cap": "WLDS.L",  # iShares MSCI World Small Cap
        "cash": "ERNS.L",  # iShares GBP Ultrashort Bond (cash proxy)
    },
    "europe": {
        "value": "IEFV.L",  # iShares Edge MSCI Europe Value Factor
        "momentum": "IWMO.L",
        "quality": "IWQU.L",
        "small_cap": "WLDS.L",
        "cash": "ERNS.L",
    },
}
# global is an alias for developed.
_STYLE_PROXIES["global"] = _STYLE_PROXIES["developed"]


def _cache() -> DiskCache:
    return DiskCache(get_settings().cache_dir)


def _yf_close(ticker: str) -> pd.Series:
    yf = get_yfinance()

    data = yf.download(ticker, period="max", auto_adjust=True, progress=False, actions=False)
    if data is None or len(data) == 0:
        return pd.Series(dtype=float)
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


def _returns_from_close(close: pd.Series, freq: str) -> pd.Series:
    if not len(close):
        return pd.Series(dtype=float)
    if freq == "M":
        close = close.resample("ME").last()
    return close.pct_change().dropna()


def _longest_mapping_match(text: str, mappings: dict) -> str | None:
    """Return the most specific ticker mapping contained in ``text``."""
    text_norm = text.casefold()
    matches = [
        (len(str(substring)), str(ticker))
        for substring, ticker in mappings.items()
        if ticker and str(substring).casefold() in text_norm
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


def benchmark_proxy_match_for(fund: FundMeta) -> BenchmarkProxyMatch | None:
    """Resolve a benchmark proxy and retain whether its source was name or category."""
    with open(BENCHMARK_MAP_PATH, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    mappings = config.get("mappings", {}) or {}
    category_mappings = config.get("category_mappings", {}) or {}

    name = (fund.benchmark_name or "").strip()
    if name:
        ticker = _longest_mapping_match(name, mappings)
        if ticker:
            return BenchmarkProxyMatch(ticker, "stated_benchmark", name)

    category = (fund.category or "").strip()
    if category:
        ticker = _longest_mapping_match(category, category_mappings)
        if ticker:
            return BenchmarkProxyMatch(ticker, "category", category)
    return None


def benchmark_proxy_for(fund: FundMeta) -> str | None:
    """Look up a tradeable ETF proxy for a fund benchmark or category.

    Stated benchmark names are matched first, preferring the longest (most
    specific) configured substring. If that produces no match, the Morningstar
    category mapping is tried. Categories without a defensible configured proxy
    remain unmapped rather than receiving an unrelated global-equity default.
    """
    match = benchmark_proxy_match_for(fund)
    return match.ticker if match else None


def get_benchmark_returns(ticker: str, freq: str = "M") -> pd.Series:
    """Fetch decimal period returns for a benchmark proxy ``ticker`` (yfinance)."""
    cache = _cache()
    cache_key = f"benchmark/{ticker}/{freq}"
    cached = cache.get_df(cache_key, _CACHE_TTL_DAYS)
    if cached is not None:
        return cached.iloc[:, 0]

    close = _yf_close(ticker)
    if not len(close):
        raise RuntimeError(f"yfinance returned no data for benchmark ticker {ticker!r}")
    returns = _returns_from_close(close, freq)
    returns.name = ticker

    cache.put_df(cache_key, returns.to_frame())
    return returns


def get_style_proxies(region: str, freq: str = "M") -> pd.DataFrame:
    """Fetch style-proxy return series for ``region`` (for RBSA).

    Returns a DataFrame with one decimal-return column per style proxy that
    returned data. Tickers that yfinance cannot serve are silently dropped.
    """
    proxies = _STYLE_PROXIES.get(region, _STYLE_PROXIES["developed"])
    cache = _cache()

    columns: dict[str, pd.Series] = {}
    for style, ticker in proxies.items():
        cache_key = f"style/{ticker}/{freq}"
        cached = cache.get_df(cache_key, _CACHE_TTL_DAYS)
        if cached is not None and len(cached):
            columns[style] = cached.iloc[:, 0]
            continue
        try:
            close = _yf_close(ticker)
        except Exception:  # noqa: BLE001 - best-effort per proxy
            close = pd.Series(dtype=float)
        if not len(close):
            continue
        returns = _returns_from_close(close, freq)
        returns.name = style
        cache.put_df(cache_key, returns.to_frame())
        columns[style] = returns

    if not columns:
        raise RuntimeError(f"no style proxies returned data for region {region!r}")
    return pd.DataFrame(columns).sort_index()

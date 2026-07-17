"""Fetch fund/ETF holdings through Yahoo Finance and normalise the contract."""
from __future__ import annotations

import pandas as pd

from fundlens.cache import DiskCache
from fundlens.config import get_settings
from fundlens.data.resolver import FundMeta

# Contract columns shared by both functions below.
HOLDINGS_COLUMNS = ["ticker", "isin", "name", "weight", "sector", "country", "market_cap"]

_CACHE_TTL_DAYS = 30

def _cache() -> DiskCache:
    return DiskCache(get_settings().cache_dir)


def _normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise a legacy equity-holdings frame to the contract columns."""
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=HOLDINGS_COLUMNS)

    out = pd.DataFrame(index=range(len(raw)))
    out["ticker"] = raw["ticker"].values if "ticker" in raw.columns else pd.NA
    out["isin"] = raw["isin"].values if "isin" in raw.columns else pd.NA
    out["name"] = raw["securityName"].values if "securityName" in raw.columns else pd.NA
    if "weighting" in raw.columns:
        out["weight"] = pd.to_numeric(raw["weighting"], errors="coerce").values / 100.0
    else:
        out["weight"] = pd.NA
    out["sector"] = raw["sector"].values if "sector" in raw.columns else pd.NA
    out["country"] = raw["country"].values if "country" in raw.columns else pd.NA
    # The legacy holdings shape does not expose company market cap directly.
    out["market_cap"] = pd.NA
    return out[HOLDINGS_COLUMNS]


def _fetch_yahoo_holdings(symbol: str) -> pd.DataFrame:
    """Return Yahoo's disclosed top holdings in FundLens contract form."""
    try:
        import yfinance as yf

        raw = yf.Ticker(symbol).funds_data.top_holdings
    except Exception as exc:  # noqa: BLE001 - surface a clear message
        raise RuntimeError(f"Yahoo holdings failed for {symbol!r}: {exc}") from exc
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=HOLDINGS_COLUMNS)

    out = pd.DataFrame(index=range(len(raw)))
    out["ticker"] = [str(value) for value in raw.index]
    out["isin"] = pd.NA
    out["name"] = raw["Name"].astype(str).values if "Name" in raw.columns else out["ticker"]
    out["weight"] = (
        pd.to_numeric(raw["Holding Percent"], errors="coerce").values
        if "Holding Percent" in raw.columns
        else pd.NA
    )
    out["sector"] = pd.NA
    out["country"] = pd.NA
    out["market_cap"] = pd.NA
    return out[HOLDINGS_COLUMNS]


def get_fund_holdings(fund: FundMeta) -> pd.DataFrame:
    """Fetch and normalise the current equity holdings for ``fund``."""
    cache = _cache()
    cache_key = f"holdings/{fund.isin}"
    cached = cache.get_df(cache_key, _CACHE_TTL_DAYS)
    if cached is not None:
        return cached

    symbol = (fund.raw.get("quote") or {}).get("ticker")
    if not symbol:
        raise LookupError(f"no Yahoo Finance symbol available for {fund.isin!r}")
    holdings = _fetch_yahoo_holdings(str(symbol))
    cache.put_df(cache_key, holdings)
    return holdings


def get_etf_holdings(ticker_or_isin: str) -> pd.DataFrame:
    """Fetch and normalise the current equity holdings for an ETF.

    ``ticker_or_isin`` should normally be the benchmark ticker selected by the
    benchmark map (for example ``SWDA.L``).
    """
    cache = _cache()
    cache_key = f"etf_holdings/{ticker_or_isin}"
    cached = cache.get_df(cache_key, _CACHE_TTL_DAYS)
    if cached is not None:
        return cached

    holdings = _fetch_yahoo_holdings(ticker_or_isin)
    cache.put_df(cache_key, holdings)
    return holdings

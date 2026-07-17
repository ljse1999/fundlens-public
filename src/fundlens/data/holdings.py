"""Fetch fund/ETF holdings via mstarpy.

``Funds.holdings("equity")`` returns a wide DataFrame with ``weighting`` in
percent; we normalise it to the shared contract columns with decimal weights
and cache on disk for 30 days.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from fundlens.cache import DiskCache
from fundlens.config import get_settings
from fundlens.data.resolver import FundMeta, _search_hit, get_mstarpy, get_session

# Contract columns shared by both functions below.
HOLDINGS_COLUMNS = ["ticker", "isin", "name", "weight", "sector", "country", "market_cap"]

_CACHE_TTL_DAYS = 30

_BENCHMARK_MAP_PATH = Path(__file__).with_name("benchmark_map.yaml")

_SEARCH_FIELDS = "name,isin,ticker,investmentType,securityType,exchange,SecId"


def _isin_from_benchmark_map(ticker: str) -> str | None:
    """Look up ``ticker`` in the optional ``isins:`` section of benchmark_map.yaml."""
    try:
        with open(_BENCHMARK_MAP_PATH, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
    except OSError:
        return None
    isins = config.get("isins", {}) or {}
    return isins.get(ticker)


def _score_hit(hit: dict) -> int:
    """Rank a general_search hit by how likely it is the ETF we want."""
    score = 0
    kind = str(hit.get("investmentType") or hit.get("securityType") or "").upper()
    if "ETF" in kind or kind in {"FE", "ET"}:
        score += 2
    exchange = str(hit.get("exchange") or "").upper()
    if "LSE" in exchange or "LONDON" in exchange:
        score += 1
    return score


def _resolve_etf_identifier(ticker_or_isin: str) -> str:
    """Resolve an ETF ticker (or ISIN) to an identifier ``mstarpy.Funds`` accepts.

    Resolution order:
      1. general_search on the literal ``ticker_or_isin`` (as before).
      2. If that yields no results and the ticker has an exchange suffix
         (e.g. "SWDA.L"), search the base ticker (before the last ".") and
         prefer results that look like an ETF on the LSE.
      3. Fall back to a hard-coded ISIN from ``benchmark_map.yaml``'s
         optional ``isins:`` section, searched directly.
    """
    try:
        hit = _search_hit(ticker_or_isin)
        return hit.get("isin") or hit.get("ticker") or ticker_or_isin
    except LookupError:
        pass

    base = ticker_or_isin.rsplit(".", 1)[0] if "." in ticker_or_isin else None
    if base:
        session = get_session()
        resp = session.general_search(
            {"q": base, "fields": _SEARCH_FIELDS, "limit": 10}
        )
        results = [(r.get("value") or {}) for r in (resp or {}).get("results") or []]
        if results:
            best = max(results, key=_score_hit)
            return best.get("isin") or best.get("ticker") or base

    fallback_isin = _isin_from_benchmark_map(ticker_or_isin)
    if fallback_isin:
        return fallback_isin

    raise LookupError(
        f"could not resolve an ETF identifier for {ticker_or_isin!r} via "
        f"general_search (full ticker or base ticker) or the benchmark_map "
        f"isins fallback"
    )


def _cache() -> DiskCache:
    return DiskCache(get_settings().cache_dir)


def _normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw mstarpy equity-holdings frame to the contract columns."""
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
    # Morningstar equity holdings do not expose company market cap directly.
    out["market_cap"] = pd.NA
    return out[HOLDINGS_COLUMNS]


def _fetch_holdings(fund_obj: "mstarpy.Funds") -> pd.DataFrame:
    try:
        raw = fund_obj.holdings("equity")
    except Exception as exc:  # noqa: BLE001 - surface a clear message
        raise RuntimeError(f"holdings('equity') failed: {exc}") from exc
    return _normalise(raw)


def get_fund_holdings(fund: FundMeta) -> pd.DataFrame:
    """Fetch and normalise the current equity holdings for ``fund``."""
    cache = _cache()
    cache_key = f"holdings/{fund.isin}"
    cached = cache.get_df(cache_key, _CACHE_TTL_DAYS)
    if cached is not None:
        return cached

    mstarpy = get_mstarpy()
    fund_obj = mstarpy.Funds(fund.isin, session=get_session())
    holdings = _fetch_holdings(fund_obj)
    cache.put_df(cache_key, holdings)
    return holdings


def get_etf_holdings(ticker_or_isin: str) -> pd.DataFrame:
    """Fetch and normalise the current equity holdings for an ETF.

    Resolves ``ticker_or_isin`` via general_search, then follows the same
    ``Funds.holdings("equity")`` path as :func:`get_fund_holdings`.
    """
    cache = _cache()
    cache_key = f"etf_holdings/{ticker_or_isin}"
    cached = cache.get_df(cache_key, _CACHE_TTL_DAYS)
    if cached is not None:
        return cached

    identifier = _resolve_etf_identifier(ticker_or_isin)
    mstarpy = get_mstarpy()
    fund_obj = mstarpy.Funds(identifier, session=get_session())
    holdings = _fetch_holdings(fund_obj)
    cache.put_df(cache_key, holdings)
    return holdings

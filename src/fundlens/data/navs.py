"""Fetch NAV-derived return series for a resolved fund.

The public deployment uses Yahoo Finance adjusted prices. Mutual-fund ISINs are
resolved to Yahoo symbols first, so this path does not require Morningstar's
browser/WAF session.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fundlens.cache import DiskCache
from fundlens.config import get_settings
from fundlens.data.ft_markets import FTFund, monthly_returns as ft_monthly_returns
from fundlens.data.resolver import FundMeta
from fundlens.data.yahoo import get_yfinance

_MIN_MONTHLY_OBS = 36
# NAV history is appended monthly; cache the full history for a week so a
# resumed/re-run screen skips the per-fund provider fetch.
_CACHE_TTL_DAYS = 7


@dataclass
class ReturnsBundle:
    """Container for a fund's return series at daily and monthly frequency.

    Attributes:
        daily: Decimal daily period returns indexed by a DatetimeIndex, or
            None if daily data is unavailable for this fund.
        monthly: Decimal monthly period returns indexed by a DatetimeIndex
            with each observation dated at month-end.
        currency: The currency the returns are denominated in (should match
            ``fund.currency`` unless explicitly converted).
        provenance: Dict describing data source(s), series type, and coverage.
    """

    daily: pd.Series | None
    monthly: pd.Series
    currency: str
    provenance: dict


def _nav_index_to_returns(nav_list: list[dict], field: str = "totalReturn") -> pd.Series:
    """Convert a list of NAV-index dicts into a month-end decimal return series."""
    if not nav_list:
        return pd.Series(dtype=float)
    frame = pd.DataFrame(nav_list)
    frame = frame.dropna(subset=["date", field])
    idx = pd.to_datetime(frame["date"]) + pd.offsets.MonthEnd(0)
    level = pd.Series(frame[field].astype(float).values, index=idx).sort_index()
    level = level[~level.index.duplicated(keep="last")]
    return level.pct_change().dropna()


def _yf_monthly_returns(ticker: str, start: str | None) -> pd.Series:
    """Month-end decimal returns from yfinance adjusted close for ``ticker``."""
    yf = get_yfinance()

    if start:
        data = yf.download(
            ticker, start=start, auto_adjust=True, progress=False, actions=False
        )
    else:
        data = yf.download(
            ticker, period="max", auto_adjust=True, progress=False, actions=False
        )
    if data is None or len(data) == 0:
        return pd.Series(dtype=float)
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce")
    # Some London mutual-fund feeds publish zero placeholders on non-valuation
    # days. Treat them as missing rather than as a total loss/recovery.
    close = close.where(close > 0).dropna()
    monthly = close.resample("ME").last()
    return monthly.pct_change().dropna()


def get_returns(fund: FundMeta, start: str | None = None) -> ReturnsBundle:
    """Fetch decimal monthly (and, if cheap, daily) period returns for ``fund``.

    Returns a :class:`ReturnsBundle`. The monthly series uses Yahoo Finance's
    auto-adjusted prices. Daily data is left as ``None`` because it is not
    required downstream.

    The full-history monthly series is cached on disk keyed by ISIN for
    ``_CACHE_TTL_DAYS`` days, so resumed or re-run screens skip the expensive
    per-fund provider fetch. The ``start`` clip is applied after the
    cache lookup, so the same cached history serves any date window.
    """
    cache = DiskCache(get_settings().cache_dir)
    cache_key = f"navs/{fund.isin}/monthly"

    use_ft = fund.raw.get("provider") == "ft_markets"
    source = "ft_markets" if use_ft else "yfinance"
    series_type = "fund_price" if use_ft else "adjusted_price"

    # Prefer the cached full-history monthly series when fresh. The Series is
    # stored as a single-column DataFrame ("return") since DiskCache persists
    # via to_parquet, which is DataFrame-only.
    cached = cache.get_df(cache_key, _CACHE_TTL_DAYS)
    if cached is not None and "return" in cached.columns and len(cached) >= _MIN_MONTHLY_OBS:
        monthly = cached["return"].copy()
    else:
        ticker = (fund.raw.get("quote") or {}).get("ticker")
        if use_ft:
            ft_raw = fund.raw.get("ft_fund") or {}
            monthly = ft_monthly_returns(
                FTFund(
                    isin=fund.isin,
                    name=fund.name,
                    currency=fund.currency,
                    internal_symbol=str(ft_raw.get("internal_symbol") or fund.sec_id),
                    inception_date=ft_raw.get("inception_date") or fund.inception_date,
                )
            )
        else:
            if not ticker:
                raise LookupError(f"no Yahoo Finance symbol available for {fund.isin!r}")
            monthly = _yf_monthly_returns(str(ticker), start=None)
        if len(monthly) >= _MIN_MONTHLY_OBS:
            cache.put_df(cache_key, monthly.rename("return").to_frame())

    ticker = (fund.raw.get("quote") or {}).get("ticker")

    if start:
        cutoff = pd.Timestamp(start)
        monthly = monthly[monthly.index >= cutoff]

    monthly.name = "return"

    provenance = {
        "source": source,
        "series_type": series_type,
        "first_date": str(monthly.index.min().date()) if len(monthly) else None,
        "last_date": str(monthly.index.max().date()) if len(monthly) else None,
        "n_obs": int(len(monthly)),
        "isin": fund.isin,
        "ticker": ticker,
        "ft_symbol": (fund.raw.get("ft_fund") or {}).get("internal_symbol"),
    }

    return ReturnsBundle(
        daily=None,
        monthly=monthly,
        currency=fund.currency,
        provenance=provenance,
    )

"""Fetch NAV-derived return series for a resolved fund.

Primary source is mstarpy's ``Funds.nav(..., "monthly")``, which returns a list
of ``{"nav", "totalReturn", "date"}`` dicts. We use the ``totalReturn`` field:
this is Morningstar's total-return-reinvested NAV *index* (distributions
reinvested), so period returns computed from it are total returns, not
price-only returns. Requesting from 1990-01-01 clips to the fund's inception.

If mstarpy yields fewer than 36 monthly observations and the fund has a listed
ticker (e.g. ETFs), we fall back to yfinance adjusted-close returns.
"""
from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass

import pandas as pd

from fundlens.cache import DiskCache
from fundlens.config import get_settings
from fundlens.data.resolver import FundMeta, get_mstarpy, get_session

_MIN_MONTHLY_OBS = 36
_MIN_MONTHLY_OBS_RETRY = 24
_NAV_FETCH_ATTEMPTS = 3
_NAV_FETCH_RETRY_SLEEP_SECS = 3
# NAV history is appended monthly; cache the full history for a week so a
# resumed/re-run screen skips the expensive per-fund Morningstar fetches.
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
    import yfinance as yf

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
    monthly = close.resample("ME").last()
    return monthly.pct_change().dropna()


def get_returns(fund: FundMeta, start: str | None = None) -> ReturnsBundle:
    """Fetch decimal monthly (and, if cheap, daily) period returns for ``fund``.

    Returns a :class:`ReturnsBundle`. The monthly series is a *total return*
    series derived from Morningstar's reinvested NAV index. Daily data is left
    as ``None`` (fetching full-history daily NAV is expensive and not required
    downstream); the field is retained for future use.

    The full-history monthly series is cached on disk keyed by ISIN for
    ``_CACHE_TTL_DAYS`` days, so resumed or re-run screens skip the expensive
    per-fund Morningstar NAV fetch. The ``start`` clip is applied after the
    cache lookup, so the same cached history serves any date window.
    """
    cache = DiskCache(get_settings().cache_dir)
    cache_key = f"navs/{fund.isin}/monthly"

    start_date = dt.date(1990, 1, 1)
    end_date = dt.date.today()

    source = "mstarpy"
    series_type = "total_return"

    # Prefer the cached full-history monthly series when fresh. The Series is
    # stored as a single-column DataFrame ("return") since DiskCache persists
    # via to_parquet, which is DataFrame-only.
    cached = cache.get_df(cache_key, _CACHE_TTL_DAYS)
    if cached is not None and "return" in cached.columns and len(cached) >= _MIN_MONTHLY_OBS_RETRY:
        monthly = cached["return"].copy()
    else:
        monthly = pd.Series(dtype=float)
        for attempt in range(_NAV_FETCH_ATTEMPTS):
            try:
                mstarpy = get_mstarpy()
                fund_obj = mstarpy.Funds(fund.isin, session=get_session())
                nav_list = fund_obj.nav(start_date, end_date, "monthly") or []
            except Exception:  # noqa: BLE001 - treat as empty and retry/fall through
                nav_list = []

            monthly = _nav_index_to_returns(nav_list, "totalReturn")
            if len(monthly) >= _MIN_MONTHLY_OBS_RETRY:
                break
            if attempt < _NAV_FETCH_ATTEMPTS - 1:
                # mstarpy/Morningstar intermittently returns an empty (or truncated)
                # NAV list for a valid ISIN; retry with a fresh Funds object rather
                # than silently accepting a transient empty result.
                time.sleep(_NAV_FETCH_RETRY_SLEEP_SECS)

        # Persist the freshly-fetched full history so resume/re-run is cheap.
        # Only cache mstarpy results with enough observations to be useful; the
        # yfinance fallback below is a per-call decision, not cached here.
        # Store as a single-column DataFrame because DiskCache uses to_parquet.
        if len(monthly) >= _MIN_MONTHLY_OBS_RETRY:
            cache.put_df(cache_key, monthly.rename("return").to_frame())

    ticker = (fund.raw.get("quote") or {}).get("ticker")
    if len(monthly) < _MIN_MONTHLY_OBS and ticker:
        yf_monthly = _yf_monthly_returns(ticker, start=None)
        if len(yf_monthly) > len(monthly):
            monthly = yf_monthly
            source = "yfinance"
            series_type = "price"  # yfinance auto-adjusted (dividends reinvested via adj close)

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
    }

    return ReturnsBundle(
        daily=None,
        monthly=monthly,
        currency=fund.currency,
        provenance=provenance,
    )

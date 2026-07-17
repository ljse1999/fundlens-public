"""FX rates and currency conversion helpers for cross-currency factor/return analysis.

FX rates come from yfinance (``{base}{quote}=X`` daily close, resampled to
month-end). Risk-free rates come from FRED via pandas-datareader. Both are
cached on disk for 7 days.
"""
from __future__ import annotations

import pandas as pd

from fundlens.cache import DiskCache
from fundlens.config import get_settings
from fundlens.data.yahoo import get_yfinance

_CACHE_TTL_DAYS = 7

# Preferred FRED series per currency, with fallbacks. All are annualised
# percentage short rates.
_RISK_FREE_SERIES: dict[str, list[str]] = {
    "GBP": ["IUDSOIA"],  # SONIA
    "EUR": ["ECBESTRVOLWGTTRMDMNRT", "IR3TIB01EZM156N"],  # ESTR, fallback 3M interbank
    "USD": ["DGS1MO", "FEDFUNDS"],  # 1M Treasury, fallback fed funds
}

_FACTOR_COLUMNS = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]


def _cache() -> DiskCache:
    return DiskCache(get_settings().cache_dir)


def _month_end(series: pd.Series) -> pd.Series:
    return series.resample("ME").last()


def get_fx(quote: str, base: str = "USD", freq: str = "M") -> pd.Series:
    """Fetch the FX rate series for ``quote`` per 1 unit of ``base``.

    Uses yfinance symbol ``f"{base}{quote}=X"`` (daily close). When
    ``quote == base`` a flat series of 1.0 over a month-end range is returned.
    """
    quote = quote.upper()
    base = base.upper()

    if quote == base:
        idx = pd.date_range("1990-01-01", pd.Timestamp.today().normalize(), freq="ME")
        return pd.Series(1.0, index=idx, name=f"{base}{quote}")

    cache = _cache()
    cache_key = f"fx/{base}{quote}/{freq}"
    cached = cache.get_df(cache_key, _CACHE_TTL_DAYS)
    if cached is not None:
        return cached.iloc[:, 0]

    yf = get_yfinance()

    symbol = f"{base}{quote}=X"
    data = yf.download(symbol, period="max", auto_adjust=True, progress=False, actions=False)
    if data is None or len(data) == 0:
        raise RuntimeError(f"yfinance returned no FX data for {symbol!r}")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    if freq == "M":
        close = _month_end(close)
    close.name = f"{base}{quote}"

    cache.put_df(cache_key, close.to_frame())
    return close


def get_risk_free(currency: str, freq: str = "M") -> pd.Series:
    """Fetch a monthly decimal risk-free return series appropriate for ``currency``.

    FRED annualised percentage rates are converted to per-month decimals via
    ``(1 + r/100) ** (1/12) - 1``.
    """
    currency = currency.upper()
    series_ids = _RISK_FREE_SERIES.get(currency)
    if not series_ids:
        raise ValueError(f"no risk-free series configured for currency {currency!r}")

    cache = _cache()
    cache_key = f"rf/{currency}/{freq}"
    cached = cache.get_df(cache_key, _CACHE_TTL_DAYS)
    if cached is not None:
        return cached.iloc[:, 0]

    import pandas_datareader.data as web

    last_error: Exception | None = None
    raw: pd.Series | None = None
    for series_id in series_ids:
        try:
            frame = web.DataReader(series_id, "fred", start="1990-01-01")
            raw = frame.iloc[:, 0].dropna()
            if len(raw):
                break
        except Exception as exc:  # noqa: BLE001 - try next fallback
            last_error = exc
    if raw is None or not len(raw):
        raise RuntimeError(
            f"could not fetch a risk-free series for {currency!r} "
            f"(tried {series_ids}): {last_error}"
        )

    monthly_annual_pct = _month_end(raw)
    monthly_decimal = (1.0 + monthly_annual_pct / 100.0) ** (1.0 / 12.0) - 1.0
    monthly_decimal = monthly_decimal.dropna()
    monthly_decimal.name = f"RF_{currency}"

    cache.put_df(cache_key, monthly_decimal.to_frame())
    return monthly_decimal


_LONG_SHORT_COLUMNS = ["SMB", "HML", "RMW", "CMA", "MOM"]


def _translate(
    factors_usd: pd.DataFrame,
    fx_ret: pd.Series,
    rf_local: pd.Series,
) -> pd.DataFrame:
    """Apply leg-wise FX translation to a USD factor set given aligned inputs.

    ``factors_usd`` must already contain ``MKT_RF`` and ``RF`` (USD) plus any of
    the long-short columns; ``fx_ret`` and ``rf_local`` must already be aligned
    to ``factors_usd.index`` (reindexing/alignment is the caller's job — this
    function does pure arithmetic so it is trivially unit-testable).

    Market (``MKT_RF``): the market factor is a long-only total return, so the
    standard total-return translation identity applies to the *total* local
    return, not to the excess return in isolation. We first reconstruct the
    USD total market return ``MKT_local_total_usd = MKT_RF_usd + RF_usd``,
    translate it as a total return ``(1 + MKT_local_total_usd) * (1 + fx_ret)
    - 1``, and then re-express it as an excess return over the *local*
    risk-free rate: ``MKT_RF_out = MKT_local_total - rf_local``.

    Long-short factors (``SMB``, ``HML``, ``RMW``, ``CMA``, ``MOM``): each is a
    zero-net-investment long leg minus short leg. Translating each leg as a
    total return, ``(1 + r_long) * (1 + fx_ret) - (1 + r_short) * (1 + fx_ret)
    = (r_long - r_short) * (1 + fx_ret) = r_usd * (1 + fx_ret)``. This is the
    *exact* currency-translation identity for a long-short portfolio (no
    approximation), and is materially different from — and much smaller in
    magnitude than — the long-only identity ``(1 + r_usd) * (1 + fx_ret) - 1``
    that would otherwise inject the full FX return into every style factor.

    ``RF`` in the output is simply ``rf_local``.
    """
    result = factors_usd.copy()

    if "MKT_RF" in result.columns and "RF" in result.columns:
        mkt_local_total_usd = result["MKT_RF"] + result["RF"]
        mkt_local_total = (1.0 + mkt_local_total_usd) * (1.0 + fx_ret) - 1.0
        result["MKT_RF"] = mkt_local_total - rf_local

    for col in _LONG_SHORT_COLUMNS:
        if col in result.columns:
            result[col] = result[col] * (1.0 + fx_ret)

    result["RF"] = rf_local

    return result


def convert_factor_returns(
    factors_usd: pd.DataFrame,
    currency: str,
    *,
    rf_local: pd.Series | None = None,
) -> pd.DataFrame:
    """Convert USD-denominated factor returns into ``currency``-denominated returns.

    Translation is leg-wise and exact rather than applying a single blanket
    formula to every column:

    - ``MKT_RF`` is a long-only total return. It is re-based to its USD total
      return (adding back ``RF`` in USD), translated as a total return via
      ``(1 + r_total_usd) * (1 + fx_ret) - 1``, and then re-expressed as an
      excess return over the *local* risk-free rate.
    - The long-short factors (``SMB``, ``HML``, ``RMW``, ``CMA``, ``MOM``) are
      zero-net-investment long-minus-short portfolios. For these the exact
      currency-translation identity is ``r_out = r_usd * (1 + fx_ret)`` (the
      cross term only) — NOT ``(1 + r_usd) * (1 + fx_ret) - 1``, which would
      incorrectly inject the full FX return into every style factor.
    - ``RF`` is replaced with the target currency's own risk-free series
      (SONIA for GBP, ESTR/3M interbank fallback for EUR, US 1-month Treasury
      for USD), falling back to the FX-translated US RF where the local
      series is unavailable.

    ``rf_local``, if provided, is used directly as the local risk-free series
    (already aligned or alignable to ``factors_usd.index``) instead of calling
    :func:`get_risk_free` — this exists primarily for testability without
    network access.
    """
    currency = currency.upper()
    result = factors_usd.copy()

    if currency == "USD":
        # No FX translation needed; still refresh RF from the US series so the
        # column is consistent with get_risk_free.
        if rf_local is not None:
            result["RF"] = rf_local.reindex(result.index)
            return result
        try:
            rf = get_risk_free("USD").reindex(result.index)
            if rf.notna().any():
                result["RF"] = rf
        except Exception:  # noqa: BLE001 - keep the existing RF column on failure
            pass
        return result

    fx = get_fx(quote=currency, base="USD", freq="M")
    fx_ret = fx.pct_change().reindex(result.index)

    if rf_local is not None:
        rf = rf_local.reindex(result.index)
    else:
        try:
            rf = get_risk_free(currency).reindex(result.index)
        except Exception:  # noqa: BLE001 - fall back to FX-translated US RF
            rf = None
        if rf is None or not rf.notna().any():
            us_rf = get_risk_free("USD").reindex(result.index)
            rf = (1.0 + us_rf) * (1.0 + fx_ret) - 1.0

    return _translate(result, fx_ret, rf)

"""Return-series performance analytics.

Pure functions on pandas objects (no I/O, no network). All returns are
*decimal* period returns (e.g. 0.01 for +1%). Conventions:

* **Annualisation** uses ``periods_per_year`` (``ppy``); volatilities scale by
  ``sqrt(ppy)`` and mean-based ratios by ``sqrt(ppy)``. CAGR is the geometric
  growth rate: ``(prod(1 + r)) ** (ppy / n) - 1``.
* **Excess returns** for the Sharpe ratio are taken over ``rf`` when supplied,
  otherwise the raw returns are used.
* **Drawdowns** are computed on cumulative-return wealth and are ``<= 0``.
* When a ``benchmark`` is supplied, all series are aligned on their common
  (intersected) index before any statistic is computed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _align(*series: pd.Series | None) -> list[pd.Series]:
    """Intersect the indices of all non-None series and drop rows with NaNs."""
    present = [s for s in series if s is not None]
    frame = pd.concat(present, axis=1, join="inner").dropna()
    return [frame.iloc[:, i] for i in range(frame.shape[1])]


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute the running drawdown series for ``returns``.

    The drawdown at each date is ``wealth / running_peak - 1`` where
    ``wealth`` is the cumulative-return index ``cumprod(1 + returns)``.

    Args:
        returns: Decimal period returns indexed by DatetimeIndex.

    Returns:
        A pd.Series (same index as ``returns``) of decimal drawdowns from the
        running peak cumulative value (values ``<= 0``, e.g. ``-0.12`` for a
        12% drawdown).
    """
    r = returns.dropna()
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return dd


def _cagr(returns: pd.Series, periods_per_year: int) -> float:
    n = len(returns)
    if n == 0:
        return float("nan")
    total_growth = float((1.0 + returns).prod())
    return total_growth ** (periods_per_year / n) - 1.0


def perf_summary(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    rf: pd.Series | None = None,
    periods_per_year: int = 12,
) -> dict:
    """Compute a standard performance summary for a return series.

    Args:
        returns: Decimal period returns indexed by DatetimeIndex.
        benchmark: Optional decimal period benchmark returns. When supplied,
            relative metrics are added and every series is aligned on the
            common intersected index first.
        rf: Optional decimal period risk-free returns. When supplied the
            Sharpe/Sortino ratios use excess returns over ``rf``.
        periods_per_year: Periods per year for annualisation (12 = monthly).

    Returns:
        A dict with keys: ``cagr``, ``vol_ann``, ``sharpe`` (excess over ``rf``
        if given, else raw), ``sortino``, ``max_drawdown``, ``max_dd_start``,
        ``max_dd_end``, ``best_period``, ``worst_period``, ``positive_share``,
        ``n_obs``, ``start``, ``end``. When ``benchmark`` is given, also:
        ``tracking_error_ann``, ``information_ratio``, ``beta``,
        ``up_capture``, ``down_capture``, ``excess_cagr``, ``hit_rate``.

        Ratio conventions: Sharpe = ``mean(excess) / std(excess) * sqrt(ppy)``;
        Sortino uses downside deviation ``sqrt(mean(min(excess, 0) ** 2))``;
        information ratio = ``mean(active) / std(active) * sqrt(ppy)`` where
        ``active = returns - benchmark``; up/down capture are the ratios of
        compounded returns over periods where the benchmark is positive /
        negative; ``hit_rate`` is the share of periods with
        ``returns > benchmark``.
    """
    aligned = _align(returns, benchmark, rf)
    r = aligned[0]
    idx = 1
    bench = None
    rf_a = None
    if benchmark is not None:
        bench = aligned[idx]
        idx += 1
    if rf is not None:
        rf_a = aligned[idx]
        idx += 1

    ppy = periods_per_year
    n = len(r)
    ann_factor = np.sqrt(ppy)

    # Sharpe / Sortino excess base.
    excess = r - rf_a if rf_a is not None else r
    vol_ann = float(r.std(ddof=1) * ann_factor)
    ex_std = float(excess.std(ddof=1))
    sharpe = float(excess.mean() / ex_std * ann_factor) if ex_std > 0 else float("nan")
    downside = np.minimum(excess.to_numpy(), 0.0)
    downside_dev = float(np.sqrt(np.mean(downside**2)))
    sortino = (
        float(excess.mean() / downside_dev * ann_factor)
        if downside_dev > 0
        else float("nan")
    )

    dd = drawdown_series(r)
    max_drawdown = float(dd.min())
    dd_end = dd.idxmin()
    wealth = (1.0 + r).cumprod()
    dd_start = wealth.loc[:dd_end].idxmax()

    out: dict = {
        "cagr": _cagr(r, ppy),
        "vol_ann": vol_ann,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "max_dd_start": dd_start,
        "max_dd_end": dd_end,
        "best_period": float(r.max()),
        "worst_period": float(r.min()),
        "positive_share": float((r > 0).mean()),
        "n_obs": int(n),
        "start": r.index[0],
        "end": r.index[-1],
    }

    if bench is not None:
        active = r - bench
        te_ann = float(active.std(ddof=1) * ann_factor)
        act_std = float(active.std(ddof=1))
        info_ratio = (
            float(active.mean() / act_std * ann_factor) if act_std > 0 else float("nan")
        )
        var_b = float(bench.var(ddof=1))
        beta = float(r.cov(bench) / var_b) if var_b > 0 else float("nan")

        up = bench > 0
        down = bench < 0
        up_capture = float("nan")
        down_capture = float("nan")
        if up.any():
            fund_up = float((1.0 + r[up]).prod() - 1.0)
            bench_up = float((1.0 + bench[up]).prod() - 1.0)
            up_capture = fund_up / bench_up if bench_up != 0 else float("nan")
        if down.any():
            fund_dn = float((1.0 + r[down]).prod() - 1.0)
            bench_dn = float((1.0 + bench[down]).prod() - 1.0)
            down_capture = fund_dn / bench_dn if bench_dn != 0 else float("nan")

        out.update(
            {
                "tracking_error_ann": te_ann,
                "information_ratio": info_ratio,
                "beta": beta,
                "up_capture": up_capture,
                "down_capture": down_capture,
                "excess_cagr": _cagr(r, ppy) - _cagr(bench, ppy),
                "hit_rate": float((r > bench).mean()),
            }
        )

    return out


def rolling_excess(
    returns: pd.Series, benchmark: pd.Series, window: int = 12
) -> pd.Series:
    """Compute rolling annualised compounded excess return over ``benchmark``.

    For each trailing window the fund and benchmark returns are compounded,
    annualised as ``(prod(1 + r)) ** (12 / window) - 1`` (assuming 12 periods
    per year, matching this package's monthly convention), and the benchmark's
    annualised figure is subtracted from the fund's.

    Args:
        returns: Decimal period returns indexed by DatetimeIndex.
        benchmark: Decimal period benchmark returns.
        window: Rolling window length in periods.

    Returns:
        A pd.Series indexed like the aligned inputs (first ``window - 1``
        entries NaN) of rolling annualised excess return.
    """
    r, b = _align(returns, benchmark)
    exp = 12.0 / window

    def _ann(x: pd.Series) -> pd.Series:
        comp = x.rolling(window).apply(lambda w: (1.0 + w).prod(), raw=True)
        return comp**exp - 1.0

    return _ann(r) - _ann(b)

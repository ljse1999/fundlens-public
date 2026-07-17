"""Performance attribution: Brinson attribution and factor contribution decomposition."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fundlens.analysis.factor_model import FactorFit
from fundlens.analysis.holdings_analytics import _segment_weights


def brinson(
    fund_holdings: pd.DataFrame,
    bench_holdings: pd.DataFrame,
    fund_segment_returns: pd.Series | None,
    bench_segment_returns: pd.Series,
    by: str = "sector",
) -> pd.DataFrame:
    """Compute a single-period Brinson-Fachler attribution grouped by a segment column.

    This is a **single-period snapshot approximation**: it takes one set of
    holdings weights (fund and benchmark) and externally supplied per-segment
    returns for that same period, and decomposes the fund's over/underweight
    relative to the benchmark into allocation and selection effects. It does
    not itself compute segment returns from underlying security prices --
    those must be supplied by the caller (``fund_segment_returns`` /
    ``bench_segment_returns``).

    Weights on each side are renormalised to that side's own covered total
    (as in :func:`fundlens.analysis.holdings_analytics.tilts`), so partial
    holdings coverage is compared like-with-like rather than penalised.

    For each segment:

    * ``allocation`` = ``(w_fund - w_bench) * (r_bench_seg - r_bench_total)``
    * ``selection`` = ``w_fund * (r_fund_seg - r_bench_seg)`` when
      ``fund_segment_returns`` is given, else ``NaN``.

    Args:
        fund_holdings: Holdings DataFrame as produced by
            :mod:`fundlens.data.holdings`.
        bench_holdings: Benchmark holdings DataFrame with the same column
            contract.
        fund_segment_returns: Per-segment realised fund returns for the
            attribution period, indexed like ``by``'s distinct values.
            ``None`` if unavailable, in which case ``selection`` is ``NaN``
            for every segment.
        bench_segment_returns: Per-segment realised benchmark returns for the
            attribution period, indexed like ``by``'s distinct values.
            Required.
        by: Column to group by, e.g. "sector" or "country".

    Returns:
        A DataFrame indexed by the distinct values of ``by`` plus a "TOTAL"
        row, with columns ``fund_weight``, ``bench_weight``, ``allocation``,
        ``selection``, and ``total_effect`` (``allocation + selection``,
        ``NaN`` if ``selection`` is ``NaN``).
    """
    fund_seg = _segment_weights(fund_holdings, by)
    bench_seg = _segment_weights(bench_holdings, by)

    segments = fund_seg.index.union(bench_seg.index)
    fund_seg = fund_seg.reindex(segments, fill_value=0.0)
    bench_seg = bench_seg.reindex(segments, fill_value=0.0)

    bench_ret = bench_segment_returns.reindex(segments)
    bench_total_ret = float((bench_seg * bench_ret).sum())

    allocation = (fund_seg - bench_seg) * (bench_ret - bench_total_ret)

    if fund_segment_returns is not None:
        fund_ret = fund_segment_returns.reindex(segments)
        selection = fund_seg * (fund_ret - bench_ret)
    else:
        selection = pd.Series(np.nan, index=segments)

    total_effect = allocation + selection

    result = pd.DataFrame(
        {
            "fund_weight": fund_seg,
            "bench_weight": bench_seg,
            "allocation": allocation,
            "selection": selection,
            "total_effect": total_effect,
        }
    )
    result.index.name = by

    total_row = pd.DataFrame(
        {
            "fund_weight": [float(fund_seg.sum())],
            "bench_weight": [float(bench_seg.sum())],
            "allocation": [float(allocation.sum())],
            "selection": [float(selection.sum(skipna=False))],
            "total_effect": [float(total_effect.sum(skipna=False))],
        },
        index=pd.Index(["TOTAL"], name=by),
    )
    return pd.concat([result, total_row])


def factor_contributions(fit: FactorFit, factors: pd.DataFrame) -> pd.DataFrame:
    """Decompose the fund's average excess return over the fit window by factor.

    For each factor in ``fit.betas``, the annualised factor return over the
    fit window ``[fit.start, fit.end]`` is computed from the mean per-period
    factor return, compounded as ``(1 + mean_period_return) ** ppy - 1`` with
    ``ppy = 12`` (monthly convention, consistent with
    :mod:`fundlens.analysis.factor_model`). Its contribution to the fund's
    annualised return is ``beta * factor_return_ann``. An ``alpha`` row is
    appended with ``beta = NaN`` and ``contribution_ann = fit.alpha_ann``.

    This is an **approximation**: it arithmetically decomposes what is
    fundamentally a geometric (compounded) quantity, by applying each period
    beta to the annualised mean factor return rather than compounding the
    period-by-period product. It is intended as a directional/rough
    attribution, not an exact reconciliation of the fund's total annualised
    return.

    Args:
        fit: A fitted :class:`fundlens.analysis.factor_model.FactorFit`.
        factors: Factor return DataFrame covering (at least) ``fit.start``
            to ``fit.end``, with the same columns the model was fit on.

    Returns:
        A DataFrame indexed by component name (factor names plus "alpha"),
        with columns ``beta``, ``factor_return_ann``, ``contribution_ann``.
    """
    ppy = 12
    window = factors.loc[fit.start : fit.end]

    rows = []
    index = []
    for factor_name, beta in fit.betas.items():
        mean_period = float(window[factor_name].mean())
        factor_return_ann = (1.0 + mean_period) ** ppy - 1.0
        contribution_ann = beta * factor_return_ann
        rows.append(
            {
                "beta": beta,
                "factor_return_ann": factor_return_ann,
                "contribution_ann": contribution_ann,
            }
        )
        index.append(factor_name)

    rows.append(
        {
            "beta": float("nan"),
            "factor_return_ann": float("nan"),
            "contribution_ann": fit.alpha_ann,
        }
    )
    index.append("alpha")

    return pd.DataFrame(rows, index=pd.Index(index, name="component"))

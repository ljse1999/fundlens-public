"""Holdings-based analytics: active share, concentration, and tilts.

Holdings DataFrames follow the contract used throughout ``fundlens``:
columns ``["ticker", "isin", "name", "weight", "sector", "country",
"market_cap"]``, with ``weight`` a decimal fraction of the portfolio.

**Partial coverage.** Morningstar (and similar data providers) sometimes
publish only a subset of a fund's holdings -- e.g. "top 10" holdings whose
weights sum to well under 1.0 -- while other funds report holdings summing
close to (but rarely exactly) 1.0. Every function here handles partial
coverage explicitly rather than silently assuming full coverage; see each
docstring for how.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_SUFFIX_RE = re.compile(
    r"\b(plc|inc|incorporated|ltd|limited|corp|corporation|co|sa|nv|ag|se|"
    r"class\s+[a-z]|adr|ord|shs)\b\.?",
    flags=re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def _norm_ticker(ticker) -> str | None:
    if ticker is None or (isinstance(ticker, float) and np.isnan(ticker)):
        return None
    t = str(ticker).strip()
    if not t:
        return None
    # Strip exchange suffix, e.g. "AAPL.US" / "AAPL:US" / "AAPL US"
    t = re.split(r"[.: ]", t)[0]
    return t.upper()


def _norm_name(name) -> str | None:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return None
    n = str(name).strip()
    if not n:
        return None
    n = n.casefold()
    n = _SUFFIX_RE.sub(" ", n)
    n = _PUNCT_RE.sub(" ", n)
    n = _WS_RE.sub(" ", n).strip()
    return n or None


def _norm_isin(isin) -> str | None:
    if isin is None or (isinstance(isin, float) and np.isnan(isin)):
        return None
    i = str(isin).strip().upper()
    return i or None


def _stage_match(
    fund_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    key_col: str,
    normalizer,
    stage: str,
    fund_weights: dict,
    bench_weights: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match remaining rows on ``key_col`` (normalised), for keys present on
    both sides only. Matched rows' weights are aggregated into
    ``fund_weights`` / ``bench_weights`` (keyed by ``(stage, key)`` so keys
    don't collide across matching stages) and removed from the returned
    "remaining" frames.
    """
    f_keys = fund_df[key_col].map(normalizer)
    b_keys = bench_df[key_col].map(normalizer)
    common = {k for k in f_keys if k is not None} & {k for k in b_keys if k is not None}
    if not common:
        return fund_df, bench_df
    f_mask = f_keys.isin(common)
    b_mask = b_keys.isin(common)
    for key in common:
        fund_weights[(stage, key)] = float(fund_df.loc[f_keys == key, "weight"].sum())
        bench_weights[(stage, key)] = float(bench_df.loc[b_keys == key, "weight"].sum())
    return (
        fund_df.loc[~f_mask].reset_index(drop=True),
        bench_df.loc[~b_mask].reset_index(drop=True),
    )


def active_share(fund_holdings: pd.DataFrame, bench_holdings: pd.DataFrame) -> float:
    """Compute active share between a fund's holdings and a benchmark's holdings.

    Active share is 0.5 * sum(|w_fund - w_bench|) over the union of
    positions, where positions are matched by ISIN when both sides provide
    one, else by upper-cased ticker with exchange suffix stripped, else by a
    normalised name (casefold, punctuation stripped, common suffixes such as
    "PLC"/"Inc"/"Ltd"/share-class markers like "Class A" removed).

    This is computed on RAW (unscaled) weights, i.e. weights are used as
    published without renormalising to 100%. When either side has partial
    holdings coverage (weights summing to less than 1.0 -- see
    :func:`concentration`'s ``coverage`` key), the true active share against
    full portfolios is generally *higher* than what is returned here: missing
    positions on one side are simply absent from the union rather than
    contributing their true (unknown) weight difference. Callers should
    therefore treat the returned value as a **lower bound** on true active
    share whenever coverage is below 1.0, and check coverage via
    :func:`concentration` before over-interpreting a small number.

    Args:
        fund_holdings: Holdings DataFrame as produced by
            :mod:`fundlens.data.holdings`.
        bench_holdings: Benchmark holdings DataFrame with the same column
            contract.

    Returns:
        Active share as a decimal fraction in [0, 1] (a lower bound under
        partial coverage).
    """
    fund_weights: dict = {}
    bench_weights: dict = {}

    fund_remaining, bench_remaining = _stage_match(
        fund_holdings, bench_holdings, "isin", _norm_isin, "isin", fund_weights, bench_weights
    )
    fund_remaining, bench_remaining = _stage_match(
        fund_remaining, bench_remaining, "ticker", _norm_ticker, "ticker", fund_weights, bench_weights
    )
    fund_remaining, bench_remaining = _stage_match(
        fund_remaining, bench_remaining, "name", _norm_name, "name", fund_weights, bench_weights
    )

    keys = set(fund_weights) | set(bench_weights)
    matched_abs_diff = sum(
        abs(fund_weights.get(k, 0.0) - bench_weights.get(k, 0.0)) for k in keys
    )
    # Rows that matched nothing on the other side contribute their full
    # weight to the difference (equivalent to comparing against zero).
    unmatched_fund = float(fund_remaining["weight"].sum()) if len(fund_remaining) else 0.0
    unmatched_bench = float(bench_remaining["weight"].sum()) if len(bench_remaining) else 0.0

    return 0.5 * (matched_abs_diff + unmatched_fund + unmatched_bench)


def concentration(fund_holdings: pd.DataFrame) -> dict:
    """Compute concentration metrics for a fund's holdings.

    Args:
        fund_holdings: Holdings DataFrame as produced by
            :mod:`fundlens.data.holdings`.

    Returns:
        A dict with keys:

        * ``n_holdings``: number of holdings rows.
        * ``top5_weight`` / ``top10_weight``: sum of the 5 / 10 largest raw
          weights (fewer if the fund has fewer positions).
        * ``hhi``: Herfindahl-Hirschman index, sum of squared weights *after
          renormalising weights to their own sum* (so partial-coverage funds
          aren't penalised for missing holdings).
        * ``effective_n``: ``1 / hhi``, the effective number of holdings.
        * ``coverage``: sum of raw (unscaled) weights -- the share of the
          portfolio the published holdings represent. 1.0 for full
          disclosure, less than 1.0 for partial (e.g. "top N") disclosure.
    """
    weights = fund_holdings["weight"].astype(float)
    n_holdings = int(len(weights))
    sorted_w = weights.sort_values(ascending=False)
    top5_weight = float(sorted_w.iloc[:5].sum())
    top10_weight = float(sorted_w.iloc[:10].sum())
    coverage = float(weights.sum())

    if coverage > 0:
        renorm = weights / coverage
        hhi = float((renorm ** 2).sum())
    else:
        hhi = float("nan")
    effective_n = (1.0 / hhi) if hhi and not np.isnan(hhi) else float("nan")

    return {
        "n_holdings": n_holdings,
        "top5_weight": top5_weight,
        "top10_weight": top10_weight,
        "hhi": hhi,
        "effective_n": effective_n,
        "coverage": coverage,
    }


def _segment_weights(holdings: pd.DataFrame, by: str) -> pd.Series:
    """Sum raw weights per segment, renormalised to the holdings' own total.

    NaN/missing segment values are bucketed into "Unclassified".
    """
    df = holdings[[by, "weight"]].copy()
    df[by] = df[by].where(df[by].notna(), "Unclassified")
    df[by] = df[by].apply(lambda v: v if isinstance(v, str) and v.strip() else "Unclassified")
    grouped = df.groupby(by)["weight"].sum().astype(float)
    total = float(holdings["weight"].astype(float).sum())
    if total > 0:
        grouped = grouped / total
    return grouped


def tilts(
    fund_holdings: pd.DataFrame,
    bench_holdings: pd.DataFrame,
    by: str = "sector",
) -> pd.DataFrame:
    """Compute the fund's over/underweight tilts relative to a benchmark, grouped by a column.

    Each side's weights are renormalised to that side's own covered total
    (i.e. the fund's segment weights sum to 1.0, and likewise for the
    benchmark) before computing the difference, so that funds/benchmarks
    with partial holdings coverage are compared like-with-like rather than
    penalised for missing weight. Rows with a missing/NaN ``by`` value are
    grouped into an "Unclassified" segment.

    Args:
        fund_holdings: Holdings DataFrame as produced by
            :mod:`fundlens.data.holdings`.
        bench_holdings: Benchmark holdings DataFrame with the same column
            contract.
        by: Column to group by, e.g. "sector" or "country".

    Returns:
        A DataFrame indexed by the distinct values of ``by`` (plus possibly
        "Unclassified"), with columns ``fund_weight``, ``bench_weight``, and
        ``active_weight`` (``fund_weight - bench_weight``), sorted by
        ``|active_weight|`` descending.
    """
    fund_seg = _segment_weights(fund_holdings, by)
    bench_seg = _segment_weights(bench_holdings, by)

    segments = fund_seg.index.union(bench_seg.index)
    fund_seg = fund_seg.reindex(segments, fill_value=0.0)
    bench_seg = bench_seg.reindex(segments, fill_value=0.0)

    result = pd.DataFrame(
        {
            "fund_weight": fund_seg,
            "bench_weight": bench_seg,
        }
    )
    result["active_weight"] = result["fund_weight"] - result["bench_weight"]
    result = result.reindex(result["active_weight"].abs().sort_values(ascending=False).index)
    result.index.name = by
    return result

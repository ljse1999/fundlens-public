"""Tests for fundlens.analysis.holdings_analytics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fundlens.analysis.holdings_analytics import active_share, concentration, tilts


def _holdings(rows: list[dict]) -> pd.DataFrame:
    """Build a holdings DataFrame, filling any missing contract columns with NaN."""
    cols = ["ticker", "isin", "name", "weight", "sector", "country", "market_cap"]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]


# --------------------------------------------------------------------------
# active_share
# --------------------------------------------------------------------------


def test_active_share_known_overlap_isin_and_ticker_matching():
    # Fund: 4 positions, full coverage (sums to 1.0).
    fund = _holdings(
        [
            {"isin": "XXXX1", "ticker": "AAA", "weight": 0.40},
            {"isin": "XXXX2", "ticker": "BBB", "weight": 0.30},
            # No ISIN on this row -- must be matched to bench by ticker.
            {"isin": None, "ticker": "CCC", "weight": 0.20},
            {"isin": "XXXX4", "ticker": "DDD", "weight": 0.10},
        ]
    )
    # Bench: 5 positions, full coverage (sums to 1.0).
    bench = _holdings(
        [
            {"isin": "XXXX1", "ticker": "AAA", "weight": 0.25},
            {"isin": "XXXX2", "ticker": "BBB", "weight": 0.25},
            # Different (or absent) ISIN vs fund's CCC row -- ticker match required.
            {"isin": "XXXX3", "ticker": "CCC", "weight": 0.25},
            {"isin": "XXXX5", "ticker": "EEE", "weight": 0.15},
            {"isin": "XXXX6", "ticker": "FFF", "weight": 0.10},
        ]
    )
    # |0.40-0.25| + |0.30-0.25| + |0.20-0.25| + |0.10-0| + |0-0.15| + |0-0.10|
    # = 0.15 + 0.05 + 0.05 + 0.10 + 0.15 + 0.10 = 0.60 -> * 0.5 = 0.30
    assert active_share(fund, bench) == pytest.approx(0.30)


def test_active_share_identical_portfolios_is_zero():
    fund = _holdings(
        [
            {"isin": "XXXX1", "ticker": "AAA", "weight": 0.6},
            {"isin": "XXXX2", "ticker": "BBB", "weight": 0.4},
        ]
    )
    bench = fund.copy()
    assert active_share(fund, bench) == pytest.approx(0.0)


def test_active_share_fully_disjoint_full_coverage_is_one():
    fund = _holdings(
        [
            {"isin": "XXXX1", "ticker": "AAA", "weight": 0.7},
            {"isin": "XXXX2", "ticker": "BBB", "weight": 0.3},
        ]
    )
    bench = _holdings(
        [
            {"isin": "YYYY1", "ticker": "ZZZ", "weight": 0.5},
            {"isin": "YYYY2", "ticker": "WWW", "weight": 0.5},
        ]
    )
    assert active_share(fund, bench) == pytest.approx(1.0)


def test_active_share_partial_coverage_is_a_lower_bound():
    # Fund only covers 0.5 of the true portfolio (e.g. "top N" disclosure).
    fund = _holdings(
        [
            {"isin": "I1", "ticker": "AAA", "weight": 0.30},
            {"isin": "I2", "ticker": "BBB", "weight": 0.20},
        ]
    )
    # Bench fully covers the same two names.
    bench = _holdings(
        [
            {"isin": "I1", "ticker": "AAA", "weight": 0.60},
            {"isin": "I2", "ticker": "BBB", "weight": 0.40},
        ]
    )
    # 0.5 * (|0.30-0.60| + |0.20-0.40|) = 0.5 * (0.30 + 0.20) = 0.25
    assert active_share(fund, bench) == pytest.approx(0.25)
    assert concentration(fund)["coverage"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# concentration
# --------------------------------------------------------------------------


def test_concentration_known_weights_exact():
    weights = [0.20, 0.15, 0.15, 0.10, 0.10, 0.08, 0.07, 0.05, 0.05, 0.03, 0.01, 0.01]
    assert sum(weights) == pytest.approx(1.00)
    fund = _holdings([{"weight": w, "ticker": f"T{i}"} for i, w in enumerate(weights)])

    result = concentration(fund)

    assert result["n_holdings"] == 12
    assert result["top5_weight"] == pytest.approx(0.70)
    assert result["top10_weight"] == pytest.approx(0.98)

    expected_hhi = sum(w**2 for w in weights)  # already renormalised: sum(weights) == 1.0
    assert result["hhi"] == pytest.approx(expected_hhi)
    assert result["effective_n"] == pytest.approx(1.0 / expected_hhi)
    assert result["coverage"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# tilts
# --------------------------------------------------------------------------


def test_tilts_renormalised_active_weights_and_unclassified_bucket():
    # Fund: partial coverage (0.6 total), one row with no sector.
    fund = _holdings(
        [
            {"sector": "Tech", "weight": 0.30},
            {"sector": "Health", "weight": 0.20},
            {"sector": np.nan, "weight": 0.10},
        ]
    )
    # Bench: full coverage (1.0 total), no unclassified rows.
    bench = _holdings(
        [
            {"sector": "Tech", "weight": 0.40},
            {"sector": "Health", "weight": 0.40},
            {"sector": "Utilities", "weight": 0.20},
        ]
    )

    result = tilts(fund, bench, by="sector")

    # Fund segment weights renormalised to its own total (0.6):
    # Tech = 0.5, Health = 1/3, Unclassified = 1/6.
    assert result.loc["Tech", "fund_weight"] == pytest.approx(0.5)
    assert result.loc["Health", "fund_weight"] == pytest.approx(1 / 3)
    assert result.loc["Unclassified", "fund_weight"] == pytest.approx(1 / 6)
    assert result.loc["Utilities", "fund_weight"] == pytest.approx(0.0)

    assert result.loc["Tech", "bench_weight"] == pytest.approx(0.4)
    assert result.loc["Health", "bench_weight"] == pytest.approx(0.4)
    assert result.loc["Utilities", "bench_weight"] == pytest.approx(0.2)
    assert result.loc["Unclassified", "bench_weight"] == pytest.approx(0.0)

    assert result.loc["Tech", "active_weight"] == pytest.approx(0.5 - 0.4)
    assert result.loc["Health", "active_weight"] == pytest.approx(1 / 3 - 0.4)
    assert result.loc["Utilities", "active_weight"] == pytest.approx(0.0 - 0.2)
    assert result.loc["Unclassified", "active_weight"] == pytest.approx(1 / 6 - 0.0)

    # Sorted by |active_weight| descending.
    abs_active = result["active_weight"].abs().tolist()
    assert abs_active == sorted(abs_active, reverse=True)

"""Tests for fundlens.analysis.attribution."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fundlens.analysis.attribution import brinson, factor_contributions
from fundlens.analysis.factor_model import FactorFit


def _holdings(rows: list[dict]) -> pd.DataFrame:
    cols = ["ticker", "isin", "name", "weight", "sector", "country", "market_cap"]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]


# --------------------------------------------------------------------------
# brinson
# --------------------------------------------------------------------------


def _classic_two_segment_holdings():
    fund = _holdings(
        [
            {"sector": "Equities", "weight": 0.60},
            {"sector": "Bonds", "weight": 0.40},
        ]
    )
    bench = _holdings(
        [
            {"sector": "Equities", "weight": 0.50},
            {"sector": "Bonds", "weight": 0.50},
        ]
    )
    return fund, bench


def test_brinson_classic_textbook_example_with_selection():
    fund, bench = _classic_two_segment_holdings()
    bench_returns = pd.Series({"Equities": 0.10, "Bonds": 0.04})
    fund_returns = pd.Series({"Equities": 0.12, "Bonds": 0.03})

    result = brinson(fund, bench, fund_returns, bench_returns, by="sector")

    # bench_total = 0.5*0.10 + 0.5*0.04 = 0.07
    # allocation_Equities = (0.60-0.50)*(0.10-0.07) = 0.003
    # allocation_Bonds = (0.40-0.50)*(0.04-0.07) = 0.003
    assert result.loc["Equities", "allocation"] == pytest.approx(0.003)
    assert result.loc["Bonds", "allocation"] == pytest.approx(0.003)

    # selection_Equities = 0.60*(0.12-0.10) = 0.012
    # selection_Bonds = 0.40*(0.03-0.04) = -0.004
    assert result.loc["Equities", "selection"] == pytest.approx(0.012)
    assert result.loc["Bonds", "selection"] == pytest.approx(-0.004)

    assert result.loc["Equities", "total_effect"] == pytest.approx(0.015)
    assert result.loc["Bonds", "total_effect"] == pytest.approx(-0.001)

    assert result.loc["TOTAL", "allocation"] == pytest.approx(0.006)
    assert result.loc["TOTAL", "selection"] == pytest.approx(0.008)
    assert result.loc["TOTAL", "total_effect"] == pytest.approx(0.014)


def test_brinson_no_fund_returns_selection_is_all_nan_allocation_exact():
    fund, bench = _classic_two_segment_holdings()
    bench_returns = pd.Series({"Equities": 0.10, "Bonds": 0.04})

    result = brinson(fund, bench, None, bench_returns, by="sector")

    assert result.loc["Equities", "allocation"] == pytest.approx(0.003)
    assert result.loc["Bonds", "allocation"] == pytest.approx(0.003)
    assert result.loc["TOTAL", "allocation"] == pytest.approx(0.006)

    assert result["selection"].isna().all()
    assert pd.isna(result.loc["TOTAL", "selection"])
    assert result["total_effect"].isna().all()


# --------------------------------------------------------------------------
# factor_contributions
# --------------------------------------------------------------------------


def test_factor_contributions_exact_with_constant_factor_returns():
    start = pd.Timestamp("2020-01-31")
    end = pd.Timestamp("2022-12-31")
    fit = FactorFit(
        model="capm",
        alpha_ann=0.02,
        alpha_t=1.5,
        alpha_p_bootstrap=None,
        betas={"MKT_RF": 1.2},
        beta_t={"MKT_RF": 3.0},
        r2=0.8,
        resid_vol_ann=0.1,
        nobs=36,
        start=start,
        end=end,
    )
    dates = pd.date_range(start=start, end=end, freq="ME")
    factors = pd.DataFrame({"MKT_RF": 0.01}, index=dates)

    result = factor_contributions(fit, factors)

    expected_factor_return_ann = (1.01) ** 12 - 1.0
    expected_contribution = 1.2 * expected_factor_return_ann

    assert result.loc["MKT_RF", "beta"] == pytest.approx(1.2)
    assert result.loc["MKT_RF", "factor_return_ann"] == pytest.approx(expected_factor_return_ann)
    assert result.loc["MKT_RF", "contribution_ann"] == pytest.approx(expected_contribution)

    assert np.isnan(result.loc["alpha", "beta"])
    assert result.loc["alpha", "contribution_ann"] == pytest.approx(0.02)

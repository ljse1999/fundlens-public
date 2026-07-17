"""Tests for fundlens.analysis.factor_model on synthetic fixtures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from fundlens.analysis.factor_model import (
    _default_nw_lags,
    fit_factor_model,
    rolling_betas,
    subperiod_alphas,
)


def _midx(n: int, start: str = "2000-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


def _make_factors(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MKT_RF": rng.normal(0.005, 0.045, n),
            "SMB": rng.normal(0.001, 0.030, n),
            "HML": rng.normal(0.001, 0.030, n),
            "RMW": rng.normal(0.001, 0.020, n),
            "CMA": rng.normal(0.001, 0.020, n),
            "MOM": rng.normal(0.003, 0.040, n),
        },
        index=_midx(n),
    )


def test_factor_recovery():
    rng = np.random.default_rng(42)
    n = 240
    factors = _make_factors(n, rng)
    true_alpha = 0.002
    fund_excess = pd.Series(
        true_alpha
        + 1.1 * factors["MKT_RF"]
        + 0.3 * factors["SMB"]
        - 0.2 * factors["HML"]
        + rng.normal(0, 0.005, n),
        index=factors.index,
    )
    fit = fit_factor_model(fund_excess, factors, model="ff3", bootstrap=True)

    assert fit.betas["MKT_RF"] == pytest.approx(1.1, abs=0.05)
    assert fit.betas["SMB"] == pytest.approx(0.3, abs=0.05)
    assert fit.betas["HML"] == pytest.approx(-0.2, abs=0.05)
    exp_alpha_ann = (1 + true_alpha) ** 12 - 1
    assert fit.alpha_ann == pytest.approx(exp_alpha_ann, abs=0.01)
    assert fit.alpha_t > 2
    assert fit.r2 > 0.9
    assert fit.alpha_p_bootstrap is not None and fit.alpha_p_bootstrap < 0.05
    assert fit.nobs == n


def test_zero_alpha_fund():
    rng = np.random.default_rng(42)
    n = 240
    factors = _make_factors(n, rng)
    fund_excess = pd.Series(
        0.0
        + 1.0 * factors["MKT_RF"]
        + 0.2 * factors["SMB"]
        + rng.normal(0, 0.010, n),
        index=factors.index,
    )
    fit = fit_factor_model(fund_excess, factors, model="ff3", bootstrap=True)
    assert abs(fit.alpha_t) < 2
    assert fit.alpha_p_bootstrap > 0.05


def test_newey_west_alpha_t_cross_check():
    rng = np.random.default_rng(42)
    n = 200
    factors = _make_factors(n, rng)
    fund_excess = pd.Series(
        0.001 + 0.9 * factors["MKT_RF"] + rng.normal(0, 0.008, n),
        index=factors.index,
    )
    fit = fit_factor_model(fund_excess, factors, model="capm", bootstrap=False)

    lags = _default_nw_lags(n)
    X = sm.add_constant(factors[["MKT_RF"]])
    res = sm.OLS(fund_excess.to_numpy(), X.to_numpy()).fit().get_robustcov_results(
        "HAC", maxlags=lags
    )
    assert fit.alpha_t == pytest.approx(float(res.tvalues[0]), abs=1e-8)
    assert fit.betas["MKT_RF"] == pytest.approx(float(res.params[1]), abs=1e-8)


def test_rolling_betas_regime_shift():
    rng = np.random.default_rng(42)
    n = 240
    factors = _make_factors(n, rng)
    mkt = factors["MKT_RF"].to_numpy()
    beta_path = np.where(np.arange(n) < n // 2, 0.8, 1.2)
    fund = pd.Series(beta_path * mkt + rng.normal(0, 0.003, n), index=factors.index)

    window = 36
    rb = rolling_betas(fund, factors, model="capm", window=window)
    assert list(rb.columns) == ["alpha_ann", "MKT_RF"]
    assert rb.index[-1] == factors.index[-1]

    # Window ending well inside the first regime.
    early = rb["MKT_RF"].iloc[window - 1 + 5]
    # Window ending well inside the second regime (last window is all regime 2).
    late = rb["MKT_RF"].iloc[-1]
    assert early == pytest.approx(0.8, abs=0.1)
    assert late == pytest.approx(1.2, abs=0.1)


def test_subperiod_alphas_shape():
    rng = np.random.default_rng(42)
    n = 180
    factors = _make_factors(n, rng)
    fund = pd.Series(
        0.002 + 1.0 * factors["MKT_RF"] + rng.normal(0, 0.006, n),
        index=factors.index,
    )
    sub = subperiod_alphas(fund, factors, model="capm", n_periods=3)
    assert len(sub) == 3
    assert set(sub.columns) == {"alpha_ann", "alpha_t", "r2", "nobs", "start", "end"}
    assert sub["nobs"].sum() == n
    assert sub["start"].iloc[0] == factors.index[0]
    assert sub["end"].iloc[-1] == factors.index[-1]


def test_too_few_observations_raises():
    rng = np.random.default_rng(42)
    n = 20  # < max(24, k+10)
    factors = _make_factors(n, rng)
    fund = pd.Series(rng.normal(0, 0.01, n), index=factors.index)
    with pytest.raises(ValueError, match="Insufficient observations"):
        fit_factor_model(fund, factors, model="ff3", bootstrap=False)


def test_missing_factor_column_raises():
    rng = np.random.default_rng(42)
    n = 60
    factors = _make_factors(n, rng).drop(columns=["MOM"])
    fund = pd.Series(rng.normal(0, 0.01, n), index=factors.index)
    with pytest.raises(ValueError, match="missing required column"):
        fit_factor_model(fund, factors, model="ff5_mom", bootstrap=False)

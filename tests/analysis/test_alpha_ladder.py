from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fundlens.analysis.alpha_ladder import (
    build_alpha_ladder,
    fit_benchmark_residual_alpha,
    fit_exposure_model,
)
from fundlens.analysis.factor_model import FactorFit


def _midx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2015-01-31", periods=n, freq="ME")


def test_benchmark_residual_alpha_zero_for_benchmark_beta():
    rng = np.random.default_rng(123)
    n = 120
    idx = _midx(n)
    rf = pd.Series(0.001, index=idx)
    benchmark = pd.Series(rng.normal(0.006, 0.035, n), index=idx)
    fund = rf + 1.2 * (benchmark - rf) + rng.normal(0.0, 0.002, n)

    step = fit_benchmark_residual_alpha(
        fund,
        benchmark,
        rf,
        benchmark_ticker="SWDA.L",
        bootstrap=False,
    )

    assert step.id == "benchmark"
    assert step.label == "Benchmark residual alpha"
    assert step.exposures["SWDA.L_excess"] == pytest.approx(1.2, abs=0.03)
    assert abs(step.alpha_ann) < 0.01
    assert abs(step.alpha_t or 0.0) < 2.0


def test_exposure_model_retains_true_intercept():
    rng = np.random.default_rng(456)
    n = 180
    idx = _midx(n)
    proxy = pd.Series(rng.normal(0.005, 0.030, n), index=idx, name="proxy")
    true_alpha = 0.002
    fund_excess = true_alpha + 0.8 * proxy + rng.normal(0.0, 0.003, n)

    step = fit_exposure_model(
        fund_excess,
        proxy,
        model_id="proxy",
        label="Proxy alpha",
        bootstrap=False,
    )

    assert step.alpha_ann == pytest.approx((1.0 + true_alpha) ** 12 - 1.0, abs=0.01)
    assert step.alpha_t is not None and step.alpha_t > 2.0
    assert step.exposures["proxy"] == pytest.approx(0.8, abs=0.04)


def test_alpha_ladder_verdict_uses_benchmark_step_without_changing_factor_fit():
    idx = _midx(60)
    ff5 = FactorFit(
        model="ff5_mom",
        alpha_ann=0.04,
        alpha_t=2.5,
        alpha_p_bootstrap=0.01,
        betas={"MKT_RF": 1.0},
        beta_t={"MKT_RF": 5.0},
        r2=0.8,
        resid_vol_ann=0.05,
        nobs=60,
        start=idx[0],
        end=idx[-1],
    )
    benchmark = fit_exposure_model(
        pd.Series(np.linspace(-0.01, 0.01, 60), index=idx),
        pd.Series(np.linspace(-0.01, 0.01, 60), index=idx, name="proxy"),
        model_id="benchmark",
        label="Benchmark residual alpha",
        bootstrap=False,
    )

    ladder = build_alpha_ladder({"ff5_mom": ff5}, benchmark_step=benchmark)

    assert ladder.steps["ff5_mom"].alpha_ann == ff5.alpha_ann
    assert ladder.verdict == "benchmark_explained"

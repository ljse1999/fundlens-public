"""Tests for fundlens.analysis.returns on synthetic fixtures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fundlens.analysis.returns import drawdown_series, perf_summary, rolling_excess


def _midx(n: int, start: str = "2005-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


def test_perf_summary_hand_computed_cagr_and_drawdown():
    # 12-point series with a clean, hand-computable drawdown.
    r = pd.Series(
        [0.10, 0.10, -0.20, -0.20, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        index=_midx(12),
    )
    s = perf_summary(r, periods_per_year=12)

    total_growth = float((1 + r).prod())
    expected_cagr = total_growth ** (12 / 12) - 1
    assert s["cagr"] == pytest.approx(expected_cagr, abs=1e-12)
    assert s["n_obs"] == 12

    # Peak at index 1 (after two +10% months), trough at index 3.
    wealth = (1 + r).cumprod()
    expected_max_dd = float((wealth / wealth.cummax() - 1).min())
    assert s["max_drawdown"] == pytest.approx(expected_max_dd, abs=1e-12)
    assert s["max_drawdown"] == pytest.approx(1.20 * 0.8 * 0.8 / (1.20) - 1, abs=1e-12)
    assert s["max_dd_start"] == r.index[1]
    assert s["max_dd_end"] == r.index[3]
    assert s["best_period"] == pytest.approx(0.10)
    assert s["worst_period"] == pytest.approx(-0.20)
    assert s["positive_share"] == pytest.approx(10 / 12)


def test_drawdown_series_nonpositive_and_zero_at_peak():
    r = pd.Series([0.05, 0.05, -0.10, 0.20], index=_midx(4))
    dd = drawdown_series(r)
    assert (dd <= 1e-15).all()
    assert dd.iloc[0] == pytest.approx(0.0)  # first period is a new peak
    assert dd.iloc[1] == pytest.approx(0.0)  # still rising -> at peak


def test_up_down_capture_constructed():
    # Benchmark alternates up/down; fund captures 50% up, 50% down moves.
    bench = pd.Series([0.10, -0.10, 0.10, -0.10, 0.10, -0.10], index=_midx(6))
    fund = 0.5 * bench
    s = perf_summary(fund, benchmark=bench, periods_per_year=12)

    up = bench > 0
    dn = bench < 0
    exp_up = ((1 + fund[up]).prod() - 1) / ((1 + bench[up]).prod() - 1)
    exp_dn = ((1 + fund[dn]).prod() - 1) / ((1 + bench[dn]).prod() - 1)
    assert s["up_capture"] == pytest.approx(exp_up, abs=1e-12)
    assert s["down_capture"] == pytest.approx(exp_dn, abs=1e-12)
    # 50% linear capture -> geometric capture near 0.5 (not exact under compounding).
    assert s["up_capture"] == pytest.approx(0.5, abs=0.05)
    assert s["hit_rate"] == pytest.approx(0.5)


def test_benchmark_keys_and_beta():
    rng = np.random.default_rng(42)
    n = 180
    idx = _midx(n)
    bench = pd.Series(rng.normal(0.005, 0.04, n), index=idx)
    fund = pd.Series(1.3 * bench.to_numpy() + rng.normal(0, 0.005, n), index=idx)
    s = perf_summary(fund, benchmark=bench, rf=None, periods_per_year=12)
    for key in (
        "tracking_error_ann",
        "information_ratio",
        "beta",
        "up_capture",
        "down_capture",
        "excess_cagr",
        "hit_rate",
        "sortino",
    ):
        assert key in s
    assert s["beta"] == pytest.approx(1.3, abs=0.05)


def test_misaligned_indices_are_intersected():
    rng = np.random.default_rng(1)
    full = _midx(24)
    fund = pd.Series(rng.normal(0.01, 0.03, 24), index=full)
    # Benchmark covers only a sub-window.
    bench = pd.Series(rng.normal(0.008, 0.03, 12), index=full[6:18])
    s = perf_summary(fund, benchmark=bench, periods_per_year=12)
    assert s["n_obs"] == 12
    assert s["start"] == full[6]
    assert s["end"] == full[17]


def test_sharpe_uses_excess_over_rf():
    rng = np.random.default_rng(7)
    n = 120
    idx = _midx(n)
    fund = pd.Series(rng.normal(0.01, 0.03, n), index=idx)
    rf = pd.Series(np.full(n, 0.002), index=idx)
    s_raw = perf_summary(fund, periods_per_year=12)
    s_rf = perf_summary(fund, rf=rf, periods_per_year=12)
    # With a positive rf, excess Sharpe is lower than the raw Sharpe.
    assert s_rf["sharpe"] < s_raw["sharpe"]


def test_rolling_excess_window_matches_manual():
    rng = np.random.default_rng(3)
    n = 60
    idx = _midx(n)
    fund = pd.Series(rng.normal(0.01, 0.03, n), index=idx)
    bench = pd.Series(rng.normal(0.008, 0.03, n), index=idx)
    re = rolling_excess(fund, bench, window=12)
    assert re.iloc[:11].isna().all()
    # window=12 -> annualisation exponent 1 -> trailing 12m compound excess.
    w = slice(0, 12)
    exp = (1 + fund.iloc[w]).prod() - (1 + bench.iloc[w]).prod()
    assert re.iloc[11] == pytest.approx(exp, abs=1e-12)

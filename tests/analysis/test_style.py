"""Tests for fundlens.analysis.style on synthetic fixtures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fundlens.analysis.style import rbsa, style_drift_score


def _midx(n: int, start: str = "2000-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


def _proxies(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": rng.normal(0.006, 0.045, n),
            "B": rng.normal(0.004, 0.035, n),
            "C": rng.normal(0.005, 0.050, n),
        },
        index=_midx(n),
    )


def test_rbsa_recovers_blend_and_rows_sum_to_one():
    rng = np.random.default_rng(42)
    n = 120
    proxies = _proxies(n, rng)
    fund = pd.Series(
        0.6 * proxies["A"] + 0.4 * proxies["B"] + rng.normal(0, 0.0005, n),
        index=proxies.index,
    )
    w = rbsa(fund, proxies, window=36)

    assert list(w.columns) == ["A", "B", "C"]
    assert np.allclose(w.sum(axis=1).to_numpy(), 1.0, atol=1e-6)
    assert (w.to_numpy() >= -1e-9).all()

    last = w.iloc[-1]
    assert last["A"] == pytest.approx(0.6, abs=0.05)
    assert last["B"] == pytest.approx(0.4, abs=0.05)
    assert last["C"] == pytest.approx(0.0, abs=0.05)


def test_style_drift_score_constant_blend_near_zero():
    rng = np.random.default_rng(42)
    n = 120
    proxies = _proxies(n, rng)
    fund = pd.Series(0.5 * proxies["A"] + 0.5 * proxies["C"], index=proxies.index)
    w = rbsa(fund, proxies, window=36)
    score = style_drift_score(w)
    assert score == pytest.approx(0.0, abs=1e-3)


def test_style_drift_score_definition():
    # Deterministic hand-check of the drift definition.
    w = pd.DataFrame(
        {"A": [0.5, 0.6, 0.4], "B": [0.5, 0.4, 0.6]}, index=_midx(3)
    )
    # diff() rows: [NaN->0.0], |0.1|+|-0.1|=0.2, |-0.2|+|0.2|=0.4; mean([0,0.2,0.4])=0.2
    assert style_drift_score(w) == pytest.approx(0.2, abs=1e-12)

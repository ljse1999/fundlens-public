"""Pure-math unit tests for leg-wise FX translation in fundlens.data.fx.

These tests hit the internal ``_translate`` helper directly with hand-built,
already-aligned inputs, so no network access (yfinance / FRED) is required.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fundlens.data.fx import _translate


def _idx(n: int = 1) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-31", periods=n, freq="ME")


def test_hand_computed_single_period_market_translation():
    """MKT_RF_usd=0.02, RF_usd=0.004, fx=0.01, rf_local=0.003.

    MKT_RF_out must equal (1.024)(1.01) - 1 - 0.003 exactly.
    """
    idx = _idx(1)
    factors_usd = pd.DataFrame({"MKT_RF": [0.02], "RF": [0.004]}, index=idx)
    fx_ret = pd.Series([0.01], index=idx)
    rf_local = pd.Series([0.003], index=idx)

    out = _translate(factors_usd, fx_ret, rf_local)

    expected = (1.0 + 0.02 + 0.004) * (1.0 + 0.01) - 1.0 - 0.003
    assert out["MKT_RF"].iloc[0] == pytest.approx(expected, abs=1e-15)


def test_long_short_factor_uses_cross_term_only():
    """SMB=0.01, fx=0.05 -> out = 0.0105 exactly, NOT the long-only 0.0605."""
    idx = _idx(1)
    factors_usd = pd.DataFrame(
        {"MKT_RF": [0.0], "RF": [0.0], "SMB": [0.01]}, index=idx
    )
    fx_ret = pd.Series([0.05], index=idx)
    rf_local = pd.Series([0.0], index=idx)

    out = _translate(factors_usd, fx_ret, rf_local)

    expected = 0.01 * (1.0 + 0.05)
    assert out["SMB"].iloc[0] == pytest.approx(expected, abs=1e-15)
    assert out["SMB"].iloc[0] == pytest.approx(0.0105, abs=1e-12)

    wrong_long_only = (1.0 + 0.01) * (1.0 + 0.05) - 1.0
    assert out["SMB"].iloc[0] != pytest.approx(wrong_long_only, abs=1e-9)
    assert wrong_long_only == pytest.approx(0.0605, abs=1e-12)


def test_zero_fx_and_matching_rf_is_a_no_op():
    """fx=0 and rf_local == RF_usd -> output equals input for all columns."""
    idx = _idx(1)
    rf_usd = 0.004
    factors_usd = pd.DataFrame(
        {
            "MKT_RF": [0.02],
            "RF": [rf_usd],
            "SMB": [0.01],
            "HML": [-0.005],
            "RMW": [0.003],
            "CMA": [0.002],
            "MOM": [0.015],
        },
        index=idx,
    )
    fx_ret = pd.Series([0.0], index=idx)
    rf_local = pd.Series([rf_usd], index=idx)

    out = _translate(factors_usd, fx_ret, rf_local)

    for col in ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]:
        assert out[col].iloc[0] == pytest.approx(factors_usd[col].iloc[0], abs=1e-15)
    assert out["RF"].iloc[0] == pytest.approx(rf_local.iloc[0], abs=1e-15)

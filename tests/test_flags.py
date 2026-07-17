"""Tests for fundlens.analysis.flags.evaluate_flags."""
from __future__ import annotations

import pandas as pd
import pytest

from fundlens.analysis.factor_model import FactorFit
from fundlens.analysis.flags import THRESHOLDS, evaluate_flags


def _fit(model="ff5_mom", alpha_ann=0.0, alpha_t=0.0, alpha_p=None, betas=None, beta_t=None):
    betas = betas or {"MKT_RF": 1.0}
    beta_t = beta_t or {"MKT_RF": 5.0}
    return FactorFit(
        model=model,
        alpha_ann=alpha_ann,
        alpha_t=alpha_t,
        alpha_p_bootstrap=alpha_p,
        betas=betas,
        beta_t=beta_t,
        r2=0.8,
        resid_vol_ann=0.05,
        nobs=60,
        start=pd.Timestamp("2018-01-31"),
        end=pd.Timestamp("2023-12-31"),
    )


def _ids(flags):
    return {f.id for f in flags}


def test_missing_inputs_only_yields_no_crash():
    flags = evaluate_flags({})
    assert flags == []


def test_alpha_verdict_significant():
    ctx = {"factor_fits": {"ff5_mom": _fit(alpha_t=2.5, alpha_p=0.01, alpha_ann=0.05)}}
    flags = evaluate_flags(ctx)
    verdict = next(f for f in flags if f.id == "alpha_verdict")
    assert verdict.severity == "green"


def test_alpha_verdict_suggestive():
    ctx = {"factor_fits": {"ff5_mom": _fit(alpha_t=1.5, alpha_p=0.2, alpha_ann=0.03)}}
    flags = evaluate_flags(ctx)
    verdict = next(f for f in flags if f.id == "alpha_verdict")
    assert verdict.severity == "info"
    assert "suggestive" in verdict.title.lower() or "inconclusive" in verdict.title.lower()


def test_alpha_verdict_none():
    ctx = {"factor_fits": {"ff5_mom": _fit(alpha_t=0.5, alpha_ann=0.01)}}
    flags = evaluate_flags(ctx)
    verdict = next(f for f in flags if f.id == "alpha_verdict")
    assert verdict.severity == "info"


def test_alpha_verdict_negative():
    ctx = {"factor_fits": {"ff5_mom": _fit(alpha_t=-2.0, alpha_ann=-0.04)}}
    flags = evaluate_flags(ctx)
    verdict = next(f for f in flags if f.id == "alpha_verdict")
    assert verdict.severity == "amber"


def test_closet_indexer_fires_red_when_all_three_conditions_hold():
    ctx = {
        "perf": {"tracking_error_ann": 0.02},
        "meta": {"ongoing_charge": 0.008},
        "holdings_stats": {
            "active_share": 0.50,
            "concentration": {"coverage": 0.95},
        },
    }
    flags = evaluate_flags(ctx)
    flag = next((f for f in flags if f.id == "closet_indexer"), None)
    assert flag is not None
    assert flag.severity == "red"


def test_closet_indexer_amber_with_fee_and_one_other_condition():
    ctx = {
        "perf": {"tracking_error_ann": 0.02},
        "meta": {"ongoing_charge": 0.008},
        "holdings_stats": {
            "active_share": 0.90,  # not low -> only the fee + TE conditions hold
            "concentration": {"coverage": 0.95},
        },
    }
    flags = evaluate_flags(ctx)
    flag = next((f for f in flags if f.id == "closet_indexer"), None)
    assert flag is not None
    assert flag.severity == "amber"


def test_closet_indexer_skips_active_share_when_coverage_low():
    ctx = {
        "perf": {"tracking_error_ann": 0.02},
        "meta": {"ongoing_charge": 0.008},
        "holdings_stats": {
            "active_share": 0.10,
            "concentration": {"coverage": 0.5},  # below 0.8 -> skip active-share condition
        },
    }
    flags = evaluate_flags(ctx)
    flag = next((f for f in flags if f.id == "closet_indexer"), None)
    assert flag is not None
    assert flag.severity == "amber"
    assert "coverage" in flag.detail.lower()


def test_closet_indexer_does_not_fire_with_missing_inputs():
    ctx = {"perf": {}, "meta": {}}
    flags = evaluate_flags(ctx)
    assert "closet_indexer" not in _ids(flags)


def test_closet_indexer_requires_fee_condition():
    # Low active share and low tracking error alone (no active fee) must not
    # fire closet_indexer -- charging a passive-level fee for passive
    # positioning is not closet indexing, it's just a good tracker.
    ctx = {
        "perf": {"tracking_error_ann": 0.0157},
        "meta": {"ongoing_charge": 0.0006},
        "holdings_stats": {
            "active_share": 0.10,
            "concentration": {"coverage": 0.95},
        },
    }
    flags = evaluate_flags(ctx)
    assert "closet_indexer" not in _ids(flags)


def test_cheap_tracker_fires_and_not_closet_indexer():
    # The Vanguard-style scenario: low fee + low active share + low(ish)
    # tracking error should read as a good cheap tracker (green), not amber
    # closet indexing.
    ctx = {
        "perf": {"tracking_error_ann": 0.0157},
        "meta": {"ongoing_charge": 0.0006},
        "holdings_stats": {
            "active_share": 0.10,
            "concentration": {"coverage": 0.95},
        },
    }
    flags = evaluate_flags(ctx)
    ids = _ids(flags)
    assert "closet_indexer" not in ids
    tracker_flag = next(f for f in flags if f.id == "cheap_tracker_ok")
    assert tracker_flag.severity == "green"


def test_factor_explained_fires():
    ctx = {
        "factor_fits": {
            "capm": _fit(model="capm", alpha_t=2.5),
            "ff5_mom": _fit(model="ff5_mom", alpha_t=0.3),
        }
    }
    flags = evaluate_flags(ctx)
    assert "factor_explained" in _ids(flags)


def test_factor_explained_absent_when_only_one_fit_present():
    ctx = {"factor_fits": {"capm": _fit(model="capm", alpha_t=2.5)}}
    flags = evaluate_flags(ctx)
    assert "factor_explained" not in _ids(flags)


def test_style_drift_via_score():
    ctx = {"style_drift": THRESHOLDS["style_drift_score_max"] + 0.01}
    flags = evaluate_flags(ctx)
    assert "style_drift" in _ids(flags)


def test_style_drift_via_rolling_beta_shift():
    dates = pd.date_range("2022-01-31", periods=12, freq="ME")
    rolling = pd.DataFrame({"MKT_RF": [1.5] * 12, "SMB": [0.0] * 12, "HML": [0.0] * 12}, index=dates)
    ctx = {
        "rolling_betas": rolling,
        "factor_fits": {"ff3": _fit(model="ff3", betas={"MKT_RF": 1.0, "SMB": 0.0, "HML": 0.0})},
    }
    flags = evaluate_flags(ctx)
    assert "style_drift" in _ids(flags)


def test_style_drift_not_fired_when_within_bounds():
    ctx = {"style_drift": THRESHOLDS["style_drift_score_max"] - 0.01}
    flags = evaluate_flags(ctx)
    assert "style_drift" not in _ids(flags)


def test_concentration_amber_on_low_effective_n():
    ctx = {"holdings_stats": {"concentration": {"top10_weight": 0.3, "effective_n": 10.0}}}
    flags = evaluate_flags(ctx)
    flag = next(f for f in flags if f.id == "concentration")
    assert flag.severity == "amber"


def test_concentration_info_on_high_top10():
    ctx = {"holdings_stats": {"concentration": {"top10_weight": 0.5, "effective_n": 30.0}}}
    flags = evaluate_flags(ctx)
    flag = next(f for f in flags if f.id == "concentration")
    assert flag.severity == "info"


def test_capture_asymmetry_fires():
    ctx = {"perf": {"up_capture": 0.9, "down_capture": 1.0}}
    flags = evaluate_flags(ctx)
    assert "capture_asymmetry" in _ids(flags)


def test_capture_asymmetry_not_fired_within_gap():
    ctx = {"perf": {"up_capture": 0.98, "down_capture": 1.0}}
    flags = evaluate_flags(ctx)
    assert "capture_asymmetry" not in _ids(flags)


def test_expensive_beta_fires():
    ctx = {"perf": {"tracking_error_ann": 0.02}, "meta": {"ongoing_charge": 0.015}}
    flags = evaluate_flags(ctx)
    assert "expensive_beta" in _ids(flags)


def test_tenure_mismatch_fires():
    ctx = {"meta": {"manager_tenure_years": 2.0}, "provenance": {"n_obs": 120}}
    flags = evaluate_flags(ctx)
    assert "tenure_mismatch" in _ids(flags)


def test_holdings_coverage_info():
    ctx = {"holdings_stats": {"concentration": {"coverage": 0.5}}}
    flags = evaluate_flags(ctx)
    flag = next(f for f in flags if f.id == "holdings_coverage")
    assert flag.severity == "info"


def test_cheap_tracker_ok_fires():
    ctx = {"perf": {"tracking_error_ann": 0.005}, "meta": {"ongoing_charge": 0.001}}
    flags = evaluate_flags(ctx)
    flag = next(f for f in flags if f.id == "cheap_tracker_ok")
    assert flag.severity == "green"

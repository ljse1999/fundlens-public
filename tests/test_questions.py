"""Tests for fundlens.analysis.questions.questions_for."""
from __future__ import annotations

import re

from fundlens.analysis.flags import Flag
from fundlens.analysis.questions import questions_for

_PLACEHOLDER_RE = re.compile(r"\{[a-zA-Z_]+\}")


def _flag(id_, severity, metrics, title="Title", detail="Detail"):
    return Flag(id=id_, severity=severity, title=title, detail=detail, metrics=metrics)


def test_green_flags_yield_no_questions():
    flags = [_flag("cheap_tracker_ok", "green", {"ongoing_charge": 0.001, "tracking_error_ann": 0.005})]
    assert questions_for(flags) == []


def test_alpha_verdict_suggestive_yields_question():
    flags = [
        _flag(
            "alpha_verdict",
            "info",
            {"alpha_ann": 0.03, "alpha_t": 1.5, "alpha_p_bootstrap": None},
            title="Suggestive but inconclusive FF5+MOM alpha",
        )
    ]
    qs = questions_for(flags)
    assert len(qs) == 1
    assert qs[0]["flag_id"] == "alpha_verdict"
    assert "FF5+MOM alpha" in qs[0]["question"]
    assert not _PLACEHOLDER_RE.search(qs[0]["question"])


def test_alpha_verdict_none_info_yields_no_question():
    flags = [
        _flag(
            "alpha_verdict",
            "info",
            {"alpha_ann": 0.01, "alpha_t": 0.2},
            title="No detectable FF5+MOM alpha",
        )
    ]
    assert questions_for(flags) == []


def _assert_questions_ok(flags):
    qs = questions_for(flags)
    assert 1 <= len(qs) <= 3
    for q in qs:
        assert q["flag_id"] == flags[0].id
        assert "topic" in q and q["topic"]
        assert not _PLACEHOLDER_RE.search(q["question"])


def test_closet_indexer_questions():
    flags = [
        _flag(
            "closet_indexer",
            "red",
            {"active_share": 0.5, "tracking_error_ann": 0.02, "ongoing_charge": 0.008, "coverage": 0.95},
        )
    ]
    _assert_questions_ok(flags)


def test_factor_explained_questions():
    flags = [_flag("factor_explained", "amber", {"capm_alpha_t": 2.5, "ff5_mom_alpha_t": 0.3})]
    _assert_questions_ok(flags)


def test_style_drift_questions():
    flags = [_flag("style_drift", "amber", {"style_drift_score": 0.08, "shifted_betas": {"MKT_RF": 0.4}})]
    _assert_questions_ok(flags)


def test_concentration_questions():
    flags = [_flag("concentration", "amber", {"top10_weight": 0.5, "effective_n": 10.0})]
    _assert_questions_ok(flags)


def test_capture_asymmetry_questions():
    flags = [_flag("capture_asymmetry", "amber", {"up_capture": 0.9, "down_capture": 1.0})]
    _assert_questions_ok(flags)


def test_expensive_beta_questions():
    flags = [_flag("expensive_beta", "amber", {"ongoing_charge": 0.015, "tracking_error_ann": 0.02})]
    _assert_questions_ok(flags)


def test_tenure_mismatch_info_severity_yields_no_question():
    # tenure_mismatch always fires as "info" severity; per spec only
    # red/amber flags (plus the special-cased alpha_verdict "suggestive")
    # generate questions.
    flags = [_flag("tenure_mismatch", "info", {"manager_tenure_years": 2.0, "track_record_years": 10.0})]
    assert questions_for(flags) == []


def test_holdings_coverage_info_severity_yields_no_question():
    flags = [_flag("holdings_coverage", "info", {"coverage": 0.5})]
    assert questions_for(flags) == []

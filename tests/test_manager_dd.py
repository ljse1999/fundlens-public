"""Tests for local Codex-enhanced manager DD agenda generation."""
from __future__ import annotations

import json
import subprocess

import pytest

from fundlens.analysis import manager_dd
from fundlens.analysis.flags import Flag


def _valid_agenda() -> dict:
    return {
        "summary": "Probe whether the active fee is buying stock-selection skill.",
        "anomalies": [
            {
                "priority": "high",
                "framework_section": "Portfolio construction",
                "title": "Possible closet indexing",
                "evidence": "Active share is low and fees are active.",
                "why_it_matters": "The client may be overpaying for benchmark exposure.",
            }
        ],
        "priority_questions": [
            {
                "priority": "high",
                "topic": "Active management versus fees",
                "question": "Which active weights generated stock-specific value?",
                "evidence_used": ["closet_indexer"],
                "follow_up_if_evasive": "Show stock-level attribution net of factor tilts.",
                "evidence_request": "Provide active weights and attribution by security.",
            }
        ],
        "evidence_requests": [
            {
                "priority": "high",
                "request": "Provide full active weights and stock attribution.",
                "linked_topic": "Active management versus fees",
            }
        ],
        "data_gaps": ["Full holdings history unavailable."],
        "tripwires": [
            {
                "metric": "Active share floor",
                "proposed_level": "Below 60% for two consecutive quarters.",
                "breach_action": "Manager call plus drift review.",
                "rationale": "Low active risk undermines active fee justification.",
            }
        ],
    }


def _result_with_flags() -> dict:
    return {
        "meta": {
            "isin": "GB00TEST1234",
            "name": "Test Fund",
            "currency": "GBP",
            "category": "Global Equity",
            "ongoing_charge": 0.009,
            "manager_tenure_years": 2.0,
        },
        "perf": {"tracking_error_ann": 0.02, "information_ratio": 0.2},
        "holdings_stats": {
            "active_share": 0.5,
            "concentration": {"top10_weight": 0.5, "effective_n": 10.0, "coverage": 0.8},
        },
        "factor_fits": {},
        "flags": [
            {
                "id": "closet_indexer",
                "severity": "red",
                "title": "Possible closet indexing",
                "detail": "Low active risk.",
                "metrics": {
                    "active_share": 0.5,
                    "tracking_error_ann": 0.02,
                    "ongoing_charge": 0.009,
                    "coverage": 0.95,
                },
            },
            {
                "id": "factor_explained",
                "severity": "amber",
                "title": "Outperformance explained by static factor tilts",
                "detail": "Factor tilts explain it.",
                "metrics": {"capm_alpha_t": 2.5, "ff5_mom_alpha_t": 0.3},
            },
            {
                "id": "style_drift",
                "severity": "amber",
                "title": "Meaningful style drift",
                "detail": "Style shifted.",
                "metrics": {"style_drift_score": 0.2, "shifted_betas": {"MKT_RF": 0.4}},
            },
            {
                "id": "concentration",
                "severity": "amber",
                "title": "High portfolio concentration",
                "detail": "Concentrated.",
                "metrics": {"top10_weight": 0.5, "effective_n": 10.0},
            },
            {
                "id": "expensive_beta",
                "severity": "amber",
                "title": "Expensive for the beta on offer",
                "detail": "Fees high.",
                "metrics": {"ongoing_charge": 0.015, "tracking_error_ann": 0.02},
            },
            {
                "id": "tenure_mismatch",
                "severity": "info",
                "title": "Track record predates current manager",
                "detail": "Tenure mismatch.",
                "metrics": {"manager_tenure_years": 2.0, "track_record_years": 10.0},
            },
            {
                "id": "holdings_coverage",
                "severity": "info",
                "title": "Partial holdings disclosure",
                "detail": "Partial disclosure.",
                "metrics": {"coverage": 0.5},
            },
        ],
        "questions": [
            {
                "flag_id": "closet_indexer",
                "topic": "Active management versus fees",
                "question": "What justifies active fees?",
            }
        ],
        "provenance": {"date_range": ["2020-01-31", "2024-12-31"], "n_obs": 60},
        "errors": {},
    }


def test_evidence_cards_cover_framework_flags():
    cards = manager_dd.build_evidence_cards(_result_with_flags())
    by_id = {card["flag_id"]: card for card in cards}

    assert by_id["closet_indexer"]["framework_section"] == "Portfolio construction"
    assert by_id["factor_explained"]["framework_section"] == "Performance evidence and equity attribution"
    assert by_id["style_drift"]["framework_section"] == "Risk, style drift, and drawdown management"
    assert by_id["concentration"]["topic"] == "Position sizing and liquidity"
    assert by_id["expensive_beta"]["framework_section"] == "People, alignment, integrity, and fees"
    assert by_id["tenure_mismatch"]["framework_section"] == "Track record integrity"
    assert by_id["holdings_coverage"]["topic"] == "Data completeness"


def test_codex_backend_success(monkeypatch):
    def fake_run(args, **kwargs):
        assert args[:2] == ["codex", "exec"]
        assert "--ephemeral" in args
        assert "--sandbox" in args and "read-only" in args
        assert "--output-schema" in args
        assert "-C" in args
        assert "evidence_cards" in kwargs["input"]
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps(_valid_agenda()), stderr="")

    monkeypatch.setattr(manager_dd.subprocess, "run", fake_run)

    agenda, metadata = manager_dd.enhance_manager_dd_agenda(_result_with_flags(), timeout_seconds=5)

    assert agenda == _valid_agenda()
    assert metadata["status"] == "ok"
    assert metadata["fallback_used"] is False
    assert metadata["timeout_seconds"] == 5


@pytest.mark.parametrize(
    ("runner", "error_text"),
    [
        (lambda args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing")), "not found"),
        (
            lambda args, **kwargs: (_ for _ in ()).throw(
                subprocess.TimeoutExpired(cmd=args, timeout=kwargs["timeout"])
            ),
            "timed out",
        ),
        (
            lambda args, **kwargs: subprocess.CompletedProcess(args, 1, stdout="", stderr="auth failed"),
            "auth failed",
        ),
        (
            lambda args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="not json", stderr=""),
            "invalid JSON",
        ),
        (
            lambda args, **kwargs: subprocess.CompletedProcess(
                args, 0, stdout=json.dumps({"summary": "missing fields"}), stderr=""
            ),
            "missing required",
        ),
    ],
)
def test_codex_backend_falls_back(monkeypatch, runner, error_text):
    monkeypatch.setattr(manager_dd.subprocess, "run", runner)

    agenda, metadata = manager_dd.enhance_manager_dd_agenda(_result_with_flags(), timeout_seconds=5)

    assert agenda is None
    assert metadata["status"] == "fallback"
    assert metadata["fallback_used"] is True
    assert error_text in metadata["error"]


def test_apply_codex_dd_agenda_records_fallback_error(monkeypatch):
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="not logged in")

    result = _result_with_flags()
    monkeypatch.setattr(manager_dd.subprocess, "run", fake_run)

    manager_dd.apply_codex_dd_agenda(result, timeout_seconds=5)

    assert "dd_agenda" not in result
    assert result["question_generation"]["status"] == "fallback"
    assert result["errors"]["questions_codex"] == "not logged in"

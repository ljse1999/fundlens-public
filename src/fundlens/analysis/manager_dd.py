"""Manager due-diligence agenda generation.

This module keeps the evidence trail deterministic and optional Codex usage
strictly local. The rule engine and metric calculations remain the source of
truth; Codex is only asked to turn compact evidence cards into a sharper
meeting agenda.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from fundlens.analysis.flags import Flag
from fundlens.config import get_settings

DEFAULT_CODEX_TIMEOUT_SECS = 180


DD_FRAMEWORK_SECTIONS: dict[str, str] = {
    "Mandate fit and equity role": (
        "Define the job, benchmark, peer universe, desired exposures, and role "
        "in the client portfolio before accepting the manager's framing."
    ),
    "Philosophy and stock-selection edge": (
        "Test whether the claimed inefficiency is specific, persistent, "
        "falsifiable, and visible in holdings and outcomes."
    ),
    "Research process and decision quality": (
        "Force examples of buys, sells, rejected ideas, mistakes, challenge, "
        "and post-mortems rather than accepting process diagrams."
    ),
    "Portfolio construction": (
        "Assess how insight becomes active weights, concentration, turnover, "
        "and active risk."
    ),
    "Performance evidence and equity attribution": (
        "Separate stock selection from factor, style, sector, country, "
        "currency, peer, fee, and regime effects."
    ),
    "Track record integrity": (
        "Verify the record was earned by this team, in this strategy, at "
        "comparable AUM, and over a meaningful period."
    ),
    "Risk, style drift, and drawdown management": (
        "Separate intentional active risk from hidden, accidental, crowded, or "
        "poorly governed risk."
    ),
    "Liquidity, trading, and capacity": (
        "Test whether current and future AUM can be implemented without "
        "eroding alpha through liquidity or transaction costs."
    ),
    "People, alignment, integrity, and fees": (
        "Evaluate decision rights, succession, incentives, co-investment, "
        "integrity, operational basics, terms, and net client value."
    ),
    "Monitoring, tripwires, and kill criteria": (
        "Convert underwriting into numeric tripwires and thesis-break triggers "
        "before capital is allocated."
    ),
}


FLAG_FRAMEWORK_MAP: dict[str, dict[str, str]] = {
    "alpha_verdict": {
        "framework_section": "Performance evidence and equity attribution",
        "topic": "Alpha durability",
        "why_it_matters": (
            "The central underwriting question is whether any apparent alpha "
            "survives factor, style, benchmark, peer, and fee adjustment."
        ),
        "suggested_challenge": (
            "Ask the manager to connect net excess return to specific "
            "stock-level decisions and to state what remains after neutralising "
            "common exposures."
        ),
        "required_evidence": (
            "Rolling alpha, stock/sector/country attribution, top contributors "
            "and detractors, and examples tied to the stated philosophy."
        ),
    },
    "closet_indexer": {
        "framework_section": "Portfolio construction",
        "topic": "Active management versus fees",
        "why_it_matters": (
            "Active fees are hard to justify if active share and tracking error "
            "show little intentional departure from the benchmark."
        ),
        "suggested_challenge": (
            "Ask what the client would lose if the exposure were replaced by a "
            "cheaper tracker plus any explicit factor tilt."
        ),
        "required_evidence": (
            "Active weights, active share history, tracking error budget, "
            "security attribution, and fee comparison versus a credible passive "
            "alternative."
        ),
    },
    "factor_explained": {
        "framework_section": "Performance evidence and equity attribution",
        "topic": "Selection skill versus factor tilts",
        "why_it_matters": (
            "Outperformance that disappears after factor controls may be cheap "
            "style beta rather than stock-selection skill."
        ),
        "suggested_challenge": (
            "Ask which specific stock decisions prove skill beyond the same "
            "static factor exposures."
        ),
        "required_evidence": (
            "Factor-neutral attribution, active weights, sector/country "
            "attribution, and examples where stock selection drove return."
        ),
    },
    "style_drift": {
        "framework_section": "Risk, style drift, and drawdown management",
        "topic": "Style consistency",
        "why_it_matters": (
            "Unexplained style drift can mean the fund is no longer delivering "
            "the exposure the client hired it to provide."
        ),
        "suggested_challenge": (
            "Ask whether the drift was deliberate, how it was approved, and "
            "what governance would force a reversal or memo update."
        ),
        "required_evidence": (
            "Rolling style/factor exposures, dated portfolio changes, risk "
            "committee notes, and stated style bands."
        ),
    },
    "concentration": {
        "framework_section": "Portfolio construction",
        "topic": "Position sizing and liquidity",
        "why_it_matters": (
            "Concentration can express edge, but only if position size is "
            "linked to upside, downside, liquidity, and correlation."
        ),
        "suggested_challenge": (
            "Ask how the largest positions would be trimmed or exited in a "
            "stress period and which risk budget allowed them to reach size."
        ),
        "required_evidence": (
            "Position-sizing rules, liquidity buckets, active weights, top "
            "holding thesis summaries, and stress exit analysis."
        ),
    },
    "capture_asymmetry": {
        "framework_section": "Risk, style drift, and drawdown management",
        "topic": "Downside risk management",
        "why_it_matters": (
            "A fund that captures more downside than upside needs a clear "
            "explanation of what risk controls are supposed to protect clients."
        ),
        "suggested_challenge": (
            "Ask which controls failed in the worst drawdown and what changed "
            "after the post-mortem."
        ),
        "required_evidence": (
            "Drawdown attribution, cash/hedging/sizing history, risk overrides, "
            "and dated decision logs from stress periods."
        ),
    },
    "expensive_beta": {
        "framework_section": "People, alignment, integrity, and fees",
        "topic": "Fee justification",
        "why_it_matters": (
            "Fees and transaction costs consume a large share of plausible "
            "alpha when active risk is low."
        ),
        "suggested_challenge": (
            "Ask what net alpha is needed to beat the cheapest credible "
            "alternative and who retains the value created."
        ),
        "required_evidence": (
            "OCF/TER, transaction costs, clean share-class availability, "
            "securities lending economics, and fee comparison."
        ),
    },
    "tenure_mismatch": {
        "framework_section": "Track record integrity",
        "topic": "Manager track record",
        "why_it_matters": (
            "A record that predates the current manager may not be attributable "
            "to the team now asking for capital."
        ),
        "suggested_challenge": (
            "Ask which years were earned by the current decision-makers and how "
            "prior-firm or predecessor records are verified."
        ),
        "required_evidence": (
            "Strategy AUM by year, team history, composite report, portability "
            "documentation, and PM decision rights through the record."
        ),
    },
    "holdings_coverage": {
        "framework_section": "Portfolio construction",
        "topic": "Data completeness",
        "why_it_matters": (
            "Partial holdings disclosure can understate active share, tilts, "
            "concentration, and crowding."
        ),
        "suggested_challenge": (
            "Ask for fuller holdings disclosure or enough attribution evidence "
            "to triangulate the missing portfolio."
        ),
        "required_evidence": (
            "Full holdings, active weights, sector/country/currency exposures, "
            "and top buys/sells."
        ),
    },
    "cheap_tracker_ok": {
        "framework_section": "People, alignment, integrity, and fees",
        "topic": "Low-cost passive delivery",
        "why_it_matters": (
            "Low active risk can be acceptable when the product is priced and "
            "sold as a tracker."
        ),
        "suggested_challenge": (
            "Confirm that the product role is passive exposure rather than "
            "stock-selection alpha."
        ),
        "required_evidence": (
            "Fee schedule, tracking difference, replication method, securities "
            "lending policy, and benchmark fit."
        ),
    },
}


AGENDA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "summary",
        "anomalies",
        "priority_questions",
        "evidence_requests",
        "data_gaps",
        "tripwires",
    ],
    "properties": {
        "summary": {"type": "string"},
        "anomalies": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "priority",
                    "framework_section",
                    "title",
                    "evidence",
                    "why_it_matters",
                ],
                "properties": {
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    "framework_section": {"type": "string"},
                    "title": {"type": "string"},
                    "evidence": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                },
            },
        },
        "priority_questions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "priority",
                    "topic",
                    "question",
                    "evidence_used",
                    "follow_up_if_evasive",
                    "evidence_request",
                ],
                "properties": {
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    "topic": {"type": "string"},
                    "question": {"type": "string"},
                    "evidence_used": {"type": "array", "items": {"type": "string"}},
                    "follow_up_if_evasive": {"type": "string"},
                    "evidence_request": {"type": "string"},
                },
            },
        },
        "evidence_requests": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["priority", "request", "linked_topic"],
                "properties": {
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    "request": {"type": "string"},
                    "linked_topic": {"type": "string"},
                },
            },
        },
        "data_gaps": {"type": "array", "items": {"type": "string"}},
        "tripwires": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["metric", "proposed_level", "breach_action", "rationale"],
                "properties": {
                    "metric": {"type": "string"},
                    "proposed_level": {"type": "string"},
                    "breach_action": {"type": "string"},
                    "rationale": {"type": "string"},
                },
            },
        },
    },
}


def _fmt_pct(x: Any, dp: int = 1) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x) * 100:.{dp}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_num(x: Any, dp: int = 2) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{dp}f}"
    except (TypeError, ValueError):
        return "n/a"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if pd.isna(value):
            return None
        return value
    if isinstance(value, pd.DataFrame):
        return _jsonable(value.head(20).reset_index().to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _jsonable(value.head(20).reset_index().to_dict(orient="records"))
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    return str(value)


def _flag_dict(flag: Flag | dict) -> dict:
    if isinstance(flag, Flag):
        return asdict(flag)
    return dict(flag)


def _factor_summary(fit: Any) -> dict | None:
    if fit is None:
        return None
    return {
        "model": getattr(fit, "model", None),
        "alpha_ann": _jsonable(getattr(fit, "alpha_ann", None)),
        "alpha_t": _jsonable(getattr(fit, "alpha_t", None)),
        "alpha_p_bootstrap": _jsonable(getattr(fit, "alpha_p_bootstrap", None)),
        "r2": _jsonable(getattr(fit, "r2", None)),
        "nobs": _jsonable(getattr(fit, "nobs", None)),
        "start": _jsonable(getattr(fit, "start", None)),
        "end": _jsonable(getattr(fit, "end", None)),
        "betas": _jsonable(getattr(fit, "betas", None)),
    }


def _summarise_factor_fits(result: dict) -> dict:
    fits = result.get("factor_fits") or {}
    return {name: _factor_summary(fit) for name, fit in fits.items()}


def _evidence_for_flag(flag: dict) -> str:
    m = flag.get("metrics") or {}
    flag_id = flag.get("id")
    title = flag.get("title") or flag_id or "Flag"
    if flag_id == "alpha_verdict":
        p = m.get("alpha_p_bootstrap")
        p_text = f", bootstrap p={_fmt_num(p, 3)}" if p is not None else ""
        return (
            f"{title}: FF5+MOM alpha {_fmt_pct(m.get('alpha_ann'))} "
            f"annualised, t={_fmt_num(m.get('alpha_t'))}{p_text}, "
            f"n={m.get('nobs', 'n/a')}."
        )
    if flag_id == "closet_indexer":
        return (
            f"Active share {_fmt_pct(m.get('active_share'))}, tracking error "
            f"{_fmt_pct(m.get('tracking_error_ann'))}, OCF "
            f"{_fmt_pct(m.get('ongoing_charge'), 2)}, holdings coverage "
            f"{_fmt_pct(m.get('coverage'))}."
        )
    if flag_id == "factor_explained":
        return (
            f"CAPM alpha t-stat {_fmt_num(m.get('capm_alpha_t'))}, but "
            f"FF5+MOM alpha t-stat {_fmt_num(m.get('ff5_mom_alpha_t'))}."
        )
    if flag_id == "style_drift":
        shifted = m.get("shifted_betas") or {}
        shifted_text = ", ".join(
            f"{factor} shift {_fmt_num(delta)}" for factor, delta in shifted.items()
        )
        if not shifted_text:
            shifted_text = "no named shifted beta supplied"
        return (
            f"Style drift score {_fmt_num(m.get('style_drift_score'), 3)}; "
            f"{shifted_text}."
        )
    if flag_id == "concentration":
        return (
            f"Top-10 weight {_fmt_pct(m.get('top10_weight'))}; effective "
            f"number of holdings {_fmt_num(m.get('effective_n'), 1)}."
        )
    if flag_id == "capture_asymmetry":
        return (
            f"Up-capture {_fmt_num(m.get('up_capture'))}; down-capture "
            f"{_fmt_num(m.get('down_capture'))}."
        )
    if flag_id == "expensive_beta":
        return (
            f"OCF {_fmt_pct(m.get('ongoing_charge'), 2)} against tracking "
            f"error {_fmt_pct(m.get('tracking_error_ann'))}."
        )
    if flag_id == "tenure_mismatch":
        return (
            f"Current manager tenure {_fmt_num(m.get('manager_tenure_years'), 1)} "
            f"years versus analysed track record "
            f"{_fmt_num(m.get('track_record_years'), 1)} years."
        )
    if flag_id == "holdings_coverage":
        return f"Published holdings cover {_fmt_pct(m.get('coverage'))} of the portfolio."
    if flag_id == "cheap_tracker_ok":
        return (
            f"OCF {_fmt_pct(m.get('ongoing_charge'), 2)} and tracking error "
            f"{_fmt_pct(m.get('tracking_error_ann'))}."
        )
    detail = flag.get("detail")
    return str(detail or title)


def build_evidence_cards(result: dict) -> list[dict]:
    """Build framework-mapped evidence cards from fired flags."""
    cards: list[dict] = []
    for raw_flag in result.get("flags") or []:
        flag = _flag_dict(raw_flag)
        flag_id = flag.get("id")
        mapping = FLAG_FRAMEWORK_MAP.get(flag_id, {})
        cards.append(
            {
                "flag_id": flag_id,
                "severity": flag.get("severity"),
                "title": flag.get("title"),
                "framework_section": mapping.get(
                    "framework_section", "Monitoring, tripwires, and kill criteria"
                ),
                "topic": mapping.get("topic", flag.get("title") or "General"),
                "evidence": _evidence_for_flag(flag),
                "why_it_matters": mapping.get(
                    "why_it_matters",
                    "This flag should be reconciled before relying on the manager story.",
                ),
                "suggested_challenge": mapping.get(
                    "suggested_challenge",
                    "Ask the manager to reconcile the evidence with the stated process.",
                ),
                "required_evidence": mapping.get(
                    "required_evidence",
                    "Ask for supporting data, examples, and dated decision evidence.",
                ),
                "metrics": _jsonable(flag.get("metrics") or {}),
            }
        )
    return cards


def infer_data_gaps(result: dict) -> list[str]:
    """Infer data gaps that should become manager follow-up requests."""
    gaps: list[str] = []
    errors = result.get("errors") or {}
    for stage, message in errors.items():
        if stage == "questions_codex":
            continue
        gaps.append(f"{stage}: {message}")

    holdings_stats = result.get("holdings_stats") or {}
    concentration = holdings_stats.get("concentration") or {}
    coverage = concentration.get("coverage")
    if coverage is not None and coverage < 0.9:
        gaps.append(
            f"Holdings disclosure covers only {_fmt_pct(coverage)} of the portfolio."
        )
    if holdings_stats.get("active_share") is None:
        gaps.append("Active share could not be measured from available holdings.")

    meta = result.get("meta") or {}
    if meta.get("manager_tenure_years") is None:
        gaps.append("Current manager tenure was not available.")
    if meta.get("ongoing_charge") is None:
        gaps.append("Ongoing charge figure was not available.")

    if not result.get("factor_fits"):
        gaps.append("Factor-model fits were unavailable.")

    seen = set()
    deduped = []
    for gap in gaps:
        if gap not in seen:
            deduped.append(gap)
            seen.add(gap)
    return deduped


def tripwire_candidates(result: dict) -> list[dict]:
    """Build deterministic tripwire candidates from the DD framework."""
    holdings_stats = result.get("holdings_stats") or {}
    concentration = holdings_stats.get("concentration") or {}
    active_share = holdings_stats.get("active_share")
    top10 = concentration.get("top10_weight")
    style_drift = result.get("style_drift")
    perf = result.get("perf") or {}

    return [
        {
            "metric": "Active share floor",
            "current_value": _fmt_pct(active_share) if active_share is not None else "n/a",
            "default": "Below 60% for two consecutive quarters for diversified active equity.",
            "breach_action": "Manager call plus active-risk and closet-indexing review.",
        },
        {
            "metric": "Rolling 3-year net information ratio",
            "current_value": _fmt_num(perf.get("information_ratio")),
            "default": "Below 0 versus the agreed benchmark.",
            "breach_action": "Formal review of attribution, fees, and peer alternatives.",
        },
        {
            "metric": "Style regression drift",
            "current_value": _fmt_num(style_drift, 3),
            "default": "Stated style loadings outside agreed band for two consecutive quarters.",
            "breach_action": "Drift review and memo update.",
        },
        {
            "metric": "Concentration",
            "current_value": _fmt_pct(top10) if top10 is not None else "n/a",
            "default": "Top-10 weight outside the manager's stated band.",
            "breach_action": "Escalate position-sizing and liquidity review.",
        },
        {
            "metric": "Strategy AUM and capacity",
            "current_value": "manager-supplied",
            "default": "Strategy AUM above the manager's own stated capacity.",
            "breach_action": "Reduce/redeem review unless capacity evidence is refreshed.",
        },
        {
            "metric": "Named key people",
            "current_value": "n/a",
            "default": "Departure of named decision-makers or unresolved succession change.",
            "breach_action": "Automatic freeze plus re-underwrite.",
        },
    ]


def build_manager_dd_context(result: dict) -> dict:
    """Build compact JSON context for Codex agenda generation."""
    meta = result.get("meta") or {}
    provenance = result.get("provenance") or {}
    return {
        "fund": {
            "name": meta.get("name"),
            "isin": meta.get("isin"),
            "currency": meta.get("currency"),
            "category": meta.get("category"),
            "benchmark_stated": meta.get("benchmark_name"),
            "benchmark_proxy": provenance.get("benchmark_proxy"),
            "ongoing_charge": _jsonable(meta.get("ongoing_charge")),
            "manager_tenure_years": _jsonable(meta.get("manager_tenure_years")),
            "analysis_window": provenance.get("date_range"),
            "monthly_observations": provenance.get("n_obs"),
        },
        "performance": _jsonable(result.get("perf") or {}),
        "factor_fits": _summarise_factor_fits(result),
        "holdings_stats": _jsonable(result.get("holdings_stats") or {}),
        "flags": [_flag_dict(flag) for flag in (result.get("flags") or [])],
        "deterministic_questions": _jsonable(result.get("questions") or []),
        "evidence_cards": build_evidence_cards(result),
        "data_gaps": infer_data_gaps(result),
        "tripwire_candidates": tripwire_candidates(result),
        "framework_sections": DD_FRAMEWORK_SECTIONS,
        "instructions": [
            "Use only the supplied evidence; do not invent fund facts.",
            "Write manager-facing due-diligence questions suitable for a professional fund research meeting.",
            "Force examples, dates, holdings, attribution, decision logs, and data requests.",
            "Prioritise anomalies and evidence gaps over generic process questions.",
            "Where evidence is weak, say what evidence is needed rather than implying a conclusion.",
        ],
    }


def _prompt() -> str:
    return (
        "You are helping FundLens generate a local manager due-diligence agenda. "
        "Read the JSON context supplied on stdin. Return only JSON matching the "
        "provided output schema. Use the evidence cards and framework sections "
        "to produce a concise agenda with anomalies, priority manager questions, "
        "evidence requests, data gaps, and monitoring tripwires. Do not invent "
        "facts beyond the context."
    )


def _metadata(
    status: str,
    fallback_used: bool,
    error: str | None = None,
    timeout_seconds: int = DEFAULT_CODEX_TIMEOUT_SECS,
) -> dict:
    meta = {
        "mode": "codex",
        "source": "codex_exec",
        "status": status,
        "fallback_used": fallback_used,
        "timeout_seconds": timeout_seconds,
    }
    if error:
        meta["error"] = error
    model = os.getenv("FUNDLENS_CODEX_MODEL")
    if model:
        meta["model"] = model
    return meta


def _assert_keys(obj: dict, allowed: set[str], where: str) -> None:
    extra = set(obj) - allowed
    missing = allowed - set(obj)
    if extra:
        raise ValueError(f"{where} has unexpected keys: {sorted(extra)}")
    if missing:
        raise ValueError(f"{where} is missing required keys: {sorted(missing)}")


def _assert_str(value: Any, where: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a string")


def _assert_priority(value: Any, where: str) -> None:
    if value not in {"high", "medium", "low"}:
        raise ValueError(f"{where} must be high, medium, or low")


def _validate_string_list(values: Any, where: str) -> None:
    if not isinstance(values, list):
        raise ValueError(f"{where} must be a list")
    for index, item in enumerate(values):
        if not isinstance(item, str):
            raise ValueError(f"{where}[{index}] must be a string")


def validate_dd_agenda(agenda: Any) -> dict:
    """Validate Codex agenda output against the strict local schema."""
    if not isinstance(agenda, dict):
        raise ValueError("agenda must be an object")

    top_keys = {
        "summary",
        "anomalies",
        "priority_questions",
        "evidence_requests",
        "data_gaps",
        "tripwires",
    }
    _assert_keys(agenda, top_keys, "agenda")
    _assert_str(agenda["summary"], "summary")
    _validate_string_list(agenda["data_gaps"], "data_gaps")

    anomaly_keys = {"priority", "framework_section", "title", "evidence", "why_it_matters"}
    if not isinstance(agenda["anomalies"], list):
        raise ValueError("anomalies must be a list")
    for index, item in enumerate(agenda["anomalies"]):
        if not isinstance(item, dict):
            raise ValueError(f"anomalies[{index}] must be an object")
        _assert_keys(item, anomaly_keys, f"anomalies[{index}]")
        _assert_priority(item["priority"], f"anomalies[{index}].priority")
        for key in anomaly_keys - {"priority"}:
            _assert_str(item[key], f"anomalies[{index}].{key}")

    question_keys = {
        "priority",
        "topic",
        "question",
        "evidence_used",
        "follow_up_if_evasive",
        "evidence_request",
    }
    if not isinstance(agenda["priority_questions"], list):
        raise ValueError("priority_questions must be a list")
    for index, item in enumerate(agenda["priority_questions"]):
        if not isinstance(item, dict):
            raise ValueError(f"priority_questions[{index}] must be an object")
        _assert_keys(item, question_keys, f"priority_questions[{index}]")
        _assert_priority(item["priority"], f"priority_questions[{index}].priority")
        _validate_string_list(item["evidence_used"], f"priority_questions[{index}].evidence_used")
        for key in question_keys - {"priority", "evidence_used"}:
            _assert_str(item[key], f"priority_questions[{index}].{key}")

    request_keys = {"priority", "request", "linked_topic"}
    if not isinstance(agenda["evidence_requests"], list):
        raise ValueError("evidence_requests must be a list")
    for index, item in enumerate(agenda["evidence_requests"]):
        if not isinstance(item, dict):
            raise ValueError(f"evidence_requests[{index}] must be an object")
        _assert_keys(item, request_keys, f"evidence_requests[{index}]")
        _assert_priority(item["priority"], f"evidence_requests[{index}].priority")
        _assert_str(item["request"], f"evidence_requests[{index}].request")
        _assert_str(item["linked_topic"], f"evidence_requests[{index}].linked_topic")

    tripwire_keys = {"metric", "proposed_level", "breach_action", "rationale"}
    if not isinstance(agenda["tripwires"], list):
        raise ValueError("tripwires must be a list")
    for index, item in enumerate(agenda["tripwires"]):
        if not isinstance(item, dict):
            raise ValueError(f"tripwires[{index}] must be an object")
        _assert_keys(item, tripwire_keys, f"tripwires[{index}]")
        for key in tripwire_keys:
            _assert_str(item[key], f"tripwires[{index}].{key}")

    return agenda


def _run_codex(context: dict, timeout_seconds: int) -> dict:
    settings = get_settings()
    model = os.getenv("FUNDLENS_CODEX_MODEL")
    with tempfile.TemporaryDirectory(prefix="fundlens_codex_") as temp_dir:
        schema_path = Path(temp_dir) / "dd_agenda.schema.json"
        schema_path.write_text(json.dumps(AGENDA_SCHEMA), encoding="utf-8")
        args = [
            "codex",
            "exec",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--output-schema",
            str(schema_path),
            "-C",
            str(settings.project_root),
        ]
        if model:
            args.extend(["--model", model])
        args.append(_prompt())

        completed = subprocess.run(
            args,
            input=json.dumps(context, default=_jsonable),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            encoding="utf-8",
        )

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or f"codex exited with code {completed.returncode}"
        raise RuntimeError(detail)

    output = (completed.stdout or "").strip()
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"codex returned invalid JSON: {exc}") from exc
    return validate_dd_agenda(parsed)


def enhance_manager_dd_agenda(
    result: dict,
    timeout_seconds: int = DEFAULT_CODEX_TIMEOUT_SECS,
) -> tuple[dict | None, dict]:
    """Try to generate an enhanced manager DD agenda using local Codex."""
    try:
        agenda = _run_codex(build_manager_dd_context(result), timeout_seconds)
    except FileNotFoundError as exc:
        return None, _metadata(
            "fallback",
            True,
            f"codex executable not found: {exc}",
            timeout_seconds=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return None, _metadata(
            "fallback",
            True,
            f"codex timed out after {timeout_seconds} seconds",
            timeout_seconds=timeout_seconds,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return None, _metadata("fallback", True, str(exc), timeout_seconds=timeout_seconds)

    meta = _metadata("ok", False, timeout_seconds=timeout_seconds)
    return agenda, meta


def apply_codex_dd_agenda(
    result: dict,
    timeout_seconds: int = DEFAULT_CODEX_TIMEOUT_SECS,
) -> dict:
    """Mutate ``result`` with a Codex DD agenda or deterministic fallback metadata."""
    agenda, metadata = enhance_manager_dd_agenda(result, timeout_seconds=timeout_seconds)
    result["question_generation"] = metadata
    if agenda is not None:
        result["dd_agenda"] = agenda
        return result

    result.pop("dd_agenda", None)
    errors = result.setdefault("errors", {})
    if metadata.get("error"):
        errors["questions_codex"] = metadata["error"]
    return result


def deterministic_question_generation_metadata() -> dict:
    """Metadata for the default deterministic question path."""
    return {
        "mode": "deterministic",
        "source": "rules",
        "status": "ok",
        "fallback_used": False,
    }

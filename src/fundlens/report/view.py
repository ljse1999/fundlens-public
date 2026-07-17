"""Presentation view models shared by HTML and Streamlit report surfaces."""
from __future__ import annotations

import pandas as pd

from fundlens.report.labels import factor_display_value

_SEVERITY_ORDER = {"red": 0, "amber": 1, "green": 2, "info": 3}


def pct(x, dp: int = 1) -> str:
    """Format a decimal return/weight as a percentage string."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    try:
        return f"{x * 100:.{dp}f}%"
    except (TypeError, ValueError):
        return "n/a"


def num(x, dp: int = 2) -> str:
    """Format a scalar numeric value."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    try:
        return f"{x:.{dp}f}"
    except (TypeError, ValueError):
        return "n/a"


def perf_table(data: dict) -> list[dict]:
    """Return display-ready performance summary rows."""
    perf = data.get("perf")
    if not perf:
        return []
    rows = [
        ("CAGR", "cagr", pct),
        ("Volatility (ann.)", "vol_ann", pct),
        ("Sharpe", "sharpe", num),
        ("Sortino", "sortino", num),
        ("Max drawdown", "max_drawdown", pct),
        ("Tracking error (ann.)", "tracking_error_ann", pct),
        ("Information ratio", "information_ratio", num),
        ("Up capture", "up_capture", num),
        ("Down capture", "down_capture", num),
        ("Hit rate", "hit_rate", pct),
    ]
    out = []
    for label, key, fmt in rows:
        if key in perf:
            out.append({"metric": label, "value": fmt(perf.get(key))})
    return out


def flag_cards(data: dict) -> list[dict]:
    """Return flags sorted for display."""
    flags = data.get("flags") or []
    return sorted(flags, key=lambda f: _SEVERITY_ORDER.get(f.get("severity"), 9))


def flag_counts(flags: list[dict]) -> dict:
    """Count flags by severity."""
    counts = {"red": 0, "amber": 0, "green": 0, "info": 0}
    for flag in flags:
        severity = flag.get("severity")
        if severity in counts:
            counts[severity] += 1
    return counts


def questions_by_topic(data: dict) -> dict:
    """Group manager questions by topic."""
    grouped: dict = {}
    for question in data.get("questions") or []:
        display_question = factor_display_value(question)
        grouped.setdefault(display_question.get("topic", "General"), []).append(
            display_question
        )
    return grouped


def alpha_ladder_view(data: dict) -> dict:
    """Return display-ready alpha ladder rows and metadata."""
    ladder = data.get("alpha_ladder") or {}
    steps = ladder.get("steps") or {}
    if not steps:
        steps = _factor_steps_from_legacy_result(data)

    rows = []
    for step_id in ("capm", "ff3", "ff5", "ff5_mom", "benchmark"):
        step = steps.get(step_id)
        if not step:
            continue
        rows.append(
            {
                "step": _step_get(step, "label") or step_id,
                "alpha_ann": pct(_step_get(step, "alpha_ann")),
                "alpha_t": num(_step_get(step, "alpha_t")),
                "alpha_p": num(_step_get(step, "alpha_p_bootstrap"), 3),
                "r2": num(_step_get(step, "r2")),
                "resid_vol_ann": pct(_step_get(step, "resid_vol_ann")),
                "nobs": _step_get(step, "nobs") or "n/a",
                "window": _window(_step_get(step, "start"), _step_get(step, "end")),
            }
        )

    return {
        "rows": rows,
        "verdict": ladder.get("verdict"),
        "warnings": ladder.get("warnings") or [],
        "selected_proxies": ladder.get("selected_proxies") or [],
    }


def _factor_steps_from_legacy_result(data: dict) -> dict:
    labels = {
        "capm": "CAPM alpha",
        "ff3": "FF3 alpha",
        "ff5": "FF5 alpha",
        "ff5_mom": "FF5+MOM alpha",
    }
    steps = {}
    for model, fit in (data.get("factor_fits") or {}).items():
        steps[model] = {
            "label": labels.get(model, f"{model} alpha"),
            "alpha_ann": getattr(fit, "alpha_ann", None),
            "alpha_t": getattr(fit, "alpha_t", None),
            "alpha_p_bootstrap": getattr(fit, "alpha_p_bootstrap", None),
            "r2": getattr(fit, "r2", None),
            "resid_vol_ann": getattr(fit, "resid_vol_ann", None),
            "nobs": getattr(fit, "nobs", None),
            "start": getattr(fit, "start", None),
            "end": getattr(fit, "end", None),
        }
    return steps


def _step_get(step: object, key: str):
    if isinstance(step, dict):
        return step.get(key)
    return getattr(step, key, None)


def _window(start, end) -> str:
    if start is None or end is None:
        return "n/a"
    try:
        return f"{pd.Timestamp(start).date()} to {pd.Timestamp(end).date()}"
    except (TypeError, ValueError):
        return f"{start} to {end}"


def build_report_view(data: dict) -> dict:
    """Build display-ready report context independent of output channel."""
    meta = data.get("meta") or {}
    provenance = data.get("provenance") or {}
    errors = data.get("errors") or {}

    flags = factor_display_value(flag_cards(data))
    verdict = next((f for f in flags if f.get("id") == "alpha_verdict"), None)

    header = {
        "name": meta.get("name") or "(unknown fund)",
        "isin": meta.get("isin") or "n/a",
        "currency": meta.get("currency") or "n/a",
        "category": meta.get("category") or "n/a",
        "benchmark_stated": meta.get("benchmark_name") or "n/a",
        "benchmark_proxy": provenance.get("benchmark_proxy") or "n/a",
        "ocf": pct(meta.get("ongoing_charge"), 2),
        "manager_tenure": (
            f"{meta.get('manager_tenure_years'):.1f} yrs"
            if meta.get("manager_tenure_years") is not None
            else "n/a"
        ),
        "window": (
            f"{provenance.get('date_range', ['n/a', 'n/a'])[0]} "
            f"to {provenance.get('date_range', ['n/a', 'n/a'])[1]}"
        ),
    }

    return {
        "header": header,
        "verdict": verdict,
        "flag_counts": flag_counts(flags),
        "flags": flags,
        "perf_table": perf_table(data),
        "alpha_ladder": alpha_ladder_view(data),
        "dd_agenda": factor_display_value(data.get("dd_agenda")),
        "questions_by_topic": questions_by_topic(data),
        "question_generation": data.get("question_generation") or {},
        "provenance": provenance,
        "errors": errors,
    }

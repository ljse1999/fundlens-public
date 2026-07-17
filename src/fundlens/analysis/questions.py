"""Generate manager-facing due-diligence questions from raised flags.

Each red/amber flag (plus the "suggestive alpha" case of ``alpha_verdict``)
yields one to three sharp, specific questions an analyst could put directly
to the fund manager, with the underlying metrics interpolated from
``Flag.metrics`` so the questions read as concrete evidence rather than
generic prompts.
"""
from __future__ import annotations

from fundlens.analysis.flags import Flag


def _fmt_pct(x, dp: int = 1) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{x * 100:.{dp}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_num(x, dp: int = 2) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{x:.{dp}f}"
    except (TypeError, ValueError):
        return "n/a"


def _questions_alpha_verdict(flag: Flag) -> list[dict]:
    m = flag.metrics
    if flag.severity != "info" or "suggestive" not in flag.title.lower():
        return []
    return [
        {
            "flag_id": flag.id,
            "topic": "Alpha durability",
            "question": (
                f"The FF5+MOM alpha of {_fmt_pct(m.get('alpha_ann'))} "
                f"annualised (t={_fmt_num(m.get('alpha_t'))}) is positive but not "
                f"yet statistically robust. What would you point to as the "
                f"repeatable process driver of this, independent of the sample "
                f"period examined?"
            ),
        }
    ]


def _questions_closet_indexer(flag: Flag) -> list[dict]:
    m = flag.metrics
    as_text = _fmt_pct(m.get("active_share")) if m.get("active_share") is not None else "not measurable from disclosed holdings"
    return [
        {
            "flag_id": flag.id,
            "topic": "Active management vs. fees",
            "question": (
                f"Active share is {as_text} and tracking error is "
                f"{_fmt_pct(m.get('tracking_error_ann'))}, yet the OCF is "
                f"{_fmt_pct(m.get('ongoing_charge'), 2)}. What in the process "
                f"justifies active fees for positioning this close to the benchmark?"
            ),
        },
        {
            "flag_id": flag.id,
            "topic": "Active management vs. fees",
            "question": (
                "If the mandate were run at half the current fee as a lower-cost "
                "index tracker, what would clients actually give up?"
            ),
        },
    ]


def _questions_factor_explained(flag: Flag) -> list[dict]:
    m = flag.metrics
    return [
        {
            "flag_id": flag.id,
            "topic": "Selection skill vs. factor tilts",
            "question": (
                f"CAPM alpha looks strong (t={_fmt_num(m.get('capm_alpha_t'))}) but "
                f"collapses once style factors are controlled for "
                f"(FF5+MOM t={_fmt_num(m.get('ff5_mom_alpha_t'))}). "
                f"What stock-level decisions do you point to as evidence of "
                f"selection skill beyond these tilts?"
            ),
        },
        {
            "flag_id": flag.id,
            "topic": "Selection skill vs. factor tilts",
            "question": (
                "Could this excess return have been replicated more cheaply with "
                "a static tilt to the same factors, and if not, why not?"
            ),
        },
    ]


def _questions_style_drift(flag: Flag) -> list[dict]:
    m = flag.metrics
    shifted = m.get("shifted_betas") or {}
    factor_txt = ", ".join(shifted.keys()) if shifted else "the fund's style loadings"
    return [
        {
            "flag_id": flag.id,
            "topic": "Style consistency",
            "question": (
                f"Recent factor exposure on {factor_txt} has moved materially away "
                f"from the full-period average. Was this a deliberate repositioning "
                f"call, or drift from letting winners run/losers slide within the "
                f"portfolio?"
            ),
        },
        {
            "flag_id": flag.id,
            "topic": "Style consistency",
            "question": (
                "How do you define and enforce the style mandate this fund is sold "
                "against, and at what point would a shift like this trigger a "
                "process review?"
            ),
        },
    ]


def _questions_concentration(flag: Flag) -> list[dict]:
    m = flag.metrics
    return [
        {
            "flag_id": flag.id,
            "topic": "Position sizing and liquidity",
            "question": (
                f"With top-10 weight at {_fmt_pct(m.get('top10_weight'))} and an "
                f"effective holding count of {_fmt_num(m.get('effective_n'), 1)}, "
                f"what position-sizing and liquidity discipline governs how large "
                f"a single name can become, and how would the book be unwound in "
                f"a stress scenario?"
            ),
        },
    ]


def _questions_capture_asymmetry(flag: Flag) -> list[dict]:
    m = flag.metrics
    return [
        {
            "flag_id": flag.id,
            "topic": "Downside risk management",
            "question": (
                f"Down-capture ({_fmt_num(m.get('down_capture'))}) exceeds up-capture "
                f"({_fmt_num(m.get('up_capture'))}) -- what specific risk controls "
                f"(hedging, cash, sizing, sector caps) are meant to protect on the "
                f"downside, and why haven't they shown up in the capture ratios?"
            ),
        },
    ]


def _questions_expensive_beta(flag: Flag) -> list[dict]:
    m = flag.metrics
    return [
        {
            "flag_id": flag.id,
            "topic": "Fee justification",
            "question": (
                f"OCF of {_fmt_pct(m.get('ongoing_charge'), 2)} against tracking "
                f"error of only {_fmt_pct(m.get('tracking_error_ann'))} implies a "
                f"high fee per unit of active risk taken. How is this fee level "
                f"justified to investors?"
            ),
        },
    ]


def _questions_tenure_mismatch(flag: Flag) -> list[dict]:
    m = flag.metrics
    return [
        {
            "flag_id": flag.id,
            "topic": "Manager track record",
            "question": (
                f"The current manager has {_fmt_num(m.get('manager_tenure_years'), 1)} "
                f"years' tenure against a track record spanning "
                f"{_fmt_num(m.get('track_record_years'), 1)} years. How much of the "
                f"historical performance is attributable to the current manager "
                f"versus their predecessor(s)?"
            ),
        },
    ]


def _questions_holdings_coverage(flag: Flag) -> list[dict]:
    m = flag.metrics
    return [
        {
            "flag_id": flag.id,
            "topic": "Data completeness",
            "question": (
                f"Published holdings cover only {_fmt_pct(m.get('coverage'))} of the "
                f"portfolio. Can you provide fuller holdings disclosure so active "
                f"share and tilt estimates aren't understated?"
            ),
        },
    ]


_HANDLERS = {
    "alpha_verdict": _questions_alpha_verdict,
    "closet_indexer": _questions_closet_indexer,
    "factor_explained": _questions_factor_explained,
    "style_drift": _questions_style_drift,
    "concentration": _questions_concentration,
    "capture_asymmetry": _questions_capture_asymmetry,
    "expensive_beta": _questions_expensive_beta,
    "tenure_mismatch": _questions_tenure_mismatch,
    "holdings_coverage": _questions_holdings_coverage,
}

# Flags that never generate questions (positive/neutral signals), except the
# special-cased "suggestive alpha" branch of alpha_verdict handled above.
_NO_QUESTION_SEVERITIES = {"green"}


def questions_for(flags: list[Flag]) -> list[dict]:
    """Generate suggested due-diligence questions for a set of raised flags.

    Args:
        flags: A list of :class:`fundlens.analysis.flags.Flag` instances,
            typically the output of
            :func:`fundlens.analysis.flags.evaluate_flags`.

    Returns:
        A list of dicts, one or more per red/amber flag (plus the
        "suggestive alpha" case of ``alpha_verdict``), each containing
        ``flag_id``, ``topic``, and ``question``.
    """
    out: list[dict] = []
    for flag in flags:
        handler = _HANDLERS.get(flag.id)
        if handler is None:
            continue
        if flag.id == "alpha_verdict":
            out.extend(handler(flag))
            continue
        if flag.severity not in ("red", "amber"):
            continue
        out.extend(handler(flag))
    return out

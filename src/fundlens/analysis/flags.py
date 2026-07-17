"""Rule-based diligence flags derived from analysis outputs.

``evaluate_flags`` consumes the same shape of dict produced by
:func:`fundlens.pipeline.analyse_fund` (built up incrementally by the
pipeline itself, before the ``flags``/``questions`` keys are populated) and
returns the list of :class:`Flag` instances whose conditions are satisfied.
Every rule is defensive about missing inputs: if the data a rule needs is not
present in ``ctx`` (because an earlier pipeline stage failed or the fund
lacks that kind of data), the rule is skipped rather than raising.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Tunable thresholds for the rules below, gathered in one place so they can
# be adjusted without touching rule logic.
THRESHOLDS: dict = {
    "alpha_t_significant": 2.0,
    "alpha_p_significant": 0.05,
    "alpha_t_suggestive": 1.3,
    "alpha_t_negative": -1.3,
    "closet_as_max": 0.60,
    "closet_te_max": 0.03,
    "closet_ocf_min": 0.005,
    "holdings_coverage_min": 0.80,
    "factor_explained_capm_t": 2.0,
    "factor_explained_ff5mom_t": 1.0,
    "style_drift_score_max": 0.12,
    "style_drift_beta_shift_max": 0.30,
    "concentration_top10_max": 0.45,
    "concentration_effective_n_min": 15.0,
    "capture_asymmetry_gap": 0.05,
    "expensive_beta_ocf_min": 0.010,
    "expensive_beta_te_max": 0.04,
    "tenure_mismatch_years_max": 3.0,
    "tenure_mismatch_track_record_min": 5.0,
    "holdings_coverage_report_max": 0.90,
    "cheap_tracker_ocf_max": 0.002,
    "cheap_tracker_te_max": 0.02,
}


@dataclass
class Flag:
    """A single diligence flag raised by :func:`evaluate_flags`.

    Attributes:
        id: Stable machine identifier for the flag rule (e.g.
            "style_drift_high").
        severity: One of "red" (serious concern), "amber" (worth
            watching), "green" (positive signal), or "info" (neutral
            observation).
        title: Short human-readable title.
        detail: Longer human-readable explanation of why the flag fired.
        metrics: Dict of the underlying metric values that triggered the
            flag, for traceability in reports.
    """

    id: str
    severity: Literal["red", "amber", "green", "info"]
    title: str
    detail: str
    metrics: dict


def _fmt_pct(x: float | None, dp: int = 1) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.{dp}f}%"


def _get_fit(ctx: dict, model: str):
    fits = ctx.get("factor_fits") or {}
    return fits.get(model)


def _rule_alpha_verdict(ctx: dict) -> Flag | None:
    fit = _get_fit(ctx, "ff5_mom")
    if fit is None:
        return None
    t = fit.alpha_t
    p = fit.alpha_p_bootstrap
    alpha_ann = fit.alpha_ann
    window = f"{fit.start.date()} to {fit.end.date()} (n={fit.nobs})"
    metrics = {
        "alpha_ann": alpha_ann,
        "alpha_t": t,
        "alpha_p_bootstrap": p,
        "nobs": fit.nobs,
        "start": fit.start,
        "end": fit.end,
    }
    if t < THRESHOLDS["alpha_t_negative"]:
        return Flag(
            id="alpha_verdict",
            severity="amber",
            title="Negative FF5+MOM alpha",
            detail=(
                f"FF5+MOM alpha is {_fmt_pct(alpha_ann)} annualised "
                f"(t={t:.2f}) over {window} -- the fund has meaningfully "
                f"underperformed after controlling for style/factor tilts."
            ),
            metrics=metrics,
        )
    if t > THRESHOLDS["alpha_t_significant"] and p is not None and p < THRESHOLDS["alpha_p_significant"]:
        return Flag(
            id="alpha_verdict",
            severity="green",
            title="Statistically significant FF5+MOM alpha",
            detail=(
                f"FF5+MOM alpha is {_fmt_pct(alpha_ann)} annualised "
                f"(t={t:.2f}, bootstrap p={p:.3f}) over {window} -- this holds up "
                f"as broad-factor risk-adjusted outperformance, not noise."
            ),
            metrics=metrics,
        )
    if t > THRESHOLDS["alpha_t_suggestive"]:
        return Flag(
            id="alpha_verdict",
            severity="info",
            title="Suggestive but inconclusive FF5+MOM alpha",
            detail=(
                f"FF5+MOM alpha is {_fmt_pct(alpha_ann)} annualised "
                f"(t={t:.2f}"
                + (f", bootstrap p={p:.3f}" if p is not None else "")
                + f") over {window} -- positive but not yet statistically robust."
            ),
            metrics=metrics,
        )
    return Flag(
        id="alpha_verdict",
        severity="info",
        title="No detectable FF5+MOM alpha",
        detail=(
            f"FF5+MOM alpha is {_fmt_pct(alpha_ann)} annualised "
            f"(t={t:.2f}) over {window} -- statistically indistinguishable from zero."
        ),
        metrics=metrics,
    )


def _rule_closet_indexer(ctx: dict) -> Flag | None:
    perf = ctx.get("perf") or {}
    meta = ctx.get("meta") or {}
    te = perf.get("tracking_error_ann")
    ocf = meta.get("ongoing_charge")
    if te is None or ocf is None:
        return None

    # Closet indexing is about charging *active* fees for largely passive
    # positioning -- the fee condition is a precondition, not just one of
    # three equally-weighted signals. No fee -> the rule never fires,
    # regardless of how low active share/tracking error are.
    fee_cond = ocf > THRESHOLDS["closet_ocf_min"]
    if not fee_cond:
        return None

    hstats = ctx.get("holdings_stats") or {}
    concentration = hstats.get("concentration") or {}
    coverage = concentration.get("coverage")
    active_share_val = hstats.get("active_share")

    as_available = (
        active_share_val is not None
        and coverage is not None
        and coverage >= THRESHOLDS["holdings_coverage_min"]
    )

    conds: dict[str, bool] = {}
    detail_notes = []
    if as_available:
        conds["active_share"] = active_share_val < THRESHOLDS["closet_as_max"]
    else:
        detail_notes.append(
            "active-share condition skipped (holdings coverage below 80%)"
        )
    conds["tracking_error"] = te < THRESHOLDS["closet_te_max"]

    n_true = sum(conds.values())

    if as_available:
        if n_true == 2:
            severity = "red"
        elif n_true == 1:
            severity = "amber"
        else:
            return None
    else:
        if n_true == 1:
            severity = "amber"
        else:
            return None

    metrics = {
        "active_share": active_share_val,
        "tracking_error_ann": te,
        "ongoing_charge": ocf,
        "coverage": coverage,
    }
    as_text = _fmt_pct(active_share_val) if active_share_val is not None else "n/a"
    detail = (
        f"Active share {as_text}, tracking error {_fmt_pct(te)}, OCF "
        f"{_fmt_pct(ocf, 2)}: this combination is consistent with closet "
        f"indexing (active fees for largely index-like positioning)."
    )
    if detail_notes:
        detail += " (" + "; ".join(detail_notes) + ")"

    return Flag(
        id="closet_indexer",
        severity=severity,
        title="Possible closet indexing",
        detail=detail,
        metrics=metrics,
    )


def _rule_factor_explained(ctx: dict) -> Flag | None:
    capm = _get_fit(ctx, "capm")
    ff5_mom = _get_fit(ctx, "ff5_mom")
    if capm is None or ff5_mom is None:
        return None
    if capm.alpha_t > THRESHOLDS["factor_explained_capm_t"] and ff5_mom.alpha_t < THRESHOLDS["factor_explained_ff5mom_t"]:
        return Flag(
            id="factor_explained",
            severity="amber",
            title="Outperformance explained by static factor tilts",
            detail=(
                f"CAPM alpha looks strong (t={capm.alpha_t:.2f}) but the "
                f"FF5+MOM alpha collapses to t={ff5_mom.alpha_t:.2f} "
                f"once style tilts are controlled for -- outperformance appears "
                f"explained by static factor exposure rather than selection skill."
            ),
            metrics={"capm_alpha_t": capm.alpha_t, "ff5_mom_alpha_t": ff5_mom.alpha_t},
        )
    return None


def _rule_style_drift(ctx: dict) -> Flag | None:
    drift_score = ctx.get("style_drift")
    rolling = ctx.get("rolling_betas")
    ff3_fit = _get_fit(ctx, "ff3")

    fired = False
    detail_bits = []
    metrics: dict = {}

    if drift_score is not None:
        metrics["style_drift_score"] = drift_score
        if drift_score > THRESHOLDS["style_drift_score_max"]:
            fired = True
            detail_bits.append(
                f"RBSA style-weight turnover score is {drift_score:.3f} "
                f"(threshold {THRESHOLDS['style_drift_score_max']:.2f})"
            )

    shifted_factors = []
    if rolling is not None and len(rolling) >= 12 and ff3_fit is not None:
        recent_mean = rolling.tail(12).mean()
        for col in ("MKT_RF", "SMB", "HML"):
            if col in rolling.columns and col in ff3_fit.betas:
                diff = abs(float(recent_mean[col]) - ff3_fit.betas[col])
                if diff > THRESHOLDS["style_drift_beta_shift_max"]:
                    shifted_factors.append((col, diff))
        if shifted_factors:
            fired = True
            metrics["shifted_betas"] = {c: d for c, d in shifted_factors}
            detail_bits.append(
                "rolling 12m beta has shifted materially vs the full-period "
                "beta on: " + ", ".join(f"{c} (Delta={d:.2f})" for c, d in shifted_factors)
            )

    if not fired:
        return None

    return Flag(
        id="style_drift",
        severity="amber",
        title="Meaningful style drift",
        detail="; ".join(detail_bits) + ".",
        metrics=metrics,
    )


def _rule_concentration(ctx: dict) -> Flag | None:
    hstats = ctx.get("holdings_stats") or {}
    concentration = hstats.get("concentration")
    if not concentration:
        return None
    top10 = concentration.get("top10_weight")
    eff_n = concentration.get("effective_n")

    metrics = {"top10_weight": top10, "effective_n": eff_n}

    if eff_n is not None and eff_n < THRESHOLDS["concentration_effective_n_min"]:
        return Flag(
            id="concentration",
            severity="amber",
            title="High portfolio concentration",
            detail=(
                f"Effective number of holdings is {eff_n:.1f} "
                f"(top-10 weight {_fmt_pct(top10) if top10 is not None else 'n/a'}) "
                f"-- concentration risk is significant."
            ),
            metrics=metrics,
        )
    if top10 is not None and top10 > THRESHOLDS["concentration_top10_max"]:
        return Flag(
            id="concentration",
            severity="info",
            title="High conviction positioning",
            detail=(
                f"Top-10 holdings make up {_fmt_pct(top10)} of the portfolio -- "
                f"a high-conviction, concentrated book."
            ),
            metrics=metrics,
        )
    return None


def _rule_capture_asymmetry(ctx: dict) -> Flag | None:
    perf = ctx.get("perf") or {}
    up = perf.get("up_capture")
    down = perf.get("down_capture")
    if up is None or down is None:
        return None
    if down > up + THRESHOLDS["capture_asymmetry_gap"]:
        return Flag(
            id="capture_asymmetry",
            severity="amber",
            title="Asymmetric capture: worse on the downside",
            detail=(
                f"Down-market capture ({down:.2f}) exceeds up-market capture "
                f"({up:.2f}) -- the fund gives back more than it keeps, an "
                f"unfavourable asymmetry for a risk-managed process."
            ),
            metrics={"up_capture": up, "down_capture": down},
        )
    return None


def _rule_expensive_beta(ctx: dict) -> Flag | None:
    perf = ctx.get("perf") or {}
    meta = ctx.get("meta") or {}
    te = perf.get("tracking_error_ann")
    ocf = meta.get("ongoing_charge")
    if te is None or ocf is None:
        return None
    if ocf > THRESHOLDS["expensive_beta_ocf_min"] and te < THRESHOLDS["expensive_beta_te_max"]:
        return Flag(
            id="expensive_beta",
            severity="amber",
            title="Expensive for the beta on offer",
            detail=(
                f"OCF is {_fmt_pct(ocf, 2)} while tracking error is only "
                f"{_fmt_pct(te)} -- the fee looks high relative to how far the "
                f"fund departs from its benchmark."
            ),
            metrics={"ongoing_charge": ocf, "tracking_error_ann": te},
        )
    return None


def _rule_tenure_mismatch(ctx: dict) -> Flag | None:
    meta = ctx.get("meta") or {}
    tenure = meta.get("manager_tenure_years")
    provenance = ctx.get("provenance") or {}
    n_obs = provenance.get("n_obs")
    track_record_years = (n_obs / 12.0) if n_obs else None
    if tenure is None or track_record_years is None:
        return None
    if tenure < THRESHOLDS["tenure_mismatch_years_max"] and track_record_years > THRESHOLDS["tenure_mismatch_track_record_min"]:
        return Flag(
            id="tenure_mismatch",
            severity="info",
            title="Track record predates current manager",
            detail=(
                f"The current manager has {tenure:.1f} years' tenure, but the "
                f"analysed track record spans {track_record_years:.1f} years -- "
                f"much of the historical performance predates them."
            ),
            metrics={"manager_tenure_years": tenure, "track_record_years": track_record_years},
        )
    return None


def _rule_holdings_coverage(ctx: dict) -> Flag | None:
    hstats = ctx.get("holdings_stats") or {}
    concentration = hstats.get("concentration") or {}
    coverage = concentration.get("coverage")
    if coverage is None:
        return None
    if coverage < THRESHOLDS["holdings_coverage_report_max"]:
        return Flag(
            id="holdings_coverage",
            severity="info",
            title="Partial holdings disclosure",
            detail=(
                f"Published holdings cover only {_fmt_pct(coverage)} of the "
                f"portfolio; holdings-based metrics (active share, tilts, "
                f"concentration) are partial and should be read as lower bounds."
            ),
            metrics={"coverage": coverage},
        )
    return None


def _rule_cheap_tracker_ok(ctx: dict) -> Flag | None:
    perf = ctx.get("perf") or {}
    meta = ctx.get("meta") or {}
    te = perf.get("tracking_error_ann")
    ocf = meta.get("ongoing_charge")
    if te is None or ocf is None:
        return None
    if ocf < THRESHOLDS["cheap_tracker_ocf_max"] and te < THRESHOLDS["cheap_tracker_te_max"]:
        return Flag(
            id="cheap_tracker_ok",
            severity="green",
            title="Does what it says on the tin",
            detail=(
                f"OCF {_fmt_pct(ocf, 2)} and tracking error {_fmt_pct(te)} -- "
                f"a cheap, tight tracker delivering what it promises."
            ),
            metrics={"ongoing_charge": ocf, "tracking_error_ann": te},
        )
    return None


_RULES = [
    _rule_alpha_verdict,
    _rule_closet_indexer,
    _rule_factor_explained,
    _rule_style_drift,
    _rule_concentration,
    _rule_capture_asymmetry,
    _rule_expensive_beta,
    _rule_tenure_mismatch,
    _rule_holdings_coverage,
    _rule_cheap_tracker_ok,
]


def evaluate_flags(ctx: dict) -> list[Flag]:
    """Evaluate diligence rules against an analysis context and return the flags that fire.

    Args:
        ctx: A dict aggregating analysis outputs for a single fund, e.g.
            the same shape produced by
            :func:`fundlens.pipeline.analyse_fund` (keys such as ``meta``,
            ``perf``, ``factor_fits``, ``rolling_betas``, ``style_drift``,
            ``holdings_stats``, ``provenance``).

    Returns:
        A list of :class:`Flag` instances, one per rule that fired (order
        not guaranteed; callers should sort by severity if needed). Rules
        whose required inputs are missing from ``ctx`` are silently skipped.
    """
    flags: list[Flag] = []
    for rule in _RULES:
        try:
            flag = rule(ctx)
        except Exception:  # noqa: BLE001 - a single bad rule shouldn't break diligence
            flag = None
        if flag is not None:
            flags.append(flag)
    return flags

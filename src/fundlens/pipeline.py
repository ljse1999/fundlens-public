"""End-to-end analysis pipeline: resolve a fund and run the full analysis suite.

``analyse_fund`` orchestrates the data-fetch and analysis modules into a
single result dict. Every stage past fund resolution and return-series
fetch is wrapped so that one failure does not take down the whole run:
failures are recorded in ``result["errors"][stage] = message`` and
downstream stages/consumers (flags, the HTML report) check for presence of
their inputs rather than assuming they exist.

Fund resolution and the return-series fetch are the two prerequisite
stages: nothing else can proceed without them, so failures there are
recorded in ``errors`` *and* re-raised, matching the CLI's contract
(exit 1 only when resolve or returns fail; exit 0 -- with a possibly
partial report -- for any other failure).
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Literal

import pandas as pd

from fundlens.analysis.alpha_ladder import (
    alpha_ladder_to_dict,
    build_alpha_ladder,
    fit_benchmark_residual_alpha,
)
from fundlens.analysis.attribution import factor_contributions
from fundlens.analysis.factor_model import fit_factor_model, rolling_betas, subperiod_alphas
from fundlens.analysis.flags import evaluate_flags
from fundlens.analysis.holdings_analytics import active_share, concentration, tilts
from fundlens.analysis.manager_dd import apply_codex_dd_agenda, deterministic_question_generation_metadata
from fundlens.analysis.questions import questions_for
from fundlens.analysis.returns import drawdown_series, perf_summary, rolling_excess
from fundlens.analysis.style import rbsa, style_drift_score
from fundlens.data.benchmarks import (
    benchmark_proxy_for,
    benchmark_proxy_match_for,
    get_benchmark_returns,
    get_style_proxies,
)
from fundlens.data.factors import get_factors, region_for_category
from fundlens.data.fundamentals import enrich_holdings
from fundlens.data.fx import convert_factor_returns, get_risk_free
from fundlens.data.holdings import get_etf_holdings, get_fund_holdings
from fundlens.data.navs import get_returns
from fundlens.data.resolver import resolve_fund

_MIN_MONTHLY_OBS = 24
_FACTOR_MODELS = ("capm", "ff3", "ff5", "ff5_mom")


def analyse_fund(
    isin: str,
    benchmark_override: str | None = None,
    factor_region_override: str | None = None,
    question_mode: Literal["deterministic", "codex"] = "deterministic",
) -> dict:
    """Run the full fundlens analysis pipeline for a single fund.

    Resolves the fund, fetches returns/factors/benchmark/holdings data,
    runs the factor, style, holdings, and attribution analyses, evaluates
    diligence flags, and generates follow-up questions.

    Args:
        isin: The fund's ISIN (or a name resolvable via
            :func:`fundlens.data.resolver.resolve_fund`).
        benchmark_override: Optional explicit benchmark ticker to use
            instead of the auto-resolved proxy from
            :func:`fundlens.data.benchmarks.benchmark_proxy_for`.
        factor_region_override: Optional explicit
            :data:`fundlens.data.factors.Region` string to use instead of
            the auto-resolved region from
            :func:`fundlens.data.factors.region_for_category`.
        question_mode: ``"deterministic"`` keeps the existing rule-based
            questions only; ``"codex"`` also tries to build a local
            Codex-enhanced manager due-diligence agenda.

    Returns:
        A result dict; see module docstring for the error-handling
        contract. Keys: ``meta``, ``perf``, ``factor_fits``,
        ``rolling_betas``, ``subperiod``, ``style_weights``,
        ``style_drift``, ``holdings``, ``holdings_stats``,
        ``factor_contrib``, ``flags``, ``questions``, optional
        ``dd_agenda``, ``question_generation``, ``series``, ``provenance``,
        ``errors``.

    Raises:
        Exception: whatever :func:`fundlens.data.resolver.resolve_fund` or
            :func:`fundlens.data.navs.get_returns` raise (or a
            ``RuntimeError`` if there's insufficient monthly history),
            after also recording it in ``errors``. All other stage
            failures are swallowed and recorded only.
    """
    if question_mode not in ("deterministic", "codex"):
        raise ValueError("question_mode must be 'deterministic' or 'codex'")

    result: dict = {"errors": {}}
    errors = result["errors"]
    provenance: dict = {}
    result["provenance"] = provenance

    # -- 1. Resolve (fatal on failure) --------------------------------------
    try:
        fund = resolve_fund(isin)
    except Exception as exc:
        errors["resolve"] = str(exc)
        raise
    result["meta"] = asdict(fund)

    # -- 2. Returns (fatal on failure) ---------------------------------------
    try:
        returns_bundle = get_returns(fund)
        monthly = returns_bundle.monthly
        if monthly is None or len(monthly) < _MIN_MONTHLY_OBS:
            raise RuntimeError(
                f"insufficient monthly return history for {isin!r}: "
                f"{0 if monthly is None else len(monthly)} obs (need >= {_MIN_MONTHLY_OBS})"
            )
    except Exception as exc:
        errors["returns"] = str(exc)
        raise

    provenance["returns_source"] = returns_bundle.provenance
    provenance["currency"] = fund.currency
    provenance["n_obs"] = int(len(monthly))
    provenance["date_range"] = [str(monthly.index.min().date()), str(monthly.index.max().date())]
    provenance["start"] = str(monthly.index.min().date())
    provenance["end"] = str(monthly.index.max().date())
    provenance["benchmark_stated"] = fund.benchmark_name
    provenance["notes"] = [
        "Morningstar may backfill older share-class history.",
        "Active share is a lower bound under partial holdings coverage.",
        "FF5+MOM alpha does not neutralise sector, country, currency, or theme exposures; "
        "benchmark residual alpha only controls for the selected benchmark proxy.",
    ]

    # -- 3. Benchmark proxy + returns -----------------------------------------
    benchmark_ticker = None
    benchmark_returns = None
    benchmark_proxy_source = None
    try:
        benchmark_ticker = benchmark_override or benchmark_proxy_for(fund)
        if not benchmark_ticker:
            label = fund.benchmark_name or fund.category or "unknown benchmark"
            raise LookupError(f"no reliable benchmark proxy mapping for {label!r}")
        if benchmark_override:
            benchmark_proxy_source = "override"
        else:
            match = benchmark_proxy_match_for(fund)
            benchmark_proxy_source = match.source if match else "auto"
        provenance["benchmark_proxy_source"] = benchmark_proxy_source
        benchmark_returns = get_benchmark_returns(benchmark_ticker)
        provenance["benchmark_proxy"] = benchmark_ticker
    except Exception as exc:
        errors["benchmark"] = str(exc)

    # -- 4. Factor region -----------------------------------------------------
    region = factor_region_override or region_for_category(fund.category)
    provenance["factor_region"] = region

    # -- 5. Factors, converted to fund currency -------------------------------
    factors_conv = None
    try:
        factors_usd = get_factors(region, "M")
        factors_conv = convert_factor_returns(factors_usd, fund.currency)
    except Exception as exc:
        errors["factors"] = str(exc)

    # -- 6. Risk-free ----------------------------------------------------------
    rf = None
    try:
        if factors_conv is not None and "RF" in factors_conv.columns:
            rf = factors_conv["RF"]
        else:
            rf = get_risk_free(fund.currency)
    except Exception as exc:
        errors["risk_free"] = str(exc)

    # -- 7. Fund excess returns (monthly - rf, aligned) -----------------------
    fund_excess = None
    try:
        if rf is not None:
            aligned = pd.concat([monthly.rename("r"), rf.rename("rf")], axis=1, join="inner").dropna()
            fund_excess = aligned["r"] - aligned["rf"]
    except Exception as exc:
        errors["fund_excess"] = str(exc)

    # -- 8. Performance summary -------------------------------------------------
    try:
        result["perf"] = perf_summary(monthly, benchmark_returns, rf)
    except Exception as exc:
        errors["perf"] = str(exc)
        result["perf"] = None

    # -- Series for charting ----------------------------------------------------
    series: dict = {"fund_monthly": monthly}
    if benchmark_returns is not None:
        series["benchmark_monthly"] = benchmark_returns
    try:
        series["drawdown"] = drawdown_series(monthly)
    except Exception as exc:
        errors["drawdown"] = str(exc)
    if benchmark_returns is not None:
        try:
            series["rolling_excess_12m"] = rolling_excess(monthly, benchmark_returns, window=12)
        except Exception as exc:
            errors["rolling_excess"] = str(exc)
    result["series"] = series

    # -- 9. Factor model fits ----------------------------------------------------
    factor_fits: dict = {}
    if factors_conv is not None and fund_excess is not None:
        for model in _FACTOR_MODELS:
            try:
                factor_fits[model] = fit_factor_model(fund_excess, factors_conv, model=model)
            except Exception as exc:
                errors[f"factor_fit_{model}"] = str(exc)
    else:
        errors.setdefault(
            "factor_fits",
            "skipped: factors or fund_excess unavailable",
        )
    result["factor_fits"] = factor_fits

    # -- 9b. Alpha ladder (factor rows + benchmark residual alpha) --------------------
    alpha_ladder_warnings: list[str] = []
    selected_alpha_proxies: list[dict] = []
    benchmark_step = None
    if benchmark_ticker:
        selected_alpha_proxies.append(
            {
                "id": "benchmark",
                "ticker": benchmark_ticker,
                "reason": (
                    "benchmark override"
                    if benchmark_proxy_source == "override"
                    else "Morningstar category proxy"
                    if benchmark_proxy_source == "category"
                    else "stated benchmark proxy"
                ),
                "source": "benchmark_map",
            }
        )
    try:
        if benchmark_returns is None:
            raise RuntimeError("benchmark returns unavailable")
        if rf is None:
            raise RuntimeError("risk-free series unavailable")
        benchmark_step = fit_benchmark_residual_alpha(
            monthly,
            benchmark_returns,
            rf,
            benchmark_ticker=benchmark_ticker,
        )
    except Exception as exc:
        message = str(exc)
        alpha_ladder_warnings.append(f"Benchmark residual alpha unavailable: {message}")
        errors["alpha_ladder_benchmark"] = message

    result["alpha_ladder"] = alpha_ladder_to_dict(
        build_alpha_ladder(
            factor_fits,
            benchmark_step=benchmark_step,
            warnings=alpha_ladder_warnings,
            selected_proxies=selected_alpha_proxies,
        )
    )

    # -- 10. Rolling betas (ff3, 36m) ---------------------------------------------
    try:
        if factors_conv is None or fund_excess is None:
            raise RuntimeError("factors or fund_excess unavailable")
        result["rolling_betas"] = rolling_betas(fund_excess, factors_conv, model="ff3", window=36)
    except Exception as exc:
        errors["rolling_betas"] = str(exc)
        result["rolling_betas"] = None

    # -- 11. Subperiod alphas -------------------------------------------------------
    try:
        if factors_conv is None or fund_excess is None:
            raise RuntimeError("factors or fund_excess unavailable")
        result["subperiod"] = subperiod_alphas(fund_excess, factors_conv, model="ff5_mom", n_periods=3)
    except Exception as exc:
        errors["subperiod"] = str(exc)
        result["subperiod"] = None

    # -- 12. Style proxies + RBSA + drift --------------------------------------------
    result["style_weights"] = None
    result["style_drift"] = None
    try:
        proxies = get_style_proxies(region)
        style_weights = rbsa(monthly, proxies, window=36)
        result["style_weights"] = style_weights
        result["style_drift"] = style_drift_score(style_weights)
    except Exception as exc:
        errors["style"] = str(exc)

    # -- 13. Holdings + holdings analytics --------------------------------------------
    fund_holdings = None
    bench_holdings = None
    holdings_stats: dict = {}
    try:
        fund_holdings = get_fund_holdings(fund)
    except Exception as exc:
        errors["holdings"] = str(exc)
    try:
        if benchmark_ticker:
            bench_holdings = get_etf_holdings(benchmark_ticker)
    except Exception as exc:
        errors["bench_holdings"] = str(exc)

    if fund_holdings is not None and len(fund_holdings):
        try:
            holdings_stats["concentration"] = concentration(fund_holdings)
        except Exception as exc:
            errors["concentration"] = str(exc)
        if bench_holdings is not None and len(bench_holdings):
            try:
                holdings_stats["active_share"] = active_share(fund_holdings, bench_holdings)
            except Exception as exc:
                errors["active_share"] = str(exc)
            try:
                holdings_stats["tilts_sector"] = tilts(fund_holdings, bench_holdings, by="sector")
            except Exception as exc:
                errors["tilts_sector"] = str(exc)
            try:
                holdings_stats["tilts_country"] = tilts(fund_holdings, bench_holdings, by="country")
            except Exception as exc:
                errors["tilts_country"] = str(exc)

    result["holdings"] = fund_holdings
    result["holdings_stats"] = holdings_stats

    # -- 14. Factor contribution decomposition (ff5_mom only) -------------------------
    result["factor_contrib"] = None
    try:
        if "ff5_mom" in factor_fits and factors_conv is not None:
            result["factor_contrib"] = factor_contributions(factor_fits["ff5_mom"], factors_conv)
        else:
            raise RuntimeError("ff5_mom fit or factors unavailable")
    except Exception as exc:
        errors["factor_contrib"] = str(exc)

    # -- 15. Best-effort fundamentals enrichment (top holdings) -------------------------
    try:
        if fund_holdings is not None and len(fund_holdings):
            result["holdings_enriched"] = enrich_holdings(fund_holdings.head(25))
    except Exception:  # noqa: BLE001 - optional enrichment, skip silently
        pass

    # -- 16. Diligence flags -----------------------------------------------------------
    try:
        flags = evaluate_flags(result)
    except Exception as exc:
        errors["flags"] = str(exc)
        flags = []
    result["flags"] = [asdict(f) for f in flags]

    # -- 17. Manager questions ----------------------------------------------------------
    try:
        result["questions"] = questions_for(flags)
    except Exception as exc:
        errors["questions"] = str(exc)
        result["questions"] = []
    result["question_generation"] = deterministic_question_generation_metadata()

    # -- 18. Optional local Codex-enhanced manager DD agenda -----------------------------
    if question_mode == "codex":
        apply_codex_dd_agenda(result)

    return result

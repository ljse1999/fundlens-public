"""Shared helpers for batch fund screening."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from fundlens.analysis.alpha_ladder import (
    alpha_ladder_to_dict,
    build_alpha_ladder,
    fit_benchmark_residual_alpha,
)
from fundlens.analysis.factor_model import fit_factor_model
from fundlens.analysis.flags import THRESHOLDS, evaluate_flags
from fundlens.analysis.returns import perf_summary
from fundlens.data.benchmarks import benchmark_proxy_for, benchmark_proxy_match_for, get_benchmark_returns
from fundlens.data.factors import get_factors, region_for_category
from fundlens.data.fx import convert_factor_returns, get_risk_free
from fundlens.data.ia_universe import append_screen_row, load_screen_checkpoint
from fundlens.data.navs import get_returns
from fundlens.data.resolver import resolve_fund

_MIN_MONTHLY_OBS = 24

SCREEN_FIELDNAMES = [
    "isin",
    "name",
    "category",
    "currency",
    "domicile",
    "security_type",
    "cagr",
    "information_ratio",
    "tracking_error_ann",
    "alpha_ann",
    "alpha_t",
    "alpha_p_bootstrap",
    "alpha_verdict",
    "genuine_alpha",
    "n_obs",
    "active_share",
    "ongoing_charge",
    "flags",
    "error",
]

# Screen fields appended when the universe source is an IA workbook import. Kept
# separate so the base SCREEN_FIELDNAMES stays backward-compatible.
IA_SCREEN_FIELDNAMES = SCREEN_FIELDNAMES + [
    "ia_fund_name",
    "ia_management_company",
    "ia_sector",
    "resolution_confidence",
    "resolution_status",
]

ALPHA_LADDER_SCREEN_FIELDNAMES = [
    "benchmark_alpha_ann",
    "benchmark_alpha_t",
    "alpha_ladder_verdict",
]


def parse_isins_text(text: str) -> list[str]:
    """Parse one ISIN/name per line, ignoring blanks and comments."""
    return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def _alpha_verdict(flags: list[dict]) -> dict | None:
    return next((flag for flag in flags if flag.get("id") == "alpha_verdict"), None)


def is_equity_category(category: str | None) -> bool:
    """Return whether a Morningstar category label looks equity-oriented."""
    if not category:
        return False
    lowered = category.lower()
    if "equity" not in lowered:
        return False
    exclusions = ("bond", "fixed income", "money market", "allocation", "convertible", "alternative")
    return not any(term in lowered for term in exclusions)


def row_has_genuine_alpha(
    row: dict,
    *,
    alpha_t_threshold: float | None = None,
    alpha_p_threshold: float | None = None,
) -> bool:
    """Check the screen-row definition of genuine alpha."""
    custom_thresholds = alpha_t_threshold is not None or alpha_p_threshold is not None
    if row.get("genuine_alpha") is True and not custom_thresholds:
        return True
    if alpha_t_threshold is None:
        alpha_t_threshold = THRESHOLDS["alpha_t_significant"]
    if alpha_p_threshold is None:
        alpha_p_threshold = THRESHOLDS["alpha_p_significant"]
    alpha_t = row.get("alpha_t")
    alpha_p = row.get("alpha_p_bootstrap")
    try:
        return (
            alpha_t is not None
            and alpha_p is not None
            and float(alpha_t) > alpha_t_threshold
            and float(alpha_p) < alpha_p_threshold
        )
    except (TypeError, ValueError):
        return False


def result_to_screen_row(isin: str, result: dict) -> dict:
    """Summarise a successful pipeline result into one screening row."""
    meta = result.get("meta") or {}
    perf = result.get("perf") or {}
    fits = result.get("factor_fits") or {}
    ff5_mom = fits.get("ff5_mom")
    holdings_stats = result.get("holdings_stats") or {}
    provenance = result.get("provenance") or {}
    flags = result.get("flags") or []
    verdict = _alpha_verdict(flags)

    flags_fired = ",".join(flag.get("id", "") for flag in flags)
    alpha_p = getattr(ff5_mom, "alpha_p_bootstrap", None) if ff5_mom else None
    row = {
        "isin": meta.get("isin") or isin,
        "name": meta.get("name"),
        "category": meta.get("category"),
        "currency": meta.get("currency"),
        "domicile": meta.get("domicile"),
        "security_type": meta.get("security_type"),
        "cagr": perf.get("cagr"),
        "information_ratio": perf.get("information_ratio"),
        "tracking_error_ann": perf.get("tracking_error_ann"),
        "alpha_ann": getattr(ff5_mom, "alpha_ann", None) if ff5_mom else None,
        "alpha_t": getattr(ff5_mom, "alpha_t", None) if ff5_mom else None,
        "alpha_p_bootstrap": alpha_p,
        "alpha_verdict": verdict.get("title") if verdict else None,
        "genuine_alpha": bool(verdict and verdict.get("severity") == "green"),
        "n_obs": provenance.get("n_obs"),
        "active_share": holdings_stats.get("active_share"),
        "ongoing_charge": meta.get("ongoing_charge"),
        "flags": flags_fired,
        "error": None,
    }

    if not row["genuine_alpha"]:
        row["genuine_alpha"] = row_has_genuine_alpha(row)

    ladder = result.get("alpha_ladder") or {}
    steps = ladder.get("steps") or {}
    benchmark_step = steps.get("benchmark")
    if benchmark_step:
        row["benchmark_alpha_ann"] = _step_get(benchmark_step, "alpha_ann")
        row["benchmark_alpha_t"] = _step_get(benchmark_step, "alpha_t")
        row["alpha_ladder_verdict"] = ladder.get("verdict")
    return row


def error_to_screen_row(isin: str, error: BaseException | str) -> dict:
    """Summarise a failed pipeline run into one screening row."""
    return {
        "isin": isin,
        "name": None,
        "category": None,
        "currency": None,
        "domicile": None,
        "security_type": None,
        "cagr": None,
        "information_ratio": None,
        "tracking_error_ann": None,
        "alpha_ann": None,
        "alpha_t": None,
        "alpha_p_bootstrap": None,
        "alpha_verdict": None,
        "genuine_alpha": False,
        "n_obs": None,
        "active_share": None,
        "ongoing_charge": None,
        "flags": "",
        "error": str(error),
    }


def enrich_screen_row_with_ia(row: dict, ia: dict) -> dict:
    """Return a copy of ``row`` with IA source metadata merged in.

    ``ia`` is a flat dict with at least ``ia_fund_name``, ``ia_management_company``,
    ``ia_sector``, ``resolution_confidence`` and ``resolution_status`` keys.
    Used by the screen-resolved CLI to carry IA provenance through to the screen.
    """
    enriched = dict(row)
    enriched["ia_fund_name"] = ia.get("ia_fund_name")
    enriched["ia_management_company"] = ia.get("ia_management_company")
    enriched["ia_sector"] = ia.get("ia_sector")
    enriched["resolution_confidence"] = ia.get("resolution_confidence")
    enriched["resolution_status"] = ia.get("resolution_status")
    return enriched


def filter_screen_rows(
    rows: Iterable[dict],
    *,
    category: str | None = None,
    equity_only: bool = False,
    genuine_alpha_only: bool = False,
    include_errors: bool = True,
    alpha_t_threshold: float | None = None,
    alpha_p_threshold: float | None = None,
) -> list[dict]:
    """Filter screen rows for category/equity/alpha views."""
    category_norm = category.casefold() if category else None
    out: list[dict] = []
    for row in rows:
        if row.get("error"):
            if include_errors and not genuine_alpha_only:
                out.append(row)
            continue

        row_category = row.get("category")
        if category_norm and category_norm not in str(row_category or "").casefold():
            continue
        if equity_only and not is_equity_category(row_category):
            continue
        if genuine_alpha_only and not row_has_genuine_alpha(
            row,
            alpha_t_threshold=alpha_t_threshold,
            alpha_p_threshold=alpha_p_threshold,
        ):
            continue
        out.append(row)
    return out


def available_categories(rows: Iterable[dict], *, equity_only: bool = False) -> list[str]:
    """Return sorted Morningstar categories available in screen rows."""
    categories = {
        str(row.get("category"))
        for row in rows
        if row.get("category") and (not equity_only or is_equity_category(row.get("category")))
    }
    return sorted(categories)


def rank_screen_rows(
    rows: Iterable[dict],
    *,
    category: str | None = None,
    equity_only: bool = False,
    genuine_alpha_only: bool = False,
    sort_by_category: bool = False,
    include_errors: bool = True,
    alpha_t_threshold: float | None = None,
    alpha_p_threshold: float | None = None,
) -> list[dict]:
    """Rank screen rows by alpha t-stat, with optional category/alpha filters."""
    filtered = filter_screen_rows(
        rows,
        category=category,
        equity_only=equity_only,
        genuine_alpha_only=genuine_alpha_only,
        include_errors=include_errors,
        alpha_t_threshold=alpha_t_threshold,
        alpha_p_threshold=alpha_p_threshold,
    )

    def key(row: dict) -> tuple:
        category_key = str(row.get("category") or "") if sort_by_category else ""
        return (
            category_key,
            row.get("alpha_t") is None,
            -(row.get("alpha_t") or 0),
            str(row.get("name") or row.get("isin") or ""),
        )

    return sorted(filtered, key=key)


def analyse_alpha_screen(
    isin: str,
    *,
    benchmark_override: str | None = None,
    factor_region_override: str | None = None,
    bootstrap_draws: int = 2000,
    include_alpha_ladder: bool = False,
) -> dict:
    """Run the lightest analysis needed for an alpha-generation screen."""
    if bootstrap_draws <= 0:
        raise ValueError("bootstrap_draws must be positive")

    result: dict = {"errors": {}}
    errors = result["errors"]
    provenance: dict = {}
    result["provenance"] = provenance

    try:
        fund = resolve_fund(isin)
    except Exception as exc:
        errors["resolve"] = str(exc)
        raise
    result["meta"] = asdict(fund)

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

    benchmark_returns = None
    benchmark_ticker = None
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

    region = factor_region_override or region_for_category(fund.category)
    provenance["factor_region"] = region

    try:
        factors_usd = get_factors(region, "M")
        factors_conv = convert_factor_returns(factors_usd, fund.currency)
    except Exception as exc:
        errors["factors"] = str(exc)
        raise RuntimeError(f"alpha factors unavailable for {isin!r}: {exc}") from exc

    try:
        if "RF" in factors_conv.columns:
            rf = factors_conv["RF"]
        else:
            rf = get_risk_free(fund.currency)
        aligned = pd.concat([monthly.rename("r"), rf.rename("rf")], axis=1, join="inner").dropna()
        fund_excess = aligned["r"] - aligned["rf"]
    except Exception as exc:
        errors["risk_free"] = str(exc)
        raise RuntimeError(f"risk-free series unavailable for {isin!r}: {exc}") from exc

    try:
        result["perf"] = perf_summary(monthly, benchmark_returns, rf)
    except Exception as exc:
        errors["perf"] = str(exc)
        result["perf"] = None

    try:
        fit = fit_factor_model(
            fund_excess,
            factors_conv,
            model="ff5_mom",
            bootstrap=True,
            bootstrap_draws=bootstrap_draws,
        )
    except Exception as exc:
        errors["factor_fit_ff5_mom"] = str(exc)
        raise RuntimeError(f"alpha model unavailable for {isin!r}: {exc}") from exc

    result["factor_fits"] = {"ff5_mom": fit}
    if include_alpha_ladder:
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
            benchmark_step = fit_benchmark_residual_alpha(
                monthly,
                benchmark_returns,
                rf,
                benchmark_ticker=benchmark_ticker,
                bootstrap=True,
                bootstrap_draws=bootstrap_draws,
            )
        except Exception as exc:
            message = str(exc)
            alpha_ladder_warnings.append(f"Benchmark residual alpha unavailable: {message}")
            errors["alpha_ladder_benchmark"] = message

        result["alpha_ladder"] = alpha_ladder_to_dict(
            build_alpha_ladder(
                result["factor_fits"],
                benchmark_step=benchmark_step,
                warnings=alpha_ladder_warnings,
                selected_proxies=selected_alpha_proxies,
            )
        )
    result["holdings_stats"] = {}
    result["flags"] = [asdict(flag) for flag in evaluate_flags(result)]
    result["questions"] = []
    return result


def _step_get(step: object, key: str):
    if isinstance(step, dict):
        return step.get(key)
    return getattr(step, key, None)


def run_screen(
    isins: Iterable[str],
    analyse: Callable[[str], dict] | None = None,
    on_progress: Callable[[int, str, dict], Any] | None = None,
    *,
    category: str | None = None,
    equity_only: bool = False,
    genuine_alpha_only: bool = False,
    sort_by_category: bool = False,
    include_errors: bool = True,
) -> list[dict]:
    """Run the analysis pipeline across ISINs and return ranked screen rows."""
    if analyse is None:
        from fundlens.pipeline import analyse_fund

        analyse = analyse_fund

    rows: list[dict] = []
    for index, isin in enumerate(isins, start=1):
        try:
            row = result_to_screen_row(isin, analyse(isin))
        except Exception as exc:  # noqa: BLE001 - batch screens keep going
            row = error_to_screen_row(isin, exc)
        rows.append(row)
        if on_progress is not None:
            on_progress(index, isin, row)
    return rank_screen_rows(
        rows,
        category=category,
        equity_only=equity_only,
        genuine_alpha_only=genuine_alpha_only,
        sort_by_category=sort_by_category,
        include_errors=include_errors,
    )


def _coerce_checkpoint_row(row: dict) -> dict:
    """Coerce a CSV-read checkpoint row's numeric/bool fields to native types.

    Checkpoint rows come back as strings; sorting and genuine-alpha detection
    expect floats/bools. Numeric fields become float (or None), booleans become
    bool. Unknown/empty values are left as-is or set to None.
    """
    numeric_fields = {
        "cagr", "information_ratio", "tracking_error_ann", "alpha_ann",
        "alpha_t", "alpha_p_bootstrap", "n_obs", "active_share", "ongoing_charge",
        "resolution_confidence", "benchmark_alpha_ann", "benchmark_alpha_t",
    }
    bool_fields = {"genuine_alpha"}
    out = dict(row)
    for key, value in row.items():
        if key in numeric_fields:
            if value is None or value == "":
                out[key] = None
                continue
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                out[key] = None
        elif key in bool_fields:
            if value is None or value == "":
                out[key] = None
                continue
            out[key] = str(value).strip().lower() in {"true", "1", "yes"}
        elif value == "":
            out[key] = None
    return out


def run_screen_with_checkpoint(
    ia_rows: list[dict],
    *,
    analyse: Callable[[str], dict],
    checkpoint_path: str | Path | None = None,
    checkpoint_fields: list[str],
    enrich: Callable[[dict, dict], dict] | None = None,
    workers: int = 4,
    on_progress: Callable[[int, int, int, dict], Any] | None = None,
) -> list[dict]:
    """Run the alpha screen across IA rows with concurrency + resume.

    Shared driver for the CLI and Streamlit IA screens. Each IA row carries an
    ``isin`` plus source metadata (name/sector/confidence) merged via ``enrich``.

    Args:
        ia_rows: rows to screen; each must have an ``isin`` key.
        analyse: callable mapping ISIN -> result dict (e.g. a partial of
            ``analyse_alpha_screen`` with bootstrap_draws bound).
        checkpoint_path: if given, completed rows are appended here per fund
            (durable) and reloaded on entry so an interrupted run resumes.
        checkpoint_fields: CSV column order for the checkpoint.
        enrich: merges a screen row with its IA source metadata. Defaults to
            no-op enrichment.
        workers: concurrent worker count. 4 is the proven sweet spot; the
            sandbox showed no gain beyond ~4-8 and full result determinism.
        on_progress: called as ``(done_index, total, overall_done, row)`` as
            each fund completes.

    Returns:
        All screen rows in input order. Unranked/unfiltered; callers rank as
        needed.
    """
    if enrich is None:
        def enrich(row: dict, _ia: dict) -> dict:  # noqa: A001 - default no-op
            return row

    input_index_by_isin: dict[str, int] = {}
    for index, ia_row in enumerate(ia_rows):
        isin = ia_row.get("isin")
        if isin and isin not in input_index_by_isin:
            input_index_by_isin[isin] = index

    # Resume: load previously-screened rows and skip their ISINs. Checkpoint rows
    # come back as strings (CSV); coerce numeric/bool fields so they sort and
    # compare correctly alongside freshly-screened native-typed rows.
    rows_by_index: dict[int, dict] = {}
    done_isins: set[str] = set()
    if checkpoint_path is not None:
        raw_rows, _checkpoint_isins = load_screen_checkpoint(checkpoint_path, checkpoint_fields)
        for raw_row in raw_rows:
            row = _coerce_checkpoint_row(raw_row)
            isin = row.get("isin")
            index = input_index_by_isin.get(isin)
            if index is None:
                continue
            rows_by_index[index] = row
            done_isins.add(isin)

    todo = [
        (index, ia_row)
        for index, ia_row in enumerate(ia_rows)
        if ia_row.get("isin") not in done_isins
    ]
    total = len(todo)
    already_done = len(done_isins)

    def _screen_one(ia_row: dict) -> dict:
        isin = ia_row["isin"]
        try:
            row = result_to_screen_row(isin, analyse(isin))
        except Exception as exc:  # noqa: BLE001 - batch screens keep going
            row = error_to_screen_row(isin, exc)
        return enrich(row, ia_row)

    # Concurrent execution, but results are re-keyed by input index so the
    # output order (and therefore the checkpoint) stays stable across runs.
    new_rows_by_todo_index: dict[int, dict] = {}
    if workers > 1 and total > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_index = {
                pool.submit(_screen_one, ia_row): (todo_index, input_index)
                for todo_index, (input_index, ia_row) in enumerate(todo)
            }
            completed = 0
            next_checkpoint_index = 0
            for fut in as_completed(future_to_index):
                todo_index, input_index = future_to_index[fut]
                row = fut.result()  # _screen_one never raises
                rows_by_index[input_index] = row
                new_rows_by_todo_index[todo_index] = row
                while (
                    checkpoint_path is not None
                    and next_checkpoint_index in new_rows_by_todo_index
                ):
                    append_screen_row(
                        checkpoint_path,
                        checkpoint_fields,
                        new_rows_by_todo_index[next_checkpoint_index],
                    )
                    next_checkpoint_index += 1
                completed += 1
                if on_progress is not None:
                    overall_done = already_done + completed
                    on_progress(completed, total, overall_done, row)
    else:
        for todo_index, (input_index, ia_row) in enumerate(todo, start=1):
            row = _screen_one(ia_row)
            rows_by_index[input_index] = row
            if checkpoint_path is not None:
                append_screen_row(checkpoint_path, checkpoint_fields, row)
            if on_progress is not None:
                overall_done = already_done + todo_index
                on_progress(todo_index, total, overall_done, row)

    return [
        rows_by_index[index]
        for index in range(len(ia_rows))
        if index in rows_by_index
    ]

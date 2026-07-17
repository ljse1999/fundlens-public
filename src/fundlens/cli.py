"""fundlens command-line interface."""
from __future__ import annotations

import csv
import datetime as dt
import html
from pathlib import Path
from typing import Literal

import typer

from fundlens.config import get_settings
from fundlens.data.ia_universe import (
    AMBIGUOUS_THRESHOLD,
    MATCHED_THRESHOLD,
    IAResolution,
    load_screen_checkpoint,
    read_resolution_audit,
    read_ia_workbook,
    resolution_counts,
    resolve_ia_workbook,
    write_resolution_audit,
    write_resolution_review,
)
from fundlens.data.universe import candidate_isins, discover_fund_universe
from fundlens.pipeline import analyse_fund
from fundlens.report.builder import build_report
from fundlens.screening import (
    IA_SCREEN_FIELDNAMES,
    SCREEN_FIELDNAMES,
    analyse_alpha_screen,
    enrich_screen_row_with_ia,
    error_to_screen_row,
    parse_isins_text,
    rank_screen_rows,
    result_to_screen_row,
    run_screen_with_checkpoint,
)

app = typer.Typer(help="fundlens: factor decomposition and alpha analysis toolkit.")


def _verdict_summary(result: dict) -> str:
    flags = result.get("flags") or []
    verdict = next((f for f in flags if f.get("id") == "alpha_verdict"), None)
    verdict_text = verdict["title"] if verdict else "no alpha verdict available"
    red = sum(1 for f in flags if f.get("severity") == "red")
    amber = sum(1 for f in flags if f.get("severity") == "amber")
    return f"{verdict_text} -- {red} red flag(s), {amber} amber flag(s)"


@app.command()
def analyse(
    isin: str = typer.Argument(..., help="ISIN or fund name to analyse."),
    benchmark: str = typer.Option(None, "--benchmark", help="Override benchmark ticker."),
    factors: str = typer.Option(None, "--factors", help="Override factor region."),
    questions: Literal["deterministic", "codex"] = typer.Option(
        "deterministic",
        "--questions",
        help="Question generation mode: deterministic or local Codex-enhanced.",
    ),
    out: Path = typer.Option(None, "--out", help="Output path for the report."),
) -> None:
    """Run the full analysis pipeline for a single fund and build a report."""
    typer.echo(f"fundlens: resolving {isin!r}...")
    try:
        result = analyse_fund(
            isin,
            benchmark_override=benchmark,
            factor_region_override=factors,
            question_mode=questions,
        )
    except Exception as exc:  # noqa: BLE001 - resolve/returns failures are fatal
        typer.echo(f"fundlens: analysis failed: {exc}")
        raise typer.Exit(code=1)

    typer.echo("fundlens: fetched returns, factors, benchmark, and holdings (best-effort).")
    errors = result.get("errors") or {}
    for stage, msg in errors.items():
        typer.echo(f"fundlens: warning -- {stage} unavailable: {msg}")

    qmeta = result.get("question_generation") or {}
    if questions == "codex" and qmeta.get("status") == "ok":
        typer.echo("fundlens: evaluated diligence flags and generated Codex-enhanced DD agenda.")
    elif questions == "codex":
        typer.echo("fundlens: Codex agenda unavailable; using deterministic questions.")
    else:
        typer.echo("fundlens: evaluated diligence flags and generated questions.")

    if out is None:
        meta = result.get("meta") or {}
        report_isin = meta.get("isin") or isin
        stamp = dt.date.today().strftime("%Y%m%d")
        out = get_settings().reports_dir / f"{report_isin}_{stamp}.html"

    report_path = build_report(result, out)
    typer.echo(f"fundlens: report written to {report_path}")
    typer.echo(f"fundlens: {_verdict_summary(result)}")
    raise typer.Exit(code=0)


@app.command()
def screen(
    isins_file: Path = typer.Argument(..., help="Path to a file of ISINs, one per line."),
    out: Path = typer.Option(None, "--out", help="Output path for the screen CSV results."),
) -> None:
    """Run the analysis pipeline across a list of funds and summarise results."""
    isins = parse_isins_text(isins_file.read_text(encoding="utf-8"))

    rows: list[dict] = []
    for isin in isins:
        typer.echo(f"fundlens: screening {isin!r}...")
        try:
            result = analyse_fund(isin)
        except Exception as exc:  # noqa: BLE001 - keep screening the rest
            typer.echo(f"fundlens: warning -- {isin!r} failed: {exc}")
            rows.append(error_to_screen_row(isin, exc))
            continue
        rows.append(result_to_screen_row(isin, result))

    if out is None:
        stamp = dt.date.today().strftime("%Y%m%d")
        out = get_settings().reports_dir / f"screen_{stamp}.csv"
    _, html_path = _write_screen_outputs(rows, out, title="fundlens screen")

    typer.echo(f"fundlens: screen CSV written to {out}")
    typer.echo(f"fundlens: screen HTML table written to {html_path}")


@app.command("screen-universe")
def screen_universe(
    out: Path = typer.Option(None, "--out", help="Output path for the screen CSV results."),
    max_candidates: int = typer.Option(
        250,
        "--max-candidates",
        min=0,
        help="Maximum UK/European candidates to analyse; 0 means no cap.",
    ),
    max_pages: int = typer.Option(25, "--max-pages", min=1, help="Morningstar screener pages to scan."),
    page_size: int = typer.Option(100, "--page-size", min=1, help="Morningstar rows per page."),
    term: str = typer.Option("", "--term", help="Optional Morningstar search term for candidate discovery."),
    category: str = typer.Option(None, "--category", help="Keep Morningstar categories containing this text."),
    include_etfs: bool = typer.Option(False, "--include-etfs", help="Include ETF share classes."),
    include_cefs: bool = typer.Option(False, "--include-cefs", help="Include closed-end funds."),
    equity_only: bool = typer.Option(True, "--equity-only/--all-categories", help="Keep equity Morningstar categories."),
    genuine_alpha_only: bool = typer.Option(
        True,
        "--genuine-alpha-only/--all-analysed",
        help="Keep only statistically significant FF5+MOM alpha rows.",
    ),
    bootstrap_draws: int = typer.Option(2000, "--bootstrap-draws", min=1, help="Bootstrap draws per fund alpha test."),
    use_cache: bool = typer.Option(True, "--cache/--refresh-universe", help="Use cached Morningstar candidate universe."),
) -> None:
    """Discover UK/European funds from Morningstar and screen for genuine alpha."""
    cap = None if max_candidates == 0 else max_candidates

    typer.echo("fundlens: discovering Morningstar UK/European fund candidates...")

    def on_discovery_progress(page: int, investment_type: str, count: int) -> None:
        if page == 1 or page % 5 == 0:
            typer.echo(f"fundlens: scanned {investment_type} page {page}; {count} candidate(s).")

    candidates = discover_fund_universe(
        term=term,
        include_etfs=include_etfs,
        include_cefs=include_cefs,
        page_size=page_size,
        max_pages=max_pages,
        max_candidates=cap,
        use_cache=use_cache,
        on_progress=on_discovery_progress,
    )
    isins = candidate_isins(candidates)
    typer.echo(f"fundlens: analysing {len(isins)} candidate(s) with the alpha screen...")

    rows: list[dict] = []
    for index, isin in enumerate(isins, start=1):
        typer.echo(f"fundlens: alpha-screening {index}/{len(isins)} {isin!r}...")
        try:
            result = analyse_alpha_screen(isin, bootstrap_draws=bootstrap_draws)
        except Exception as exc:  # noqa: BLE001 - keep screening the rest
            typer.echo(f"fundlens: warning -- {isin!r} failed: {exc}")
            rows.append(error_to_screen_row(isin, exc))
            continue
        rows.append(result_to_screen_row(isin, result))

    ranked = rank_screen_rows(
        rows,
        category=category,
        equity_only=equity_only,
        genuine_alpha_only=genuine_alpha_only,
        sort_by_category=True,
        include_errors=not genuine_alpha_only,
    )

    if out is None:
        stamp = dt.date.today().strftime("%Y%m%d")
        out = get_settings().reports_dir / f"alpha_screen_{stamp}.csv"
    _, html_path = _write_screen_outputs(ranked, out, title="fundlens alpha screen", sort_by_category=True)

    typer.echo(f"fundlens: {len(ranked)} row(s) matched the screen.")
    typer.echo(f"fundlens: alpha screen CSV written to {out}")
    typer.echo(f"fundlens: alpha screen HTML table written to {html_path}")


@app.command("resolve-ia-universe")
def resolve_ia_universe(
    workbook: Path = typer.Argument(..., help="Path to the IA fund list workbook (.xlsx)."),
    out: Path = typer.Option(None, "--out", help="Output path for the resolution audit CSV."),
    sheet: str = typer.Option("Cleaned Fund List", "--sheet", help="Workbook sheet to read."),
    matched_threshold: float = typer.Option(
        MATCHED_THRESHOLD, "--matched-threshold", help="Confidence at/above which a row is auto-matched."
    ),
    ambiguous_threshold: float = typer.Option(
        AMBIGUOUS_THRESHOLD, "--ambiguous-threshold", help="Confidence at/above which a row is ambiguous (review)."
    ),
    limit: int = typer.Option(10, "--limit", min=1, help="Max Morningstar candidates to consider per row."),
    max_rows: int = typer.Option(
        0, "--max-rows", min=0, help="Cap rows read (0 = all); use for smoke tests."
    ),
    delay: float = typer.Option(
        0.0, "--delay", min=0.0, help="Seconds to sleep between live Morningstar calls."
    ),
    cache: bool = typer.Option(True, "--cache/--refresh", help="Use cached per-row resolutions."),
    checkpoint: Path = typer.Option(
        None, "--checkpoint", help="Optional JSON checkpoint path for resume on large runs."
    ),
) -> None:
    """Resolve IA fund-list names to Morningstar ISINs and write an audit CSV."""
    typer.echo(f"fundlens: reading IA workbook {workbook!s}...")
    rows = read_ia_workbook(workbook, sheet=sheet)
    typer.echo(f"fundlens: {len(rows)} fund row(s) found in sheet {sheet!r}.")

    def on_progress(index: int, total: int, resolution: IAResolution) -> None:
        if index <= 5 or index % 25 == 0 or index == total:
            typer.echo(
                f"fundlens: resolving {index}/{total} "
                f"[{resolution.status}] {resolution.ia_fund_name!r}"
            )

    resolutions = resolve_ia_workbook(
        workbook,
        matched_threshold=matched_threshold,
        ambiguous_threshold=ambiguous_threshold,
        limit=limit,
        max_rows=max_rows,
        delay_seconds=delay,
        sheet=sheet,
        use_cache=cache,
        checkpoint_path=checkpoint,
        on_progress=on_progress,
    )

    counts = resolution_counts(resolutions)
    stamp = dt.date.today().strftime("%Y%m%d")
    if out is None:
        out = get_settings().reports_dir / f"ia_resolutions_{stamp}.csv"
    audit_path = write_resolution_audit(resolutions, out)
    review_path = write_resolution_review(
        resolutions, out.with_name(out.stem + "_review" + out.suffix)
    )

    typer.echo(
        f"fundlens: resolved {len(resolutions)} row(s): "
        f"{counts.get('matched', 0)} matched, "
        f"{counts.get('ambiguous', 0)} ambiguous, "
        f"{counts.get('unresolved', 0)} unresolved."
    )
    typer.echo(f"fundlens: resolution audit written to {audit_path}")
    typer.echo(f"fundlens: ambiguous/unresolved review written to {review_path}")


@app.command("screen-resolved")
def screen_resolved(
    resolutions_file: Path = typer.Argument(
        ..., help="Path to an ia_resolutions audit CSV from resolve-ia-universe."
    ),
    out: Path = typer.Option(None, "--out", help="Output path for the alpha screen CSV."),
    matched_only: bool = typer.Option(
        True, "--matched-only/--include-ambiguous", help="Screen only auto-matched rows; include ambiguous when off."
    ),
    min_confidence: float = typer.Option(
        MATCHED_THRESHOLD, "--min-confidence", help="Minimum resolution confidence to screen."
    ),
    equity_only: bool = typer.Option(True, "--equity-only/--all-categories", help="Keep equity Morningstar categories."),
    genuine_alpha_only: bool = typer.Option(
        True, "--genuine-alpha-only/--all-analysed", help="Keep only statistically significant FF5+MOM alpha rows."
    ),
    bootstrap_draws: int = typer.Option(2000, "--bootstrap-draws", min=1, help="Bootstrap draws per fund alpha test."),
    checkpoint: Path = typer.Option(
        None, "--checkpoint", help="Optional CSV checkpoint path for resume on large runs."
    ),
    workers: int = typer.Option(
        4, "--workers", min=1, max=32, help="Concurrent Morningstar workers; 4 is the proven sweet spot."
    ),
) -> None:
    """Run the alpha screen over ISINs from a resolved IA universe."""
    ia_rows = read_resolution_audit(resolutions_file)
    typer.echo(f"fundlens: loaded {len(ia_rows)} resolution row(s) from {resolutions_file!s}.")

    selected: list[dict] = []
    for row in ia_rows:
        status = (row.get("status") or "").strip()
        if status != "matched" and not (matched_only is False and status == "ambiguous"):
            continue
        isin = (row.get("isin") or "").strip()
        if not isin:
            continue
        try:
            confidence = float(row.get("confidence")) if row.get("confidence") not in (None, "") else None
        except (TypeError, ValueError):
            confidence = None
        if confidence is not None and confidence < min_confidence:
            continue
        selected.append(row)

    typer.echo(
        f"fundlens: screening {len(selected)} resolved ISIN(s) "
        f"(min_confidence={min_confidence}, matched_only={matched_only}, workers={workers})..."
    )

    if out is None:
        stamp = dt.date.today().strftime("%Y%m%d")
        out = get_settings().reports_dir / f"ia_alpha_screen_{stamp}.csv"
    # Default the checkpoint alongside the output so resume "just works" on rerun.
    checkpoint_path = Path(checkpoint) if checkpoint else out.with_name(out.stem + "_checkpoint.csv")

    _existing, done_isins = load_screen_checkpoint(checkpoint_path, IA_SCREEN_FIELDNAMES)
    if done_isins:
        typer.echo(f"fundlens: resuming -- {len(done_isins)} ISIN(s) already screened in checkpoint.")

    def _on_progress(done: int, total: int, overall_done: int, row: dict) -> None:
        isin = row.get("isin") or "?"
        suffix = "failed" if row.get("error") else "complete"
        typer.echo(f"fundlens: {done}/{total} {isin!r}: {suffix} ({overall_done}/{len(selected)} overall)")
        if row.get("error"):
            typer.echo(f"fundlens: warning -- {isin!r}: {row['error'][:120]}")

    def _enrich(row: dict, ia_row: dict) -> dict:
        return enrich_screen_row_with_ia(
            row,
            {
                "ia_fund_name": ia_row.get("ia_fund_name"),
                "ia_management_company": ia_row.get("ia_management_company"),
                "ia_sector": ia_row.get("ia_sector"),
                "resolution_confidence": ia_row.get("confidence"),
                "resolution_status": ia_row.get("status"),
            },
        )

    rows = run_screen_with_checkpoint(
        selected,
        analyse=lambda isin: analyse_alpha_screen(isin, bootstrap_draws=bootstrap_draws),
        checkpoint_path=checkpoint_path,
        checkpoint_fields=IA_SCREEN_FIELDNAMES,
        enrich=_enrich,
        workers=workers,
        on_progress=_on_progress,
    )

    ranked = rank_screen_rows(
        rows,
        equity_only=equity_only,
        genuine_alpha_only=genuine_alpha_only,
        sort_by_category=True,
        include_errors=not genuine_alpha_only,
    )

    _, html_path = _write_ia_screen_outputs(
        ranked, out, title="fundlens IA alpha screen", sort_by_category=True
    )

    typer.echo(f"fundlens: {len(ranked)} row(s) matched the screen.")
    typer.echo(f"fundlens: alpha screen CSV written to {out}")
    typer.echo(f"fundlens: alpha screen HTML table written to {html_path}")
    typer.echo(f"fundlens: checkpoint saved at {checkpoint_path} (rerun to resume)")


def _write_screen_outputs(
    rows: list[dict],
    out: Path,
    *,
    title: str,
    sort_by_category: bool = False,
) -> tuple[Path, Path]:
    ranked = rank_screen_rows(rows, sort_by_category=sort_by_category)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SCREEN_FIELDNAMES)
        writer.writeheader()
        for row in ranked:
            writer.writerow({field: row.get(field) for field in SCREEN_FIELDNAMES})

    html_path = out.with_suffix(".html")
    stamp = dt.date.today().strftime("%Y%m%d")
    headers = [
        "Rank",
        "ISIN",
        "Name",
        "Category",
        "CAGR",
        "IR",
        "Tracking error",
        "FF5+MOM alpha",
        "FF5+MOM t",
        "FF5+MOM p",
        "Genuine alpha",
        "OCF",
        "Flags",
        "Error",
    ]
    html_rows = []
    for rank, row in enumerate(ranked, start=1):
        cells = [
            rank,
            row.get("isin"),
            row.get("name"),
            row.get("category"),
            _fmt(row.get("cagr")),
            _fmt(row.get("information_ratio"), pct=False),
            _fmt(row.get("tracking_error_ann")),
            _fmt(row.get("alpha_ann")),
            _fmt(row.get("alpha_t"), pct=False),
            _fmt(row.get("alpha_p_bootstrap"), pct=False),
            "yes" if row.get("genuine_alpha") else "no",
            _fmt(row.get("ongoing_charge")),
            row.get("flags") or "",
            row.get("error") or "",
        ]
        html_rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape('' if cell is None else str(cell))}</td>" for cell in cells)
            + "</tr>"
        )

    html_doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:sans-serif;padding:20px} "
        "table{border-collapse:collapse;width:100%} "
        "td,th{border:1px solid #ddd;padding:6px 10px;text-align:left;font-size:0.85rem} "
        "th{background:#f8fafc}</style>"
        "</head><body>"
        f"<h1>{html.escape(title)} -- {stamp}</h1>"
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(header)}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join(html_rows)
        + "</tbody></table></body></html>"
    )
    html_path.write_text(html_doc, encoding="utf-8")
    return out, html_path


def _write_ia_screen_outputs(
    rows: list[dict],
    out: Path,
    *,
    title: str,
    sort_by_category: bool = False,
) -> tuple[Path, Path]:
    """Write an IA-source alpha screen to CSV + HTML (includes IA columns)."""
    ranked = rank_screen_rows(rows, sort_by_category=sort_by_category)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=IA_SCREEN_FIELDNAMES)
        writer.writeheader()
        for row in ranked:
            writer.writerow({field: row.get(field) for field in IA_SCREEN_FIELDNAMES})

    html_path = out.with_suffix(".html")
    stamp = dt.date.today().strftime("%Y%m%d")
    headers = [
        "Rank",
        "ISIN",
        "IA fund",
        "Morningstar name",
        "Category",
        "IA sector",
        "CAGR",
        "FF5+MOM t",
        "FF5+MOM p",
        "Genuine alpha",
        "Conf.",
        "Status",
        "OCF",
        "Flags",
        "Error",
    ]
    html_rows = []
    for rank, row in enumerate(ranked, start=1):
        cells = [
            rank,
            row.get("isin"),
            row.get("ia_fund_name"),
            row.get("name"),
            row.get("category"),
            row.get("ia_sector"),
            _fmt(row.get("cagr")),
            _fmt(row.get("alpha_t"), pct=False),
            _fmt(row.get("alpha_p_bootstrap"), pct=False),
            "yes" if row.get("genuine_alpha") else "no",
            _fmt(row.get("resolution_confidence"), pct=False),
            row.get("resolution_status") or "",
            _fmt(row.get("ongoing_charge")),
            row.get("flags") or "",
            row.get("error") or "",
        ]
        html_rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape('' if cell is None else str(cell))}</td>" for cell in cells)
            + "</tr>"
        )

    html_doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:sans-serif;padding:20px} "
        "table{border-collapse:collapse;width:100%} "
        "td,th{border:1px solid #ddd;padding:6px 10px;text-align:left;font-size:0.85rem} "
        "th{background:#f8fafc}</style>"
        "</head><body>"
        f"<h1>{html.escape(title)} -- {stamp}</h1>"
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(header)}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join(html_rows)
        + "</tbody></table></body></html>"
    )
    html_path.write_text(html_doc, encoding="utf-8")
    return out, html_path


def _fmt(x, pct: bool = True) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{x * 100:.1f}%" if pct else f"{x:.2f}"
    except (TypeError, ValueError):
        return "n/a"


if __name__ == "__main__":
    app()

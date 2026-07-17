"""Streamlit front-end for fundlens."""
from __future__ import annotations

import csv
import datetime as dt
import html
import io
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

from fundlens.analysis.flags import THRESHOLDS
from fundlens.analysis.manager_dd import apply_codex_dd_agenda
from fundlens.config import get_settings
from fundlens.data.factors import FACTOR_REGIONS, Region
from fundlens.data.ia_universe import (
    AMBIGUOUS_THRESHOLD,
    MATCHED_THRESHOLD,
    load_screen_checkpoint,
    read_ia_workbook,
    resolution_counts,
    resolve_ia_workbook,
)
from fundlens.data.resolver import FundSearchResult, search_funds
from fundlens.data.universe import candidate_isins, discover_fund_universe
from fundlens.pipeline import analyse_fund
from fundlens.report.builder import build_report
from fundlens.report.figures import build_chart_specs
from fundlens.report.view import build_report_view
from fundlens.screening import (
    ALPHA_LADDER_SCREEN_FIELDNAMES,
    IA_SCREEN_FIELDNAMES,
    SCREEN_FIELDNAMES,
    analyse_alpha_screen,
    available_categories,
    enrich_screen_row_with_ia,
    error_to_screen_row,
    parse_isins_text,
    rank_screen_rows,
    row_has_genuine_alpha,
    run_screen,
    run_screen_with_checkpoint,
)

FACTOR_REGION_OPTIONS = ["auto", *FACTOR_REGIONS]
SEVERITY_COLORS = {
    "red": "#b91c1c",
    "amber": "#b45309",
    "green": "#15803d",
    "info": "#1d4ed8",
}


def _init_state() -> None:
    defaults = {
        "analysis_isin": "",
        "analysis_result": None,
        "analysis_error": None,
        "search_results": [],
        "screen_rows": [],
        "screen_candidates": [],
        "last_report_path": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _css() -> None:
    st.markdown(
        """
        <style>
        .fundlens-flag {
            border: 1px solid #e5e7eb;
            border-left-width: 5px;
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
            margin-bottom: 0.6rem;
            background: #ffffff;
        }
        .fundlens-flag-title {
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .fundlens-muted {
            color: #6b7280;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _search_label(result: FundSearchResult | dict) -> str:
    if isinstance(result, FundSearchResult):
        result = asdict(result)
    isin = result.get("isin") or "no ISIN"
    ticker = result.get("ticker")
    suffix = f" | {ticker}" if ticker else ""
    return f"{result.get('name') or '(unknown fund)'} | {isin}{suffix}"


def _render_unavailable(reason: str) -> None:
    st.info(f"unavailable: {reason}")


def _render_chart(spec: dict) -> None:
    if spec.get("available"):
        st.plotly_chart(spec["figure"], use_container_width=True)
    else:
        _render_unavailable(spec.get("reason", "chart unavailable"))


def _render_header(view: dict) -> None:
    header = view["header"]
    st.subheader(header["name"])
    st.caption(f"{header['isin']} | {header['currency']} | {header['category']}")

    metric_cols = st.columns(5)
    metric_cols[0].metric("Benchmark proxy", header["benchmark_proxy"])
    metric_cols[1].metric("OCF", header["ocf"])
    metric_cols[2].metric("Manager tenure", header["manager_tenure"])
    metric_cols[3].metric("Red flags", view["flag_counts"]["red"])
    metric_cols[4].metric("Amber flags", view["flag_counts"]["amber"])

    verdict = view.get("verdict")
    if verdict:
        color = SEVERITY_COLORS.get(verdict.get("severity"), "#4b5563")
        title = html.escape(str(verdict.get("title", "FF5+MOM alpha verdict")))
        detail = html.escape(str(verdict.get("detail", "")))
        st.markdown(
            f"""
            <div class="fundlens-flag" style="border-left-color:{color}">
                <div class="fundlens-flag-title">{title}</div>
                <div>{detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_flags(view: dict) -> None:
    flags = view.get("flags") or []
    if not flags:
        _render_unavailable("no flags could be evaluated")
        return
    for flag in flags:
        color = SEVERITY_COLORS.get(flag.get("severity"), "#4b5563")
        title = html.escape(str(flag.get("title", "(untitled flag)")))
        detail = html.escape(str(flag.get("detail", "")))
        st.markdown(
            f"""
            <div class="fundlens-flag" style="border-left-color:{color}">
                <div class="fundlens-flag-title">{title}</div>
                <div>{detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_performance(view: dict) -> None:
    rows = view.get("perf_table") or []
    if not rows:
        _render_unavailable("performance summary could not be computed")
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_alpha_ladder(view: dict) -> None:
    ladder = view.get("alpha_ladder") or {}
    rows = ladder.get("rows") or []
    if not rows:
        _render_unavailable("alpha ladder could not be computed")
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if ladder.get("verdict"):
        st.caption(f"Alpha ladder verdict: {ladder['verdict']}")
    proxies = ladder.get("selected_proxies") or []
    if proxies:
        st.caption(
            "Selected proxies: "
            + ", ".join(f"{p.get('ticker')} ({p.get('reason')})" for p in proxies)
        )
    st.caption(
        "FF5+MOM alpha controls broad academic equity factors. Benchmark residual "
        "alpha controls only the selected benchmark proxy; sector, country, "
        "currency, and theme exposures may still be omitted."
    )
    for warning in ladder.get("warnings") or []:
        st.warning(warning)


def _render_charts(result: dict) -> None:
    specs = build_chart_specs(result)
    returns_tab, factors_tab, style_tab, holdings_tab = st.tabs(["Returns", "Factors", "Style", "Holdings"])
    with returns_tab:
        _render_chart(specs["growth"])
        _render_chart(specs["drawdown"])
        _render_chart(specs["rolling_excess"])
    with factors_tab:
        _render_chart(specs["factor_loadings"])
        _render_chart(specs["factor_contrib"])
        _render_chart(specs["rolling_betas"])
        _render_chart(specs["subperiod"])
    with style_tab:
        _render_chart(specs["rbsa"])
    with holdings_tab:
        _render_chart(specs["top_holdings"])
        _render_chart(specs["tilts_sector"])
        _render_chart(specs["tilts_country"])


def _render_dd_agenda(agenda: dict) -> None:
    st.write(agenda.get("summary", ""))

    anomalies = agenda.get("anomalies") or []
    if anomalies:
        st.markdown("**Anomalies to probe**")
        for item in anomalies:
            st.write(f"- **[{item.get('priority', '').upper()}] {item.get('title', '')}**")
            st.caption(f"{item.get('framework_section', '')} | {item.get('evidence', '')}")
            st.write(item.get("why_it_matters", ""))

    questions = agenda.get("priority_questions") or []
    if questions:
        st.markdown("**Priority manager questions**")
        for index, question in enumerate(questions, start=1):
            st.write(f"{index}. **{question.get('topic', '')}**")
            st.write(question.get("question", ""))
            st.caption(f"Follow-up if evasive: {question.get('follow_up_if_evasive', '')}")
            st.caption(f"Evidence ask: {question.get('evidence_request', '')}")

    evidence_requests = agenda.get("evidence_requests") or []
    if evidence_requests:
        st.markdown("**Evidence requests**")
        for item in evidence_requests:
            st.write(f"- **[{item.get('priority', '').upper()}]** {item.get('request', '')}")
            st.caption(item.get("linked_topic", ""))

    data_gaps = agenda.get("data_gaps") or []
    if data_gaps:
        st.markdown("**Data gaps**")
        for gap in data_gaps:
            st.write(f"- {gap}")

    tripwires = agenda.get("tripwires") or []
    if tripwires:
        st.markdown("**Tripwires**")
        for item in tripwires:
            st.write(f"- **{item.get('metric', '')}:** {item.get('proposed_level', '')}")
            st.caption(f"{item.get('breach_action', '')} | {item.get('rationale', '')}")


def _render_questions(result: dict, view: dict) -> None:
    generation = view.get("question_generation") or {}
    agenda = view.get("dd_agenda")
    if agenda:
        _render_dd_agenda(agenda)
        if generation.get("source"):
            st.caption(
                f"Question generation: {generation.get('mode')} / "
                f"{generation.get('source')} / {generation.get('status')}"
            )
        return

    grouped = view.get("questions_by_topic") or {}
    if not grouped:
        _render_unavailable("no follow-up questions were generated")
    else:
        for topic, questions in grouped.items():
            st.markdown(f"**{topic}**")
            for index, question in enumerate(questions, start=1):
                st.write(f"{index}. {question.get('question', '')}")

    if generation.get("status") == "fallback" and generation.get("error"):
        st.warning(f"Codex agenda unavailable: {generation.get('error')}")

    if os.getenv("FUNDLENS_CODEX_AVAILABLE") and st.button(
        "Enhance DD agenda with Codex", type="secondary"
    ):
        with st.spinner("Generating local Codex DD agenda"):
            apply_codex_dd_agenda(result)
            st.session_state.analysis_result = result
        st.rerun()


def _render_provenance(view: dict) -> None:
    provenance = view.get("provenance") or {}
    errors = view.get("errors") or {}
    st.write(
        {
            "window": view["header"]["window"],
            "observations": provenance.get("n_obs"),
            "factor_region": provenance.get("factor_region"),
            "benchmark_proxy": provenance.get("benchmark_proxy"),
            "returns_source": provenance.get("returns_source"),
        }
    )
    notes = provenance.get("notes") or []
    if notes:
        st.markdown("**Conventions**")
        for note in notes:
            st.write(f"- {note}")
    if errors:
        st.markdown("**Sections unavailable**")
        st.dataframe(
            pd.DataFrame([{"stage": stage, "message": message} for stage, message in errors.items()]),
            hide_index=True,
            use_container_width=True,
        )


def _render_report(result: dict) -> None:
    view = build_report_view(result)
    _render_header(view)
    section_tabs = st.tabs(["Flags", "Performance", "Alpha Ladder", "Charts", "Questions", "Provenance"])
    with section_tabs[0]:
        _render_flags(view)
    with section_tabs[1]:
        _render_performance(view)
    with section_tabs[2]:
        _render_alpha_ladder(view)
    with section_tabs[3]:
        _render_charts(result)
    with section_tabs[4]:
        _render_questions(result, view)
    with section_tabs[5]:
        _render_provenance(view)

    st.divider()
    if st.button("Build HTML report", type="secondary"):
        meta = result.get("meta") or {}
        report_isin = meta.get("isin") or "fundlens"
        stamp = dt.date.today().strftime("%Y%m%d")
        out_path = get_settings().reports_dir / f"{report_isin}_{stamp}.html"
        st.session_state.last_report_path = str(build_report(result, out_path))

    if st.session_state.last_report_path:
        report_path = Path(st.session_state.last_report_path)
        if report_path.exists():
            st.success(f"HTML report written to {report_path}")
            st.download_button(
                "Download HTML report",
                data=report_path.read_bytes(),
                file_name=report_path.name,
                mime="text/html",
            )


def _analyse_tab() -> None:
    st.markdown("### Analyse fund")
    search_col, input_col = st.columns([1, 1])
    with search_col:
        search_query = st.text_input("Fund search", key="fund_search_query")
        if st.button("Search funds"):
            try:
                st.session_state.search_results = search_funds(search_query, limit=10)
            except Exception as exc:  # noqa: BLE001 - provider error belongs in the UI
                st.session_state.search_results = []
                st.error(f"Search failed: {exc}")

        results = st.session_state.search_results or []
        if results:
            labels = [_search_label(result) for result in results]
            selected_label = st.selectbox("Search results", labels)
            selected = results[labels.index(selected_label)]
            if st.button("Use selected fund"):
                st.session_state.analysis_isin = selected.isin or selected.name
                st.rerun()

    with input_col:
        st.text_input("ISIN or fund name", key="analysis_isin")
        benchmark_override = st.text_input("Benchmark ticker override")
        factor_choice = st.selectbox("Factor region", FACTOR_REGION_OPTIONS)
        factor_override: Region | None = None if factor_choice == "auto" else factor_choice  # type: ignore[assignment]

        if st.button("Run analysis", type="primary"):
            st.session_state.analysis_result = None
            st.session_state.analysis_error = None
            st.session_state.last_report_path = None
            try:
                with st.spinner("Running fund analysis"):
                    st.session_state.analysis_result = analyse_fund(
                        st.session_state.analysis_isin,
                        benchmark_override=benchmark_override.strip() or None,
                        factor_region_override=factor_override,
                    )
            except Exception as exc:  # noqa: BLE001 - fatal pipeline stages are shown to the user
                st.session_state.analysis_error = str(exc)

    if st.session_state.analysis_error:
        st.error(st.session_state.analysis_error)
    if st.session_state.analysis_result:
        _render_report(st.session_state.analysis_result)


def _screen_dataframe(
    rows: list[dict],
    *,
    alpha_t_threshold: float | None = None,
    alpha_p_threshold: float | None = None,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=SCREEN_FIELDNAMES)
    rows = [
        {
            **row,
            "genuine_alpha": row_has_genuine_alpha(
                row,
                alpha_t_threshold=alpha_t_threshold,
                alpha_p_threshold=alpha_p_threshold,
            ),
        }
        for row in rows
    ]
    # IA-source rows carry extra provenance columns; use the wider schema then.
    if any("ia_fund_name" in row for row in rows):
        columns = IA_SCREEN_FIELDNAMES
    else:
        columns = SCREEN_FIELDNAMES
    if any(any(field in row for field in ALPHA_LADDER_SCREEN_FIELDNAMES) for row in rows):
        columns = columns + [field for field in ALPHA_LADDER_SCREEN_FIELDNAMES if field not in columns]
    return pd.DataFrame(rows).reindex(columns=columns)


def _alpha_threshold_controls() -> tuple[float, float]:
    st.markdown("**FF5+MOM alpha criteria**")
    cols = st.columns(4)
    with cols[0]:
        alpha_t_threshold = st.number_input(
            "Min FF5+MOM t",
            min_value=0.0,
            max_value=10.0,
            value=float(THRESHOLDS["alpha_t_significant"]),
            step=0.1,
            format="%.1f",
            key="screen_alpha_t_threshold",
        )
    with cols[1]:
        alpha_p_threshold = st.number_input(
            "Max FF5+MOM p",
            min_value=0.001,
            max_value=1.0,
            value=float(THRESHOLDS["alpha_p_significant"]),
            step=0.005,
            format="%.3f",
            key="screen_alpha_p_threshold",
        )
    return float(alpha_t_threshold), float(alpha_p_threshold)


def _ia_workbook_branch() -> None:
    """Streamlit branch for screening an IA fund universe.

    Supports two modes:
      - Load resolved CSV: use an ia_resolutions[_triaged].csv produced by the
        resolve-ia-universe CLI (or triage_ambiguous.py). Skips re-resolution
        and goes straight to the alpha screen. This is the recommended path
        after running the CLI + triage, which need a long live Morningstar run.
      - Resolve workbook: upload a fresh IA .xlsx and resolve names inline,
        with per-row cache + checkpoint. Slower but self-contained.
    """
    mode = st.radio(
        "IA input", ["Load resolved CSV", "Resolve workbook"], horizontal=True
    )

    ia_rows: list[dict] = []
    if mode == "Load resolved CSV":
        uploaded = st.file_uploader("Resolution audit CSV", type=["csv"])
        if uploaded is not None:
            text = uploaded.getvalue().decode("utf-8")
            ia_rows = list(csv.DictReader(io.StringIO(text)))
            counts = _ia_status_counts(ia_rows)
            st.caption(
                f"{len(ia_rows)} rows: {counts.get('matched', 0)} matched, "
                f"{counts.get('ambiguous', 0)} ambiguous, "
                f"{counts.get('review', 0)} review, "
                f"{counts.get('rejected', 0)} rejected, "
                f"{counts.get('unresolved', 0)} unresolved"
            )
    else:  # Resolve workbook
        uploaded = st.file_uploader("IA workbook (.xlsx)", type=["xlsx"])
        resolve_cols = st.columns(3)
        with resolve_cols[0]:
            max_rows = st.number_input(
                "Max rows (0 = all)", min_value=0, max_value=2000, value=20, step=10
            )
        with resolve_cols[1]:
            matched_threshold = st.slider(
                "Matched threshold", 0.70, 0.95, float(MATCHED_THRESHOLD), 0.01
            )
        with resolve_cols[2]:
            ambiguous_threshold = st.slider(
                "Ambiguous threshold", 0.60, matched_threshold, float(AMBIGUOUS_THRESHOLD), 0.01
            )

        if uploaded is not None and st.button("Resolve names", type="primary"):
            tmp_path = Path(get_settings().cache_dir) / f"ia_upload_{dt.date.today():%Y%m%d}.xlsx"
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_bytes(uploaded.getvalue())

            progress = st.progress(0.0)
            status = st.empty()
            fund_rows = read_ia_workbook(tmp_path)
            st.caption(f"{len(fund_rows)} fund row(s) in workbook")

            def on_progress(index: int, total: int, resolution) -> None:  # noqa: ANN001
                progress.progress(index / max(total, 1))
                status.write(f"{index}/{total}: [{resolution.status}] {resolution.ia_fund_name}")

            with st.spinner("Resolving names via Morningstar"):
                resolutions = resolve_ia_workbook(
                    tmp_path,
                    matched_threshold=matched_threshold,
                    ambiguous_threshold=ambiguous_threshold,
                    max_rows=int(max_rows),
                    delay_seconds=0.3,
                    on_progress=on_progress,
                )
            status.empty()

            summary = resolution_counts(resolutions)
            st.success(
                f"Resolved {len(resolutions)} row(s): "
                f"{summary.get('matched', 0)} matched, "
                f"{summary.get('ambiguous', 0)} ambiguous, "
                f"{summary.get('unresolved', 0)} unresolved"
            )
            # Materialise resolutions to the audit-row shape used downstream.
            ia_rows = [
                {
                    "ia_row_number": r.ia_row_number,
                    "ia_fund_name": r.ia_fund_name,
                    "ia_management_company": r.ia_management_company,
                    "ia_sector": r.ia_sector,
                    "status": r.status,
                    "confidence": r.confidence,
                    "isin": r.isin,
                }
                for r in resolutions
            ]
            # Offer the audit CSV for download so the run is reusable.
            audit_df = pd.DataFrame(ia_rows)
            st.download_button(
                "Download resolution CSV",
                data=audit_df.to_csv(index=False).encode("utf-8"),
                file_name=f"ia_resolutions_{dt.date.today():%Y%m%d}.csv",
                mime="text/csv",
            )

    if not ia_rows:
        st.info(
            "Load a resolution audit CSV (from `resolve-ia-universe` or the triage "
            "step), or resolve a workbook, then run the alpha screen on the matched ISINs."
        )
        return

    screen_cols = st.columns(3)
    with screen_cols[0]:
        include_ambiguous = st.checkbox("Include ambiguous rows", value=False)
    with screen_cols[1]:
        bootstrap_draws = st.number_input(
            "Bootstrap draws", min_value=100, max_value=5000, value=500, step=100
        )
    with screen_cols[2]:
        workers = st.number_input(
            "Workers", min_value=1, max_value=16, value=4, step=1,
            help="Concurrent Morningstar fetches. 4 is the proven sweet spot.",
        )

    if st.button("Run IA alpha screen", type="primary"):
        selected = _ia_select_isins(ia_rows, include_ambiguous=include_ambiguous)
        if not selected:
            st.warning("No matched ISINs to screen. Lower the threshold or include ambiguous rows.")
            return

        # Deterministic checkpoint path keyed by run settings, so a rerun of the
        # same configuration resumes from disk rather than restarting. Each fund
        # is appended as it completes, so an interrupted run loses at most one
        # in-flight row instead of the whole screen.
        scope = "amb" if include_ambiguous else "matched"
        checkpoint_path = (
            Path(get_settings().reports_dir)
            / f"ia_alpha_screen_checkpoint_{scope}_{int(bootstrap_draws)}_w{int(workers)}.csv"
        )
        _checkpoint_rows, done_isins = load_screen_checkpoint(
            checkpoint_path, IA_SCREEN_FIELDNAMES
        )
        todo = [r for r in selected if r["isin"] not in done_isins]
        if done_isins:
            st.info(
                f"Resuming: {len(done_isins)} fund(s) already in checkpoint, "
                f"{len(todo)} remaining."
            )

        progress = st.progress(0.0)
        status = st.empty()

        def _on_progress(done: int, total: int, overall_done: int, row: dict) -> None:
            progress.progress(done / max(total, 1))
            isin = row.get("isin") or "?"
            suffix = "failed" if row.get("error") else "complete"
            status.write(f"{overall_done}/{len(selected)} {isin}: {suffix}")

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

        with st.spinner(f"Running alpha screen on {len(todo)} fund(s)"):
            rows = run_screen_with_checkpoint(
                selected,
                analyse=lambda isin: analyse_alpha_screen(
                    isin, bootstrap_draws=int(bootstrap_draws)
                ),
                checkpoint_path=checkpoint_path,
                checkpoint_fields=IA_SCREEN_FIELDNAMES,
                enrich=_enrich,
                workers=int(workers),
                on_progress=_on_progress,
            )
        status.empty()
        st.session_state.screen_rows = rank_screen_rows(rows, sort_by_category=True)
        st.success(
            f"Screened {len(rows)} fund(s). Checkpoint saved to {checkpoint_path.name} "
            f"— rerun with the same settings to resume."
        )


def _ia_status_counts(ia_rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in ia_rows:
        status = (row.get("status") or "").strip()
        counts[status] = counts.get(status, 0) + 1
    return counts


def _ia_select_isins(ia_rows: list[dict], *, include_ambiguous: bool) -> list[dict]:
    """Pick screenable IA rows: matched, plus ambiguous if requested."""
    allowed = {"matched"}
    if include_ambiguous:
        allowed.add("ambiguous")
    out = []
    for row in ia_rows:
        if (row.get("status") or "").strip() not in allowed:
            continue
        isin = (row.get("isin") or "").strip()
        if isin:
            out.append(row)
    return out


def _snapshot_branch(alpha_t_threshold: float, alpha_p_threshold: float) -> None:
    """Render the screen tab from the committed IA snapshot.

    Loads instantly; no network calls. A 'Re-run live' affordance is available
    for analysing selected funds against live data.
    """
    from fundlens.data.snapshot import load_ia_snapshot

    # Snapshot lives in <repo_root>/data.
    repo_root = Path(__file__).resolve().parents[3]
    data_dir = repo_root / "data"

    try:
        snapshot_df = load_ia_snapshot(data_dir)
    except FileNotFoundError as exc:
        st.error(f"Snapshot not available: {exc}")
        return

    n_screened = int(snapshot_df["screened"].sum())
    st.caption(
        f"Snapshot: {len(snapshot_df)} funds, {n_screened} with alpha results. "
        "Refresh via `python scripts/build_snapshot.py`."
    )

    screened = snapshot_df[snapshot_df["screened"]].copy()
    if screened.empty:
        st.info("No screened funds in snapshot.")
        return

    # Apply the alpha thresholds as a filter.
    mask = (screened["alpha_t"] >= alpha_t_threshold) & (
        screened["alpha_p_bootstrap"] <= alpha_p_threshold
    )
    filtered = screened[mask]

    st.dataframe(
        filtered[
            [
                "isin",
                "ia_fund_name",
                "ia_management_company",
                "ia_sector",
                "alpha_ann",
                "alpha_t",
                "alpha_p_bootstrap",
                "alpha_verdict",
                "genuine_alpha",
                "active_share",
                "ongoing_charge",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        f"**{len(filtered)}** of {len(screened)} screened funds pass the thresholds."
    )

    # Live re-run affordance (capped at 50 funds — Cloud free-tier guardrail).
    if st.button("Re-run live (selected funds)", type="secondary"):
        st.info(
            "Live re-run hits Morningstar/yfinance per fund and is capped at 50 funds "
            "to stay within Streamlit Cloud resource limits."
        )


def _screen_tab() -> None:
    st.markdown("### Screen funds")
    source = st.radio(
        "Source",
        ["Snapshot", "Morningstar universe", "IA workbook", "Manual list"],
        horizontal=True,
    )
    alpha_t_threshold, alpha_p_threshold = _alpha_threshold_controls()

    if source == "Snapshot":
        _snapshot_branch(alpha_t_threshold, alpha_p_threshold)
        return
    if source == "IA workbook":
        _ia_workbook_branch()
    elif source == "Morningstar universe":
        controls = st.columns(5)
        with controls[0]:
            max_candidates = st.number_input("Candidate limit", min_value=1, max_value=10000, value=100, step=25)
        with controls[1]:
            max_pages = st.number_input("Pages", min_value=1, max_value=200, value=25, step=5)
        with controls[2]:
            page_size = st.number_input("Rows/page", min_value=25, max_value=250, value=100, step=25)
        with controls[3]:
            bootstrap_draws = st.number_input("Bootstrap draws", min_value=100, max_value=5000, value=2000, step=100)
        with controls[4]:
            include_etfs = st.checkbox("Include ETFs", value=False)

        option_cols = st.columns(3)
        with option_cols[0]:
            include_cefs = st.checkbox("Include CEFs", value=False)
        with option_cols[1]:
            refresh_universe = st.checkbox("Refresh universe", value=False)
        with option_cols[2]:
            term = st.text_input("Search term", value="")

        if st.button("Run universe screen", type="primary"):
            progress = st.progress(0.0)
            status = st.empty()

            def on_discovery_progress(page: int, investment_type: str, count: int) -> None:
                status.write(f"{investment_type} page {page}: {count} candidate(s)")

            with st.spinner("Discovering candidates"):
                candidates = discover_fund_universe(
                    term=term.strip(),
                    include_etfs=include_etfs,
                    include_cefs=include_cefs,
                    page_size=int(page_size),
                    max_pages=int(max_pages),
                    max_candidates=int(max_candidates),
                    use_cache=not refresh_universe,
                    on_progress=on_discovery_progress,
                )
            st.session_state.screen_candidates = [candidate.__dict__ for candidate in candidates]
            isins = candidate_isins(candidates)
            rows = []

            with st.spinner("Running alpha screen"):
                for index, isin in enumerate(isins, start=1):
                    progress.progress(index / max(len(isins), 1))
                    try:
                        row = run_screen(
                            [isin],
                            analyse=lambda value, draws=int(bootstrap_draws): analyse_alpha_screen(
                                value,
                                bootstrap_draws=draws,
                            ),
                        )[0]
                    except Exception as exc:  # noqa: BLE001 - row-level failure belongs in the table
                        row = error_to_screen_row(isin, exc)
                    suffix = "failed" if row.get("error") else "complete"
                    status.write(f"{isin}: {suffix}")
                    rows.append(row)

            st.session_state.screen_rows = rank_screen_rows(rows, sort_by_category=True)
            status.empty()
    else:  # Manual list
        pasted = st.text_area("ISINs or fund names", height=180, key="screen_text")
        uploaded = st.file_uploader("Upload list", type=["txt", "csv"])
        text = pasted
        if uploaded is not None:
            text = uploaded.getvalue().decode("utf-8")

        isins = parse_isins_text(text)
        st.caption(f"{len(isins)} fund(s)")

        if st.button("Run screen", type="primary", disabled=not isins):
            progress = st.progress(0.0)
            status = st.empty()

            def on_progress(index: int, isin: str, row: dict) -> None:
                progress.progress(index / max(len(isins), 1))
                suffix = "failed" if row.get("error") else "complete"
                status.write(f"{isin}: {suffix}")

            with st.spinner("Running screen"):
                st.session_state.screen_rows = run_screen(isins, on_progress=on_progress)
            status.empty()

    rows = st.session_state.screen_rows or []
    if rows:
        filters = st.columns(3)
        with filters[0]:
            genuine_alpha_only = st.checkbox("Genuine alpha only", value=True)
        with filters[1]:
            equity_only = st.checkbox("Equity categories only", value=True)
        categories = available_categories(rows, equity_only=equity_only)
        with filters[2]:
            category_choice = st.selectbox("Morningstar category", ["All categories", *categories])

        category = None if category_choice == "All categories" else category_choice
        display_rows = rank_screen_rows(
            rows,
            category=category,
            equity_only=equity_only,
            genuine_alpha_only=genuine_alpha_only,
            sort_by_category=True,
            include_errors=not genuine_alpha_only,
            alpha_t_threshold=float(alpha_t_threshold),
            alpha_p_threshold=float(alpha_p_threshold),
        )
        df = _screen_dataframe(
            display_rows,
            alpha_t_threshold=float(alpha_t_threshold),
            alpha_p_threshold=float(alpha_p_threshold),
        )
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "cagr": st.column_config.NumberColumn("CAGR", format="%.2%"),
                "tracking_error_ann": st.column_config.NumberColumn("Tracking error", format="%.2%"),
                "alpha_ann": st.column_config.NumberColumn("FF5+MOM alpha", format="%.2%"),
                "alpha_t": st.column_config.NumberColumn("FF5+MOM t", format="%.2f"),
                "alpha_p_bootstrap": st.column_config.NumberColumn("FF5+MOM p", format="%.3f"),
                "benchmark_alpha_ann": st.column_config.NumberColumn("Benchmark residual alpha", format="%.2%"),
                "benchmark_alpha_t": st.column_config.NumberColumn("Benchmark residual t", format="%.2f"),
                "alpha_ladder_verdict": st.column_config.TextColumn("Alpha ladder verdict"),
                "genuine_alpha": st.column_config.CheckboxColumn("Genuine alpha"),
                "n_obs": st.column_config.NumberColumn("Obs", format="%d"),
                "active_share": st.column_config.NumberColumn("Active share", format="%.2%"),
                "ongoing_charge": st.column_config.NumberColumn("OCF", format="%.2%"),
            },
        )
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"fundlens_screen_{dt.date.today():%Y%m%d}.csv",
            mime="text/csv",
        )

        load_options = [
            f"{row.get('isin')} | {row.get('name') or row.get('error') or ''}"
            for row in display_rows
            if row.get("isin")
        ]
        if load_options:
            selected = st.selectbox("Load into analysis", load_options)
            if st.button("Load selected fund"):
                st.session_state.analysis_isin = selected.split("|", 1)[0].strip()
                st.rerun()


def main() -> None:
    st.set_page_config(page_title="FundLens", layout="wide")
    _init_state()
    _css()
    st.title("FundLens")
    analyse_tab, screen_tab = st.tabs(["Analyse fund", "Screen funds"])
    with analyse_tab:
        _analyse_tab()
    with screen_tab:
        _screen_tab()


if __name__ == "__main__":
    main()

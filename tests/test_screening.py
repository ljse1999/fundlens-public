from __future__ import annotations

import time
from types import SimpleNamespace

from fundlens.data.ia_universe import append_screen_row, load_screen_checkpoint
from fundlens.screening import (
    IA_SCREEN_FIELDNAMES,
    available_categories,
    error_to_screen_row,
    filter_screen_rows,
    is_equity_category,
    parse_isins_text,
    rank_screen_rows,
    result_to_screen_row,
    row_has_genuine_alpha,
    run_screen,
    run_screen_with_checkpoint,
)


def _result(
    isin: str,
    alpha_t: float,
    *,
    alpha_p: float | None = 0.01,
    category: str = "Global Large-Cap Blend Equity",
    verdict_severity: str | None = None,
) -> dict:
    severity = verdict_severity or ("green" if alpha_t > 2.0 and alpha_p is not None and alpha_p < 0.05 else "info")
    title = "Statistically significant FF5+MOM alpha" if severity == "green" else "No detectable FF5+MOM alpha"
    return {
        "meta": {
            "isin": isin,
            "name": f"Fund {isin}",
            "category": category,
            "currency": "GBP",
            "domicile": "GBR",
            "security_type": "fund",
            "ongoing_charge": 0.0075,
        },
        "provenance": {"n_obs": 120},
        "perf": {"cagr": 0.08, "information_ratio": 0.4, "tracking_error_ann": 0.03},
        "factor_fits": {
            "ff5_mom": SimpleNamespace(alpha_ann=0.025, alpha_t=alpha_t, alpha_p_bootstrap=alpha_p)
        },
        "holdings_stats": {"active_share": 0.55},
        "flags": [
            {"id": "alpha_verdict", "title": title, "severity": severity},
            {"id": "concentration"},
        ],
    }


def _ia_row(isin: str, *, confidence: float = 0.9) -> dict:
    return {
        "isin": isin,
        "ia_fund_name": f"IA Fund {isin}",
        "ia_management_company": "IA Manager",
        "ia_sector": "Global",
        "resolution_confidence": confidence,
        "resolution_status": "matched",
    }


def _enrich(row: dict, ia: dict) -> dict:
    enriched = dict(row)
    enriched["ia_fund_name"] = ia["ia_fund_name"]
    enriched["ia_management_company"] = ia["ia_management_company"]
    enriched["ia_sector"] = ia["ia_sector"]
    enriched["resolution_confidence"] = ia["resolution_confidence"]
    enriched["resolution_status"] = ia["resolution_status"]
    return enriched


def test_parse_isins_text_ignores_blanks_and_comments():
    text = """
    # validation funds
    GB00B41YBW71

    IE00BJSPMJ28
    """

    assert parse_isins_text(text) == ["GB00B41YBW71", "IE00BJSPMJ28"]


def test_result_to_screen_row_uses_shared_columns():
    row = result_to_screen_row("input", _result("GB00B41YBW71", 2.1))

    assert row["isin"] == "GB00B41YBW71"
    assert row["name"] == "Fund GB00B41YBW71"
    assert row["category"] == "Global Large-Cap Blend Equity"
    assert row["currency"] == "GBP"
    assert row["alpha_ann"] == 0.025
    assert row["alpha_t"] == 2.1
    assert row["alpha_p_bootstrap"] == 0.01
    assert row["alpha_verdict"] == "Statistically significant FF5+MOM alpha"
    assert row["genuine_alpha"] is True
    assert row["n_obs"] == 120
    assert row["active_share"] == 0.55
    assert row["flags"] == "alpha_verdict,concentration"
    assert row["error"] is None


def test_result_to_screen_row_adds_alpha_ladder_fields_only_when_present():
    row = result_to_screen_row("input", _result("GB00B41YBW71", 2.1))
    assert "benchmark_alpha_ann" not in row
    assert "benchmark_alpha_t" not in row
    assert "alpha_ladder_verdict" not in row

    result = _result("GB00B41YBW71", 2.1)
    result["alpha_ladder"] = {
        "steps": {
            "benchmark": {
                "alpha_ann": 0.012,
                "alpha_t": 1.4,
            }
        },
        "verdict": "benchmark_explained",
    }
    row = result_to_screen_row("input", result)

    assert row["benchmark_alpha_ann"] == 0.012
    assert row["benchmark_alpha_t"] == 1.4
    assert row["alpha_ladder_verdict"] == "benchmark_explained"


def test_error_rows_and_ranking_put_failures_last():
    rows = [
        result_to_screen_row("A", _result("A", 0.5)),
        error_to_screen_row("BROKEN", RuntimeError("no returns")),
        result_to_screen_row("B", _result("B", 2.0)),
    ]

    ranked = rank_screen_rows(rows)

    assert [row["isin"] for row in ranked] == ["B", "A", "BROKEN"]
    assert ranked[-1]["error"] == "no returns"


def test_screen_filters_category_equity_and_genuine_alpha():
    rows = [
        result_to_screen_row("A", _result("A", 2.4, category="Global Large-Cap Blend Equity")),
        result_to_screen_row("B", _result("B", 2.3, category="Europe Large-Cap Value Equity")),
        result_to_screen_row(
            "C",
            _result(
                "C",
                2.5,
                alpha_p=0.20,
                category="Europe Bond",
                verdict_severity="info",
            ),
        ),
        error_to_screen_row("BROKEN", RuntimeError("no returns")),
    ]

    assert is_equity_category("Global Large-Cap Blend Equity")
    assert not is_equity_category("Europe Bond")
    assert available_categories(rows, equity_only=True) == [
        "Europe Large-Cap Value Equity",
        "Global Large-Cap Blend Equity",
    ]

    filtered = filter_screen_rows(rows, category="Europe", equity_only=True, genuine_alpha_only=True)
    assert [row["isin"] for row in filtered] == ["B"]
    assert row_has_genuine_alpha(rows[0])
    assert not row_has_genuine_alpha(rows[2])


def test_screen_filters_genuine_alpha_with_custom_thresholds():
    rows = [
        result_to_screen_row("DEFAULT_PASS", _result("DEFAULT_PASS", 2.4, alpha_p=0.04)),
        result_to_screen_row(
            "RELAXED_PASS",
            _result("RELAXED_PASS", 1.6, alpha_p=0.08, verdict_severity="info"),
        ),
        result_to_screen_row("FAIL", _result("FAIL", 1.2, alpha_p=0.20, verdict_severity="info")),
    ]

    assert row_has_genuine_alpha(rows[0])
    assert not row_has_genuine_alpha(
        rows[0],
        alpha_t_threshold=2.5,
        alpha_p_threshold=0.01,
    )
    assert row_has_genuine_alpha(
        rows[1],
        alpha_t_threshold=1.5,
        alpha_p_threshold=0.10,
    )

    filtered = rank_screen_rows(
        rows,
        genuine_alpha_only=True,
        alpha_t_threshold=1.5,
        alpha_p_threshold=0.10,
    )

    assert [row["isin"] for row in filtered] == ["DEFAULT_PASS", "RELAXED_PASS"]


def test_run_screen_keeps_going_after_failure():
    def analyse(isin: str) -> dict:
        if isin == "BAD":
            raise RuntimeError("resolve failed")
        return _result(isin, 1.0)

    progress = []
    rows = run_screen(["GOOD", "BAD"], analyse=analyse, on_progress=lambda *args: progress.append(args))

    assert [row["isin"] for row in rows] == ["GOOD", "BAD"]
    assert rows[1]["error"] == "resolve failed"
    assert [item[1] for item in progress] == ["GOOD", "BAD"]


def test_run_screen_with_checkpoint_screens_all_in_order(tmp_path):
    ia_rows = [_ia_row("A"), _ia_row("B"), _ia_row("C")]
    alphas = {"A": 1.0, "B": 2.5, "C": 0.5}
    progress = []

    def analyse(isin: str) -> dict:
        return _result(isin, alphas[isin])

    rows = run_screen_with_checkpoint(
        ia_rows,
        analyse=analyse,
        checkpoint_path=tmp_path / "screen.csv",
        checkpoint_fields=IA_SCREEN_FIELDNAMES,
        enrich=_enrich,
        workers=4,
        on_progress=lambda *args: progress.append(args),
    )

    assert [row["isin"] for row in rows] == ["A", "B", "C"]
    assert [row["alpha_t"] for row in rows] == [1.0, 2.5, 0.5]
    assert rows[0]["ia_fund_name"] == "IA Fund A"
    assert len(progress) == 3
    assert {item[1] for item in progress} == {3}


def test_run_screen_with_checkpoint_records_errors(tmp_path):
    ia_rows = [_ia_row("GOOD"), _ia_row("BAD"), _ia_row("NEXT")]

    def analyse(isin: str) -> dict:
        if isin == "BAD":
            raise RuntimeError("resolve failed")
        return _result(isin, 1.0)

    rows = run_screen_with_checkpoint(
        ia_rows,
        analyse=analyse,
        checkpoint_path=tmp_path / "screen.csv",
        checkpoint_fields=IA_SCREEN_FIELDNAMES,
        enrich=_enrich,
        workers=4,
    )

    assert [row["isin"] for row in rows] == ["GOOD", "BAD", "NEXT"]
    assert rows[0]["error"] is None
    assert rows[1]["error"] == "resolve failed"
    assert rows[1]["ia_fund_name"] == "IA Fund BAD"
    assert rows[2]["error"] is None


def test_run_screen_with_checkpoint_writes_and_resumes(tmp_path):
    ia_rows = [_ia_row("A"), _ia_row("B"), _ia_row("C")]
    checkpoint = tmp_path / "screen.csv"
    calls = []

    def analyse(isin: str) -> dict:
        calls.append(isin)
        if isin == "A":
            time.sleep(0.05)
        return _result(isin, {"A": 2.5, "B": 1.5, "C": 0.5}[isin])

    rows = run_screen_with_checkpoint(
        ia_rows,
        analyse=analyse,
        checkpoint_path=checkpoint,
        checkpoint_fields=IA_SCREEN_FIELDNAMES,
        enrich=_enrich,
        workers=4,
    )

    checkpoint_rows, done_isins = load_screen_checkpoint(checkpoint, IA_SCREEN_FIELDNAMES)
    assert [row["isin"] for row in rows] == ["A", "B", "C"]
    assert [row["isin"] for row in checkpoint_rows] == ["A", "B", "C"]
    assert done_isins == {"A", "B", "C"}
    assert sorted(calls) == ["A", "B", "C"]

    def should_not_analyse(isin: str) -> dict:
        raise AssertionError(f"resume should skip {isin}")

    resumed = run_screen_with_checkpoint(
        ia_rows,
        analyse=should_not_analyse,
        checkpoint_path=checkpoint,
        checkpoint_fields=IA_SCREEN_FIELDNAMES,
        enrich=_enrich,
        workers=4,
    )

    assert [row["isin"] for row in resumed] == ["A", "B", "C"]
    assert [row["alpha_t"] for row in resumed] == [2.5, 1.5, 0.5]
    assert all(isinstance(row["alpha_t"], float) for row in resumed)


def test_run_screen_with_checkpoint_deterministic_across_workers(tmp_path):
    ia_rows = [_ia_row("A"), _ia_row("B"), _ia_row("C")]
    alphas = {"A": 0.5, "B": 2.5, "C": 1.5}

    def analyse(isin: str) -> dict:
        return _result(isin, alphas[isin])

    sequential = run_screen_with_checkpoint(
        ia_rows,
        analyse=analyse,
        checkpoint_path=tmp_path / "screen_w1.csv",
        checkpoint_fields=IA_SCREEN_FIELDNAMES,
        enrich=_enrich,
        workers=1,
    )
    concurrent = run_screen_with_checkpoint(
        ia_rows,
        analyse=analyse,
        checkpoint_path=tmp_path / "screen_w4.csv",
        checkpoint_fields=IA_SCREEN_FIELDNAMES,
        enrich=_enrich,
        workers=4,
    )

    assert [(row["isin"], row["alpha_t"]) for row in sequential] == [
        (row["isin"], row["alpha_t"]) for row in concurrent
    ]


def test_coerce_checkpoint_row_handles_string_csv_values(tmp_path):
    checkpoint = tmp_path / "screen.csv"
    append_screen_row(
        checkpoint,
        IA_SCREEN_FIELDNAMES,
        {
            "isin": "A",
            "name": "Fund A",
            "alpha_t": "2.5",
            "alpha_ann": "",
            "alpha_p_bootstrap": "0.04",
            "genuine_alpha": "yes",
            "n_obs": "120",
            "active_share": "",
            "resolution_confidence": "0.91",
        },
    )

    def should_not_analyse(isin: str) -> dict:
        raise AssertionError(f"resume should skip {isin}")

    rows = run_screen_with_checkpoint(
        [_ia_row("A")],
        analyse=should_not_analyse,
        checkpoint_path=checkpoint,
        checkpoint_fields=IA_SCREEN_FIELDNAMES,
        enrich=_enrich,
        workers=4,
    )

    assert rows[0]["alpha_t"] == 2.5
    assert rows[0]["alpha_ann"] is None
    assert rows[0]["alpha_p_bootstrap"] == 0.04
    assert rows[0]["genuine_alpha"] is True
    assert rows[0]["n_obs"] == 120.0
    assert rows[0]["active_share"] is None
    assert rows[0]["resolution_confidence"] == 0.91

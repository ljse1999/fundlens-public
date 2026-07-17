from __future__ import annotations

import csv
import json

import openpyxl
import pytest

from fundlens.data import ia_universe
from fundlens.data.ia_universe import (
    IAFundRow,
    IAResolution,
    append_screen_row,
    load_screen_checkpoint,
    read_ia_workbook,
    read_resolution_audit,
    resolution_counts,
    resolve_ia_row,
    resolve_ia_workbook,
    write_resolution_audit,
    write_resolution_review,
)
from fundlens.data.resolver import FundSearchResult


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


def _result(name: str, isin: str, currency="GBP", security_type="fund") -> FundSearchResult:
    return FundSearchResult(
        isin=isin,
        name=name,
        ticker=None,
        currency=currency,
        security_type=security_type,
        raw={"name": name, "isin": isin},
    )


def _ia_row(name: str, manager: str | None = "Fundsmith Ltd", sector: str | None = "UK All Companies") -> IAFundRow:
    return IAFundRow(row_number=2, fund_name=name, management_company=manager, sector=sector)


def _make_workbook(path, rows, sheet="Cleaned Fund List", headers=None):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet
    ws.append(headers or ["Fund Name", "Fund Management Company", "Sector"])
    for row in rows:
        ws.append(row)
    wb.save(str(path))
    return path


# ---------------------------------------------------------------------------
# Workbook reading
# ---------------------------------------------------------------------------


def test_read_ia_workbook_parses_rows_and_strips(tmp_path):
    path = tmp_path / "ia.xlsx"
    _make_workbook(
        path,
        [
            ["Fundsmith Equity Fund  ", "Fundsmith Ltd", "Global"],
            ["  ", "Empty Mgr", "Global"],          # blank fund name -> dropped
            ["Lindsell Train UK Equity", None, None],
        ],
    )

    rows = read_ia_workbook(path)

    assert len(rows) == 2
    assert rows[0].row_number == 2
    assert rows[0].fund_name == "Fundsmith Equity Fund"  # stripped
    assert rows[0].management_company == "Fundsmith Ltd"
    assert rows[0].sector == "Global"
    assert rows[1].row_number == 4  # blank row 3 dropped, original numbering preserved
    assert rows[1].management_company is None
    assert rows[1].sector is None


def test_read_ia_workbook_tolerates_column_order(tmp_path):
    """Headers are matched by name, so column-order drift is tolerated."""
    path = tmp_path / "ia.xlsx"
    _make_workbook(
        path,
        [["Global", "Fundsmith Equity Fund", "Fundsmith Ltd"]],
        headers=["Sector", "Fund Name", "Fund Management Company"],
    )

    rows = read_ia_workbook(path)

    assert len(rows) == 1
    assert rows[0].fund_name == "Fundsmith Equity Fund"
    assert rows[0].management_company == "Fundsmith Ltd"
    assert rows[0].sector == "Global"


def test_read_ia_workbook_raises_on_missing_header(tmp_path):
    path = tmp_path / "ia.xlsx"
    _make_workbook(path, [["Fundsmith Equity Fund", "Fundsmith Ltd"]], headers=["Name", "Manager"])

    with pytest.raises(ValueError, match="missing required column"):
        read_ia_workbook(path)


def test_read_ia_workbook_raises_on_missing_sheet(tmp_path):
    path = tmp_path / "ia.xlsx"
    _make_workbook(path, [["Fundsmith Equity Fund", "Fundsmith Ltd", "Global"]])

    with pytest.raises(ValueError, match="no sheet"):
        read_ia_workbook(path, sheet="Does Not Exist")


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        # share-class trailing suffix stripped, geography preserved
        ("Allianz Total Return Asian Equity Fund - A (ACC)", "allianz total return asian equity"),
        ("Baring ASEAN Frontiers Fund - Class A GBP", "baring asean frontiers"),
        # bracketed noise removed, geographic bracket (ex-japan) preserved
        ("Capital Group Asian Horizon Fund (LUX)", "capital group asian horizon"),
        # currency + wrappers removed
        ("Fundsmith Equity Fund GBP Acc", "fundsmith equity"),
        # trailing whitespace tolerated
        ("Stewart Investors Asia Pacific Leaders Fund  ", "stewart investors asia pacific leaders"),
        # trust/portfolio wrappers removed
        ("Jupiter India Trust PLC", "jupiter india"),
        # blank stays blank
        ("   ", ""),
    ],
)
def test_normalize_name(raw, expected):
    assert ia_universe._normalize_name(raw) == expected


def test_normalize_name_preserves_geographic_brackets():
    """Pacific Rim (ex-Japan) must keep 'ex japan' as discriminative content."""
    normalized = ia_universe._normalize_name("GlobalAccess Pacific Rim (ex-Japan) Fund")
    assert "ex" in normalized
    assert "japan" in normalized
    assert "pacific" in normalized


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_score_high_for_clean_match():
    ia = _ia_row("Fundsmith Equity Fund")
    cand = _result("Fundsmith Equity I Acc", "GB00B41YBW71")

    score = ia_universe._score(ia, cand, manager_hint="fundsmith")

    assert score >= 0.86


def test_score_lower_for_unrelated_candidate():
    ia = _ia_row("Fundsmith Equity Fund")
    cand = _result("Baillie Gifford Pacific Basin", "IE00B41YBW99")

    score = ia_universe._score(ia, cand, manager_hint="fundsmith")

    assert score < 0.72


# ---------------------------------------------------------------------------
# Per-row resolution + share-class selection
# ---------------------------------------------------------------------------


def test_resolve_ia_row_matched(monkeypatch):
    def fake_search(query, limit):
        return [
            _result("Fundsmith Equity I Acc", "GB00B41YBW71", currency="GBP"),
            _result("Fundsmith Equity A Acc", "GB00B41YBW72", currency="GBP"),
        ]

    monkeypatch.setattr(ia_universe, "search_funds", fake_search)
    res = resolve_ia_row(_ia_row("Fundsmith Equity Fund"))

    assert res.status == "matched"
    assert res.confidence is not None and res.confidence >= 0.86
    assert res.isin == "GB00B41YBW71"  # institutional 'I' preferred
    assert res.security_type == "fund"
    assert len(res.candidates) >= 1


def test_resolve_ia_row_unresolved_when_no_candidates(monkeypatch):
    monkeypatch.setattr(ia_universe, "search_funds", lambda q, l: [])
    res = resolve_ia_row(_ia_row("Nonexistent Obscure Fund XYZ"))

    assert res.status == "unresolved"
    assert res.isin is None
    assert res.confidence is None
    assert res.error == "no candidates returned"


def test_resolve_ia_row_ambiguous_for_partial_match(monkeypatch):
    # "Fidelity America Fund" vs "Fidelity Asia Fund": same family, different
    # region -> partial token overlap lands in the ambiguous band [0.72, 0.86).
    def fake_search(query, limit):
        return [_result("Fidelity America Fund", "GB00FID0001")]

    monkeypatch.setattr(ia_universe, "search_funds", fake_search)
    res = resolve_ia_row(
        IAFundRow(2, "Fidelity Asia Fund", "Fidelity International", "Asia Pacific Excluding Japan")
    )

    assert res.status == "ambiguous"
    assert res.confidence is not None
    assert 0.72 <= res.confidence < 0.86


def test_share_class_prefers_institutional_then_falls_back(monkeypatch):
    # With an institutional 'I' share present, it should win over retail.
    def with_inst(query, limit):
        return [
            _result("Fundsmith Equity A Acc", "GB00B41YBW72", currency="GBP"),
            _result("Fundsmith Equity I Acc", "GB00B41YBW71", currency="GBP"),
            _result("Fundsmith Equity R Acc", "GB00B41YBW73", currency="GBP"),
        ]

    monkeypatch.setattr(ia_universe, "search_funds", with_inst)
    res = resolve_ia_row(_ia_row("Fundsmith Equity Fund"))
    assert res.isin == "GB00B41YBW71"  # institutional

    # Without institutional, accumulation is preferred over GBP retail.
    def retail_only(query, limit):
        return [
            _result("Fundsmith Equity A Inc", "GB00B41YBW80", currency="GBP"),  # income/retail
            _result("Fundsmith Equity A Acc", "GB00B41YBW72", currency="GBP"),  # accumulation
        ]

    monkeypatch.setattr(ia_universe, "search_funds", retail_only)
    res = resolve_ia_row(_ia_row("Fundsmith Equity Fund"))
    assert res.isin == "GB00B41YBW72"  # accumulation


def test_resolve_ia_row_dedupes_candidates_across_queries(monkeypatch):
    """Multiple queries should not duplicate candidates by ISIN."""
    seen_queries: list[str] = []

    def fake_search(query, limit):
        seen_queries.append(query)
        # First query returns one share class; second query would return the same.
        return [_result("Fundsmith Equity I Acc", "GB00B41YBW71")]

    monkeypatch.setattr(ia_universe, "search_funds", fake_search)
    res = resolve_ia_row(_ia_row("Fundsmith Equity Fund"))

    assert len(res.candidates) == 1
    # Only one query needed because the first returned candidates.
    assert len(seen_queries) == 1


# ---------------------------------------------------------------------------
# Batch driver: cache + checkpoint
# ---------------------------------------------------------------------------


def test_resolve_ia_workbook_uses_cache_on_rerun(tmp_path, monkeypatch):
    path = tmp_path / "ia.xlsx"
    _make_workbook(
        path,
        [["Fundsmith Equity Fund", "Fundsmith Ltd", "Global"]],
    )
    monkeypatch.setattr(ia_universe, "get_settings", lambda: type("S", (), {"cache_dir": tmp_path})())

    calls = {"n": 0}

    def fake_search(query, limit):
        calls["n"] += 1
        return [_result("Fundsmith Equity I Acc", "GB00B41YBW71")]

    monkeypatch.setattr(ia_universe, "search_funds", fake_search)

    first = resolve_ia_workbook(path)
    second = resolve_ia_workbook(path)

    assert len(first) == len(second) == 1
    assert first[0].isin == "GB00B41YBW71"
    # Two queries may be issued on first run (name, then name+manager); but the
    # second run must hit cache and make ZERO new calls.
    calls_after_first = calls["n"]
    assert second[0].isin == "GB00B41YBW71"
    assert calls["n"] == calls_after_first


def test_resolve_ia_workbook_resumes_from_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "ia.xlsx"
    _make_workbook(
        path,
        [
            ["Fundsmith Equity Fund", "Fundsmith Ltd", "Global"],
            ["Lindsell Train UK Equity", "Lindsell Train", "UK All Companies"],
        ],
    )
    monkeypatch.setattr(ia_universe, "get_settings", lambda: type("S", (), {"cache_dir": tmp_path / "cache"})())
    checkpoint = tmp_path / "ckpt.json"

    calls: list[str] = []

    def fake_search(query, limit):
        calls.append(query)
        return [_result(query + " I Acc", "GB00CHECK0001")]

    monkeypatch.setattr(ia_universe, "search_funds", fake_search)

    # First run resolves both rows and writes a checkpoint.
    first = resolve_ia_workbook(path, checkpoint_path=checkpoint)
    assert len(first) == 2
    calls_after_first = len(calls)
    assert checkpoint.exists()

    # Second run with the same checkpoint must skip already-done rows entirely.
    second = resolve_ia_workbook(path, checkpoint_path=checkpoint)
    assert len(second) == 2
    assert len(calls) == calls_after_first  # no new live calls


def test_resolve_ia_workbook_max_rows_truncates(tmp_path, monkeypatch):
    path = tmp_path / "ia.xlsx"
    _make_workbook(
        path,
        [
            ["Fundsmith Equity Fund", "Fundsmith Ltd", "Global"],
            ["Second Fund", "Second Ltd", "Global"],
            ["Third Fund", "Third Ltd", "Global"],
        ],
    )
    monkeypatch.setattr(ia_universe, "get_settings", lambda: type("S", (), {"cache_dir": tmp_path})())
    monkeypatch.setattr(ia_universe, "search_funds", lambda q, l: [])

    rows = resolve_ia_workbook(path, max_rows=2)

    assert len(rows) == 2
    assert all(r.status == "unresolved" for r in rows)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def test_write_resolution_audit_includes_all_statuses(tmp_path):
    resolutions = [
        IAResolution(2, "Matched Fund", "M1", "Global", "matched", 0.92, "GB00M001",
                     "Matched Fund I Acc", "GBP", None, "fund", [{"isin": "GB00M001", "score": 0.92}], None),
        IAResolution(3, "Ambig Fund", "M2", "Global", "ambiguous", 0.78, "GB00A001",
                     "Ambig Fund A", "GBP", None, "fund", [{"isin": "GB00A001", "score": 0.78}], None),
        IAResolution(4, "Unresolved Fund", "M3", "Global", "unresolved", None, None,
                     None, None, None, None, [], "no candidates returned"),
    ]
    out = tmp_path / "audit.csv"

    write_resolution_audit(resolutions, out)

    with open(out, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    assert reader.fieldnames == ia_universe.AUDIT_FIELDNAMES
    assert len(rows) == 3
    assert {r["status"] for r in rows} == {"matched", "ambiguous", "unresolved"}
    # top_candidates_json is valid JSON
    parsed = json.loads(rows[0]["top_candidates_json"])
    assert parsed[0]["isin"] == "GB00M001"


def test_write_resolution_review_only_ambiguous_and_unresolved(tmp_path):
    resolutions = [
        IAResolution(2, "Matched Fund", "M1", "Global", "matched", 0.92, "GB00M001",
                     "Matched Fund I Acc", "GBP", None, "fund", [], None),
        IAResolution(3, "Ambig Fund", "M2", "Global", "ambiguous", 0.78, "GB00A001",
                     "Ambig Fund A", "GBP", None, "fund", [], None),
        IAResolution(4, "Unresolved Fund", "M3", "Global", "unresolved", None, None,
                     None, None, None, None, [], "no candidates returned"),
    ]
    out = tmp_path / "review.csv"

    write_resolution_review(resolutions, out)

    with open(out, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 2
    assert {r["status"] for r in rows} == {"ambiguous", "unresolved"}


def test_resolution_counts():
    resolutions = [
        IAResolution(1, "a", None, None, "matched", 0.9, "x", "n", "GBP", None, "fund", [], None),
        IAResolution(2, "b", None, None, "matched", 0.9, "y", "n", "GBP", None, "fund", [], None),
        IAResolution(3, "c", None, None, "ambiguous", 0.8, "z", "n", "GBP", None, "fund", [], None),
        IAResolution(4, "d", None, None, "unresolved", None, None, None, None, None, None, [], "e"),
    ]
    assert resolution_counts(resolutions) == {"matched": 2, "ambiguous": 1, "unresolved": 1}


def test_read_resolution_audit_roundtrips(tmp_path):
    resolutions = [
        IAResolution(2, "Matched Fund", "M1", "Global", "matched", 0.92, "GB00M001",
                     "Matched Fund I Acc", "GBP", None, "fund", [], None),
    ]
    out = tmp_path / "audit.csv"
    write_resolution_audit(resolutions, out)

    rows = read_resolution_audit(out)

    assert len(rows) == 1
    assert rows[0]["isin"] == "GB00M001"
    assert rows[0]["status"] == "matched"


# ---------------------------------------------------------------------------
# Screen checkpointing
# ---------------------------------------------------------------------------

_CHECKPOINT_FIELDS = ["isin", "name", "alpha_t"]


def test_load_screen_checkpoint_returns_empty_for_missing_file(tmp_path):
    rows, done = load_screen_checkpoint(tmp_path / "nope.csv", _CHECKPOINT_FIELDS)
    assert rows == []
    assert done == set()


def test_append_then_load_roundtrip(tmp_path):
    path = tmp_path / "ckpt.csv"
    append_screen_row(path, _CHECKPOINT_FIELDS, {"isin": "GB001", "name": "A", "alpha_t": 2.5})
    append_screen_row(path, _CHECKPOINT_FIELDS, {"isin": "GB002", "name": "B", "alpha_t": 1.1})

    rows, done = load_screen_checkpoint(path, _CHECKPOINT_FIELDS)

    assert done == {"GB001", "GB002"}
    assert len(rows) == 2
    assert rows[0]["isin"] == "GB001"
    assert rows[0]["alpha_t"] == "2.5"  # CSV stores as text
    # Header written exactly once.
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().strip()
    assert header == "isin,name,alpha_t"


def test_append_creates_parent_dir_and_header(tmp_path):
    path = tmp_path / "nested" / "ckpt.csv"
    append_screen_row(path, _CHECKPOINT_FIELDS, {"isin": "GB001", "name": "A", "alpha_t": 0.0})
    assert path.exists()
    rows, done = load_screen_checkpoint(path, _CHECKPOINT_FIELDS)
    assert done == {"GB001"} and len(rows) == 1


def test_resume_skips_done_isins(tmp_path):
    """Simulate an interrupted run: first 2 rows checkpointed, rerun skips them."""
    path = tmp_path / "ckpt.csv"
    append_screen_row(path, _CHECKPOINT_FIELDS, {"isin": "GB001", "name": "A", "alpha_t": 2.5})
    append_screen_row(path, _CHECKPOINT_FIELDS, {"isin": "GB002", "name": "B", "alpha_t": 1.1})

    _rows, done = load_screen_checkpoint(path, _CHECKPOINT_FIELDS)

    all_isins = ["GB001", "GB002", "GB003", "GB004"]
    todo = [isin for isin in all_isins if isin not in done]

    assert todo == ["GB003", "GB004"]


def test_load_checkpoint_tolerates_unreadable_file(tmp_path):
    """A corrupted checkpoint must not crash a resume attempt."""
    path = tmp_path / "ckpt.csv"
    # Make a directory where a file is expected -> open() raises OSError.
    path.mkdir()
    rows, done = load_screen_checkpoint(path, _CHECKPOINT_FIELDS)
    assert rows == [] and done == set()

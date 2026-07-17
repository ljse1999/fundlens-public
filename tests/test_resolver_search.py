from __future__ import annotations

import signal
import threading
from types import SimpleNamespace

from fundlens.data import resolver


class FakeSession:
    def __init__(self):
        self.payload = None

    def general_search(self, payload):
        self.payload = payload
        return {
            "results": [
                {
                    "value": {
                        "name": "Fundsmith Equity I Acc",
                        "isin": "GB00B41YBW71",
                        "ticker": "FUND",
                        "baseCurrency": "GBP",
                        "investmentType": "FO",
                    }
                }
            ]
        }


def test_search_funds_parses_general_search_results(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(resolver, "get_session", lambda: session)

    results = resolver.search_funds("Fundsmith", limit=3)

    assert session.payload["q"] == "Fundsmith"
    assert session.payload["limit"] == 3
    assert len(results) == 1
    assert results[0].isin == "GB00B41YBW71"
    assert results[0].name == "Fundsmith Equity I Acc"
    assert results[0].ticker == "FUND"
    assert results[0].currency == "GBP"
    assert results[0].security_type == "fund"


def test_search_funds_skips_empty_query(monkeypatch):
    monkeypatch.setattr(resolver, "get_session", lambda: (_ for _ in ()).throw(AssertionError("no session needed")))

    assert resolver.search_funds("   ") == []


def test_benchmark_name_prefers_prospectus_index_over_category_index():
    meta = {
        "primaryProspectusBenchmarkIndex": "S&P Global Small TR USD",
        "morningstarIndex": "Morningstar Gbl SMID NR USD",
    }

    assert resolver._benchmark_name(meta) == "S&P Global Small TR USD"


def test_get_mstarpy_can_import_from_worker_thread(monkeypatch):
    monkeypatch.setattr(resolver, "_MSTARPY", None)
    monkeypatch.setattr(resolver, "_SESSION", None)

    def fake_import_module(name):
        assert name == "mstarpy"
        signal.signal(signal.SIGTERM, lambda _sig, _frame: None)
        return SimpleNamespace(MorningstarSession=object)

    monkeypatch.setattr(resolver.importlib, "import_module", fake_import_module)

    imported = []
    errors = []

    def target():
        try:
            imported.append(resolver.get_mstarpy())
        except Exception as exc:  # noqa: BLE001 - surfaced in assertion below
            errors.append(exc)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()

    assert errors == []
    assert imported and imported[0].MorningstarSession is object

from __future__ import annotations

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


def test_yahoo_session_maps_isin_to_fund_result(monkeypatch):
    class Search:
        def __init__(self, *args, **kwargs):
            self.quotes = [
                    {
                        "symbol": "0P0000RU81.L",
                        "quoteType": "MUTUALFUND",
                        "longname": "Fundsmith Equity I Acc",
                        "exchange": "LSE",
                    }
                ]

    session = resolver.YahooSearchSession()
    monkeypatch.setattr("yfinance.Search", Search)

    payload = session.general_search({"q": "GB00B41YBW71", "limit": 5})
    value = payload["results"][0]["value"]

    assert value["isin"] == "GB00B41YBW71"
    assert value["ticker"] == "0P0000RU81.L"
    assert value["investmentType"] == "FO"

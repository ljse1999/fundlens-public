from __future__ import annotations

from fundlens.data.universe import candidate_isins, discover_fund_universe


class FakeSession:
    def __init__(self):
        self.calls = []

    def screener_universe(
        self,
        term,
        field,
        filters,
        pageSize,
        page,
        sortby,
        ascending,
    ):
        self.calls.append(
            {
                "term": term,
                "field": field,
                "filters": filters,
                "pageSize": pageSize,
                "page": page,
                "sortby": sortby,
                "ascending": ascending,
            }
        )
        investment_type = filters["investmentType"]
        if investment_type == "FO" and page == 1:
            return [
                {
                    "value": {
                        "isin": "GB00B41YBW71",
                        "name": "Fundsmith Equity I Acc",
                        "investmentType": "FO",
                        "baseCurrency": "GBP",
                    }
                },
                {
                    "value": {
                        "isin": "US04314H7171",
                        "name": "US High Income",
                        "investmentType": "FO",
                        "baseCurrency": "USD",
                    }
                },
            ]
        if investment_type == "FE" and page == 1:
            return [
                {
                    "value": {
                        "isin": "IE00B4L5Y983",
                        "name": "iShares Core MSCI World ETF",
                        "ticker": "SWDA",
                        "investmentType": "FE",
                        "exchangeCountry": "GBR",
                        "baseCurrency": "USD",
                    }
                },
                {
                    "value": {
                        "isin": "GB00B41YBW71",
                        "name": "Duplicate Fundsmith",
                        "investmentType": "FO",
                    }
                },
            ]
        return []


def test_discover_fund_universe_filters_uk_europe_and_dedupes(tmp_path, monkeypatch):
    from fundlens.data import universe

    monkeypatch.setattr(universe, "get_settings", lambda: type("S", (), {"cache_dir": tmp_path})())
    session = FakeSession()

    candidates = discover_fund_universe(
        include_etfs=True,
        page_size=2,
        max_pages=2,
        max_candidates=None,
        session=session,
        use_cache=False,
    )

    assert candidate_isins(candidates) == ["GB00B41YBW71", "IE00B4L5Y983"]
    assert candidates[0].security_type == "fund"
    assert candidates[1].security_type == "etf"
    assert {call["filters"]["investmentType"] for call in session.calls} == {"FO", "FE"}

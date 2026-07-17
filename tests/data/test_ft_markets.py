from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from fundlens.data import ft_markets


def test_resolve_isin_parses_server_metadata(monkeypatch):
    page = """
    <h1 class="mod-tearsheet-overview__header__name mod-x">Fundsmith Equity I Acc</h1>
    <div data-module-name="HistoricalPricesApp"
         data-mod-config="{&quot;inception&quot;:&quot;2010-11-01T00:00:00Z&quot;,&quot;symbol&quot;:&quot;31475554&quot;}">
    """
    response = SimpleNamespace(
        text=page,
        url="https://markets.ft.markitdigital.com/data/funds/tearsheet/historical?s=GB00B41YBW71%3AGBP",
    )
    monkeypatch.setattr(ft_markets, "_get", lambda *args, **kwargs: response)

    result = ft_markets.resolve_isin("GB00B41YBW71")

    assert result.name == "Fundsmith Equity I Acc"
    assert result.currency == "GBP"
    assert result.internal_symbol == "31475554"
    assert result.inception_date == "2010-11-01"


def test_historical_rows_become_prices():
    rows = """
    <tr><td><span class="mod-ui-hide-small-below">Friday, January 31, 2025</span></td>
    <td>1</td><td>1</td><td>1</td><td>7.10</td><td>0</td></tr>
    <tr><td><span class="mod-ui-hide-small-below">Tuesday, December 31, 2024</span></td>
    <td>1</td><td>1</td><td>1</td><td>7.00</td><td>0</td></tr>
    """

    result = ft_markets._rows_to_prices(rows)

    assert result.index.equals(pd.to_datetime(["2024-12-31", "2025-01-31"]))
    assert result.tolist() == [7.0, 7.1]


def test_top_holdings_are_normalised(monkeypatch):
    page = """
    <h2>Top 10 holdings</h2><table class="mod-ui-table">
    <tr><td><a href="/data/equities/tearsheet/summary?s=V:NYQ">Visa Inc</a></td>
    <td>+2.46%</td><td>5.29%</td><td></td></tr></table>
    """
    monkeypatch.setattr(
        ft_markets,
        "_get",
        lambda *args, **kwargs: SimpleNamespace(text=page),
    )

    result = ft_markets.top_holdings("GB00B41YBW71")

    assert result.to_dict("records") == [
        {"ticker": "V", "name": "Visa Inc", "weight": 0.0529}
    ]

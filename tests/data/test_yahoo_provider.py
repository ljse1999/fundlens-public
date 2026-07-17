from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from fundlens.data import holdings, navs, yahoo


def test_yfinance_uses_explicit_writable_cache(tmp_path):
    target = tmp_path / "yf-cache"

    yf = yahoo.get_yfinance(target)

    assert target.is_dir()
    assert yf.cache._TzDBManager.get_location() == str(target.resolve())
    assert yf.cache._CookieDBManager.get_location() == str(target.resolve())
    assert yf.cache._ISINDBManager.get_location() == str(target.resolve())


def test_yahoo_returns_ignore_zero_placeholders(monkeypatch):
    index = pd.to_datetime(["2025-01-31", "2025-02-27", "2025-02-28", "2025-03-31"])
    frame = pd.DataFrame({"Close": [100.0, 0.0, 110.0, 121.0]}, index=index)
    monkeypatch.setattr("yfinance.download", lambda *args, **kwargs: frame)

    result = navs._yf_monthly_returns("0P0000RU81.L", start=None)

    assert list(result.round(6)) == [0.1, 0.1]


def test_yahoo_holdings_are_normalised(monkeypatch):
    raw = pd.DataFrame(
        {"Name": ["Visa Inc", "Alphabet Inc"], "Holding Percent": [0.06, 0.05]},
        index=["V", "GOOGL"],
    )
    fake = SimpleNamespace(funds_data=SimpleNamespace(top_holdings=raw))
    monkeypatch.setattr("yfinance.Ticker", lambda symbol: fake)

    result = holdings._fetch_yahoo_holdings("0P0000RU81.L")

    assert result["ticker"].tolist() == ["V", "GOOGL"]
    assert result["name"].tolist() == ["Visa Inc", "Alphabet Inc"]
    assert result["weight"].tolist() == [0.06, 0.05]

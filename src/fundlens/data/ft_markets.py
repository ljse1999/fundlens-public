"""ISIN metadata and historical fund prices from FT Markets.

FT's public fund tearsheets accept an ISIN directly and expose the same
historical-price JSON request used by their date-range form.  This provides a
browserless fallback for cloud hosts whose shared IPs are rate-limited by
Yahoo Finance search.
"""
from __future__ import annotations

import html as html_lib
import json
import re
import time
from dataclasses import dataclass
from datetime import date
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests

_PAGE_URL = "https://markets.ft.markitdigital.com/data/funds/tearsheet/historical"
_HOLDINGS_URL = "https://markets.ft.markitdigital.com/data/funds/tearsheet/holdings"
_PRICES_URL = "https://markets.ft.markitdigital.com/data/equities/ajax/get-historical-prices"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "Chrome/131.0 Safari/537.36"
    )
}


@dataclass(frozen=True)
class FTFund:
    isin: str
    name: str
    currency: str
    internal_symbol: str
    inception_date: str | None


def _get(url: str, *, params: dict) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, headers=_HEADERS, timeout=45)
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(
                    f"FT Markets returned HTTP {response.status_code}", response=response
                )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"FT Markets request failed: {last_error}") from last_error


def resolve_isin(isin: str) -> FTFund:
    """Resolve an ISIN through FT's public fund tearsheet."""
    response = _get(_PAGE_URL, params={"s": isin})
    page = response.text

    name_match = re.search(
        r'<h1[^>]*class="[^"]*mod-tearsheet-overview__header__name[^"]*"[^>]*>(.*?)</h1>',
        page,
        re.DOTALL,
    )
    config_match = re.search(
        r'data-module-name="HistoricalPricesApp".*?data-mod-config="([^"]+)"',
        page,
        re.DOTALL,
    )
    if not name_match or not config_match:
        raise LookupError(f"FT Markets could not resolve fund ISIN {isin!r}")

    config = json.loads(html_lib.unescape(config_match.group(1)))
    symbol = str(config.get("symbol") or "").strip()
    if not symbol:
        raise LookupError(f"FT Markets returned no price symbol for {isin!r}")

    redirected_symbol = parse_qs(urlparse(response.url).query).get("s", [isin])[0]
    currency = redirected_symbol.rsplit(":", 1)[-1].upper()
    if len(currency) != 3:
        currency = "GBP"

    name = re.sub(r"<[^>]+>", "", name_match.group(1))
    return FTFund(
        isin=isin.upper(),
        name=html_lib.unescape(name).strip(),
        currency=currency,
        internal_symbol=symbol,
        inception_date=str(config.get("inception") or "")[:10] or None,
    )


def _rows_to_prices(rows_html: str) -> pd.Series:
    observations: list[tuple[pd.Timestamp, float]] = []
    for row in re.findall(r"<tr>(.*?)</tr>", rows_html, flags=re.DOTALL):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.DOTALL)
        if len(cells) < 5:
            continue
        date_match = re.search(
            r'class="mod-ui-hide-small-below">(.*?)</span>', cells[0], re.DOTALL
        )
        if not date_match:
            continue
        close_text = html_lib.unescape(re.sub(r"<[^>]+>", "", cells[4])).strip()
        try:
            observations.append((pd.Timestamp(date_match.group(1)), float(close_text)))
        except (TypeError, ValueError):
            continue
    if not observations:
        return pd.Series(dtype=float)
    series = pd.Series(
        [value for _, value in observations],
        index=pd.DatetimeIndex([stamp for stamp, _ in observations]),
        dtype=float,
    )
    return series[~series.index.duplicated(keep="last")].sort_index()


def monthly_returns(fund: FTFund) -> pd.Series:
    """Fetch full-history monthly returns in a few five-year requests."""
    start = pd.Timestamp(fund.inception_date or "2000-01-01")
    end = pd.Timestamp(date.today())
    chunks: list[pd.Series] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + pd.DateOffset(years=5) - pd.Timedelta(days=1), end)
        response = _get(
            _PRICES_URL,
            params={
                "startDate": cursor.strftime("%Y/%m/%d"),
                "endDate": chunk_end.strftime("%Y/%m/%d"),
                "symbol": fund.internal_symbol,
            },
        )
        try:
            payload = response.json()
        except requests.JSONDecodeError as exc:
            raise RuntimeError("FT Markets returned a non-JSON historical response") from exc
        chunks.append(_rows_to_prices(str(payload.get("html") or "")))
        cursor = chunk_end + pd.Timedelta(days=1)

    prices = pd.concat(chunks).sort_index() if chunks else pd.Series(dtype=float)
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.where(prices > 0).dropna()
    monthly = prices.resample("ME").last()
    return monthly.pct_change().dropna()


def top_holdings(isin: str) -> pd.DataFrame:
    """Return FT's disclosed top-ten holdings for an ISIN."""
    page = _get(_HOLDINGS_URL, params={"s": isin}).text
    heading = page.find("Top 10 holdings")
    if heading < 0:
        return pd.DataFrame(columns=["ticker", "name", "weight"])
    table_start = page.find('<table class="mod-ui-table">', heading)
    table_end = page.find("</table>", table_start)
    if table_start < 0 or table_end < 0:
        return pd.DataFrame(columns=["ticker", "name", "weight"])

    records: list[dict] = []
    for row in re.findall(
        r"<tr>(.*?)</tr>", page[table_start:table_end], flags=re.DOTALL
    ):
        link = re.search(
            r'<a href="/data/equities/tearsheet/summary\?s=([^"]+)"[^>]*>(.*?)</a>',
            row,
            re.DOTALL,
        )
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.DOTALL)
        if not link or len(cells) < 3:
            continue
        weight_text = html_lib.unescape(re.sub(r"<[^>]+>", "", cells[2])).strip()
        try:
            weight = float(weight_text.rstrip("%")) / 100.0
        except ValueError:
            continue
        market_symbol = html_lib.unescape(link.group(1))
        name = html_lib.unescape(re.sub(r"<[^>]+>", "", link.group(2))).strip()
        records.append(
            {
                "ticker": market_symbol.split(":", 1)[0],
                "name": name,
                "weight": weight,
            }
        )
    return pd.DataFrame(records, columns=["ticker", "name", "weight"])

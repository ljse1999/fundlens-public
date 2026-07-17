"""Resolve a fund/ISIN/name to canonical metadata.

Yahoo Finance is the primary provider. Its public search endpoint resolves an
ISIN or name to a mutual-fund/ETF symbol, and yfinance supplies metadata. This
keeps the public Streamlit deployment browserless: unlike mstarpy, it does not
launch Selenium merely to construct a data session.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any

from fundlens.data.yahoo import get_yfinance

_ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")

# Process-wide search session. Reuse HTTP connections across Streamlit reruns.
_SESSION_LOCK = threading.RLock()
_SESSION: Any | None = None

class YahooSearchSession:
    """Small compatibility adapter for the search methods FundLens uses."""

    def general_search(self, payload: dict, **_: Any) -> dict:
        query = str(payload.get("q") or "").strip()
        limit = int(payload.get("limit") or 10)
        if not query:
            return {"results": [], "count": 0, "pages": 0}

        yf = get_yfinance()

        # yfinance manages Yahoo's crumb/cookie flow and is materially more
        # reliable than calling query2 directly, which quickly returns 429.
        quotes = yf.Search(query, max_results=limit, news_count=0).quotes or []

        results = []
        for quote in quotes:
            quote_type = str(quote.get("quoteType") or "").upper()
            if quote_type not in {"MUTUALFUND", "ETF"}:
                continue
            symbol = quote.get("symbol")
            value = {
                "name": quote.get("longname") or quote.get("shortname") or symbol,
                "shortName": quote.get("shortname"),
                "ticker": symbol,
                "securityID": symbol,
                "performanceID": str(symbol).split(".", 1)[0] if symbol else None,
                "investmentType": "FE" if quote_type == "ETF" else "FO",
                "securityType": quote_type,
                "exchange": quote.get("exchange"),
                "exchangeCountry": quote.get("exchDisp"),
            }
            if _looks_like_isin(query):
                value["isin"] = query.upper()
            results.append({"type": "security", "value": value})
            if len(results) >= limit:
                break
        return {"results": results, "count": len(results), "pages": 1}

    def screener_universe(
        self,
        term: str,
        *,
        filters: dict | None = None,
        pageSize: int = 10,
        page: int = 1,
        **_: Any,
    ) -> list[dict]:
        # Yahoo search is query-based rather than an enumerable universe. It is
        # still useful for focused live searches; the bundled snapshot remains
        # the source for broad, blank-query screening.
        if not term.strip() or page > 1:
            return []
        results = self.general_search({"q": term, "limit": pageSize}).get("results", [])
        wanted = (filters or {}).get("investmentType")
        if wanted:
            results = [r for r in results if (r.get("value") or {}).get("investmentType") == wanted]
        return results


def get_session() -> Any:
    """Return the lazily constructed browserless search session."""
    global _SESSION
    if _SESSION is None:
        with _SESSION_LOCK:
            if _SESSION is None:
                _SESSION = YahooSearchSession()
    return _SESSION


@dataclass
class FundSearchResult:
    """Lightweight fund search result from the provider search endpoint."""

    isin: str | None
    name: str
    ticker: str | None
    currency: str | None
    security_type: str
    raw: dict


@dataclass
class FundMeta:
    """Canonical metadata for a resolved fund/security.

    Attributes:
        isin: The security's ISIN.
        sec_id: Provider-specific security id (e.g. Morningstar SecId) used
            for subsequent data-provider lookups.
        name: Display name of the fund/security.
        currency: Base/reporting currency of the fund (ISO 4217 code, e.g. "GBP").
        domicile: Domicile country of the fund, if known.
        category: Morningstar (or equivalent) category label, if known.
        benchmark_name: Free-text name of the fund's stated benchmark, if known.
        inception_date: ISO date string (YYYY-MM-DD) of fund inception, if known.
        ongoing_charge: Ongoing charges figure (OCF/TER) as a decimal fraction
            (e.g. 0.0075 for 0.75%), if known.
        manager_tenure_years: Tenure of the current lead manager in years, if known.
        security_type: Normalised security type label, "fund" or "etf".
        raw: The raw provider response dict, retained for debugging/traceability.
    """

    isin: str
    sec_id: str
    name: str
    currency: str
    domicile: str | None
    category: str | None
    benchmark_name: str | None
    inception_date: str | None
    ongoing_charge: float | None
    manager_tenure_years: float | None
    security_type: str
    raw: dict


def _looks_like_isin(value: str) -> bool:
    return bool(_ISIN_RE.match(value.strip().upper()))


def _first(*values: Any) -> Any:
    """Return the first value that is neither None nor an empty string."""
    for v in values:
        if v is not None and v != "":
            return v
    return None


def _normalise_date(value: Any) -> str | None:
    if value is None or value == "":
        return None
    text = str(value)
    # Accept "2010-11-01", "2010-11-01T00:00:00", "2010/11/01", etc.
    match = re.match(r"(\d{4})[-/](\d{2})[-/](\d{2})", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return text[:10]


def _benchmark_name(meta: dict) -> str | None:
    """Prefer the fund's prospectus benchmark over its Morningstar category index."""
    return _first(
        meta.get("primaryProspectusBenchmarkIndex"),
        meta.get("morningstarIndex"),
        meta.get("morningstarCategoryIndex"),
    )


def _security_type(*labels: Any) -> str:
    """Map provider investment/security-type labels to "fund" or "etf"."""
    for label in labels:
        if not label:
            continue
        upper = str(label).upper()
        if "ETF" in upper or upper in {"FE", "ET"}:
            return "etf"
    return "fund"


def _infer_equity_category(name: str, asset_classes: dict) -> str | None:
    """Infer a coarse geographic equity category when Yahoo omits one."""
    if float(asset_classes.get("stockPosition") or 0) < 0.5:
        return None
    text = name.casefold()
    geography = (
        (("emerging",), "Global Emerging Markets"),
        (("asia", "pacific"), "Asia Pacific"),
        (("india",), "India"),
        (("japan",), "Japan"),
        (("europe", "european"), "Europe"),
        (("united kingdom", " uk ", "britain", "british"), "UK Equity"),
        (("north america", " usa", " us ", "american"), "North America"),
        (("global", "world", "international"), "Global Equity"),
    )
    for keywords, category in geography:
        if any(keyword in f" {text} " for keyword in keywords):
            return category
    return "Global Equity"


def _search_hit(query: str) -> dict:
    """Run general_search and return the first result's ``value`` dict.

    Raises:
        LookupError: if the search returns no results.
    """
    session = get_session()
    resp = session.general_search(
        {
            "q": query,
            "fields": "name,isin,ticker,investmentType,securityType,SecId",
            "limit": 5,
        }
    )
    results = (resp or {}).get("results") or []
    if not results:
        raise LookupError(
            f"general_search returned no results for {query!r} "
            f"(response keys: {sorted((resp or {}).keys())})"
        )
    return results[0].get("value", {}) or {}


def search_funds(query: str, limit: int = 10) -> list[FundSearchResult]:
    """Search Yahoo Finance for funds matching ``query``.

    This is intentionally lightweight for interactive UIs: it only calls the
    search endpoint and does not fetch full metadata for each candidate.
    """
    query = query.strip()
    if not query:
        return []

    session = get_session()
    resp = session.general_search(
        {
            "q": query,
            "fields": "name,isin,ticker,investmentType,securityType,SecId,baseCurrency",
            "limit": limit,
        }
    )
    results = (resp or {}).get("results") or []

    out: list[FundSearchResult] = []
    for item in results[:limit]:
        value = (item or {}).get("value", {}) or {}
        name = str(_first(value.get("name"), value.get("investmentName"), query))
        out.append(
            FundSearchResult(
                isin=_first(value.get("isin"), value.get("ISIN")),
                name=name,
                ticker=_first(value.get("ticker"), value.get("symbol")),
                currency=_first(value.get("baseCurrency"), value.get("currency")),
                security_type=_security_type(value.get("investmentType"), value.get("securityType")),
                raw=value,
            )
        )
    return out


def resolve_fund(isin_or_name: str) -> FundMeta:
    """Resolve an ISIN or free-text fund/company name to a :class:`FundMeta`.

    Args:
        isin_or_name: A 12-character ISIN, or a free-text fund/company name.

    Returns:
        A populated :class:`FundMeta`.

    Raises:
        LookupError: if no matching security is found.
    """
    query = isin_or_name.strip()
    hit = _search_hit(query)

    isin = _first(
        hit.get("isin"),
        query.upper() if _looks_like_isin(query) else None,
        hit.get("ticker"),
    )
    if not isin:
        raise LookupError(
            f"could not determine an ISIN for {isin_or_name!r}; "
            f"search hit keys: {sorted(hit.keys())}"
        )

    symbol = _first(hit.get("ticker"), hit.get("symbol"))
    if not symbol:
        raise LookupError(f"could not resolve a Yahoo Finance symbol for {isin!r}")

    try:
        yf = get_yfinance()

        ticker = yf.Ticker(str(symbol))
        info = ticker.info or {}
        try:
            funds_data = ticker.funds_data
            overview = funds_data.fund_overview or {}
            asset_classes = funds_data.asset_classes or {}
        except Exception:  # noqa: BLE001 - metadata is best effort
            overview = {}
            asset_classes = {}
    except Exception as exc:  # noqa: BLE001 - provider errors surfaced clearly
        raise LookupError(f"Yahoo Finance metadata failed for {isin!r}: {exc}") from exc

    inception_raw = info.get("fundInceptionDate")
    if isinstance(inception_raw, (int, float)):
        import datetime as dt

        inception_date = dt.datetime.fromtimestamp(inception_raw, tz=dt.UTC).date().isoformat()
    else:
        inception_date = _normalise_date(inception_raw)

    expense_raw = info.get("annualReportExpenseRatio")
    ongoing_charge = float(expense_raw) if expense_raw not in (None, 0, 0.0) else None
    security_type = _security_type(hit.get("investmentType"), hit.get("securityType"))
    currency = str(_first(info.get("currency"), hit.get("baseCurrency")) or "")
    name = str(_first(info.get("longName"), info.get("shortName"), hit.get("name"), isin))
    category = _first(
        info.get("category"),
        overview.get("categoryName"),
        _infer_equity_category(name, asset_classes),
    )
    family = _first(info.get("fundFamily"), overview.get("family"))
    sec_id = str(_first(hit.get("securityID"), hit.get("performanceID"), symbol))

    return FundMeta(
        isin=str(isin),
        sec_id=sec_id,
        name=name,
        currency=currency,
        domicile=None,
        category=category,
        benchmark_name=None,
        inception_date=inception_date,
        ongoing_charge=ongoing_charge,
        manager_tenure_years=None,
        security_type=security_type,
        raw={
            "provider": "yfinance",
            "search": hit,
            "quote": {"ticker": symbol},
            "info": info,
            "fund_overview": overview,
            "asset_classes": asset_classes,
            "fund_family": family,
        },
    )

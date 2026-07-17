"""Resolve a fund/ISIN/name to canonical metadata.

Uses mstarpy (Morningstar) as the primary provider. Resolution flow:

1. ``MorningstarSession.general_search`` on the raw input (ISIN or name) to
   locate the security and its provider ids.
2. ``Funds(isin, session=...)`` then ``.metaData()``, ``.quote(2)`` and
   ``.people()`` to populate the canonical :class:`FundMeta` fields.

Several fields are ``None`` on ``metaData()`` for some share classes but are
reliably present on ``quote(2)`` (ongoing charge, category name) or
``people()`` (inception date, manager tenure), so those endpoints are used as
fallbacks.
"""
from __future__ import annotations

import importlib
import re
import signal
import threading
from dataclasses import dataclass
from types import ModuleType
from typing import Any

_ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")

# Process-wide Morningstar session. Building a session performs a Selenium
# handshake to obtain WAF cookies, so it is expensive; reuse one instance.
_MSTARPY: ModuleType | None = None
_MSTARPY_LOCK = threading.RLock()
_SESSION: Any | None = None


def get_mstarpy() -> ModuleType:
    """Import and return ``mstarpy``, safely under Streamlit.

    ``mstarpy`` registers signal handlers at import time. Streamlit executes
    app scripts in a worker thread, where Python forbids ``signal.signal``.
    The CLI still imports normally on the main thread; UI-triggered imports
    temporarily guard ``signal.signal`` so mstarpy can finish importing.
    """
    global _MSTARPY
    if _MSTARPY is not None:
        return _MSTARPY

    with _MSTARPY_LOCK:
        if _MSTARPY is not None:
            return _MSTARPY

        if threading.current_thread() is threading.main_thread():
            _MSTARPY = importlib.import_module("mstarpy")
            return _MSTARPY

        real_signal = signal.signal

        def streamlit_safe_signal(sig, handler):
            try:
                return real_signal(sig, handler)
            except ValueError as exc:
                if "main thread" in str(exc):
                    return None
                raise

        signal.signal = streamlit_safe_signal
        try:
            _MSTARPY = importlib.import_module("mstarpy")
        finally:
            signal.signal = real_signal
        return _MSTARPY


def get_session() -> Any:
    """Return a lazily-constructed, process-wide :class:`MorningstarSession`."""
    global _SESSION
    if _SESSION is None:
        mstarpy = get_mstarpy()
        _SESSION = mstarpy.MorningstarSession()
    return _SESSION


@dataclass
class FundSearchResult:
    """Lightweight fund search result from Morningstar general_search."""

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
    """Search Morningstar for funds matching ``query``.

    This is intentionally lightweight for interactive UIs: it only calls
    ``general_search`` and does not instantiate ``mstarpy.Funds`` or fetch
    metadata/quotes/people for each candidate.
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

    isin = _first(hit.get("isin"), query if _looks_like_isin(query) else None)
    if not isin:
        raise LookupError(
            f"could not determine an ISIN for {isin_or_name!r}; "
            f"search hit keys: {sorted(hit.keys())}"
        )

    session = get_session()
    mstarpy = get_mstarpy()
    fund = mstarpy.Funds(isin, session=session)

    # metaData is the primary source; quote(2) and people fill gaps.
    try:
        meta = fund.metaData() or {}
    except Exception as exc:  # noqa: BLE001 - provider errors surfaced as LookupError
        raise LookupError(f"metaData() failed for {isin!r}: {exc}") from exc
    try:
        quote = fund.quote(2) or {}
    except Exception:  # noqa: BLE001 - quote is a best-effort fallback source
        quote = {}
    try:
        people = fund.people() or {}
    except Exception:  # noqa: BLE001 - people is a best-effort fallback source
        people = {}

    sec_id = _first(meta.get("secId"), quote.get("secId"), hit.get("securityID"))
    if not sec_id:
        raise LookupError(f"could not resolve a secId for {isin!r}")

    ongoing_raw = _first(meta.get("onGoingCharge"), quote.get("onGoingCharge"))
    ongoing_charge = float(ongoing_raw) / 100.0 if ongoing_raw is not None else None

    tenure_raw = _first(
        people.get("averageManagerTenure"), meta.get("averageManagerTenure")
    )
    manager_tenure_years = float(tenure_raw) if tenure_raw is not None else None

    return FundMeta(
        isin=str(_first(meta.get("isin"), isin)),
        sec_id=str(sec_id),
        name=str(_first(meta.get("name"), quote.get("investmentName"), hit.get("name"), isin)),
        currency=str(_first(meta.get("baseCurrencyId"), quote.get("currency"), hit.get("baseCurrency")) or ""),
        domicile=_first(meta.get("domicileCountryId"), quote.get("domicileCountryId")),
        category=_first(quote.get("categoryName"), meta.get("categoryName")),
        benchmark_name=_benchmark_name(meta),
        inception_date=_normalise_date(
            _first(people.get("inceptionDate"), meta.get("inceptionDate"))
        ),
        ongoing_charge=ongoing_charge,
        manager_tenure_years=manager_tenure_years,
        security_type=_security_type(
            meta.get("securityType"), quote.get("securityType"), hit.get("investmentType")
        ),
        raw={"search": hit, "metaData": meta, "quote": quote, "people": people},
    )

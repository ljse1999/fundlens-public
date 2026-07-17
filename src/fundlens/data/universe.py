"""Morningstar-backed fund universe discovery for screening workflows."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from fundlens.cache import DiskCache
from fundlens.config import get_settings
from fundlens.data.resolver import get_session

_CACHE_TTL_DAYS = 7

_INVESTMENT_TYPE_MAP = {
    "FO": "fund",
    "FV": "fund",
    "FM": "fund",
    "FE": "etf",
    "FC": "cef",
}

_DEFAULT_FIELDS = [
    "isin",
    "name",
    "shortName",
    "ticker",
    "investmentType",
    "baseCurrency",
    "currency",
    "exchangeCountry",
    "exchange",
    "exchangeName",
    "hasReport",
]

# Country prefixes for UK, Crown Dependencies, EU, EEA, Switzerland and
# closely associated European fund domiciles commonly seen in UCITS universes.
UK_EUROPE_ISIN_PREFIXES = {
    "AT",
    "BE",
    "BG",
    "CH",
    "CY",
    "CZ",
    "DE",
    "DK",
    "EE",
    "ES",
    "FI",
    "FR",
    "GB",
    "GG",
    "GR",
    "HR",
    "HU",
    "IE",
    "IM",
    "IS",
    "IT",
    "JE",
    "LI",
    "LT",
    "LU",
    "LV",
    "MT",
    "NL",
    "NO",
    "PL",
    "PT",
    "RO",
    "SE",
    "SI",
    "SK",
}

UK_EUROPE_EXCHANGE_COUNTRIES = {
    "AUT",
    "BEL",
    "BGR",
    "CHE",
    "CYP",
    "CZE",
    "DEU",
    "DNK",
    "ESP",
    "EST",
    "FIN",
    "FRA",
    "GBR",
    "GGY",
    "GRC",
    "HRV",
    "HUN",
    "IMN",
    "IRL",
    "ISL",
    "ITA",
    "JEY",
    "LIE",
    "LTU",
    "LUX",
    "LVA",
    "MLT",
    "NLD",
    "NOR",
    "POL",
    "PRT",
    "ROU",
    "SVK",
    "SVN",
    "SWE",
    "UNITED KINGDOM",
    "UK",
}


@dataclass(frozen=True)
class FundUniverseCandidate:
    """A lightweight candidate returned by Morningstar's public screener."""

    isin: str
    name: str
    ticker: str | None
    currency: str | None
    security_type: str
    exchange_country: str | None
    source: str
    raw: dict


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _row_value(row: dict) -> dict:
    """Normalise current and older mstarpy search row shapes."""
    value = row.get("value")
    if isinstance(value, dict):
        return value

    fields = row.get("fields")
    if isinstance(fields, dict):
        out = {}
        for key, payload in fields.items():
            if isinstance(payload, dict) and "value" in payload:
                out[key] = payload["value"]
            else:
                out[key] = payload
        meta = row.get("meta")
        if isinstance(meta, dict):
            out.update({k: v for k, v in meta.items() if k not in out})
        return out

    return {}


def _security_type(value: dict) -> str:
    investment_type = str(_first(value.get("investmentType"), value.get("universe")) or "").upper()
    return _INVESTMENT_TYPE_MAP.get(investment_type, "fund")


def _is_uk_europe(value: dict) -> bool:
    isin = str(_first(value.get("isin"), value.get("ISIN")) or "").upper()
    if len(isin) >= 2 and isin[:2] in UK_EUROPE_ISIN_PREFIXES:
        return True

    exchange_country = str(value.get("exchangeCountry") or "").upper()
    return bool(exchange_country and exchange_country in UK_EUROPE_EXCHANGE_COUNTRIES)


def _candidate_from_value(value: dict) -> FundUniverseCandidate | None:
    isin = _first(value.get("isin"), value.get("ISIN"))
    if not isin:
        return None

    name = str(_first(value.get("name"), value.get("shortName"), isin))
    return FundUniverseCandidate(
        isin=str(isin),
        name=name,
        ticker=_first(value.get("ticker"), value.get("symbol")),
        currency=_first(value.get("baseCurrency"), value.get("currency")),
        security_type=_security_type(value),
        exchange_country=value.get("exchangeCountry"),
        source="morningstar_screener",
        raw=value,
    )


def _investment_types(include_etfs: bool, include_cefs: bool) -> list[str]:
    types = ["FO"]
    if include_etfs:
        types.append("FE")
    if include_cefs:
        types.append("FC")
    return types


def _cache_key(
    *,
    term: str,
    include_etfs: bool,
    include_cefs: bool,
    page_size: int,
    max_pages: int,
    max_candidates: int | None,
) -> str:
    type_bits = "-".join(_investment_types(include_etfs, include_cefs))
    term_bit = term or "blank"
    cap_bit = "all" if max_candidates is None else str(max_candidates)
    return f"universe/morningstar/uk_europe/{term_bit}/{type_bits}/{page_size}/{max_pages}/{cap_bit}"


def _call_screener(
    session: Any,
    *,
    term: str,
    investment_type: str,
    page_size: int,
    page: int,
) -> list[dict]:
    filters = {"investmentType": investment_type}
    try:
        return session.screener_universe(
            term,
            field=_DEFAULT_FIELDS,
            filters=filters,
            pageSize=page_size,
            page=page,
            sortby="name",
            ascending=True,
        ) or []
    except ValueError:
        return session.screener_universe(
            term,
            field=["isin", "name"],
            filters=filters,
            pageSize=page_size,
            page=page,
            sortby="name",
            ascending=True,
        ) or []


def discover_fund_universe(
    *,
    term: str = "",
    include_etfs: bool = False,
    include_cefs: bool = False,
    page_size: int = 100,
    max_pages: int = 25,
    max_candidates: int | None = 250,
    session: Any | None = None,
    use_cache: bool = True,
    on_progress: Callable[[int, str, int], Any] | None = None,
) -> list[FundUniverseCandidate]:
    """Discover UK/European fund candidates from Morningstar's public screener.

    Morningstar's search endpoint currently ignores several documented exact
    filters such as domicile/category. This function therefore uses only the
    stable investment-type filter at source, then applies a transparent local
    UK/Europe test from ISIN country prefix or exchange-country code.
    """
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if max_pages <= 0:
        raise ValueError("max_pages must be positive")
    if max_candidates is not None and max_candidates <= 0:
        max_candidates = None

    cache = DiskCache(get_settings().cache_dir)
    key = _cache_key(
        term=term,
        include_etfs=include_etfs,
        include_cefs=include_cefs,
        page_size=page_size,
        max_pages=max_pages,
        max_candidates=max_candidates,
    )
    if use_cache:
        cached = cache.get_json(key, _CACHE_TTL_DAYS)
        if cached is not None:
            candidates = [FundUniverseCandidate(**item) for item in cached]
            return candidates[:max_candidates] if max_candidates is not None else candidates

    session = session or get_session()
    candidates: list[FundUniverseCandidate] = []
    seen: set[str] = set()

    for investment_type in _investment_types(include_etfs, include_cefs):
        for page in range(1, max_pages + 1):
            rows = _call_screener(
                session,
                term=term,
                investment_type=investment_type,
                page_size=page_size,
                page=page,
            )
            if not rows:
                break

            for row in rows:
                value = _row_value(row)
                if not value or not _is_uk_europe(value):
                    continue
                candidate = _candidate_from_value(value)
                if candidate is None or candidate.isin in seen:
                    continue
                seen.add(candidate.isin)
                candidates.append(candidate)

                if max_candidates is not None and len(candidates) >= max_candidates:
                    cache.put_json(key, [asdict(item) for item in candidates])
                    return candidates

            if on_progress is not None:
                on_progress(page, investment_type, len(candidates))

            if len(rows) < page_size:
                break

    cache.put_json(key, [asdict(item) for item in candidates])
    return candidates


def candidate_isins(candidates: list[FundUniverseCandidate]) -> list[str]:
    """Return unique ISINs from candidates, preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for candidate in candidates:
        if candidate.isin in seen:
            continue
        seen.add(candidate.isin)
        out.append(candidate.isin)
    return out

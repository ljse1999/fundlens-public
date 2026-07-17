"""Smoke test for the fundlens data layer against live providers.

Runs the full resolve -> returns -> holdings -> benchmark -> factors ->
risk-free chain for three validation ISINs and prints PASS/FAIL per section.
Exits 1 if any section fails.

Run:
    C:\\Python314\\python.exe scripts\\smoke_data.py
"""
from __future__ import annotations

import sys
import traceback

from fundlens.data.resolver import resolve_fund
from fundlens.data.navs import get_returns
from fundlens.data.holdings import get_fund_holdings
from fundlens.data.benchmarks import benchmark_proxy_for, get_benchmark_returns
from fundlens.data.factors import get_factors, region_for_category
from fundlens.data.fx import convert_factor_returns, get_risk_free

VALIDATION = [
    ("GB00B41YBW71", "Fundsmith Equity"),
    ("IE00BJSPMJ28", "Lindsell Train Global Equity"),
    ("GB00B3X7QG63", "Vanguard FTSE UK All Share"),
]

_MIN_OBS = 36


def section(name: str, results: list[tuple[str, bool, str]], func):
    try:
        detail = func()
        results.append((name, True, detail))
        print(f"  PASS {name}: {detail}")
    except Exception as exc:  # noqa: BLE001 - smoke records all failures
        results.append((name, False, f"{type(exc).__name__}: {exc}"))
        print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()


def run_fund(isin: str, label: str, results: list[tuple[str, bool, str]]) -> None:
    print(f"\n=== {label} [{isin}] ===")

    try:
        fund = resolve_fund(isin)
    except Exception as exc:  # noqa: BLE001
        msg = (
            f"RESOLUTION FAILURE for {isin} ({label}): {type(exc).__name__}: {exc}. "
            f"Tried general_search(q={isin!r}) then Funds(isin).metaData/quote/people."
        )
        print(f"  {msg}")
        results.append((f"{isin}:resolve", False, msg))
        return

    print(
        f"  resolved: name={fund.name!r} currency={fund.currency!r} "
        f"category={fund.category!r} benchmark={fund.benchmark_name!r} "
        f"ongoing_charge={fund.ongoing_charge} tenure={fund.manager_tenure_years} "
        f"inception={fund.inception_date} type={fund.security_type}"
    )
    results.append((f"{isin}:resolve", True, fund.name))

    def _returns():
        bundle = get_returns(fund)
        m = bundle.monthly
        if len(m) < _MIN_OBS:
            raise AssertionError(f"only {len(m)} monthly obs (<{_MIN_OBS})")
        return (
            f"n_obs={len(m)} range={m.index.min().date()}..{m.index.max().date()} "
            f"source={bundle.provenance['source']} type={bundle.provenance['series_type']}"
        )

    section(f"{isin}:returns", results, _returns)

    def _holdings():
        h = get_fund_holdings(fund)
        if len(h) == 0:
            raise AssertionError("empty holdings")
        wsum = float(h["weight"].sum())
        top5 = h[["name", "weight"]].head(5).to_dict("records")
        top5_str = ", ".join(f"{r['name']}={r['weight']:.3f}" for r in top5)
        return f"shape={h.shape} weight_sum={wsum:.3f} top5=[{top5_str}]"

    section(f"{isin}:holdings", results, _holdings)

    def _benchmark():
        ticker = benchmark_proxy_for(fund)
        br = get_benchmark_returns(ticker)
        if len(br) < _MIN_OBS:
            raise AssertionError(f"only {len(br)} benchmark obs for {ticker} (<{_MIN_OBS})")
        return f"proxy={ticker} n={len(br)} range={br.index.min().date()}..{br.index.max().date()}"

    section(f"{isin}:benchmark", results, _benchmark)

    def _factors():
        region = region_for_category(fund.category)
        usd = get_factors(region, "M")
        conv = convert_factor_returns(usd, fund.currency)
        factor_cols = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]
        overlap = conv.dropna(subset=factor_cols)
        if len(overlap) < _MIN_OBS:
            raise AssertionError(
                f"only {len(overlap)} rows with non-NaN converted factors (<{_MIN_OBS}); "
                "FX/factor alignment likely broken"
            )
        tail = conv[["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM", "RF"]].tail(3)
        return f"region={region} currency={fund.currency} cols={list(conv.columns)}\n{tail.to_string()}"

    section(f"{isin}:factors", results, _factors)

    def _risk_free():
        rf = get_risk_free(fund.currency)
        if len(rf) == 0:
            raise AssertionError("empty risk-free series")
        return f"currency={fund.currency}\n{rf.tail(3).to_string()}"

    section(f"{isin}:risk_free", results, _risk_free)


def main() -> int:
    results: list[tuple[str, bool, str]] = []
    for isin, label in VALIDATION:
        run_fund(isin, label, results)

    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n===== SUMMARY: {passed}/{len(results)} sections passed =====")
    failures = [(n, d) for n, ok, d in results if not ok]
    if failures:
        print("FAILURES:")
        for name, detail in failures:
            print(f"  - {name}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import pytest

from fundlens.data.benchmarks import benchmark_proxy_for, benchmark_proxy_match_for
from fundlens.data.resolver import FundMeta


def _fund(*, benchmark_name: str | None, category: str | None) -> FundMeta:
    return FundMeta(
        isin="GB0000000001",
        sec_id="TEST",
        name="Test Fund",
        currency="GBP",
        domicile="GB",
        category=category,
        benchmark_name=benchmark_name,
        inception_date=None,
        ongoing_charge=None,
        manager_tenure_years=None,
        security_type="fund",
        raw={},
    )


@pytest.mark.parametrize(
    ("benchmark_name", "expected"),
    [
        ("MSCI World Small Cap NR USD", "WLDS.L"),
        ("FTSE Developed Europe ex UK Index", "VERX.L"),
        ("MSCI Japan NR JPY", "CPJ1.L"),
        ("MSCI Pacific ex Japan NR USD", "CPXJ.L"),
        ("MSCI North America NR USD", "VNRT.L"),
        ("MSCI Emerging Markets NR USD", "EMIM.L"),
    ],
)
def test_benchmark_name_uses_most_specific_mapping(benchmark_name, expected):
    fund = _fund(benchmark_name=benchmark_name, category="Global Equity")
    assert benchmark_proxy_for(fund) == expected


def test_unknown_stated_benchmark_falls_back_to_morningstar_category():
    fund = _fund(benchmark_name="Bespoke Sustainable Transition Index", category="Global Equity")
    match = benchmark_proxy_match_for(fund)
    assert match is not None
    assert match.ticker == "SWDA.L"
    assert match.source == "category"


def test_global_smid_trial_resolves_to_world_small_cap_proxy():
    fund = _fund(
        benchmark_name="Morningstar Gbl SMID NR USD",
        category="EAA Fund Global Small/Mid-Cap Equity",
    )
    assert benchmark_proxy_for(fund) == "WLDS.L"


def test_unmapped_category_index_uses_global_smid_category():
    fund = _fund(
        benchmark_name="Unmapped Provider Category Index",
        category="EAA Fund Global Small/Mid-Cap Equity",
    )
    match = benchmark_proxy_match_for(fund)
    assert match is not None
    assert match.ticker == "WLDS.L"
    assert match.source == "category"


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("Global Emerging Markets Equity", "EMIM.L"),
        ("Japan Equity", "CPJ1.L"),
        ("Asia Pacific Excluding Japan Equity", "CPXJ.L"),
        ("North America Equity", "VNRT.L"),
        ("Europe ex UK Equity", "VERX.L"),
        ("UK All Companies Equity", "FTAL.L"),
        ("Global Large-Cap Blend Equity", "SWDA.L"),
    ],
)
def test_category_mapping_is_used_only_when_benchmark_is_missing(category, expected):
    fund = _fund(benchmark_name=None, category=category)
    assert benchmark_proxy_for(fund) == expected


def test_missing_benchmark_and_unmapped_category_has_no_proxy():
    fund = _fund(benchmark_name=None, category="Property Other")
    assert benchmark_proxy_for(fund) is None


def test_unknown_benchmark_and_unmapped_category_has_no_proxy():
    fund = _fund(benchmark_name="Bespoke Property Index", category="Property Other")
    assert benchmark_proxy_for(fund) is None

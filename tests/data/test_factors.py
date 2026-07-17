from __future__ import annotations

import pandas as pd
import pytest

from fundlens.data.factors import FACTOR_REGIONS, _daily_names, _to_daily_index, region_for_category


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        (None, "developed"),
        ("Global Large-Cap Blend Equity", "developed"),
        ("Developed World ex-US Equity", "developed_ex_us"),
        ("Europe ex-UK Equity", "europe"),
        ("Japan Large-Cap Equity", "japan"),
        ("Asia Pacific Excluding Japan Equity", "asia_pacific_ex_japan"),
        ("North America Equity", "north_america"),
        ("Canada Equity", "north_america"),
        ("Global Emerging Markets Equity", "emerging"),
        ("India Equity", "emerging"),
        ("US Large-Cap Growth Equity", "us"),
        ("American Equity", "us"),
    ],
)
def test_region_for_category_uses_most_specific_available_region(category, expected):
    assert region_for_category(category) == expected


def test_factor_regions_cover_all_published_regional_five_factor_sets():
    assert set(FACTOR_REGIONS) == {
        "developed",
        "developed_ex_us",
        "europe",
        "japan",
        "asia_pacific_ex_japan",
        "north_america",
        "emerging",
        "us",
    }


def test_daily_dataset_names_follow_library_capitalisation():
    assert _daily_names("japan") == ("Japan_5_Factors_Daily", "Japan_Mom_Factor_Daily")
    assert _daily_names("us") == (
        "F-F_Research_Data_5_Factors_2x3_daily",
        "F-F_Momentum_Factor_daily",
    )


def test_emerging_factors_are_monthly_only():
    with pytest.raises(ValueError, match="does not publish daily emerging"):
        _daily_names("emerging")


def test_daily_period_index_is_converted_for_modern_pandas():
    periods = pd.period_range("2026-01-01", periods=2, freq="D")
    result = _to_daily_index(periods)
    assert isinstance(result, pd.DatetimeIndex)
    assert list(result) == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")]

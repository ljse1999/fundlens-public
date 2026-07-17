"""Fetch Fama-French (+ momentum) style factor return series.

Data comes from Ken French's data library via
``pandas_datareader.DataReader(name, "famafrench")``. Percent returns are
converted to decimals, the momentum factor is merged into the 5-factor set,
and the result is cached on disk for 7 days.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd
import pandas_datareader.data as web

from fundlens.cache import DiskCache
from fundlens.config import get_settings

Region = Literal[
    "developed",
    "developed_ex_us",
    "europe",
    "japan",
    "asia_pacific_ex_japan",
    "north_america",
    "emerging",
    "us",
]

FACTOR_REGIONS: tuple[Region, ...] = (
    "developed",
    "developed_ex_us",
    "europe",
    "japan",
    "asia_pacific_ex_japan",
    "north_america",
    "emerging",
    "us",
)

_START = "1990-01-01"
_CACHE_TTL_DAYS = 7

# Monthly (base) dataset names per region: (five_factor, momentum).
_DATASETS: dict[str, tuple[str, str]] = {
    "developed": ("Developed_5_Factors", "Developed_Mom_Factor"),
    "developed_ex_us": ("Developed_ex_US_5_Factors", "Developed_ex_US_Mom_Factor"),
    "europe": ("Europe_5_Factors", "Europe_Mom_Factor"),
    "japan": ("Japan_5_Factors", "Japan_Mom_Factor"),
    "asia_pacific_ex_japan": (
        "Asia_Pacific_ex_Japan_5_Factors",
        "Asia_Pacific_ex_Japan_MOM_Factor",
    ),
    "north_america": ("North_America_5_Factors", "North_America_Mom_Factor"),
    "emerging": ("Emerging_5_Factors", "Emerging_MOM_Factor"),
    "us": ("F-F_Research_Data_5_Factors_2x3", "F-F_Momentum_Factor"),
}

_OUTPUT_COLUMNS = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM", "RF"]


def _to_month_end_index(idx: pd.Index) -> pd.DatetimeIndex:
    if isinstance(idx, pd.PeriodIndex):
        ts = idx.to_timestamp(how="end").normalize()
        return pd.DatetimeIndex(ts) + pd.offsets.MonthEnd(0)
    return pd.DatetimeIndex(pd.to_datetime(idx))


def _to_daily_index(idx: pd.Index) -> pd.DatetimeIndex:
    if isinstance(idx, pd.PeriodIndex):
        return pd.DatetimeIndex(idx.to_timestamp(how="start"))
    return pd.DatetimeIndex(pd.to_datetime(idx))


def _daily_names(region: str) -> tuple[str, str]:
    if region == "emerging":
        raise ValueError("Ken French does not publish daily emerging-market factors")
    five, mom = _DATASETS[region]
    if region == "us":
        # US five-factor daily is "F-F_Research_Data_5_Factors_2x3_daily";
        # US momentum daily is "F-F_Momentum_Factor_daily".
        return f"{five}_daily", f"{mom}_daily"
    return f"{five}_Daily", f"{mom}_Daily"


def _fetch(name: str) -> pd.DataFrame:
    """Fetch the primary table (key 0) for a famafrench dataset."""
    try:
        data = web.DataReader(name, "famafrench", start=_START)
    except Exception as exc:  # noqa: BLE001 - surface a clear message
        raise RuntimeError(f"failed to fetch Ken French dataset {name!r}: {exc}") from exc
    if 0 not in data:
        raise RuntimeError(f"Ken French dataset {name!r} has no primary table (key 0)")
    return data[0]


def get_factors(region: Region, freq: Literal["M", "D"] = "M") -> pd.DataFrame:
    """Fetch Fama-French 5-factor + momentum returns for ``region``.

    Args:
        region: Which regional factor set to fetch.
        freq: "M" for monthly (month-end indexed) or "D" for daily.

    Returns:
        A DataFrame indexed by DatetimeIndex (month-end when ``freq="M"``)
        with columns exactly ``["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM",
        "RF"]``, all as decimal (not percentage) period returns.
    """
    if region not in _DATASETS:
        raise ValueError(f"unknown region {region!r}; expected one of {sorted(_DATASETS)}")
    if freq not in ("M", "D"):
        raise ValueError(f"unknown freq {freq!r}; expected 'M' or 'D'")

    cache = DiskCache(get_settings().cache_dir)
    cache_key = f"factors/{region}/{freq}"
    cached = cache.get_df(cache_key, _CACHE_TTL_DAYS)
    if cached is not None:
        return cached

    if freq == "M":
        five_name, mom_name = _DATASETS[region]
    else:
        five_name, mom_name = _daily_names(region)

    five = _fetch(five_name)
    mom = _fetch(mom_name)

    # Percent -> decimal.
    five = five.astype(float) / 100.0
    mom = mom.astype(float) / 100.0

    # Momentum table has a single column; name varies ('WML' or 'Mom').
    mom_col = mom.iloc[:, 0].rename("MOM")

    df = five.rename(columns={"Mkt-RF": "MKT_RF"})
    df = df.join(mom_col, how="inner")

    missing = [c for c in _OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"famafrench dataset {five_name!r}/{mom_name!r} missing columns {missing}; "
            f"got {list(df.columns)}"
        )
    df = df[_OUTPUT_COLUMNS]

    if freq == "M":
        df.index = _to_month_end_index(df.index)
    else:
        df.index = _to_daily_index(df.index)
    df = df.sort_index()

    cache.put_df(cache_key, df)
    return df


def region_for_category(category: str | None) -> Region:
    """Map a fund category label to a factor :data:`Region`.

    Specific exclusions and regional phrases are checked before broader names;
    for example, ``Asia Pacific ex Japan`` must not be classified as Japan.
    Single-country categories use the nearest region that Ken French publishes.
    """
    if not category:
        return "developed"

    lowered = category.casefold()
    normalised = " ".join(lowered.replace("-", " ").replace("/", " ").split())

    developed_ex_us_keys = (
        "developed ex us",
        "developed ex usa",
        "world ex us",
        "world ex usa",
        "global ex us",
        "global ex usa",
    )
    if any(key in normalised for key in developed_ex_us_keys):
        return "developed_ex_us"

    emerging_keys = (
        "emerging",
        "bric",
        "china",
        "india",
        "latin america",
        "brazil",
        "mexico",
        "south africa",
        "south korea",
        "taiwan",
    )
    if any(key in normalised for key in emerging_keys):
        return "emerging"

    asia_pacific_keys = (
        "asia pacific ex japan",
        "asia pacific excluding japan",
        "asia ex japan",
        "asia excluding japan",
        "pacific ex japan",
        "pacific excluding japan",
        "asia pacific",
        "pacific basin",
        "australia",
        "new zealand",
        "hong kong",
        "singapore",
    )
    if any(key in normalised for key in asia_pacific_keys):
        return "asia_pacific_ex_japan"

    if "japan" in normalised:
        return "japan"

    if "north america" in normalised or "canada" in normalised:
        return "north_america"

    europe_keys = ("europe", "uk", "united kingdom", "eurozone")
    if any(key in normalised for key in europe_keys):
        return "europe"

    us_keys = ("united states", "usa", "american")
    if any(key in normalised for key in us_keys):
        return "us"
    # Guard "us" against matching inside unrelated words by requiring a token match.
    if "us" in set(normalised.split()):
        return "us"
    return "developed"

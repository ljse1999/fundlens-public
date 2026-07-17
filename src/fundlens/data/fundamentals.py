"""Best-effort fundamental data enrichment for a holdings table.

Enrichment is best-effort via yfinance ``Ticker.info`` for the top holdings
only. Any per-ticker failure yields NaN for that row; the function never raises.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

_TOP_N = 25
_SLEEP_SECONDS = 0.2


def _safe_float(value: object) -> float:
    try:
        if value is None:
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def enrich_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    """Best-effort enrichment of a holdings table with fundamental metrics.

    Adds ``pe`` (trailing P/E), ``growth`` (forward revenue/earnings growth,
    decimal) and ``roe`` (return on equity, decimal) columns using yfinance
    for the top ``_TOP_N`` holdings by weight. Missing values are NaN.
    """
    result = holdings.copy()
    for col in ("pe", "growth", "roe"):
        if col not in result.columns:
            result[col] = np.nan

    if len(result) == 0 or "ticker" not in result.columns:
        return result

    try:
        import yfinance as yf
    except Exception:  # noqa: BLE001 - enrichment is optional
        return result

    if "weight" in result.columns:
        order = result["weight"].fillna(0).sort_values(ascending=False).index
    else:
        order = result.index
    top_index = list(order)[:_TOP_N]

    for idx in top_index:
        ticker = result.at[idx, "ticker"]
        if ticker is None or (isinstance(ticker, float) and pd.isna(ticker)) or ticker == "":
            continue
        try:
            info = yf.Ticker(str(ticker)).info or {}
            pe = _safe_float(info.get("trailingPE"))
            growth = _safe_float(
                info.get("revenueGrowth")
                if info.get("revenueGrowth") is not None
                else info.get("earningsGrowth")
            )
            roe = _safe_float(info.get("returnOnEquity"))
            result.at[idx, "pe"] = pe
            result.at[idx, "growth"] = growth
            result.at[idx, "roe"] = roe
        except Exception:  # noqa: BLE001 - per-ticker best-effort
            pass
        time.sleep(_SLEEP_SECONDS)

    return result

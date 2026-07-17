"""Load committed IA snapshot data for the screen tab.

The screen tab can run against either (a) committed snapshot data shipped in the
repo, or (b) live network fetches. This module handles (a): it reads the cleaned
IA fund list and the pre-computed alpha-screen results, joins them on ISIN, and
returns a single DataFrame marked with a ``screened`` column.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

#: Columns kept from the fund-list snapshot (public, non-debugging fields).
FUND_COLUMNS = [
    "isin",
    "ia_fund_name",
    "ia_management_company",
    "ia_sector",
    "resolved_name",
    "status",
    "confidence",
    "triage_reason",
]

#: Columns kept from the alpha-screen snapshot.
SCREEN_COLUMNS = [
    "isin",
    "name",
    "alpha_ann",
    "alpha_t",
    "alpha_p_bootstrap",
    "alpha_verdict",
    "genuine_alpha",
    "n_obs",
    "active_share",
    "ongoing_charge",
    "flags",
]

#: Directory holding the committed snapshot files. Resolved relative to this
#: module so it works under any install mode (editable, wheel, Streamlit Cloud).
SNAPSHOT_DIR = Path(__file__).with_name("snapshot_data")


def load_ia_snapshot(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Load and join the IA fund list and alpha-screen snapshots.

    Args:
        data_dir: Directory containing ``ia_funds.csv`` and (optionally)
            ``ia_screen_results.parquet``. Defaults to the package-bundled
            ``snapshot_data/`` directory shipped inside the wheel, so callers
            don't need to compute repo paths (which differ across editable,
            wheel, and Streamlit Cloud installs).

    Returns:
        A DataFrame keyed by ``isin``. Every fund in ``ia_funds.csv`` appears
        exactly once. Funds present in the screen results have their alpha
        metrics filled in and ``screened=True``; the rest have ``screened=False``
        and null metrics.

    Raises:
        FileNotFoundError: if ``ia_funds.csv`` does not exist in ``data_dir``.
    """
    if data_dir is None:
        data_dir = SNAPSHOT_DIR
    data_dir = Path(data_dir)
    funds_path = data_dir / "ia_funds.csv"
    if not funds_path.exists():
        raise FileNotFoundError(f"IA fund list not found: {funds_path}")

    funds = pd.read_csv(funds_path)
    # Keep only the documented public columns that are present.
    funds = funds[[c for c in FUND_COLUMNS if c in funds.columns]]

    screen_path = data_dir / "ia_screen_results.parquet"
    if screen_path.exists():
        screen = pd.read_parquet(screen_path)
        screen = screen[[c for c in SCREEN_COLUMNS if c in screen.columns]]
        merged = funds.merge(screen, on="isin", how="left", suffixes=("", "_screen"))
        merged["screened"] = merged["alpha_verdict"].notna()
    else:
        merged = funds.copy()
        merged["screened"] = False
        for col in SCREEN_COLUMNS:
            if col != "isin" and col not in merged.columns:
                merged[col] = pd.NA

    return merged

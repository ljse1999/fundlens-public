"""Tests for the snapshot data loader."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fundlens.data.snapshot import load_ia_snapshot


def _make_fake_snapshot_dir(tmp_path: Path) -> Path:
    funds = pd.DataFrame(
        {
            "isin": ["GB0000000001", "GB0000000002", "GB0000000003"],
            "ia_fund_name": ["Fund A", "Fund B", "Fund C"],
            "ia_management_company": ["Mgr A", "Mgr B", "Mgr C"],
            "ia_sector": ["UK", "Europe", "Global"],
            "resolved_name": ["Fund A Inc", "Fund B Inc", "Fund C Inc"],
            "status": ["matched", "matched", "matched"],
            "confidence": [1.0, 1.0, 0.5],
            "triage_reason": ["", "", "ambiguous"],
        }
    )
    screen = pd.DataFrame(
        {
            "isin": ["GB0000000001", "GB0000000002"],
            "name": ["Fund A Inc", "Fund B Inc"],
            "alpha_ann": [0.02, -0.01],
            "alpha_t": [1.5, -0.8],
            "alpha_p_bootstrap": [0.05, 0.4],
            "alpha_verdict": ["Some alpha", "No detectable alpha"],
            "genuine_alpha": [True, False],
            "n_obs": [60, 60],
            "active_share": [0.9, 0.4],
            "ongoing_charge": [0.0075, 0.005],
            "flags": ["", ""],
            "ia_fund_name": ["Fund A", "Fund B"],
            "ia_management_company": ["Mgr A", "Mgr B"],
            "ia_sector": ["UK", "Europe"],
        }
    )
    funds.to_csv(tmp_path / "ia_funds.csv", index=False)
    screen.to_parquet(tmp_path / "ia_screen_results.parquet", index=False)
    return tmp_path


def test_load_ia_snapshot_joins_on_isin(tmp_path):
    data_dir = _make_fake_snapshot_dir(tmp_path)
    df = load_ia_snapshot(data_dir)
    # All three funds present (left join from ia_funds.csv).
    assert len(df) == 3
    assert set(df["isin"]) == {"GB0000000001", "GB0000000002", "GB0000000003"}
    # Screened flag distinguishes matched vs unmatched.
    assert not df.loc[df["isin"] == "GB0000000003", "screened"].iloc[0]
    assert df.loc[df["isin"] == "GB0000000001", "screened"].iloc[0]


def test_load_ia_snapshot_missing_screen_file_returns_unscreened(tmp_path):
    """If the parquet is absent, all funds are returned as unscreened."""
    funds = pd.DataFrame(
        {
            "isin": ["GB0000000001"],
            "ia_fund_name": ["Fund A"],
            "ia_management_company": ["Mgr A"],
            "ia_sector": ["UK"],
            "resolved_name": ["Fund A Inc"],
            "status": ["matched"],
            "confidence": [1.0],
            "triage_reason": [""],
        }
    )
    funds.to_csv(tmp_path / "ia_funds.csv", index=False)
    df = load_ia_snapshot(tmp_path)
    assert len(df) == 1
    assert not df["screened"].iloc[0]


def test_load_ia_snapshot_no_files_raises(tmp_path):
    """If neither file exists, raise a clear error."""
    with pytest.raises(FileNotFoundError):
        load_ia_snapshot(tmp_path)

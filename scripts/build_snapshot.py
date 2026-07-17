"""Build the committed IA snapshot for the public repo.

Reads the source repo's triaged resolution CSV and the alpha-screen checkpoint
CSV, cleans them down to public columns, and writes:

  - ``src/fundlens/data/snapshot_data/ia_funds.csv``               (fund universe)
  - ``src/fundlens/data/snapshot_data/ia_screen_results.parquet``  (alpha-screen results)

Run from the target repo (fundlens-public) after refreshing the source data:

    python scripts/build_snapshot.py \\
        --funds "C:/path/to/reports/ia_resolutions_20260708_triaged.csv" \\
        --screen "C:/path/to/reports/ia_alpha_screen_checkpoint_matched_500_w4.csv"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FUND_KEEP = [
    "ia_fund_name",
    "ia_management_company",
    "ia_sector",
    "isin",
    "resolved_name",
    "status",
    "confidence",
    "triage_reason",
]
SCREEN_KEEP = [
    "isin",
    "name",
    "category",
    "currency",
    "domicile",
    "security_type",
    "cagr",
    "information_ratio",
    "tracking_error_ann",
    "alpha_ann",
    "alpha_t",
    "alpha_p_bootstrap",
    "alpha_verdict",
    "genuine_alpha",
    "n_obs",
    "active_share",
    "ongoing_charge",
    "flags",
    "ia_fund_name",
    "ia_management_company",
    "ia_sector",
    "resolution_confidence",
    "resolution_status",
]


def build(funds_csv: Path, screen_csv: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    funds = pd.read_csv(funds_csv)
    funds_kept = [c for c in FUND_KEEP if c in funds.columns]
    funds_out = funds[funds_kept]
    funds_out.to_csv(out_dir / "ia_funds.csv", index=False)

    screen = pd.read_csv(screen_csv)
    screen_kept = [c for c in SCREEN_KEEP if c in screen.columns]
    screen_out = screen[screen_kept]
    screen_out.to_parquet(out_dir / "ia_screen_results.parquet", index=False)

    print(f"Wrote {len(funds_out)} funds -> {out_dir / 'ia_funds.csv'}")
    print(f"Wrote {len(screen_out)} screen results -> {out_dir / 'ia_screen_results.parquet'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--funds",
        type=Path,
        required=True,
        help="Path to ia_resolutions_*_triaged.csv in the source repo's reports/.",
    )
    p.add_argument(
        "--screen",
        type=Path,
        required=True,
        help="Path to ia_alpha_screen_checkpoint_*.csv in the source repo's reports/.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("src/fundlens/data/snapshot_data"),
        help="Output directory (default: src/fundlens/data/snapshot_data, so the "
        "snapshot ships inside the package and resolves under any install mode)",
    )
    args = p.parse_args()
    build(args.funds, args.screen, args.out)


if __name__ == "__main__":
    main()

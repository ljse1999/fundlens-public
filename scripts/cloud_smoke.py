"""Run the public app's critical analysis path in a clean cloud-like runtime."""
from __future__ import annotations

from fundlens.pipeline import analyse_fund


def main() -> None:
    result = analyse_fund("GB00B41YBW71")
    assert result["meta"]["name"] == "Fundsmith Equity I Acc"
    assert result["provenance"]["n_obs"] >= 120
    assert set(result["factor_fits"]) == {"capm", "ff3", "ff5", "ff5_mom"}
    assert result["holdings"] is not None and len(result["holdings"]) > 0
    assert result["errors"] == {}
    print(
        "CLOUD_SMOKE_OK",
        result["meta"]["name"],
        result["provenance"]["n_obs"],
        len(result["holdings"]),
    )


if __name__ == "__main__":
    main()

"""Phase-5 validation harness: run the pipeline on the validation funds and
print the metrics the sanity gates are judged on."""
from __future__ import annotations

import sys

from fundlens.pipeline import analyse_fund

FUNDS = [
    ("GB00B41YBW71", "Fundsmith Equity"),
    ("IE00BJSPMJ28", "Lindsell Train Global Equity"),
    ("GB00B3X7QG63", "Vanguard FTSE UK All Share (control)"),
]


def main() -> int:
    failures = 0
    for isin, label in FUNDS:
        print(f"\n=== {label} [{isin}] ===")
        try:
            r = analyse_fund(isin)
        except Exception as exc:  # noqa: BLE001
            print(f"  FATAL: {exc}")
            failures += 1
            continue
        perf = r.get("perf") or {}
        fits = r.get("factor_fits") or {}
        hs = r.get("holdings_stats") or {}
        conc = hs.get("concentration") or {}
        print(f"  window: {r['provenance'].get('start')}..{r['provenance'].get('end')}"
              f"  n_obs={perf.get('n_obs')}  benchmark_proxy={r['provenance'].get('benchmark_proxy')}")
        for model in ("capm", "ff5_mom"):
            f = fits.get(model)
            if f is None:
                print(f"  {model}: MISSING")
                continue
            beta_mkt = f.betas.get("MKT_RF")
            print(f"  {model}: alpha_ann={f.alpha_ann:+.2%} t={f.alpha_t:+.2f} "
                  f"p_boot={f.alpha_p_bootstrap} beta_mkt={beta_mkt:.3f} r2={f.r2:.3f}")
        print(f"  TE={perf.get('tracking_error_ann')}  IR={perf.get('information_ratio')}"
              f"  up/down capture={perf.get('up_capture')}/{perf.get('down_capture')}")
        print(f"  perf beta_vs_benchmark={perf.get('beta')}")
        print(f"  active_share={hs.get('active_share')}  coverage={conc.get('coverage')}"
              f"  top10={conc.get('top10_weight')}  eff_n={conc.get('effective_n')}")
        print(f"  style_drift={r.get('style_drift')}")
        flags = r.get("flags") or []
        print("  flags: " + (", ".join(f"{f['severity']}:{f['id']}" for f in flags) or "none"))
        errs = r.get("errors") or {}
        if errs:
            print("  errors: " + "; ".join(f"{k}: {v}" for k, v in errs.items()))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

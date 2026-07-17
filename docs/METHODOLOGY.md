# Methodology

*Adapted from the FundLens internal documentation.*

This document describes how FundLens computes returns, factors, alpha,
holdings analytics, and diligence flags. It is a research-desk methodology,
not a fund-rating product: it surfaces what's worth asking a manager, not a
verdict.

For a single ISIN, FundLens resolves the fund via Morningstar, pulls its
monthly total-return history, fits it against regional Fama-French factor
sets (converted into the fund's own currency), tests whether any alpha
survives that adjustment, tracks style drift over time, computes holdings-based
concentration and active share where disclosure allows, evaluates a fixed set
of rule-based diligence flags, and turns any flags that fire into specific
questions for a manager conversation.

## Returns

Monthly total-return NAVs come from Morningstar via mstarpy
(`Funds.nav(..., "monthly")`, `totalReturn` field — i.e. distributions
reinvested, not price-only). Empty/truncated NAV responses are retried (up to
3 attempts, 3s apart) since Morningstar intermittently returns an empty list
for a valid ISIN. If fewer than 36 monthly observations come back and the
fund/ETF has a listed ticker, returns fall back to yfinance adjusted-close
(price series, dividends reflected via adjustment, not a true reinvested
total return). A minimum of 24 monthly observations is required for the
pipeline to proceed at all.

## Factors

Regional Fama-French 5-factor + momentum sets (`MKT_RF`, `SMB`, `HML`, `RMW`,
`CMA`, `MOM`, `RF`) come from Ken French's data library via
`pandas_datareader`. The region (`developed`, `developed_ex_us`, `europe`,
`japan`, `asia_pacific_ex_japan`, `north_america`, `emerging`, or `us`) is
chosen from the fund's Morningstar category via specific-first substring
matching (`developed` is the default/global fallback), or can be overridden
in the UI. These are all regional five-factor-plus-momentum pairs currently
published in the library; emerging-market factors are monthly only.

Factors are USD-denominated at source and are FX-translated into the fund's
own share-class currency leg-wise, not with a single blanket formula.
`MKT_RF` is a long-only total return: it is re-based to its USD total return
(`MKT_RF_usd + RF_usd`), translated as a total return via
`(1 + r_total_usd) * (1 + fx_ret) - 1`, and then re-expressed as an excess
return over the *local* risk-free rate. The long-short factors (`SMB`, `HML`,
`RMW`, `CMA`, `MOM`) are zero-net-investment long-minus-short portfolios, for
which the exact currency-translation identity is `r_out = r_usd * (1 + fx_ret)`
(the cross term only) — applying the long-only formula to these would
incorrectly inject the full FX return into every style factor, creating
engineered collinearity between the FX return and every style beta. The
risk-free leg is replaced with a local series appropriate to the fund's
currency (SONIA for GBP, €STR with a 3M interbank fallback for EUR, US
1-month Treasury with a fed funds fallback for USD), sourced from FRED,
falling back to the FX-translated US risk-free rate if the local series is
unavailable.

## Regression

Excess returns (fund minus local risk-free) are regressed on each factor
model (CAPM, FF3, FF5, FF5+MOM) by OLS with Newey-West HAC standard errors
(lag count `floor(4 * (n/100)^(2/9))` by default). Annualised alpha compounds
the per-period intercept: `(1 + alpha)^12 - 1`. Alpha significance is reported
two ways: the HAC t-statistic, and a two-sided p-value from a 2,000-draw
circular block bootstrap of the regression residuals under the null of zero
alpha (block length `n^(1/3)`, refit under HAC at each draw) — this doesn't
rely on asymptotic normality and gives a second, more conservative read on
whether an apparent alpha is likely noise. Rolling 36-month FF3 betas are
also computed to track how the fund's factor exposure evolves.

## Alpha ladder

Reports label the existing factor result as **FF5+MOM alpha** rather than
plain "alpha". This controls broad academic equity factors, but it does not
neutralise sector, country, currency, or thematic exposures. Single-fund
reports also add **benchmark residual alpha**, an OLS regression of fund
excess return on the selected benchmark-proxy ETF's excess return. This is a
stricter sanity check against the stated/mapped benchmark, but it still
should not be read as fully ETF-replicated or holdings-neutral stock-selection
alpha.

## Universe alpha screen

The screen uses Morningstar's public search endpoint as a candidate source,
applies UK/Europe inclusion locally from ISIN country prefixes or
exchange-country codes, then resolves each candidate through the normal
metadata/returns stack. The screen path deliberately skips
holdings/style/report sections and fits only the FF5+MOM alpha model, so it
is practical for a candidate universe. A row is marked `genuine_alpha` only
when the FF5+MOM alpha t-stat is above 2.0 and the bootstrap p-value is below
0.05, matching the green `alpha_verdict` rule. Morningstar category filtering
is applied after resolution because the current public search endpoint does
not reliably honour category or domicile filters.

## Style analysis

Returns-based style analysis (Sharpe, 1992): over rolling 36-month windows,
fund returns are regressed on a basket of style-proxy ETF returns
(value/momentum/quality/small-cap/cash; different proxy sets for
`developed`/`global` vs. `europe`) subject to non-negative weights summing to
one, solved by constrained least squares (SLSQP). A style drift score is the
mean period-over-period total absolute change in style weights — a rough
measure of how much the RBSA-implied style mix moves around, independent of
the factor-model betas above.

## Active share and holdings analytics

Holdings are matched between fund and benchmark ETF in three cascading stages
— first by ISIN, then by ticker (exchange suffix stripped), then by
normalised name (casefolded, common corporate suffixes and share-class
markers removed) — with each stage's matches removed before the next runs.
Active share is `0.5 * sum(|w_fund - w_bench|)` over raw (unrescaled) weights
across the union of matched and unmatched positions. **This is a lower
bound** whenever either side's holdings disclosure is partial (weights
summing to less than 1.0): missing positions on one side simply drop out of
the union rather than contributing their true (unknown) weight difference.
Coverage (the sum of raw disclosed weights) is always computed and reported
alongside active share so this limitation is visible, not hidden.
Concentration metrics (top-5/top-10 weight, HHI, effective number of
holdings) and sector/country tilts are computed similarly, with tilts
renormalised to each side's own covered total so partial-coverage funds are
compared like-for-like rather than penalised for missing weight.

## Attribution

A factor-contribution decomposition splits the fund's annualised excess
return into each factor's beta times its annualised return over the fit
window, plus an alpha residual. This is an arithmetic approximation of what
is fundamentally a geometric (compounded) quantity — treat it as directional,
not an exact reconciliation. Brinson-Fachler sector/country attribution is
implemented but **not wired into the report**: it requires per-segment
realised returns for both fund and benchmark, which the pipeline does not
currently compute from holdings history.

## Diligence flags

Rules live in `src/fundlens/analysis/flags.py`; every threshold is gathered
in a single `THRESHOLDS` dict at the top of that file so they can be retuned
without touching rule logic. Each rule is defensive: if the inputs it needs
are missing (an earlier pipeline stage failed, or the fund lacks that kind of
data), the rule is silently skipped rather than raising. Thresholds below are
**first-pass calibrations**, validated only against three reference funds
(Fundsmith Equity, Lindsell Train Global Equity, and Vanguard FTSE UK All
Share as the passive control) — they are meant to be tuned as more funds are
run through the pipeline, not treated as settled.

| Flag | Severity | Fires when |
|---|---|---|
| `alpha_verdict` | red/green/info | Always evaluated when the FF5+MOM fit exists. `t < -1.3` -> amber "negative FF5+MOM alpha"; `t > 2.0` and bootstrap `p < 0.05` -> green "significant FF5+MOM alpha"; `t > 1.3` -> info "suggestive but inconclusive FF5+MOM alpha"; else info "no detectable FF5+MOM alpha". |
| `closet_indexer` | red/amber | Fee precondition `OCF > 0.5%` must hold. Then checks active share `< 60%` (only if holdings coverage `>= 80%`) and tracking error `< 3%`; both true -> red, one true -> amber. |
| `factor_explained` | amber | CAPM alpha t-stat `> 2.0` but FF5+MOM alpha t-stat `< 1.0` — apparent outperformance collapses once style tilts are controlled for. |
| `style_drift` | amber | RBSA style-weight turnover score `> 0.12`, or trailing-12m mean rolling FF3 beta has shifted `> 0.30` from the full-period beta on MKT_RF/SMB/HML. |
| `concentration` | amber/info | Effective number of holdings `< 15` -> amber "high concentration"; else top-10 weight `> 45%` -> info "high conviction positioning". |
| `capture_asymmetry` | amber | Down-market capture exceeds up-market capture by more than 5 percentage points. |
| `expensive_beta` | amber | OCF `> 1.0%` while tracking error `< 4%` — high fee for how little the fund departs from its benchmark. |
| `tenure_mismatch` | info | Current manager tenure `< 3` years but the analysed track record spans `> 5` years. |
| `holdings_coverage` | info | Published holdings cover `< 90%` of the portfolio — a reminder that holdings-based metrics above are partial. |
| `cheap_tracker_ok` | green | OCF `< 0.2%` and tracking error `< 2%` — a cheap, tight tracker doing what it says. |

## Caveats and data-quality notes

- **Backfilled share-class history.** Morningstar can backfill NAV history for
  newer share classes from an older, related one. Treat very long histories
  on newer share classes with some suspicion.
- **Partial holdings disclosure varies widely.** Some funds publish top-10
  holdings only, covering roughly half the portfolio. Active share, tilts,
  and concentration metrics degrade gracefully under this (see above) but are
  genuinely partial — always check the reported `coverage` figure before
  trusting them.
- **Benchmark returns are proxy-ETF returns, not index returns.** The
  benchmark side of tracking error, information ratio, and up/down capture is
  computed against a tradeable ETF (see `benchmark_map.yaml`), not the fund's
  stated benchmark index directly. This means benchmark-relative metrics
  include the ETF's own tracking noise and NAV-timing effects on top of the
  fund's true active risk.
- **Benchmark mapping is a hand-maintained substring table.**
  `benchmark_map.yaml` matches the fund's stated benchmark name against known
  index names and uses the longest case-insensitive match. Categories without
  a defensible configured proxy remain unmapped rather than falling back
  universally to a single global ETF.
- **Factor granularity is coarse.** Ken French's regional factor sets are
  academic, broad-market constructions. Japan has its own set, but other
  single-country funds map to the nearest broad region, which can still
  under-explain country-specific exposure.
- **Transient empty Morningstar responses are retried**, not silently
  accepted. Persistent empty responses still surface as a hard failure (fund
  resolution and the returns fetch are the only two pipeline stages that are
  fatal; everything else degrades to a recorded error and a partial report).
- **Disk cache TTLs** (parquet + JSON, keyed by logical path, each with a
  timestamp sidecar): factors 7 days, FX/risk-free 7 days, benchmark and
  style-proxy returns 1 day, holdings 30 days. Delete a cache file (or the
  whole cache directory) to force a refetch before the TTL expires.

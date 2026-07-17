"""Alpha ladder regression helpers.

The ladder keeps the existing Fama-French alpha intact and adds stricter
residual-alpha checks beside it. Phase 1/2 covers factor-fit rows plus a
single benchmark-proxy regression; later phases can add ETF replication steps
without changing the public result shape.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from fundlens.analysis.factor_model import (
    FactorFit,
    _align_xy,
    _bootstrap_alpha_p,
    _fit_hac,
    _infer_ppy,
)


@dataclass
class AlphaStep:
    """One rung in the alpha ladder."""

    id: str
    label: str
    alpha_ann: float
    alpha_t: float | None
    alpha_p_bootstrap: float | None
    r2: float | None
    resid_vol_ann: float | None
    nobs: int
    start: pd.Timestamp
    end: pd.Timestamp
    exposures: dict[str, float]
    notes: list[str]


@dataclass
class AlphaLadder:
    """Collection of alpha checks and a simple Phase 2 verdict."""

    steps: dict[str, AlphaStep]
    verdict: str
    warnings: list[str]
    selected_proxies: list[dict]


_FACTOR_STEP_LABELS = {
    "capm": "CAPM alpha",
    "ff3": "FF3 alpha",
    "ff5": "FF5 alpha",
    "ff5_mom": "FF5+MOM alpha",
}


def fit_exposure_model(
    fund_excess: pd.Series,
    exposures: pd.DataFrame | pd.Series,
    *,
    model_id: str,
    label: str,
    nw_lags: int | None = None,
    bootstrap: bool = True,
    bootstrap_draws: int = 2000,
    min_obs: int | None = None,
    notes: list[str] | None = None,
) -> AlphaStep:
    """Fit fund excess returns on arbitrary exposure return columns.

    This mirrors :func:`fundlens.analysis.factor_model.fit_factor_model`, but
    accepts caller-supplied regressors so benchmark and ETF-proxy models can
    share the same HAC/bootstrap inference style without destabilising the
    factor-model API.
    """
    if isinstance(exposures, pd.Series):
        exposures = exposures.to_frame()
    if exposures.empty or len(exposures.columns) == 0:
        raise ValueError("exposures must contain at least one regressor column")

    X_raw = exposures.copy()
    X_raw.columns = [str(col) for col in X_raw.columns]
    cols = list(X_raw.columns)
    y, X = _align_xy(fund_excess, X_raw, cols)

    k = len(cols)
    n = len(y)
    needed = max(24, k + 10) if min_obs is None else int(min_obs)
    if n < needed:
        raise ValueError(
            f"Insufficient observations for exposure model {model_id!r}: "
            f"got {n}, need at least {needed}."
        )

    res, lags = _fit_hac(y, X, nw_lags)
    ppy = _infer_ppy(y.index)

    alpha = float(res.params[0])
    alpha_t = float(res.tvalues[0]) if len(res.tvalues) else None
    exposure_betas = {c: float(res.params[i + 1]) for i, c in enumerate(cols)}
    resid_vol_ann = float(np.sqrt(res.scale) * np.sqrt(ppy))

    alpha_p = None
    if bootstrap and alpha_t is not None and np.isfinite(alpha_t):
        alpha_p = _bootstrap_alpha_p(
            y,
            X,
            lags,
            alpha_t,
            n_draws=bootstrap_draws,
        )

    return AlphaStep(
        id=model_id,
        label=label,
        alpha_ann=(1.0 + alpha) ** ppy - 1.0,
        alpha_t=alpha_t,
        alpha_p_bootstrap=alpha_p,
        r2=float(res.rsquared_adj),
        resid_vol_ann=resid_vol_ann,
        nobs=int(n),
        start=y.index[0],
        end=y.index[-1],
        exposures=exposure_betas,
        notes=list(notes or []),
    )


def fit_benchmark_residual_alpha(
    fund_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf: pd.Series,
    *,
    benchmark_ticker: str | None = None,
    bootstrap: bool = True,
    bootstrap_draws: int = 2000,
) -> AlphaStep:
    """Fit fund excess returns on benchmark-proxy excess returns."""
    frame = pd.concat(
        [
            fund_returns.rename("fund"),
            benchmark_returns.rename("benchmark"),
            rf.rename("rf"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if frame.empty:
        raise ValueError("fund, benchmark, and risk-free series have no overlapping observations")

    fund_excess = frame["fund"] - frame["rf"]
    benchmark_name = benchmark_ticker or "benchmark_proxy"
    benchmark_excess = pd.DataFrame(
        {f"{benchmark_name}_excess": frame["benchmark"] - frame["rf"]},
        index=frame.index,
    )
    notes = [
        "Benchmark residual alpha controls for the selected proxy only; "
        "sector, country, and theme exposures may still be omitted."
    ]
    if benchmark_ticker:
        notes.append(f"Benchmark proxy: {benchmark_ticker}.")
    return fit_exposure_model(
        fund_excess,
        benchmark_excess,
        model_id="benchmark",
        label="Benchmark residual alpha",
        bootstrap=bootstrap,
        bootstrap_draws=bootstrap_draws,
        notes=notes,
    )


def factor_fit_to_step(fit: FactorFit) -> AlphaStep:
    """Convert an existing factor-model fit into an alpha-ladder step."""
    return AlphaStep(
        id=fit.model,
        label=_FACTOR_STEP_LABELS.get(fit.model, f"{fit.model} alpha"),
        alpha_ann=fit.alpha_ann,
        alpha_t=fit.alpha_t,
        alpha_p_bootstrap=fit.alpha_p_bootstrap,
        r2=fit.r2,
        resid_vol_ann=fit.resid_vol_ann,
        nobs=fit.nobs,
        start=fit.start,
        end=fit.end,
        exposures=dict(fit.betas),
        notes=[],
    )


def build_alpha_ladder(
    factor_fits: dict[str, FactorFit],
    *,
    benchmark_step: AlphaStep | None = None,
    warnings: list[str] | None = None,
    selected_proxies: list[dict] | None = None,
) -> AlphaLadder:
    """Build the Phase 2 alpha ladder from available fits."""
    steps: dict[str, AlphaStep] = {}
    for model in ("capm", "ff3", "ff5", "ff5_mom"):
        fit = factor_fits.get(model)
        if fit is not None:
            steps[model] = factor_fit_to_step(fit)
    if benchmark_step is not None:
        steps[benchmark_step.id] = benchmark_step

    return AlphaLadder(
        steps=steps,
        verdict=_phase2_verdict(steps),
        warnings=list(warnings or []),
        selected_proxies=list(selected_proxies or []),
    )


def alpha_ladder_to_dict(ladder: AlphaLadder) -> dict:
    """Return a plain-dict representation suitable for pipeline results."""
    return {
        "steps": {key: asdict(step) for key, step in ladder.steps.items()},
        "verdict": ladder.verdict,
        "warnings": list(ladder.warnings),
        "selected_proxies": list(ladder.selected_proxies),
    }


def _phase2_verdict(steps: dict[str, AlphaStep]) -> str:
    ff5_mom = steps.get("ff5_mom")
    benchmark = steps.get("benchmark")
    if ff5_mom is None:
        return "not_evaluated"
    if not _is_significant(ff5_mom):
        return "inconclusive"
    if benchmark is None:
        return "ff5_mom_only"
    if _is_significant(benchmark):
        return "benchmark_resilient"
    return "benchmark_explained"


def _is_significant(step: AlphaStep) -> bool:
    if step.alpha_t is None or not np.isfinite(step.alpha_t) or step.alpha_t <= 2.0:
        return False
    if step.alpha_p_bootstrap is None:
        return True
    return step.alpha_p_bootstrap < 0.05

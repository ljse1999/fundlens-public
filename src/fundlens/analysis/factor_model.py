"""Fama-French style factor regression models.

Pure functions on pandas objects. Regressions are OLS of fund *excess*
returns on factor returns, with Newey-West (HAC) covariance for inference.

Conventions:

* **Excess returns**: ``fund_excess`` is already fund-minus-risk-free; the
  factors themselves (e.g. ``MKT_RF``) are excess/long-short by construction.
* **Annualisation**: the number of periods per year (``ppy``) is inferred from
  the index frequency (monthly -> 12, quarterly -> 4, daily -> 252),
  defaulting to 12. Annualised alpha compounds the per-period intercept:
  ``(1 + alpha) ** ppy - 1``. Residual volatility scales by ``sqrt(ppy)``.
* **HAC lags**: ``nw_lags`` defaults to ``floor(4 * (n / 100) ** (2 / 9))``.
* **R-squared** reported is the adjusted R-squared.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm

Model = Literal["capm", "ff3", "ff5", "ff5_mom"]

_MODEL_COLS: dict[str, list[str]] = {
    "capm": ["MKT_RF"],
    "ff3": ["MKT_RF", "SMB", "HML"],
    "ff5": ["MKT_RF", "SMB", "HML", "RMW", "CMA"],
    "ff5_mom": ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"],
}


@dataclass
class FactorFit:
    """Result of a single factor-model regression.

    Attributes:
        model: Which model was fit (see :data:`Model`).
        alpha_ann: Annualised alpha (decimal), compounded from the per-period
            intercept as ``(1 + intercept) ** ppy - 1``.
        alpha_t: HAC t-statistic of the intercept.
        alpha_p_bootstrap: Two-sided circular-block-bootstrap p-value for the
            alpha under the null ``alpha = 0`` (None if ``bootstrap=False``).
        betas: Factor name -> estimated coefficient.
        beta_t: Factor name -> HAC t-statistic.
        r2: Adjusted R-squared of the regression.
        resid_vol_ann: Annualised residual (idiosyncratic) volatility.
        nobs: Number of observations used.
        start: Timestamp of the first observation used.
        end: Timestamp of the last observation used.
    """

    model: str
    alpha_ann: float
    alpha_t: float
    alpha_p_bootstrap: float | None
    betas: dict[str, float]
    beta_t: dict[str, float]
    r2: float
    resid_vol_ann: float
    nobs: int
    start: pd.Timestamp
    end: pd.Timestamp


def _infer_ppy(index: pd.Index) -> int:
    """Infer periods-per-year from a DatetimeIndex frequency; default 12."""
    try:
        freq = pd.infer_freq(index)
    except (ValueError, TypeError):
        freq = None
    if freq is None:
        return 12
    f = freq.upper()
    if f.startswith(("D", "B")):
        return 252
    if f.startswith("W"):
        return 52
    if f.startswith(("Q",)):
        return 4
    if f.startswith(("A", "Y")):
        return 1
    if f.startswith("M") or "M" in f:
        return 12
    return 12


def _default_nw_lags(n: int) -> int:
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def _model_columns(model: str, factors: pd.DataFrame) -> list[str]:
    if model not in _MODEL_COLS:
        raise ValueError(
            f"Unknown model {model!r}; expected one of {sorted(_MODEL_COLS)}."
        )
    cols = _MODEL_COLS[model]
    missing = [c for c in cols if c not in factors.columns]
    if missing:
        raise ValueError(
            f"factors is missing required column(s) {missing} for model "
            f"{model!r}; available columns: {list(factors.columns)}."
        )
    return cols


def _align_xy(
    fund_excess: pd.Series, factors: pd.DataFrame, cols: list[str]
) -> tuple[pd.Series, pd.DataFrame]:
    frame = pd.concat([fund_excess.rename("_y"), factors[cols]], axis=1).dropna()
    return frame["_y"], frame[cols]


def _fit_hac(
    y: pd.Series, X: pd.DataFrame, nw_lags: int | None
) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, int]:
    n = len(y)
    lags = _default_nw_lags(n) if nw_lags is None else int(nw_lags)
    Xc = sm.add_constant(X, has_constant="add")
    res = sm.OLS(y.to_numpy(), Xc.to_numpy()).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    return res, lags


def _bootstrap_alpha_p(
    y: pd.Series,
    X: pd.DataFrame,
    nw_lags: int,
    t_obs: float,
    n_draws: int = 2000,
    seed: int = 20240517,
) -> float:
    """Two-sided p-value for alpha via circular block bootstrap of residuals.

    Under the null ``alpha = 0`` the response is rebuilt as the factor-only
    fitted values plus resampled residuals (circular block bootstrap), the
    full model (with intercept) is refit under HAC covariance, and the share
    of bootstrap intercept t-stats at least as extreme (in absolute value) as
    the observed one is returned.
    """
    n = len(y)
    Xc = sm.add_constant(X, has_constant="add").to_numpy()
    yv = y.to_numpy()
    # Full-fit residuals and factor-only (null) fitted values.
    beta_full = np.linalg.lstsq(Xc, yv, rcond=None)[0]
    resid = yv - Xc @ beta_full
    fitted_null = Xc @ np.concatenate([[0.0], beta_full[1:]])

    block = max(1, int(round(n ** (1.0 / 3.0))))
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)

    count = 0
    for _ in range(n_draws):
        starts = rng.integers(0, n, size=n_blocks)
        offsets = (starts[:, None] + np.arange(block)[None, :]) % n
        idx = offsets.reshape(-1)[:n]
        y_star = fitted_null + resid[idx]
        res_b = sm.OLS(y_star, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
        if abs(float(res_b.tvalues[0])) >= abs(t_obs):
            count += 1
    return count / n_draws


def fit_factor_model(
    fund_excess: pd.Series,
    factors: pd.DataFrame,
    model: Model = "ff5_mom",
    nw_lags: int | None = None,
    bootstrap: bool = True,
    bootstrap_draws: int = 2000,
) -> FactorFit:
    """Fit a factor regression of fund excess returns on the chosen model.

    Args:
        fund_excess: Decimal period excess returns indexed by DatetimeIndex.
        factors: DataFrame of factor returns aligned to ``fund_excess``.
        model: "capm" (MKT_RF), "ff3" (+SMB, HML), "ff5" (+RMW, CMA), or
            "ff5_mom" (+MOM).
        nw_lags: HAC lag count; None uses ``floor(4 * (n / 100) ** (2 / 9))``.
        bootstrap: Whether to compute the bootstrap alpha p-value.
        bootstrap_draws: Number of circular-block-bootstrap draws when
            ``bootstrap`` is true.

    Returns:
        A populated :class:`FactorFit`.

    Raises:
        ValueError: If ``model`` is unknown, a required factor column is
            missing, or fewer than ``max(24, k + 10)`` observations remain
            (``k`` = number of factors).
    """
    cols = _model_columns(model, factors)
    y, X = _align_xy(fund_excess, factors, cols)
    k = len(cols)
    n = len(y)
    min_obs = max(24, k + 10)
    if n < min_obs:
        raise ValueError(
            f"Insufficient observations for model {model!r}: got {n}, need at "
            f"least {min_obs} (max(24, k+10) with k={k})."
        )

    res, lags = _fit_hac(y, X, nw_lags)
    ppy = _infer_ppy(y.index)

    alpha = float(res.params[0])
    alpha_t = float(res.tvalues[0])
    betas = {c: float(res.params[i + 1]) for i, c in enumerate(cols)}
    beta_t = {c: float(res.tvalues[i + 1]) for i, c in enumerate(cols)}
    resid_vol_ann = float(np.sqrt(res.scale) * np.sqrt(ppy))

    alpha_p = None
    if bootstrap:
        alpha_p = _bootstrap_alpha_p(y, X, lags, alpha_t, n_draws=bootstrap_draws)

    return FactorFit(
        model=model,
        alpha_ann=(1.0 + alpha) ** ppy - 1.0,
        alpha_t=alpha_t,
        alpha_p_bootstrap=alpha_p,
        betas=betas,
        beta_t=beta_t,
        r2=float(res.rsquared_adj),
        resid_vol_ann=resid_vol_ann,
        nobs=int(n),
        start=y.index[0],
        end=y.index[-1],
    )


def rolling_betas(
    fund_excess: pd.Series,
    factors: pd.DataFrame,
    model: Model = "ff3",
    window: int = 36,
) -> pd.DataFrame:
    """Compute rolling-window OLS factor betas plus an annualised alpha.

    Args:
        fund_excess: Decimal period excess returns indexed by DatetimeIndex.
        factors: Factor return DataFrame aligned to ``fund_excess``.
        model: Which factor set to use (see :data:`Model`).
        window: Rolling window length in periods.

    Returns:
        A DataFrame indexed by window-end date with one column per factor beta
        (named as in the model) plus an ``alpha_ann`` column.
    """
    cols = _model_columns(model, factors)
    y, X = _align_xy(fund_excess, factors, cols)
    ppy = _infer_ppy(y.index)
    Xc = sm.add_constant(X, has_constant="add")
    Xv = Xc.to_numpy()
    yv = y.to_numpy()
    n = len(y)

    rows = []
    dates = []
    for end in range(window, n + 1):
        sl = slice(end - window, end)
        beta = np.linalg.lstsq(Xv[sl], yv[sl], rcond=None)[0]
        alpha_ann = (1.0 + beta[0]) ** ppy - 1.0
        rows.append([alpha_ann, *beta[1:]])
        dates.append(y.index[end - 1])

    return pd.DataFrame(rows, index=pd.Index(dates, name=y.index.name), columns=["alpha_ann", *cols])


def subperiod_alphas(
    fund_excess: pd.Series,
    factors: pd.DataFrame,
    model: Model = "ff5_mom",
    n_periods: int = 3,
) -> pd.DataFrame:
    """Split the sample into ``n_periods`` contiguous equal parts and fit each.

    Each sub-period is fit by OLS with HAC covariance (no minimum-observation
    guard and no bootstrap); one row is returned per sub-period.

    Args:
        fund_excess: Decimal period excess returns indexed by DatetimeIndex.
        factors: Factor return DataFrame aligned to ``fund_excess``.
        model: Which factor set to use (see :data:`Model`).
        n_periods: Number of contiguous, roughly equal-length sub-periods.

    Returns:
        A DataFrame with one row per sub-period and columns ``alpha_ann``,
        ``alpha_t``, ``r2``, ``nobs``, ``start``, ``end``.
    """
    cols = _model_columns(model, factors)
    y, X = _align_xy(fund_excess, factors, cols)
    ppy = _infer_ppy(y.index)
    n = len(y)
    bounds = np.linspace(0, n, n_periods + 1).round().astype(int)

    records = []
    for p in range(n_periods):
        lo, hi = int(bounds[p]), int(bounds[p + 1])
        ys, Xs = y.iloc[lo:hi], X.iloc[lo:hi]
        res, _ = _fit_hac(ys, Xs, None)
        alpha = float(res.params[0])
        records.append(
            {
                "alpha_ann": (1.0 + alpha) ** ppy - 1.0,
                "alpha_t": float(res.tvalues[0]),
                "r2": float(res.rsquared_adj),
                "nobs": int(len(ys)),
                "start": ys.index[0],
                "end": ys.index[-1],
            }
        )
    return pd.DataFrame.from_records(records)

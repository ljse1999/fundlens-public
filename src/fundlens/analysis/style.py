"""Returns-based style analysis (RBSA).

Pure functions on pandas objects. Implements Sharpe's (1992) constrained
returns-based style analysis: over each rolling window the fund return is
regressed on style-proxy returns subject to non-negative weights that sum to
one (a long-only, fully-invested portfolio of style proxies).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _solve_weights(r: np.ndarray, X: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Minimise ||r - X w||^2 s.t. w >= 0 and sum(w) == 1 via SLSQP."""
    k = X.shape[1]

    def obj(w: np.ndarray) -> float:
        resid = r - X @ w
        return float(resid @ resid)

    def grad(w: np.ndarray) -> np.ndarray:
        return -2.0 * X.T @ (r - X @ w)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * k
    res = minimize(
        obj,
        x0,
        jac=grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 500},
    )
    w = np.clip(res.x, 0.0, None)
    total = w.sum()
    return w / total if total > 0 else np.full(k, 1.0 / k)


def rbsa(fund_returns: pd.Series, proxies: pd.DataFrame, window: int = 36) -> pd.DataFrame:
    """Rolling constrained returns-based style analysis (Sharpe, 1992).

    For each trailing window of length ``window`` the constrained least-squares
    problem ``min ||r - X w||^2  s.t.  w >= 0, sum(w) = 1`` is solved by SLSQP.

    Args:
        fund_returns: Decimal period returns for the fund.
        proxies: DataFrame of style-proxy decimal period returns.
        window: Rolling window length in periods.

    Returns:
        A DataFrame indexed by window-end date with one column per proxy
        (matching ``proxies.columns``); every row is non-negative and sums to
        1 (tolerance 1e-6).
    """
    frame = pd.concat([fund_returns.rename("_y"), proxies], axis=1).dropna()
    y = frame["_y"].to_numpy()
    cols = list(proxies.columns)
    X = frame[cols].to_numpy()
    n = len(frame)
    k = len(cols)

    rows = []
    dates = []
    x0 = np.full(k, 1.0 / k)
    for end in range(window, n + 1):
        sl = slice(end - window, end)
        w = _solve_weights(y[sl], X[sl], x0)
        rows.append(w)
        dates.append(frame.index[end - 1])
        x0 = w  # warm-start next window

    return pd.DataFrame(
        rows, index=pd.Index(dates, name=frame.index.name), columns=cols
    )


def style_drift_score(weights: pd.DataFrame) -> float:
    """Average total style-weight turnover per step.

    Defined as ``weights.diff().abs().sum(axis=1).mean()``: the mean over time
    of the total absolute period-over-period change in style weights. Higher
    values indicate more style drift; a constant style mix scores ~0.

    Args:
        weights: A rolling style-weights DataFrame as returned by :func:`rbsa`.

    Returns:
        A single non-negative scalar drift score.
    """
    return float(weights.diff().abs().sum(axis=1).mean())

"""Reusable Plotly figure builders for fundlens report surfaces."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from fundlens.report.labels import factor_label

_LAYOUT_DEFAULTS = dict(template="plotly_white", height=380, margin=dict(l=50, r=20, t=50, b=40))


def unavailable(reason: str) -> dict:
    """Return a chart spec for an unavailable chart."""
    return {"available": False, "reason": reason}


def available(fig: go.Figure) -> dict:
    """Return a chart spec containing a Plotly figure."""
    return {"available": True, "figure": fig}


def _date(x) -> str:
    if x is None:
        return "n/a"
    try:
        return str(pd.Timestamp(x).date())
    except (TypeError, ValueError):
        return str(x)


def _chart_growth(data: dict) -> dict:
    series = data.get("series") or {}
    fund = series.get("fund_monthly")
    if fund is None or len(fund) == 0:
        return unavailable("fund return series unavailable")
    bench = series.get("benchmark_monthly")

    fig = go.Figure()
    growth = (1.0 + fund.dropna()).cumprod() * 100.0
    fig.add_trace(go.Scatter(x=growth.index, y=growth.values, name="Fund", mode="lines"))
    if bench is not None and len(bench):
        aligned = pd.concat([fund.rename("f"), bench.rename("b")], axis=1, join="inner").dropna()
        if len(aligned):
            g2 = (1.0 + aligned["b"]).cumprod() * 100.0
            fig.add_trace(go.Scatter(x=g2.index, y=g2.values, name="Benchmark", mode="lines"))
    fig.update_yaxes(type="log", title="Growth of 100 (log)")
    fig.update_layout(title="Cumulative growth", **_LAYOUT_DEFAULTS)
    return available(fig)


def _chart_drawdown(data: dict) -> dict:
    series = data.get("series") or {}
    dd = series.get("drawdown")
    if dd is None or len(dd) == 0:
        return unavailable("drawdown series unavailable")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values * 100.0,
            fill="tozeroy",
            name="Drawdown",
            line=dict(color="crimson"),
        )
    )
    fig.update_yaxes(title="Drawdown (%)")
    fig.update_layout(title="Drawdown", **_LAYOUT_DEFAULTS)
    return available(fig)


def _chart_rolling_excess(data: dict) -> dict:
    series = data.get("series") or {}
    ex = series.get("rolling_excess_12m")
    if ex is None or ex.dropna().empty:
        return unavailable("rolling excess return unavailable (needs a benchmark)")
    ex = ex.dropna()
    colors = ["seagreen" if v >= 0 else "crimson" for v in ex.values]
    fig = go.Figure(go.Bar(x=ex.index, y=ex.values * 100.0, marker_color=colors, name="12m excess"))
    fig.update_yaxes(title="Rolling 12m annualised excess return (%)")
    fig.update_layout(title="Rolling 12-month excess return vs benchmark", **_LAYOUT_DEFAULTS)
    return available(fig)


def _chart_rolling_betas(data: dict) -> dict:
    rb = data.get("rolling_betas")
    if rb is None or len(rb) == 0:
        return unavailable("rolling betas unavailable")
    fig = go.Figure()
    for col in ("MKT_RF", "SMB", "HML"):
        if col in rb.columns:
            fig.add_trace(
                go.Scatter(x=rb.index, y=rb[col], mode="lines", name=factor_label(col))
            )
    fig.update_yaxes(title="Rolling 36m beta")
    fig.update_layout(title="Rolling FF3 factor betas (36m window)", **_LAYOUT_DEFAULTS)
    return available(fig)


def _chart_subperiod(data: dict) -> dict:
    sp = data.get("subperiod")
    if sp is None or len(sp) == 0:
        return unavailable("sub-period alpha analysis unavailable")
    labels = [f"{_date(r['start'])} to {_date(r['end'])}" for _, r in sp.iterrows()]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=sp["alpha_ann"] * 100.0,
            text=[f"t={t:.2f}" for t in sp["alpha_t"]],
            textposition="outside",
            marker_color="steelblue",
        )
    )
    fig.update_yaxes(title="Annualised alpha (%)")
    fig.update_layout(title="Sub-period FF5+MOM alpha", **_LAYOUT_DEFAULTS)
    return available(fig)


def _chart_factor_loadings(data: dict) -> dict:
    fits = data.get("factor_fits") or {}
    if not fits:
        return unavailable("factor model fits unavailable")
    all_factors = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]
    factor_labels = [factor_label(f) for f in all_factors]
    fig = go.Figure()
    for model, fit in fits.items():
        ys = [fit.betas.get(f) for f in all_factors]
        texts = [f"t={fit.beta_t[f]:.1f}" if f in fit.beta_t else "" for f in all_factors]
        fig.add_trace(
            go.Bar(name=model, x=factor_labels, y=ys, text=texts, textposition="outside")
        )
    fig.update_layout(barmode="group", title="Factor loadings by model (t-stats annotated)", **_LAYOUT_DEFAULTS)
    return available(fig)


def _chart_factor_contrib(data: dict) -> dict:
    fc = data.get("factor_contrib")
    if fc is None or len(fc) == 0:
        return unavailable("factor contribution decomposition unavailable")
    fig = go.Figure(
        go.Bar(
            x=[factor_label(f) for f in fc.index],
            y=fc["contribution_ann"] * 100.0,
            marker_color="darkorange",
        )
    )
    fig.update_yaxes(title="Contribution to annualised return (%)")
    fig.update_layout(title="Factor contribution to annualised return", **_LAYOUT_DEFAULTS)
    return available(fig)


def _chart_rbsa(data: dict) -> dict:
    sw = data.get("style_weights")
    if sw is None or len(sw) == 0:
        return unavailable("returns-based style analysis unavailable")
    fig = go.Figure()
    for col in sw.columns:
        fig.add_trace(go.Scatter(x=sw.index, y=sw[col] * 100.0, mode="lines", stackgroup="one", name=col))
    fig.update_yaxes(title="Style weight (%)")
    drift = data.get("style_drift")
    title = "Rolling style weights (RBSA)"
    if drift is not None:
        title += f" -- drift score {drift:.3f}"
    fig.update_layout(title=title, **_LAYOUT_DEFAULTS)
    return available(fig)


def _chart_tilts(data: dict, by: str) -> dict:
    hs = data.get("holdings_stats") or {}
    df = hs.get(f"tilts_{by}")
    if df is None or len(df) == 0:
        return unavailable(f"{by} tilts unavailable (needs fund + benchmark holdings)")
    top = df.head(15).sort_values("active_weight")
    colors = ["seagreen" if v >= 0 else "crimson" for v in top["active_weight"]]
    fig = go.Figure(go.Bar(x=top["active_weight"] * 100.0, y=list(top.index), orientation="h", marker_color=colors))
    fig.update_xaxes(title="Active weight (%)")
    fig.update_layout(title=f"{by.title()} tilts vs benchmark", **_LAYOUT_DEFAULTS)
    return available(fig)


def _chart_top_holdings(data: dict) -> dict:
    holdings = data.get("holdings")
    if holdings is None or len(holdings) == 0:
        return unavailable("holdings unavailable")
    top = holdings.sort_values("weight", ascending=False).head(10).iloc[::-1]
    labels = top["name"].fillna(top.get("ticker")).fillna("(unknown)")
    fig = go.Figure(go.Bar(x=top["weight"] * 100.0, y=labels, orientation="h", marker_color="steelblue"))
    fig.update_xaxes(title="Weight (%)")
    hs = data.get("holdings_stats") or {}
    conc = hs.get("concentration") or {}
    coverage = conc.get("coverage")
    title = "Top 10 holdings"
    if coverage is not None:
        title += f" (published holdings cover {coverage * 100:.0f}% of the portfolio)"
    fig.update_layout(title=title, **_LAYOUT_DEFAULTS)
    return available(fig)


def build_chart_specs(data: dict) -> dict[str, dict]:
    """Build all report chart specs from a pipeline result dict."""
    return {
        "growth": _chart_growth(data),
        "drawdown": _chart_drawdown(data),
        "rolling_excess": _chart_rolling_excess(data),
        "rolling_betas": _chart_rolling_betas(data),
        "subperiod": _chart_subperiod(data),
        "factor_loadings": _chart_factor_loadings(data),
        "factor_contrib": _chart_factor_contrib(data),
        "rbsa": _chart_rbsa(data),
        "tilts_sector": _chart_tilts(data, "sector"),
        "tilts_country": _chart_tilts(data, "country"),
        "top_holdings": _chart_top_holdings(data),
    }

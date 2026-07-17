"""Tests for fundlens.report.builder.build_report."""
from __future__ import annotations

import re

import pandas as pd
import pytest

from fundlens.analysis.factor_model import FactorFit
from fundlens.report.builder import build_report
from fundlens.report.figures import build_chart_specs
from fundlens.report.view import build_report_view

# Detects un-rendered Jinja2 syntax that should never survive into the
# rendered HTML. Deliberately narrow (rather than a bare "{{" / "{%" scan)
# because the inlined Plotly.js bundle contains incidental "{%" substrings
# in unrelated minified string literals.
_PLACEHOLDER_RE = re.compile(
    r"\{\{\s*[\w.]+\s*\}\}|\{%-?\s*(if|for|endif|endfor|block|extends|include|set)\b"
)


def _fit(model, alpha_ann=0.03, alpha_t=1.8, betas=None, beta_t=None):
    betas = betas or {"MKT_RF": 0.9, "SMB": 0.1, "HML": -0.05}
    beta_t = beta_t or {"MKT_RF": 8.0, "SMB": 1.2, "HML": -0.4}
    return FactorFit(
        model=model,
        alpha_ann=alpha_ann,
        alpha_t=alpha_t,
        alpha_p_bootstrap=0.04,
        betas=betas,
        beta_t=beta_t,
        r2=0.85,
        resid_vol_ann=0.04,
        nobs=72,
        start=pd.Timestamp("2018-01-31"),
        end=pd.Timestamp("2023-12-31"),
    )


def _full_data() -> dict:
    dates = pd.date_range("2018-01-31", periods=72, freq="ME")
    fund = pd.Series(0.01, index=dates, name="return")
    bench = pd.Series(0.008, index=dates, name="bench")
    dd = pd.Series(-0.02, index=dates)
    rolling_ex = pd.Series(0.01, index=dates)

    rolling_betas = pd.DataFrame(
        {"alpha_ann": 0.02, "MKT_RF": 0.9, "SMB": 0.1, "HML": -0.05}, index=dates[35:]
    )
    subperiod = pd.DataFrame(
        [
            {"alpha_ann": 0.02, "alpha_t": 1.5, "r2": 0.8, "nobs": 24, "start": dates[0], "end": dates[23]},
            {"alpha_ann": 0.03, "alpha_t": 2.1, "r2": 0.82, "nobs": 24, "start": dates[24], "end": dates[47]},
            {"alpha_ann": 0.01, "alpha_t": 0.5, "r2": 0.79, "nobs": 24, "start": dates[48], "end": dates[71]},
        ]
    )
    style_weights = pd.DataFrame(
        {"value": 0.3, "momentum": 0.2, "quality": 0.3, "small_cap": 0.1, "cash": 0.1}, index=dates[35:]
    )

    holdings = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "isin": ["US0000000001", "US0000000002", "US0000000003"],
            "name": ["Alpha Corp", "Beta Inc", "Gamma Ltd"],
            "weight": [0.2, 0.15, 0.1],
            "sector": ["Tech", "Health", "Tech"],
            "country": ["US", "US", "GB"],
            "market_cap": [None, None, None],
        }
    )
    tilts_sector = pd.DataFrame(
        {"fund_weight": [0.5, 0.5], "bench_weight": [0.3, 0.7], "active_weight": [0.2, -0.2]},
        index=["Tech", "Health"],
    )
    tilts_country = pd.DataFrame(
        {"fund_weight": [0.6, 0.4], "bench_weight": [0.5, 0.5], "active_weight": [0.1, -0.1]},
        index=["US", "GB"],
    )
    factor_contrib = pd.DataFrame(
        {"beta": [0.9, 0.1], "factor_return_ann": [0.08, 0.02], "contribution_ann": [0.072, 0.002]},
        index=["MKT_RF", "alpha"],
    )

    flags = [
        {
            "id": "alpha_verdict",
            "severity": "green",
            "title": "Statistically significant FF5+MOM alpha",
            "detail": "FF5+MOM alpha is 3.0% annualised (t=2.10, p=0.040).",
            "metrics": {"alpha_ann": 0.03, "alpha_t": 2.1},
        },
        {
            "id": "concentration",
            "severity": "amber",
            "title": "High portfolio concentration",
            "detail": "Effective number of holdings is 8.0.",
            "metrics": {"top10_weight": 0.5, "effective_n": 8.0},
        },
    ]
    questions = [
        {"flag_id": "concentration", "topic": "Position sizing and liquidity", "question": "How large can a single position get before it triggers a review?"},
    ]

    return {
        "meta": {
            "isin": "GB00B41YBW71",
            "name": "Test Global Growth Fund",
            "currency": "GBP",
            "category": "Global Large-Cap Blend Equity",
            "benchmark_name": "MSCI World",
            "ongoing_charge": 0.0075,
            "manager_tenure_years": 6.5,
        },
        "perf": {
            "cagr": 0.09,
            "vol_ann": 0.14,
            "sharpe": 0.6,
            "sortino": 0.8,
            "max_drawdown": -0.22,
            "tracking_error_ann": 0.03,
            "information_ratio": 0.4,
            "up_capture": 1.05,
            "down_capture": 0.95,
            "hit_rate": 0.55,
        },
        "factor_fits": {
            "capm": _fit("capm"),
            "ff3": _fit("ff3"),
            "ff5": _fit("ff5"),
            "ff5_mom": _fit("ff5_mom"),
        },
        "alpha_ladder": {
            "steps": {
                "capm": {
                    "id": "capm",
                    "label": "CAPM alpha",
                    "alpha_ann": 0.03,
                    "alpha_t": 1.8,
                    "alpha_p_bootstrap": 0.04,
                    "r2": 0.85,
                    "resid_vol_ann": 0.04,
                    "nobs": 72,
                    "start": pd.Timestamp("2018-01-31"),
                    "end": pd.Timestamp("2023-12-31"),
                    "exposures": {"MKT_RF": 0.9},
                    "notes": [],
                },
                "ff5_mom": {
                    "id": "ff5_mom",
                    "label": "FF5+MOM alpha",
                    "alpha_ann": 0.03,
                    "alpha_t": 1.8,
                    "alpha_p_bootstrap": 0.04,
                    "r2": 0.85,
                    "resid_vol_ann": 0.04,
                    "nobs": 72,
                    "start": pd.Timestamp("2018-01-31"),
                    "end": pd.Timestamp("2023-12-31"),
                    "exposures": {"MKT_RF": 0.9},
                    "notes": [],
                },
                "benchmark": {
                    "id": "benchmark",
                    "label": "Benchmark residual alpha",
                    "alpha_ann": 0.006,
                    "alpha_t": 0.7,
                    "alpha_p_bootstrap": 0.45,
                    "r2": 0.9,
                    "resid_vol_ann": 0.03,
                    "nobs": 72,
                    "start": pd.Timestamp("2018-01-31"),
                    "end": pd.Timestamp("2023-12-31"),
                    "exposures": {"SWDA.L_excess": 1.1},
                    "notes": ["Benchmark proxy: SWDA.L."],
                },
            },
            "verdict": "benchmark_explained",
            "warnings": [],
            "selected_proxies": [
                {
                    "id": "benchmark",
                    "ticker": "SWDA.L",
                    "reason": "stated benchmark proxy",
                    "source": "benchmark_map",
                }
            ],
        },
        "rolling_betas": rolling_betas,
        "subperiod": subperiod,
        "style_weights": style_weights,
        "style_drift": 0.04,
        "holdings": holdings,
        "holdings_stats": {
            "active_share": 0.55,
            "concentration": {"top10_weight": 0.45, "effective_n": 12.0, "coverage": 0.92},
            "tilts_sector": tilts_sector,
            "tilts_country": tilts_country,
        },
        "factor_contrib": factor_contrib,
        "flags": flags,
        "questions": questions,
        "series": {
            "fund_monthly": fund,
            "benchmark_monthly": bench,
            "drawdown": dd,
            "rolling_excess_12m": rolling_ex,
        },
        "provenance": {
            "date_range": ["2018-01-31", "2023-12-31"],
            "n_obs": 72,
            "currency": "GBP",
            "factor_region": "developed",
            "benchmark_proxy": "SWDA.L",
            "notes": ["Morningstar may backfill older share-class history."],
        },
        "errors": {},
    }


def test_build_report_golden_path(tmp_path):
    data = _full_data()
    out_path = tmp_path / "report.html"
    result_path = build_report(data, out_path)

    assert result_path == out_path
    assert out_path.exists()
    html = out_path.read_text(encoding="utf-8")

    assert "Test Global Growth Fund" in html
    assert "Alpha ladder" in html
    assert "Benchmark residual alpha" in html
    assert "FF5+MOM alpha controls broad academic equity factors" in html
    assert "<script>" in html and "Plotly" in html
    assert "How large can a single position get before it triggers a review?" in html
    assert not _PLACEHOLDER_RE.search(html)


def test_build_report_degraded_path_still_writes_valid_html(tmp_path):
    data = {
        "meta": {"isin": "GB00B41YBW71", "name": "Broken Fund", "currency": "GBP"},
        "errors": {
            "benchmark": "yfinance returned no data for benchmark ticker 'XXXX.L'",
            "holdings": "holdings('equity') failed: timeout",
        },
        "series": {},
    }
    out_path = tmp_path / "degraded.html"
    build_report(data, out_path)

    assert out_path.exists()
    html = out_path.read_text(encoding="utf-8")
    assert "Broken Fund" in html
    assert "alpha ladder could not be computed" in html
    assert "unavailable" in html.lower()
    assert not _PLACEHOLDER_RE.search(html)


def test_report_view_and_figures_are_reusable():
    data = _full_data()

    view = build_report_view(data)
    charts = build_chart_specs(data)

    assert view["header"]["name"] == "Test Global Growth Fund"
    assert view["flag_counts"] == {"red": 0, "amber": 1, "green": 1, "info": 0}
    assert any(row["metric"] == "CAGR" and row["value"] == "9.0%" for row in view["perf_table"])
    assert any(row["step"] == "Benchmark residual alpha" for row in view["alpha_ladder"]["rows"])
    assert "Position sizing and liquidity" in view["questions_by_topic"]

    assert charts["growth"]["available"] is True
    assert charts["growth"]["figure"].layout.title.text == "Cumulative growth"
    assert list(charts["factor_loadings"]["figure"].data[0].x) == [
        "Market",
        "Size",
        "Value",
        "Profitability",
        "Investment",
        "Momentum",
    ]
    assert [trace.name for trace in charts["rolling_betas"]["figure"].data] == [
        "Market",
        "Size",
        "Value",
    ]
    assert list(charts["factor_contrib"]["figure"].data[0].x) == ["Market", "Alpha"]
    assert charts["top_holdings"]["available"] is True


def test_report_view_relabels_factor_codes_in_display_text():
    data = {
        "flags": [
            {
                "id": "style_drift",
                "severity": "amber",
                "title": "Meaningful style drift",
                "detail": (
                    "Rolling beta shifted on: MKT_RF (Delta=0.40), "
                    "SMB (Delta=0.35)."
                ),
                "metrics": {"shifted_betas": {"MKT_RF": 0.4, "SMB": 0.35}},
            }
        ],
        "questions": [
            {
                "flag_id": "style_drift",
                "topic": "Style consistency",
                "question": "Recent factor exposure on MKT_RF and SMB has moved.",
            }
        ],
    }

    view = build_report_view(data)

    assert view["flags"][0]["detail"] == (
        "Rolling beta shifted on: Market (Delta=0.40), Size (Delta=0.35)."
    )
    question = view["questions_by_topic"]["Style consistency"][0]["question"]
    assert question == "Recent factor exposure on Market and Size has moved."


def test_build_report_renders_enhanced_dd_agenda(tmp_path):
    data = _full_data()
    data["dd_agenda"] = {
        "summary": "Probe whether concentration is intentional and liquid.",
        "anomalies": [
            {
                "priority": "high",
                "framework_section": "Portfolio construction",
                "title": "High portfolio concentration",
                "evidence": "Effective number of holdings is 8.0.",
                "why_it_matters": "Position size must connect to risk and liquidity.",
            }
        ],
        "priority_questions": [
            {
                "priority": "high",
                "topic": "Position sizing and liquidity",
                "question": "How would the largest position be unwound in stress?",
                "evidence_used": ["concentration"],
                "follow_up_if_evasive": "Show the liquidity ladder and decision log.",
                "evidence_request": "Provide position-size limits and liquidity buckets.",
            }
        ],
        "evidence_requests": [
            {
                "priority": "high",
                "request": "Provide liquidity buckets for the top ten positions.",
                "linked_topic": "Position sizing and liquidity",
            }
        ],
        "data_gaps": ["Full holdings history unavailable."],
        "tripwires": [
            {
                "metric": "Top-10 concentration",
                "proposed_level": "Outside manager-stated band.",
                "breach_action": "Escalation call.",
                "rationale": "Concentration is a core underwriting assumption.",
            }
        ],
    }

    out_path = tmp_path / "agenda.html"
    build_report(data, out_path)

    html = out_path.read_text(encoding="utf-8")
    assert "Manager DD agenda" in html
    assert "Probe whether concentration is intentional and liquid." in html
    assert "How would the largest position be unwound in stress?" in html
    assert "Questions for the manager" not in html
    assert not _PLACEHOLDER_RE.search(html)

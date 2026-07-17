"""Pipeline tests for deterministic and Codex question modes."""
from __future__ import annotations

import pandas as pd
import pytest

from fundlens import pipeline
from fundlens.analysis.alpha_ladder import AlphaStep
from fundlens.analysis.flags import Flag
from fundlens.analysis.factor_model import FactorFit
from fundlens.data.navs import ReturnsBundle
from fundlens.data.resolver import FundMeta


def _raise(message: str):
    raise RuntimeError(message)


def _patch_minimal_pipeline(monkeypatch, codex_apply=None) -> None:
    dates = pd.date_range("2021-01-31", periods=24, freq="ME")
    monthly = pd.Series(0.01, index=dates, name="return")
    fund = FundMeta(
        isin="GB00TEST1234",
        sec_id="SECID",
        name="Test Fund",
        currency="GBP",
        domicile="GB",
        category="Global Equity",
        benchmark_name="MSCI World",
        inception_date="2020-01-01",
        ongoing_charge=0.01,
        manager_tenure_years=2.0,
        security_type="fund",
        raw={},
    )

    monkeypatch.setattr(pipeline, "resolve_fund", lambda isin: fund)
    monkeypatch.setattr(
        pipeline,
        "get_returns",
        lambda fund: ReturnsBundle(
            daily=None,
            monthly=monthly,
            currency="GBP",
            provenance={"source": "test"},
        ),
    )
    monkeypatch.setattr(pipeline, "benchmark_proxy_for", lambda fund: _raise("no benchmark"))
    monkeypatch.setattr(pipeline, "region_for_category", lambda category: "developed")
    monkeypatch.setattr(pipeline, "get_factors", lambda region, freq: _raise("no factors"))
    monkeypatch.setattr(pipeline, "get_risk_free", lambda currency: _raise("no risk-free"))
    monkeypatch.setattr(
        pipeline,
        "perf_summary",
        lambda monthly, benchmark_returns, rf: {
            "tracking_error_ann": 0.02,
            "information_ratio": 0.3,
        },
    )
    monkeypatch.setattr(pipeline, "drawdown_series", lambda monthly: monthly.cumsum())
    monkeypatch.setattr(pipeline, "get_style_proxies", lambda region: _raise("no style"))
    monkeypatch.setattr(pipeline, "get_fund_holdings", lambda fund: _raise("no holdings"))
    monkeypatch.setattr(
        pipeline,
        "evaluate_flags",
        lambda result: [
            Flag(
                id="expensive_beta",
                severity="amber",
                title="Expensive for the beta on offer",
                detail="Fee looks high relative to active risk.",
                metrics={"ongoing_charge": 0.01, "tracking_error_ann": 0.02},
            )
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "questions_for",
        lambda flags: [
            {
                "flag_id": "expensive_beta",
                "topic": "Fee justification",
                "question": "How is this fee justified to investors?",
            }
        ],
    )
    if codex_apply is not None:
        monkeypatch.setattr(pipeline, "apply_codex_dd_agenda", codex_apply)


def test_analyse_fund_deterministic_question_mode_preserves_current_behavior(monkeypatch):
    def fail_if_called(result):
        pytest.fail("Codex agenda should not run in deterministic mode")

    _patch_minimal_pipeline(monkeypatch, codex_apply=fail_if_called)

    result = pipeline.analyse_fund("GB00TEST1234", question_mode="deterministic")

    assert result["questions"][0]["topic"] == "Fee justification"
    assert "dd_agenda" not in result
    assert result["question_generation"] == {
        "mode": "deterministic",
        "source": "rules",
        "status": "ok",
        "fallback_used": False,
    }


def test_analyse_fund_codex_question_mode_adds_agenda(monkeypatch):
    def fake_apply(result):
        result["dd_agenda"] = {
            "summary": "Enhanced agenda",
            "anomalies": [],
            "priority_questions": [],
            "evidence_requests": [],
            "data_gaps": [],
            "tripwires": [],
        }
        result["question_generation"] = {
            "mode": "codex",
            "source": "codex_exec",
            "status": "ok",
            "fallback_used": False,
        }
        return result

    _patch_minimal_pipeline(monkeypatch, codex_apply=fake_apply)

    result = pipeline.analyse_fund("GB00TEST1234", question_mode="codex")

    assert result["questions"][0]["topic"] == "Fee justification"
    assert result["dd_agenda"]["summary"] == "Enhanced agenda"
    assert result["question_generation"]["status"] == "ok"


def test_analyse_fund_codex_question_mode_falls_back_cleanly(monkeypatch):
    def fake_apply(result):
        result["question_generation"] = {
            "mode": "codex",
            "source": "codex_exec",
            "status": "fallback",
            "fallback_used": True,
            "error": "not logged in",
        }
        result.setdefault("errors", {})["questions_codex"] = "not logged in"
        return result

    _patch_minimal_pipeline(monkeypatch, codex_apply=fake_apply)

    result = pipeline.analyse_fund("GB00TEST1234", question_mode="codex")

    assert "dd_agenda" not in result
    assert result["questions"][0]["question"] == "How is this fee justified to investors?"
    assert result["question_generation"]["status"] == "fallback"
    assert result["errors"]["questions_codex"] == "not logged in"


def test_analyse_fund_stores_alpha_ladder_with_benchmark_step(monkeypatch):
    dates = pd.date_range("2021-01-31", periods=30, freq="ME")
    monthly = pd.Series(0.01, index=dates, name="return")
    benchmark = pd.Series(0.008, index=dates, name="benchmark")
    factors = pd.DataFrame(
        {
            "MKT_RF": 0.006,
            "SMB": 0.0,
            "HML": 0.0,
            "RMW": 0.0,
            "CMA": 0.0,
            "MOM": 0.0,
            "RF": 0.001,
        },
        index=dates,
    )
    fund = FundMeta(
        isin="GB00TEST1234",
        sec_id="SECID",
        name="Test Fund",
        currency="GBP",
        domicile="GB",
        category="Global Equity",
        benchmark_name="MSCI World",
        inception_date="2020-01-01",
        ongoing_charge=0.01,
        manager_tenure_years=2.0,
        security_type="fund",
        raw={},
    )

    def fake_fit(_fund_excess, _factors, model="ff5_mom", **_kwargs):
        return FactorFit(
            model=model,
            alpha_ann=0.03,
            alpha_t=2.2,
            alpha_p_bootstrap=0.01,
            betas={"MKT_RF": 1.0},
            beta_t={"MKT_RF": 5.0},
            r2=0.8,
            resid_vol_ann=0.05,
            nobs=30,
            start=dates[0],
            end=dates[-1],
        )

    def fake_benchmark_step(*_args, **_kwargs):
        return AlphaStep(
            id="benchmark",
            label="Benchmark residual alpha",
            alpha_ann=0.004,
            alpha_t=0.6,
            alpha_p_bootstrap=0.55,
            r2=0.9,
            resid_vol_ann=0.03,
            nobs=30,
            start=dates[0],
            end=dates[-1],
            exposures={"SWDA.L_excess": 1.0},
            notes=[],
        )

    monkeypatch.setattr(pipeline, "resolve_fund", lambda isin: fund)
    monkeypatch.setattr(
        pipeline,
        "get_returns",
        lambda fund: ReturnsBundle(daily=None, monthly=monthly, currency="GBP", provenance={"source": "test"}),
    )
    monkeypatch.setattr(pipeline, "benchmark_proxy_for", lambda fund: "SWDA.L")
    monkeypatch.setattr(pipeline, "get_benchmark_returns", lambda ticker: benchmark)
    monkeypatch.setattr(pipeline, "region_for_category", lambda category: "developed")
    monkeypatch.setattr(pipeline, "get_factors", lambda region, freq: factors)
    monkeypatch.setattr(pipeline, "convert_factor_returns", lambda factors_usd, currency: factors_usd)
    monkeypatch.setattr(pipeline, "perf_summary", lambda monthly, benchmark_returns, rf: {})
    monkeypatch.setattr(pipeline, "drawdown_series", lambda monthly: monthly.cumsum())
    monkeypatch.setattr(pipeline, "rolling_excess", lambda monthly, benchmark_returns, window: monthly - benchmark_returns)
    monkeypatch.setattr(pipeline, "fit_factor_model", fake_fit)
    monkeypatch.setattr(pipeline, "fit_benchmark_residual_alpha", fake_benchmark_step)
    monkeypatch.setattr(pipeline, "rolling_betas", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(pipeline, "subperiod_alphas", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(pipeline, "get_style_proxies", lambda region: _raise("no style"))
    monkeypatch.setattr(pipeline, "get_fund_holdings", lambda fund: _raise("no holdings"))
    monkeypatch.setattr(pipeline, "get_etf_holdings", lambda ticker: _raise("no benchmark holdings"))
    monkeypatch.setattr(pipeline, "factor_contributions", lambda fit, factors: pd.DataFrame())
    monkeypatch.setattr(pipeline, "evaluate_flags", lambda result: [])
    monkeypatch.setattr(pipeline, "questions_for", lambda flags: [])

    result = pipeline.analyse_fund("GB00TEST1234")

    ladder = result["alpha_ladder"]
    assert ladder["steps"]["ff5_mom"]["label"] == "FF5+MOM alpha"
    assert ladder["steps"]["benchmark"]["label"] == "Benchmark residual alpha"
    assert ladder["verdict"] == "benchmark_explained"
    assert ladder["selected_proxies"][0]["ticker"] == "SWDA.L"

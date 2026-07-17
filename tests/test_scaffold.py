"""Scaffold sanity tests: every module imports cleanly and the CLI --help works."""
from __future__ import annotations

import importlib

from typer.testing import CliRunner

MODULES = [
    "fundlens",
    "fundlens.config",
    "fundlens.cache",
    "fundlens.cli",
    "fundlens.pipeline",
    "fundlens.data",
    "fundlens.data.resolver",
    "fundlens.data.navs",
    "fundlens.data.factors",
    "fundlens.data.fx",
    "fundlens.data.benchmarks",
    "fundlens.data.holdings",
    "fundlens.data.fundamentals",
    "fundlens.analysis",
    "fundlens.analysis.alpha_ladder",
    "fundlens.analysis.returns",
    "fundlens.analysis.factor_model",
    "fundlens.analysis.style",
    "fundlens.analysis.holdings_analytics",
    "fundlens.analysis.attribution",
    "fundlens.analysis.flags",
    "fundlens.analysis.questions",
    "fundlens.analysis.manager_dd",
    "fundlens.screening",
    "fundlens.report",
    "fundlens.report.builder",
    "fundlens.report.figures",
    "fundlens.report.view",
    "fundlens.ui",
]


def test_all_modules_import():
    for module_name in MODULES:
        importlib.import_module(module_name)


def test_cli_help():
    from fundlens.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "fundlens" in result.output.lower()

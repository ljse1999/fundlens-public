"""Render an analysis result dict into a single self-contained HTML report."""
from __future__ import annotations

from pathlib import Path

import plotly.io as pio
import plotly.offline as pyo
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from fundlens.report.figures import build_chart_specs
from fundlens.report.view import build_report_view

TEMPLATE_DIR = Path(__file__).with_name("templates")


def _to_div(fig, div_id: str) -> Markup:
    html = pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id=div_id)
    return Markup(html)


def _html_chart_specs(data: dict) -> dict[str, dict]:
    charts = {}
    for key, spec in build_chart_specs(data).items():
        if spec.get("available"):
            charts[key] = {"available": True, "html": _to_div(spec["figure"], f"chart-{key.replace('_', '-')}")}
        else:
            charts[key] = spec
    return charts


def _build_context(data: dict) -> dict:
    context = build_report_view(data)
    context["charts"] = _html_chart_specs(data)
    context["plotly_js"] = Markup(f"<script>{pyo.get_plotlyjs()}</script>")
    return context


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
    )


def build_report(data: dict, out_path: Path) -> Path:
    """Render an analysis result dict to a self-contained HTML report file.

    Args:
        data: The result dict produced by
            :func:`fundlens.pipeline.analyse_fund`.
        out_path: Destination path for the rendered HTML report. Parent
            directories are created if missing.

    Returns:
        The ``out_path`` the report was written to, for convenience.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    context = _build_context(data)
    template = _env().get_template("report.html.j2")
    html = template.render(**context)
    out_path.write_text(html, encoding="utf-8")
    return out_path

"""Unit tests for Self-Contained Interactive HTML Report Generator.

Functional Purpose:
    Verifies HTML document construction, embedded SVG chart rendering, monthly heatmap
    matrix layout, and structured JSON telemetry extraction.

Explicit Dependency Tracking:
    - pytest, json.
    - quant.analytics.html_report: HtmlReportGenerator.
    - quant.analytics.tearsheet: PerformanceAnalyticsEngine.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from quant.analytics.html_report import HtmlReportGenerator
from quant.analytics.tearsheet import PerformanceAnalyticsEngine


def _create_sample_tearsheet_report() -> any:
    """Helper creating a sample TearsheetReport."""
    n = 100
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.015, n)
    t0 = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1e9)
    timestamps = np.array([t0 + i * int(86400 * 1e9) for i in range(n)], dtype=np.int64)

    return PerformanceAnalyticsEngine.calculate_tearsheet(
        strategy_returns=returns,
        timestamps_ns=timestamps,
        risk_free_rate=0.04,
        metadata={"strategy": "FracDiff_Swarm", "version": "1.0.0"},
    )


def test_html_report_rendering_structure() -> None:
    """Verify generated HTML string contains required SVG charts and tables."""
    report = _create_sample_tearsheet_report()
    html_out = HtmlReportGenerator.render_html(report=report, title="Audit Tearsheet")

    # Document integrity
    assert "<!DOCTYPE html>" in html_out
    assert "<title>Audit Tearsheet</title>" in html_out
    assert "#090D16" in html_out  # Dark mode background
    assert "#38BDF8" in html_out  # Accent blue

    # SVG charts present
    assert "<svg" in html_out
    assert 'class="chart-svg"' in html_out

    # Monthly return heatmap table present
    assert "Monthly Return Heatmap Matrix" in html_out
    assert "<th>M01</th>" in html_out
    assert "<th>YTD</th>" in html_out

    # JSON telemetry extraction verification
    assert '<script id="quant-report-data" type="application/json">' in html_out
    json_start = html_out.find('<script id="quant-report-data" type="application/json">')
    json_end = html_out.find("</script>", json_start)
    raw_json = html_out[
        json_start + len('<script id="quant-report-data" type="application/json">') : json_end
    ].strip()

    parsed = json.loads(raw_json)
    assert "metrics" in parsed
    assert "sharpe_ratio" in parsed["metrics"]
    assert parsed["metadata"]["strategy"] == "FracDiff_Swarm"


def test_html_report_file_export(tmp_path: Path) -> None:
    """Verify report export writes clean HTML file to filesystem."""
    report = _create_sample_tearsheet_report()
    out_file = tmp_path / "test_report.html"

    exported_path = HtmlReportGenerator.export_html_report(
        report=report,
        output_path=out_file,
        title="File Export Test",
    )

    assert exported_path.exists()
    assert exported_path.is_file()
    assert exported_path.stat().st_size > 1000

    content = exported_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "File Export Test" in content

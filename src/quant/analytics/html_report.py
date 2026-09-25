"""Self-Contained Interactive HTML Report Generator for CFA Tearsheets.

Functional Purpose:
    Generates standalone, institutional dark-theme HTML performance reports containing
    SVG equity curves, underwater drawdown charts, calendar monthly return heatmaps,
    and structured JSON telemetry. Has zero external runtime network or CDN dependencies,
    ensuring 100% hermetic rendering in offline environments.

Explicit Dependency Tracking:
    - quant.analytics.tearsheet: TearsheetReport, CFAMetrics, TearsheetError.
    - pathlib: Path creation and file export.
    - json: Structured telemetry serialization.

Structural Relationship:
    Export layer for Phase 15. Consumed by Backtest CLI runners, FastAPI `/tearsheet.html`
    endpoints, and Web BacktestStudio download buttons.

Defensive Invariants:
    - Zero External Dependencies: All styles and vector charts embedded inline.
    - Strict Sanitization: HTML output escapes special characters to prevent XSS.
    - JSON Invariant: Embeds reproducible machine-readable telemetry payload in script tag.
"""

from __future__ import annotations

import html
import json
import logging
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from quant.analytics.tearsheet import TearsheetError, TearsheetReport

logger = logging.getLogger(__name__)

ERR_RPT_HTML_EXPORT_FAILED: Final[str] = "ERR-RPT-010"


def _format_pct(val: float | None, precision: int = 2) -> str:
    if val is None or not math.isfinite(val):
        return "N/A"
    return f"{val * 100.0:+.{precision}f}%"


def _format_num(val: float | None, precision: int = 2) -> str:
    if val is None or not math.isfinite(val):
        return "N/A"
    return f"{val:.{precision}f}"


def _generate_svg_sparkline(
    y_values: list[float],
    width: int = 800,
    height: int = 250,
    stroke_color: str = "#38BDF8",
    fill_color: str | None = "rgba(56, 189, 248, 0.1)",
    is_drawdown: bool = False,
) -> str:
    """Generate a clean self-contained responsive SVG path for time series data."""
    if not y_values:
        return f'<svg viewBox="0 0 {width} {height}"><text x="10" y="20" fill="#94A3B8">No Data</text></svg>'

    n = len(y_values)
    y_min = min(y_values)
    y_max = max(y_values)

    if math.isclose(y_max, y_min):
        y_max += 1e-4
        y_min -= 1e-4

    if is_drawdown:
        y_max = 0.0  # Anchor top to 0%

    pad_top = 20
    pad_bottom = 20
    usable_h = height - pad_top - pad_bottom

    points: list[tuple[float, float]] = []
    for i, val in enumerate(y_values):
        x = (i / max(1, n - 1)) * (width - 20) + 10
        norm_y = (val - y_min) / (y_max - y_min)
        y = height - pad_bottom - (norm_y * usable_h)
        points.append((x, y))

    path_data = f"M {points[0][0]:.1f} {points[0][1]:.1f} " + " ".join(
        f"L {p[0]:.1f} {p[1]:.1f}" for p in points[1:]
    )

    fill_polygon = ""
    if fill_color:
        if is_drawdown:
            zero_y = height - pad_bottom - ((0.0 - y_min) / (y_max - y_min) * usable_h)
            fill_data = f"{path_data} L {points[-1][0]:.1f} {zero_y:.1f} L {points[0][0]:.1f} {zero_y:.1f} Z"
        else:
            fill_data = f"{path_data} L {points[-1][0]:.1f} {height - pad_bottom:.1f} L {points[0][0]:.1f} {height - pad_bottom:.1f} Z"
        fill_polygon = f'<path d="{fill_data}" fill="{fill_color}" />'

    return f"""<svg viewBox="0 0 {width} {height}" class="chart-svg">
      <defs>
        <linearGradient id="chartGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="{stroke_color}" stop-opacity="0.3"/>
          <stop offset="100%" stop-color="{stroke_color}" stop-opacity="0.0"/>
        </linearGradient>
      </defs>
      {fill_polygon}
      <path d="{path_data}" fill="none" stroke="{stroke_color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />
    </svg>"""


class HtmlReportGenerator:
    """Renders standalone institutional performance HTML documents."""

    @staticmethod
    def render_html(
        report: TearsheetReport,
        title: str = "Institutional Quantitative Strategy Tearsheet",
    ) -> str:
        """Render complete HTML string from a TearsheetReport."""
        m = report.metrics

        # Convert equity & drawdowns to Python float lists
        equity_list = [float(x) for x in report.equity_curve]
        dd_list = [float(x) for x in report.drawdown_series]

        equity_svg = _generate_svg_sparkline(
            equity_list, stroke_color="#38BDF8", fill_color="url(#chartGrad)"
        )
        dd_svg = _generate_svg_sparkline(
            dd_list,
            stroke_color="#EF4444",
            fill_color="rgba(239, 68, 68, 0.15)",
            is_drawdown=True,
        )

        # Build monthly heatmap rows
        month_headers = "".join(f"<th>M{i:02d}</th>" for i in range(1, 13))
        table_rows: list[str] = []

        for year, months in sorted(report.monthly_matrix.items(), reverse=True):
            tds: list[str] = [f"<td><strong>{year}</strong></td>"]
            for i in range(1, 13):
                m_key = f"M{i:02d}"
                val = months.get(m_key, 0.0)
                if math.isclose(val, 0.0, abs_tol=1e-6):
                    tds.append('<td class="val-zero">-</td>')
                elif val > 0:
                    alpha = min(1.0, val * 10.0)
                    tds.append(
                        f'<td class="val-pos" style="background-color: rgba(16, 185, 129, {alpha * 0.3:.2f})">+{val * 100:.1f}%</td>'
                    )
                else:
                    alpha = min(1.0, abs(val) * 10.0)
                    tds.append(
                        f'<td class="val-neg" style="background-color: rgba(239, 68, 68, {alpha * 0.3:.2f})">{val * 100:.1f}%</td>'
                    )

            ytd_val = months.get("YTD", 0.0)
            ytd_class = "val-pos" if ytd_val >= 0 else "val-neg"
            tds.append(
                f'<td class="{ytd_class} ytd-cell"><strong>{ytd_val * 100:+.2f}%</strong></td>'
            )
            table_rows.append(f"<tr>{''.join(tds)}</tr>")

        monthly_table_html = "\n".join(table_rows)

        # JSON Telemetry payload
        telemetry_dict = {
            "metrics": {
                "total_return": m.total_return,
                "cagr": m.cagr,
                "annualized_volatility": m.annualized_volatility,
                "sharpe_ratio": m.sharpe_ratio,
                "sortino_ratio": m.sortino_ratio,
                "calmar_ratio": m.calmar_ratio,
                "max_drawdown": m.max_drawdown,
                "max_drawdown_duration": m.max_drawdown_duration_bars,
                "win_rate": m.win_rate,
                "profit_factor": m.profit_factor,
                "alpha": m.alpha,
                "beta": m.beta,
                "information_ratio": m.information_ratio,
            },
            "observation_count": len(report.timestamps),
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "metadata": report.metadata,
        }
        telemetry_json = json.dumps(telemetry_dict, indent=2).replace("</", "<\\/")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg-dark: #090D16;
      --card-bg: #111827;
      --border: #1E293B;
      --text-main: #F1F5F9;
      --text-muted: #94A3B8;
      --accent-blue: #38BDF8;
      --accent-green: #10B981;
      --accent-red: #EF4444;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background-color: var(--bg-dark);
      color: var(--text-main);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      padding: 32px;
      line-height: 1.5;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    .header {{
      display: flex; justify-content: space-between; align-items: flex-end;
      border-bottom: 1px solid var(--border); padding-bottom: 20px; margin-bottom: 28px;
    }}
    .title {{ font-size: 24px; font-weight: 700; color: var(--text-main); letter-spacing: -0.5px; }}
    .subtitle {{ font-size: 13px; color: var(--text-muted); margin-top: 4px; }}
    .badge {{
      display: inline-block; padding: 4px 10px; border-radius: 4px;
      background: rgba(56, 189, 248, 0.15); color: var(--accent-blue);
      font-size: 12px; font-weight: 600; text-transform: uppercase;
    }}
    .grid-metrics {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px; margin-bottom: 32px;
    }}
    .metric-card {{
      background: var(--card-bg); border: 1px solid var(--border);
      border-radius: 8px; padding: 16px;
    }}
    .metric-label {{ font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 600; }}
    .metric-value {{ font-size: 22px; font-weight: 700; color: var(--text-main); margin-top: 6px; }}
    .metric-sub {{ font-size: 11px; color: var(--text-muted); margin-top: 2px; }}
    .val-pos {{ color: var(--accent-green); }}
    .val-neg {{ color: var(--accent-red); }}
    .val-zero {{ color: var(--text-muted); }}

    .chart-section {{
      background: var(--card-bg); border: 1px solid var(--border);
      border-radius: 8px; padding: 20px; margin-bottom: 28px;
    }}
    .section-title {{ font-size: 16px; font-weight: 600; margin-bottom: 16px; color: var(--text-main); }}
    .chart-svg {{ width: 100%; height: 260px; display: block; }}

    .table-container {{
      background: var(--card-bg); border: 1px solid var(--border);
      border-radius: 8px; padding: 20px; margin-bottom: 28px; overflow-x: auto;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: right; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); }}
    th {{ color: var(--text-muted); font-weight: 600; font-size: 11px; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .ytd-cell {{ background: rgba(255, 255, 255, 0.03); border-left: 1px solid var(--border); }}

    footer {{ text-align: center; color: var(--text-muted); font-size: 12px; margin-top: 40px; }}
  </style>
</head>
<body>
  <div class="container">
    <header class="header">
      <div>
        <h1 class="title">{html.escape(title)}</h1>
        <p class="subtitle">Generated on {datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")} | Universe: {m.total_trades_bars} Observations</p>
      </div>
      <div>
        <span class="badge">CFA Tier-1 Attribution</span>
      </div>
    </header>

    <!-- Key Performance Metrics Grid -->
    <div class="grid-metrics">
      <div class="metric-card">
        <div class="metric-label">Cumulative Return</div>
        <div class="metric-value {"val-pos" if m.total_return >= 0 else "val-neg"}">{_format_pct(m.total_return)}</div>
        <div class="metric-sub">CAGR: {_format_pct(m.cagr)}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Sharpe Ratio</div>
        <div class="metric-value {"val-pos" if m.sharpe_ratio >= 1.0 else "val-neg"}">{_format_num(m.sharpe_ratio)}</div>
        <div class="metric-sub">Sortino: {_format_num(m.sortino_ratio)}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Max Drawdown</div>
        <div class="metric-value val-neg">{_format_pct(m.max_drawdown)}</div>
        <div class="metric-sub">Duration: {m.max_drawdown_duration_bars} bars</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Ann. Volatility</div>
        <div class="metric-value">{_format_pct(m.annualized_volatility)}</div>
        <div class="metric-sub">Calmar: {_format_num(m.calmar_ratio)}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Win Rate</div>
        <div class="metric-value">{_format_pct(m.win_rate)}</div>
        <div class="metric-sub">Profit Factor: {_format_num(m.profit_factor)}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">CAPM Alpha / Beta</div>
        <div class="metric-value">{_format_pct(m.alpha)}</div>
        <div class="metric-sub">Beta: {_format_num(m.beta)} | IR: {_format_num(m.information_ratio)}</div>
      </div>
    </div>

    <!-- Equity Curve Chart -->
    <div class="chart-section">
      <h2 class="section-title">Cumulative Performance Growth (Equity Curve)</h2>
      {equity_svg}
    </div>

    <!-- Underwater Drawdown Chart -->
    <div class="chart-section">
      <h2 class="section-title">Underwater Drawdown Profile</h2>
      {dd_svg}
    </div>

    <!-- Monthly Return Matrix Heatmap -->
    <div class="table-container">
      <h2 class="section-title">Monthly Return Heatmap Matrix</h2>
      <table>
        <thead>
          <tr>
            <th>Year</th>
            {month_headers}
            <th>YTD</th>
          </tr>
        </thead>
        <tbody>
          {monthly_table_html}
        </tbody>
      </table>
    </div>

    <footer>
      Quant Engine Institutional Backtesting Studio &bull; Verified Zero-Lookahead Model Context
    </footer>
  </div>

  <script id="quant-report-data" type="application/json">
{telemetry_json}
  </script>
</body>
</html>"""

    @classmethod
    def export_html_report(
        cls,
        report: TearsheetReport,
        output_path: str | Path | None = None,
        title: str = "Institutional Quantitative Strategy Tearsheet",
    ) -> Path:
        """Render and write HTML report to file."""
        html_content = cls.render_html(report=report, title=title)

        if output_path is None:
            os.makedirs("reports", exist_ok=True)
            ts_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            out_file = Path(f"reports/tearsheet_{ts_str}.html")
        else:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            out_file.write_text(html_content, encoding="utf-8")
            logger.info(f"Tearsheet HTML report exported successfully to {out_file}")
            return out_file
        except Exception as e:
            raise TearsheetError(
                f"Failed to export HTML report to {out_file}: {e}",
                code=ERR_RPT_HTML_EXPORT_FAILED,
            ) from e

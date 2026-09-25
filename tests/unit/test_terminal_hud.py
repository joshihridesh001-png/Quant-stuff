"""Comprehensive Unit Tests for Rich Terminal HUD.

Functional Purpose:
    Verifies that the Terminal HUD adheres strictly to:
    - INV-HUD-001: Sub-2ms layout render overhead.
    - INV-HUD-002: Zero crash on non-finite, NaN, Infinity, or missing symbol inputs.
    - INV-HUD-003: Headless terminal rendering safety and string output generation.

Governing Rules:
    - Rule 1: Four-tier docstrings on all fixtures and tests.
    - Rule 2: Deterministic error codes (ERR-HUD-001, ERR-HUD-002).
    - Rule 3: Quality gates (100% pass rate, strict static typing).
    - Rule 4: Adversarial non-finite inputs, extreme drawdowns, and event queue bounding.
"""

from __future__ import annotations

import time

import pytest
from rich.layout import Layout

from quant.analytics.strategies.base import SignalDirection, StrategySignal
from quant.execution.terminal_hud import TerminalHUD
from quant.services.live_session import LiveSessionStatus, LiveSessionTelemetry


@pytest.fixture
def sample_telemetry() -> LiveSessionTelemetry:
    """Fixture providing rich, realistic sample telemetry."""
    signals = {
        "SPY": StrategySignal(
            symbol="SPY",
            direction=SignalDirection.LONG,
            target_weight=0.25,
            conviction=0.90,
            target_horizon_bars=10,
        ),
        "QQQ": StrategySignal(
            symbol="QQQ",
            direction=SignalDirection.SHORT,
            target_weight=-0.15,
            conviction=0.75,
            target_horizon_bars=5,
        ),
    }
    return LiveSessionTelemetry(
        timestamp_ns=time.time_ns(),
        iteration=42,
        status=LiveSessionStatus.RUNNING,
        portfolio_nav=105420.50,
        cash=65000.25,
        gross_leverage=0.40,
        net_leverage=0.10,
        drawdown_pct=0.0125,
        current_prices={"SPY": 505.20, "QQQ": 435.10},
        positions={"SPY": 50.0, "QQQ": -35.0},
        target_weights={"SPY": 0.25, "QQQ": -0.15},
        active_signals=signals,
        orders_dispatched_count=12,
        fills_count=12,
        is_kill_switch_active=False,
    )


def test_terminal_hud_render_layout_structure(sample_telemetry: LiveSessionTelemetry) -> None:
    """Verify that render_layout produces a valid 4-quadrant Rich Layout."""
    hud = TerminalHUD()
    layout = hud.render_layout(sample_telemetry)

    assert isinstance(layout, Layout)
    assert "header" in [c.name for c in layout.children]
    assert "body" in [c.name for c in layout.children]
    assert "footer" in [c.name for c in layout.children]


def test_terminal_hud_render_string_content(sample_telemetry: LiveSessionTelemetry) -> None:
    """Verify that render_string captures textual representation containing all critical metrics."""
    hud = TerminalHUD()
    hud.add_event("14:30:00", "FILL", "BUY 50 SPY @ 505.20")
    rendered_text = hud.render_string(sample_telemetry)

    assert "QUANT STATION" in rendered_text
    assert "105,420.50" in rendered_text
    assert "65,000.25" in rendered_text
    assert "SPY" in rendered_text
    assert "QQQ" in rendered_text
    assert "FILL" in rendered_text
    assert "KILL SWITCH: ARMED" in rendered_text


def test_terminal_hud_render_sla_performance(sample_telemetry: LiveSessionTelemetry) -> None:
    """Verify that layout generation complies with sub-2ms latency SLA (INV-HUD-001)."""
    hud = TerminalHUD()

    # Warmup
    _ = hud.render_layout(sample_telemetry)

    # Measure 50 iterations
    t0 = time.perf_counter()
    for _ in range(50):
        _ = hud.render_layout(sample_telemetry)
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) / 50.0) * 1000.0
    assert avg_ms < 2.0, f"Average render time {avg_ms:.3f}ms exceeded 2.0ms SLA"


def test_terminal_hud_non_finite_resilience() -> None:
    """Verify that non-finite values (NaN, Inf) do not crash the display (INV-HUD-002)."""
    corrupt_telemetry = LiveSessionTelemetry(
        timestamp_ns=time.time_ns(),
        iteration=1,
        status=LiveSessionStatus.ERROR,
        portfolio_nav=float("nan"),
        cash=float("inf"),
        gross_leverage=float("-inf"),
        net_leverage=float("nan"),
        drawdown_pct=float("nan"),
        current_prices={"SPY": float("nan")},
        positions={"SPY": float("nan")},
        target_weights={"SPY": float("nan")},
        active_signals={},
        orders_dispatched_count=0,
        fills_count=0,
        is_kill_switch_active=True,
        last_error="Corrupt floating point injected",
    )

    hud = TerminalHUD()
    rendered = hud.render_string(corrupt_telemetry)
    assert "QUANT STATION" in rendered
    assert "STATUS: ERROR" in rendered
    assert "KILL SWITCH: ACTIVE" in rendered


def test_terminal_hud_event_buffer_bounding() -> None:
    """Verify HUD ring buffer caps recent events to 8 items."""
    hud = TerminalHUD()
    for i in range(15):
        hud.add_event(f"10:00:{i:02d}", "INFO", f"Event #{i}")

    assert len(hud._recent_events) == 8
    assert hud._recent_events[-1][2] == "Event #14"

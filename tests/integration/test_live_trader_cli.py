"""Integration Tests for Turnkey Live Trader CLI Runner.

Functional Purpose:
    Verifies that the standalone CLI entrypoint scripts/run_live_trader.py executes
    cleanly across diverse strategy configurations, terminates cleanly after N iterations,
    and adheres to all system quality gates.

Governing Rules:
    - Rule 1: Four-tier docstrings on all fixtures and tests.
    - Rule 2: Deterministic error codes and exit codes.
    - Rule 3: Quality gates (100% pass rate, strict static typing).
    - Rule 4: Zero unhandled subprocess exceptions and clean shutdown verification.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_live_trader_cli_help() -> None:
    """Verify that --help exits cleanly with exit code 0 and displays valid options."""
    cmd = [sys.executable, "scripts/run_live_trader.py", "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert "Turnkey Continuous Live Paper Trading Station" in result.stdout
    assert "--strategy" in result.stdout
    assert "--capital" in result.stdout


@pytest.mark.parametrize("strat", ["swarm", "kalman", "momentum"])
def test_live_trader_cli_headless_execution(strat: str) -> None:
    """Verify that live trader runs headlessly for 3 iterations and shuts down with code 0."""
    cmd = [
        sys.executable,
        "scripts/run_live_trader.py",
        "--symbols",
        "SPY,QQQ",
        "--strategy",
        strat,
        "--capital",
        "50000",
        "--poll-interval",
        "0.05",
        "--max-iterations",
        "3",
        "--headless",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, f"Process failed with stderr: {result.stderr}"
    # Verify that telemetry lines were output
    assert "NAV:" in result.stdout or "Cash:" in result.stdout

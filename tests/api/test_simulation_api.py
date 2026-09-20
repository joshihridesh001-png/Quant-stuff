"""Integration tests for Live Replay Simulation & Backtest REST API endpoints.

Governing Standards:
- Rules.md (Rule 1: Line annotations, Rule 2: Diagnostic error codes, Rule 3: Quality gates)
- Endpoints: POST /api/v1/simulation/run
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_unauthenticated_simulation_rejected(client: AsyncClient) -> None:
    """Verify simulation endpoints return 401 when unauthenticated."""
    resp = await client.post("/api/v1/simulation/run", json={"asset_id": "NVDA"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_simulation_run_nominal(
    client: AsyncClient, researcher_jwt_headers: dict[str, str]
) -> None:
    """Verify POST /api/v1/simulation/run executes successfully and emits full tear sheet."""
    payload = {
        "asset_id": "NVDA",
        "bar_count": 60,
        "initial_capital": 10000.0,
        "fee_bps": 2.0,
        "spread_bps": 1.0,
        "impact_coefficient": 0.10,
    }
    resp = await client.post(
        "/api/v1/simulation/run",
        json=payload,
        headers=researcher_jwt_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["asset_id"] == "NVDA"
    assert data["bar_count"] == 60
    assert data["initial_capital"] == 10000.0
    assert data["final_equity"] > 0.0
    assert "sharpe_ratio" in data
    assert "deflated_sharpe_ratio" in data
    assert "max_drawdown_pct" in data
    assert "equity_curve" in data
    assert len(data["equity_curve"]) == 61  # initial + 60 closed bars
    assert "benchmarks" in data
    assert len(data["benchmarks"]) >= 1


@pytest.mark.asyncio
async def test_simulation_run_invalid_bars(
    client: AsyncClient, researcher_jwt_headers: dict[str, str]
) -> None:
    """Verify POST /api/v1/simulation/run rejects bar_count < 30 with 422."""
    payload = {
        "asset_id": "NVDA",
        "bar_count": 10,
        "initial_capital": 10000.0,
    }
    resp = await client.post(
        "/api/v1/simulation/run",
        json=payload,
        headers=researcher_jwt_headers,
    )
    assert resp.status_code == 422

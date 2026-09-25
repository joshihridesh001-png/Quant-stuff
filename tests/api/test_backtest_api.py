"""Integration & API Tests for Backtesting Studio REST and WebSocket Endpoints.

Functional Purpose:
    Validates the end-to-end execution of quantitative backtests via FastAPI endpoints:
    - POST /api/v1/backtest/run (synchronous dispatch with worker threadpool)
    - GET /api/v1/backtest/{id}/status (metrics, curves, monthly matrix)
    - GET /api/v1/backtest/{id}/tearsheet.html (standalone HTML report export)
    - GET /api/v1/backtest/history (historical job log)
    - Deterministic error handling (404 on missing jobs, 400 on invalid strategies).

Explicit Dependency Tracking:
    - pytest, httpx.AsyncClient.
    - quant.api.v1.schemas: BacktestRunRequest, BacktestStatusResponse.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_run_backtest_fracdiff_swarm_success(client: AsyncClient) -> None:
    """Verify executing a FracDiff_Swarm backtest produces CFA metrics and equity curve."""
    payload = {
        "strategy_type": "FracDiff_Swarm",
        "symbols": ["SPY", "QQQ"],
        "initial_cash": 100000.0,
        "benchmark_symbol": "SPY",
        "cost_bps": 2.0,
        "parameters": {"learning_rate": 0.05, "min_warmup_bars": 35},
    }

    response = await client.post("/api/v1/backtest/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "backtest_id" in data
    assert data["status"] == "COMPLETED"
    backtest_id = data["backtest_id"]

    # Check status endpoint
    status_resp = await client.get(f"/api/v1/backtest/{backtest_id}/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()

    assert status_data["backtest_id"] == backtest_id
    assert status_data["status"] == "COMPLETED"
    assert status_data["progress"] == 1.0
    assert status_data["strategy_type"] == "FracDiff_Swarm"

    # CFA Metrics presence
    metrics = status_data["metrics"]
    assert metrics is not None
    assert "sharpe_ratio" in metrics
    assert "sortino_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "annualized_return" in metrics
    assert metrics["initial_capital"] == 100000.0
    assert metrics["final_equity"] > 0.0

    # Curves and monthly matrix
    assert status_data["equity_curve"] is not None
    assert len(status_data["equity_curve"]) > 20
    assert status_data["drawdown_series"] is not None
    assert status_data["monthly_matrix"] is not None

    # HTML tearsheet endpoint
    html_resp = await client.get(f"/api/v1/backtest/{backtest_id}/tearsheet.html")
    assert html_resp.status_code == 200
    assert "text/html" in html_resp.headers.get("content-type", "")
    assert "<!DOCTYPE html>" in html_resp.text
    assert "quant-report-data" in html_resp.text


@pytest.mark.asyncio
async def test_run_backtest_kalman_statarb_success(client: AsyncClient) -> None:
    """Verify executing KalmanPairsTradingStrategy backtest across asset pairs."""
    payload = {
        "strategy_type": "Kalman_StatArb",
        "symbols": ["SPY", "QQQ"],
        "initial_cash": 250000.0,
        "benchmark_symbol": "SPY",
        "cost_bps": 1.5,
        "parameters": {"entry_z_score": 1.8, "exit_z_score": 0.4},
    }

    response = await client.post("/api/v1/backtest/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "COMPLETED"
    backtest_id = data["backtest_id"]

    status_resp = await client.get(f"/api/v1/backtest/{backtest_id}/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["metrics"]["initial_capital"] == 250000.0


@pytest.mark.asyncio
async def test_run_backtest_vol_breakout_success(client: AsyncClient) -> None:
    """Verify executing VolatilityBreakoutStrategy backtest."""
    payload = {
        "strategy_type": "Vol_Breakout",
        "symbols": ["AAPL", "NVDA"],
        "initial_cash": 50000.0,
        "benchmark_symbol": "SPY",
        "cost_bps": 2.0,
        "parameters": {"bollinger_window": 20, "squeeze_lookback": 25},
    }

    response = await client.post("/api/v1/backtest/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_backtest_history_endpoint(client: AsyncClient) -> None:
    """Verify history endpoint returns list of completed backtests."""
    response = await client.get("/api/v1/backtest/history")
    assert response.status_code == 200
    history = response.json()
    assert isinstance(history, list)
    assert len(history) >= 1


@pytest.mark.asyncio
async def test_backtest_not_found_404(client: AsyncClient) -> None:
    """Verify 404 response on querying non-existent backtest ID."""
    response = await client.get("/api/v1/backtest/non_existent_bkt_9999/status")
    assert response.status_code == 404
    err = response.json()
    assert err["detail"]["code"] == "ERR-BKT-007"


@pytest.mark.asyncio
async def test_backtest_invalid_strategy_400(client: AsyncClient) -> None:
    """Verify 500/400 error on requesting non-existent strategy."""
    payload = {
        "strategy_type": "NonExistentMagicalStrategy",
        "symbols": ["SPY"],
        "initial_cash": 100000.0,
    }
    response = await client.post("/api/v1/backtest/run", json=payload)
    assert response.status_code in (400, 500)

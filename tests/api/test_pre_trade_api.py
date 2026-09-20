"""Integration tests for Pre-Trade Decision Gate REST API endpoints.

Governing Standards:
- Rules.md (Rule 1: Line annotations, Rule 2: Diagnostic error codes, Rule 3: Quality gates)
- Endpoints: POST /api/v1/pre-trade/evaluate, GET /api/v1/pre-trade/decisions, GET /api/v1/pre-trade/status
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_unauthenticated_pre_trade_endpoints_rejected(client: AsyncClient) -> None:
    """Verify pre-trade endpoints return 401 when unauthenticated."""
    routes = [
        ("POST", "/api/v1/pre-trade/evaluate"),
        ("GET", "/api/v1/pre-trade/decisions"),
        ("GET", "/api/v1/pre-trade/status"),
    ]
    for method, path in routes:
        resp = await client.request(method, path)
        assert resp.status_code == 401, f"{method} {path} returned {resp.status_code}, expected 401"


@pytest.mark.asyncio
async def test_evaluate_order_nominal_pass(
    client: AsyncClient, researcher_jwt_headers: dict[str, str]
) -> None:
    """Verify POST /api/v1/pre-trade/evaluate returns PASS for nominal order."""
    payload = {
        "symbol": "SPY",
        "action": "BUY",
        "quantity": 50.0,
        "reference_price": 510.0,
        "market_spread_bps": 2.0,
        "order_book_imbalance": 0.05,
        "macro_yield_spread": 0.20,
        "trailing_volatility_pct": 1.1,
        "current_bar_return_pct": 0.02,
        "consecutive_losses": 0,
        "current_drawdown_pct": 0.5,
        "data_age_seconds": 1.0,
    }
    resp = await client.post(
        "/api/v1/pre-trade/evaluate",
        json=payload,
        headers=researcher_jwt_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["allowed"] is True
    assert data["decision"] == "PASS"
    assert data["primary_code"] == "OK"
    assert "checks" in data
    assert len(data["checks"]) == 5


@pytest.mark.asyncio
async def test_evaluate_order_adverse_reject(
    client: AsyncClient, researcher_jwt_headers: dict[str, str]
) -> None:
    """Verify POST /api/v1/pre-trade/evaluate returns REJECT for stale or toxic flow."""
    payload = {
        "symbol": "NVDA",
        "action": "BUY",
        "quantity": 10.0,
        "reference_price": 120.0,
        "data_age_seconds": 200.0,  # Stale data
    }
    resp = await client.post(
        "/api/v1/pre-trade/evaluate",
        json=payload,
        headers=researcher_jwt_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["allowed"] is False
    assert data["decision"] == "REJECT"
    assert data["primary_code"] == "ERR-GATE-001"


@pytest.mark.asyncio
async def test_get_pre_trade_decisions_and_status(
    client: AsyncClient, researcher_jwt_headers: dict[str, str]
) -> None:
    """Verify GET /api/v1/pre-trade/decisions and GET /api/v1/pre-trade/status endpoints."""
    # Check decisions history
    dec_resp = await client.get("/api/v1/pre-trade/decisions", headers=researcher_jwt_headers)
    assert dec_resp.status_code == 200
    decisions = dec_resp.json()
    assert isinstance(decisions, list)

    # Check status
    stat_resp = await client.get("/api/v1/pre-trade/status", headers=researcher_jwt_headers)
    assert stat_resp.status_code == 200
    status_data = stat_resp.json()
    assert "max_toxic_probability_threshold" in status_data
    assert "max_data_age_seconds" in status_data
    assert "decisions_count" in status_data

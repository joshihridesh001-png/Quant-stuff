"""API tests for system health and authentication token issuance."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check_endpoint(client: AsyncClient) -> None:
    response = await client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["database"] == "healthy"
    assert "environment" in data


@pytest.mark.asyncio
async def test_auth_token_issuance(client: AsyncClient) -> None:
    # 1. Invalid credentials -> 401
    bad_login = await client.post(
        "/api/v1/auth/token",
        json={"username": "trader", "password": "wrong-password", "role": "RESEARCHER"},
    )
    assert bad_login.status_code == 401

    # 2. Valid credentials -> 200 with JWT
    good_login = await client.post(
        "/api/v1/auth/token",
        json={"username": "lead_quant", "password": "quant-secret-pass", "role": "RESEARCHER"},
    )
    assert good_login.status_code == 200
    token_data = good_login.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"

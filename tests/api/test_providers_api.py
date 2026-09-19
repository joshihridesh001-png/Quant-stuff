"""Integration tests for External Quantitative Data Providers REST API endpoints.

Governing Standards:
- Rules.md (Rule 1: Line annotations, Rule 2: Diagnostic error codes, Rule 3: Quality gates)
- Endpoints: GET /api/v1/providers/status, POST /api/v1/providers/macro/sync, GET /api/v1/providers/company-news/{symbol}
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_unauthenticated_provider_endpoints_rejected(client: AsyncClient) -> None:
    """Verify provider endpoints return 401 when unauthenticated.

    Args:
        client: Test HTTP client.
    """
    routes = [
        ("GET", "/api/v1/providers/status"),
        ("POST", "/api/v1/providers/macro/sync"),
        ("GET", "/api/v1/providers/company-news/AAPL"),
    ]
    for method, path in routes:
        resp = await client.request(method, path)
        assert resp.status_code == 401, f"{method} {path} returned {resp.status_code}, expected 401"


@pytest.mark.asyncio
async def test_get_provider_status_endpoint(
    client: AsyncClient, researcher_jwt_headers: dict[str, str]
) -> None:
    """Verify GET /api/v1/providers/status returns provider health and configuration states.

    Args:
        client: Test HTTP client.
        researcher_jwt_headers: Authenticated JWT request headers.
    """
    resp = await client.get("/api/v1/providers/status", headers=researcher_jwt_headers)
    assert resp.status_code == 200
    data = resp.json()

    assert "providers" in data
    providers = data["providers"]
    for pid in ("FRED", "FINNHUB", "NEWSAPI", "POLYGON"):
        assert pid in providers
        item = providers[pid]
        assert "is_configured" in item
        assert "is_healthy" in item
        assert "calls_made" in item
        assert isinstance(item["is_healthy"], bool)


@pytest.mark.asyncio
async def test_macro_sync_endpoint(
    client: AsyncClient, researcher_jwt_headers: dict[str, str]
) -> None:
    """Verify POST /api/v1/providers/macro/sync returns yield spread and fed funds rate.

    Args:
        client: Test HTTP client.
        researcher_jwt_headers: Authenticated JWT request headers.
    """
    resp = await client.post("/api/v1/providers/macro/sync", headers=researcher_jwt_headers)
    assert resp.status_code == 200
    data = resp.json()

    assert "yield_spread_10y_2y" in data
    assert "fed_funds_rate" in data
    assert "source" in data
    assert "status" in data
    assert data["status"] == "synced"
    assert isinstance(data["yield_spread_10y_2y"], float)
    assert isinstance(data["fed_funds_rate"], float)


@pytest.mark.asyncio
async def test_company_news_endpoint(
    client: AsyncClient, researcher_jwt_headers: dict[str, str]
) -> None:
    """Verify GET /api/v1/providers/company-news/{symbol} returns news list.

    Args:
        client: Test HTTP client.
        researcher_jwt_headers: Authenticated JWT request headers.
    """
    resp = await client.get("/api/v1/providers/company-news/AAPL", headers=researcher_jwt_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)

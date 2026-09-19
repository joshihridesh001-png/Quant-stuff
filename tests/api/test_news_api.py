"""Integration tests for Financial News & Price Reaction REST API endpoints.

Governing Standards:
- Rules.md (Rule 1: Line annotations, Rule 2: Diagnostic error codes, Rule 3: Quality gates)
- Endpoints: GET /api/v1/news/latest, POST /api/v1/news/harvest, POST /api/v1/news/predict
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_unauthenticated_news_endpoints_rejected(client: AsyncClient) -> None:
    """Verify all news prediction endpoints return 401 when unauthenticated.

    Args:
        client: Test HTTP client.
    """
    routes = [
        ("GET", "/api/v1/news/latest"),
        ("POST", "/api/v1/news/harvest"),
        ("POST", "/api/v1/news/predict"),
    ]
    for method, path in routes:
        resp = await client.request(method, path)
        assert resp.status_code == 401, f"{method} {path} returned {resp.status_code}, expected 401"


@pytest.mark.asyncio
async def test_predict_headline_endpoint(
    client: AsyncClient, researcher_jwt_headers: dict[str, str]
) -> None:
    """Verify POST /api/v1/news/predict forecasts price shock and breakout probabilities.

    Args:
        client: Test HTTP client.
        researcher_jwt_headers: Authenticated JWT request headers.
    """
    payload = {
        "headline": "Apple Q4 iPhone revenue surges 12% beating Wall Street expectations",
        "summary": "Services revenue hit all-time high with expanding operating margin.",
        "ticker": "AAPL",
        "current_price": 220.0,
        "volatility": 0.018,
    }

    resp = await client.post("/api/v1/news/predict", json=payload, headers=researcher_jwt_headers)
    assert resp.status_code == 200
    data = resp.json()

    assert isinstance(data, list)
    assert len(data) >= 1
    pred = data[0]

    assert pred["ticker"] == "AAPL"
    assert pred["current_price"] == 220.0
    assert pred["expected_delta_price"] > 0.0
    assert pred["target_price"] > pred["current_price"]
    assert pred["prob_up"] > 0.60
    assert pred["signal"] == "BUY"
    assert pred["confidence"] > 0.20
    assert pred["barrier_upper"] > pred["current_price"]
    assert pred["barrier_lower"] < pred["current_price"]
    assert len(pred["predicted_trajectory"]) >= 5


@pytest.mark.asyncio
async def test_harvest_and_latest_news_endpoints(
    client: AsyncClient, researcher_jwt_headers: dict[str, str]
) -> None:
    """Verify POST /api/v1/news/harvest and GET /api/v1/news/latest endpoints.

    Args:
        client: Test HTTP client.
        researcher_jwt_headers: Authenticated JWT request headers.
    """
    # 1. Trigger harvest cycle
    harvest_resp = await client.post("/api/v1/news/harvest", headers=researcher_jwt_headers)
    assert harvest_resp.status_code == 200
    harvest_data = harvest_resp.json()
    assert isinstance(harvest_data, list)

    # 2. Query latest predictions
    latest_resp = await client.get("/api/v1/news/latest?limit=10", headers=researcher_jwt_headers)
    assert latest_resp.status_code == 200
    latest_data = latest_resp.json()
    assert isinstance(latest_data, list)
    assert len(latest_data) >= len(harvest_data)


@pytest.mark.asyncio
async def test_predict_invalid_input_rejection(
    client: AsyncClient, researcher_jwt_headers: dict[str, str]
) -> None:
    """Verify POST /api/v1/news/predict rejects invalid prices or empty headlines.

    Args:
        client: Test HTTP client.
        researcher_jwt_headers: Authenticated JWT request headers.
    """
    # Empty headline
    resp = await client.post(
        "/api/v1/news/predict",
        json={"headline": "", "current_price": 100.0, "volatility": 0.02},
        headers=researcher_jwt_headers,
    )
    assert resp.status_code in (400, 422)

    # Negative price
    resp2 = await client.post(
        "/api/v1/news/predict",
        json={"headline": "Valid headline", "current_price": -10.0, "volatility": 0.02},
        headers=researcher_jwt_headers,
    )
    assert resp2.status_code in (400, 422)

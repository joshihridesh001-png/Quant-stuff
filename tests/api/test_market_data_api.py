"""API tests for market data endpoints: batch ingestion, range queries, and volatility summaries.

Purpose: Verifies HTTP status codes, authorization gates, schema validation, and error responses.
Dependencies: pytest, httpx.AsyncClient, conftest fixtures.
Relationship: Validates /api/v1/market-data HTTP contracts.
Invariants: Ingestion requires X-API-Key; queries require Bearer JWT.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_batch_ingest_bars_unauthorized(client: AsyncClient) -> None:
    """Verify market data ingestion is rejected without valid X-API-Key."""
    payload = {
        "resolution": "1m",
        "bars": [
            {
                "asset_id": "SPY",
                "timestamp": 1000,
                "open": 450.0,
                "high": 455.0,
                "low": 449.0,
                "close": 452.0,
                "volume": 10000.0,
                "vwap": 451.0,
                "resolution": "1m",
            }
        ],
    }
    response = await client.post("/api/v1/market-data/bars/batch", json=payload)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_batch_ingest_bars_success(
    client: AsyncClient, api_key_headers: dict[str, str]
) -> None:
    """Verify authorized batch ingestion returns 201 and persisted counts."""
    payload = {
        "resolution": "1m",
        "bars": [
            {
                "asset_id": "SPY",
                "timestamp": 1000,
                "open": 450.0,
                "high": 455.0,
                "low": 449.0,
                "close": 452.0,
                "volume": 10000.0,
                "vwap": 451.0,
                "resolution": "1m",
            },
            {
                "asset_id": "SPY",
                "timestamp": 2000,
                "open": 452.0,
                "high": 456.0,
                "low": 451.0,
                "close": 454.0,
                "volume": 12000.0,
                "vwap": 453.5,
                "resolution": "1m",
            },
        ],
    }
    response = await client.post(
        "/api/v1/market-data/bars/batch", json=payload, headers=api_key_headers
    )
    assert response.status_code == 201
    data = response.json()
    assert data["processed_count"] == 2
    assert data["persisted_count"] == 2
    assert data["resolution"] == "1m"


@pytest.mark.asyncio
async def test_batch_ingest_invalid_resolution(
    client: AsyncClient, api_key_headers: dict[str, str]
) -> None:
    """Verify invalid resolution string produces 422 Unprocessable Entity."""
    payload = {
        "resolution": "999years",
        "bars": [
            {
                "asset_id": "SPY",
                "timestamp": 1000,
                "open": 450.0,
                "high": 455.0,
                "low": 449.0,
                "close": 452.0,
                "volume": 10000.0,
                "vwap": 451.0,
                "resolution": "1m",
            }
        ],
    }
    response = await client.post(
        "/api/v1/market-data/bars/batch", json=payload, headers=api_key_headers
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_batch_ingest_invalid_price_invariants(
    client: AsyncClient, api_key_headers: dict[str, str]
) -> None:
    """Verify mathematically invalid bar (High < Low) produces 400 Bad Request."""
    payload = {
        "resolution": "1m",
        "bars": [
            {
                "asset_id": "SPY",
                "timestamp": 1000,
                "open": 450.0,
                "high": 440.0,  # Invalid: high < open
                "low": 449.0,
                "close": 445.0,
                "volume": 1000.0,
                "vwap": 445.0,
                "resolution": "1m",
            }
        ],
    }
    response = await client.post(
        "/api/v1/market-data/bars/batch", json=payload, headers=api_key_headers
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_get_bars_range_unauthorized(client: AsyncClient) -> None:
    """Verify range query endpoint rejects unauthenticated access."""
    response = await client.get("/api/v1/market-data/bars?asset_id=SPY&start_time=100&end_time=500")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_bars_range_and_latest_lifecycle(
    client: AsyncClient,
    api_key_headers: dict[str, str],
    researcher_jwt_headers: dict[str, str],
) -> None:
    """Verify end-to-end lifecycle: ingestion followed by range and latest queries with volatility."""
    # Ingest 25 continuous bars
    bars = [
        {
            "asset_id": "QQQ",
            "timestamp": 1000 * (i + 1),
            "open": 350.0 + i,
            "high": 352.0 + i,
            "low": 349.0 + i,
            "close": 351.0 + i,
            "volume": 5000.0,
            "vwap": 350.5 + i,
            "resolution": "1m",
        }
        for i in range(25)
    ]
    ingest_res = await client.post(
        "/api/v1/market-data/bars/batch",
        json={"resolution": "1m", "bars": bars},
        headers=api_key_headers,
    )
    assert ingest_res.status_code == 201

    # Query range covering all bars
    range_res = await client.get(
        "/api/v1/market-data/bars?asset_id=QQQ&start_time=1000&end_time=25000&resolution=1m",
        headers=researcher_jwt_headers,
    )
    assert range_res.status_code == 200
    range_data = range_res.json()
    assert range_data["asset_id"] == "QQQ"
    assert range_data["count"] == 25
    assert range_data["start_time"] == 1000
    assert range_data["end_time"] == 25000
    assert range_data["latest_close"] == 375.0
    assert range_data["realized_volatility_latest"] is not None

    # Query latest 5 bars
    latest_res = await client.get(
        "/api/v1/market-data/bars/latest?asset_id=QQQ&count=5&resolution=1m",
        headers=researcher_jwt_headers,
    )
    assert latest_res.status_code == 200
    latest_data = latest_res.json()
    assert latest_data["count"] == 5
    assert latest_data["latest_close"] == 375.0


@pytest.mark.asyncio
async def test_get_bars_invalid_time_range(
    client: AsyncClient, researcher_jwt_headers: dict[str, str]
) -> None:
    """Verify range query with start_time > end_time produces 400 Bad Request."""
    response = await client.get(
        "/api/v1/market-data/bars?asset_id=QQQ&start_time=5000&end_time=1000",
        headers=researcher_jwt_headers,
    )
    assert response.status_code == 400

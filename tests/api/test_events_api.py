"""API tests for news ingestion and state vector endpoints."""

from datetime import UTC, datetime

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_ingest_event_unauthorized(client: AsyncClient) -> None:
    payload = {
        "headline": "Fed signals neutral policy stance",
        "raw_text": "Federal Reserve officials noted balanced inflation trends today.",
        "timestamp": datetime.now(UTC).isoformat(),
        "ticker_weights": {"SPY": 1.0},
    }
    response = await client.post("/api/v1/events/ingest", json=payload)
    assert response.status_code == 401
    assert "Invalid or missing X-API-Key" in response.json()["detail"]


@pytest.mark.asyncio
async def test_ingest_event_and_retrieve_lifecycle(
    client: AsyncClient, api_key_headers: dict
) -> None:
    now_iso = datetime.now(UTC).isoformat()
    payload = {
        "headline": "Semiconductor manufacturing subsidies approved in Senate",
        "raw_text": "Comprehensive legislation providing forty billion in direct factory funding.",
        "timestamp": now_iso,
        "ticker_weights": {"NVDA": 0.8, "TSM": 0.9},
        "dense_embedding": [0.12, -0.45, 0.88],
        "sentiment_polarity": 0.70,
        "sentiment_subjectivity": 0.30,
        "sentiment_novelty": 0.85,
        "urgency": 0.90,
        "source": "BLOOMBERG",
    }

    # 1. Ingest Event
    create_res = await client.post("/api/v1/events/ingest", json=payload, headers=api_key_headers)
    assert create_res.status_code == 201
    data = create_res.json()
    event_id = data["id"]
    assert data["headline"] == payload["headline"]
    assert data["sentiment_polarity"] == 0.70
    assert "X-Request-ID" in create_res.headers
    assert "X-Process-Time-Ms" in create_res.headers

    # 2. Retrieve Event by ID
    get_res = await client.get(f"/api/v1/events/{event_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == event_id

    # 3. Query Active News State for Ticker
    state_res = await client.get("/api/v1/events/state/NVDA")
    assert state_res.status_code == 200
    state_data = state_res.json()
    assert state_data["ticker"] == "NVDA"
    assert state_data["event_count"] == 1
    assert len(state_data["state_vector"]) > 0


@pytest.mark.asyncio
async def test_get_nonexistent_event(client: AsyncClient) -> None:
    fake_uuid = "00000000-0000-0000-0000-000000000000"
    response = await client.get(f"/api/v1/events/{fake_uuid}")
    assert response.status_code == 404
    assert f"Event {fake_uuid} not found" in response.json()["detail"]

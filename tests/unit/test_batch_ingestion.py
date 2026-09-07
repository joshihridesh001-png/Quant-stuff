"""Tests for batch event ingestion and multi-asset query capabilities."""

from datetime import UTC, datetime

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_batch_event_ingest_api(client: AsyncClient, api_key_headers: dict[str, str]) -> None:
    batch_payload = {
        "events": [
            {
                "headline": "Fed Signals Interest Rate Pause Amid Moderate Inflation",
                "raw_text": "Federal Reserve officials indicated interest rates will remain steady.",
                "timestamp": datetime.now(UTC).isoformat(),
                "ticker_weights": {"SPY": 0.8, "QQQ": 0.6},
                "sentiment_polarity": 0.45,
                "urgency": 0.7,
                "source": "REUTERS",
            },
            {
                "headline": "Tech Sector Leads Rally on Semiconductor Earnings Beat",
                "raw_text": "Strong semiconductor demand drove major chipmakers to report record revenues.",
                "timestamp": datetime.now(UTC).isoformat(),
                "ticker_weights": {"QQQ": 0.9, "NVDA": 1.0},
                "sentiment_polarity": 0.8,
                "urgency": 0.9,
                "source": "BLOOMBERG",
            },
        ]
    }

    response = await client.post(
        "/api/v1/events/batch", json=batch_payload, headers=api_key_headers
    )
    assert response.status_code == 201
    data = response.json()
    assert data["ingested_count"] == 2
    assert len(data["events"]) == 2
    assert (
        data["events"][0]["headline"] == "Fed Signals Interest Rate Pause Amid Moderate Inflation"
    )
    assert data["events"][1]["headline"] == "Tech Sector Leads Rally on Semiconductor Earnings Beat"


@pytest.mark.asyncio
async def test_batch_event_ingest_unauthorized(client: AsyncClient) -> None:
    response = await client.post("/api/v1/events/batch", json={"events": []})
    assert response.status_code == 401

"""API tests for Econometric Stationarity Rig, FFD, Volatility, and Triple Barrier endpoints."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_ffd_search_unauthorized(client: AsyncClient) -> None:
    """Ensure FFD search endpoint rejects requests without authenticated JWT."""
    response = await client.get("/api/v1/econometrics/ffd/search?symbol=AAPL")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_ffd_search_success(
    client: AsyncClient, researcher_jwt_headers: dict[str, str]
) -> None:
    """Test successful FFD stationarity search returning d in [0, 1] points and optimal d*."""
    response = await client.get(
        "/api/v1/econometrics/ffd/search?symbol=NVDA&threshold=0.05&bar_count=200",
        headers=researcher_jwt_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "NVDA"
    assert 0.0 <= data["optimal_d"] <= 1.0
    assert data["threshold_pvalue"] == 0.05
    assert len(data["points"]) > 0
    first_pt = data["points"][0]
    assert "d" in first_pt
    assert "adf_stat" in first_pt
    assert "adf_pvalue" in first_pt
    assert "correlation" in first_pt
    assert "is_stationary" in first_pt


@pytest.mark.asyncio
async def test_realized_volatility_success(
    client: AsyncClient, researcher_jwt_headers: dict[str, str]
) -> None:
    """Test realized Parkinson and Garman-Klass volatility endpoint."""
    response = await client.get(
        "/api/v1/econometrics/volatility/parkinson?symbol=NVDA&window=20&bar_count=100",
        headers=researcher_jwt_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "NVDA"
    assert data["window"] == 20
    assert data["latest_close"] > 0.0
    assert data["annualized_parkinson_pct"] >= 0.0
    assert data["annualized_garman_klass_pct"] >= 0.0
    assert len(data["series"]) > 0
    first_series = data["series"][0]
    assert "timestamp_ns" in first_series
    assert "close" in first_series
    assert "parkinson_vol" in first_series
    assert "garman_klass_vol" in first_series


@pytest.mark.asyncio
async def test_triple_barrier_simulate_success(
    client: AsyncClient, researcher_jwt_headers: dict[str, str]
) -> None:
    """Test dynamic volatility triple-barrier simulation endpoint."""
    payload = {
        "symbol": "NVDA",
        "profit_multiplier": 2.0,
        "stop_multiplier": 1.0,
        "horizon_bars": 20,
        "volatility_window": 15,
        "side": 1,
    }
    response = await client.post(
        "/api/v1/econometrics/triple-barrier/simulate",
        json=payload,
        headers=researcher_jwt_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "NVDA"
    assert data["total_events"] > 0
    assert data["take_profit_hits"] >= 0
    assert data["stop_loss_hits"] >= 0
    assert data["vertical_expiration_hits"] >= 0
    assert round(
        data["take_profit_pct"] + data["stop_loss_pct"] + data["vertical_expiration_pct"]
    ) in (99, 100, 101)
    assert len(data["sample_trajectory"]) > 0
    point = data["sample_trajectory"][0]
    assert "bar_index" in point
    assert "close_price" in point
    assert "upper_barrier" in point
    assert "lower_barrier" in point

"""API tests for Bayesian Game Theory, Regime Inference, and Minimax Regret endpoints."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_regimes_unauthorized(client: AsyncClient) -> None:
    """Ensure regimes endpoint requires authenticated JWT."""
    response = await client.get("/api/v1/game-theory/regimes?symbol=NVDA")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_regimes_success(
    client: AsyncClient, researcher_jwt_headers: dict[str, str]
) -> None:
    """Test successful 3-simplex regime estimation and CUSUM jump detection."""
    response = await client.get(
        "/api/v1/game-theory/regimes?symbol=NVDA&cusum_threshold=3.0&cusum_drift=0.5&bar_count=100",
        headers=researcher_jwt_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "NVDA"
    assert data["current_regime"] in (
        "LOW_VOL_ABSORPTION",
        "MOMENTUM_CASCADE",
        "PANIC_LIQUIDITY_TRAP",
    )
    # Simplex conservation: sum equals 1.0
    prob_sum = data["p_absorption"] + data["p_momentum"] + data["p_panic"]
    assert round(prob_sum, 2) == 1.0
    assert len(data["dirichlet_alphas"]) == 3
    assert data["confidence_pct"] >= 0.0
    assert len(data["cusum_series"]) > 0
    assert len(data["regime_history"]) > 0


@pytest.mark.asyncio
async def test_payoff_matrix_success(
    client: AsyncClient, researcher_jwt_headers: dict[str, str]
) -> None:
    """Test Stackelberg Minimax Regret payoff matrix calculation."""
    payload = {
        "ambiguity_beta": 1.5,
        "risk_aversion": 2.0,
    }
    response = await client.post(
        "/api/v1/game-theory/payoff-matrix",
        json=payload,
        headers=researcher_jwt_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["actions"]) == 3
    assert len(data["counterparties"]) == 3
    assert len(data["payoff_matrix"]) == 3
    assert len(data["regret_matrix"]) == 3
    assert len(data["worst_case_regrets"]) == 3
    assert data["optimal_action"] in data["actions"]
    # All regrets >= 0
    for row in data["regret_matrix"]:
        for val in row:
            assert val >= 0.0
    # Worst case counterparty probabilities sum to 1.0
    assert round(sum(data["worst_case_counterparty_probs"]), 2) == 1.0

"""API tests for evolutionary strategy genotype endpoints."""

import pytest
from httpx import AsyncClient

from quant.core.security import create_access_token


@pytest.mark.asyncio
async def test_create_and_evaluate_genotype_lifecycle(client: AsyncClient) -> None:
    # 1. Register Genotype
    payload = {
        "generation": 0,
        "cohort": "ASPIRANT",
        "chromosome_repr": {"tau_fast": 3600.0, "tau_slow": 86400.0, "alpha": 0.5},
        "chromosome_game": {"risk_aversion_lambda": 1.2, "belief_priors": [0.4, 0.4, 0.2]},
        "chromosome_infer": {"model_depth": 4, "threshold": 0.55},
        "chromosome_risk": {"vol_target": 0.15, "max_drawdown_limit": 0.08},
    }
    create_res = await client.post("/api/v1/genotypes", json=payload)
    assert create_res.status_code == 201
    created_data = create_res.json()
    genotype_id = created_data["id"]
    assert created_data["generation"] == 0
    assert created_data["cohort"] == "ASPIRANT"

    # 2. Evaluate Genotype
    eval_payload = {
        "deflated_sharpe": 1.95,
        "max_drawdown": 0.04,
        "regret_score": 0.85,
        "novelty_score": 0.50,
    }
    eval_res = await client.post(f"/api/v1/genotypes/{genotype_id}/evaluate", json=eval_payload)
    assert eval_res.status_code == 200
    assert eval_res.json()["genotype_id"] == genotype_id
    assert eval_res.json()["fitness_score"] > 0.0


@pytest.mark.asyncio
async def test_seed_population_rbac_guards(
    client: AsyncClient, researcher_jwt_headers: dict
) -> None:
    # 1. Unauthenticated Request -> 401
    unauth_res = await client.post("/api/v1/genotypes/seed", json={"population_size": 10})
    assert unauth_res.status_code == 401

    # 2. Insufficient Role -> 403
    guest_token = create_access_token(payload={"sub": "guest-user", "role": "GUEST"})
    forbidden_res = await client.post(
        "/api/v1/genotypes/seed",
        json={"population_size": 10},
        headers={"Authorization": f"Bearer {guest_token}"},
    )
    assert forbidden_res.status_code == 403
    assert "Insufficient permissions" in forbidden_res.json()["detail"]

    # 3. Authorized RESEARCHER -> 201
    seed_res = await client.post(
        "/api/v1/genotypes/seed",
        json={"population_size": 10},
        headers=researcher_jwt_headers,
    )
    assert seed_res.status_code == 201
    population = seed_res.json()
    assert len(population) == 10

    # 4. Check Alpha Cohort Endpoint
    alpha_res = await client.get("/api/v1/genotypes/alpha")
    assert alpha_res.status_code == 200
    alphas = alpha_res.json()
    assert len(alphas) >= 2
    for a in alphas:
        assert a["cohort"] == "ALPHA"

    # 5. Check Pareto Ranking Endpoint
    pareto_res = await client.get("/api/v1/genotypes/pareto?generation=0")
    assert pareto_res.status_code == 200
    ranked = pareto_res.json()
    assert len(ranked) == 10
    assert "pareto_rank" in ranked[0]
    assert "crowding_distance" in ranked[0]

"""Evolutionary strategy population and fitness evaluation endpoints."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from quant.api.dependencies import get_genotype_service, require_role
from quant.api.v1.schemas import (
    GenotypeCreateRequest,
    GenotypeEvaluateRequest,
    GenotypeResponse,
    ParetoRankedGenotypeResponse,
    PopulationSeedRequest,
)
from quant.domain.models import Genotype, GenotypeCohort
from quant.services.genotype_service import GenotypeService

router = APIRouter(prefix="/genotypes", tags=["Evolutionary Population"])


def _to_response(genotype: Genotype) -> GenotypeResponse:
    return GenotypeResponse(
        id=genotype.id,
        generation=genotype.generation,
        cohort=genotype.cohort.value,
        chromosome_repr=genotype.chromosome_repr,
        chromosome_game=genotype.chromosome_game,
        chromosome_infer=genotype.chromosome_infer,
        chromosome_risk=genotype.chromosome_risk,
        fitness_score=genotype.fitness_score,
        deflated_sharpe=genotype.deflated_sharpe,
        max_drawdown=genotype.max_drawdown,
        regret_score=genotype.regret_score,
        novelty_score=genotype.novelty_score,
        created_at=genotype.created_at,
    )


@router.post(
    "",
    response_model=GenotypeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new strategy chromosome",
)
async def create_genotype(
    payload: GenotypeCreateRequest,
    service: GenotypeService = Depends(get_genotype_service),
) -> GenotypeResponse:
    """Insert a strategy individual into the evolutionary population."""
    genotype = await service.register_genotype(
        generation=payload.generation,
        cohort=GenotypeCohort(payload.cohort),
        chromosome_repr=payload.chromosome_repr,
        chromosome_game=payload.chromosome_game,
        chromosome_infer=payload.chromosome_infer,
        chromosome_risk=payload.chromosome_risk,
    )
    return _to_response(genotype)


@router.post(
    "/seed",
    response_model=list[GenotypeResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Seed generation 0 population",
)
async def seed_population(
    payload: PopulationSeedRequest,
    service: GenotypeService = Depends(get_genotype_service),
    _user: dict[str, Any] = Depends(require_role(["ADMIN", "RESEARCHER"])),
) -> list[GenotypeResponse]:
    """Seed generation 0 with stratified Alpha and Aspirant chromosomes."""
    population = await service.seed_initial_population(population_size=payload.population_size)
    return [_to_response(g) for g in population]


@router.get(
    "/alpha",
    response_model=list[GenotypeResponse],
    summary="Retrieve elite Alpha cohort",
)
async def get_alpha_cohort(
    limit: int = Query(20, ge=1, le=100),
    service: GenotypeService = Depends(get_genotype_service),
) -> list[GenotypeResponse]:
    """Fetch top performing Alpha cohort members sorted by fitness."""
    alphas = await service.get_alpha_cohort(limit=limit)
    return [_to_response(g) for g in alphas]


@router.get(
    "/pareto",
    response_model=list[ParetoRankedGenotypeResponse],
    summary="Retrieve population sorted by NSGA-II Pareto dominance and crowding distance",
)
async def get_pareto_ranking(
    generation: int = Query(0, ge=0, description="Generation epoch to rank"),
    service: GenotypeService = Depends(get_genotype_service),
) -> list[ParetoRankedGenotypeResponse]:
    """Calculate and return NSGA-II non-dominated Pareto fronts for a given generation."""
    population = await service.genotype_repo.get_generation(generation)
    ranked = service.rank_population_pareto(population)
    return [
        ParetoRankedGenotypeResponse(
            genotype=_to_response(g),
            pareto_rank=rank,
            crowding_distance=dist if dist != float("inf") else 1e9,
        )
        for g, rank, dist in ranked
    ]


@router.post(
    "/{genotype_id}/evaluate",
    response_model=dict[str, Any],
    summary="Record multi-objective fitness evaluation",
)
async def evaluate_genotype(
    genotype_id: UUID,
    payload: GenotypeEvaluateRequest,
    service: GenotypeService = Depends(get_genotype_service),
) -> dict[str, Any]:
    """Evaluate and persist Deflated Sharpe, MaxDD, and composite fitness."""
    existing = await service.genotype_repo.get_by_id(genotype_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Genotype {genotype_id} not found",
        )

    fitness = await service.evaluate_genotype(
        genotype_id=genotype_id,
        deflated_sharpe=payload.deflated_sharpe,
        max_drawdown=payload.max_drawdown,
        regret_score=payload.regret_score,
        novelty_score=payload.novelty_score,
    )
    return {"genotype_id": str(genotype_id), "fitness_score": fitness}

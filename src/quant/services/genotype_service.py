"""Application service managing evolutionary strategy chromosomes and fitness evaluations."""

import math
from typing import Any
from uuid import UUID

from quant.domain.interfaces import IGenotypeRepository
from quant.domain.models import Genotype, GenotypeCohort


def compute_multiobjective_fitness(
    deflated_sharpe: float,
    max_drawdown: float,
    regret_score: float = 0.0,
    novelty_score: float = 0.0,
    psi: float = 2.0,
    omega1: float = 0.3,
    omega2: float = 0.2,
) -> float:
    """Compute institutional multi-objective fitness function F(I_i).

    F(I_i) = DeflatedSharpe * exp(-psi * MaxDD) + omega1 * V_i(Regret) + omega2 * H_novelty
    """
    dd_penalty = math.exp(-psi * max(max_drawdown, 0.0))
    fitness = (deflated_sharpe * dd_penalty) + (omega1 * regret_score) + (omega2 * novelty_score)
    return float(fitness)


class GenotypeService:
    """Service orchestrating population storage, fitness scoring, and cohort queries."""

    def __init__(self, genotype_repo: IGenotypeRepository) -> None:
        self.genotype_repo = genotype_repo

    async def register_genotype(
        self,
        generation: int,
        cohort: GenotypeCohort,
        chromosome_repr: dict[str, Any],
        chromosome_game: dict[str, Any],
        chromosome_infer: dict[str, Any],
        chromosome_risk: dict[str, Any],
    ) -> Genotype:
        """Persist a new algorithmic individual into the population."""
        genotype = Genotype(
            generation=generation,
            cohort=cohort,
            chromosome_repr=chromosome_repr,
            chromosome_game=chromosome_game,
            chromosome_infer=chromosome_infer,
            chromosome_risk=chromosome_risk,
        )
        return await self.genotype_repo.add(genotype)

    async def evaluate_genotype(
        self,
        genotype_id: UUID,
        deflated_sharpe: float,
        max_drawdown: float,
        regret_score: float = 0.0,
        novelty_score: float = 0.0,
        psi: float = 2.0,
        omega1: float = 0.3,
        omega2: float = 0.2,
    ) -> float:
        """Calculate multi-objective fitness and persist evaluated scores."""
        fitness = compute_multiobjective_fitness(
            deflated_sharpe=deflated_sharpe,
            max_drawdown=max_drawdown,
            regret_score=regret_score,
            novelty_score=novelty_score,
            psi=psi,
            omega1=omega1,
            omega2=omega2,
        )
        await self.genotype_repo.update_evaluation(
            genotype_id=genotype_id,
            fitness_score=fitness,
            deflated_sharpe=deflated_sharpe,
            max_drawdown=max_drawdown,
            regret_score=regret_score,
        )
        return fitness

    async def get_alpha_cohort(self, limit: int = 50) -> list[Genotype]:
        """Fetch current elite Alpha cohort sorted by fitness."""
        return await self.genotype_repo.get_alpha_cohort(limit=limit)

    async def seed_initial_population(self, population_size: int = 20) -> list[Genotype]:
        """Seed generation 0 with baseline parameter configurations."""
        alpha_count = max(int(population_size * 0.2), 1)
        created: list[Genotype] = []

        for i in range(population_size):
            cohort = GenotypeCohort.ALPHA if i < alpha_count else GenotypeCohort.ASPIRANT
            genotype = await self.register_genotype(
                generation=0,
                cohort=cohort,
                chromosome_repr={
                    "tau_fast": 3600.0 * (1.0 + (i * 0.1)),
                    "tau_slow": 86400.0 * (1.0 + (i * 0.05)),
                    "alpha": 0.5,
                },
                chromosome_game={
                    "risk_aversion_lambda": 1.0 + (i * 0.05),
                    "belief_prior": [0.33, 0.33, 0.34],
                },
                chromosome_infer={"model_depth": 3, "sensitivity": 0.5},
                chromosome_risk={"vol_target": 0.15, "max_drawdown_limit": 0.10},
            )
            created.append(genotype)

        return created

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


def fast_non_dominated_sort(objectives: list[list[float]]) -> list[list[int]]:
    """Fast Non-dominated Sorting Algorithm (NSGA-II).

    Partitions population indices into Pareto dominance fronts [F_0, F_1, ...].
    Each objective vector is assumed to be maximized (higher is better).
    Returns list of fronts where F_0 is the non-dominated Pareto-optimal elite front.
    """
    n = len(objectives)
    if n == 0:
        return []

    domination_counts = [0] * n
    dominated_sets: list[list[int]] = [[] for _ in range(n)]
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            p_dominates_q = True
            q_dominates_p = True
            p_better_at_least_one = False
            q_better_at_least_one = False

            for f_idx in range(len(objectives[p])):
                val_p = objectives[p][f_idx]
                val_q = objectives[q][f_idx]
                if val_p < val_q:
                    p_dominates_q = False
                elif val_p > val_q:
                    p_better_at_least_one = True

                if val_q < val_p:
                    q_dominates_p = False
                elif val_q > val_p:
                    q_better_at_least_one = True

            if p_dominates_q and p_better_at_least_one:
                dominated_sets[p].append(q)
            elif q_dominates_p and q_better_at_least_one:
                domination_counts[p] += 1

        if domination_counts[p] == 0:
            fronts[0].append(p)

    curr_front = 0
    while fronts[curr_front]:
        next_front: list[int] = []
        for p in fronts[curr_front]:
            for q in dominated_sets[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    next_front.append(q)
        curr_front += 1
        fronts.append(next_front)

    if not fronts[-1]:
        fronts.pop()

    return fronts


def calculate_crowding_distance(
    front: list[int],
    objectives: list[list[float]],
) -> dict[int, float]:
    """Calculate crowding distance for individuals within a Pareto front to maintain diversity."""
    distance: dict[int, float] = dict.fromkeys(front, 0.0)
    n_front = len(front)
    if n_front <= 2:
        for idx in front:
            distance[idx] = float("inf")
        return distance

    num_objectives = len(objectives[0])
    for m in range(num_objectives):
        sorted_front = sorted(front, key=lambda idx: objectives[idx][m])
        distance[sorted_front[0]] = float("inf")
        distance[sorted_front[-1]] = float("inf")

        f_min = objectives[sorted_front[0]][m]
        f_max = objectives[sorted_front[-1]][m]
        diff = f_max - f_min
        if diff <= 1e-8:
            continue

        for i in range(1, n_front - 1):
            distance[sorted_front[i]] += (
                objectives[sorted_front[i + 1]][m] - objectives[sorted_front[i - 1]][m]
            ) / diff

    return distance


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

    def rank_population_pareto(
        self,
        genotypes: list[Genotype],
    ) -> list[tuple[Genotype, int, float]]:
        """Rank a population using NSGA-II non-dominated sorting and crowding distance.

        Returns list of tuples: (genotype, pareto_rank, crowding_distance).
        Rank 0 is the non-dominated Pareto-optimal elite front.
        """
        if not genotypes:
            return []

        # Construct objective vectors: [DSR (max), -MaxDD (max), Regret (max), Novelty (max)]
        objectives: list[list[float]] = []
        for g in genotypes:
            dsr = g.deflated_sharpe if g.deflated_sharpe is not None else 0.0
            mdd = g.max_drawdown if g.max_drawdown is not None else 1.0
            regret = g.regret_score if g.regret_score is not None else 0.0
            novelty = g.novelty_score if g.novelty_score is not None else 0.0
            objectives.append([dsr, -mdd, regret, novelty])

        fronts = fast_non_dominated_sort(objectives)
        ranked_results: list[tuple[Genotype, int, float]] = []

        for rank, front in enumerate(fronts):
            crowding_distances = calculate_crowding_distance(front, objectives)
            sorted_front = sorted(front, key=lambda idx: crowding_distances[idx], reverse=True)
            for idx in sorted_front:
                ranked_results.append((genotypes[idx], rank, crowding_distances[idx]))

        return ranked_results

    def select_aspirant_by_novelty(
        self,
        aspirants: list[Genotype],
        elite_cohort: list[Genotype],
    ) -> Genotype | None:
        """Select an aspirant that maximizes behavioral novelty distance from elite cohort.

        Replaces arbitrary random fallback selection with structured novelty search,
        preventing genetic drift and population degeneracy.
        """
        if not aspirants:
            return None
        if not elite_cohort:
            return aspirants[0]

        best_aspirant = aspirants[0]
        max_novelty = -1.0

        for asp in aspirants:
            total_dist = 0.0
            for elite in elite_cohort:
                t_fast_diff = (
                    abs(
                        asp.chromosome_repr.get("tau_fast", 3600.0)
                        - elite.chromosome_repr.get("tau_fast", 3600.0)
                    )
                    / 3600.0
                )
                t_slow_diff = (
                    abs(
                        asp.chromosome_repr.get("tau_slow", 86400.0)
                        - elite.chromosome_repr.get("tau_slow", 86400.0)
                    )
                    / 86400.0
                )
                total_dist += t_fast_diff + t_slow_diff

            avg_dist = total_dist / len(elite_cohort)
            if avg_dist > max_novelty:
                max_novelty = avg_dist
                best_aspirant = asp

        return best_aspirant

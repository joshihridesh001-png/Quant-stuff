"""Hypergamic Assortative Selection & Residual Orthogonality Gating Engine.

This module implements the evolutionary mating and reproduction architecture
for financial strategy discovery, replacing naive genetic algorithm crossovers with:
1. Front-Preserving Pareto Cohort Stratification (Alpha and Aspirant cohorts).
2. Bidirectional Absolute Residual Orthogonality Gating (eliminating direct and inverse clones).
3. Adaptive Assortative Mating with Deadlock Guarantee (bounded relaxation and phenotypic fallback).
4. Asymmetric Latent Unit-Hypercube Crossover (SBX in [0, 1]^20 preserving risk parameters).
5. Elitism Preservation of Front-1 champions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from quant.analytics.chromosomes import StrategyChromosome
from quant.analytics.pareto_sorting import CandidateFitness, RankingResult


# =====================================================================
# Domain Exceptions
# =====================================================================


class HypergamicSelectionError(Exception):
    """Base exception for all hypergamic selection and mating failures."""


class InvalidCohortException(HypergamicSelectionError):
    """Raised when cohort partitioning violates population invariants."""


class MatingDeadlockException(HypergamicSelectionError):
    """Raised when prospective mating partner search deadlocks."""


# =====================================================================
# Domain Dataclasses
# =====================================================================


@dataclass(frozen=True)
class HypergamicConfig:
    """Hyperparameter configuration for hypergamic assortative mating.

    Attributes:
        alpha_ratio: Target proportion of population allocated to Alpha cohort [0.05, 0.90].
        orthogonality_threshold: Minimum required unexplained variance delta_ortho [0.0, 1.0].
        max_mating_attempts: Max attempts before adaptive gate relaxation >= 1.
        relaxation_factor: Multiplier applied to orthogonality threshold per relaxation (0.0, 1.0].
        tournament_size: Size of tournament pool when selecting Aspirant candidates >= 1.
        crossover_distribution_index: Spread parameter eta_c for Simulated Binary Crossover > 0.
        alpha_risk_inheritance_prob: Probability of inheriting risk/game genes from Alpha parent [0.50, 1.0].
        aspirant_repr_inheritance_prob: Probability of inheriting repr/infer genes from Aspirant parent [0.50, 1.0].
        elitism_count: Number of top Front-1 champions preserved without crossover >= 0.
    """

    alpha_ratio: float = 0.25
    orthogonality_threshold: float = 0.30
    max_mating_attempts: int = 5
    relaxation_factor: float = 0.80
    tournament_size: int = 3
    crossover_distribution_index: float = 15.0
    alpha_risk_inheritance_prob: float = 0.75
    aspirant_repr_inheritance_prob: float = 0.65
    elitism_count: int = 2

    def __post_init__(self) -> None:
        """Validate configuration domain bounds and defensive invariants."""
        if not (0.05 <= self.alpha_ratio <= 0.90):
            raise HypergamicSelectionError(
                f"alpha_ratio must be in [0.05, 0.90], got {self.alpha_ratio}"
            )
        if not (0.0 <= self.orthogonality_threshold <= 1.0):
            raise HypergamicSelectionError(
                f"orthogonality_threshold must be in [0.0, 1.0], got {self.orthogonality_threshold}"
            )
        if self.max_mating_attempts < 1:
            raise HypergamicSelectionError(
                f"max_mating_attempts must be >= 1, got {self.max_mating_attempts}"
            )
        if not (0.0 < self.relaxation_factor <= 1.0):
            raise HypergamicSelectionError(
                f"relaxation_factor must be in (0.0, 1.0], got {self.relaxation_factor}"
            )
        if self.tournament_size < 1:
            raise HypergamicSelectionError(
                f"tournament_size must be >= 1, got {self.tournament_size}"
            )
        if not (math.isfinite(self.crossover_distribution_index) and self.crossover_distribution_index > 0.0):
            raise HypergamicSelectionError(
                f"crossover_distribution_index must be > 0, got {self.crossover_distribution_index}"
            )
        if not (0.50 <= self.alpha_risk_inheritance_prob <= 1.0):
            raise HypergamicSelectionError(
                f"alpha_risk_inheritance_prob must be in [0.50, 1.0], got {self.alpha_risk_inheritance_prob}"
            )
        if not (0.50 <= self.aspirant_repr_inheritance_prob <= 1.0):
            raise HypergamicSelectionError(
                f"aspirant_repr_inheritance_prob must be in [0.50, 1.0], got {self.aspirant_repr_inheritance_prob}"
            )
        if self.elitism_count < 0:
            raise HypergamicSelectionError(
                f"elitism_count must be >= 0, got {self.elitism_count}"
            )


@dataclass(frozen=True)
class MatingPair:
    """Record of an accepted hypergamic mating pair.

    Attributes:
        alpha_id: Candidate ID of the Alpha parent.
        aspirant_id: Candidate ID of the Aspirant parent.
        residual_correlation: Empirical sample Pearson correlation between residual vectors.
        is_relaxed: Flag indicating whether threshold was relaxed to admit the pair.
        relaxation_level: Number of adaptive relaxation steps applied (0 = strict).
    """

    alpha_id: str
    aspirant_id: str
    residual_correlation: float
    is_relaxed: bool = False
    relaxation_level: int = 0


@dataclass(frozen=True)
class OffspringResult:
    """Complete reproductive result bundle for generation t + 1.

    Attributes:
        offspring_chromosomes: Decoded StrategyChromosome instances for offspring population.
        mating_pairs: Details of accepted mating partnerships.
        alpha_ids: Unique candidate IDs belonging to the Alpha cohort.
        aspirant_ids: Unique candidate IDs belonging to the Aspirant cohort.
        elite_ids: Candidate IDs directly preserved via elitism without crossover.
        rejection_count: Total number of prospective pairings rejected by the orthogonality gate.
        relaxation_count: Total number of adaptive relaxation steps applied across all pairs.
    """

    offspring_chromosomes: Tuple[StrategyChromosome, ...]
    mating_pairs: Tuple[MatingPair, ...]
    alpha_ids: Tuple[str, ...]
    aspirant_ids: Tuple[str, ...]
    elite_ids: Tuple[str, ...] = ()
    rejection_count: int = 0
    relaxation_count: int = 0


# =====================================================================
# 1. Front-Preserving Pareto Cohort Stratifier
# =====================================================================


class ParetoCohortStratifier:
    """Stratifies a multi-objective ranked population into Alpha and Aspirant cohorts.

    Enforces Front-Preserving Pareto Stratification:
    - Front 1 is the primary core of the Alpha cohort.
    - If |Front 1| >= target_alpha_size, all of Front 1 is retained to prevent
      arbitrarily severing non-dominated solutions.
    - If |Front 1| < target_alpha_size, the deficit is filled using the top
      APD-ranked candidates from Front 2.
    - Infeasible candidates are strictly excluded from both cohorts.
    """

    def __init__(self, config: HypergamicConfig | None = None) -> None:
        """Initialize stratifier with configuration bounds."""
        self._config = config or HypergamicConfig()

    def stratify(
        self,
        candidates: list[CandidateFitness],
        ranking: RankingResult,
    ) -> tuple[list[CandidateFitness], list[CandidateFitness]]:
        """Partition viable candidates into Alpha and Aspirant cohorts.

        Args:
            candidates: Population candidate fitness records.
            ranking: Multi-objective ranking result from Step 2 Pareto sorting.

        Returns:
            Tuple of (alpha_cohort, aspirant_cohort).

        Raises:
            InvalidCohortException: If no viable candidates exist to form cohorts.
        """
        cand_by_id = {c.candidate_id: c for c in candidates if c.is_feasible}
        infeasible_set = set(ranking.infeasible_ids)

        viable_cands = {
            cid: cand
            for cid, cand in cand_by_id.items()
            if cid not in infeasible_set and cand.dsr >= 0.50
        }
        if not viable_cands:
            raise InvalidCohortException(
                "No viable candidates available for hypergamic stratification."
            )

        target_alpha = max(1, math.ceil(self._config.alpha_ratio * len(viable_cands)))

        alpha_ids: list[str] = []

        if not ranking.fronts:
            # Fallback if no fronts present
            sorted_cands = sorted(viable_cands.values(), key=lambda c: c.dsr, reverse=True)
            alpha_cands = sorted_cands[:target_alpha]
            asp_cands = sorted_cands[target_alpha:] or alpha_cands
            return alpha_cands, asp_cands

        # Front 1
        f1 = ranking.fronts[0]
        f1_viable = [cid for cid in f1.candidate_ids if cid in viable_cands]

        if len(f1_viable) >= target_alpha:
            # Front-preserving: retain all of Front 1
            alpha_ids.extend(f1_viable)
        else:
            alpha_ids.extend(f1_viable)
            needed = target_alpha - len(alpha_ids)
            if len(ranking.fronts) > 1:
                f2 = ranking.fronts[1]
                # Sort Front 2 by APD score (lower is better)
                f2_pairs = [
                    (cid, apd)
                    for cid, apd in zip(f2.candidate_ids, f2.apd_scores, strict=False)
                    if cid in viable_cands and cid not in alpha_ids
                ]
                f2_sorted = sorted(f2_pairs, key=lambda x: x[1])
                for cid, _ in f2_sorted[:needed]:
                    alpha_ids.append(cid)

        # Aspirants are all remaining viable candidates
        alpha_set = set(alpha_ids)
        aspirant_ids = [cid for cid in viable_cands if cid not in alpha_set]

        # If aspirant cohort is empty (e.g. whole population in Front 1),
        # fallback to self-mating across the Alpha cohort
        if not aspirant_ids:
            aspirant_ids = list(alpha_ids)

        alphas = [viable_cands[cid] for cid in alpha_ids]
        aspirants = [viable_cands[cid] for cid in aspirant_ids]

        return alphas, aspirants


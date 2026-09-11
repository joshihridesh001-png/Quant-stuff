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

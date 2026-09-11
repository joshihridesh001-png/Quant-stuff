"""Adaptive Volatility Mutation & Generational Lifecycle Engine.

This module provides the closed-loop evolutionary lifecycle architecture for
quantitative strategy discovery, replacing standard fixed GA operators with:
1. Self-Adaptive Truncated Cauchy Mutation in Unit Hypercube space u in [0, 1]^20.
2. Mirror Boundary Reflection eliminating edge-clamping stickiness.
3. Gene-Family Sensitivity Scaling (conservative risk vs. exploratory search).
4. Continuous APD-Progress Rechenberg Volatility Adaptation.
5. Dual-Space Stagnation Monitoring (genotypic hypercube dispersion and phenotypic residual collinearity).
6. Cataclysmic Re-Diversification preserving Front-1 non-dominated champions.
7. (mu + lambda) Environmental Pareto Selection guaranteeing monotonic frontier preservation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quant.analytics.chromosomes import (
    GENE_REGISTRY,
    ChromosomeVectorCodec,
    StrategyChromosome,
)
from quant.analytics.pareto_sorting import CandidateFitness, RankingResult

# =====================================================================
# Domain Exceptions
# =====================================================================


class LifecycleError(Exception):
    """Base exception for all evolutionary lifecycle and mutation failures."""


class InvalidGenerationalStateException(LifecycleError):
    """Raised when generational state transitions violate population or step invariants."""


class StagnationException(LifecycleError):
    """Raised when population stagnation calculations encounter mathematical errors."""


# =====================================================================
# Domain Dataclasses & Configuration
# =====================================================================


@dataclass(frozen=True)
class MutationConfig:
    """Hyperparameter configuration for self-adaptive Cauchy mutation and lifecycle dynamics.

    Attributes:
        initial_step_size: Starting global mutation standard deviation sigma_mut in [min, max].
        min_step_size: Lower clamp bound for mutation step size > 0.
        max_step_size: Upper clamp bound for mutation step size > min_step_size.
        mutation_probability: Probability of mutating an individual gene coordinate in (0, 1].
        cauchy_clipping_bound: Truncation bound theta_clip for standard Cauchy random draws > 0.
        expansion_factor: Multiplier gamma_expand applied to sigma when success ratio > 0.20 (> 1.0).
        contraction_factor: Multiplier gamma_contract applied to sigma when success ratio < 0.20 in (0, 1).
        smoothing_factor: Exponential smoothing weight alpha_smooth for success ratio in (0, 1].
        risk_gene_scale: Sensitivity multiplier kappa_risk for Risk sub-chromosome > 0.
        game_gene_scale: Sensitivity multiplier kappa_game for Game Theory sub-chromosome > 0.
        search_gene_scale: Sensitivity multiplier kappa_search for Repr and Infer sub-chromosomes > 0.
        stagnation_diversity_threshold: Minimum acceptable mean pairwise Euclidean parameter dispersion > 0.
        stagnation_correlation_threshold: Maximum allowable population mean residual correlation in [0, 1].
        stagnation_generations_limit: Consecutive stagnant generations required to trigger cataclysm >= 1.
        cataclysmic_step_size: Elevated mutation step size during cataclysmic re-diversification > 0.
    """

    initial_step_size: float = 0.05
    min_step_size: float = 0.005
    max_step_size: float = 0.25
    mutation_probability: float = 0.20
    cauchy_clipping_bound: float = 2.0
    expansion_factor: float = 1.10
    contraction_factor: float = 0.90
    smoothing_factor: float = 0.20
    risk_gene_scale: float = 0.50
    game_gene_scale: float = 0.75
    search_gene_scale: float = 1.25
    stagnation_diversity_threshold: float = 0.05
    stagnation_correlation_threshold: float = 0.80
    stagnation_generations_limit: int = 3
    cataclysmic_step_size: float = 0.35

    def __post_init__(self) -> None:
        """Validate configuration domain bounds and defensive invariants."""
        if self.min_step_size <= 0.0 or not math.isfinite(self.min_step_size):
            raise LifecycleError(
                f"min_step_size must be positive and finite, got {self.min_step_size}"
            )
        if self.max_step_size <= self.min_step_size or not math.isfinite(self.max_step_size):
            raise LifecycleError(
                f"max_step_size must be strictly greater than min_step_size, got min={self.min_step_size}, max={self.max_step_size}"
            )
        if not (self.min_step_size <= self.initial_step_size <= self.max_step_size):
            raise LifecycleError(
                f"initial_step_size must be within [{self.min_step_size}, {self.max_step_size}], got {self.initial_step_size}"
            )
        if not (0.0 < self.mutation_probability <= 1.0):
            raise LifecycleError(
                f"mutation_probability must be in (0.0, 1.0], got {self.mutation_probability}"
            )
        if self.cauchy_clipping_bound <= 0.0 or not math.isfinite(self.cauchy_clipping_bound):
            raise LifecycleError(
                f"cauchy_clipping_bound must be positive and finite, got {self.cauchy_clipping_bound}"
            )
        if self.expansion_factor <= 1.0 or not math.isfinite(self.expansion_factor):
            raise LifecycleError(f"expansion_factor must be > 1.0, got {self.expansion_factor}")
        if not (0.0 < self.contraction_factor < 1.0):
            raise LifecycleError(
                f"contraction_factor must be in (0.0, 1.0), got {self.contraction_factor}"
            )
        if not (0.0 < self.smoothing_factor <= 1.0):
            raise LifecycleError(
                f"smoothing_factor must be in (0.0, 1.0], got {self.smoothing_factor}"
            )
        if self.risk_gene_scale <= 0.0 or not math.isfinite(self.risk_gene_scale):
            raise LifecycleError(
                f"risk_gene_scale must be positive and finite, got {self.risk_gene_scale}"
            )
        if self.game_gene_scale <= 0.0 or not math.isfinite(self.game_gene_scale):
            raise LifecycleError(
                f"game_gene_scale must be positive and finite, got {self.game_gene_scale}"
            )
        if self.search_gene_scale <= 0.0 or not math.isfinite(self.search_gene_scale):
            raise LifecycleError(
                f"search_gene_scale must be positive and finite, got {self.search_gene_scale}"
            )
        if self.stagnation_diversity_threshold <= 0.0 or not math.isfinite(
            self.stagnation_diversity_threshold
        ):
            raise LifecycleError(
                f"stagnation_diversity_threshold must be positive, got {self.stagnation_diversity_threshold}"
            )
        if not (0.0 <= self.stagnation_correlation_threshold <= 1.0):
            raise LifecycleError(
                f"stagnation_correlation_threshold must be in [0.0, 1.0], got {self.stagnation_correlation_threshold}"
            )
        if self.stagnation_generations_limit < 1:
            raise LifecycleError(
                f"stagnation_generations_limit must be >= 1, got {self.stagnation_generations_limit}"
            )
        if self.cataclysmic_step_size <= 0.0 or not math.isfinite(self.cataclysmic_step_size):
            raise LifecycleError(
                f"cataclysmic_step_size must be positive, got {self.cataclysmic_step_size}"
            )


@dataclass(frozen=True)
class GenerationalState:
    """Immutable audit record and control state of the evolutionary generational loop.

    Attributes:
        generation_index: Zero-indexed current generation count (t >= 0).
        population_size: Total number of individuals in the active population (N > 0).
        active_step_size: Current global mutation standard deviation sigma_mut.
        smoothed_success_ratio: Exponential moving average of mutation success ratio in [0, 1].
        phenotypic_diversity: Current population parameter dispersion bar_D_param >= 0.
        mean_residual_correlation: Current population mean residual correlation bar_rho_pop in [0, 1].
        stagnation_count: Consecutive generations satisfying stagnation criteria >= 0.
        is_cataclysm_triggered: Boolean flag indicating if cataclysmic hyper-mutation was fired.
        surviving_candidate_ids: Ordered tuple of candidate IDs surviving into this generation.
        front_1_count: Number of non-dominated Pareto Front-1 champions in this generation.
    """

    generation_index: int
    population_size: int
    active_step_size: float
    smoothed_success_ratio: float
    phenotypic_diversity: float
    mean_residual_correlation: float
    stagnation_count: int
    is_cataclysm_triggered: bool
    surviving_candidate_ids: tuple[str, ...]
    front_1_count: int

    def __post_init__(self) -> None:
        """Validate defensive state invariants."""
        if self.generation_index < 0:
            raise InvalidGenerationalStateException(
                f"generation_index must be >= 0, got {self.generation_index}"
            )
        if self.population_size <= 0:
            raise InvalidGenerationalStateException(
                f"population_size must be > 0, got {self.population_size}"
            )
        if self.active_step_size <= 0.0 or not math.isfinite(self.active_step_size):
            raise InvalidGenerationalStateException(
                f"active_step_size must be positive, got {self.active_step_size}"
            )
        if not (0.0 <= self.smoothed_success_ratio <= 1.0):
            raise InvalidGenerationalStateException(
                f"smoothed_success_ratio must be in [0.0, 1.0], got {self.smoothed_success_ratio}"
            )
        if self.phenotypic_diversity < 0.0 or not math.isfinite(self.phenotypic_diversity):
            raise InvalidGenerationalStateException(
                f"phenotypic_diversity must be non-negative, got {self.phenotypic_diversity}"
            )
        if not (0.0 <= self.mean_residual_correlation <= 1.0):
            raise InvalidGenerationalStateException(
                f"mean_residual_correlation must be in [0.0, 1.0], got {self.mean_residual_correlation}"
            )
        if self.stagnation_count < 0:
            raise InvalidGenerationalStateException(
                f"stagnation_count must be >= 0, got {self.stagnation_count}"
            )
        if self.front_1_count < 0:
            raise InvalidGenerationalStateException(
                f"front_1_count must be >= 0, got {self.front_1_count}"
            )
        if (
            self.surviving_candidate_ids
            and len(self.surviving_candidate_ids) != self.population_size
        ):
            raise InvalidGenerationalStateException(
                f"surviving_candidate_ids count ({len(self.surviving_candidate_ids)}) does not match population_size ({self.population_size})"
            )


@dataclass(frozen=True)
class LifecycleStepResult:
    """Complete bundle returned after executing one generational transition step.

    Attributes:
        next_chromosomes: Map of candidate_id -> StrategyChromosome for the surviving population.
        surviving_fitness: Tuple of CandidateFitness records for the surviving population.
        ranking: RankingResult bundle from Pareto ranking.
        state: Updated GenerationalState recording step metrics.
    """

    next_chromosomes: dict[str, StrategyChromosome]
    surviving_fitness: tuple[CandidateFitness, ...]
    ranking: RankingResult
    state: GenerationalState


# =====================================================================
# Adaptive Volatility Mutator
# =====================================================================


class AdaptiveVolatilityMutator:
    """Self-adaptive Truncated Cauchy Mutator with Gene-Family Differential Scaling and Mirror Reflection.

    Operates in the scale-free unit hypercube space u in [0, 1]^20. Applies truncated Cauchy
    perturbations scaled differentially across gene families (conservative risk, intermediate game
    theory, exploratory search), reflects crossing trajectories back into [0, 1] to prevent edge-stickiness,
    and decodes back to strictly invariant-satisfying StrategyChromosome domain models.
    """

    def __init__(
        self,
        config: MutationConfig | None = None,
        codec: ChromosomeVectorCodec | None = None,
    ) -> None:
        self._config = config or MutationConfig()
        self._codec = codec or ChromosomeVectorCodec()

        # Construct scale vector for the 20 genes
        scales: list[float] = []
        for spec in GENE_REGISTRY:
            if spec.block == "risk":
                scales.append(self._config.risk_gene_scale)
            elif spec.block == "game":
                scales.append(self._config.game_gene_scale)
            else:  # "repr", "infer"
                scales.append(self._config.search_gene_scale)
        self._scale_vector: np.ndarray = np.array(scales, dtype=np.float64)

    @property
    def config(self) -> MutationConfig:
        """Hyperparameter configuration."""
        return self._config

    @property
    def codec(self) -> ChromosomeVectorCodec:
        """Chromosome vector codec."""
        return self._codec

    def mutate(
        self,
        chromosome: StrategyChromosome,
        step_size: float | None = None,
        is_cataclysmic: bool = False,
        rng: np.random.Generator | None = None,
    ) -> StrategyChromosome:
        """Mutate a StrategyChromosome in unit hypercube space with Cauchy fat tails and mirror reflection.

        Args:
            chromosome: Parent StrategyChromosome to mutate.
            step_size: Optional step size override sigma_mut. If None, defaults to config.initial_step_size.
            is_cataclysmic: If True, uses elevated cataclysmic_step_size for high-dispersion re-diversification.
            rng: Optional numpy Generator for deterministic reproduction.

        Returns:
            Mutated StrategyChromosome guaranteed to satisfy all domain invariants.
        """
        if rng is None:
            rng = np.random.default_rng()

        if is_cataclysmic:
            effective_sigma = self._config.cataclysmic_step_size
        elif step_size is not None:
            if step_size <= 0.0 or not math.isfinite(step_size):
                raise LifecycleError(f"step_size must be positive and finite, got {step_size}")
            effective_sigma = step_size
        else:
            effective_sigma = self._config.initial_step_size

        u = self._codec.encode(chromosome)

        # Draw gene mutation mask
        mask = rng.random(self._codec.dimension) < self._config.mutation_probability

        # Standard Cauchy random draws: xi ~ Cauchy(0, 1)
        xi = rng.standard_cauchy(self._codec.dimension)
        xi_clipped = np.clip(
            xi, -self._config.cauchy_clipping_bound, self._config.cauchy_clipping_bound
        )

        # Differential perturbation: Delta u = sigma * kappa * xi
        delta_u = np.where(mask, effective_sigma * self._scale_vector * xi_clipped, 0.0)

        # Mirror boundary reflection
        u_raw = u + delta_u
        u_refl = np.where(
            u_raw < 0.0,
            -u_raw,
            np.where(u_raw > 1.0, 2.0 - u_raw, u_raw),
        )
        u_clamped = np.clip(u_refl, 0.0, 1.0)

        return self._codec.decode(u_clamped)

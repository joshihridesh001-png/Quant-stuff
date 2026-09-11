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
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist

from quant.analytics.chromosomes import (
    GENE_REGISTRY,
    ChromosomeVectorCodec,
    StrategyChromosome,
)
from quant.analytics.hypergamic_selection import (
    HypergamicConfig,
    HypergamicSelectionEngine,
)
from quant.analytics.pareto_sorting import (
    BoundaryAnchoredRVEARanker,
    CandidateFitness,
    ParetoFront,
    RankingResult,
)

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

    def mutate_vector(
        self,
        u: np.ndarray,
        step_size: float | None = None,
        is_cataclysmic: bool = False,
        rng: np.random.Generator | None = None,
    ) -> StrategyChromosome:
        """Mutate encoded unit vector u directly with Cauchy fat tails and mirror reflection."""
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

        u_mut = np.clip(u_refl, 0.0, 1.0)
        return self._codec.decode(u_mut)

    def mutate(
        self,
        chromosome: StrategyChromosome,
        step_size: float | None = None,
        is_cataclysmic: bool = False,
        rng: np.random.Generator | None = None,
    ) -> StrategyChromosome:
        """Mutate a StrategyChromosome in unit hypercube space with Cauchy fat tails and mirror reflection."""
        return self.mutate_vector(
            self._codec.encode(chromosome),
            step_size=step_size,
            is_cataclysmic=is_cataclysmic,
            rng=rng,
        )

    def adapt_step_size(
        self,
        current_step_size: float,
        current_smoothed_ratio: float,
        instantaneous_success_ratio: float,
    ) -> tuple[float, float]:
        r"""Adapt global mutation step size via Rechenberg 1/5th rule with exponential smoothing.

        Args:
            current_step_size: Current mutation step size sigma_mut^(t).
            current_smoothed_ratio: Previous generation smoothed success ratio \bar{r}^(t-1).
            instantaneous_success_ratio: Instantaneous success ratio r^(t) in [0, 1].

        Returns:
            Tuple of (new_step_size, new_smoothed_ratio) strictly bounded in [min, max] and [0, 1].
        """
        if current_step_size <= 0.0 or not math.isfinite(current_step_size):
            raise LifecycleError(
                f"current_step_size must be positive and finite, got {current_step_size}"
            )
        if not (0.0 <= current_smoothed_ratio <= 1.0):
            raise LifecycleError(
                f"current_smoothed_ratio must be in [0.0, 1.0], got {current_smoothed_ratio}"
            )
        if not (0.0 <= instantaneous_success_ratio <= 1.0):
            raise LifecycleError(
                f"instantaneous_success_ratio must be in [0.0, 1.0], got {instantaneous_success_ratio}"
            )

        # Exponential smoothing: \bar{r}^(t) = \alpha * r^(t) + (1 - \alpha) * \bar{r}^(t-1)
        alpha = self._config.smoothing_factor
        new_smoothed_ratio = float(
            np.clip(
                alpha * instantaneous_success_ratio + (1.0 - alpha) * current_smoothed_ratio,
                0.0,
                1.0,
            )
        )

        # 1/5th rule: expand if > 0.20, contract if < 0.20
        if new_smoothed_ratio > 0.20:
            new_step_size = current_step_size * self._config.expansion_factor
        elif new_smoothed_ratio < 0.20:
            new_step_size = current_step_size * self._config.contraction_factor
        else:
            new_step_size = current_step_size

        # Clamp strictly to [min_step_size, max_step_size]
        new_step_size = float(
            np.clip(new_step_size, self._config.min_step_size, self._config.max_step_size)
        )

        return new_step_size, new_smoothed_ratio

    def compute_apd_success_ratio(
        self,
        offspring_to_parent: dict[str, str],
        ranking: RankingResult,
    ) -> float:
        """Compute instantaneous APD progress success ratio r_succ^(t) in [0, 1].

        Offspring q_j is successful (s_j = 1) if:
          FrontRank(q_j) < FrontRank(p_j)  OR  (FrontRank(q_j) == FrontRank(p_j) AND APD(q_j) < APD(p_j))
        Infeasible offspring always receive s_j = 0.

        Args:
            offspring_to_parent: Mapping of offspring candidate_id -> parent candidate_id.
            ranking: Multi-objective RankingResult of joint (parents + offspring) candidate pool.

        Returns:
            Float empirical success ratio in [0.0, 1.0].
        """
        if not offspring_to_parent:
            return 0.0

        # Build candidate -> (front_rank, apd_score) map
        infeasible_set = set(ranking.infeasible_ids)
        candidate_meta: dict[str, tuple[int, float]] = {}
        for front in ranking.fronts:
            for cid, apd in zip(front.candidate_ids, front.apd_scores, strict=True):
                candidate_meta[cid] = (front.rank, apd)

        success_count = 0
        total_count = len(offspring_to_parent)

        for off_id, parent_id in offspring_to_parent.items():
            if off_id in infeasible_set:
                continue

            off_meta = candidate_meta.get(off_id)
            if off_meta is None:
                continue

            parent_meta = candidate_meta.get(parent_id)
            if parent_id in infeasible_set or parent_meta is None:
                # Parent failed or missing, but offspring is feasible and ranked
                success_count += 1
                continue

            off_rank, off_apd = off_meta
            parent_rank, parent_apd = parent_meta

            if off_rank < parent_rank or off_rank == parent_rank and off_apd < parent_apd:
                success_count += 1

        return float(success_count / total_count)


# =====================================================================
# Dual-Space Stagnation Detector
# =====================================================================


@dataclass(frozen=True)
class StagnationReport:
    r"""Audit report and diversification triggers evaluated by StagnationDetector.

    Attributes:
        phenotypic_diversity: Mean pairwise Euclidean parameter dispersion \bar{D}_param in [0, \sqrt{20}].
        mean_residual_correlation: Mean pairwise absolute residual Pearson correlation \bar{\rho}_pop in [0, 1].
        is_stagnant: True if both diversity < threshold and correlation > threshold.
        stagnation_count: Updated consecutive stagnant generations count.
        is_cataclysm_triggered: True if stagnation_count reached the limit, triggering hyper-mutation.
    """

    phenotypic_diversity: float
    mean_residual_correlation: float
    is_stagnant: bool
    stagnation_count: int
    is_cataclysm_triggered: bool


class StagnationDetector:
    r"""Dual-space population diversity monitor and cataclysmic re-diversification trigger.

    Monitors:
    1. Genotypic Hypercube Dispersion:
       \bar{D}_param = \frac{2}{N(N-1)} \sum_{i < j} ||u_i - u_j||_2
    2. Phenotypic Residual Collinearity:
       \bar{\rho}_pop = \frac{2}{N(N-1)} \sum_{i < j} |Corr(e_i, e_j)|

    Triggers cataclysmic hyper-mutation when both criteria bind simultaneously for
    `stagnation_generations_limit` consecutive generations.
    """

    def __init__(
        self,
        config: MutationConfig | None = None,
        codec: ChromosomeVectorCodec | None = None,
    ) -> None:
        self._config = config or MutationConfig()
        self._codec = codec or ChromosomeVectorCodec()

    @property
    def config(self) -> MutationConfig:
        return self._config

    @property
    def codec(self) -> ChromosomeVectorCodec:
        return self._codec

    def evaluate(
        self,
        chromosomes: Sequence[StrategyChromosome],
        residuals_or_fitnesses: Sequence[CandidateFitness] | Sequence[np.ndarray],
        current_stagnation_count: int,
    ) -> StagnationReport:
        """Evaluate dual-space population diversity metrics and update stagnation state.

        Args:
            chromosomes: Sequence of StrategyChromosome instances in the surviving population.
            residuals_or_fitnesses: Sequence of CandidateFitness objects or 1D residual arrays.
            current_stagnation_count: Number of consecutive stagnant generations prior to this step.

        Returns:
            StagnationReport detailing diversity, collinearity, and cataclysm triggers.
        """
        if current_stagnation_count < 0:
            raise StagnationException(
                f"current_stagnation_count must be >= 0, got {current_stagnation_count}"
            )

        n = len(chromosomes)
        if len(residuals_or_fitnesses) != n:
            raise StagnationException(
                f"Count mismatch: {n} chromosomes vs {len(residuals_or_fitnesses)} residual series"
            )

        if n < 2:
            return StagnationReport(
                phenotypic_diversity=0.0,
                mean_residual_correlation=0.0,
                is_stagnant=False,
                stagnation_count=0,
                is_cataclysm_triggered=False,
            )

        # 1. Genotypic hypercube dispersion \bar{D}_param
        u_matrix = np.stack([self._codec.encode(c) for c in chromosomes], axis=0)  # (N, 20)
        dists = pdist(u_matrix, metric="euclidean")
        mean_param_dispersion = float(np.mean(dists))

        # 2. Phenotypic residual collinearity \bar{\rho}_pop
        residuals: list[np.ndarray] = []
        for item in residuals_or_fitnesses:
            if isinstance(item, CandidateFitness):
                residuals.append(item.residual_series)
            elif isinstance(item, np.ndarray):
                residuals.append(item)
            else:
                raise StagnationException(f"Unsupported residual item type: {type(item)}")

        # Validate residual lengths
        t_len = len(residuals[0])
        for idx, res in enumerate(residuals):
            if res.ndim != 1 or len(res) != t_len:
                raise StagnationException(
                    f"Residual at index {idx} has length {len(res)}, expected {t_len}"
                )

        e_matrix = np.stack(residuals, axis=0)  # (N, T)
        e_centered = e_matrix - np.mean(e_matrix, axis=1, keepdims=True)
        stds = np.std(e_matrix, axis=1)  # (N,)

        valid_std_mask = stds > 1e-12
        e_norm = np.zeros_like(e_centered)
        e_norm[valid_std_mask] = e_centered[valid_std_mask] / stds[valid_std_mask, np.newaxis]

        corr_matrix = (e_norm @ e_norm.T) / float(t_len)
        triu_indices = np.triu_indices(n, k=1)
        mean_residual_correlation = float(np.mean(np.abs(corr_matrix)[triu_indices]))
        mean_residual_correlation = float(np.clip(mean_residual_correlation, 0.0, 1.0))

        # Stagnation rule: BOTH dispersion < eps AND correlation > rho
        is_stagnant = (
            mean_param_dispersion < self._config.stagnation_diversity_threshold
            and mean_residual_correlation > self._config.stagnation_correlation_threshold
        )

        new_stagnation_count = current_stagnation_count + 1 if is_stagnant else 0

        if new_stagnation_count >= self._config.stagnation_generations_limit:
            is_cataclysm_triggered = True
            new_stagnation_count = 0
        else:
            is_cataclysm_triggered = False

        return StagnationReport(
            phenotypic_diversity=mean_param_dispersion,
            mean_residual_correlation=mean_residual_correlation,
            is_stagnant=is_stagnant,
            stagnation_count=new_stagnation_count,
            is_cataclysm_triggered=is_cataclysm_triggered,
        )


# =====================================================================
# Master Generational Lifecycle Engine
# =====================================================================


class GenerationalLifecycleEngine:
    r"""Master (\mu + \lambda) APD-Adaptive Evolutionary Lifecycle Engine.

    Executes closed-loop generational strategy evolution:
    1. Cataclysmic Re-Diversification handling when state.is_cataclysm_triggered is True.
    2. Hypergamic Assortative Reproduction via Pareto Cohort Stratification and Orthogonality Gating.
    3. Self-Adaptive Truncated Cauchy Mutation with gene-family scaling and mirror boundary reflection.
    4. Offspring candidate evaluation via caller-provided simulation evaluator or precomputed fitness.
    5. Joint (\mu + \lambda) candidate pool formation (|U_t| = 2N).
    6. Multi-objective Pareto ranking via BoundaryAnchoredRVEARanker (ENS-SS, RVEA APD).
    7. Environmental Selection: selects surviving P_{t+1} (|P_{t+1}| = N) filling fronts by APD ascending.
    8. APD-Progress Rechenberg Volatility Adaptation: dynamically adjusts sigma_mut.
    9. Dual-Space Stagnation Monitoring: computes \bar{D}_param and \bar{\rho}_pop.
    10. Immutable GenerationalState progression.
    """

    def __init__(
        self,
        mutation_config: MutationConfig | None = None,
        hypergamic_config: HypergamicConfig | None = None,
        rvea_ranker: BoundaryAnchoredRVEARanker | None = None,
        mutator: AdaptiveVolatilityMutator | None = None,
        stagnation_detector: StagnationDetector | None = None,
        selection_engine: HypergamicSelectionEngine | None = None,
        codec: ChromosomeVectorCodec | None = None,
    ) -> None:
        self._mutation_config = mutation_config or MutationConfig()
        self._hypergamic_config = hypergamic_config or HypergamicConfig(elitism_count=0)
        self._codec = codec or ChromosomeVectorCodec()
        self._rvea_ranker = rvea_ranker or BoundaryAnchoredRVEARanker()
        self._mutator = mutator or AdaptiveVolatilityMutator(self._mutation_config, self._codec)
        self._stagnation_detector = stagnation_detector or StagnationDetector(
            self._mutation_config, self._codec
        )
        self._selection_engine = selection_engine or HypergamicSelectionEngine(
            self._hypergamic_config
        )
        self._cached_ranking: tuple[tuple[str, ...], RankingResult] | None = None

    @property
    def mutation_config(self) -> MutationConfig:
        """Hyperparameter configuration for mutation and lifecycle."""
        return self._mutation_config

    @property
    def hypergamic_config(self) -> HypergamicConfig:
        """Hyperparameter configuration for hypergamic reproduction."""
        return self._hypergamic_config

    def initialize_state(
        self,
        chromosomes: dict[str, StrategyChromosome],
        fitness: Sequence[CandidateFitness],
        initial_step_size: float | None = None,
        max_generations: int = 100,
    ) -> GenerationalState:
        """Initialize GenerationalState at generation 0 with baseline ranking and diversity metrics.

        Args:
            chromosomes: Map of candidate_id -> StrategyChromosome for initial population.
            fitness: Sequence of CandidateFitness records for initial population.
            initial_step_size: Optional step size override. Defaults to config.initial_step_size.
            max_generations: Total evolutionary horizon.

        Returns:
            GenerationalState for generation 0.
        """
        n = len(chromosomes)
        if len(fitness) != n:
            raise LifecycleError(
                f"Mismatch between chromosomes count ({n}) and fitness count ({len(fitness)})"
            )
        for f in fitness:
            if f.candidate_id not in chromosomes:
                raise LifecycleError(
                    f"Candidate '{f.candidate_id}' in fitness not found in chromosomes."
                )

        ranking = self._rvea_ranker.rank_population(
            list(fitness), generation=0, max_generations=max_generations, auto_admit=False
        )
        ordered_ids: list[str] = []
        for front in ranking.fronts:
            ordered_ids.extend(front.candidate_ids)
        for inf_id in ranking.infeasible_ids:
            ordered_ids.append(inf_id)

        front_1_count = len(ranking.fronts[0].candidate_ids) if ranking.fronts else 0

        chrom_list = [chromosomes[cid] for cid in ordered_ids]
        fitness_list = [next(f for f in fitness if f.candidate_id == cid) for cid in ordered_ids]

        stagnation_report = self._stagnation_detector.evaluate(
            chromosomes=chrom_list,
            residuals_or_fitnesses=fitness_list,
            current_stagnation_count=0,
        )

        step_size = (
            initial_step_size
            if initial_step_size is not None
            else self._mutation_config.initial_step_size
        )

        self._cached_ranking = (tuple(f.candidate_id for f in fitness), ranking)

        return GenerationalState(
            generation_index=0,
            population_size=n,
            active_step_size=step_size,
            smoothed_success_ratio=0.20,
            phenotypic_diversity=stagnation_report.phenotypic_diversity,
            mean_residual_correlation=stagnation_report.mean_residual_correlation,
            stagnation_count=stagnation_report.stagnation_count,
            is_cataclysm_triggered=stagnation_report.is_cataclysm_triggered,
            surviving_candidate_ids=tuple(ordered_ids),
            front_1_count=front_1_count,
        )

    def step_generation(
        self,
        current_chromosomes: dict[str, StrategyChromosome],
        current_fitness: Sequence[CandidateFitness],
        current_state: GenerationalState,
        evaluator: (
            Callable[[dict[str, StrategyChromosome]], Sequence[CandidateFitness]] | None
        ) = None,
        offspring_fitness: Sequence[CandidateFitness] | None = None,
        seed: int | None = None,
        max_generations: int = 100,
    ) -> LifecycleStepResult:
        """Execute one complete evolutionary generation step under (mu + lambda) selection.

        Args:
            current_chromosomes: Dict mapping candidate_id -> StrategyChromosome for parent population P_t.
            current_fitness: Sequence of CandidateFitness records for parent population P_t.
            current_state: Current GenerationalState audit record.
            evaluator: Optional callable executing simulation/backtest to evaluate offspring.
            offspring_fitness: Optional precomputed CandidateFitness sequence for offspring.
            seed: Optional random seed for reproducible mating and mutation draws.
            max_generations: Planned total generation horizon for RVEA ray penalty escalation.

        Returns:
            LifecycleStepResult containing next_chromosomes, surviving_fitness, ranking, and updated state.
        """
        n = len(current_chromosomes)
        if len(current_fitness) != n:
            raise LifecycleError(
                f"Population size mismatch: {n} chromosomes vs {len(current_fitness)} fitness records."
            )
        if current_state.population_size != n:
            raise InvalidGenerationalStateException(
                f"State population size ({current_state.population_size}) does not match chromosomes ({n})."
            )
        if evaluator is None and offspring_fitness is None:
            raise LifecycleError("Either evaluator or offspring_fitness must be provided.")

        rng = np.random.default_rng(seed)

        # 1. Obtain parent ranking (fast-path: check if ranking already cached for this population)
        parent_candidate_ids = tuple(f.candidate_id for f in current_fitness)
        if self._cached_ranking is not None and self._cached_ranking[0] == parent_candidate_ids:
            parent_ranking = self._cached_ranking[1]
        else:
            parent_ranking = self._rvea_ranker.rank_population(
                list(current_fitness),
                generation=current_state.generation_index,
                max_generations=max_generations,
                auto_admit=False,
            )

        # 2. Hypergamic Reproduction: produce N raw offspring
        repro_result = self._selection_engine.reproduce(
            candidates=list(current_fitness),
            ranking=parent_ranking,
            chromosome_map=dict(current_chromosomes),
            target_population_size=n,
            seed=seed,
        )

        # 3. Adaptive Cauchy Mutation
        is_cataclysmic = current_state.is_cataclysm_triggered
        mutated_offspring_map: dict[str, StrategyChromosome] = {}
        offspring_to_parent: dict[str, str] = {}

        next_gen_idx = current_state.generation_index + 1

        for i, raw_child in enumerate(repro_result.offspring_chromosomes):
            child_id = f"gen_{next_gen_idx}_ind_{i}"
            mutated_child = self._mutator.mutate(
                chromosome=raw_child,
                step_size=current_state.active_step_size,
                is_cataclysmic=is_cataclysmic,
                rng=rng,
            )
            mutated_offspring_map[child_id] = mutated_child

            if i < len(repro_result.mating_pairs):
                offspring_to_parent[child_id] = repro_result.mating_pairs[i].alpha_id
            elif i - len(repro_result.mating_pairs) < len(repro_result.elite_ids):
                offspring_to_parent[child_id] = repro_result.elite_ids[
                    i - len(repro_result.mating_pairs)
                ]
            elif repro_result.alpha_ids:
                offspring_to_parent[child_id] = repro_result.alpha_ids[0]
            else:
                offspring_to_parent[child_id] = next(iter(current_chromosomes))

        # 4. Offspring Fitness Evaluation
        if offspring_fitness is not None:
            if len(offspring_fitness) != n:
                raise LifecycleError(
                    f"offspring_fitness length ({len(offspring_fitness)}) must match population size ({n})"
                )
            evaluated_offspring = tuple(offspring_fitness)
        else:
            assert evaluator is not None
            eval_res = evaluator(mutated_offspring_map)
            if len(eval_res) != n:
                raise LifecycleError(
                    f"evaluator returned {len(eval_res)} fitness records, expected {n}"
                )
            evaluated_offspring = tuple(eval_res)

        # 5. Form Joint Pool U_t = P_t \cup Q_t (|U_t| = 2N)
        joint_fitness = tuple(current_fitness) + evaluated_offspring
        joint_chromosomes: dict[str, StrategyChromosome] = dict(current_chromosomes)
        joint_chromosomes.update(mutated_offspring_map)

        offspring_ids_in_map = list(mutated_offspring_map.keys())
        for idx, off_fit in enumerate(evaluated_offspring):
            if off_fit.candidate_id not in joint_chromosomes:
                joint_chromosomes[off_fit.candidate_id] = mutated_offspring_map[
                    offspring_ids_in_map[idx]
                ]
                old_id = offspring_ids_in_map[idx]
                if old_id in offspring_to_parent:
                    offspring_to_parent[off_fit.candidate_id] = offspring_to_parent[old_id]

        # 6. Multi-Objective Pareto Ranking on U_t
        joint_ranking = self._rvea_ranker.rank_population(
            list(joint_fitness),
            generation=next_gen_idx,
            max_generations=max_generations,
            auto_admit=True,
        )

        # 7. (mu + lambda) Environmental Selection: select top N survivors by APD
        surviving_ids: list[str] = []
        for front in joint_ranking.fronts:
            for cid in front.candidate_ids:
                surviving_ids.append(cid)
                if len(surviving_ids) == n:
                    break
            if len(surviving_ids) == n:
                break

        if len(surviving_ids) < n:
            for inf_id in joint_ranking.infeasible_ids:
                surviving_ids.append(inf_id)
                if len(surviving_ids) == n:
                    break

        fitness_lookup = {f.candidate_id: f for f in joint_fitness}
        surviving_fitness_tuple = tuple(fitness_lookup[cid] for cid in surviving_ids)
        next_chromosomes = {cid: joint_chromosomes[cid] for cid in surviving_ids}

        surviving_front_1 = (
            set(joint_ranking.fronts[0].candidate_ids) if joint_ranking.fronts else set()
        )
        front_1_count = sum(1 for cid in surviving_ids if cid in surviving_front_1)

        # 8. Rechenberg Volatility Adaptation
        instantaneous_success_ratio = self._mutator.compute_apd_success_ratio(
            offspring_to_parent, joint_ranking
        )
        new_step_size, new_smoothed_ratio = self._mutator.adapt_step_size(
            current_step_size=current_state.active_step_size,
            current_smoothed_ratio=current_state.smoothed_success_ratio,
            instantaneous_success_ratio=instantaneous_success_ratio,
        )

        # 9. Dual-Space Stagnation Monitoring
        surviving_chrom_list = [next_chromosomes[cid] for cid in surviving_ids]
        stagnation_report = self._stagnation_detector.evaluate(
            chromosomes=surviving_chrom_list,
            residuals_or_fitnesses=surviving_fitness_tuple,
            current_stagnation_count=current_state.stagnation_count,
        )

        # 10. Construct next GenerationalState
        next_state = GenerationalState(
            generation_index=next_gen_idx,
            population_size=n,
            active_step_size=new_step_size,
            smoothed_success_ratio=new_smoothed_ratio,
            phenotypic_diversity=stagnation_report.phenotypic_diversity,
            mean_residual_correlation=stagnation_report.mean_residual_correlation,
            stagnation_count=stagnation_report.stagnation_count,
            is_cataclysm_triggered=stagnation_report.is_cataclysm_triggered,
            surviving_candidate_ids=tuple(surviving_ids),
            front_1_count=front_1_count,
        )

        # Cache survivor ranking for next generation to eliminate redundant parent ranking
        surviving_id_set = set(surviving_ids)
        surv_fronts: list[ParetoFront] = []
        for front in joint_ranking.fronts:
            surv_pairs = [
                (cid, score)
                for cid, score in zip(front.candidate_ids, front.apd_scores, strict=True)
                if cid in surviving_id_set
            ]
            if surv_pairs:
                surv_cands = tuple(p[0] for p in surv_pairs)
                surv_scores = tuple(p[1] for p in surv_pairs)
                surv_fronts.append(
                    ParetoFront(
                        rank=front.rank,
                        candidate_ids=surv_cands,
                        apd_scores=surv_scores,
                    )
                )
        surv_infeasible = tuple(
            cid for cid in joint_ranking.infeasible_ids if cid in surviving_id_set
        )
        survivor_ranking = RankingResult(
            fronts=tuple(surv_fronts),
            infeasible_ids=surv_infeasible,
            active_reference_rays=joint_ranking.active_reference_rays,
            archive_size=joint_ranking.archive_size,
            subspace_rank=joint_ranking.subspace_rank,
        )
        self._cached_ranking = (tuple(surviving_ids), survivor_ranking)

        return LifecycleStepResult(
            next_chromosomes=next_chromosomes,
            surviving_fitness=surviving_fitness_tuple,
            ranking=joint_ranking,
            state=next_state,
        )

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

import numpy as np

from quant.analytics.chromosomes import ChromosomeVectorCodec, StrategyChromosome
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
        if not (
            math.isfinite(self.crossover_distribution_index)
            and self.crossover_distribution_index > 0.0
        ):
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
            raise HypergamicSelectionError(f"elitism_count must be >= 0, got {self.elitism_count}")


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

    offspring_chromosomes: tuple[StrategyChromosome, ...]
    mating_pairs: tuple[MatingPair, ...]
    alpha_ids: tuple[str, ...]
    aspirant_ids: tuple[str, ...]
    elite_ids: tuple[str, ...] = ()
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


# =====================================================================
# 2. Bidirectional Residual Orthogonality Gate
# =====================================================================


class ResidualOrthogonalityGate:
    """Evaluates whether prospective mating partners exhibit orthogonal residuals.

    Enforces bidirectional absolute correlation rejection:
    - Rejects positive collinear clones (rho -> +1.0).
    - Rejects negative inverse clones (rho -> -1.0).
    - Accepts pairs where unexplained variance 1.0 - |rho| >= delta_eff.
    - Supports adaptive relaxation levels k >= 0: delta_eff = delta * (relaxation_factor)^k.
    """

    def __init__(self, config: HypergamicConfig | None = None) -> None:
        """Initialize gate with configuration thresholds."""
        self._config = config or HypergamicConfig()

    def evaluate(
        self,
        alpha: CandidateFitness,
        aspirant: CandidateFitness,
        relaxation_level: int = 0,
    ) -> tuple[bool, float]:
        """Evaluate orthogonality between Alpha and Aspirant residual vectors.

        Args:
            alpha: Alpha candidate fitness record.
            aspirant: Aspirant candidate fitness record.
            relaxation_level: Number of adaptive relaxation steps applied (default 0).

        Returns:
            Tuple of (is_accepted, sample_pearson_correlation).
        """
        res_a = alpha.residual_series
        res_b = aspirant.residual_series

        t_bars = min(len(res_a), len(res_b))
        if t_bars < 2:
            return False, 1.0

        dev_a = res_a[:t_bars] - np.mean(res_a[:t_bars])
        dev_b = res_b[:t_bars] - np.mean(res_b[:t_bars])

        norm_a = float(np.linalg.norm(dev_a))
        norm_b = float(np.linalg.norm(dev_b))

        # ERR-EVO-HYP-002: Guard against zero variance residuals
        if norm_a < 1e-12 or norm_b < 1e-12:
            return False, 1.0

        rho = float(np.dot(dev_a, dev_b) / (norm_a * norm_b))
        rho = max(-1.0, min(1.0, rho))

        # Effective threshold under adaptive relaxation
        factor = self._config.relaxation_factor ** max(0, relaxation_level)
        delta_eff = self._config.orthogonality_threshold * factor

        # Bidirectional unexplained variance condition: 1.0 - |rho| >= delta_eff
        unexplained_variance = 1.0 - abs(rho)
        is_accepted = bool(unexplained_variance >= delta_eff)

        return is_accepted, rho


# =====================================================================
# 3. Assortative Hypergamic Partner Matcher
# =====================================================================


class HypergamicPartnerMatcher:
    """Matches Alpha and Aspirant candidates into complementary mating pairs.

    Features:
    - Bounded tournament selection for Aspirant petitions.
    - Evaluation through Bidirectional Residual Orthogonality Gate.
    - Adaptive threshold relaxation upon consecutive rejections.
    - Phenotypic distance fallback in unit-hypercube space upon retry exhaustion,
      guaranteeing zero deadlocks and strictly bounded execution time.
    """

    def __init__(
        self,
        gate: ResidualOrthogonalityGate | None = None,
        config: HypergamicConfig | None = None,
    ) -> None:
        """Initialize matcher with orthogonality gate and configuration."""
        self._config = config or HypergamicConfig()
        self._gate = gate or ResidualOrthogonalityGate(self._config)
        self._codec = ChromosomeVectorCodec()
        self._last_rejection_count: int = 0
        self._last_relaxation_count: int = 0

    @property
    def last_rejection_count(self) -> int:
        """Total number of candidate rejections during the last match_pairs run."""
        return self._last_rejection_count

    @property
    def last_relaxation_count(self) -> int:
        """Total number of relaxed pairs during the last match_pairs run."""
        return self._last_relaxation_count

    def match_pairs(
        self,
        alphas: list[CandidateFitness],
        aspirants: list[CandidateFitness],
        target_pair_count: int,
        chromosome_map: dict[str, StrategyChromosome] | None = None,
        seed: int | None = None,
    ) -> list[MatingPair]:
        """Match target_pair_count mating pairs between Alphas and Aspirants.

        Args:
            alphas: Viable Alpha cohort candidates.
            aspirants: Viable Aspirant cohort candidates.
            target_pair_count: Number of parent pairs to generate.
            chromosome_map: Optional map of candidate_id -> StrategyChromosome for phenotypic distance.
            seed: Optional PRNG seed for deterministic testing.

        Returns:
            List of accepted MatingPair instances of length target_pair_count.

        Raises:
            InvalidCohortException: If alphas or aspirants list is empty.
        """
        if not alphas:
            raise InvalidCohortException("Alpha cohort cannot be empty for partner matching.")
        if not aspirants:
            raise InvalidCohortException("Aspirant cohort cannot be empty for partner matching.")
        if target_pair_count <= 0:
            self._last_rejection_count = 0
            self._last_relaxation_count = 0
            return []

        rng = np.random.default_rng(seed)
        n_alphas = len(alphas)
        n_aspirants = len(aspirants)
        k_tourn = min(self._config.tournament_size, n_aspirants)
        max_attempts = self._config.max_mating_attempts
        exhaustion_limit = 2 * max_attempts

        pairs: list[MatingPair] = []
        total_rejections = 0
        total_relaxations = 0

        for pair_idx in range(target_pair_count):
            # Select Alpha parent via round-robin
            alpha = alphas[pair_idx % n_alphas]

            accepted = False
            attempts = 0
            best_pair: MatingPair | None = None

            while not accepted and attempts < exhaustion_limit:
                # Determine adaptive relaxation level
                level = 0
                if attempts >= max_attempts:
                    level = attempts - max_attempts + 1

                # Tournament selection among aspirants
                tourn_indices = rng.choice(
                    n_aspirants, size=k_tourn, replace=(k_tourn > n_aspirants)
                )
                best_idx = int(max(tourn_indices, key=lambda idx: aspirants[idx].dsr))
                aspirant = aspirants[best_idx]

                # Evaluate through gate
                is_accepted, rho = self._gate.evaluate(alpha, aspirant, relaxation_level=level)

                if is_accepted:
                    accepted = True
                    best_pair = MatingPair(
                        alpha_id=alpha.candidate_id,
                        aspirant_id=aspirant.candidate_id,
                        residual_correlation=rho,
                        is_relaxed=(level > 0),
                        relaxation_level=level,
                    )
                else:
                    attempts += 1
                    total_rejections += 1

            # Fallback upon retry exhaustion
            if not accepted or best_pair is None:
                # Phenotypic distance fallback in unit hypercube space
                if chromosome_map and alpha.candidate_id in chromosome_map:
                    u_alpha = self._codec.encode(chromosome_map[alpha.candidate_id])
                    max_dist = -1.0
                    chosen_asp = aspirants[0]
                    chosen_rho = 1.0

                    for asp in aspirants:
                        if asp.candidate_id in chromosome_map:
                            u_asp = self._codec.encode(chromosome_map[asp.candidate_id])
                            dist = float(np.linalg.norm(u_alpha - u_asp))
                            if dist > max_dist:
                                max_dist = dist
                                chosen_asp = asp
                                _, chosen_rho = self._gate.evaluate(alpha, asp, relaxation_level=0)
                else:
                    # Fallback to aspirant with minimum absolute residual correlation
                    min_abs_corr = 2.0
                    chosen_asp = aspirants[0]
                    chosen_rho = 1.0

                    for asp in aspirants:
                        _, rho_val = self._gate.evaluate(alpha, asp, relaxation_level=0)
                        if abs(rho_val) < min_abs_corr:
                            min_abs_corr = abs(rho_val)
                            chosen_asp = asp
                            chosen_rho = rho_val

                best_pair = MatingPair(
                    alpha_id=alpha.candidate_id,
                    aspirant_id=chosen_asp.candidate_id,
                    residual_correlation=chosen_rho,
                    is_relaxed=True,
                    relaxation_level=max(1, attempts),
                )

            if best_pair.is_relaxed:
                total_relaxations += 1

            pairs.append(best_pair)

        self._last_rejection_count = total_rejections
        self._last_relaxation_count = total_relaxations
        return pairs


# =====================================================================
# Asymmetric Latent Unit-Hypercube Crossover
# =====================================================================


class AsymmetricLatentCrossover:
    """Asymmetric Latent Unit-Hypercube Simulated Binary Crossover (SBX).

    Performs continuous genetic recombination strictly in the scale-free unit hypercube
    u in [0, 1]^20. Applies asymmetric inheritance bias:
    - Risk & Game Theory genes are biased toward Alpha parent (P_alpha >= 0.75).
    - Representation & Inference genes are biased toward Aspirant parent (P_aspirant >= 0.65).
    Decodes via ChromosomeVectorCodec, ensuring domain invariants (tau_fast < tau_slow,
    regime simplex normalization) are strictly satisfied by construction.
    """

    def __init__(
        self,
        config: HypergamicConfig | None = None,
        codec: ChromosomeVectorCodec | None = None,
    ) -> None:
        """Initialize crossover operator with configuration and vector codec."""
        self._config = config or HypergamicConfig()
        self._codec = codec or ChromosomeVectorCodec()
        self._registry = self._codec.registry

    def cross(
        self,
        alpha: StrategyChromosome,
        aspirant: StrategyChromosome,
        rng: np.random.Generator | None = None,
    ) -> StrategyChromosome:
        """Cross an Alpha and Aspirant chromosome to produce a valid offspring.

        Args:
            alpha: Alpha parent chromosome.
            aspirant: Aspirant parent chromosome.
            rng: Optional NumPy random generator for deterministic sampling.

        Returns:
            New decoded StrategyChromosome offspring satisfying domain invariants.
        """
        if rng is None:
            rng = np.random.default_rng()

        u_a = self._codec.encode(alpha)
        u_b = self._codec.encode(aspirant)

        eta_c = self._config.crossover_distribution_index
        p_alpha = self._config.alpha_risk_inheritance_prob
        p_aspirant = self._config.aspirant_repr_inheritance_prob

        n_dim = len(self._registry)
        u_child = np.empty(n_dim, dtype=np.float64)

        for d in range(n_dim):
            spec = self._registry[d]
            u_ad = u_a[d]
            u_bd = u_b[d]

            is_risk_or_game = spec.block in ("risk", "game")

            # Simulated Binary Crossover (SBX) spread calculation
            # Safe uniform draw bounded away from 0.0 and 1.0 to prevent division by zero
            r = float(rng.uniform(1e-7, 1.0 - 1e-7))
            if r <= 0.5:
                beta = (2.0 * r) ** (1.0 / (eta_c + 1.0))
            else:
                beta = (1.0 / (2.0 * (1.0 - r))) ** (1.0 / (eta_c + 1.0))

            # Candidate offspring genes: u1 is centered on Alpha, u2 is centered on Aspirant
            u1 = 0.5 * ((1.0 + beta) * u_ad + (1.0 - beta) * u_bd)
            u2 = 0.5 * ((1.0 - beta) * u_ad + (1.0 + beta) * u_bd)

            # Asymmetric role-biased selection
            coin = float(rng.uniform(0.0, 1.0))
            if is_risk_or_game:
                chosen = u1 if coin < p_alpha else u2
            else:
                chosen = u2 if coin < p_aspirant else u1

            # Strictly clamp to continuous unit hypercube bounds [0.0, 1.0]
            u_child[d] = min(max(chosen, 0.0), 1.0)

        return self._codec.decode(u_child)


# =====================================================================
# Master Evolutionary Selection Engine Facade
# =====================================================================


class HypergamicSelectionEngine:
    """Master facade orchestrating hypergamic selection, gating, and reproduction.

    End-to-end evolutionary mating pipeline:
    1. Pareto Cohort Stratification (Alpha and Aspirant cohorts).
    2. Monotonic Elitism preservation of top Front-1 champions.
    3. Bidirectional Residual Orthogonality Gating with adaptive relaxation.
    4. Bounded tournament matching with phenotypic hypercube fallback.
    5. Asymmetric Latent Unit-Hypercube Crossover preserving risk controls.
    """

    def __init__(
        self,
        config: HypergamicConfig | None = None,
        stratifier: ParetoCohortStratifier | None = None,
        gate: ResidualOrthogonalityGate | None = None,
        matcher: HypergamicPartnerMatcher | None = None,
        crossover: AsymmetricLatentCrossover | None = None,
        codec: ChromosomeVectorCodec | None = None,
    ) -> None:
        """Initialize reproduction engine facade with modular components."""
        self._config = config or HypergamicConfig()
        self._codec = codec or ChromosomeVectorCodec()
        self._stratifier = stratifier or ParetoCohortStratifier(self._config)
        self._gate = gate or ResidualOrthogonalityGate(self._config)
        self._matcher = matcher or HypergamicPartnerMatcher(self._gate, self._config)
        self._crossover = crossover or AsymmetricLatentCrossover(self._config, codec=self._codec)

    def reproduce(
        self,
        candidates: list[CandidateFitness] | tuple[CandidateFitness, ...],
        ranking: RankingResult,
        chromosome_map: dict[str, StrategyChromosome] | dict[str, StrategyChromosome],
        target_population_size: int,
        seed: int | None = None,
    ) -> OffspringResult:
        """Execute one complete evolutionary reproduction cycle.

        Args:
            candidates: Sequence of evaluated CandidateFitness records.
            ranking: Multi-objective RankingResult from Pareto sorting.
            chromosome_map: Mapping of candidate_id -> StrategyChromosome.
            target_population_size: Desired total offspring population size (N_target > 0).
            seed: Optional PRNG seed for deterministic reproduction.

        Returns:
            OffspringResult containing generated offspring chromosomes, mating pairs,
            cohort allocations, elitism IDs, and rejection/relaxation counters.

        Raises:
            HypergamicSelectionError: If target_population_size <= 0 or chromosomes are missing.
            InvalidCohortException: If cohorts cannot be stratified.
        """
        if target_population_size <= 0:
            raise HypergamicSelectionError(
                f"target_population_size must be positive, got {target_population_size}"
            )

        rng = np.random.default_rng(seed)

        # 1. Stratify candidates into Alpha and Aspirant cohorts
        alphas, aspirants = self._stratifier.stratify(list(candidates), ranking)
        alpha_ids = tuple(a.candidate_id for a in alphas)
        aspirant_ids = tuple(asp.candidate_id for asp in aspirants)

        # 2. Elitism: preserve top N_elite Front-1 champions bitwise identical
        front_1 = next((f for f in ranking.fronts if f.rank == 1), None)
        front_1_cands = front_1.candidate_ids if front_1 is not None else ()
        n_elite = min(self._config.elitism_count, len(front_1_cands), target_population_size)
        elite_ids = tuple(front_1_cands[:n_elite])

        elite_chromosomes: list[StrategyChromosome] = []
        for eid in elite_ids:
            if eid not in chromosome_map:
                raise HypergamicSelectionError(
                    f"Elite candidate '{eid}' not found in chromosome_map."
                )
            elite_chromosomes.append(chromosome_map[eid])

        # 3. Match remaining required pairs via HypergamicPartnerMatcher
        needed_pairs = target_population_size - n_elite
        mating_pairs: list[MatingPair] = []
        crossover_chromosomes: list[StrategyChromosome] = []

        if needed_pairs > 0:
            mating_pairs = self._matcher.match_pairs(
                alphas=alphas,
                aspirants=aspirants,
                target_pair_count=needed_pairs,
                chromosome_map=dict(chromosome_map),
                seed=seed,
            )

            # 4. Asymmetric Crossover
            for pair in mating_pairs:
                if pair.alpha_id not in chromosome_map:
                    raise HypergamicSelectionError(
                        f"Alpha candidate '{pair.alpha_id}' not found in chromosome_map."
                    )
                if pair.aspirant_id not in chromosome_map:
                    raise HypergamicSelectionError(
                        f"Aspirant candidate '{pair.aspirant_id}' not found in chromosome_map."
                    )
                p_alpha = chromosome_map[pair.alpha_id]
                p_aspirant = chromosome_map[pair.aspirant_id]
                child = self._crossover.cross(p_alpha, p_aspirant, rng=rng)
                crossover_chromosomes.append(child)

        all_offspring = tuple(elite_chromosomes + crossover_chromosomes)

        return OffspringResult(
            offspring_chromosomes=all_offspring,
            mating_pairs=tuple(mating_pairs),
            alpha_ids=alpha_ids,
            aspirant_ids=aspirant_ids,
            elite_ids=elite_ids,
            rejection_count=self._matcher.last_rejection_count,
            relaxation_count=self._matcher.last_relaxation_count,
        )

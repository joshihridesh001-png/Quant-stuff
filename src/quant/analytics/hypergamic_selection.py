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
            return []

        rng = np.random.default_rng(seed)
        n_alphas = len(alphas)
        n_aspirants = len(aspirants)
        k_tourn = min(self._config.tournament_size, n_aspirants)
        max_attempts = self._config.max_mating_attempts
        exhaustion_limit = 2 * max_attempts

        pairs: list[MatingPair] = []

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
                tourn_indices = rng.choice(n_aspirants, size=k_tourn, replace=(k_tourn > n_aspirants))
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

            pairs.append(best_pair)

        return pairs




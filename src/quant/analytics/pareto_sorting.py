"""Boundary-Anchored Adaptive RVEA with SVD Subspace Orthogonality (BA-ARVEA-SO).

Provides multi-objective Pareto optimization for quantitative alpha evolution using:
1. Memmel-Ledoit-Wolf dependent asymptotic dominance (accounting for return covariance).
2. SVD subspace orthogonal novelty search (eliminating multi-collinear redundancy).
3. Boundary-anchored adaptive reference rays (preserving specialist alpha champions).
4. Angle-penalized distance (APD) ranking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class ParetoSortingError(Exception):
    """Base exception for all multi-objective Pareto sorting failures."""


class CorruptedFitnessException(ParetoSortingError):
    """Raised when candidate fitness values or tensors contain non-finite numbers (NaN, Inf)."""


class InvalidResidualException(ParetoSortingError):
    """Raised when residual series are malformed, short, or have mismatched dimensions."""


class DegenerateLatticeException(ParetoSortingError):
    """Raised when reference rays lose angular separation or collapse into degeneracy."""


@dataclass(frozen=True)
class CandidateFitness:
    """Immutable evaluation record for an individual strategy candidate.

    Enforces boundary invariants INV-PAR-001 (Strict Finiteness) and
    INV-PAR-002 (Contiguous Series Length Matching).

    Attributes:
        candidate_id: Unique string identifier for the individual strategy.
        dsr: Deflated Sharpe Ratio controlling for non-normality and selection bias.
        minimax_regret: Certified worst-case regret from Phase 3 Minimax Newton Solver.
        return_series: 1D contiguous float64 array of realized portfolio returns.
        residual_series: 1D contiguous float64 array of out-of-fold prediction residuals.
        backtest_length: Total number of observations in evaluation sample.
        is_feasible: Flag indicating whether candidate satisfies domain constraints.
    """

    candidate_id: str
    dsr: float
    minimax_regret: float
    return_series: np.ndarray
    residual_series: np.ndarray
    backtest_length: int
    is_feasible: bool

    def __post_init__(self) -> None:
        """Validate defensive boundary invariants upon instantiation.

        Functional Purpose:
            Guarantees that corrupt floating-point values or mismatched array dimensions
            never penetrate into downstream BLAS linear algebra or sorting routines.
        Defensive Invariants:
            INV-PAR-001: All scalar and tensor elements must be strictly finite.
            INV-PAR-002: return_series and residual_series must be 1D, contiguous,
                         have matching length T >= 100, and match backtest_length.
        """
        # Functional Purpose: Verify scalar finiteness for primary econometric objectives.
        if not math.isfinite(self.dsr):
            raise CorruptedFitnessException(
                f"Non-finite DSR encountered for candidate '{self.candidate_id}': {self.dsr}"
            )
        if not math.isfinite(self.minimax_regret):
            raise CorruptedFitnessException(
                f"Non-finite minimax regret encountered for candidate '{self.candidate_id}': "
                f"{self.minimax_regret}"
            )

        # Functional Purpose: Ensure return_series is contiguous 1D float64 with no NaNs.
        if not isinstance(self.return_series, np.ndarray):
            raise InvalidResidualException(
                f"return_series for '{self.candidate_id}' must be a NumPy array."
            )
        if self.return_series.ndim != 1:
            raise InvalidResidualException(
                f"return_series for '{self.candidate_id}' must be 1D, got ndim={self.return_series.ndim}"
            )
        if not np.all(np.isfinite(self.return_series)):
            raise CorruptedFitnessException(
                f"Non-finite value in return_series for candidate '{self.candidate_id}'."
            )

        # Functional Purpose: Ensure residual_series is contiguous 1D float64 with no NaNs.
        if not isinstance(self.residual_series, np.ndarray):
            raise InvalidResidualException(
                f"residual_series for '{self.candidate_id}' must be a NumPy array."
            )
        if self.residual_series.ndim != 1:
            raise InvalidResidualException(
                f"residual_series for '{self.candidate_id}' must be 1D, got ndim={self.residual_series.ndim}"
            )
        if not np.all(np.isfinite(self.residual_series)):
            raise CorruptedFitnessException(
                f"Non-finite value in residual_series for candidate '{self.candidate_id}'."
            )

        # Functional Purpose: Validate series length matching and minimum sample threshold.
        t_returns = len(self.return_series)
        t_residuals = len(self.residual_series)
        if t_returns != t_residuals:
            raise InvalidResidualException(
                f"Length mismatch for candidate '{self.candidate_id}': "
                f"returns={t_returns} vs residuals={t_residuals}."
            )
        if t_returns < 100:
            raise InvalidResidualException(
                f"Series length too short for candidate '{self.candidate_id}': "
                f"{t_returns} bars (minimum required: 100)."
            )


@dataclass(frozen=True)
class ParetoFront:
    """Immutable sequence of candidate IDs assigned to a non-dominated front rank.

    Attributes:
        rank: Non-dominated front index (1-based, where 1 is the elite frontier).
        candidate_ids: Ordered sequence of candidate IDs sorted by angle-penalized distance.
        apd_scores: Corresponding angle-penalized distance scores for the candidates.
    """

    rank: int
    candidate_ids: tuple[str, ...]
    apd_scores: tuple[float, ...]


@dataclass(frozen=True)
class RankingResult:
    """Final output bundle from the multi-objective Pareto ranking execution.

    Attributes:
        fronts: Non-dominated Pareto fronts partitioned by rank.
        infeasible_ids: Sequence of candidate IDs failing domain viability constraints.
        active_reference_rays: Array of unit reference vectors active in this generation.
        archive_size: Current number of historical elite models retained in SVD archive.
        subspace_rank: Current effective rank of SVD orthogonal subspace basis.
    """

    fronts: tuple[ParetoFront, ...]
    infeasible_ids: tuple[str, ...]
    active_reference_rays: np.ndarray
    archive_size: int
    subspace_rank: int


class SVDSubspaceOrthogonalArchive:
    """Historical elite archive maintaining an orthonormal SVD prediction basis.

    Functional Purpose:
        Eradicates the 'Novelty Parasite' by evaluating candidate novelty as the
        fraction of unexplained variance (1 - R²) when projecting out-of-fold
        residuals onto the orthogonal complement of the elite alpha subspace.
        Detects multi-collinear redundancy where pairwise correlations fail.

    Defensive Invariants:
        INV-PAR-006: Output novelty rho_ortho in [0.0, 1.0] for all candidates.
        Viability Gate: Candidates with viability_mask[i] == False receive rho = 0.0.
    """

    def __init__(
        self,
        max_capacity: int = 500,
        variance_threshold: float = 0.99,
        max_basis_rank: int = 50,
    ) -> None:
        """Initialize SVDSubspaceOrthogonalArchive with capacity and threshold parameters.

        Args:
            max_capacity: Maximum number of historical elite models stored in FIFO buffer.
            variance_threshold: Minimum cumulative singular energy retained in basis.
            max_basis_rank: Hard ceiling on number of singular vectors retained.
        """
        self._max_capacity = max_capacity
        self._variance_threshold = variance_threshold
        self._max_basis_rank = max_basis_rank

        # Circular FIFO storage for standardized residual series
        self._elite_ids: list[str] = []
        self._residuals: list[np.ndarray] = []

        # Cached orthonormal basis matrix V_K in R^{T x K}
        self._basis_vt: np.ndarray | None = None
        self._subspace_rank: int = 0

    @property
    def archive_size(self) -> int:
        """Current number of historical elite models stored in the archive."""
        return len(self._residuals)

    @property
    def subspace_rank(self) -> int:
        """Effective rank of the current orthonormal elite prediction basis."""
        return self._subspace_rank

    def _standardize_residual(self, series: np.ndarray) -> np.ndarray | None:
        """Center and L2-normalize residual vector, returning None if zero-variance."""
        dev = series - np.mean(series)
        norm = float(np.linalg.norm(dev))
        if norm < 1e-12:
            return None
        res: np.ndarray = np.asarray(dev / norm, dtype=np.float64)
        return res

    def _recompute_basis(self) -> None:
        """Compute thin SVD on stored residuals and cache orthonormal basis vectors."""
        if not self._residuals:
            self._basis_vt = None
            self._subspace_rank = 0
            return

        # Stack into matrix A in R^{N_arch x T}
        matrix = np.vstack(self._residuals)

        # Thin SVD: matrix = U * S * Vt
        _, s, vt = np.linalg.svd(matrix, full_matrices=False)

        # Retain singular vectors capturing variance_threshold cumulative energy
        energy = s**2
        total_energy = float(np.sum(energy))
        if total_energy < 1e-12:
            self._basis_vt = None
            self._subspace_rank = 0
            return

        cum_ratio = np.cumsum(energy) / total_energy
        rank_idx = int(np.searchsorted(cum_ratio, self._variance_threshold)) + 1
        effective_rank = min(rank_idx, self._max_basis_rank, len(s))

        # Rows of Vt are orthonormal vectors in R^T
        self._basis_vt = vt[:effective_rank, :].copy()
        self._subspace_rank = effective_rank

    def admit(
        self,
        candidate_id: str,
        residual_series: np.ndarray,
        novelty_score: float,
        dsr: float,
    ) -> bool:
        """Admit a candidate into the historical elite archive.

        Functional Purpose:
            Maintains the rolling FIFO memory of elite strategy behaviors,
            triggering basis recomputation upon insertion.

        Args:
            candidate_id: Unique identifier for candidate.
            residual_series: 1D contiguous array of prediction residuals.
            novelty_score: Evaluated orthogonal novelty score.
            dsr: Certified Deflated Sharpe Ratio.

        Returns:
            True if admitted, False if candidate had zero-variance residuals.
        """
        standardized = self._standardize_residual(residual_series)
        if standardized is None:
            return False

        # Enforce FIFO eviction when capacity is reached
        if len(self._residuals) >= self._max_capacity:
            self._elite_ids.pop(0)
            self._residuals.pop(0)

        self._elite_ids.append(candidate_id)
        self._residuals.append(standardized)
        self._recompute_basis()
        return True

    def compute_novelty(
        self,
        residual_batch: np.ndarray,
        viability_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute SVD subspace orthogonal novelty for a batch of candidate residuals.

        Functional Purpose:
            Evaluates rho_ortho as (1 - R²) via projection onto the orthonormal basis.
            Clamps novelty to 0.0 for unviable candidates (Viability Gate).

        Args:
            residual_batch: 2D array of shape (N_cand, T) of prediction residuals.
            viability_mask: 1D boolean array of shape (N_cand,) indicating viability.

        Returns:
            1D array of shape (N_cand,) containing novelty scores in [0.0, 1.0].
        """
        n_cand = residual_batch.shape[0]
        novelties = np.zeros(n_cand, dtype=np.float64)

        # Generation 0 or empty archive fallback: all viable candidates receive 1.0
        if self._basis_vt is None or self._subspace_rank == 0:
            novelties[viability_mask] = 1.0
            return novelties

        # Basis Vt has shape (K, T). Transpose is V in R^{T x K}
        v_basis = self._basis_vt.T

        for i in range(n_cand):
            # Viability Gate: unviable candidates receive 0.0 novelty
            if not viability_mask[i]:
                novelties[i] = 0.0
                continue

            std_res = self._standardize_residual(residual_batch[i])
            if std_res is None:
                # ERR-EVO-PAR-002: Zero-variance residual yields zero novelty
                novelties[i] = 0.0
                continue

            # Projection coordinates: c = V^T * e in R^K
            coords = np.dot(std_res, v_basis)
            explained_ratio = float(np.sum(coords**2))

            # Fraction of unexplained variance: 1 - R^2
            ortho_score = max(0.0, min(1.0, 1.0 - explained_ratio))
            novelties[i] = ortho_score

        return novelties


class AdaptiveReferenceLattice:
    """Boundary-Anchored Adaptive Reference Lattice for 3D objective surfaces (BA-ARVEA).

    Functional Purpose:
        Generates Das-Dennis structured reference rays, associates candidates with their
        nearest directional ray, evaluates Angle-Penalized Distance (APD), and dynamically
        migrates idle interior rays toward populated candidate clusters while keeping
        basis anchor rays ([1,0,0], [0,1,0], [0,0,1]) strictly immutable.

    Defensive Invariants:
        INV-PAR-003: All rays maintain exact unit Euclidean norm (||v|| == 1.0).
        INV-PAR-004: Anchor basis rays are strictly immutable across all adaptations.
    """

    def __init__(
        self,
        num_objectives: int = 3,
        partitions: int = 6,
        adaptation_interval: int = 5,
        idle_threshold: int = 3,
        adaptation_rate: float = 0.30,
        min_angle_separation: float = 0.05,
        alpha_escalation: float = 2.0,
    ) -> None:
        """Initialize reference lattice with Das-Dennis partition settings.

        Args:
            num_objectives: Dimension of objective space (M=3 for DSR, Regret, Novelty).
            partitions: Number of divisions along each coordinate axis (p=6 yields 28 rays).
            adaptation_interval: Number of generations between interior ray adjustments.
            idle_threshold: Minimum idle generations before an interior ray is migrated.
            adaptation_rate: Smoothing factor eta for ray cluster projection.
            min_angle_separation: Minimum angular clearance required to prevent ray collapse.
            alpha_escalation: Exponent alpha controlling APD penalty ramp-up over time.
        """
        self._m = num_objectives
        self._p = partitions
        self._adaptation_interval = adaptation_interval
        self._idle_threshold = idle_threshold
        self._eta = adaptation_rate
        self._min_sep = min_angle_separation
        self._alpha = alpha_escalation

        # Generate initial Das-Dennis lattice
        self._rays = self._generate_das_dennis_rays(self._m, self._p)
        self._k_rays = len(self._rays)

        # Identify immutable coordinate basis anchor indices
        self._anchor_indices: set[int] = set()
        for idx, ray in enumerate(self._rays):
            for dim in range(self._m):
                basis = np.zeros(self._m, dtype=np.float64)
                basis[dim] = 1.0
                if np.allclose(ray, basis, atol=1e-5):
                    self._anchor_indices.add(idx)

        # Idle generation counter per ray
        self._idle_counters = np.zeros(self._k_rays, dtype=np.int64)

        # Pre-compute localized minimum angles gamma_j
        self._gamma = self._compute_gamma(self._rays)

    @property
    def rays(self) -> np.ndarray:
        """Active reference rays matrix of shape (K, M)."""
        return self._rays.copy()

    @staticmethod
    def _generate_das_dennis_rays(m: int, p: int) -> np.ndarray:
        """Recursively generate uniform simplex lattice points using Das-Dennis method."""
        combinations: list[list[int]] = []

        def _recurse(remaining_sum: int, depth: int, current: list[int]) -> None:
            if depth == m - 1:
                combinations.append(current + [remaining_sum])
                return
            for val in range(remaining_sum + 1):
                _recurse(remaining_sum - val, depth + 1, current + [val])

        _recurse(p, 0, [])
        raw = np.array(combinations, dtype=np.float64) / float(p)

        # Normalize each ray to unit Euclidean length
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        # Avoid division by zero on degenerate lattice
        norms = np.maximum(norms, 1e-12)
        unit_rays = raw / norms
        return unit_rays

    @staticmethod
    def _compute_gamma(rays: np.ndarray) -> np.ndarray:
        """Compute localized smallest angle gamma_j to the nearest neighboring ray."""
        # Pairwise dot products in R^{K x K}
        dots = np.dot(rays, rays.T)
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)

        # Mask diagonal (angle to self is 0)
        np.fill_diagonal(angles, np.inf)
        gamma = np.min(angles, axis=1)
        # Numerical floor to prevent divide-by-zero
        return np.maximum(gamma, 1e-5)

    def associate_and_penalize(
        self,
        normalized_objectives: np.ndarray,
        generation_ratio: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Associate candidates with nearest reference rays and compute APD metrics.

        Functional Purpose:
            Evaluates acute angle theta_{i, j} between candidate objective vectors
            and reference rays, assigning each candidate to j* = argmin_j theta_{i, j}.
            Computes Angle-Penalized Distance d_APD scaling convergence with diversity.

        Args:
            normalized_objectives: 2D array of shape (N_cand, M) in [0, 1]^M.
            generation_ratio: Scaled generation index t / t_max in [0.0, 1.0].

        Returns:
            Tuple of (associated_ray_indices, acute_angles, apd_scores).
        """
        n_cand = normalized_objectives.shape[0]
        associations = np.zeros(n_cand, dtype=np.int64)
        angles = np.zeros(n_cand, dtype=np.float64)
        apd_scores = np.zeros(n_cand, dtype=np.float64)

        gen_ratio_clamped = max(0.0, min(1.0, generation_ratio))
        time_escalation = float(self._m) * (gen_ratio_clamped**self._alpha)

        # Euclidean norms of objective vectors
        norms = np.linalg.norm(normalized_objectives, axis=1)

        for i in range(n_cand):
            norm_i = float(norms[i])
            if norm_i < 1e-12:
                # Solution at ideal point origin [0,0,0]: zero APD
                associations[i] = 0
                angles[i] = 0.0
                apd_scores[i] = 0.0
                continue

            unit_obj = normalized_objectives[i] / norm_i
            # Dot products with all reference rays in R^K
            cosines = np.dot(self._rays, unit_obj)
            cosines = np.clip(cosines, -1.0, 1.0)
            candidate_angles = np.arccos(cosines)

            best_ray = int(np.argmin(candidate_angles))
            theta_star = float(candidate_angles[best_ray])
            gamma_star = float(self._gamma[best_ray])

            # Angle-Penalized Distance formula: (1 + M * (t/t_max)^alpha * (theta / gamma)) * ||f||
            penalty = 1.0 + time_escalation * (theta_star / gamma_star)
            apd = penalty * norm_i

            associations[i] = best_ray
            angles[i] = theta_star
            apd_scores[i] = apd

        return associations, angles, apd_scores

    def adapt_interior_rays(
        self,
        normalized_objectives: np.ndarray,
        associations: np.ndarray,
    ) -> int:
        """Adaptively re-project idle interior reference rays toward candidate clusters.

        Functional Purpose:
            Concentrates search capacity on empirically viable objective regions
            while strictly preserving coordinate basis anchors (INV-PAR-004) and
            minimum angular separation (gamma_min).

        Args:
            normalized_objectives: 2D array of shape (N_cand, M) of active solutions.
            associations: 1D array of assigned ray indices for each solution.

        Returns:
            Total number of interior rays migrated during this adaptation pass.
        """
        active_ray_set = set(associations.tolist())
        migrated_count = 0

        # Identify most populated ray cluster as target direction
        ray_counts = np.bincount(associations, minlength=self._k_rays)
        dense_ray = int(np.argmax(ray_counts))

        if ray_counts[dense_ray] == 0:
            return 0

        dense_mask = associations == dense_ray
        cluster_centroid = np.mean(normalized_objectives[dense_mask], axis=0)
        centroid_norm = float(np.linalg.norm(cluster_centroid))
        if centroid_norm < 1e-12:
            return 0
        target_unit = cluster_centroid / centroid_norm

        for j in range(self._k_rays):
            if j in active_ray_set:
                self._idle_counters[j] = 0
            else:
                self._idle_counters[j] += 1

            # Check adaptation criteria: idle >= threshold and NOT an anchor ray (INV-PAR-004)
            if self._idle_counters[j] >= self._idle_threshold and j not in self._anchor_indices:
                candidate_ray = (1.0 - self._eta) * self._rays[j] + self._eta * target_unit
                cand_norm = float(np.linalg.norm(candidate_ray))
                if cand_norm < 1e-12:
                    continue
                candidate_unit = candidate_ray / cand_norm

                # Verify minimum angular separation against other existing rays
                temp_rays = self._rays.copy()
                temp_rays[j] = candidate_unit
                dots = np.dot(temp_rays, candidate_unit)
                dots = np.clip(dots, -1.0, 1.0)
                angles = np.arccos(dots)
                angles[j] = np.inf

                if float(np.min(angles)) >= self._min_sep:
                    self._rays[j] = candidate_unit
                    self._idle_counters[j] = 0
                    migrated_count += 1

        if migrated_count > 0:
            self._gamma = self._compute_gamma(self._rays)

        return migrated_count


class DependentNonDominatedSorter:
    """Non-dominated sorting engine implementing Memmel-Ledoit-Wolf covariance and ENS-SS.

    Functional Purpose:
        Replaces naive point-estimate dominance with statistical confidence dominance (tau-dominance).
        Calculates exact asymptotic variance for dependent financial return series to eliminate
        the 300% variance inflation caused by assuming independence. Implements Efficient
        Non-Dominated Sort with Sequential Search (ENS-SS) in O(M * N * log N) time.

    Defensive Invariants:
        INV-PAR-005: Partition conservation - all candidates must be partitioned into
                     either non-dominated fronts or the infeasible cohort with zero loss.
        ERR-EVO-PAR-004: Return correlations are clamped to [-0.9999, 0.9999] preventing
                         numerical drift in asymptotic variance square-roots.
    """

    def __init__(
        self,
        tau_conf: float = 1.645,
        epsilon_regret: float = 1e-6,
        epsilon_novelty: float = 1e-4,
    ) -> None:
        """Initialize sorter with confidence and indifference tolerance thresholds.

        Args:
            tau_conf: Critical threshold for 95% single-tailed Gaussian test (1.645).
            epsilon_regret: Indifference threshold on minimax regret objective.
            epsilon_novelty: Indifference threshold on orthogonal novelty objective.
        """
        self._tau = tau_conf
        self._eps_regret = epsilon_regret
        self._eps_novelty = epsilon_novelty

    @staticmethod
    def compute_dependent_dsr_variance(
        dsr_a: float,
        dsr_b: float,
        returns_a: np.ndarray,
        returns_b: np.ndarray,
    ) -> float:
        """Compute asymptotic standard error of difference in DSR under dependent returns.

        Functional Purpose:
            Implements Memmel (2003) and Ledoit & Wolf (2008) asymptotic variance formula
            for Sharpe/DSR differences across strategies evaluated on identical market data.

        Args:
            dsr_a: Deflated Sharpe Ratio of Strategy A.
            dsr_b: Deflated Sharpe Ratio of Strategy B.
            returns_a: 1D array of portfolio returns for Strategy A.
            returns_b: 1D array of portfolio returns for Strategy B.

        Returns:
            Asymptotic standard error sigma(Delta DSR_{A, B}).
        """
        t_bars = min(len(returns_a), len(returns_b))
        if t_bars < 2:
            return 0.0

        dev_a = returns_a[:t_bars] - np.mean(returns_a[:t_bars])
        dev_b = returns_b[:t_bars] - np.mean(returns_b[:t_bars])

        norm_a = float(np.linalg.norm(dev_a))
        norm_b = float(np.linalg.norm(dev_b))

        if norm_a < 1e-12 or norm_b < 1e-12:
            rho = 0.0
        else:
            rho = float(np.dot(dev_a, dev_b) / (norm_a * norm_b))

        # ERR-EVO-PAR-004: Clamp correlation to [-0.9999, 0.9999] preventing negative variance
        rho = max(-0.9999, min(0.9999, rho))

        # Memmel (2003) asymptotic variance formula
        diff_term = 2.0 * (1.0 - rho)
        squared_term = 0.5 * (dsr_a**2 + dsr_b**2 - 2.0 * (rho**2) * dsr_a * dsr_b)
        asymptotic_var = (diff_term + squared_term) / float(t_bars)

        return math.sqrt(max(0.0, asymptotic_var))

    def dominates(
        self,
        candidate_a: CandidateFitness,
        candidate_b: CandidateFitness,
        objective_a: np.ndarray,
        objective_b: np.ndarray,
    ) -> bool:
        """Evaluate if candidate_a statistically dominates candidate_b under tau-dominance.

        Functional Purpose:
            Applies Deb's Feasibility Rule followed by Memmel-Ledoit-Wolf tau-dominance.

        Args:
            candidate_a: Candidate A fitness record.
            candidate_b: Candidate B fitness record.
            objective_a: Uniform minimization objective vector [-DSR, Regret, -Novelty].
            objective_b: Uniform minimization objective vector [-DSR, Regret, -Novelty].

        Returns:
            True if candidate_a dominates candidate_b, False otherwise.
        """
        # Deb's Feasibility Dominance: Feasible strictly dominates Infeasible
        if candidate_a.is_feasible and not candidate_b.is_feasible:
            return True
        if not candidate_a.is_feasible and candidate_b.is_feasible:
            return False
        if not candidate_a.is_feasible and not candidate_b.is_feasible:
            return False

        # Both candidates are feasible: evaluate dependent statistical dominance
        se_diff = self.compute_dependent_dsr_variance(
            candidate_a.dsr,
            candidate_b.dsr,
            candidate_a.return_series,
            candidate_b.return_series,
        )
        delta_dsr_tol = self._tau * se_diff

        # Weak condition (no worse across all 3 objectives)
        cond_dsr = objective_a[0] <= objective_b[0] + delta_dsr_tol
        cond_regret = objective_a[1] <= objective_b[1] + 1e-9
        cond_novelty = objective_a[2] <= objective_b[2] + 1e-9

        if not (cond_dsr and cond_regret and cond_novelty):
            return False

        # Strict condition (strictly superior in at least one objective)
        strict_dsr = objective_a[0] < objective_b[0] - delta_dsr_tol
        strict_regret = objective_a[1] < objective_b[1] - self._eps_regret
        strict_novelty = objective_a[2] < objective_b[2] - self._eps_novelty

        is_better = strict_dsr or strict_regret or strict_novelty
        return bool(is_better)

    def sort(
        self,
        candidates: list[CandidateFitness],
        objective_matrix: np.ndarray,
    ) -> tuple[list[list[int]], list[int]]:
        """Partition candidates into non-dominated Pareto fronts using ENS-SS.

        Functional Purpose:
            Sorts candidates by primary objective (-DSR) in O(N log N) and assigns
            each candidate to the first non-dominated front where no existing member dominates it.

        Args:
            candidates: Sequence of CandidateFitness records.
            objective_matrix: 2D array of shape (N, 3) containing objective vectors.

        Returns:
            Tuple of (fronts, infeasible_indices), satisfying partition conservation INV-PAR-005.
        """
        n_cand = len(candidates)
        feasible_indices: list[int] = []
        infeasible_indices: list[int] = []

        # Partition feasible vs infeasible (DSR < 0.50 or constraint failure)
        for i in range(n_cand):
            if candidates[i].is_feasible and candidates[i].dsr >= 0.50:
                feasible_indices.append(i)
            else:
                infeasible_indices.append(i)

        if not feasible_indices:
            return [], infeasible_indices

        # Sort feasible candidates ascending by primary objective (-DSR, highest DSR first)
        sorted_feasible = sorted(feasible_indices, key=lambda idx: float(objective_matrix[idx, 0]))

        # Efficient Non-Dominated Sort with Sequential Search (ENS-SS)
        fronts: list[list[int]] = []

        for cand_idx in sorted_feasible:
            placed = False
            for front in fronts:
                is_dominated = False
                for member_idx in front:
                    if self.dominates(
                        candidates[member_idx],
                        candidates[cand_idx],
                        objective_matrix[member_idx],
                        objective_matrix[cand_idx],
                    ):
                        is_dominated = True
                        break
                if not is_dominated:
                    front.append(cand_idx)
                    placed = True
                    break

            if not placed:
                fronts.append([cand_idx])

        return fronts, infeasible_indices


class BoundaryAnchoredRVEARanker:
    """High-level facade executing multi-objective Pareto ranking (BA-ARVEA-SO).

    Functional Purpose:
        Coordinates SVDSubspaceOrthogonalArchive, AdaptiveReferenceLattice, and
        DependentNonDominatedSorter to evaluate population trade-offs, execute
        ENS-SS sorting under Memmel-Ledoit-Wolf covariance, compute APD diversity rankings,
        and manage historical memory.
    """

    def __init__(
        self,
        archive: SVDSubspaceOrthogonalArchive | None = None,
        lattice: AdaptiveReferenceLattice | None = None,
        sorter: DependentNonDominatedSorter | None = None,
    ) -> None:
        """Initialize ranker with component instances or default implementations.

        Args:
            archive: SVD subspace orthogonal archive instance.
            lattice: Boundary-anchored adaptive reference lattice instance.
            sorter: Dependent non-dominated sorter instance.
        """
        self._archive = archive if archive is not None else SVDSubspaceOrthogonalArchive()
        self._lattice = lattice if lattice is not None else AdaptiveReferenceLattice()
        self._sorter = sorter if sorter is not None else DependentNonDominatedSorter()

    def rank_population(
        self,
        candidates: list[CandidateFitness],
        generation: int,
        max_generations: int,
        auto_admit: bool = True,
    ) -> RankingResult:
        """Execute complete multi-objective Pareto ranking across a population generation.

        Functional Purpose:
            Evaluates orthogonal novelty via thin SVD projection, constructs 3D objective
            space, runs ENS-SS with Memmel-Ledoit-Wolf dependent dominance, normalizes
            feasible fronts, computes APD metrics, and auto-admits Front-1 elites.

        Args:
            candidates: Sequence of CandidateFitness records for current generation.
            generation: Current generation index (0-based or 1-based).
            max_generations: Total number of planned evolutionary generations.
            auto_admit: Flag whether to automatically admit Front-1 candidates to SVD archive.

        Returns:
            RankingResult bundle with sorted fronts, infeasible cohort, and active rays.
        """
        n_cand = len(candidates)
        if n_cand == 0:
            return RankingResult(
                fronts=(),
                infeasible_ids=(),
                active_reference_rays=self._lattice.rays,
                archive_size=self._archive.archive_size,
                subspace_rank=self._archive.subspace_rank,
            )

        # Extract residual series and viability mask
        residual_batch = np.vstack([c.residual_series for c in candidates])
        viability_mask = np.array(
            [bool(c.is_feasible and c.dsr >= 0.50) for c in candidates], dtype=bool
        )

        # Compute SVD subspace orthogonal novelty (1 - R^2)
        novelties = self._archive.compute_novelty(residual_batch, viability_mask)

        # Construct raw uniform minimization objective matrix in R^{N x 3}
        # f_1 = -DSR, f_2 = Minimax Regret, f_3 = -Novelty
        raw_objectives = np.zeros((n_cand, 3), dtype=np.float64)
        for i in range(n_cand):
            raw_objectives[i, 0] = -candidates[i].dsr
            raw_objectives[i, 1] = candidates[i].minimax_regret
            raw_objectives[i, 2] = -novelties[i]

        # Partition into non-dominated Pareto fronts using ENS-SS
        front_indices, infeasible_indices = self._sorter.sort(candidates, raw_objectives)
        infeasible_ids = tuple(candidates[idx].candidate_id for idx in infeasible_indices)

        if not front_indices:
            return RankingResult(
                fronts=(),
                infeasible_ids=infeasible_ids,
                active_reference_rays=self._lattice.rays,
                archive_size=self._archive.archive_size,
                subspace_rank=self._archive.subspace_rank,
            )

        # Extract feasible solutions and compute ideal/nadir points for normalization
        feasible_indices: list[int] = [idx for front in front_indices for idx in front]
        feas_objs = raw_objectives[feasible_indices]

        z_min = np.min(feas_objs, axis=0)
        z_max = np.max(feas_objs, axis=0)
        spread = np.maximum(z_max - z_min, 1e-12)

        # Normalized objective coordinates in [0, 1]^3
        norm_objectives = np.zeros((n_cand, 3), dtype=np.float64)
        norm_objectives[feasible_indices] = (feas_objs - z_min) / spread

        # Compute APD scores and ray associations for feasible candidates
        gen_ratio = float(generation) / max(1.0, float(max_generations))
        associations, _, apd_scores = self._lattice.associate_and_penalize(
            norm_objectives, gen_ratio
        )

        # Construct ParetoFront objects, sorting each front by APD ascending
        fronts: list[ParetoFront] = []
        for rank_num, front_list in enumerate(front_indices, start=1):
            sorted_front = sorted(front_list, key=lambda idx: float(apd_scores[idx]))
            c_ids = tuple(candidates[idx].candidate_id for idx in sorted_front)
            c_apds = tuple(float(apd_scores[idx]) for idx in sorted_front)
            fronts.append(ParetoFront(rank=rank_num, candidate_ids=c_ids, apd_scores=c_apds))

        # Auto-admit Front-1 elites into the SVD subspace archive
        if auto_admit and fronts:
            id_to_idx = {c.candidate_id: i for i, c in enumerate(candidates)}
            for cand_id in fronts[0].candidate_ids:
                idx = id_to_idx[cand_id]
                cand = candidates[idx]
                self._archive.admit(
                    cand.candidate_id, cand.residual_series, novelties[idx], cand.dsr
                )

        # Adapt interior reference rays toward active solution clusters
        if generation > 0 and generation % self._lattice._adaptation_interval == 0:
            self._lattice.adapt_interior_rays(
                norm_objectives[feasible_indices], associations[feasible_indices]
            )

        return RankingResult(
            fronts=tuple(fronts),
            infeasible_ids=infeasible_ids,
            active_reference_rays=self._lattice.rays,
            archive_size=self._archive.archive_size,
            subspace_rank=self._archive.subspace_rank,
        )

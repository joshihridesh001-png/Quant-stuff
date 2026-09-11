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

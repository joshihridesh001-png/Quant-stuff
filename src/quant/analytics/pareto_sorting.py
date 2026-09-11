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

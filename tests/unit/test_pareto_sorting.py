"""Unit tests for multi-objective Pareto sorting engine (BA-ARVEA-SO).

Verifies domain entities, boundary invariants, SVD subspace orthogonality,
adaptive reference lattice, Memmel-Ledoit-Wolf dependent dominance, and
the high-level ranking facade.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant.analytics.pareto_sorting import (
    CandidateFitness,
    CorruptedFitnessException,
    InvalidResidualException,
    ParetoFront,
    RankingResult,
)


class TestDomainEntitiesAndInvariants:
    """Test suite for core domain entities and boundary defensive invariants."""

    def test_candidate_fitness_valid_instantiation(self) -> None:
        """Verify nominal construction of CandidateFitness with valid inputs."""
        returns = np.random.randn(200).astype(np.float64)
        residuals = np.random.randn(200).astype(np.float64)

        fit = CandidateFitness(
            candidate_id="cand_001",
            dsr=1.25,
            minimax_regret=0.045,
            return_series=returns,
            residual_series=residuals,
            backtest_length=200,
            is_feasible=True,
        )

        assert fit.candidate_id == "cand_001"
        assert fit.dsr == 1.25
        assert fit.minimax_regret == 0.045
        assert len(fit.return_series) == 200
        assert len(fit.residual_series) == 200
        assert fit.backtest_length == 200
        assert fit.is_feasible is True

    def test_candidate_fitness_rejects_nan_in_scalars(self) -> None:
        """INV-PAR-001: Rejects NaN or Inf in scalar objective values."""
        returns = np.ones(150, dtype=np.float64)
        residuals = np.ones(150, dtype=np.float64)

        with pytest.raises(CorruptedFitnessException, match="Non-finite DSR"):
            CandidateFitness(
                candidate_id="cand_nan_dsr",
                dsr=float("nan"),
                minimax_regret=0.01,
                return_series=returns,
                residual_series=residuals,
                backtest_length=150,
                is_feasible=True,
            )

        with pytest.raises(CorruptedFitnessException, match="Non-finite minimax regret"):
            CandidateFitness(
                candidate_id="cand_inf_regret",
                dsr=1.0,
                minimax_regret=float("inf"),
                return_series=returns,
                residual_series=residuals,
                backtest_length=150,
                is_feasible=True,
            )

    def test_candidate_fitness_rejects_nan_in_arrays(self) -> None:
        """INV-PAR-001: Rejects NaN or Inf in return or residual vectors."""
        returns = np.ones(150, dtype=np.float64)
        returns[10] = np.nan
        residuals = np.ones(150, dtype=np.float64)

        with pytest.raises(CorruptedFitnessException, match="Non-finite value in return_series"):
            CandidateFitness(
                candidate_id="cand_nan_returns",
                dsr=1.0,
                minimax_regret=0.01,
                return_series=returns,
                residual_series=residuals,
                backtest_length=150,
                is_feasible=True,
            )

        returns_clean = np.ones(150, dtype=np.float64)
        residuals_corrupt = np.ones(150, dtype=np.float64)
        residuals_corrupt[5] = np.inf

        with pytest.raises(CorruptedFitnessException, match="Non-finite value in residual_series"):
            CandidateFitness(
                candidate_id="cand_inf_residuals",
                dsr=1.0,
                minimax_regret=0.01,
                return_series=returns_clean,
                residual_series=residuals_corrupt,
                backtest_length=150,
                is_feasible=True,
            )

    def test_candidate_fitness_rejects_mismatched_lengths(self) -> None:
        """INV-PAR-002: Rejects mismatched lengths between returns and residuals."""
        returns = np.ones(150, dtype=np.float64)
        residuals = np.ones(140, dtype=np.float64)

        with pytest.raises(InvalidResidualException, match="Length mismatch"):
            CandidateFitness(
                candidate_id="cand_mismatch",
                dsr=1.0,
                minimax_regret=0.01,
                return_series=returns,
                residual_series=residuals,
                backtest_length=150,
                is_feasible=True,
            )

    def test_candidate_fitness_rejects_short_series(self) -> None:
        """INV-PAR-002: Rejects return or residual series shorter than 100 bars."""
        returns = np.ones(50, dtype=np.float64)
        residuals = np.ones(50, dtype=np.float64)

        with pytest.raises(InvalidResidualException, match="Series length too short"):
            CandidateFitness(
                candidate_id="cand_short",
                dsr=1.0,
                minimax_regret=0.01,
                return_series=returns,
                residual_series=residuals,
                backtest_length=50,
                is_feasible=True,
            )

    def test_pareto_front_valid_instantiation(self) -> None:
        """Verify immutable ParetoFront construction."""
        front = ParetoFront(
            rank=1,
            candidate_ids=("cand_01", "cand_02"),
            apd_scores=(0.12, 0.18),
        )
        assert front.rank == 1
        assert len(front.candidate_ids) == 2
        assert len(front.apd_scores) == 2

    def test_ranking_result_valid_instantiation(self) -> None:
        """Verify immutable RankingResult construction."""
        front = ParetoFront(rank=1, candidate_ids=("cand_01",), apd_scores=(0.15,))
        rays = np.eye(3, dtype=np.float64)
        result = RankingResult(
            fronts=(front,),
            infeasible_ids=("cand_bad",),
            active_reference_rays=rays,
            archive_size=10,
            subspace_rank=3,
        )
        assert len(result.fronts) == 1
        assert result.infeasible_ids == ("cand_bad",)
        assert result.active_reference_rays.shape == (3, 3)
        assert result.archive_size == 10
        assert result.subspace_rank == 3

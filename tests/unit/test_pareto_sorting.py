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


class TestSVDSubspaceOrthogonalArchive:
    """Test suite for SVDSubspaceOrthogonalArchive."""

    def test_empty_archive_returns_unit_novelty_for_viable(self) -> None:
        """Empty archive returns 1.0 for viable candidates and 0.0 for unviable."""
        from quant.analytics.pareto_sorting import SVDSubspaceOrthogonalArchive

        archive = SVDSubspaceOrthogonalArchive(max_capacity=50)
        assert archive.archive_size == 0
        assert archive.subspace_rank == 0

        residuals = np.random.randn(3, 120).astype(np.float64)
        viability = np.array([True, False, True], dtype=bool)

        novelties = archive.compute_novelty(residuals, viability)
        assert len(novelties) == 3
        assert np.isclose(novelties[0], 1.0)
        assert np.isclose(novelties[1], 0.0)  # Viability gated
        assert np.isclose(novelties[2], 1.0)

    def test_svd_subspace_detects_multi_collinear_redundancy(self) -> None:
        """Verifies that a candidate spanned by multiple archive elites yields rho ≈ 0.0."""
        from quant.analytics.pareto_sorting import SVDSubspaceOrthogonalArchive

        archive = SVDSubspaceOrthogonalArchive(max_capacity=50)

        # Create two orthogonal basis residual series
        t_bars = 200
        e1 = np.sin(np.linspace(0, 10 * np.pi, t_bars))
        e2 = np.cos(np.linspace(0, 10 * np.pi, t_bars))

        archive.admit("elite_1", e1, novelty_score=1.0, dsr=1.5)
        archive.admit("elite_2", e2, novelty_score=1.0, dsr=1.6)
        assert archive.archive_size == 2
        assert archive.subspace_rank == 2

        # Create a candidate that is an exact linear combination: e_cand = 0.6 * e1 + 0.8 * e2
        e_redundant = 0.6 * e1 + 0.8 * e2
        # Create a truly orthogonal candidate: independent random noise
        np.random.seed(42)
        e_orthogonal = np.random.randn(t_bars)
        # Ensure exact orthogonality to e1 and e2
        e_orthogonal -= (np.dot(e_orthogonal, e1) / np.dot(e1, e1)) * e1
        e_orthogonal -= (np.dot(e_orthogonal, e2) / np.dot(e2, e2)) * e2

        batch = np.vstack([e_redundant, e_orthogonal])
        viability = np.array([True, True], dtype=bool)

        novelties = archive.compute_novelty(batch, viability)

        # Redundant candidate must have near-zero novelty (spanned by e1 and e2)
        assert novelties[0] < 1e-4, f"Expected near-zero novelty, got {novelties[0]}"
        # Orthogonal candidate must have near 1.0 novelty
        assert novelties[1] > 0.95, f"Expected high novelty, got {novelties[1]}"

    def test_archive_fifo_eviction(self) -> None:
        """Verifies FIFO buffer eviction when capacity exceeds max_capacity."""
        from quant.analytics.pareto_sorting import SVDSubspaceOrthogonalArchive

        archive = SVDSubspaceOrthogonalArchive(max_capacity=5)
        t_bars = 100

        for i in range(7):
            res = np.random.randn(t_bars)
            admitted = archive.admit(f"cand_{i}", res, novelty_score=0.8, dsr=1.0 + i * 0.1)
            assert admitted is True

        assert archive.archive_size == 5

    def test_zero_variance_residuals_clamped_safely(self) -> None:
        """ERR-EVO-PAR-002: Zero-variance residual series does not trigger divide-by-zero NaN."""
        from quant.analytics.pareto_sorting import SVDSubspaceOrthogonalArchive

        archive = SVDSubspaceOrthogonalArchive(max_capacity=10)
        t_bars = 100
        archive.admit("elite_1", np.random.randn(t_bars), 1.0, 1.2)

        # Candidate with identical constant values (zero variance)
        zero_var_res = np.ones((1, t_bars), dtype=np.float64) * 42.0
        viability = np.array([True], dtype=bool)

        novelties = archive.compute_novelty(zero_var_res, viability)
        assert np.isfinite(novelties[0])
        assert novelties[0] == 0.0


class TestAdaptiveReferenceLattice:
    """Test suite for AdaptiveReferenceLattice (BA-ARVEA)."""

    def test_lattice_initialization_dimension_and_anchors(self) -> None:
        """Das-Dennis with M=3, p=6 generates 28 unit-norm rays with 3 basis anchors."""
        from quant.analytics.pareto_sorting import AdaptiveReferenceLattice

        lattice = AdaptiveReferenceLattice(num_objectives=3, partitions=6)
        rays = lattice.rays

        assert rays.shape == (28, 3)
        # Verify unit norm for all rays (INV-PAR-003)
        norms = np.linalg.norm(rays, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

        # Verify basis anchors [1,0,0], [0,1,0], [0,0,1] exist (INV-PAR-004)
        assert any(np.allclose(r, [1.0, 0.0, 0.0], atol=1e-6) for r in rays)
        assert any(np.allclose(r, [0.0, 1.0, 0.0], atol=1e-6) for r in rays)
        assert any(np.allclose(r, [0.0, 0.0, 1.0], atol=1e-6) for r in rays)

    def test_anchor_rays_are_strictly_immutable(self) -> None:
        """INV-PAR-004: Basis anchors never change during adaptation cycles."""
        from quant.analytics.pareto_sorting import AdaptiveReferenceLattice

        lattice = AdaptiveReferenceLattice(num_objectives=3, partitions=6, idle_threshold=1)

        # Create a cluster far from [1,0,0], say near [0.2, 0.5, 0.8]
        cluster = np.array([[0.2, 0.5, 0.8], [0.22, 0.48, 0.82]], dtype=np.float64)
        associations, _, _ = lattice.associate_and_penalize(cluster, generation_ratio=0.5)

        # Force multiple adaptation cycles
        for _ in range(5):
            lattice.adapt_interior_rays(cluster, associations)

        updated_rays = lattice.rays
        # Verify the 3 anchors are exactly preserved
        assert any(np.allclose(r, [1.0, 0.0, 0.0], atol=1e-6) for r in updated_rays)
        assert any(np.allclose(r, [0.0, 1.0, 0.0], atol=1e-6) for r in updated_rays)
        assert any(np.allclose(r, [0.0, 0.0, 1.0], atol=1e-6) for r in updated_rays)

    def test_associate_and_penalize_escalates_with_generation(self) -> None:
        """Angle-penalized distance escalates with generation ratio t / t_max."""
        from quant.analytics.pareto_sorting import AdaptiveReferenceLattice

        lattice = AdaptiveReferenceLattice(num_objectives=3, partitions=6)

        # An objective vector with non-zero angle to all rays
        objs = np.array([[0.3, 0.7, 0.2]], dtype=np.float64)

        _, _, apd_early = lattice.associate_and_penalize(objs, generation_ratio=0.0)
        _, _, apd_late = lattice.associate_and_penalize(objs, generation_ratio=1.0)

        # Late generation penalty must be strictly greater than early generation
        assert apd_late[0] > apd_early[0]

    def test_zero_norm_objective_vector_safe_handling(self) -> None:
        """Solution at origin [0,0,0] yields zero APD without division-by-zero exception."""
        from quant.analytics.pareto_sorting import AdaptiveReferenceLattice

        lattice = AdaptiveReferenceLattice(num_objectives=3, partitions=6)
        objs = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

        assoc, angles, apd = lattice.associate_and_penalize(objs, generation_ratio=0.5)
        assert len(assoc) == 1
        assert apd[0] == 0.0
        assert np.isfinite(angles[0])


class TestDependentNonDominatedSorter:
    """Test suite for DependentNonDominatedSorter (Memmel-Ledoit-Wolf & ENS-SS)."""

    def test_memmel_ledoit_wolf_covariance_scales_with_correlation(self) -> None:
        """Memmel-Ledoit-Wolf SE is strictly lower when returns are strongly correlated."""
        from quant.analytics.pareto_sorting import DependentNonDominatedSorter

        sorter = DependentNonDominatedSorter()
        t_bars = 500
        np.random.seed(42)

        base_returns = np.random.randn(t_bars)
        # Highly correlated returns (rho ≈ 0.95)
        r_corr = 0.95 * base_returns + 0.05 * np.random.randn(t_bars)
        # Independent returns (rho ≈ 0.0)
        r_indep = np.random.randn(t_bars)

        se_correlated = sorter.compute_dependent_dsr_variance(
            dsr_a=1.2, dsr_b=1.1, returns_a=base_returns, returns_b=r_corr
        )
        se_independent = sorter.compute_dependent_dsr_variance(
            dsr_a=1.2, dsr_b=1.1, returns_a=base_returns, returns_b=r_indep
        )

        assert se_correlated < se_independent
        assert se_correlated > 0.0

    def test_dependent_dominance_detects_real_advantage_when_correlated(self) -> None:
        """Delta DSR of 0.08 is statistically significant when rho=0.95, but not when rho=0.0."""
        from quant.analytics.pareto_sorting import (
            CandidateFitness,
            DependentNonDominatedSorter,
        )

        sorter = DependentNonDominatedSorter()
        t_bars = 300
        np.random.seed(123)

        r_base = np.random.randn(t_bars)
        r_high_corr = 0.95 * r_base + 0.05 * np.random.randn(t_bars)
        r_zero_corr = np.random.randn(t_bars)

        res = np.random.randn(t_bars)

        cand_a = CandidateFitness(
            candidate_id="cand_a",
            dsr=1.20,
            minimax_regret=0.05,
            return_series=r_base,
            residual_series=res,
            backtest_length=t_bars,
            is_feasible=True,
        )
        cand_b_corr = CandidateFitness(
            candidate_id="cand_b_corr",
            dsr=1.12,  # Diff = 0.08
            minimax_regret=0.05,
            return_series=r_high_corr,
            residual_series=res,
            backtest_length=t_bars,
            is_feasible=True,
        )
        cand_b_indep = CandidateFitness(
            candidate_id="cand_b_indep",
            dsr=1.12,  # Diff = 0.08
            minimax_regret=0.05,
            return_series=r_zero_corr,
            residual_series=res,
            backtest_length=t_bars,
            is_feasible=True,
        )

        # Uniform minimization: [-DSR, Regret, -Novelty]
        obj_a = np.array([-1.20, 0.05, -0.80])
        obj_b_corr = np.array([-1.12, 0.05, -0.80])
        obj_b_indep = np.array([-1.12, 0.05, -0.80])

        # Under high correlation, delta DSR of 0.08 is statistically significant -> A dominates B
        assert sorter.dominates(cand_a, cand_b_corr, obj_a, obj_b_corr) is True

        # Under independence, delta DSR of 0.08 is swallowed by noise -> A does not dominate B
        assert sorter.dominates(cand_a, cand_b_indep, obj_a, obj_b_indep) is False

    def test_deb_feasibility_rule(self) -> None:
        """Feasible candidate dominates infeasible candidate regardless of objective values."""
        from quant.analytics.pareto_sorting import (
            CandidateFitness,
            DependentNonDominatedSorter,
        )

        sorter = DependentNonDominatedSorter()
        t_bars = 150
        r = np.random.randn(t_bars)

        cand_feasible = CandidateFitness(
            candidate_id="cand_feas",
            dsr=0.60,
            minimax_regret=0.10,
            return_series=r,
            residual_series=r,
            backtest_length=t_bars,
            is_feasible=True,
        )
        cand_infeasible = CandidateFitness(
            candidate_id="cand_infeas",
            dsr=2.50,  # Huge DSR, but violates constraints!
            minimax_regret=0.001,
            return_series=r,
            residual_series=r,
            backtest_length=t_bars,
            is_feasible=False,
        )

        obj_feas = np.array([-0.60, 0.10, -0.50])
        obj_infeas = np.array([-2.50, 0.001, -0.99])

        # Feasible strictly dominates infeasible
        assert sorter.dominates(cand_feasible, cand_infeasible, obj_feas, obj_infeas) is True
        assert sorter.dominates(cand_infeasible, cand_feasible, obj_infeas, obj_feas) is False

    def test_ens_ss_sorting_partition_conservation(self) -> None:
        """INV-PAR-005: All candidates are partitioned into fronts or infeasible cohort."""
        from quant.analytics.pareto_sorting import (
            CandidateFitness,
            DependentNonDominatedSorter,
        )

        sorter = DependentNonDominatedSorter()
        t_bars = 100
        np.random.seed(99)

        candidates: list[CandidateFitness] = []
        obj_list: list[list[float]] = []

        for i in range(10):
            r = np.random.randn(t_bars)
            is_feas = i != 8  # 1 candidate is infeasible
            dsr_val = 1.0 + float(i) * 0.1
            fit = CandidateFitness(
                candidate_id=f"cand_{i}",
                dsr=dsr_val,
                minimax_regret=0.05 + float(i) * 0.01,
                return_series=r,
                residual_series=r,
                backtest_length=t_bars,
                is_feasible=is_feas,
            )
            candidates.append(fit)
            obj_list.append([-dsr_val, fit.minimax_regret, -0.5])

        obj_matrix = np.array(obj_list)
        fronts, infeasible = sorter.sort(candidates, obj_matrix)

        # Infeasible candidate cand_8 must be in infeasible list
        assert 8 in infeasible

        # Partition conservation: union of fronts and infeasible set equals all 10 indices
        partitioned = set(infeasible)
        for f in fronts:
            for idx in f:
                assert idx not in partitioned, f"Duplicate candidate {idx} across fronts"
                partitioned.add(idx)

        assert partitioned == set(range(10))

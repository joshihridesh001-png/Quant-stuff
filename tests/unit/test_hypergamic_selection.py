"""Unit tests for Phase 4, Step 3: Hypergamic Assortative Selection & Residual Orthogonality Gating."""

import pytest

from quant.analytics.hypergamic_selection import (
    HypergamicConfig,
    HypergamicSelectionError,
    InvalidCohortException,
    MatingDeadlockException,
    MatingPair,
    OffspringResult,
)


class TestDomainEntitiesAndInvariants:
    """Test domain dataclasses, invariants, and validation rules."""

    def test_hypergamic_config_defaults_and_validation(self) -> None:
        """Nominal defaults and boundary invariant validation in HypergamicConfig."""
        cfg = HypergamicConfig()
        assert cfg.alpha_ratio == 0.25
        assert cfg.orthogonality_threshold == 0.30
        assert cfg.max_mating_attempts == 5
        assert cfg.relaxation_factor == 0.80
        assert cfg.tournament_size == 3
        assert cfg.crossover_distribution_index == 15.0
        assert cfg.alpha_risk_inheritance_prob == 0.75
        assert cfg.aspirant_repr_inheritance_prob == 0.65
        assert cfg.elitism_count == 2

        # Reject out-of-bounds alpha_ratio
        with pytest.raises(HypergamicSelectionError, match="alpha_ratio must be in"):
            HypergamicConfig(alpha_ratio=0.0)
        with pytest.raises(HypergamicSelectionError, match="alpha_ratio must be in"):
            HypergamicConfig(alpha_ratio=1.0)

        # Reject out-of-bounds orthogonality_threshold
        with pytest.raises(HypergamicSelectionError, match="orthogonality_threshold must be in"):
            HypergamicConfig(orthogonality_threshold=-0.1)
        with pytest.raises(HypergamicSelectionError, match="orthogonality_threshold must be in"):
            HypergamicConfig(orthogonality_threshold=1.05)

        # Reject non-positive max_mating_attempts
        with pytest.raises(HypergamicSelectionError, match="max_mating_attempts must be >= 1"):
            HypergamicConfig(max_mating_attempts=0)

        # Reject out-of-bounds relaxation_factor
        with pytest.raises(HypergamicSelectionError, match="relaxation_factor must be in"):
            HypergamicConfig(relaxation_factor=0.0)
        with pytest.raises(HypergamicSelectionError, match="relaxation_factor must be in"):
            HypergamicConfig(relaxation_factor=1.5)

        # Reject non-positive tournament_size
        with pytest.raises(HypergamicSelectionError, match="tournament_size must be >= 1"):
            HypergamicConfig(tournament_size=0)

        # Reject non-positive crossover_distribution_index
        with pytest.raises(HypergamicSelectionError, match="crossover_distribution_index must be > 0"):
            HypergamicConfig(crossover_distribution_index=0.0)

        # Reject negative elitism_count
        with pytest.raises(HypergamicSelectionError, match="elitism_count must be >= 0"):
            HypergamicConfig(elitism_count=-1)

    def test_mating_pair_and_offspring_result_immutability(self) -> None:
        """MatingPair and OffspringResult are immutable frozen dataclasses."""
        pair = MatingPair(
            alpha_id="alpha_1",
            aspirant_id="asp_2",
            residual_correlation=0.12,
            is_relaxed=False,
            relaxation_level=0,
        )
        assert pair.alpha_id == "alpha_1"
        assert pair.residual_correlation == 0.12

        with pytest.raises(Exception):
            pair.alpha_id = "alpha_2"  # type: ignore[misc]

        res = OffspringResult(
            offspring_chromosomes=(),
            mating_pairs=(pair,),
            alpha_ids=("alpha_1",),
            aspirant_ids=("asp_2",),
            elite_ids=(),
            rejection_count=0,
            relaxation_count=0,
        )
        assert len(res.mating_pairs) == 1
        with pytest.raises(Exception):
            res.rejection_count = 5  # type: ignore[misc]


class TestParetoCohortStratifier:
    """Test Pareto cohort stratification logic."""

    def test_stratification_preserves_large_front_1(self) -> None:
        """When Front 1 size exceeds alpha_ratio * N, all of Front 1 is preserved."""
        import numpy as np
        from quant.analytics.hypergamic_selection import (
            HypergamicConfig,
            ParetoCohortStratifier,
        )
        from quant.analytics.pareto_sorting import (
            CandidateFitness,
            ParetoFront,
            RankingResult,
        )

        t_bars = 100
        candidates = []
        for i in range(20):
            r = np.random.randn(t_bars)
            fit = CandidateFitness(
                candidate_id=f"cand_{i:02d}",
                dsr=1.0 + float(i) * 0.05,
                minimax_regret=0.05,
                return_series=r,
                residual_series=r,
                backtest_length=t_bars,
                is_feasible=True,
            )
            candidates.append(fit)

        # Front 1 has 8 members (0..7), Front 2 has 12 members (8..19)
        f1_ids = tuple(f"cand_{i:02d}" for i in range(8))
        f2_ids = tuple(f"cand_{i:02d}" for i in range(8, 20))
        ranking = RankingResult(
            fronts=(
                ParetoFront(rank=1, candidate_ids=f1_ids, apd_scores=tuple(0.1 * i for i in range(8))),
                ParetoFront(rank=2, candidate_ids=f2_ids, apd_scores=tuple(0.2 * i for i in range(12))),
            ),
            infeasible_ids=(),
            active_reference_rays=np.zeros((28, 3)),
            archive_size=8,
            subspace_rank=3,
        )

        stratifier = ParetoCohortStratifier(HypergamicConfig(alpha_ratio=0.25))
        alphas, aspirants = stratifier.stratify(candidates, ranking)

        # Target alpha count was 5, but Front 1 has 8 -> Front-preserving retains all 8!
        assert len(alphas) == 8
        assert {a.candidate_id for a in alphas} == set(f1_ids)
        assert len(aspirants) == 12
        assert {asp.candidate_id for asp in aspirants} == set(f2_ids)

    def test_stratification_pads_from_front_2_when_small(self) -> None:
        """When Front 1 is smaller than target alpha count, pad from Front 2 using APD order."""
        import numpy as np
        from quant.analytics.hypergamic_selection import (
            HypergamicConfig,
            ParetoCohortStratifier,
        )
        from quant.analytics.pareto_sorting import (
            CandidateFitness,
            ParetoFront,
            RankingResult,
        )

        t_bars = 100
        candidates = []
        for i in range(20):
            r = np.random.randn(t_bars)
            fit = CandidateFitness(
                candidate_id=f"cand_{i:02d}",
                dsr=1.0 + float(i) * 0.05,
                minimax_regret=0.05,
                return_series=r,
                residual_series=r,
                backtest_length=t_bars,
                is_feasible=True,
            )
            candidates.append(fit)

        # Front 1 has only 2 members (target is 5). Front 2 has 6 members.
        f1_ids = ("cand_00", "cand_01")
        f2_ids = ("cand_02", "cand_03", "cand_04", "cand_05", "cand_06", "cand_07")
        f3_ids = tuple(f"cand_{i:02d}" for i in range(8, 20))

        # APD scores: lower is better. cand_04 has best (0.01), cand_02 has (0.05), cand_05 has (0.10)
        f2_apd = (0.05, 0.20, 0.01, 0.10, 0.30, 0.40)

        ranking = RankingResult(
            fronts=(
                ParetoFront(rank=1, candidate_ids=f1_ids, apd_scores=(0.1, 0.2)),
                ParetoFront(rank=2, candidate_ids=f2_ids, apd_scores=f2_apd),
                ParetoFront(rank=3, candidate_ids=f3_ids, apd_scores=tuple(0.5 * i for i in range(12))),
            ),
            infeasible_ids=(),
            active_reference_rays=np.zeros((28, 3)),
            archive_size=2,
            subspace_rank=1,
        )

        stratifier = ParetoCohortStratifier(HypergamicConfig(alpha_ratio=0.25))  # target = 5
        alphas, aspirants = stratifier.stratify(candidates, ranking)

        assert len(alphas) == 5
        alpha_ids = {a.candidate_id for a in alphas}
        # Includes all of Front 1
        assert "cand_00" in alpha_ids and "cand_01" in alpha_ids
        # Top 3 from Front 2 by APD: cand_04 (0.01), cand_02 (0.05), cand_05 (0.10)
        assert "cand_04" in alpha_ids
        assert "cand_02" in alpha_ids
        assert "cand_05" in alpha_ids

        # Remaining 15 candidates are in aspirants
        assert len(aspirants) == 15
        assert alpha_ids.isdisjoint({asp.candidate_id for asp in aspirants})

    def test_stratification_excludes_infeasible_candidates(self) -> None:
        """Infeasible candidates are rejected from both Alpha and Aspirant cohorts."""
        import numpy as np
        from quant.analytics.hypergamic_selection import (
            HypergamicConfig,
            ParetoCohortStratifier,
        )
        from quant.analytics.pareto_sorting import (
            CandidateFitness,
            ParetoFront,
            RankingResult,
        )

        t_bars = 100
        candidates = []
        for i in range(10):
            r = np.random.randn(t_bars)
            fit = CandidateFitness(
                candidate_id=f"cand_{i}",
                dsr=1.0,
                minimax_regret=0.05,
                return_series=r,
                residual_series=r,
                backtest_length=t_bars,
                is_feasible=(i != 9),  # cand_9 is infeasible
            )
            candidates.append(fit)

        ranking = RankingResult(
            fronts=(
                ParetoFront(rank=1, candidate_ids=("cand_0", "cand_1"), apd_scores=(0.1, 0.2)),
                ParetoFront(rank=2, candidate_ids=("cand_2", "cand_3"), apd_scores=(0.3, 0.4)),
            ),
            infeasible_ids=("cand_9",),
            active_reference_rays=np.zeros((28, 3)),
            archive_size=2,
            subspace_rank=1,
        )

        stratifier = ParetoCohortStratifier(HypergamicConfig(alpha_ratio=0.30))
        alphas, aspirants = stratifier.stratify(candidates, ranking)

        all_cohort_ids = {c.candidate_id for c in alphas} | {c.candidate_id for c in aspirants}
        assert "cand_9" not in all_cohort_ids

    def test_stratification_no_viable_raises_invalid_cohort(self) -> None:
        """When all candidates are infeasible or below DSR threshold, raises InvalidCohortException."""
        import numpy as np
        from quant.analytics.hypergamic_selection import (
            HypergamicConfig,
            InvalidCohortException,
            ParetoCohortStratifier,
        )
        from quant.analytics.pareto_sorting import (
            CandidateFitness,
            RankingResult,
        )

        t_bars = 100
        r = np.random.randn(t_bars)
        cand = CandidateFitness(
            candidate_id="inf_0",
            dsr=0.10,
            minimax_regret=0.05,
            return_series=r,
            residual_series=r,
            backtest_length=t_bars,
            is_feasible=False,
        )
        ranking = RankingResult(
            fronts=(),
            infeasible_ids=("inf_0",),
            active_reference_rays=np.zeros((28, 3)),
            archive_size=0,
            subspace_rank=0,
        )

        stratifier = ParetoCohortStratifier(HypergamicConfig())
        with pytest.raises(InvalidCohortException, match="No viable candidates available"):
            stratifier.stratify([cand], ranking)


class TestResidualOrthogonalityGate:
    """Test bidirectional absolute residual orthogonality gating."""

    def test_gate_rejects_positive_collinear_clones(self) -> None:
        """Residual correlation rho = 0.95 fails gate (unexplained variance only 5%)."""
        import numpy as np
        from quant.analytics.hypergamic_selection import (
            HypergamicConfig,
            ResidualOrthogonalityGate,
        )
        from quant.analytics.pareto_sorting import CandidateFitness

        np.random.seed(42)
        t_bars = 150
        base_res = np.random.randn(t_bars)
        clone_res = 0.95 * base_res + 0.05 * np.random.randn(t_bars)

        ret = np.random.randn(t_bars)
        alpha = CandidateFitness("alpha", 1.2, 0.05, ret, base_res, t_bars, True)
        clone = CandidateFitness("clone", 1.0, 0.06, ret, clone_res, t_bars, True)

        gate = ResidualOrthogonalityGate(HypergamicConfig(orthogonality_threshold=0.30))
        accepted, rho = gate.evaluate(alpha, clone, relaxation_level=0)

        assert accepted is False
        assert rho > 0.90

    def test_gate_rejects_negative_inverse_clones(self) -> None:
        """Residual correlation rho = -0.95 fails gate (catches inverse clones that naive signed corr passes)."""
        import numpy as np
        from quant.analytics.hypergamic_selection import (
            HypergamicConfig,
            ResidualOrthogonalityGate,
        )
        from quant.analytics.pareto_sorting import CandidateFitness

        np.random.seed(42)
        t_bars = 150
        base_res = np.random.randn(t_bars)
        inv_res = -0.95 * base_res + 0.05 * np.random.randn(t_bars)

        ret = np.random.randn(t_bars)
        alpha = CandidateFitness("alpha", 1.2, 0.05, ret, base_res, t_bars, True)
        inv_clone = CandidateFitness("inv_clone", 1.0, 0.06, ret, inv_res, t_bars, True)

        gate = ResidualOrthogonalityGate(HypergamicConfig(orthogonality_threshold=0.30))
        accepted, rho = gate.evaluate(alpha, inv_clone, relaxation_level=0)

        # Naive signed correlation (rho < 0.30) would have mistakenly accepted this!
        # Our bidirectional gate correctly rejects inverse clones!
        assert accepted is False
        assert rho < -0.90

    def test_gate_accepts_orthogonal_residuals(self) -> None:
        """Uncorrelated residuals pass gate."""
        import numpy as np
        from quant.analytics.hypergamic_selection import (
            HypergamicConfig,
            ResidualOrthogonalityGate,
        )
        from quant.analytics.pareto_sorting import CandidateFitness

        np.random.seed(123)
        t_bars = 200
        res_a = np.random.randn(t_bars)
        res_b = np.random.randn(t_bars)

        ret = np.random.randn(t_bars)
        alpha = CandidateFitness("alpha", 1.2, 0.05, ret, res_a, t_bars, True)
        aspirant = CandidateFitness("asp", 1.0, 0.06, ret, res_b, t_bars, True)

        gate = ResidualOrthogonalityGate(HypergamicConfig(orthogonality_threshold=0.30))
        accepted, rho = gate.evaluate(alpha, aspirant, relaxation_level=0)

        assert accepted is True
        assert abs(rho) < 0.20

    def test_gate_adaptive_relaxation_lowers_threshold(self) -> None:
        """Relaxation level > 0 multiplies threshold by relaxation_factor."""
        import numpy as np
        from quant.analytics.hypergamic_selection import (
            HypergamicConfig,
            ResidualOrthogonalityGate,
        )
        from quant.analytics.pareto_sorting import CandidateFitness

        np.random.seed(99)
        t_bars = 150
        base_res = np.random.randn(t_bars)
        # Moderate correlation: rho ~ 0.75 -> unexplained variance ~ 0.25
        mod_res = 0.75 * base_res + 0.66 * np.random.randn(t_bars)

        ret = np.random.randn(t_bars)
        alpha = CandidateFitness("alpha", 1.2, 0.05, ret, base_res, t_bars, True)
        aspirant = CandidateFitness("asp", 1.0, 0.06, ret, mod_res, t_bars, True)

        # Base threshold = 0.30 -> 0.25 < 0.30 (rejected at level 0)
        gate = ResidualOrthogonalityGate(
            HypergamicConfig(orthogonality_threshold=0.30, relaxation_factor=0.80)
        )
        accepted_0, _ = gate.evaluate(alpha, aspirant, relaxation_level=0)
        assert accepted_0 is False

        # Level 2 threshold = 0.30 * 0.80^2 = 0.192 -> 0.25 >= 0.192 (accepted at level 2)
        accepted_2, _ = gate.evaluate(alpha, aspirant, relaxation_level=2)
        assert accepted_2 is True

    def test_gate_zero_variance_residuals_safe_rejection(self) -> None:
        """Zero-variance flat residual series rejected without division by zero."""
        import numpy as np
        from quant.analytics.hypergamic_selection import (
            HypergamicConfig,
            ResidualOrthogonalityGate,
        )
        from quant.analytics.pareto_sorting import CandidateFitness

        t_bars = 100
        ret = np.random.randn(t_bars)
        flat_res = np.ones(t_bars)
        normal_res = np.random.randn(t_bars)

        alpha = CandidateFitness("alpha", 1.0, 0.05, ret, flat_res, t_bars, True)
        asp = CandidateFitness("asp", 1.0, 0.05, ret, normal_res, t_bars, True)

        gate = ResidualOrthogonalityGate(HypergamicConfig())
        accepted, rho = gate.evaluate(alpha, asp, relaxation_level=0)
        assert accepted is False
        assert rho == 1.0




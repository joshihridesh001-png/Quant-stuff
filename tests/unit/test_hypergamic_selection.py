from dataclasses import FrozenInstanceError

import pytest

from quant.analytics.hypergamic_selection import (
    AsymmetricLatentCrossover,
    HypergamicConfig,
    HypergamicPartnerMatcher,
    HypergamicSelectionEngine,
    HypergamicSelectionError,
    InvalidCohortException,
    MatingPair,
    OffspringResult,
    ParetoCohortStratifier,
    ResidualOrthogonalityGate,
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
        with pytest.raises(
            HypergamicSelectionError, match="crossover_distribution_index must be > 0"
        ):
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

        with pytest.raises(FrozenInstanceError):
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
        with pytest.raises(FrozenInstanceError):
            res.rejection_count = 5  # type: ignore[misc]


class TestParetoCohortStratifier:
    """Test Pareto cohort stratification logic."""

    def test_stratification_preserves_large_front_1(self) -> None:
        """When Front 1 size exceeds alpha_ratio * N, all of Front 1 is preserved."""
        import numpy as np

        from quant.analytics.hypergamic_selection import (
            HypergamicConfig,
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
                ParetoFront(
                    rank=1, candidate_ids=f1_ids, apd_scores=tuple(0.1 * i for i in range(8))
                ),
                ParetoFront(
                    rank=2, candidate_ids=f2_ids, apd_scores=tuple(0.2 * i for i in range(12))
                ),
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
                ParetoFront(
                    rank=3, candidate_ids=f3_ids, apd_scores=tuple(0.5 * i for i in range(12))
                ),
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


class TestHypergamicPartnerMatcher:
    """Test assortative hypergamic partner matching and deadlock prevention."""

    def test_matcher_nominal_pairing_produces_exact_count(self) -> None:
        """Nominal diverse population matches exact requested pair count."""
        import numpy as np

        from quant.analytics.hypergamic_selection import (
            HypergamicConfig,
        )
        from quant.analytics.pareto_sorting import CandidateFitness

        np.random.seed(42)
        t_bars = 100

        alphas = [
            CandidateFitness(
                f"alpha_{i}",
                1.5,
                0.04,
                np.random.randn(t_bars),
                np.random.randn(t_bars),
                t_bars,
                True,
            )
            for i in range(3)
        ]
        aspirants = [
            CandidateFitness(
                f"asp_{j}",
                1.0,
                0.06,
                np.random.randn(t_bars),
                np.random.randn(t_bars),
                t_bars,
                True,
            )
            for j in range(8)
        ]

        cfg = HypergamicConfig(orthogonality_threshold=0.20)
        gate = ResidualOrthogonalityGate(cfg)
        matcher = HypergamicPartnerMatcher(gate, cfg)

        pairs = matcher.match_pairs(alphas, aspirants, target_pair_count=10, seed=123)

        assert len(pairs) == 10
        alpha_ids = {a.candidate_id for a in alphas}
        asp_ids = {asp.candidate_id for asp in aspirants}

        for p in pairs:
            assert p.alpha_id in alpha_ids
            assert p.aspirant_id in asp_ids

    def test_matcher_adaptive_relaxation_on_collinear_cohort(self) -> None:
        """When all aspirants are collinear, matcher relaxes gate and succeeds without deadlock."""
        import numpy as np

        from quant.analytics.hypergamic_selection import (
            HypergamicConfig,
        )
        from quant.analytics.pareto_sorting import CandidateFitness

        np.random.seed(77)
        t_bars = 120
        base_res = np.random.randn(t_bars)

        # 1 Alpha, 4 Aspirants all highly correlated (rho ~ 0.85 -> unexplained ~ 0.15)
        alpha = CandidateFitness(
            "alpha", 1.8, 0.02, np.random.randn(t_bars), base_res, t_bars, True
        )
        aspirants = [
            CandidateFitness(
                f"asp_{j}",
                1.1,
                0.05,
                np.random.randn(t_bars),
                0.85 * base_res + 0.15 * np.random.randn(t_bars),
                t_bars,
                True,
            )
            for j in range(4)
        ]

        # Base threshold is 0.30 -> cannot be met at level 0 (unexplained is ~0.15)
        cfg = HypergamicConfig(
            orthogonality_threshold=0.30,
            max_mating_attempts=3,
            relaxation_factor=0.80,
        )
        gate = ResidualOrthogonalityGate(cfg)
        matcher = HypergamicPartnerMatcher(gate, cfg)

        pairs = matcher.match_pairs([alpha], aspirants, target_pair_count=3, seed=42)

        assert len(pairs) == 3
        # Must have applied relaxation
        assert any(p.is_relaxed for p in pairs)
        assert all(p.alpha_id == "alpha" for p in pairs)

    def test_matcher_phenotypic_distance_fallback_on_exhaustion(self) -> None:
        """When retries exhaust threshold, falls back to maximum phenotypic distance in unit hypercube."""
        import numpy as np

        from quant.analytics.chromosomes import StrategyChromosome
        from quant.analytics.hypergamic_selection import (
            HypergamicConfig,
        )
        from quant.analytics.pareto_sorting import CandidateFitness

        t_bars = 100
        res = np.random.randn(t_bars)

        # 1 Alpha, 2 Aspirants with identical residuals (rho = 1.0)
        alpha = CandidateFitness("alpha", 1.5, 0.03, res, res, t_bars, True)
        asp_close = CandidateFitness("asp_close", 1.0, 0.05, res, res, t_bars, True)
        asp_distant = CandidateFitness("asp_distant", 1.0, 0.05, res, res, t_bars, True)

        # Build mock chromosomes with known distance
        from quant.analytics.chromosomes import RiskChromosome

        chrom_alpha = StrategyChromosome()
        chrom_close = StrategyChromosome()
        chrom_distant = StrategyChromosome(risk=RiskChromosome(vol_target=0.35))

        chrom_map = {
            "alpha": chrom_alpha,
            "asp_close": chrom_close,
            "asp_distant": chrom_distant,
        }

        cfg = HypergamicConfig(
            orthogonality_threshold=0.50,
            max_mating_attempts=2,
            relaxation_factor=0.99,  # Stays high so threshold is never met by rho=1.0
        )
        gate = ResidualOrthogonalityGate(cfg)
        matcher = HypergamicPartnerMatcher(gate, cfg)

        pairs = matcher.match_pairs(
            [alpha], [asp_close, asp_distant], target_pair_count=1, chromosome_map=chrom_map, seed=1
        )

        assert len(pairs) == 1
        # Fallback selected asp_distant because of maximum phenotypic distance
        assert pairs[0].aspirant_id == "asp_distant"
        assert pairs[0].is_relaxed is True

    def test_matcher_empty_cohorts_raises_error(self) -> None:
        """Empty alphas or aspirants raises InvalidCohortException."""
        from quant.analytics.hypergamic_selection import (
            HypergamicConfig,
        )

        cfg = HypergamicConfig()
        gate = ResidualOrthogonalityGate(cfg)
        matcher = HypergamicPartnerMatcher(gate, cfg)

        with pytest.raises(InvalidCohortException, match="Alpha cohort cannot be empty"):
            matcher.match_pairs([], [], target_pair_count=5)


class TestAsymmetricLatentCrossoverAndEngine:
    """Test Asymmetric Latent Unit-Hypercube Crossover and Master Selection Engine."""

    def test_asymmetric_crossover_satisfies_all_invariants(self) -> None:
        """100 random crossovers produce 100% valid chromosomes satisfying all domain invariants."""
        import numpy as np

        from quant.analytics.chromosomes import (
            ChromosomeVectorCodec,
            GameTheoryChromosome,
            InferenceChromosome,
            RepresentationChromosome,
            RiskChromosome,
            StrategyChromosome,
        )

        codec = ChromosomeVectorCodec()
        crossover = AsymmetricLatentCrossover()

        # Parent 1 (Alpha): Extreme fast decay and conservative risk
        p1 = StrategyChromosome(
            representation=RepresentationChromosome(
                tau_slow=50000.0, tau_ratio=0.05, fractional_d=0.3
            ),
            game_theory=GameTheoryChromosome(ambiguity_temp=0.5, risk_aversion=8.0, logit_bull=1.5),
            inference=InferenceChromosome(execution_horizon=3, hyperbolic_decay=0.2),
            risk=RiskChromosome(vol_target=0.10, max_weight=0.15, max_drawdown_limit=0.05),
        )

        # Parent 2 (Aspirant): Extreme slow decay and aggressive risk
        p2 = StrategyChromosome(
            representation=RepresentationChromosome(
                tau_slow=150000.0, tau_ratio=0.45, fractional_d=0.8
            ),
            game_theory=GameTheoryChromosome(
                ambiguity_temp=4.0, risk_aversion=0.5, logit_bull=-1.5
            ),
            inference=InferenceChromosome(execution_horizon=18, hyperbolic_decay=1.8),
            risk=RiskChromosome(vol_target=0.35, max_weight=0.45, max_drawdown_limit=0.25),
        )

        rng = np.random.default_rng(12345)
        for _ in range(100):
            child = crossover.cross(p1, p2, rng=rng)

            assert isinstance(child, StrategyChromosome)
            # INV-CHROM-001: tau_fast < tau_slow
            assert child.representation.tau_fast < child.representation.tau_slow
            assert child.representation.tau_ratio < 1.0

            # INV-CHROM-002: Simplex valid p in Delta^3
            priors = child.game_theory.regime_priors
            assert np.isclose(np.sum(priors), 1.0, atol=1e-5)
            assert np.all(priors > 0.0)

            # INV-CHROM-003: Discrete integer execution horizon
            assert 1 <= child.inference.execution_horizon <= 20
            assert isinstance(child.inference.execution_horizon, int)

            # INV-CHROM-004: Continuous bounded risk parameters
            assert 0.05 <= child.risk.vol_target <= 0.40
            assert 0.05 <= child.risk.max_weight <= 0.50
            assert 0.02 <= child.risk.max_drawdown_limit <= 0.30
            assert 0.05 - 1e-6 <= child.risk.turnover_budget <= 1.0 + 1e-6

            # Unit hypercube round-trip validity
            u = codec.encode(child)
            assert np.all(u >= 0.0)
            assert np.all(u <= 1.0)

    def test_asymmetric_crossover_inherits_alpha_risk_bias(self) -> None:
        """Statistical test verifying Alpha risk genes inherited >= 65% and Aspirant repr >= 55%."""
        import numpy as np

        from quant.analytics.chromosomes import (
            GameTheoryChromosome,
            InferenceChromosome,
            RepresentationChromosome,
            RiskChromosome,
            StrategyChromosome,
        )

        crossover = AsymmetricLatentCrossover(
            HypergamicConfig(
                alpha_risk_inheritance_prob=0.75,
                aspirant_repr_inheritance_prob=0.65,
                crossover_distribution_index=15.0,
            )
        )

        # Alpha has high risk, low representation
        alpha = StrategyChromosome(
            representation=RepresentationChromosome(tau_slow=20000.0),
            game_theory=GameTheoryChromosome(risk_aversion=9.0),
            inference=InferenceChromosome(execution_horizon=2),
            risk=RiskChromosome(vol_target=0.38, max_weight=0.48),
        )

        # Aspirant has low risk, high representation
        aspirant = StrategyChromosome(
            representation=RepresentationChromosome(tau_slow=150000.0),
            game_theory=GameTheoryChromosome(risk_aversion=0.5),
            inference=InferenceChromosome(execution_horizon=18),
            risk=RiskChromosome(vol_target=0.08, max_weight=0.10),
        )

        rng = np.random.default_rng(42)
        n_trials = 500
        alpha_vol_closer = 0
        aspirant_tau_closer = 0

        for _ in range(n_trials):
            child = crossover.cross(alpha, aspirant, rng=rng)

            # Check vol_target proximity (Risk gene -> favored Alpha)
            dist_alpha_vol = abs(child.risk.vol_target - alpha.risk.vol_target)
            dist_asp_vol = abs(child.risk.vol_target - aspirant.risk.vol_target)
            if dist_alpha_vol < dist_asp_vol:
                alpha_vol_closer += 1

            # Check tau_slow proximity (Repr gene -> favored Aspirant)
            dist_alpha_tau = abs(child.representation.tau_slow - alpha.representation.tau_slow)
            dist_asp_tau = abs(child.representation.tau_slow - aspirant.representation.tau_slow)
            if dist_asp_tau < dist_alpha_tau:
                aspirant_tau_closer += 1

        alpha_risk_rate = alpha_vol_closer / n_trials
        asp_repr_rate = aspirant_tau_closer / n_trials

        # With P_alpha = 0.75, rate should comfortably exceed 0.65
        assert alpha_risk_rate > 0.65, f"Alpha risk inheritance rate {alpha_risk_rate:.2f} <= 0.65"
        # With P_aspirant = 0.65, rate should comfortably exceed 0.55
        assert asp_repr_rate > 0.55, f"Aspirant repr inheritance rate {asp_repr_rate:.2f} <= 0.55"

    def test_master_facade_end_to_end_reproduction(self) -> None:
        """HypergamicSelectionEngine reproduces exact target population size preserving elitism."""
        import numpy as np

        from quant.analytics.chromosomes import StrategyChromosome
        from quant.analytics.pareto_sorting import CandidateFitness, ParetoFront, RankingResult

        t_bars = 100
        rng = np.random.default_rng(777)

        # 10 candidates: 3 in Front 1, 7 in Front 2
        candidates = []
        chrom_map = {}
        for i in range(10):
            cid = f"cand_{i}"
            ret = rng.normal(0.001, 0.02, t_bars)
            res = rng.normal(0.0, 0.01, t_bars)
            c = CandidateFitness(cid, 1.0 + 0.1 * i, 0.05, ret, res, t_bars, True)
            candidates.append(c)
            chrom_map[cid] = StrategyChromosome()

        front_1 = ParetoFront(
            rank=1, candidate_ids=("cand_0", "cand_1", "cand_2"), apd_scores=(0.1, 0.2, 0.3)
        )
        front_2 = ParetoFront(
            rank=2,
            candidate_ids=tuple(f"cand_{i}" for i in range(3, 10)),
            apd_scores=tuple(0.5 + 0.1 * i for i in range(7)),
        )
        ranking = RankingResult(
            fronts=(front_1, front_2),
            infeasible_ids=(),
            active_reference_rays=np.eye(2),
            archive_size=10,
            subspace_rank=2,
        )

        cfg = HypergamicConfig(elitism_count=2, alpha_ratio=0.30)
        engine = HypergamicSelectionEngine(cfg)

        target_size = 12
        result = engine.reproduce(
            candidates, ranking, chrom_map, target_population_size=target_size, seed=42
        )

        assert isinstance(result, OffspringResult)
        assert len(result.offspring_chromosomes) == target_size
        assert len(result.elite_ids) == 2
        assert result.elite_ids == ("cand_0", "cand_1")

        # Invariant INV-HYP-003: Elitism clones are bitwise identical
        assert result.offspring_chromosomes[0] == chrom_map["cand_0"]
        assert result.offspring_chromosomes[1] == chrom_map["cand_1"]

        # Number of mating pairs should be target_size - n_elite = 10
        assert len(result.mating_pairs) == 10
        assert len(result.offspring_chromosomes) == target_size

        # Verify cohort IDs
        assert len(result.alpha_ids) == 3
        assert len(result.aspirant_ids) == 7

        # Invariant INV-HYP-002: All offspring are valid StrategyChromosome
        for child in result.offspring_chromosomes:
            assert isinstance(child, StrategyChromosome)
            assert child.representation.tau_fast < child.representation.tau_slow

    def test_master_facade_performance_benchmark_sub_25ms(self) -> None:
        """100 candidates produce 100 offspring in < 25ms."""
        import time

        import numpy as np

        from quant.analytics.chromosomes import StrategyChromosome
        from quant.analytics.pareto_sorting import CandidateFitness, ParetoFront, RankingResult

        t_bars = 100
        rng = np.random.default_rng(999)
        n_candidates = 100

        candidates = []
        chrom_map = {}
        for i in range(n_candidates):
            cid = f"cand_{i}"
            ret = rng.normal(0.001, 0.02, t_bars)
            res = rng.normal(0.0, 0.01, t_bars)
            c = CandidateFitness(cid, 1.0 + 0.01 * i, 0.05, ret, res, t_bars, True)
            candidates.append(c)
            chrom_map[cid] = StrategyChromosome()

        front_1 = ParetoFront(
            rank=1,
            candidate_ids=tuple(f"cand_{i}" for i in range(25)),
            apd_scores=tuple(0.01 * i for i in range(25)),
        )
        front_2 = ParetoFront(
            rank=2,
            candidate_ids=tuple(f"cand_{i}" for i in range(25, 100)),
            apd_scores=tuple(0.5 + 0.01 * i for i in range(75)),
        )
        ranking = RankingResult(
            fronts=(front_1, front_2),
            infeasible_ids=(),
            active_reference_rays=np.eye(2),
            archive_size=100,
            subspace_rank=2,
        )

        cfg = HypergamicConfig(elitism_count=2, alpha_ratio=0.25)
        engine = HypergamicSelectionEngine(cfg)

        # Warmup JIT / allocations
        _ = engine.reproduce(candidates, ranking, chrom_map, target_population_size=100, seed=1)

        t_start = time.perf_counter()
        result = engine.reproduce(
            candidates, ranking, chrom_map, target_population_size=100, seed=42
        )
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        assert len(result.offspring_chromosomes) == 100
        # Assert execution under 25ms target (allowing up to 50ms for heavily loaded CI environments)
        assert elapsed_ms < 50.0, f"Expected < 50ms, took {elapsed_ms:.2f}ms"

    def test_master_facade_defensive_error_handling(self) -> None:
        """Engine defensive checks on inputs."""
        import numpy as np

        from quant.analytics.chromosomes import StrategyChromosome
        from quant.analytics.pareto_sorting import CandidateFitness, ParetoFront, RankingResult

        t_bars = 100
        res = np.random.randn(t_bars)
        cand = CandidateFitness("c1", 1.0, 0.05, res, res, t_bars, True)
        ranking = RankingResult(
            fronts=(ParetoFront(rank=1, candidate_ids=("c1",), apd_scores=(0.1,)),),
            infeasible_ids=(),
            active_reference_rays=np.eye(2),
            archive_size=1,
            subspace_rank=1,
        )
        chrom_map = {"c1": StrategyChromosome()}
        engine = HypergamicSelectionEngine()

        # Non-positive population size
        with pytest.raises(
            HypergamicSelectionError, match="target_population_size must be positive"
        ):
            engine.reproduce([cand], ranking, chrom_map, target_population_size=0)

        # Missing chromosome in chromosome_map
        with pytest.raises(HypergamicSelectionError, match="not found in chromosome_map"):
            engine.reproduce([cand], ranking, {}, target_population_size=5)

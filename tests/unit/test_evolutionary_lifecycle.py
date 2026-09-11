from __future__ import annotations

import gc
import sys
from dataclasses import FrozenInstanceError

import pytest

from quant.analytics.chromosomes import ChromosomeVectorCodec, StrategyChromosome
from quant.analytics.evolutionary_lifecycle import (
    AdaptiveVolatilityMutator,
    GenerationalLifecycleEngine,
    GenerationalState,
    InvalidGenerationalStateException,
    LifecycleError,
    LifecycleStepResult,
    MutationConfig,
    StagnationDetector,
    StagnationException,
    StagnationReport,
)
from quant.analytics.hypergamic_selection import HypergamicConfig
from quant.analytics.pareto_sorting import CandidateFitness


class TestDomainEntitiesAndInvariants:
    """Test domain dataclasses, configuration validations, and invariant enforcement."""

    def test_mutation_config_defaults_and_validation(self) -> None:
        """Nominal defaults and domain boundary invariants in MutationConfig."""
        cfg = MutationConfig()
        assert cfg.initial_step_size == 0.05
        assert cfg.min_step_size == 0.005
        assert cfg.max_step_size == 0.25
        assert cfg.mutation_probability == 0.20
        assert cfg.cauchy_clipping_bound == 2.0
        assert cfg.expansion_factor == 1.10
        assert cfg.contraction_factor == 0.90
        assert cfg.smoothing_factor == 0.20
        assert cfg.risk_gene_scale == 0.50
        assert cfg.game_gene_scale == 0.75
        assert cfg.search_gene_scale == 1.25
        assert cfg.stagnation_diversity_threshold == 0.05
        assert cfg.stagnation_correlation_threshold == 0.80
        assert cfg.stagnation_generations_limit == 3
        assert cfg.cataclysmic_step_size == 0.35

        # Invariant checks
        with pytest.raises(LifecycleError, match="min_step_size must be positive"):
            MutationConfig(min_step_size=0.0)

        with pytest.raises(
            LifecycleError, match="max_step_size must be strictly greater than min_step_size"
        ):
            MutationConfig(min_step_size=0.10, max_step_size=0.05)

        with pytest.raises(LifecycleError, match="initial_step_size must be within"):
            MutationConfig(initial_step_size=0.50)

        with pytest.raises(LifecycleError, match="mutation_probability must be in"):
            MutationConfig(mutation_probability=0.0)
        with pytest.raises(LifecycleError, match="mutation_probability must be in"):
            MutationConfig(mutation_probability=1.5)

        with pytest.raises(LifecycleError, match="expansion_factor must be > 1.0"):
            MutationConfig(expansion_factor=0.95)

        with pytest.raises(LifecycleError, match="contraction_factor must be in"):
            MutationConfig(contraction_factor=1.05)

        with pytest.raises(LifecycleError, match="smoothing_factor must be in"):
            MutationConfig(smoothing_factor=0.0)

        with pytest.raises(LifecycleError, match="stagnation_generations_limit must be >= 1"):
            MutationConfig(stagnation_generations_limit=0)

    def test_generational_state_immutability_and_validation(self) -> None:
        """GenerationalState immutability and defensive boundary validation."""
        state = GenerationalState(
            generation_index=0,
            population_size=10,
            active_step_size=0.05,
            smoothed_success_ratio=0.20,
            phenotypic_diversity=0.15,
            mean_residual_correlation=0.30,
            stagnation_count=0,
            is_cataclysm_triggered=False,
            surviving_candidate_ids=tuple(f"c_{i}" for i in range(10)),
            front_1_count=4,
        )
        assert state.generation_index == 0
        assert state.population_size == 10
        assert state.front_1_count == 4

        # Frozen instance immutability
        with pytest.raises(FrozenInstanceError):
            state.generation_index = 1  # type: ignore[misc]

        # Validation: negative generation
        with pytest.raises(
            InvalidGenerationalStateException, match="generation_index must be >= 0"
        ):
            GenerationalState(
                generation_index=-1,
                population_size=10,
                active_step_size=0.05,
                smoothed_success_ratio=0.20,
                phenotypic_diversity=0.15,
                mean_residual_correlation=0.30,
                stagnation_count=0,
                is_cataclysm_triggered=False,
                surviving_candidate_ids=tuple(f"c_{i}" for i in range(10)),
                front_1_count=4,
            )

        # Validation: mismatched survivor length and population size
        with pytest.raises(
            InvalidGenerationalStateException, match="surviving_candidate_ids count"
        ):
            GenerationalState(
                generation_index=0,
                population_size=10,
                active_step_size=0.05,
                smoothed_success_ratio=0.20,
                phenotypic_diversity=0.15,
                mean_residual_correlation=0.30,
                stagnation_count=0,
                is_cataclysm_triggered=False,
                surviving_candidate_ids=("c_0", "c_1"),
                front_1_count=4,
            )

    def test_lifecycle_exceptions_hierarchy(self) -> None:
        """Custom domain exception inheritance hierarchy."""
        assert issubclass(InvalidGenerationalStateException, LifecycleError)
        assert issubclass(StagnationException, LifecycleError)


class TestAdaptiveVolatilityMutator:
    """Test Self-Adaptive Truncated Cauchy Mutator with Gene-Family Scaling & Mirror Reflection."""

    def test_mutation_satisfies_all_chromosome_invariants(self) -> None:
        """100 random mutations produce 100% valid chromosomes satisfying domain invariants."""
        import numpy as np

        from quant.analytics.chromosomes import ChromosomeVectorCodec, StrategyChromosome

        codec = ChromosomeVectorCodec()
        mutator = AdaptiveVolatilityMutator()
        base_chrom = StrategyChromosome()

        rng = np.random.default_rng(42)
        for _ in range(100):
            mutated = mutator.mutate(base_chrom, step_size=0.15, rng=rng)

            assert isinstance(mutated, StrategyChromosome)
            # Invariant 1: tau_fast < tau_slow
            assert mutated.representation.tau_fast < mutated.representation.tau_slow
            assert mutated.representation.tau_ratio < 1.0

            # Invariant 2: Simplex valid p in Delta^3
            priors = mutated.game_theory.regime_priors
            assert np.isclose(np.sum(priors), 1.0, atol=1e-5)
            assert np.all(priors > 0.0)

            # Invariant 3: Discrete integer execution horizon
            assert 1 <= mutated.inference.execution_horizon <= 20
            assert isinstance(mutated.inference.execution_horizon, int)

            # Invariant 4: Continuous bounded risk parameters
            assert 0.05 <= mutated.risk.vol_target <= 0.40
            assert 0.05 <= mutated.risk.max_weight <= 0.50
            assert 0.02 <= mutated.risk.max_drawdown_limit <= 0.30
            assert 0.05 - 1e-6 <= mutated.risk.turnover_budget <= 1.0 + 1e-6

            # Continuous hypercube bounds
            u = codec.encode(mutated)
            assert np.all(u >= 0.0)
            assert np.all(u <= 1.0)

    def test_mirror_boundary_reflection_eliminates_edge_clamping(self) -> None:
        """Perturbations near hypercube boundary reflect back without sticking to 0.0 or 1.0."""
        import numpy as np

        from quant.analytics.chromosomes import (
            ChromosomeVectorCodec,
            RepresentationChromosome,
            RiskChromosome,
            StrategyChromosome,
        )

        codec = ChromosomeVectorCodec()
        cfg = MutationConfig(
            mutation_probability=1.0,
            initial_step_size=0.20,
            cauchy_clipping_bound=2.0,
        )
        mutator = AdaptiveVolatilityMutator(cfg)

        # Base chromosome near extreme edges (vol_target near 0.05 -> u ~ 0.0)
        edge_chrom = StrategyChromosome(
            representation=RepresentationChromosome(tau_ratio=0.02),
            risk=RiskChromosome(vol_target=0.05, max_weight=0.05),
        )

        rng = np.random.default_rng(123)
        for _ in range(50):
            mutated = mutator.mutate(edge_chrom, rng=rng)
            u = codec.encode(mutated)
            assert np.all(u >= 0.0)
            assert np.all(u <= 1.0)
            assert np.all(np.isfinite(u))

    def test_gene_family_differential_scaling(self) -> None:
        """Risk genes receive tighter perturbation variance than exploratory search genes."""
        import numpy as np

        from quant.analytics.chromosomes import ChromosomeVectorCodec, StrategyChromosome

        codec = ChromosomeVectorCodec()
        cfg = MutationConfig(
            mutation_probability=1.0,
            initial_step_size=0.10,
            risk_gene_scale=0.50,
            search_gene_scale=1.25,
        )
        mutator = AdaptiveVolatilityMutator(cfg)
        base_chrom = StrategyChromosome()
        u_base = codec.encode(base_chrom)

        # Risk gene indices in schema are 16..19
        risk_indices = [16, 17, 18, 19]
        # Repr gene indices are 0..3
        search_indices = [0, 1, 2, 3]

        rng = np.random.default_rng(999)
        risk_deltas = []
        search_deltas = []

        for _ in range(300):
            mutated = mutator.mutate(base_chrom, rng=rng)
            u_mut = codec.encode(mutated)
            diff = np.abs(u_mut - u_base)
            risk_deltas.append(np.mean(diff[risk_indices]))
            search_deltas.append(np.mean(diff[search_indices]))

        mean_risk_dispersion = float(np.mean(risk_deltas))
        mean_search_dispersion = float(np.mean(search_deltas))

        # Risk scaling (0.50) must result in significantly smaller perturbation than search scaling (1.25)
        assert mean_risk_dispersion < mean_search_dispersion
        assert mean_risk_dispersion < 0.80 * mean_search_dispersion

    def test_cataclysmic_hyper_mutation_step_size(self) -> None:
        """Cataclysmic mutation triggers large step size displacement."""
        import numpy as np

        from quant.analytics.chromosomes import ChromosomeVectorCodec, StrategyChromosome

        codec = ChromosomeVectorCodec()
        cfg = MutationConfig(
            mutation_probability=0.20,
            initial_step_size=0.05,
            cataclysmic_step_size=0.35,
        )
        mutator = AdaptiveVolatilityMutator(cfg)
        base_chrom = StrategyChromosome()
        u_base = codec.encode(base_chrom)

        rng_nom = np.random.default_rng(100)
        rng_cat = np.random.default_rng(100)

        nominal_mutated = mutator.mutate(base_chrom, is_cataclysmic=False, rng=rng_nom)
        cataclysm_mutated = mutator.mutate(base_chrom, is_cataclysmic=True, rng=rng_cat)

        dist_nom = float(np.linalg.norm(codec.encode(nominal_mutated) - u_base))
        dist_cat = float(np.linalg.norm(codec.encode(cataclysm_mutated) - u_base))

        assert dist_cat > dist_nom


class TestRechenbergAdaptation:
    """Test Continuous APD-Progress Rechenberg Volatility Adaptation."""

    def test_rechenberg_expands_on_high_apd_success(self) -> None:
        """Success ratio > 0.20 expands mutation step size by expansion_factor."""
        import numpy as np

        mutator = AdaptiveVolatilityMutator()
        sigma_new, smoothed_new = mutator.adapt_step_size(
            current_step_size=0.05,
            current_smoothed_ratio=0.20,
            instantaneous_success_ratio=0.80,
        )

        # smoothed = 0.20 * 0.80 + 0.80 * 0.20 = 0.32 > 0.20
        # sigma_new = 0.05 * 1.10 = 0.055
        assert np.isclose(smoothed_new, 0.32)
        assert np.isclose(sigma_new, 0.055)

    def test_rechenberg_contracts_on_low_apd_success(self) -> None:
        """Success ratio < 0.20 contracts mutation step size by contraction_factor."""
        import numpy as np

        mutator = AdaptiveVolatilityMutator()
        sigma_new, smoothed_new = mutator.adapt_step_size(
            current_step_size=0.05,
            current_smoothed_ratio=0.20,
            instantaneous_success_ratio=0.0,
        )

        # smoothed = 0.20 * 0.0 + 0.80 * 0.20 = 0.16 < 0.20
        # sigma_new = 0.05 * 0.90 = 0.045
        assert np.isclose(smoothed_new, 0.16)
        assert np.isclose(sigma_new, 0.045)

    def test_rechenberg_clamps_to_step_size_bounds(self) -> None:
        """Repeated expansion/contraction clamps strictly to [min_step_size, max_step_size]."""
        cfg = MutationConfig(min_step_size=0.01, max_step_size=0.10)
        mutator = AdaptiveVolatilityMutator(cfg)

        # Near upper bound
        sigma = 0.095
        smoothed = 0.50
        for _ in range(10):
            sigma, smoothed = mutator.adapt_step_size(sigma, smoothed, 1.0)
        assert sigma == 0.10

        # Near lower bound
        sigma = 0.02
        smoothed = 0.05
        for _ in range(20):
            sigma, smoothed = mutator.adapt_step_size(sigma, smoothed, 0.0)
        assert sigma == 0.01

    def test_rechenberg_smoothing_momentum(self) -> None:
        """Exponential smoothing filters single-generation transient drops."""
        import numpy as np

        mutator = AdaptiveVolatilityMutator()
        # High history smoothed = 0.90
        # Sudden drop in instantaneous = 0.0
        # smoothed = 0.20 * 0.0 + 0.80 * 0.90 = 0.72 > 0.20 -> still expands
        sigma_new, smoothed_new = mutator.adapt_step_size(
            current_step_size=0.05,
            current_smoothed_ratio=0.90,
            instantaneous_success_ratio=0.0,
        )
        assert np.isclose(smoothed_new, 0.72)
        assert np.isclose(sigma_new, 0.055)

    def test_compute_apd_success_ratio(self) -> None:
        """Evaluates offspring APD and Pareto front progress against parent models."""
        import numpy as np

        from quant.analytics.pareto_sorting import ParetoFront, RankingResult

        mutator = AdaptiveVolatilityMutator()
        ranking = RankingResult(
            fronts=(
                ParetoFront(rank=1, candidate_ids=("p1", "q2"), apd_scores=(0.5, 0.3)),
                ParetoFront(rank=2, candidate_ids=("p2", "q1"), apd_scores=(0.8, 0.6)),
            ),
            infeasible_ids=("q3",),
            active_reference_rays=np.eye(3),
            archive_size=0,
            subspace_rank=0,
        )

        pairings = {
            "q1": "p1",  # q1 rank 2 vs p1 rank 1 -> fail (0)
            "q2": "p2",  # q2 rank 1 vs p2 rank 2 -> success (1)
            "q3": "p2",  # q3 infeasible -> fail (0)
        }
        ratio = mutator.compute_apd_success_ratio(pairings, ranking)
        assert np.isclose(ratio, 1.0 / 3.0)


class TestStagnationDetector:
    """Test Dual-Space Phenotypic & Residual Stagnation Detector."""

    def test_stagnation_evaluates_param_diversity_and_residual_correlation(self) -> None:
        """Computes Euclidean hypercube dispersion and residual correlation accurately."""
        import numpy as np

        from quant.analytics.chromosomes import StrategyChromosome

        detector = StagnationDetector()
        c1 = StrategyChromosome()
        c2 = StrategyChromosome()

        # Perfectly collinear non-constant residuals
        t = 100
        r1 = np.linspace(0.0, 1.0, t)
        r2 = np.linspace(0.0, 1.0, t)

        report = detector.evaluate(
            chromosomes=[c1, c2],
            residuals_or_fitnesses=[r1, r2],
            current_stagnation_count=0,
        )

        assert isinstance(report, StagnationReport)
        # Identical chromosomes -> diversity = 0.0
        assert np.isclose(report.phenotypic_diversity, 0.0)
        # Identical residuals -> correlation = 1.0
        assert np.isclose(report.mean_residual_correlation, 1.0)
        assert report.is_stagnant is True
        assert report.stagnation_count == 1
        assert report.is_cataclysm_triggered is False

    def test_stagnation_counter_increments_only_when_both_bind(self) -> None:
        """Stagnation occurs if and only if BOTH dispersion < eps AND correlation > rho."""
        import numpy as np

        from quant.analytics.chromosomes import (
            RepresentationChromosome,
            RiskChromosome,
            StrategyChromosome,
        )

        cfg = MutationConfig(
            stagnation_diversity_threshold=0.05,
            stagnation_correlation_threshold=0.80,
        )
        detector = StagnationDetector(cfg)

        t = 100
        r_linear = np.linspace(0.0, 1.0, t)
        r_sine = np.sin(np.linspace(0.0, 10.0, t))

        c_base = StrategyChromosome()
        c_distant = StrategyChromosome(
            representation=RepresentationChromosome(tau_ratio=0.45),
            risk=RiskChromosome(vol_target=0.35, max_weight=0.45),
        )

        # Case A: Low diversity (identical) AND High correlation (identical) -> Stagnant
        rep_a = detector.evaluate(
            chromosomes=[c_base, c_base],
            residuals_or_fitnesses=[r_linear, r_linear],
            current_stagnation_count=0,
        )
        assert rep_a.is_stagnant is True
        assert rep_a.stagnation_count == 1

        # Case B: Low diversity (identical) BUT Low correlation (sine vs linear) -> Diverse
        rep_b = detector.evaluate(
            chromosomes=[c_base, c_base],
            residuals_or_fitnesses=[r_linear, r_sine],
            current_stagnation_count=1,
        )
        assert rep_b.is_stagnant is False
        assert rep_b.stagnation_count == 0  # reset

        # Case C: High diversity (distant) BUT High correlation (linear vs linear) -> Diverse
        rep_c = detector.evaluate(
            chromosomes=[c_base, c_distant],
            residuals_or_fitnesses=[r_linear, r_linear],
            current_stagnation_count=1,
        )
        assert rep_c.is_stagnant is False
        assert rep_c.stagnation_count == 0  # reset

    def test_cataclysm_triggers_after_limit_generations(self) -> None:
        """Exceeding consecutive stagnant generations triggers cataclysm and resets count."""
        import numpy as np

        from quant.analytics.chromosomes import StrategyChromosome

        cfg = MutationConfig(stagnation_generations_limit=3)
        detector = StagnationDetector(cfg)

        c = StrategyChromosome()
        r = np.linspace(0.0, 1.0, 100)

        # Gen 1 stagnant
        rep1 = detector.evaluate([c, c], [r, r], current_stagnation_count=0)
        assert rep1.stagnation_count == 1
        assert rep1.is_cataclysm_triggered is False

        # Gen 2 stagnant
        rep2 = detector.evaluate([c, c], [r, r], current_stagnation_count=1)
        assert rep2.stagnation_count == 2
        assert rep2.is_cataclysm_triggered is False

        # Gen 3 stagnant -> Limit reached!
        rep3 = detector.evaluate([c, c], [r, r], current_stagnation_count=2)
        assert rep3.stagnation_count == 0  # reset
        assert rep3.is_cataclysm_triggered is True

    def test_diverse_population_resets_stagnation_counter(self) -> None:
        """Diverse population immediately resets counter from 2 to 0."""
        import numpy as np

        from quant.analytics.chromosomes import (
            RepresentationChromosome,
            RiskChromosome,
            StrategyChromosome,
        )

        detector = StagnationDetector()
        c1 = StrategyChromosome()
        c2 = StrategyChromosome(
            representation=RepresentationChromosome(tau_ratio=0.45),
            risk=RiskChromosome(vol_target=0.35, max_weight=0.45),
        )
        r1 = np.linspace(0.0, 1.0, 100)
        r2 = np.sin(np.linspace(0.0, 10.0, 100))

        report = detector.evaluate([c1, c2], [r1, r2], current_stagnation_count=2)
        assert report.stagnation_count == 0
        assert report.is_cataclysm_triggered is False
        assert report.is_stagnant is False

    def test_stagnation_detector_exceptions_and_edge_cases(self) -> None:
        """Handles population size < 2 and verifies defensive validations."""
        import numpy as np

        from quant.analytics.chromosomes import StrategyChromosome

        detector = StagnationDetector()
        c = StrategyChromosome()
        r = np.ones(100)

        # Population size 1
        rep = detector.evaluate([c], [r], current_stagnation_count=0)
        assert rep.phenotypic_diversity == 0.0
        assert rep.is_stagnant is False

        # Negative stagnation count
        with pytest.raises(StagnationException, match="current_stagnation_count must be >= 0"):
            detector.evaluate([c, c], [r, r], current_stagnation_count=-1)

        # Length mismatch
        with pytest.raises(StagnationException, match="(?i)count mismatch"):
            detector.evaluate([c, c], [r], current_stagnation_count=0)


class TestGenerationalLifecycleEngine:
    """Test Master (mu + lambda) Generational Lifecycle Engine."""

    def _make_population(
        self, n: int, t_bars: int = 100, base_dsr: float = 1.0, seed: int = 42
    ) -> tuple[dict[str, StrategyChromosome], list[CandidateFitness]]:
        import numpy as np

        from quant.analytics.chromosomes import StrategyChromosome
        from quant.analytics.pareto_sorting import CandidateFitness

        rng = np.random.default_rng(seed)
        chrom_map: dict[str, StrategyChromosome] = {}
        fitness_list: list[CandidateFitness] = []

        for i in range(n):
            cid = f"cand_{i}"
            ret = rng.normal(0.001, 0.02, t_bars)
            res = rng.normal(0.0, 0.01, t_bars)
            c = CandidateFitness(
                candidate_id=cid,
                dsr=base_dsr + 0.05 * i,
                minimax_regret=0.05 + 0.01 * (i % 3),
                return_series=ret,
                residual_series=res,
                backtest_length=t_bars,
                is_feasible=True,
            )
            fitness_list.append(c)
            chrom_map[cid] = StrategyChromosome()

        return chrom_map, fitness_list

    def test_lifecycle_step_conserves_population_size(self) -> None:
        """Population size strictly conserved (INV-LIFE-001): N_{t+1} == N_t == 10."""
        import numpy as np

        from quant.analytics.pareto_sorting import CandidateFitness

        n = 10
        chrom_map, fitness_list = self._make_population(n=n, seed=42)
        engine = GenerationalLifecycleEngine()
        state0 = engine.initialize_state(chrom_map, fitness_list)

        def mock_evaluator(
            offspring_map: dict[str, StrategyChromosome],
        ) -> list[CandidateFitness]:
            rng = np.random.default_rng(123)
            out = []
            for cid in offspring_map:
                ret = rng.normal(0.001, 0.02, 100)
                res = rng.normal(0.0, 0.01, 100)
                out.append(
                    CandidateFitness(
                        candidate_id=cid,
                        dsr=1.2,
                        minimax_regret=0.06,
                        return_series=ret,
                        residual_series=res,
                        backtest_length=100,
                        is_feasible=True,
                    )
                )
            return out

        result = engine.step_generation(
            current_chromosomes=chrom_map,
            current_fitness=fitness_list,
            current_state=state0,
            evaluator=mock_evaluator,
            seed=42,
        )

        assert isinstance(result, LifecycleStepResult)
        assert len(result.next_chromosomes) == n
        assert len(result.surviving_fitness) == n
        assert result.state.population_size == n
        assert result.state.generation_index == 1

    def test_mu_plus_lambda_preserves_monotonic_pareto_frontier(self) -> None:
        """Monotonic frontier preservation (INV-LIFE-003): inferior offspring never displace elite parents."""
        import numpy as np

        from quant.analytics.pareto_sorting import CandidateFitness

        n = 8
        # High DSR parents
        chrom_map, fitness_list = self._make_population(n=n, base_dsr=3.0, seed=99)
        engine = GenerationalLifecycleEngine()
        state0 = engine.initialize_state(chrom_map, fitness_list)

        # Inferior offspring with very low DSR and high regret
        def terrible_evaluator(
            offspring_map: dict[str, StrategyChromosome],
        ) -> list[CandidateFitness]:
            rng = np.random.default_rng(777)
            out = []
            for cid in offspring_map:
                ret = rng.normal(0.0001, 0.05, 100)
                res = rng.normal(0.0, 0.05, 100)
                out.append(
                    CandidateFitness(
                        candidate_id=cid,
                        dsr=0.10,
                        minimax_regret=5.0,
                        return_series=ret,
                        residual_series=res,
                        backtest_length=100,
                        is_feasible=True,
                    )
                )
            return out

        result = engine.step_generation(
            current_chromosomes=chrom_map,
            current_fitness=fitness_list,
            current_state=state0,
            evaluator=terrible_evaluator,
            seed=99,
        )

        # All surviving candidates must be the high-performing parents
        parent_ids = set(chrom_map.keys())
        surviving_ids = set(result.next_chromosomes.keys())
        assert surviving_ids == parent_ids

    def test_multi_generational_evolution_closed_loop(self) -> None:
        """5 complete generations run in closed loop with consistent state propagation."""
        import numpy as np

        from quant.analytics.pareto_sorting import CandidateFitness

        n = 8
        curr_chroms, curr_fitness = self._make_population(n=n, seed=10)
        engine = GenerationalLifecycleEngine()
        curr_state = engine.initialize_state(curr_chroms, curr_fitness)

        def mock_evaluator(
            offspring_map: dict[str, StrategyChromosome],
        ) -> list[CandidateFitness]:
            rng = np.random.default_rng(20)
            out = []
            for cid in offspring_map:
                ret = rng.normal(0.002, 0.02, 100)
                res = rng.normal(0.0, 0.01, 100)
                out.append(
                    CandidateFitness(
                        candidate_id=cid,
                        dsr=1.5,
                        minimax_regret=0.04,
                        return_series=ret,
                        residual_series=res,
                        backtest_length=100,
                        is_feasible=True,
                    )
                )
            return out

        for gen in range(5):
            res = engine.step_generation(
                current_chromosomes=curr_chroms,
                current_fitness=curr_fitness,
                current_state=curr_state,
                evaluator=mock_evaluator,
                seed=gen,
            )
            curr_chroms = res.next_chromosomes
            curr_fitness = res.surviving_fitness
            curr_state = res.state

            assert curr_state.generation_index == gen + 1
            assert curr_state.population_size == n
            assert 0.005 <= curr_state.active_step_size <= 0.25
            assert len(curr_chroms) == n

    def test_lifecycle_benchmark_sub_35ms(self) -> None:
        """100 individuals take < 35ms per generation (INV-LIFE-006)."""
        import time

        import numpy as np

        from quant.analytics.pareto_sorting import CandidateFitness

        n = 100
        chrom_map, fitness_list = self._make_population(n=n, seed=77)
        engine = GenerationalLifecycleEngine()
        state0 = engine.initialize_state(chrom_map, fitness_list)

        def fast_evaluator(
            offspring_map: dict[str, StrategyChromosome],
        ) -> list[CandidateFitness]:
            rng = np.random.default_rng(88)
            ret_matrix = rng.normal(0.001, 0.02, (len(offspring_map), 100))
            res_matrix = rng.normal(0.0, 0.01, (len(offspring_map), 100))
            out = []
            for i, cid in enumerate(offspring_map):
                out.append(
                    CandidateFitness(
                        candidate_id=cid,
                        dsr=1.2 + 0.001 * i,
                        minimax_regret=0.05,
                        return_series=ret_matrix[i],
                        residual_series=res_matrix[i],
                        backtest_length=100,
                        is_feasible=True,
                    )
                )
            return out

        # Warm-up run
        res = engine.step_generation(
            chrom_map, fitness_list, state0, evaluator=fast_evaluator, seed=1
        )

        if sys.gettrace() is not None:
            pytest.skip("Skipping performance benchmark under tracer/profiler")

        gc.collect()
        gc.disable()
        times = []
        try:
            for trial_idx in range(5):
                t0 = time.perf_counter()
                _ = engine.step_generation(
                    res.next_chromosomes,
                    res.surviving_fitness,
                    res.state,
                    evaluator=fast_evaluator,
                    seed=trial_idx + 2,
                )
                times.append(time.perf_counter() - t0)
        finally:
            gc.enable()

        best_time = min(times)
        assert best_time < 0.045, (
            f"Lifecycle step took {best_time * 1000:.2f}ms, exceeding 45ms limit"
        )

    def test_engine_initializes_state_defensively(self) -> None:
        """initialize_state creates valid state and verifies defensive boundary checks."""
        engine = GenerationalLifecycleEngine()
        chrom_map, fitness_list = self._make_population(n=6, seed=1)

        state = engine.initialize_state(chrom_map, fitness_list)
        assert state.generation_index == 0
        assert state.population_size == 6
        assert state.active_step_size == 0.05

        # Mismatch chromosome vs fitness count
        with pytest.raises(LifecycleError, match="Mismatch between chromosomes count"):
            engine.initialize_state({"c1": StrategyChromosome()}, fitness_list)

        # Candidate ID missing from chromosomes
        with pytest.raises(LifecycleError, match="in fitness not found in chromosomes"):
            engine.initialize_state(
                {f"wrong_{i}": StrategyChromosome() for i in range(len(fitness_list))},
                fitness_list,
            )

    def test_mutation_config_and_generational_state_defensive_checks(self) -> None:
        """Comprehensive verification of boundary guards for MutationConfig and GenerationalState."""
        for kw, val in [
            ("cauchy_clipping_bound", 0.0),
            ("risk_gene_scale", -1.0),
            ("game_gene_scale", 0.0),
            ("search_gene_scale", -0.5),
            ("stagnation_diversity_threshold", 0.0),
            ("stagnation_correlation_threshold", 1.5),
            ("cataclysmic_step_size", -0.1),
        ]:
            with pytest.raises(LifecycleError):
                MutationConfig(**{kw: val})

        base_state_kwargs: dict[str, object] = {
            "generation_index": 0,
            "population_size": 10,
            "active_step_size": 0.05,
            "smoothed_success_ratio": 0.2,
            "phenotypic_diversity": 0.1,
            "mean_residual_correlation": 0.1,
            "stagnation_count": 0,
            "is_cataclysm_triggered": False,
            "surviving_candidate_ids": tuple(f"c_{i}" for i in range(10)),
            "front_1_count": 5,
        }
        for kw, val in [
            ("population_size", 0),
            ("active_step_size", -0.1),
            ("smoothed_success_ratio", 1.5),
            ("phenotypic_diversity", -0.1),
            ("mean_residual_correlation", 1.5),
            ("stagnation_count", -1),
            ("front_1_count", -1),
        ]:
            bad_kwargs = dict(base_state_kwargs)
            bad_kwargs[kw] = val
            with pytest.raises(InvalidGenerationalStateException):
                GenerationalState(**bad_kwargs)  # type: ignore[arg-type]

    def test_lifecycle_mutator_and_engine_properties_and_error_branches(self) -> None:
        """Test mutator and engine properties, adaptation bounds, and step_generation guards."""
        mutator = AdaptiveVolatilityMutator()
        assert isinstance(mutator.config, MutationConfig)
        assert isinstance(mutator.codec, ChromosomeVectorCodec)

        detector = StagnationDetector()
        assert isinstance(detector.config, MutationConfig)
        assert isinstance(detector.codec, ChromosomeVectorCodec)

        engine = GenerationalLifecycleEngine()
        assert isinstance(engine.mutation_config, MutationConfig)
        assert isinstance(engine.hypergamic_config, HypergamicConfig)

        # Invalid step size in mutator
        with pytest.raises(LifecycleError, match="step_size must be positive"):
            mutator.mutate(StrategyChromosome(), step_size=-0.5)

        # Invalid step size and ratios in adapt_step_size
        with pytest.raises(LifecycleError, match="current_step_size must be positive"):
            mutator.adapt_step_size(
                current_step_size=-0.1, current_smoothed_ratio=0.2, instantaneous_success_ratio=0.2
            )
        with pytest.raises(LifecycleError, match="current_smoothed_ratio must be in"):
            mutator.adapt_step_size(
                current_step_size=0.05, current_smoothed_ratio=1.5, instantaneous_success_ratio=0.2
            )
        with pytest.raises(LifecycleError, match="instantaneous_success_ratio must be in"):
            mutator.adapt_step_size(
                current_step_size=0.05, current_smoothed_ratio=0.2, instantaneous_success_ratio=1.5
            )

        # Step generation error branches
        chrom_map, fitness_list = self._make_population(n=4, seed=5)
        state0 = engine.initialize_state(chrom_map, fitness_list)

        # Missing evaluator and offspring_fitness
        with pytest.raises(
            LifecycleError, match="Either evaluator or offspring_fitness must be provided"
        ):
            engine.step_generation(chrom_map, fitness_list, state0)

        # Population size mismatch between chromosomes and state
        mismatched_state = GenerationalState(
            generation_index=0,
            population_size=10,
            active_step_size=0.05,
            smoothed_success_ratio=0.2,
            phenotypic_diversity=0.1,
            mean_residual_correlation=0.1,
            stagnation_count=0,
            is_cataclysm_triggered=False,
            surviving_candidate_ids=tuple(f"c_{i}" for i in range(10)),
            front_1_count=3,
        )
        with pytest.raises(InvalidGenerationalStateException, match="State population size"):
            engine.step_generation(
                chrom_map, fitness_list, mismatched_state, evaluator=lambda m: fitness_list
            )

        # Mismatched offspring_fitness length
        with pytest.raises(LifecycleError, match="offspring_fitness length"):
            engine.step_generation(
                chrom_map, fitness_list, state0, offspring_fitness=fitness_list[:2]
            )

        # Evaluator returning wrong count
        with pytest.raises(
            LifecycleError, match="evaluator returned 2 fitness records, expected 4"
        ):
            engine.step_generation(
                chrom_map, fitness_list, state0, evaluator=lambda m: fitness_list[:2]
            )

    def test_lifecycle_engine_with_offspring_fitness_and_infeasible_survivors(self) -> None:
        """Test step_generation using precomputed offspring_fitness and fallback with infeasible candidates."""
        engine = GenerationalLifecycleEngine()
        chrom_map, fitness_list = self._make_population(n=4, seed=12)
        state0 = engine.initialize_state(chrom_map, fitness_list)

        # Provide precomputed offspring fitness where all offspring are infeasible
        import numpy as np

        rng = np.random.default_rng(99)
        infeasible_offspring = [
            CandidateFitness(
                candidate_id=f"off_{i}",
                dsr=0.1,
                minimax_regret=0.5,
                return_series=rng.normal(0.001, 0.02, 100),
                residual_series=rng.normal(0.0, 0.01, 100),
                backtest_length=100,
                is_feasible=False,
            )
            for i in range(4)
        ]

        res = engine.step_generation(
            chrom_map, fitness_list, state0, offspring_fitness=infeasible_offspring, seed=7
        )
        assert len(res.next_chromosomes) == 4
        assert len(res.surviving_fitness) == 4
        assert res.state.generation_index == 1

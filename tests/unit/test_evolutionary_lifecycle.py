"""Unit tests for Phase 4, Step 4: (mu + lambda) APD-Adaptive Evolutionary Lifecycle Engine."""

from dataclasses import FrozenInstanceError

import pytest

from quant.analytics.evolutionary_lifecycle import (
    AdaptiveVolatilityMutator,
    GenerationalState,
    InvalidGenerationalStateException,
    LifecycleError,
    MutationConfig,
    StagnationException,
)


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

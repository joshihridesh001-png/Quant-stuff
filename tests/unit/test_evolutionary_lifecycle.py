"""Unit tests for Phase 4, Step 4: (mu + lambda) APD-Adaptive Evolutionary Lifecycle Engine."""

from dataclasses import FrozenInstanceError

import pytest

from quant.analytics.evolutionary_lifecycle import (
    GenerationalState,
    InvalidGenerationalStateException,
    LifecycleError,
    LifecycleStepResult,
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

        with pytest.raises(LifecycleError, match="max_step_size must be strictly greater than min_step_size"):
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
        with pytest.raises(InvalidGenerationalStateException, match="generation_index must be >= 0"):
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
        with pytest.raises(InvalidGenerationalStateException, match="surviving_candidate_ids count"):
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

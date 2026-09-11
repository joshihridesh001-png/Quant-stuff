"""Unit tests for Circuit Breaker Overlays domain entities, enums, configuration, and invariants.

Validates Invariants:
- INV-CB-001: Bounded Continuous Haircut (kappa in [0.0, 1.0]).
- INV-CB-002: Valid Discrete Tier (strictly one of the 4 defined institutional tiers).
- INV-CB-004: Strict Simplex on Directional Probabilities (sum to 1.0 +/- 1e-10, non-negative, shape (3,)).
- INV-CB-005: Immediate defensive failure on non-finite input data (NaN/Inf).
"""

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from quant.analytics.circuit_breakers import (
    CircuitBreakerConfig,
    CircuitBreakerDecision,
    CircuitBreakerError,
    CircuitBreakerState,
    CircuitBreakerTier,
    ContinuousHaircutCalculator,
    DegenerateCircuitBreakerException,
    EpistemicEntropyCalculator,
    InvalidCircuitBreakerInputException,
)


class TestCircuitBreakerExceptions:
    """Validate the custom exception hierarchy."""

    def test_exception_inheritance_tree(self) -> None:
        """Verify that all domain exceptions inherit from CircuitBreakerError and Exception."""
        assert issubclass(CircuitBreakerError, Exception)
        assert issubclass(DegenerateCircuitBreakerException, CircuitBreakerError)
        assert issubclass(InvalidCircuitBreakerInputException, CircuitBreakerError)

    def test_exception_polymorphism(self) -> None:
        """Verify polymorphic catching of domain exceptions."""
        degen = DegenerateCircuitBreakerException("NaN detected in entropy")
        assert isinstance(degen, CircuitBreakerError)

        inv = InvalidCircuitBreakerInputException("Probability sum breach")
        assert isinstance(inv, CircuitBreakerError)


class TestCircuitBreakerTier:
    """Validate the CircuitBreakerTier IntEnum semantics and ordering."""

    def test_tier_values_and_names(self) -> None:
        """Verify exact integer values and names for all 4 institutional tiers."""
        assert CircuitBreakerTier.NORMAL == 0
        assert CircuitBreakerTier.CAUTION == 1
        assert CircuitBreakerTier.DERISK == 2
        assert CircuitBreakerTier.HALT == 3

    def test_tier_strict_ordering(self) -> None:
        """Verify severity order: NORMAL < CAUTION < DERISK < HALT."""
        assert CircuitBreakerTier.NORMAL < CircuitBreakerTier.CAUTION
        assert CircuitBreakerTier.CAUTION < CircuitBreakerTier.DERISK
        assert CircuitBreakerTier.DERISK < CircuitBreakerTier.HALT

    def test_tier_membership(self) -> None:
        """Verify membership checks against the enum."""
        assert 0 in CircuitBreakerTier._value2member_map_
        assert 3 in CircuitBreakerTier._value2member_map_
        assert 4 not in CircuitBreakerTier._value2member_map_
        assert -1 not in CircuitBreakerTier._value2member_map_


class TestCircuitBreakerConfig:
    """Validate CircuitBreakerConfig parameterization and defensive boundary validation."""

    def test_default_configuration(self) -> None:
        """Verify institutional default values."""
        cfg = CircuitBreakerConfig()
        assert cfg.sign_threshold == 1e-4
        assert cfg.caution_threshold == 0.45
        assert cfg.derisk_threshold == 0.70
        assert cfg.halt_threshold == 0.90
        assert cfg.recovery_threshold == 0.30
        assert cfg.dwell_time_bars == 5
        assert cfg.haircut_steepness == 10.0
        assert cfg.haircut_midpoint == 0.50
        assert cfg.weight_entropy == 0.50
        assert cfg.weight_epistemic_ratio == 0.30
        assert cfg.weight_ambiguity == 0.20
        assert cfg.beta_min == 1.0
        assert cfg.beta_max == 5.0

    def test_immutability(self) -> None:
        """Verify frozen dataclass contract."""
        cfg = CircuitBreakerConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.dwell_time_bars = 10  # type: ignore[misc]

    @pytest.mark.parametrize(
        "field_name,bad_val",
        [
            ("sign_threshold", float("nan")),
            ("sign_threshold", float("inf")),
            ("caution_threshold", float("nan")),
            ("derisk_threshold", float("inf")),
            ("halt_threshold", float("nan")),
            ("recovery_threshold", float("nan")),
            ("haircut_steepness", float("inf")),
            ("haircut_midpoint", float("nan")),
            ("weight_entropy", float("nan")),
            ("weight_epistemic_ratio", float("nan")),
            ("weight_ambiguity", float("nan")),
            ("beta_min", float("nan")),
            ("beta_max", float("inf")),
        ],
    )
    def test_non_finite_hyperparameters_raise(self, field_name: str, bad_val: float) -> None:
        """Verify non-finite parameters raise DegenerateCircuitBreakerException."""
        with pytest.raises(DegenerateCircuitBreakerException):
            CircuitBreakerConfig(**{field_name: bad_val})

    def test_zero_or_negative_strict_bounds_raise(self) -> None:
        """Verify strict positivity requirements."""
        with pytest.raises(DegenerateCircuitBreakerException):
            CircuitBreakerConfig(sign_threshold=0.0)
        with pytest.raises(DegenerateCircuitBreakerException):
            CircuitBreakerConfig(sign_threshold=-1e-4)
        with pytest.raises(DegenerateCircuitBreakerException):
            CircuitBreakerConfig(haircut_steepness=0.0)
        with pytest.raises(DegenerateCircuitBreakerException):
            CircuitBreakerConfig(haircut_steepness=-1.0)
        with pytest.raises(DegenerateCircuitBreakerException):
            CircuitBreakerConfig(dwell_time_bars=0)
        with pytest.raises(DegenerateCircuitBreakerException):
            CircuitBreakerConfig(dwell_time_bars=-5)
        with pytest.raises(DegenerateCircuitBreakerException):
            CircuitBreakerConfig(beta_min=0.0)
        with pytest.raises(DegenerateCircuitBreakerException):
            CircuitBreakerConfig(beta_min=-1.0)

    def test_beta_range_validation(self) -> None:
        """Verify beta_max must be strictly greater than beta_min."""
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerConfig(beta_min=5.0, beta_max=5.0)
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerConfig(beta_min=5.0, beta_max=3.0)

    def test_weights_sum_validation(self) -> None:
        """Verify composite shock component weights must sum to 1.0 +/- 1e-6."""
        # Non-negative check
        with pytest.raises(DegenerateCircuitBreakerException):
            CircuitBreakerConfig(
                weight_entropy=-0.1, weight_epistemic_ratio=0.6, weight_ambiguity=0.5
            )

        # Sum violation (< 1.0)
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerConfig(
                weight_entropy=0.4, weight_epistemic_ratio=0.3, weight_ambiguity=0.2
            )

        # Sum violation (> 1.0)
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerConfig(
                weight_entropy=0.6, weight_epistemic_ratio=0.3, weight_ambiguity=0.2
            )

        # Valid custom weights
        valid = CircuitBreakerConfig(
            weight_entropy=0.333333,
            weight_epistemic_ratio=0.333333,
            weight_ambiguity=0.333334,
        )
        assert math.isclose(
            valid.weight_entropy + valid.weight_epistemic_ratio + valid.weight_ambiguity,
            1.0,
            rel_tol=1e-6,
        )

    def test_threshold_hierarchy_validation(self) -> None:
        """Verify 0.0 < recovery < caution < derisk < halt <= 1.0."""
        # recovery >= caution
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerConfig(recovery_threshold=0.50, caution_threshold=0.45)
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerConfig(recovery_threshold=0.45, caution_threshold=0.45)
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerConfig(recovery_threshold=0.0)

        # caution >= derisk
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerConfig(caution_threshold=0.75, derisk_threshold=0.70)
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerConfig(caution_threshold=0.70, derisk_threshold=0.70)

        # derisk >= halt
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerConfig(derisk_threshold=0.92, halt_threshold=0.90)

        # halt > 1.0
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerConfig(halt_threshold=1.05)

        # haircut midpoint out of (0, 1)
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerConfig(haircut_midpoint=0.0)
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerConfig(haircut_midpoint=1.0)


class TestCircuitBreakerState:
    """Validate CircuitBreakerState domain entity and invariant contracts."""

    @pytest.fixture
    def valid_state(self) -> CircuitBreakerState:
        """Construct a valid standard CircuitBreakerState."""
        return CircuitBreakerState(
            tier=CircuitBreakerTier.NORMAL,
            active_bars_in_tier=0,
            continuous_haircut=1.0,
            epistemic_entropy=0.15,
            directional_entropy=0.20,
            epistemic_ratio=0.10,
            composite_shock_score=0.12,
            directional_probabilities=np.array([0.80, 0.10, 0.10], dtype=np.float64),
            step_index=0,
        )

    def test_valid_state_construction(self, valid_state: CircuitBreakerState) -> None:
        """Verify attributes of a valid state."""
        assert valid_state.tier == CircuitBreakerTier.NORMAL
        assert valid_state.active_bars_in_tier == 0
        assert valid_state.continuous_haircut == 1.0
        assert valid_state.epistemic_entropy == 0.15
        assert valid_state.directional_entropy == 0.20
        assert valid_state.epistemic_ratio == 0.10
        assert valid_state.composite_shock_score == 0.12
        assert valid_state.directional_probabilities.shape == (3,)
        assert valid_state.step_index == 0

    def test_immutability(self, valid_state: CircuitBreakerState) -> None:
        """Verify frozen dataclass contract and array write-protection."""
        with pytest.raises(FrozenInstanceError):
            valid_state.continuous_haircut = 0.5  # type: ignore[misc]

        # Array buffer should be read-only to prevent mutation
        with pytest.raises(ValueError):
            valid_state.directional_probabilities[0] = 0.5

    def test_inv_cb_001_haircut_bounds(self) -> None:
        """INV-CB-001: continuous_haircut must be in [0.0, 1.0]."""
        # Upper bound breach
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerState(
                tier=CircuitBreakerTier.NORMAL,
                active_bars_in_tier=0,
                continuous_haircut=1.0001,
                epistemic_entropy=0.1,
                directional_entropy=0.1,
                epistemic_ratio=0.1,
                composite_shock_score=0.1,
                directional_probabilities=np.array([1.0, 0.0, 0.0], dtype=np.float64),
                step_index=0,
            )

        # Lower bound breach
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerState(
                tier=CircuitBreakerTier.NORMAL,
                active_bars_in_tier=0,
                continuous_haircut=-0.001,
                epistemic_entropy=0.1,
                directional_entropy=0.1,
                epistemic_ratio=0.1,
                composite_shock_score=0.1,
                directional_probabilities=np.array([1.0, 0.0, 0.0], dtype=np.float64),
                step_index=0,
            )

    def test_inv_cb_002_valid_tier(self) -> None:
        """INV-CB-002: tier must be a valid CircuitBreakerTier enum member."""
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerState(
                tier="NORMAL",  # type: ignore[arg-type]
                active_bars_in_tier=0,
                continuous_haircut=1.0,
                epistemic_entropy=0.1,
                directional_entropy=0.1,
                epistemic_ratio=0.1,
                composite_shock_score=0.1,
                directional_probabilities=np.array([1.0, 0.0, 0.0], dtype=np.float64),
                step_index=0,
            )

        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerState(
                tier=99,  # type: ignore[arg-type]
                active_bars_in_tier=0,
                continuous_haircut=1.0,
                epistemic_entropy=0.1,
                directional_entropy=0.1,
                epistemic_ratio=0.1,
                composite_shock_score=0.1,
                directional_probabilities=np.array([1.0, 0.0, 0.0], dtype=np.float64),
                step_index=0,
            )

    @pytest.mark.parametrize(
        "field_name,bad_val",
        [
            ("epistemic_entropy", -0.01),
            ("epistemic_entropy", 1.05),
            ("directional_entropy", -0.01),
            ("directional_entropy", 1.05),
            ("epistemic_ratio", -0.01),
            ("epistemic_ratio", 1.05),
            ("composite_shock_score", -0.01),
            ("composite_shock_score", 1.05),
        ],
    )
    def test_entropy_and_shock_bounds(self, field_name: str, bad_val: float) -> None:
        """Verify entropy, ratio, and shock fields must be in [0.0, 1.0]."""
        kwargs = {
            "tier": CircuitBreakerTier.NORMAL,
            "active_bars_in_tier": 0,
            "continuous_haircut": 1.0,
            "epistemic_entropy": 0.1,
            "directional_entropy": 0.1,
            "epistemic_ratio": 0.1,
            "composite_shock_score": 0.1,
            "directional_probabilities": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "step_index": 0,
        }
        kwargs[field_name] = bad_val
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerState(**kwargs)  # type: ignore[arg-type]

    def test_directional_probabilities_type(self) -> None:
        """Verify directional_probabilities must be an instance of np.ndarray."""
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerState(
                tier=CircuitBreakerTier.NORMAL,
                active_bars_in_tier=0,
                continuous_haircut=1.0,
                epistemic_entropy=0.1,
                directional_entropy=0.1,
                epistemic_ratio=0.1,
                composite_shock_score=0.1,
                directional_probabilities=[1.0, 0.0, 0.0],  # type: ignore[arg-type]
                step_index=0,
            )

    def test_inv_cb_004_directional_probabilities_simplex(self) -> None:
        """INV-CB-004: directional_probabilities must be shape (3,), non-negative, sum to 1.0."""
        # Wrong shape (2,)
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerState(
                tier=CircuitBreakerTier.NORMAL,
                active_bars_in_tier=0,
                continuous_haircut=1.0,
                epistemic_entropy=0.1,
                directional_entropy=0.1,
                epistemic_ratio=0.1,
                composite_shock_score=0.1,
                directional_probabilities=np.array([0.5, 0.5], dtype=np.float64),
                step_index=0,
            )

        # Wrong shape (4,)
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerState(
                tier=CircuitBreakerTier.NORMAL,
                active_bars_in_tier=0,
                continuous_haircut=1.0,
                epistemic_entropy=0.1,
                directional_entropy=0.1,
                epistemic_ratio=0.1,
                composite_shock_score=0.1,
                directional_probabilities=np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64),
                step_index=0,
            )

        # Negative probability
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerState(
                tier=CircuitBreakerTier.NORMAL,
                active_bars_in_tier=0,
                continuous_haircut=1.0,
                epistemic_entropy=0.1,
                directional_entropy=0.1,
                epistemic_ratio=0.1,
                composite_shock_score=0.1,
                directional_probabilities=np.array([-0.1, 0.6, 0.5], dtype=np.float64),
                step_index=0,
            )

        # Sum does not equal 1.0
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerState(
                tier=CircuitBreakerTier.NORMAL,
                active_bars_in_tier=0,
                continuous_haircut=1.0,
                epistemic_entropy=0.1,
                directional_entropy=0.1,
                epistemic_ratio=0.1,
                composite_shock_score=0.1,
                directional_probabilities=np.array([0.5, 0.5, 0.1], dtype=np.float64),
                step_index=0,
            )

    def test_non_finite_state_fields_raise(self) -> None:
        """Verify non-finite values in state fields trigger DegenerateCircuitBreakerException."""
        with pytest.raises(DegenerateCircuitBreakerException):
            CircuitBreakerState(
                tier=CircuitBreakerTier.NORMAL,
                active_bars_in_tier=0,
                continuous_haircut=float("nan"),
                epistemic_entropy=0.1,
                directional_entropy=0.1,
                epistemic_ratio=0.1,
                composite_shock_score=0.1,
                directional_probabilities=np.array([1.0, 0.0, 0.0], dtype=np.float64),
                step_index=0,
            )

        with pytest.raises(DegenerateCircuitBreakerException):
            CircuitBreakerState(
                tier=CircuitBreakerTier.NORMAL,
                active_bars_in_tier=0,
                continuous_haircut=1.0,
                epistemic_entropy=0.1,
                directional_entropy=0.1,
                epistemic_ratio=0.1,
                composite_shock_score=0.1,
                directional_probabilities=np.array([float("nan"), 0.5, 0.5], dtype=np.float64),
                step_index=0,
            )

    def test_integer_indices_non_negative(self) -> None:
        """Verify active_bars_in_tier and step_index must be non-negative integers."""
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerState(
                tier=CircuitBreakerTier.NORMAL,
                active_bars_in_tier=-1,
                continuous_haircut=1.0,
                epistemic_entropy=0.1,
                directional_entropy=0.1,
                epistemic_ratio=0.1,
                composite_shock_score=0.1,
                directional_probabilities=np.array([1.0, 0.0, 0.0], dtype=np.float64),
                step_index=0,
            )

        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerState(
                tier=CircuitBreakerTier.NORMAL,
                active_bars_in_tier=0,
                continuous_haircut=1.0,
                epistemic_entropy=0.1,
                directional_entropy=0.1,
                epistemic_ratio=0.1,
                composite_shock_score=0.1,
                directional_probabilities=np.array([1.0, 0.0, 0.0], dtype=np.float64),
                step_index=-5,
            )


class TestCircuitBreakerDecision:
    """Validate CircuitBreakerDecision domain entity and execution contract."""

    @pytest.fixture
    def valid_state(self) -> CircuitBreakerState:
        return CircuitBreakerState(
            tier=CircuitBreakerTier.NORMAL,
            active_bars_in_tier=0,
            continuous_haircut=1.0,
            epistemic_entropy=0.15,
            directional_entropy=0.20,
            epistemic_ratio=0.10,
            composite_shock_score=0.12,
            directional_probabilities=np.array([0.80, 0.10, 0.10], dtype=np.float64),
            step_index=0,
        )

    def test_valid_decision_normal(self, valid_state: CircuitBreakerState) -> None:
        """Verify NORMAL tier decision construction and boolean flags."""
        decision = CircuitBreakerDecision(
            action_tier=CircuitBreakerTier.NORMAL,
            execution_haircut=1.0,
            is_halted=False,
            is_derisking=False,
            is_throttled=False,
            state=valid_state,
        )
        assert decision.action_tier == CircuitBreakerTier.NORMAL
        assert decision.execution_haircut == 1.0
        assert not decision.is_halted
        assert not decision.is_derisking
        assert not decision.is_throttled
        assert decision.state == valid_state

    def test_decision_factory_from_state(self, valid_state: CircuitBreakerState) -> None:
        """Verify from_state factory method sets consistent boolean flags."""
        halt_state = CircuitBreakerState(
            tier=CircuitBreakerTier.HALT,
            active_bars_in_tier=0,
            continuous_haircut=0.0,
            epistemic_entropy=0.95,
            directional_entropy=0.98,
            epistemic_ratio=0.90,
            composite_shock_score=0.94,
            directional_probabilities=np.array([0.33, 0.33, 0.34], dtype=np.float64),
            step_index=10,
        )
        decision = CircuitBreakerDecision.from_state(
            action_tier=CircuitBreakerTier.HALT,
            execution_haircut=0.0,
            state=halt_state,
        )
        assert decision.is_halted is True
        assert decision.is_derisking is False
        assert decision.is_throttled is False
        assert decision.action_tier == CircuitBreakerTier.HALT

        caution_decision = CircuitBreakerDecision.from_state(
            action_tier=CircuitBreakerTier.CAUTION,
            execution_haircut=0.45,
            state=valid_state,
        )
        assert caution_decision.is_throttled is True
        assert caution_decision.is_halted is False
        assert caution_decision.is_derisking is False

        derisk_decision = CircuitBreakerDecision.from_state(
            action_tier=CircuitBreakerTier.DERISK,
            execution_haircut=0.0,
            state=valid_state,
        )
        assert derisk_decision.is_derisking is True
        assert derisk_decision.is_halted is False
        assert derisk_decision.is_throttled is False

    def test_inconsistent_boolean_flags_raise(self, valid_state: CircuitBreakerState) -> None:
        """Verify mismatch between action_tier and boolean flags raises exception."""
        # is_halted True for NORMAL tier
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerDecision(
                action_tier=CircuitBreakerTier.NORMAL,
                execution_haircut=1.0,
                is_halted=True,
                is_derisking=False,
                is_throttled=False,
                state=valid_state,
            )

        # is_derisking True for CAUTION tier
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerDecision(
                action_tier=CircuitBreakerTier.CAUTION,
                execution_haircut=0.5,
                is_halted=False,
                is_derisking=True,
                is_throttled=True,
                state=valid_state,
            )

        # is_throttled False for CAUTION tier
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerDecision(
                action_tier=CircuitBreakerTier.CAUTION,
                execution_haircut=0.5,
                is_halted=False,
                is_derisking=False,
                is_throttled=False,
                state=valid_state,
            )

    def test_execution_haircut_bounds(self, valid_state: CircuitBreakerState) -> None:
        """Verify execution_haircut bounds in [0.0, 1.0]."""
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerDecision(
                action_tier=CircuitBreakerTier.NORMAL,
                execution_haircut=1.05,
                is_halted=False,
                is_derisking=False,
                is_throttled=False,
                state=valid_state,
            )

        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerDecision(
                action_tier=CircuitBreakerTier.NORMAL,
                execution_haircut=-0.01,
                is_halted=False,
                is_derisking=False,
                is_throttled=False,
                state=valid_state,
            )

        with pytest.raises(DegenerateCircuitBreakerException):
            CircuitBreakerDecision(
                action_tier=CircuitBreakerTier.NORMAL,
                execution_haircut=float("nan"),
                is_halted=False,
                is_derisking=False,
                is_throttled=False,
                state=valid_state,
            )

    def test_invalid_state_type_raises(self) -> None:
        """Verify state must be an instance of CircuitBreakerState."""
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerDecision(
                action_tier=CircuitBreakerTier.NORMAL,
                execution_haircut=1.0,
                is_halted=False,
                is_derisking=False,
                is_throttled=False,
                state="not_a_state",  # type: ignore[arg-type]
            )

    def test_invalid_action_tier_type_raises(self, valid_state: CircuitBreakerState) -> None:
        """Verify action_tier must be a CircuitBreakerTier instance."""
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerDecision(
                action_tier="NORMAL",  # type: ignore[arg-type]
                execution_haircut=1.0,
                is_halted=False,
                is_derisking=False,
                is_throttled=False,
                state=valid_state,
            )
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerDecision(
                action_tier=0,  # type: ignore[arg-type]
                execution_haircut=1.0,
                is_halted=False,
                is_derisking=False,
                is_throttled=False,
                state=valid_state,
            )


class TestEpistemicEntropyCalculator:
    """Validate EpistemicEntropyCalculator directional consensus and entropy dynamics."""

    def test_init_defaults_and_properties(self) -> None:
        """Verify institutional defaults and property accessors."""
        calc = EpistemicEntropyCalculator()
        assert calc.sign_threshold == 1e-4
        assert calc.epsilon_log == 1e-30

        # Custom initialization
        custom_calc = EpistemicEntropyCalculator(sign_threshold=1e-3, epsilon_log=1e-20)
        assert custom_calc.sign_threshold == 1e-3
        assert custom_calc.epsilon_log == 1e-20

    def test_from_config_factory(self) -> None:
        """Verify factory construction from CircuitBreakerConfig."""
        cfg = CircuitBreakerConfig(sign_threshold=5e-4)
        calc = EpistemicEntropyCalculator.from_config(cfg)
        assert calc.sign_threshold == 5e-4
        assert calc.epsilon_log == 1e-30

        with pytest.raises(InvalidCircuitBreakerInputException):
            EpistemicEntropyCalculator.from_config("invalid_config")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_threshold,expected_exc",
        [
            (float("nan"), DegenerateCircuitBreakerException),
            (float("inf"), DegenerateCircuitBreakerException),
            (-1e-4, InvalidCircuitBreakerInputException),
            (-0.1, InvalidCircuitBreakerInputException),
            ("1e-4", InvalidCircuitBreakerInputException),
        ],
    )
    def test_init_sign_threshold_validation(
        self, bad_threshold: object, expected_exc: type[Exception]
    ) -> None:
        """Verify invalid sign_threshold inputs raise appropriate exceptions."""
        with pytest.raises(expected_exc):
            EpistemicEntropyCalculator(sign_threshold=bad_threshold)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_epsilon,expected_exc",
        [
            (float("nan"), DegenerateCircuitBreakerException),
            (float("inf"), DegenerateCircuitBreakerException),
            (0.0, InvalidCircuitBreakerInputException),
            (-1e-30, InvalidCircuitBreakerInputException),
            ("1e-30", InvalidCircuitBreakerInputException),
        ],
    )
    def test_init_epsilon_log_validation(
        self, bad_epsilon: object, expected_exc: type[Exception]
    ) -> None:
        """Verify invalid epsilon_log inputs raise appropriate exceptions."""
        with pytest.raises(expected_exc):
            EpistemicEntropyCalculator(epsilon_log=bad_epsilon)  # type: ignore[arg-type]

    def test_directional_consensus_balanced_partition(self) -> None:
        """Verify 3-bucket partitioning into positive, negative, and neutral consensus."""
        calc = EpistemicEntropyCalculator(sign_threshold=1e-4)
        predictions = np.array([0.05, -0.02, 0.00001, 0.08, -0.01], dtype=np.float64)
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float64)

        probs = calc.compute_directional_consensus(predictions, weights)

        # Expected: pos=[0.05, 0.08] (0.4), neg=[-0.02, -0.01] (0.4), neu=[0.00001] (0.2)
        assert probs.shape == (3,)
        assert probs.dtype == np.float64
        assert np.allclose(probs, [0.4, 0.4, 0.2], atol=1e-12)
        assert math.isclose(float(np.sum(probs)), 1.0, abs_tol=1e-10)

    def test_directional_consensus_single_model(self) -> None:
        """Verify directional consensus with a single model K=1."""
        calc = EpistemicEntropyCalculator(sign_threshold=1e-4)
        probs_pos = calc.compute_directional_consensus(
            np.array([0.01], dtype=np.float64), np.array([1.0], dtype=np.float64)
        )
        assert np.allclose(probs_pos, [1.0, 0.0, 0.0])

        probs_neg = calc.compute_directional_consensus(
            np.array([-0.01], dtype=np.float64), np.array([1.0], dtype=np.float64)
        )
        assert np.allclose(probs_neg, [0.0, 1.0, 0.0])

        probs_neu = calc.compute_directional_consensus(
            np.array([0.0], dtype=np.float64), np.array([1.0], dtype=np.float64)
        )
        assert np.allclose(probs_neu, [0.0, 0.0, 1.0])

    def test_directional_consensus_exact_boundary(self) -> None:
        """Verify deadband boundary conditions |y_k| <= delta_sign are assigned to neutral."""
        calc = EpistemicEntropyCalculator(sign_threshold=1e-4)
        predictions = np.array([1e-4, -1e-4, 1.0001e-4, -1.0001e-4], dtype=np.float64)
        weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)

        probs = calc.compute_directional_consensus(predictions, weights)
        # 1e-4 and -1e-4 are <= 1e-4 (neutral) -> 0.50
        # 1.0001e-4 > 1e-4 (positive) -> 0.25
        # -1.0001e-4 < -1e-4 (negative) -> 0.25
        assert np.allclose(probs, [0.25, 0.25, 0.50], atol=1e-12)

    def test_directional_consensus_inv_cb_004_compliance(self) -> None:
        """Verify strict INV-CB-004 simplex compliance across heterogeneous weights."""
        calc = EpistemicEntropyCalculator(sign_threshold=1e-4)
        rng = np.random.default_rng(42)
        for _ in range(50):
            k = rng.integers(1, 101)
            raw_w = rng.uniform(0.01, 1.0, size=k)
            w = raw_w / np.sum(raw_w)
            # Re-normalize to guarantee sum within 1e-15
            w = w / np.sum(w)
            preds = rng.normal(0.0, 0.05, size=k)

            probs = calc.compute_directional_consensus(preds, w)
            assert probs.shape == (3,)
            assert probs.dtype == np.float64
            assert np.all(probs >= 0.0)
            assert abs(float(np.sum(probs)) - 1.0) <= 1e-10

    def test_directional_consensus_defensive_failures(self) -> None:
        """Verify defensive failures on malformed or non-finite inputs."""
        calc = EpistemicEntropyCalculator()

        # Non-ndarray
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_directional_consensus([0.1, -0.1], [0.5, 0.5])  # type: ignore[arg-type]
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_directional_consensus(
                np.array([0.1, -0.1]),
                [0.5, 0.5],  # type: ignore[arg-type]
            )

        # Wrong dimensionality (2D)
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_directional_consensus(np.array([[0.1], [-0.1]]), np.array([[0.5], [0.5]]))
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_directional_consensus(np.array([0.1, -0.1]), np.array([[0.5], [0.5]]))

        # Length mismatch
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_directional_consensus(np.array([0.1, 0.2, 0.3]), np.array([0.5, 0.5]))

        # Empty array K=0
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_directional_consensus(np.array([]), np.array([]))

        # INV-CB-005: Non-finite values in predictions
        with pytest.raises(DegenerateCircuitBreakerException):
            calc.compute_directional_consensus(np.array([0.1, float("nan")]), np.array([0.5, 0.5]))
        with pytest.raises(DegenerateCircuitBreakerException):
            calc.compute_directional_consensus(np.array([0.1, float("inf")]), np.array([0.5, 0.5]))

        # INV-CB-005: Non-finite values in weights
        with pytest.raises(DegenerateCircuitBreakerException):
            calc.compute_directional_consensus(np.array([0.1, 0.2]), np.array([float("nan"), 0.5]))

        # Negative weights
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_directional_consensus(np.array([0.1, 0.2]), np.array([-0.1, 1.1]))

        # Weights sum != 1.0
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_directional_consensus(np.array([0.1, 0.2]), np.array([0.4, 0.4]))

    def test_directional_entropy_unanimous_consensus(self) -> None:
        """Verify unanimous consensus yields normalized directional entropy exactly 0.0."""
        calc = EpistemicEntropyCalculator()

        # All positive
        assert calc.compute_directional_entropy(np.array([1.0, 0.0, 0.0])) == 0.0
        # All negative
        assert calc.compute_directional_entropy(np.array([0.0, 1.0, 0.0])) == 0.0
        # All neutral
        assert calc.compute_directional_entropy(np.array([0.0, 0.0, 1.0])) == 0.0

    def test_directional_entropy_maximum_confusion(self) -> None:
        """Verify maximum confusion [1/3, 1/3, 1/3] yields normalized entropy exactly 1.0."""
        calc = EpistemicEntropyCalculator()
        p_uniform = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
        entropy = calc.compute_directional_entropy(p_uniform)
        assert math.isclose(entropy, 1.0, rel_tol=1e-12)
        assert 0.0 <= entropy <= 1.0

    def test_directional_entropy_50_50_polarization(self) -> None:
        """Verify 50/50 polarization yields ln(2)/ln(3) ≈ 0.63092975."""
        calc = EpistemicEntropyCalculator()
        p_bipolar = np.array([0.5, 0.5, 0.0], dtype=np.float64)
        entropy = calc.compute_directional_entropy(p_bipolar)
        expected = math.log(2.0) / math.log(3.0)
        assert math.isclose(entropy, expected, rel_tol=1e-10)
        assert math.isclose(entropy, 0.6309297535714574, rel_tol=1e-10)

        # Bull vs neutral
        p_bull_neu = np.array([0.5, 0.0, 0.5], dtype=np.float64)
        assert math.isclose(calc.compute_directional_entropy(p_bull_neu), expected, rel_tol=1e-10)

    def test_directional_entropy_defensive_failures(self) -> None:
        """Verify directional entropy rejects non-finite or non-simplex vectors."""
        calc = EpistemicEntropyCalculator()

        # Non-ndarray
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_directional_entropy([1.0, 0.0, 0.0])  # type: ignore[arg-type]

        # Wrong shape
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_directional_entropy(np.array([0.5, 0.5]))

        # INV-CB-005: Non-finite probabilities
        with pytest.raises(DegenerateCircuitBreakerException):
            calc.compute_directional_entropy(np.array([float("nan"), 0.5, 0.5]))
        with pytest.raises(DegenerateCircuitBreakerException):
            calc.compute_directional_entropy(np.array([float("inf"), 0.0, 0.0]))

        # Negative probability
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_directional_entropy(np.array([-0.1, 0.6, 0.5]))

        # Sum != 1.0
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_directional_entropy(np.array([0.3, 0.3, 0.3]))

    def test_epistemic_ratio_mathematical_bounds(self) -> None:
        """Verify epistemic uncertainty ratio bounds and monotonicity in [0.0, 1.0)."""
        calc = EpistemicEntropyCalculator()

        # Zero epistemic variance -> rho = 0.0
        assert calc.compute_epistemic_ratio(aleatoric_variance=0.01, epistemic_variance=0.0) == 0.0

        # Equal variances -> rho = 0.5
        assert math.isclose(
            calc.compute_epistemic_ratio(aleatoric_variance=0.04, epistemic_variance=0.04),
            0.5,
            rel_tol=1e-12,
        )

        # Dominant epistemic variance -> rho approaches 1.0 but strictly < 1.0
        rho_high = calc.compute_epistemic_ratio(aleatoric_variance=1e-6, epistemic_variance=1.0)
        assert 0.9999 < rho_high < 1.0

        # Monotonicity with respect to epistemic variance
        r1 = calc.compute_epistemic_ratio(0.01, 0.005)
        r2 = calc.compute_epistemic_ratio(0.01, 0.01)
        r3 = calc.compute_epistemic_ratio(0.01, 0.02)
        assert r1 < r2 < r3

    def test_epistemic_ratio_defensive_failures(self) -> None:
        """Verify epistemic ratio validates positivity and finiteness."""
        calc = EpistemicEntropyCalculator()

        # Aleatoric <= 0.0
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_epistemic_ratio(aleatoric_variance=0.0, epistemic_variance=0.01)
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_epistemic_ratio(aleatoric_variance=-0.01, epistemic_variance=0.01)

        # Epistemic < 0.0
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_epistemic_ratio(aleatoric_variance=0.01, epistemic_variance=-0.001)

        # INV-CB-005: Non-finite inputs
        with pytest.raises(DegenerateCircuitBreakerException):
            calc.compute_epistemic_ratio(aleatoric_variance=float("nan"), epistemic_variance=0.01)
        with pytest.raises(DegenerateCircuitBreakerException):
            calc.compute_epistemic_ratio(aleatoric_variance=0.01, epistemic_variance=float("nan"))
        with pytest.raises(DegenerateCircuitBreakerException):
            calc.compute_epistemic_ratio(aleatoric_variance=float("inf"), epistemic_variance=0.01)

        # Invalid type
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_epistemic_ratio("0.01", 0.01)  # type: ignore[arg-type]
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_epistemic_ratio(0.01, "0.01")  # type: ignore[arg-type]
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_epistemic_ratio(True, 0.01)  # type: ignore[arg-type]
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_epistemic_ratio(0.01, False)  # type: ignore[arg-type]

    def test_composite_epistemic_entropy_dynamics(self) -> None:
        """Verify composite epistemic entropy H_epistemic = H_dir * sqrt(rho)."""
        calc = EpistemicEntropyCalculator(sign_threshold=1e-4)

        # Case 1: Unanimous consensus -> H_epistemic = 0.0 even if epistemic variance is huge
        preds_unanimous = np.array([0.05, 0.04, 0.06], dtype=np.float64)
        weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
        h_epi, h_dir, rho, probs = calc.compute_epistemic_entropy(
            preds_unanimous, weights, aleatoric_variance=0.001, epistemic_variance=1.0
        )
        assert h_epi == 0.0
        assert h_dir == 0.0
        assert rho > 0.99
        assert np.allclose(probs, [1.0, 0.0, 0.0])

        # Case 2: Maximum disagreement, but epistemic variance is 0.0 -> H_epistemic = 0.0
        preds_disagree = np.array([0.05, -0.05, 0.0], dtype=np.float64)
        h_epi, h_dir, rho, probs = calc.compute_epistemic_entropy(
            preds_disagree, weights, aleatoric_variance=0.01, epistemic_variance=0.0
        )
        assert h_epi == 0.0
        assert math.isclose(h_dir, 1.0, rel_tol=1e-12)
        assert rho == 0.0
        assert np.allclose(probs, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

        # Case 3: 50/50 polarization with equal variances (rho = 0.5)
        preds_polar = np.array([0.05, -0.05], dtype=np.float64)
        w_polar = np.array([0.5, 0.5], dtype=np.float64)
        h_epi, h_dir, rho, probs = calc.compute_epistemic_entropy(
            preds_polar, w_polar, aleatoric_variance=0.02, epistemic_variance=0.02
        )
        expected_h_dir = math.log(2.0) / math.log(3.0)
        expected_h_epi = expected_h_dir * math.sqrt(0.5)
        assert math.isclose(h_dir, expected_h_dir, rel_tol=1e-10)
        assert math.isclose(rho, 0.5, rel_tol=1e-10)
        assert math.isclose(h_epi, expected_h_epi, rel_tol=1e-10)
        assert 0.0 <= h_epi <= 1.0

        # Case 4: Output contracts
        assert isinstance(h_epi, float)
        assert isinstance(h_dir, float)
        assert isinstance(rho, float)
        assert isinstance(probs, np.ndarray)
        assert probs.shape == (3,)
        assert probs.dtype == np.float64

    def test_state_construction_from_calculator_outputs(self) -> None:
        """Verify seamless downstream integration into CircuitBreakerState."""
        calc = EpistemicEntropyCalculator(sign_threshold=1e-4)
        preds = np.array([0.02, -0.03, 0.00001], dtype=np.float64)
        weights = np.array([0.5, 0.3, 0.2], dtype=np.float64)

        h_epi, h_dir, rho, probs = calc.compute_epistemic_entropy(
            predictions=preds,
            weights=weights,
            aleatoric_variance=0.01,
            epistemic_variance=0.005,
        )

        state = CircuitBreakerState(
            tier=CircuitBreakerTier.NORMAL,
            active_bars_in_tier=1,
            continuous_haircut=0.85,
            epistemic_entropy=h_epi,
            directional_entropy=h_dir,
            epistemic_ratio=rho,
            composite_shock_score=0.25,
            directional_probabilities=probs,
            step_index=1,
        )

        assert state.epistemic_entropy == h_epi
        assert state.directional_entropy == h_dir
        assert state.epistemic_ratio == rho
        assert np.array_equal(state.directional_probabilities, probs)


class TestContinuousHaircutCalculator:
    """Validate ContinuousHaircutCalculator composite shock and continuous haircut dynamics."""

    def test_init_defaults_and_properties(self) -> None:
        """Verify institutional defaults and read-only property accessors."""
        calc = ContinuousHaircutCalculator()
        assert calc.steepness == 10.0
        assert calc.midpoint == 0.50
        assert calc.weight_entropy == 0.50
        assert calc.weight_epistemic_ratio == 0.30
        assert calc.weight_ambiguity == 0.20
        assert calc.beta_min == 1.0
        assert calc.beta_max == 5.0

        # Custom initialization
        custom = ContinuousHaircutCalculator(
            steepness=15.0,
            midpoint=0.60,
            weight_entropy=0.40,
            weight_epistemic_ratio=0.40,
            weight_ambiguity=0.20,
            beta_min=2.0,
            beta_max=8.0,
        )
        assert custom.steepness == 15.0
        assert custom.midpoint == 0.60
        assert custom.weight_entropy == 0.40
        assert custom.weight_epistemic_ratio == 0.40
        assert custom.weight_ambiguity == 0.20
        assert custom.beta_min == 2.0
        assert custom.beta_max == 8.0

        # Read-only property accessors
        with pytest.raises(AttributeError):
            calc.steepness = 20.0  # type: ignore[misc]
        with pytest.raises(AttributeError):
            calc.midpoint = 0.40  # type: ignore[misc]

    def test_from_config_factory(self) -> None:
        """Verify factory construction from CircuitBreakerConfig."""
        cfg = CircuitBreakerConfig(
            haircut_steepness=12.0,
            haircut_midpoint=0.55,
            weight_entropy=0.60,
            weight_epistemic_ratio=0.25,
            weight_ambiguity=0.15,
            beta_min=1.5,
            beta_max=6.0,
        )
        calc = ContinuousHaircutCalculator.from_config(cfg)
        assert calc.steepness == 12.0
        assert calc.midpoint == 0.55
        assert calc.weight_entropy == 0.60
        assert calc.weight_epistemic_ratio == 0.25
        assert calc.weight_ambiguity == 0.15
        assert calc.beta_min == 1.5
        assert calc.beta_max == 6.0

        with pytest.raises(InvalidCircuitBreakerInputException):
            ContinuousHaircutCalculator.from_config("invalid_config")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "field_name,bad_val",
        [
            ("steepness", float("nan")),
            ("steepness", float("inf")),
            ("midpoint", float("nan")),
            ("midpoint", float("inf")),
            ("weight_entropy", float("nan")),
            ("weight_epistemic_ratio", float("inf")),
            ("weight_ambiguity", float("nan")),
            ("beta_min", float("nan")),
            ("beta_max", float("inf")),
        ],
    )
    def test_init_non_finite_validation(self, field_name: str, bad_val: float) -> None:
        """Verify non-finite parameters trigger DegenerateCircuitBreakerException (INV-CB-005)."""
        kwargs: dict[str, float] = {
            "steepness": 10.0,
            "midpoint": 0.50,
            "weight_entropy": 0.50,
            "weight_epistemic_ratio": 0.30,
            "weight_ambiguity": 0.20,
            "beta_min": 1.0,
            "beta_max": 5.0,
        }
        kwargs[field_name] = bad_val
        with pytest.raises(DegenerateCircuitBreakerException):
            ContinuousHaircutCalculator(**kwargs)

    @pytest.mark.parametrize(
        "field_name,bad_val",
        [
            ("steepness", "10.0"),
            ("steepness", True),
            ("midpoint", False),
            ("weight_entropy", None),
            ("weight_epistemic_ratio", "0.3"),
            ("weight_ambiguity", True),
            ("beta_min", "1.0"),
            ("beta_max", None),
        ],
    )
    def test_init_type_validation(self, field_name: str, bad_val: object) -> None:
        """Verify non-float and boolean parameters raise InvalidCircuitBreakerInputException."""
        kwargs: dict[str, object] = {
            "steepness": 10.0,
            "midpoint": 0.50,
            "weight_entropy": 0.50,
            "weight_epistemic_ratio": 0.30,
            "weight_ambiguity": 0.20,
            "beta_min": 1.0,
            "beta_max": 5.0,
        }
        kwargs[field_name] = bad_val
        with pytest.raises(InvalidCircuitBreakerInputException):
            ContinuousHaircutCalculator(**kwargs)  # type: ignore[arg-type]

    def test_init_boundary_validation(self) -> None:
        """Verify parameter range invariants and constraints."""
        # steepness <= 0.0
        with pytest.raises(InvalidCircuitBreakerInputException):
            ContinuousHaircutCalculator(steepness=0.0)
        with pytest.raises(InvalidCircuitBreakerInputException):
            ContinuousHaircutCalculator(steepness=-5.0)

        # midpoint not in (0.0, 1.0)
        with pytest.raises(InvalidCircuitBreakerInputException):
            ContinuousHaircutCalculator(midpoint=0.0)
        with pytest.raises(InvalidCircuitBreakerInputException):
            ContinuousHaircutCalculator(midpoint=1.0)
        with pytest.raises(InvalidCircuitBreakerInputException):
            ContinuousHaircutCalculator(midpoint=-0.1)
        with pytest.raises(InvalidCircuitBreakerInputException):
            ContinuousHaircutCalculator(midpoint=1.1)

        # negative weights
        with pytest.raises(InvalidCircuitBreakerInputException):
            ContinuousHaircutCalculator(
                weight_entropy=-0.1, weight_epistemic_ratio=0.6, weight_ambiguity=0.5
            )
        with pytest.raises(InvalidCircuitBreakerInputException):
            ContinuousHaircutCalculator(
                weight_entropy=0.5, weight_epistemic_ratio=-0.1, weight_ambiguity=0.6
            )
        with pytest.raises(InvalidCircuitBreakerInputException):
            ContinuousHaircutCalculator(
                weight_entropy=0.5, weight_epistemic_ratio=0.6, weight_ambiguity=-0.1
            )

        # weights sum violation
        with pytest.raises(InvalidCircuitBreakerInputException):
            ContinuousHaircutCalculator(
                weight_entropy=0.40, weight_epistemic_ratio=0.30, weight_ambiguity=0.20
            )
        with pytest.raises(InvalidCircuitBreakerInputException):
            ContinuousHaircutCalculator(
                weight_entropy=0.60, weight_epistemic_ratio=0.30, weight_ambiguity=0.20
            )

        # beta_min <= 0.0
        with pytest.raises(InvalidCircuitBreakerInputException):
            ContinuousHaircutCalculator(beta_min=0.0)
        with pytest.raises(InvalidCircuitBreakerInputException):
            ContinuousHaircutCalculator(beta_min=-1.0)

        # beta_max <= beta_min
        with pytest.raises(InvalidCircuitBreakerInputException):
            ContinuousHaircutCalculator(beta_min=5.0, beta_max=5.0)
        with pytest.raises(InvalidCircuitBreakerInputException):
            ContinuousHaircutCalculator(beta_min=5.0, beta_max=3.0)

    def test_compute_composite_shock_mathematical_precision(self) -> None:
        """Verify thermodynamic normalized ambiguity and composite shock score precision."""
        calc = ContinuousHaircutCalculator(
            weight_entropy=0.50,
            weight_epistemic_ratio=0.30,
            weight_ambiguity=0.20,
            beta_min=1.0,
            beta_max=5.0,
        )

        # Minimum shock: H=0, rho=0, beta <= beta_min -> Xi = 0.0
        shock_min = calc.compute_composite_shock(
            epistemic_entropy=0.0, epistemic_ratio=0.0, ambiguity_beta=0.5
        )
        assert shock_min == 0.0

        # Maximum shock: H=1, rho=1, beta >= beta_max -> Xi = 1.0
        shock_max = calc.compute_composite_shock(
            epistemic_entropy=1.0, epistemic_ratio=1.0, ambiguity_beta=6.0
        )
        assert shock_max == 1.0

        # Midpoint ambiguity: beta = 3.0 -> tilde_beta = (3-1)/(5-1) = 0.50
        # Xi = 0.50 * 0.4 + 0.30 * 0.6 + 0.20 * 0.50 = 0.20 + 0.18 + 0.10 = 0.48
        shock_mid = calc.compute_composite_shock(
            epistemic_entropy=0.40, epistemic_ratio=0.60, ambiguity_beta=3.0
        )
        assert math.isclose(shock_mid, 0.48, rel_tol=1e-12)
        assert 0.0 <= shock_mid <= 1.0

        # Boundary beta exactly at beta_min and beta_max
        shock_bmin = calc.compute_composite_shock(
            epistemic_entropy=0.20, epistemic_ratio=0.30, ambiguity_beta=1.0
        )
        # tilde_beta = 0.0 -> Xi = 0.50 * 0.20 + 0.30 * 0.30 = 0.19
        assert math.isclose(shock_bmin, 0.19, rel_tol=1e-12)

        shock_bmax = calc.compute_composite_shock(
            epistemic_entropy=0.20, epistemic_ratio=0.30, ambiguity_beta=5.0
        )
        # tilde_beta = 1.0 -> Xi = 0.19 + 0.20 = 0.39
        assert math.isclose(shock_bmax, 0.39, rel_tol=1e-12)

    def test_compute_composite_shock_defensive_failures(self) -> None:
        """Verify composite shock calculation rejects invalid or non-finite inputs."""
        calc = ContinuousHaircutCalculator()

        # Non-finite inputs raise DegenerateCircuitBreakerException (INV-CB-005)
        with pytest.raises(DegenerateCircuitBreakerException):
            calc.compute_composite_shock(
                epistemic_entropy=float("nan"), epistemic_ratio=0.5, ambiguity_beta=2.0
            )
        with pytest.raises(DegenerateCircuitBreakerException):
            calc.compute_composite_shock(
                epistemic_entropy=0.5, epistemic_ratio=float("inf"), ambiguity_beta=2.0
            )
        with pytest.raises(DegenerateCircuitBreakerException):
            calc.compute_composite_shock(
                epistemic_entropy=0.5, epistemic_ratio=0.5, ambiguity_beta=float("nan")
            )

        # Out of bounds raises InvalidCircuitBreakerInputException
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_composite_shock(
                epistemic_entropy=-0.01, epistemic_ratio=0.5, ambiguity_beta=2.0
            )
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_composite_shock(
                epistemic_entropy=1.01, epistemic_ratio=0.5, ambiguity_beta=2.0
            )
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_composite_shock(
                epistemic_entropy=0.5, epistemic_ratio=-0.01, ambiguity_beta=2.0
            )
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_composite_shock(
                epistemic_entropy=0.5, epistemic_ratio=1.01, ambiguity_beta=2.0
            )

        # Non-positive ambiguity_beta
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_composite_shock(
                epistemic_entropy=0.5, epistemic_ratio=0.5, ambiguity_beta=0.0
            )
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_composite_shock(
                epistemic_entropy=0.5, epistemic_ratio=0.5, ambiguity_beta=-1.0
            )

        # Type errors
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_composite_shock(
                epistemic_entropy="0.5",  # type: ignore[arg-type]
                epistemic_ratio=0.5,
                ambiguity_beta=2.0,
            )
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_composite_shock(
                epistemic_entropy=0.5,
                epistemic_ratio=True,  # type: ignore[arg-type]
                ambiguity_beta=2.0,
            )
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_composite_shock(
                epistemic_entropy=0.5,
                epistemic_ratio=0.5,
                ambiguity_beta=False,  # type: ignore[arg-type]
            )

    def test_compute_haircut_boundary_contracts(self) -> None:
        """INV-CB-001: kappa(0.0) == 1.0, kappa(1.0) == 0.0, kappa(midpoint) == 0.50."""
        # Symmetric midpoint (0.50)
        calc = ContinuousHaircutCalculator(steepness=10.0, midpoint=0.50)

        assert calc.compute_haircut(0.0) == 1.0
        assert calc.compute_haircut(1.0) == 0.0
        assert math.isclose(calc.compute_haircut(0.50), 0.50, rel_tol=1e-12)

        # Asymmetric midpoint (0.40, k=12.0)
        calc_asym = ContinuousHaircutCalculator(steepness=12.0, midpoint=0.40)
        assert calc_asym.compute_haircut(0.0) == 1.0
        assert calc_asym.compute_haircut(1.0) == 0.0
        assert math.isclose(calc_asym.compute_haircut(0.40), 0.50, abs_tol=1e-2)

        # Asymmetric midpoint (0.70, k=8.0) approx 0.50 within 0.05
        calc_high = ContinuousHaircutCalculator(steepness=8.0, midpoint=0.70)
        assert calc_high.compute_haircut(0.0) == 1.0
        assert calc_high.compute_haircut(1.0) == 0.0
        assert math.isclose(calc_high.compute_haircut(0.70), 0.50, abs_tol=0.05)

    def test_compute_haircut_monotonicity_and_smoothness(self) -> None:
        """Verify haircut is strictly monotonic non-increasing and continuously differentiable."""
        calc = ContinuousHaircutCalculator(steepness=10.0, midpoint=0.50)
        shocks = np.linspace(0.0, 1.0, 1001)
        haircuts = np.array([calc.compute_haircut(float(s)) for s in shocks])

        # All values strictly in [0.0, 1.0]
        assert np.all(haircuts >= 0.0)
        assert np.all(haircuts <= 1.0)

        # Exact endpoints
        assert haircuts[0] == 1.0
        assert haircuts[-1] == 0.0

        # Monotonicity: d_kappa / d_Xi <= 0 everywhere
        diffs = np.diff(haircuts)
        assert np.all(diffs <= 0.0)

        # Smoothness: no discontinuous jumps
        max_diff = float(np.max(np.abs(diffs)))
        # For k=10, max slope is at midpoint: |d_kappa/d_Xi| ~ k/4 = 2.5
        # Step size h = 1e-3, so max step difference ~ 2.5 * 1e-3 = 2.5e-3
        assert max_diff < 5e-3

    def test_compute_haircut_steepness_sensitivity(self) -> None:
        """Verify steeper k yields sharper transition around midpoint."""
        calc_gentle = ContinuousHaircutCalculator(steepness=4.0, midpoint=0.50)
        calc_steep = ContinuousHaircutCalculator(steepness=25.0, midpoint=0.50)

        # At Xi = 0.20 (low shock):
        # gentle haircut preserves some risk reduction
        # steep haircut stays much closer to 1.0
        k_gentle_low = calc_gentle.compute_haircut(0.20)
        k_steep_low = calc_steep.compute_haircut(0.20)
        assert k_steep_low > k_gentle_low

        # At Xi = 0.80 (high shock):
        # gentle haircut retains some small exposure
        # steep haircut cuts exposure close to 0.0
        k_gentle_high = calc_gentle.compute_haircut(0.80)
        k_steep_high = calc_steep.compute_haircut(0.80)
        assert k_steep_high < k_gentle_high

    def test_compute_haircut_defensive_failures(self) -> None:
        """Verify haircut computation rejects out-of-bounds, non-finite, and invalid types."""
        calc = ContinuousHaircutCalculator()

        # Non-finite values raise DegenerateCircuitBreakerException (INV-CB-005)
        with pytest.raises(DegenerateCircuitBreakerException):
            calc.compute_haircut(float("nan"))
        with pytest.raises(DegenerateCircuitBreakerException):
            calc.compute_haircut(float("inf"))
        with pytest.raises(DegenerateCircuitBreakerException):
            calc.compute_haircut(-float("inf"))

        # Out of bounds raise InvalidCircuitBreakerInputException
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_haircut(-0.001)
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_haircut(1.001)

        # Invalid types
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_haircut("0.5")  # type: ignore[arg-type]
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_haircut(True)  # type: ignore[arg-type]
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_haircut(False)  # type: ignore[arg-type]
        with pytest.raises(InvalidCircuitBreakerInputException):
            calc.compute_haircut(None)  # type: ignore[arg-type]

    def test_end_to_end_circuit_breaker_pipeline_integration(self) -> None:
        """Verify seamless integration from entropy calculator to haircut calculator and domain state."""
        entropy_calc = EpistemicEntropyCalculator(sign_threshold=1e-4)
        haircut_calc = ContinuousHaircutCalculator(steepness=10.0, midpoint=0.50)

        # Disagreement scenario: split models
        predictions = np.array([0.04, -0.03, 0.05, -0.04], dtype=np.float64)
        weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)

        h_epi, h_dir, rho, probs = entropy_calc.compute_epistemic_entropy(
            predictions=predictions,
            weights=weights,
            aleatoric_variance=0.01,
            epistemic_variance=0.04,  # epistemic dominant: rho = 0.04 / 0.05 = 0.80
        )

        ambiguity_beta = 3.5  # elevated macro ambiguity

        # Compute composite shock
        shock = haircut_calc.compute_composite_shock(
            epistemic_entropy=h_epi,
            epistemic_ratio=rho,
            ambiguity_beta=ambiguity_beta,
        )
        assert 0.0 <= shock <= 1.0

        # Compute continuous haircut
        haircut = haircut_calc.compute_haircut(shock)
        assert 0.0 <= haircut <= 1.0

        # Construct immutable state
        state = CircuitBreakerState(
            tier=CircuitBreakerTier.CAUTION if haircut < 0.50 else CircuitBreakerTier.NORMAL,
            active_bars_in_tier=1,
            continuous_haircut=haircut,
            epistemic_entropy=h_epi,
            directional_entropy=h_dir,
            epistemic_ratio=rho,
            composite_shock_score=shock,
            directional_probabilities=probs,
            step_index=1,
        )

        # Form execution decision
        decision = CircuitBreakerDecision.from_state(
            action_tier=state.tier,
            execution_haircut=state.continuous_haircut,
            state=state,
        )

        assert decision.execution_haircut == haircut
        assert decision.state == state

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
    DegenerateCircuitBreakerException,
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

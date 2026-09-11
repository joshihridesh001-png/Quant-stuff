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
    CircuitBreakerOverlayEngine,
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


class TestCircuitBreakerOverlayEngine:
    """Validate CircuitBreakerOverlayEngine state machine, multi-tier escalation, and hysteresis."""

    @pytest.fixture
    def default_engine(self) -> CircuitBreakerOverlayEngine:
        """Provide a default CircuitBreakerOverlayEngine."""
        return CircuitBreakerOverlayEngine()

    @pytest.fixture
    def quiescent_inputs(
        self,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        """Low uncertainty inputs producing composite shock Xi_t = 0.0."""
        preds = np.array([0.05, 0.05, 0.05], dtype=np.float64)
        weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
        aleatoric = 0.05
        epistemic = 0.0
        beta = 1.0  # beta_min -> normalized ambiguity = 0.0
        return preds, weights, aleatoric, epistemic, beta

    @pytest.fixture
    def caution_inputs(
        self,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        """Moderate uncertainty inputs producing composite shock Xi_t in [0.45, 0.70)."""
        preds = np.array([0.05, -0.05], dtype=np.float64)
        weights = np.array([0.5, 0.5], dtype=np.float64)
        aleatoric = 0.01
        epistemic = 0.04  # rho = 0.80, H_dir ~ 0.6309, H_epi ~ 0.5643
        beta = 3.0  # tilde_beta = 0.50 -> Xi ~ 0.622
        return preds, weights, aleatoric, epistemic, beta

    @pytest.fixture
    def derisk_inputs(
        self,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        """High uncertainty inputs producing composite shock Xi_t in [0.70, 0.90)."""
        preds = np.array([0.05, -0.05, 0.0], dtype=np.float64)
        weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
        aleatoric = 0.04
        epistemic = 0.06  # rho = 0.60, H_dir = 1.0, H_epi ~ 0.7746
        beta = 4.0  # tilde_beta = 0.75 -> Xi ~ 0.717
        return preds, weights, aleatoric, epistemic, beta

    @pytest.fixture
    def halt_inputs(
        self,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        """Extreme uncertainty inputs producing composite shock Xi_t >= 0.90."""
        preds = np.array([0.05, -0.05, 0.0], dtype=np.float64)
        weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
        aleatoric = 0.001
        epistemic = 0.099  # rho = 0.99, H_dir = 1.0, H_epi ~ 0.9950
        beta = 5.0  # tilde_beta = 1.0 -> Xi ~ 0.994
        return preds, weights, aleatoric, epistemic, beta

    def test_init_defaults_and_properties(
        self, default_engine: CircuitBreakerOverlayEngine
    ) -> None:
        """Verify default configuration and subordinate calculator properties."""
        assert isinstance(default_engine.config, CircuitBreakerConfig)
        assert default_engine.config == CircuitBreakerConfig()
        assert isinstance(default_engine.entropy_calculator, EpistemicEntropyCalculator)
        assert isinstance(default_engine.haircut_calculator, ContinuousHaircutCalculator)
        assert (
            default_engine.entropy_calculator.sign_threshold == default_engine.config.sign_threshold
        )
        assert (
            default_engine.haircut_calculator.steepness == default_engine.config.haircut_steepness
        )

    def test_init_custom_config_and_invalid_type(self) -> None:
        """Verify custom configuration injection and type defense."""
        custom_cfg = CircuitBreakerConfig(
            dwell_time_bars=8,
            caution_threshold=0.40,
            derisk_threshold=0.65,
            halt_threshold=0.85,
            recovery_threshold=0.25,
        )
        engine = CircuitBreakerOverlayEngine(custom_cfg)
        assert engine.config == custom_cfg
        assert engine.config.dwell_time_bars == 8

        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerOverlayEngine("invalid_cfg")  # type: ignore[arg-type]
        with pytest.raises(InvalidCircuitBreakerInputException):
            CircuitBreakerOverlayEngine(123)  # type: ignore[arg-type]

    def test_initialize_state(self, default_engine: CircuitBreakerOverlayEngine) -> None:
        """Verify clean initial state generation at step 0 in NORMAL tier."""
        state = default_engine.initialize_state()
        assert state.tier == CircuitBreakerTier.NORMAL
        assert state.active_bars_in_tier == 0
        assert state.continuous_haircut == 1.0
        assert state.epistemic_entropy == 0.0
        assert state.directional_entropy == 0.0
        assert state.epistemic_ratio == 0.0
        assert state.composite_shock_score == 0.0
        assert np.allclose(
            state.directional_probabilities,
            np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64),
        )
        assert state.step_index == 0

    def test_evaluate_input_validation_and_defensive_failures(
        self,
        default_engine: CircuitBreakerOverlayEngine,
        quiescent_inputs: tuple[np.ndarray, np.ndarray, float, float, float],
    ) -> None:
        """Verify strict invariant INV-CB-005 non-finite checks and type validations."""
        preds, weights, aleatoric, epistemic, beta = quiescent_inputs
        state = default_engine.initialize_state()

        # State must be CircuitBreakerState
        with pytest.raises(InvalidCircuitBreakerInputException):
            default_engine.evaluate(
                preds,
                weights,
                aleatoric,
                epistemic,
                beta,
                state="invalid_state",  # type: ignore[arg-type]
            )
        with pytest.raises(InvalidCircuitBreakerInputException):
            default_engine.evaluate(
                preds,
                weights,
                aleatoric,
                epistemic,
                beta,
                state=None,  # type: ignore[arg-type]
            )

        # cusum_shock and regime_is_panic must be strictly boolean
        with pytest.raises(InvalidCircuitBreakerInputException):
            default_engine.evaluate(
                preds,
                weights,
                aleatoric,
                epistemic,
                beta,
                state,
                cusum_shock=1,  # type: ignore[arg-type]
            )
        with pytest.raises(InvalidCircuitBreakerInputException):
            default_engine.evaluate(
                preds,
                weights,
                aleatoric,
                epistemic,
                beta,
                state,
                cusum_shock="True",  # type: ignore[arg-type]
            )
        with pytest.raises(InvalidCircuitBreakerInputException):
            default_engine.evaluate(
                preds,
                weights,
                aleatoric,
                epistemic,
                beta,
                state,
                regime_is_panic=0,  # type: ignore[arg-type]
            )
        with pytest.raises(InvalidCircuitBreakerInputException):
            default_engine.evaluate(
                preds,
                weights,
                aleatoric,
                epistemic,
                beta,
                state,
                regime_is_panic="False",  # type: ignore[arg-type]
            )

        # INV-CB-005 non-finite predictions
        with pytest.raises(DegenerateCircuitBreakerException):
            default_engine.evaluate(
                np.array([0.05, float("nan")], dtype=np.float64),
                np.array([0.5, 0.5], dtype=np.float64),
                aleatoric,
                epistemic,
                beta,
                state,
            )
        with pytest.raises(DegenerateCircuitBreakerException):
            default_engine.evaluate(
                np.array([0.05, float("inf")], dtype=np.float64),
                np.array([0.5, 0.5], dtype=np.float64),
                aleatoric,
                epistemic,
                beta,
                state,
            )

        # INV-CB-005 non-finite weights
        with pytest.raises(DegenerateCircuitBreakerException):
            default_engine.evaluate(
                preds,
                np.array([float("nan"), 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64),
                aleatoric,
                epistemic,
                beta,
                state,
            )

        # INV-CB-005 non-finite variances
        with pytest.raises(DegenerateCircuitBreakerException):
            default_engine.evaluate(preds, weights, float("nan"), epistemic, beta, state)
        with pytest.raises(DegenerateCircuitBreakerException):
            default_engine.evaluate(preds, weights, float("inf"), epistemic, beta, state)
        with pytest.raises(DegenerateCircuitBreakerException):
            default_engine.evaluate(preds, weights, aleatoric, float("nan"), beta, state)
        with pytest.raises(DegenerateCircuitBreakerException):
            default_engine.evaluate(preds, weights, aleatoric, float("inf"), beta, state)

        # INV-CB-005 non-finite ambiguity beta
        with pytest.raises(DegenerateCircuitBreakerException):
            default_engine.evaluate(preds, weights, aleatoric, epistemic, float("nan"), state)
        with pytest.raises(DegenerateCircuitBreakerException):
            default_engine.evaluate(preds, weights, aleatoric, epistemic, float("inf"), state)

        # Out-of-bounds ambiguity beta (<= 0)
        with pytest.raises(InvalidCircuitBreakerInputException):
            default_engine.evaluate(preds, weights, aleatoric, epistemic, 0.0, state)
        with pytest.raises(InvalidCircuitBreakerInputException):
            default_engine.evaluate(preds, weights, aleatoric, epistemic, -1.0, state)

    def test_evaluate_normal_state_progression(
        self,
        default_engine: CircuitBreakerOverlayEngine,
        quiescent_inputs: tuple[np.ndarray, np.ndarray, float, float, float],
    ) -> None:
        """Verify normal progression, step_index increments, and active_bars_in_tier increments."""
        preds, weights, aleatoric, epistemic, beta = quiescent_inputs
        state_0 = default_engine.initialize_state()

        # Step 1
        decision_1, state_1 = default_engine.evaluate(
            preds, weights, aleatoric, epistemic, beta, state_0
        )
        assert state_1.tier == CircuitBreakerTier.NORMAL
        assert state_1.step_index == 1
        assert state_1.active_bars_in_tier == 1
        assert state_1.composite_shock_score == 0.0
        assert state_1.continuous_haircut == 1.0
        assert decision_1.action_tier == CircuitBreakerTier.NORMAL
        assert decision_1.execution_haircut == 1.0
        assert not decision_1.is_halted
        assert not decision_1.is_derisking
        assert not decision_1.is_throttled

        # Step 2
        decision_2, state_2 = default_engine.evaluate(
            preds, weights, aleatoric, epistemic, beta, state_1
        )
        assert state_2.tier == CircuitBreakerTier.NORMAL
        assert state_2.step_index == 2
        assert state_2.active_bars_in_tier == 2
        assert decision_2.execution_haircut == 1.0

        # Step 3
        decision_3, state_3 = default_engine.evaluate(
            preds, weights, aleatoric, epistemic, beta, state_2
        )
        assert state_3.tier == CircuitBreakerTier.NORMAL
        assert state_3.step_index == 3
        assert state_3.active_bars_in_tier == 3

    def test_evaluate_instantaneous_escalation_to_caution(
        self,
        default_engine: CircuitBreakerOverlayEngine,
        quiescent_inputs: tuple[np.ndarray, np.ndarray, float, float, float],
        caution_inputs: tuple[np.ndarray, np.ndarray, float, float, float],
    ) -> None:
        """Verify instantaneous escalation from NORMAL to CAUTION with active_bars reset to 1 and throttled haircut."""
        state = default_engine.initialize_state()
        preds_q, w_q, a_q, e_q, b_q = quiescent_inputs
        _, state = default_engine.evaluate(preds_q, w_q, a_q, e_q, b_q, state)
        _, state = default_engine.evaluate(preds_q, w_q, a_q, e_q, b_q, state)
        assert state.tier == CircuitBreakerTier.NORMAL
        assert state.active_bars_in_tier == 2

        # Trigger CAUTION
        preds_c, w_c, a_c, e_c, b_c = caution_inputs
        decision, state = default_engine.evaluate(preds_c, w_c, a_c, e_c, b_c, state)
        assert state.tier == CircuitBreakerTier.CAUTION
        assert state.active_bars_in_tier == 1
        assert state.step_index == 3
        assert decision.action_tier == CircuitBreakerTier.CAUTION
        assert decision.is_throttled is True
        assert not decision.is_derisking
        assert not decision.is_halted
        # CAUTION caps execution haircut at min(0.50, continuous_haircut)
        assert decision.execution_haircut <= 0.50
        assert decision.execution_haircut == min(0.50, state.continuous_haircut)

        # Second bar sustaining CAUTION
        decision_2, state_2 = default_engine.evaluate(preds_c, w_c, a_c, e_c, b_c, state)
        assert state_2.tier == CircuitBreakerTier.CAUTION
        assert state_2.active_bars_in_tier == 2
        assert state_2.step_index == 4

    def test_evaluate_instantaneous_escalation_to_derisk(
        self,
        default_engine: CircuitBreakerOverlayEngine,
        derisk_inputs: tuple[np.ndarray, np.ndarray, float, float, float],
    ) -> None:
        """Verify instantaneous escalation from NORMAL to DERISK with haircut 0.0."""
        state = default_engine.initialize_state()
        preds_d, w_d, a_d, e_d, b_d = derisk_inputs

        decision, state = default_engine.evaluate(preds_d, w_d, a_d, e_d, b_d, state)
        assert state.tier == CircuitBreakerTier.DERISK
        assert state.active_bars_in_tier == 1
        assert decision.action_tier == CircuitBreakerTier.DERISK
        assert decision.is_derisking is True
        assert not decision.is_halted
        assert not decision.is_throttled
        assert decision.execution_haircut == 0.0

    def test_evaluate_instantaneous_escalation_to_halt(
        self,
        default_engine: CircuitBreakerOverlayEngine,
        halt_inputs: tuple[np.ndarray, np.ndarray, float, float, float],
    ) -> None:
        """Verify instantaneous escalation to HALT with haircut 0.0."""
        state = default_engine.initialize_state()
        preds_h, w_h, a_h, e_h, b_h = halt_inputs

        decision, state = default_engine.evaluate(preds_h, w_h, a_h, e_h, b_h, state)
        assert state.tier == CircuitBreakerTier.HALT
        assert state.active_bars_in_tier == 1
        assert decision.action_tier == CircuitBreakerTier.HALT
        assert decision.is_halted is True
        assert not decision.is_derisking
        assert not decision.is_throttled
        assert decision.execution_haircut == 0.0

    def test_evaluate_exogenous_cusum_panic_shock_combinations(
        self,
        default_engine: CircuitBreakerOverlayEngine,
        quiescent_inputs: tuple[np.ndarray, np.ndarray, float, float, float],
    ) -> None:
        """Verify exogenous CUSUM panic shock triggers immediate HALT only when both flags are active."""
        preds, weights, aleatoric, epistemic, beta = quiescent_inputs
        state = default_engine.initialize_state()

        # cusum_shock=True, regime_is_panic=False -> Candidate is NORMAL based on shock
        dec, s = default_engine.evaluate(
            preds,
            weights,
            aleatoric,
            epistemic,
            beta,
            state,
            cusum_shock=True,
            regime_is_panic=False,
        )
        assert s.tier == CircuitBreakerTier.NORMAL

        # cusum_shock=False, regime_is_panic=True -> Candidate is NORMAL based on shock
        dec, s = default_engine.evaluate(
            preds,
            weights,
            aleatoric,
            epistemic,
            beta,
            state,
            cusum_shock=False,
            regime_is_panic=True,
        )
        assert s.tier == CircuitBreakerTier.NORMAL

        # cusum_shock=True, regime_is_panic=True -> Candidate is HALT immediately!
        dec, s_halt = default_engine.evaluate(
            preds,
            weights,
            aleatoric,
            epistemic,
            beta,
            state,
            cusum_shock=True,
            regime_is_panic=True,
        )
        assert s_halt.tier == CircuitBreakerTier.HALT
        assert s_halt.active_bars_in_tier == 1
        assert dec.action_tier == CircuitBreakerTier.HALT
        assert dec.is_halted is True
        assert dec.execution_haircut == 0.0

    def test_evaluate_hysteresis_lockout_in_halt_inv_cb_003(
        self,
        default_engine: CircuitBreakerOverlayEngine,
        halt_inputs: tuple[np.ndarray, np.ndarray, float, float, float],
        quiescent_inputs: tuple[np.ndarray, np.ndarray, float, float, float],
    ) -> None:
        """Verify INV-CB-003 anti-chattering lockout: HALT requires dwell_time_bars and Xi < recovery_threshold to de-escalate."""
        preds_h, w_h, a_h, e_h, b_h = halt_inputs
        preds_q, w_q, a_q, e_q, b_q = quiescent_inputs

        # Escalate to HALT (dwell_time_bars default = 5, recovery_threshold = 0.30)
        state = default_engine.initialize_state()
        dec, state = default_engine.evaluate(preds_h, w_h, a_h, e_h, b_h, state)
        assert state.tier == CircuitBreakerTier.HALT
        assert state.active_bars_in_tier == 1

        # Bars 2 to 5: Quiescent inputs (Xi = 0.0 < 0.30), but active_bars < 5
        for expected_bar in range(2, 6):
            dec, state = default_engine.evaluate(preds_q, w_q, a_q, e_q, b_q, state)
            assert state.tier == CircuitBreakerTier.HALT, f"Premature exit at bar {expected_bar}"
            assert state.active_bars_in_tier == expected_bar
            assert dec.action_tier == CircuitBreakerTier.HALT
            assert dec.execution_haircut == 0.0

        assert state.active_bars_in_tier == 5  # Now eligible on next evaluation!

        # Next evaluation under quiescent conditions: both active >= 5 AND Xi < 0.30 met!
        # Must step down one tier to DERISK (not skipping to NORMAL)
        dec, state = default_engine.evaluate(preds_q, w_q, a_q, e_q, b_q, state)
        assert state.tier == CircuitBreakerTier.DERISK
        assert state.active_bars_in_tier == 1  # Reset to 1 upon tier transition
        assert dec.action_tier == CircuitBreakerTier.DERISK
        assert dec.execution_haircut == 0.0

    def test_evaluate_halt_sustains_if_recovery_threshold_not_met(
        self,
        default_engine: CircuitBreakerOverlayEngine,
        halt_inputs: tuple[np.ndarray, np.ndarray, float, float, float],
        caution_inputs: tuple[np.ndarray, np.ndarray, float, float, float],
    ) -> None:
        """Verify HALT holds even when active_bars >= dwell_time_bars if Xi >= recovery_threshold."""
        preds_h, w_h, a_h, e_h, b_h = halt_inputs
        preds_c, w_c, a_c, e_c, b_c = caution_inputs

        state = default_engine.initialize_state()
        _, state = default_engine.evaluate(preds_h, w_h, a_h, e_h, b_h, state)

        # Fast forward active bars to 5
        for _ in range(4):
            _, state = default_engine.evaluate(preds_h, w_h, a_h, e_h, b_h, state)
        assert state.tier == CircuitBreakerTier.HALT
        assert state.active_bars_in_tier == 5

        # Evaluate with CAUTION inputs (Xi ~ 0.622 >= 0.30): candidate is CAUTION, but recovery threshold not met
        dec, state = default_engine.evaluate(preds_c, w_c, a_c, e_c, b_c, state)
        assert state.tier == CircuitBreakerTier.HALT
        assert state.active_bars_in_tier == 6
        assert dec.action_tier == CircuitBreakerTier.HALT

    def test_evaluate_hysteresis_lockout_in_derisk_inv_cb_003(
        self,
        default_engine: CircuitBreakerOverlayEngine,
        derisk_inputs: tuple[np.ndarray, np.ndarray, float, float, float],
        quiescent_inputs: tuple[np.ndarray, np.ndarray, float, float, float],
    ) -> None:
        """Verify INV-CB-003: DERISK requires dwell_time_bars before stepping down to CAUTION."""
        preds_d, w_d, a_d, e_d, b_d = derisk_inputs
        preds_q, w_q, a_q, e_q, b_q = quiescent_inputs

        state = default_engine.initialize_state()
        _, state = default_engine.evaluate(preds_d, w_d, a_d, e_d, b_d, state)
        assert state.tier == CircuitBreakerTier.DERISK
        assert state.active_bars_in_tier == 1

        # Bars 2 to 5: locked in DERISK
        for expected_bar in range(2, 6):
            dec, state = default_engine.evaluate(preds_q, w_q, a_q, e_q, b_q, state)
            assert state.tier == CircuitBreakerTier.DERISK
            assert state.active_bars_in_tier == expected_bar
            assert dec.execution_haircut == 0.0

        # Bar 6: active_bars = 5 >= 5 AND Xi < 0.30 -> Step down to CAUTION
        dec, state = default_engine.evaluate(preds_q, w_q, a_q, e_q, b_q, state)
        assert state.tier == CircuitBreakerTier.CAUTION
        assert state.active_bars_in_tier == 1
        assert dec.action_tier == CircuitBreakerTier.CAUTION
        assert dec.is_throttled is True
        assert dec.execution_haircut == min(0.50, state.continuous_haircut)

    def test_evaluate_hysteresis_caution_to_normal_transition(
        self,
        default_engine: CircuitBreakerOverlayEngine,
        caution_inputs: tuple[np.ndarray, np.ndarray, float, float, float],
        quiescent_inputs: tuple[np.ndarray, np.ndarray, float, float, float],
    ) -> None:
        """Verify CAUTION transitions to NORMAL immediately when Xi < recovery_threshold."""
        preds_c, w_c, a_c, e_c, b_c = caution_inputs
        preds_q, w_q, a_q, e_q, b_q = quiescent_inputs

        state = default_engine.initialize_state()
        _, state = default_engine.evaluate(preds_c, w_c, a_c, e_c, b_c, state)
        assert state.tier == CircuitBreakerTier.CAUTION
        assert state.active_bars_in_tier == 1

        # Immediate de-escalation to NORMAL when Xi < 0.30
        dec, state = default_engine.evaluate(preds_q, w_q, a_q, e_q, b_q, state)
        assert state.tier == CircuitBreakerTier.NORMAL
        assert state.active_bars_in_tier == 1
        assert dec.action_tier == CircuitBreakerTier.NORMAL
        assert dec.execution_haircut == state.continuous_haircut

    def test_evaluate_caution_deadband_retention(
        self,
        default_engine: CircuitBreakerOverlayEngine,
        caution_inputs: tuple[np.ndarray, np.ndarray, float, float, float],
    ) -> None:
        """Verify CAUTION tier retention when shock score is in deadband [recovery_threshold, caution_threshold)."""
        preds_c, w_c, a_c, e_c, b_c = caution_inputs

        state = default_engine.initialize_state()
        _, state = default_engine.evaluate(preds_c, w_c, a_c, e_c, b_c, state)
        assert state.tier == CircuitBreakerTier.CAUTION
        assert state.active_bars_in_tier == 1

        # Synthesize deadband inputs: candidate is NORMAL (Xi < 0.45), but Xi >= 0.30
        # For example: unanimous predictions, epistemic_ratio = 0.0, ambiguity_beta = 2.5
        # tilde_beta = (2.5 - 1.0) / 4.0 = 0.375
        # Xi = 0.20 * 0.375 = 0.075? No, we want Xi in [0.30, 0.45).
        # Let ambiguity_beta = 1.0 (tilde_beta = 0.0), H_dir = 0.0, epistemic_ratio = 1.0 (weight 0.30)
        # Xi = 0.30 * 1.0 = 0.3000! Candidate is NORMAL since 0.30 < 0.45, but 0.30 >= 0.30 (not < 0.30).
        preds_deadband = np.array([0.05, 0.05], dtype=np.float64)
        weights_deadband = np.array([0.5, 0.5], dtype=np.float64)
        aleatoric_deadband = 0.0001
        epistemic_deadband = 0.1  # rho ~ 1.0, H_dir = 0.0 -> weight_ratio = 0.30
        beta_deadband = (
            1.6  # tilde_beta = 0.6 / 4.0 = 0.15 -> weight_beta * 0.15 = 0.03 -> Xi ~ 0.33
        )

        shock = default_engine.haircut_calculator.compute_composite_shock(
            epistemic_entropy=0.0,
            epistemic_ratio=default_engine.entropy_calculator.compute_epistemic_ratio(
                aleatoric_deadband, epistemic_deadband
            ),
            ambiguity_beta=beta_deadband,
        )
        assert 0.30 <= shock < 0.45

        dec, state = default_engine.evaluate(
            preds_deadband,
            weights_deadband,
            aleatoric_deadband,
            epistemic_deadband,
            beta_deadband,
            state,
        )
        # Should stay in CAUTION because shock >= 0.30
        assert state.tier == CircuitBreakerTier.CAUTION
        assert state.active_bars_in_tier == 2
        assert dec.action_tier == CircuitBreakerTier.CAUTION
        assert dec.is_throttled is True

    def test_evaluate_full_lifecycle_simulation(
        self,
        default_engine: CircuitBreakerOverlayEngine,
        quiescent_inputs: tuple[np.ndarray, np.ndarray, float, float, float],
        caution_inputs: tuple[np.ndarray, np.ndarray, float, float, float],
        derisk_inputs: tuple[np.ndarray, np.ndarray, float, float, float],
        halt_inputs: tuple[np.ndarray, np.ndarray, float, float, float],
    ) -> None:
        """Full institutional lifecycle simulation traversing NORMAL -> CAUTION -> DERISK -> HALT -> DERISK -> CAUTION -> NORMAL."""
        p_q, w_q, a_q, e_q, b_q = quiescent_inputs
        p_c, w_c, a_c, e_c, b_c = caution_inputs
        p_d, w_d, a_d, e_d, b_d = derisk_inputs
        p_h, w_h, a_h, e_h, b_h = halt_inputs

        state = default_engine.initialize_state()
        history_tiers: list[CircuitBreakerTier] = []

        # 1. NORMAL (2 bars)
        for _ in range(2):
            dec, state = default_engine.evaluate(p_q, w_q, a_q, e_q, b_q, state)
            history_tiers.append(state.tier)
            assert dec.execution_haircut == 1.0

        # 2. CAUTION (1 bar)
        dec, state = default_engine.evaluate(p_c, w_c, a_c, e_c, b_c, state)
        history_tiers.append(state.tier)
        assert dec.action_tier == CircuitBreakerTier.CAUTION
        assert dec.execution_haircut <= 0.50

        # 3. DERISK (1 bar)
        dec, state = default_engine.evaluate(p_d, w_d, a_d, e_d, b_d, state)
        history_tiers.append(state.tier)
        assert dec.action_tier == CircuitBreakerTier.DERISK
        assert dec.execution_haircut == 0.0

        # 4. HALT (1 bar)
        dec, state = default_engine.evaluate(p_h, w_h, a_h, e_h, b_h, state)
        history_tiers.append(state.tier)
        assert dec.action_tier == CircuitBreakerTier.HALT
        assert dec.execution_haircut == 0.0

        # 5. Locked in HALT for 4 more bars (total 5 bars in HALT) under quiescent conditions
        for _ in range(4):
            dec, state = default_engine.evaluate(p_q, w_q, a_q, e_q, b_q, state)
            history_tiers.append(state.tier)
            assert dec.action_tier == CircuitBreakerTier.HALT
            assert dec.execution_haircut == 0.0

        # 6. De-escalates to DERISK (step down one tier, resets active to 1)
        dec, state = default_engine.evaluate(p_q, w_q, a_q, e_q, b_q, state)
        history_tiers.append(state.tier)
        assert dec.action_tier == CircuitBreakerTier.DERISK
        assert state.active_bars_in_tier == 1

        # 7. Locked in DERISK for 4 more bars (total 5 bars in DERISK)
        for _ in range(4):
            dec, state = default_engine.evaluate(p_q, w_q, a_q, e_q, b_q, state)
            history_tiers.append(state.tier)
            assert dec.action_tier == CircuitBreakerTier.DERISK

        # 8. De-escalates to CAUTION (step down one tier, resets active to 1)
        dec, state = default_engine.evaluate(p_q, w_q, a_q, e_q, b_q, state)
        history_tiers.append(state.tier)
        assert dec.action_tier == CircuitBreakerTier.CAUTION
        assert state.active_bars_in_tier == 1
        assert dec.execution_haircut <= 0.50

        # 9. De-escalates to NORMAL (immediate since Xi < 0.30)
        dec, state = default_engine.evaluate(p_q, w_q, a_q, e_q, b_q, state)
        history_tiers.append(state.tier)
        assert dec.action_tier == CircuitBreakerTier.NORMAL
        assert state.active_bars_in_tier == 1
        assert dec.execution_haircut == 1.0

        expected_sequence = [
            CircuitBreakerTier.NORMAL,
            CircuitBreakerTier.NORMAL,
            CircuitBreakerTier.CAUTION,
            CircuitBreakerTier.DERISK,
            CircuitBreakerTier.HALT,
            CircuitBreakerTier.HALT,
            CircuitBreakerTier.HALT,
            CircuitBreakerTier.HALT,
            CircuitBreakerTier.HALT,
            CircuitBreakerTier.DERISK,
            CircuitBreakerTier.DERISK,
            CircuitBreakerTier.DERISK,
            CircuitBreakerTier.DERISK,
            CircuitBreakerTier.DERISK,
            CircuitBreakerTier.CAUTION,
            CircuitBreakerTier.NORMAL,
        ]
        assert history_tiers == expected_sequence
        assert state.step_index == 16


class TestCircuitBreakerMasterIntegration:
    """Master integration tests for Circuit Breaker Overlays subsystem (Phase 5 Step 2 Task 5).

    Covers:
    - evaluate_prediction integration with synthetic and real EnsemblePrediction.
    - Automatic panic regime derivation and explicit override.
    - Strict input validation and defensive error trapping.
    - 50-bar rolling multi-regime simulation validating invariant preservation:
      * INV-CB-001: continuous_haircut and execution_haircut in [0.0, 1.0].
      * INV-CB-002: action_tier is strictly a CircuitBreakerTier.
      * INV-CB-003: Hysteresis dwell-time and recovery barriers enforced.
      * INV-CB-004: Directional consensus probabilities strictly sum to 1.0.
      * INV-CB-005: Non-finite inputs trap.
      * Lookahead-free causal flow: step_index incrementing monotonically.
    - Cross-subsystem end-to-end pipeline with RegimeConditionedDMAEngine.
    - Benchmark SLA: K=100 models evaluation latency <= 0.20ms (INV-CB-006).
    - Public export of all 10 domain symbols in quant.analytics.__init__.py and __all__.
    """

    def test_public_symbol_exports(self) -> None:
        """Verify all 10 circuit breaker domain symbols are exported in quant.analytics.__init__ and __all__."""
        import quant.analytics as qa

        expected_symbols = [
            "CircuitBreakerTier",
            "CircuitBreakerConfig",
            "CircuitBreakerState",
            "CircuitBreakerDecision",
            "CircuitBreakerError",
            "DegenerateCircuitBreakerException",
            "InvalidCircuitBreakerInputException",
            "EpistemicEntropyCalculator",
            "ContinuousHaircutCalculator",
            "CircuitBreakerOverlayEngine",
        ]

        for sym in expected_symbols:
            assert hasattr(qa, sym), f"quant.analytics is missing exported symbol '{sym}'"
            assert sym in qa.__all__, f"quant.analytics.__all__ is missing symbol '{sym}'"
            exported_obj = getattr(qa, sym)
            assert exported_obj is not None

    def test_evaluate_prediction_with_synthetic_ensemble_prediction(self) -> None:
        """Verify evaluate_prediction extracts properties and matches direct evaluate output."""
        from quant.analytics.ensemble import EnsemblePrediction

        engine = CircuitBreakerOverlayEngine()
        state = engine.initialize_state()

        weights = np.array([0.4, 0.4, 0.2], dtype=np.float64)
        preds = np.array([0.02, -0.01, 0.00005], dtype=np.float64)
        aleatoric = 0.0004
        epistemic = 0.0001

        prediction = EnsemblePrediction(
            point_prediction=0.005,
            aleatoric_variance=aleatoric,
            epistemic_variance=epistemic,
            total_variance=aleatoric + epistemic,
            model_weights=weights,
            regime_probabilities=np.array([0.8, 0.1, 0.1], dtype=np.float64),
            effective_models=2.8,
            volatility_forgetting_factor=0.96,
            ambiguity_shrinkage_weight=0.05,
        )

        dec, next_state = engine.evaluate_prediction(
            prediction=prediction,
            predictions=preds,
            ambiguity_beta=1.5,
            state=state,
            cusum_shock=False,
            regime_is_panic=None,
        )

        # Direct evaluation comparison
        expected_dec, expected_state = engine.evaluate(
            predictions=preds,
            weights=weights,
            aleatoric_variance=aleatoric,
            epistemic_variance=epistemic,
            ambiguity_beta=1.5,
            state=state,
            cusum_shock=False,
            regime_is_panic=False,  # argmax([0.8, 0.1, 0.1]) == 0 != 2
        )

        assert dec.action_tier == expected_dec.action_tier
        assert dec.execution_haircut == expected_dec.execution_haircut
        assert dec.is_halted == expected_dec.is_halted
        assert dec.is_throttled == expected_dec.is_throttled
        assert next_state.tier == expected_state.tier
        assert next_state.active_bars_in_tier == expected_state.active_bars_in_tier
        assert math.isclose(next_state.continuous_haircut, expected_state.continuous_haircut)
        assert math.isclose(next_state.epistemic_entropy, expected_state.epistemic_entropy)
        assert math.isclose(next_state.directional_entropy, expected_state.directional_entropy)
        assert math.isclose(next_state.epistemic_ratio, expected_state.epistemic_ratio)
        assert math.isclose(next_state.composite_shock_score, expected_state.composite_shock_score)
        assert np.allclose(
            next_state.directional_probabilities, expected_state.directional_probabilities
        )
        assert next_state.step_index == expected_state.step_index
        assert next_state.step_index == 1
        assert 0.0 <= dec.execution_haircut <= 1.0

    def test_evaluate_prediction_panic_derivation_from_regime_probabilities(self) -> None:
        """Verify automatic derivation of panic regime from regime_probabilities argmax == 2."""
        from quant.analytics.ensemble import EnsemblePrediction

        engine = CircuitBreakerOverlayEngine()
        state = engine.initialize_state()

        weights = np.array([0.5, 0.5], dtype=np.float64)
        preds = np.array([0.001, 0.001], dtype=np.float64)

        # 1. Regime 2 dominant -> panic derived as True; with cusum_shock=True triggers emergency HALT
        pred_panic = EnsemblePrediction(
            point_prediction=0.001,
            aleatoric_variance=0.0004,
            epistemic_variance=0.0001,
            total_variance=0.0005,
            model_weights=weights,
            regime_probabilities=np.array([0.1, 0.1, 0.8], dtype=np.float64),
            effective_models=2.0,
            volatility_forgetting_factor=0.95,
            ambiguity_shrinkage_weight=0.1,
        )
        dec_panic, state_panic = engine.evaluate_prediction(
            prediction=pred_panic,
            predictions=preds,
            ambiguity_beta=1.0,
            state=state,
            cusum_shock=True,
            regime_is_panic=None,
        )
        assert dec_panic.action_tier == CircuitBreakerTier.HALT
        assert dec_panic.is_halted is True
        assert state_panic.tier == CircuitBreakerTier.HALT

        # 2. Regime 1 dominant (Volatile, not Crisis) -> panic derived as False; cusum_shock does not force HALT
        pred_volatile = EnsemblePrediction(
            point_prediction=0.001,
            aleatoric_variance=0.0004,
            epistemic_variance=0.0001,
            total_variance=0.0005,
            model_weights=weights,
            regime_probabilities=np.array([0.1, 0.8, 0.1], dtype=np.float64),
            effective_models=2.0,
            volatility_forgetting_factor=0.95,
            ambiguity_shrinkage_weight=0.1,
        )
        dec_vol, state_vol = engine.evaluate_prediction(
            prediction=pred_volatile,
            predictions=preds,
            ambiguity_beta=1.0,
            state=state,
            cusum_shock=True,
            regime_is_panic=None,
        )
        assert dec_vol.action_tier == CircuitBreakerTier.NORMAL
        assert dec_vol.is_halted is False

        # 3. Explicit override: regime_is_panic=False overrides panic regime probabilities
        dec_override_false, _ = engine.evaluate_prediction(
            prediction=pred_panic,
            predictions=preds,
            ambiguity_beta=1.0,
            state=state,
            cusum_shock=True,
            regime_is_panic=False,
        )
        assert dec_override_false.action_tier == CircuitBreakerTier.NORMAL

        # 4. Explicit override: regime_is_panic=True overrides non-panic regime probabilities
        dec_override_true, _ = engine.evaluate_prediction(
            prediction=pred_volatile,
            predictions=preds,
            ambiguity_beta=1.0,
            state=state,
            cusum_shock=True,
            regime_is_panic=True,
        )
        assert dec_override_true.action_tier == CircuitBreakerTier.HALT

    def test_evaluate_prediction_input_validation(self) -> None:
        """Verify defensive error handling in evaluate_prediction."""
        from quant.analytics.ensemble import EnsemblePrediction

        engine = CircuitBreakerOverlayEngine()
        state = engine.initialize_state()

        weights = np.array([0.5, 0.5], dtype=np.float64)
        preds = np.array([0.001, 0.001], dtype=np.float64)
        valid_pred = EnsemblePrediction(
            point_prediction=0.001,
            aleatoric_variance=0.0004,
            epistemic_variance=0.0001,
            total_variance=0.0005,
            model_weights=weights,
            regime_probabilities=np.array([0.7, 0.2, 0.1], dtype=np.float64),
            effective_models=2.0,
            volatility_forgetting_factor=0.95,
            ambiguity_shrinkage_weight=0.1,
        )

        # 1. Invalid prediction object type
        with pytest.raises(
            InvalidCircuitBreakerInputException,
            match="prediction must be an instance of EnsemblePrediction",
        ):
            engine.evaluate_prediction(
                prediction="not_a_prediction",  # type: ignore[arg-type]
                predictions=preds,
                ambiguity_beta=1.0,
                state=state,
            )

        # 2. Invalid regime_is_panic type
        with pytest.raises(
            InvalidCircuitBreakerInputException,
            match="regime_is_panic must be a boolean or None",
        ):
            engine.evaluate_prediction(
                prediction=valid_pred,
                predictions=preds,
                ambiguity_beta=1.0,
                state=state,
                regime_is_panic="true",  # type: ignore[arg-type]
            )

        with pytest.raises(
            InvalidCircuitBreakerInputException,
            match="regime_is_panic must be a boolean or None",
        ):
            engine.evaluate_prediction(
                prediction=valid_pred,
                predictions=preds,
                ambiguity_beta=1.0,
                state=state,
                regime_is_panic=1,  # type: ignore[arg-type]
            )

        # 3. Shape mismatch between predictions and prediction.model_weights
        with pytest.raises(
            InvalidCircuitBreakerInputException,
            match="must match weights",
        ):
            engine.evaluate_prediction(
                prediction=valid_pred,
                predictions=np.array([0.001, 0.002, 0.003], dtype=np.float64),
                ambiguity_beta=1.0,
                state=state,
            )

        # 4. Non-finite predictions
        with pytest.raises(DegenerateCircuitBreakerException, match="non-finite"):
            engine.evaluate_prediction(
                prediction=valid_pred,
                predictions=np.array([np.nan, 0.001], dtype=np.float64),
                ambiguity_beta=1.0,
                state=state,
            )

        # 5. Non-finite ambiguity_beta
        with pytest.raises(DegenerateCircuitBreakerException, match="INV-CB-005"):
            engine.evaluate_prediction(
                prediction=valid_pred,
                predictions=preds,
                ambiguity_beta=float("inf"),
                state=state,
            )

    def test_evaluate_prediction_with_real_rd_dma_engine(self) -> None:
        """Verify cross-subsystem integration between RegimeConditionedDMAEngine and evaluate_prediction."""
        from quant.analytics.ensemble import RegimeConditionedDMAEngine

        dma_engine = RegimeConditionedDMAEngine()
        cb_engine = CircuitBreakerOverlayEngine()

        K = 5
        dma_state = dma_engine.initialize_state(n_models=K)
        cb_state = cb_engine.initialize_state()

        P_trans = np.array(
            [
                [0.8, 0.15, 0.05],
                [0.1, 0.8, 0.1],
                [0.05, 0.15, 0.8],
            ],
            dtype=np.float64,
        )

        preds = np.array([0.01, 0.012, -0.008, 0.015, -0.005], dtype=np.float64)
        variances = np.full(K, 0.0004, dtype=np.float64)
        regime_probs = np.array([0.7, 0.2, 0.1], dtype=np.float64)

        ens_pred, dma_state = dma_engine.predict_and_update(
            predictions=preds,
            variances=variances,
            realized_return=0.005,
            realized_vol=0.015,
            regime_probs=regime_probs,
            transition_matrix=P_trans,
            ambiguity_beta=1.2,
            state=dma_state,
        )

        cb_dec, cb_state = cb_engine.evaluate_prediction(
            prediction=ens_pred,
            predictions=preds,
            ambiguity_beta=1.2,
            state=cb_state,
        )

        assert isinstance(cb_dec, CircuitBreakerDecision)
        assert isinstance(cb_state, CircuitBreakerState)
        assert cb_state.step_index == 1
        assert 0.0 <= cb_dec.execution_haircut <= 1.0
        assert cb_dec.action_tier in (
            CircuitBreakerTier.NORMAL,
            CircuitBreakerTier.CAUTION,
            CircuitBreakerTier.DERISK,
            CircuitBreakerTier.HALT,
        )
        assert math.isclose(float(np.sum(cb_state.directional_probabilities)), 1.0, abs_tol=1e-10)

    def test_rolling_50_bar_simulation_across_regimes(self) -> None:
        """50-bar rolling simulation across multiple regimes verifying stability, causal flow, and invariants."""
        from quant.analytics.ensemble import EnsemblePrediction

        engine = CircuitBreakerOverlayEngine(
            CircuitBreakerConfig(dwell_time_bars=5, recovery_threshold=0.30)
        )
        state = engine.initialize_state()

        rng = np.random.default_rng(123)
        K = 6
        step_indices: list[int] = []
        action_tiers: list[CircuitBreakerTier] = []

        for bar in range(1, 51):
            if bar <= 15:
                # Regime 0: Quiescent low-vol bull
                preds = rng.normal(0.01, 0.001, size=K)  # strong positive consensus
                weights = np.full(K, 1.0 / K)
                aleatoric = 0.0001
                epistemic = 0.00001
                beta = 1.0
                regime_probs = np.array([0.90, 0.08, 0.02])
                cusum = False
            elif bar <= 25:
                # Regime 1: Moderate volatility surge / disagreement
                preds = np.array([0.02, -0.02, 0.015, -0.018, 0.005, -0.006])  # polarization
                weights = np.full(K, 1.0 / K)
                aleatoric = 0.0005
                epistemic = 0.0015  # high epistemic ratio
                beta = 2.5
                regime_probs = np.array([0.20, 0.70, 0.10])
                cusum = False
            elif bar <= 30:
                # Regime 2: Severe panic crisis
                preds = rng.normal(-0.05, 0.02, size=K)
                weights = np.full(K, 1.0 / K)
                aleatoric = 0.002
                epistemic = 0.005
                beta = 4.5
                regime_probs = np.array([0.05, 0.10, 0.85])  # Panic regime
                cusum = True  # CUSUM jump shock
            elif bar <= 42:
                # Tranquil recovery attempt, but within hysteresis dwell window
                preds = rng.normal(0.008, 0.0005, size=K)  # strong positive consensus
                weights = np.full(K, 1.0 / K)
                aleatoric = 0.0001
                epistemic = 0.000001
                beta = 1.0
                regime_probs = np.array([0.85, 0.10, 0.05])
                cusum = False
            else:
                # Continued tranquil conditions
                preds = rng.normal(0.008, 0.0005, size=K)
                weights = np.full(K, 1.0 / K)
                aleatoric = 0.0001
                epistemic = 0.000001
                beta = 1.0
                regime_probs = np.array([0.90, 0.08, 0.02])
                cusum = False

            pred_payload = EnsemblePrediction(
                point_prediction=float(np.mean(preds)),
                aleatoric_variance=aleatoric,
                epistemic_variance=epistemic,
                total_variance=aleatoric + epistemic,
                model_weights=weights,
                regime_probabilities=regime_probs,
                effective_models=float(K),
                volatility_forgetting_factor=0.95,
                ambiguity_shrinkage_weight=0.1,
            )

            dec, state = engine.evaluate_prediction(
                prediction=pred_payload,
                predictions=preds,
                ambiguity_beta=beta,
                state=state,
                cusum_shock=cusum,
            )

            # Invariant checks per bar
            # INV-CB-001
            assert 0.0 <= state.continuous_haircut <= 1.0
            assert 0.0 <= dec.execution_haircut <= 1.0
            # INV-CB-002
            assert isinstance(dec.action_tier, CircuitBreakerTier)
            assert isinstance(state.tier, CircuitBreakerTier)
            assert dec.action_tier == state.tier
            # INV-CB-004
            assert state.directional_probabilities.shape == (3,)
            assert math.isclose(float(np.sum(state.directional_probabilities)), 1.0, abs_tol=1e-10)
            assert np.all(state.directional_probabilities >= 0.0)
            # INV-CB-005
            assert math.isfinite(state.epistemic_entropy)
            assert math.isfinite(state.directional_entropy)
            assert math.isfinite(state.epistemic_ratio)
            assert math.isfinite(state.composite_shock_score)
            assert math.isfinite(dec.execution_haircut)
            # Strict causal ordering (no lookahead, monotonic step_index)
            assert state.step_index == bar

            step_indices.append(state.step_index)
            action_tiers.append(dec.action_tier)

        assert step_indices == list(range(1, 51))
        # Bar 26 was crisis with cusum shock and panic -> must be HALT
        assert action_tiers[25] == CircuitBreakerTier.HALT
        # Bars 26 to 30 were in crisis -> HALT sustained
        for b in range(25, 30):
            assert action_tiers[b] == CircuitBreakerTier.HALT
        # Hysteresis lockout (INV-CB-003): at bar 31 (bar index 30), quiescent resumes.
        # But dwell_time_bars is 5, and state was sustained in HALT.
        # At bar 31: active_bars_in_tier was 5 during crisis (bars 26-30).
        # At bar 31, dwell requirement (>= 5) is met and shock < 0.30 -> de-escalates to DERISK (not straight to NORMAL!).
        assert action_tiers[30] == CircuitBreakerTier.DERISK
        # Locked in DERISK for 5 bars: bars 31 (index 30), 32 (31), 33 (32), 34 (33), 35 (34).
        for b in range(30, 35):
            assert action_tiers[b] == CircuitBreakerTier.DERISK
        # At bar 36 (index 35), steps down to CAUTION
        assert action_tiers[35] == CircuitBreakerTier.CAUTION
        # At bar 37 (index 36), CAUTION de-escalates to NORMAL
        assert action_tiers[36] == CircuitBreakerTier.NORMAL

    def test_cross_subsystem_rolling_pipeline_with_dma_engine(self) -> None:
        """50-bar end-to-end rolling pipeline integration with RegimeConditionedDMAEngine."""
        from quant.analytics.ensemble import RegimeConditionedDMAEngine

        dma_engine = RegimeConditionedDMAEngine()
        cb_engine = CircuitBreakerOverlayEngine(
            CircuitBreakerConfig(dwell_time_bars=3, recovery_threshold=0.30)
        )

        K = 4
        dma_state = dma_engine.initialize_state(n_models=K)
        cb_state = cb_engine.initialize_state()

        P_trans = np.array(
            [
                [0.85, 0.10, 0.05],
                [0.10, 0.80, 0.10],
                [0.05, 0.15, 0.80],
            ],
            dtype=np.float64,
        )

        rng = np.random.default_rng(999)
        regime_probs = np.array([0.70, 0.20, 0.10], dtype=np.float64)

        for bar in range(1, 51):
            preds = rng.normal(0.002, 0.01, size=K)
            variances = np.full(K, 0.0004)
            ret = float(rng.normal(0.001, 0.01))
            vol = float(max(0.001, rng.normal(0.015, 0.002)))

            ens_pred, dma_state = dma_engine.predict_and_update(
                predictions=preds,
                variances=variances,
                realized_return=ret,
                realized_vol=vol,
                regime_probs=regime_probs,
                transition_matrix=P_trans,
                ambiguity_beta=1.2,
                state=dma_state,
            )

            # Update regime_probs using ens_pred forward prior for next bar
            regime_probs = ens_pred.regime_probabilities

            dec, cb_state = cb_engine.evaluate_prediction(
                prediction=ens_pred,
                predictions=preds,
                ambiguity_beta=1.2,
                state=cb_state,
            )

            assert cb_state.step_index == bar
            assert 0.0 <= dec.execution_haircut <= 1.0
            assert dec.action_tier in CircuitBreakerTier
            assert math.isclose(
                float(np.sum(cb_state.directional_probabilities)), 1.0, abs_tol=1e-10
            )

    def test_benchmark_sla_k100_sub_020ms(self) -> None:
        """Verify Benchmark SLA: K=100 models evaluation latency <= 0.20ms (INV-CB-006)."""
        import gc
        import sys
        import time

        from quant.analytics.ensemble import EnsemblePrediction

        if sys.gettrace() is not None:
            pytest.skip("Skipping performance benchmark SLA test under active tracer/profiler")

        engine = CircuitBreakerOverlayEngine()
        state = engine.initialize_state()

        K = 100
        rng = np.random.default_rng(42)
        raw_preds = rng.normal(0.001, 0.005, size=K)
        weights = rng.uniform(0.1, 1.0, size=K)
        weights /= np.sum(weights)

        prediction = EnsemblePrediction(
            point_prediction=float(np.sum(weights * raw_preds)),
            aleatoric_variance=0.0004,
            epistemic_variance=0.0002,
            total_variance=0.0006,
            model_weights=weights,
            regime_probabilities=np.array([0.6, 0.3, 0.1], dtype=np.float64),
            effective_models=float(1.0 / np.sum(weights**2)),
            volatility_forgetting_factor=0.95,
            ambiguity_shrinkage_weight=0.1,
        )

        # Warmup (15 iterations) to prime CPU cache, JIT/interpreter, and branch predictors
        for _ in range(15):
            _, state = engine.evaluate_prediction(
                prediction=prediction,
                predictions=raw_preds,
                ambiguity_beta=1.5,
                state=state,
            )

        gc.collect()

        # Timed benchmark: 50 iterations
        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            _, state = engine.evaluate_prediction(
                prediction=prediction,
                predictions=raw_preds,
                ambiguity_beta=1.5,
                state=state,
            )
            latencies.append(time.perf_counter() - t0)

        median_latency_ms = float(np.median(latencies)) * 1000.0
        assert median_latency_ms <= 0.20, (
            f"INV-CB-006 Benchmark SLA violated: median latency {median_latency_ms:.4f}ms > 0.20ms for K={K}"
        )

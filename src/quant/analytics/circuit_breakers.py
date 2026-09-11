"""Epistemic Disagreement Entropy & Circuit Breaker Overlays subsystem.

Provides domain entities, enums, configuration, state representations, and invariant contracts
for institutional-grade continuous haircutting and multi-tier circuit breaker overlays.

Invariants Enforced:
- INV-CB-001 (Bounded Continuous Haircut): kappa_t in [0.0, 1.0] and execution_haircut in [0.0, 1.0].
- INV-CB-002 (Valid Discrete Tier): Discrete state is strictly one of the 4 defined institutional tiers
  (NORMAL, CAUTION, DERISK, HALT) with ordered severity.
- INV-CB-004 (Strict Simplex Conservation on Directional Probabilities):
  sum_{s in {+, 0, -}} p_s == 1.0 +/- 1e-10, p_s >= 0.0, shape (3,).
- INV-CB-005 (Non-Finite Input Data Protection):
  Immediate defensive failure on non-finite values (NaN / Inf).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class CircuitBreakerError(Exception):
    """Base exception for all circuit breaker overlay errors and invariant violations."""


class DegenerateCircuitBreakerException(CircuitBreakerError):
    """Raised when non-finite inputs, NaN/Inf, or degenerate bounds are detected."""


class InvalidCircuitBreakerInputException(CircuitBreakerError):
    """Raised when dimensionality mismatch, out-of-bounds probabilities, or threshold hierarchy violations occur."""


class CircuitBreakerTier(IntEnum):
    """Institutional discrete risk tiers for circuit breaker execution overlay.

    Ordered by severity:
        NORMAL (0): Standard operations, full alpha execution.
        CAUTION (1): Tier 1 warning, throttled execution sizing (kappa <= 0.50).
        DERISK (2): Tier 2 de-leveraging, orderly liquidation into cash (kappa = 0.0).
        HALT (3): Tier 3 emergency trading halt, zero execution, active orders cancelled.
    """

    NORMAL = 0
    CAUTION = 1
    DERISK = 2
    HALT = 3


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Hyperparameter configuration and threshold barriers for Circuit Breaker Overlays.

    Enforces boundary validations and threshold hierarchy contracts.

    Attributes:
        sign_threshold: Directional deadband threshold delta_sign > 0.0 (default 1e-4).
        caution_threshold: Tier 1 caution threshold theta_caution in (0.0, derisk_threshold) (default 0.45).
        derisk_threshold: Tier 2 de-risk threshold theta_derisk in (caution_threshold, halt_threshold) (default 0.70).
        halt_threshold: Tier 3 emergency halt threshold theta_halt in (derisk_threshold, 1.0] (default 0.90).
        recovery_threshold: Hysteresis recovery barrier theta_recovery in (0.0, caution_threshold) (default 0.30).
        dwell_time_bars: Minimum bars to remain locked in DERISK/HALT before recovery tau_dwell >= 1 (default 5).
        haircut_steepness: Logistic sigmoid steepness coefficient k_steep > 0.0 (default 10.0).
        haircut_midpoint: Logistic sigmoid midpoint inflection Xi_mid in (0.0, 1.0) (default 0.50).
        weight_entropy: Composite shock weight for epistemic entropy omega_H >= 0.0 (default 0.50).
        weight_epistemic_ratio: Composite shock weight for epistemic ratio omega_rho >= 0.0 (default 0.30).
        weight_ambiguity: Composite shock weight for thermodynamic ambiguity omega_beta >= 0.0 (default 0.20).
        beta_min: Macroeconomic ambiguity lower normalization bound beta_min > 0.0 (default 1.0).
        beta_max: Macroeconomic ambiguity upper normalization bound beta_max > beta_min (default 5.0).
    """

    sign_threshold: float = 1e-4
    caution_threshold: float = 0.45
    derisk_threshold: float = 0.70
    halt_threshold: float = 0.90
    recovery_threshold: float = 0.30
    dwell_time_bars: int = 5
    haircut_steepness: float = 10.0
    haircut_midpoint: float = 0.50
    weight_entropy: float = 0.50
    weight_epistemic_ratio: float = 0.30
    weight_ambiguity: float = 0.20
    beta_min: float = 1.0
    beta_max: float = 5.0

    def __post_init__(self) -> None:
        """Validate hyperparameter contracts and boundary invariants."""
        # Finiteness check
        for field_name in (
            "sign_threshold",
            "caution_threshold",
            "derisk_threshold",
            "halt_threshold",
            "recovery_threshold",
            "haircut_steepness",
            "haircut_midpoint",
            "weight_entropy",
            "weight_epistemic_ratio",
            "weight_ambiguity",
            "beta_min",
            "beta_max",
        ):
            val = getattr(self, field_name)
            if not (isinstance(val, (int, float)) and math.isfinite(val)):
                raise DegenerateCircuitBreakerException(
                    f"{field_name} must be a finite float, got {val}"
                )

        if (
            not isinstance(self.dwell_time_bars, int)
            or isinstance(self.dwell_time_bars, bool)
            or self.dwell_time_bars < 1
        ):
            raise DegenerateCircuitBreakerException(
                f"dwell_time_bars must be an integer >= 1, got {self.dwell_time_bars}"
            )

        if self.sign_threshold <= 0.0:
            raise DegenerateCircuitBreakerException(
                f"sign_threshold must be strictly positive, got {self.sign_threshold}"
            )

        if self.haircut_steepness <= 0.0:
            raise DegenerateCircuitBreakerException(
                f"haircut_steepness must be strictly positive, got {self.haircut_steepness}"
            )

        if self.beta_min <= 0.0:
            raise DegenerateCircuitBreakerException(
                f"beta_min must be strictly positive, got {self.beta_min}"
            )

        if self.beta_max <= self.beta_min:
            raise InvalidCircuitBreakerInputException(
                f"beta_max ({self.beta_max}) must be strictly greater than beta_min ({self.beta_min})"
            )

        if (
            self.weight_entropy < 0.0
            or self.weight_epistemic_ratio < 0.0
            or self.weight_ambiguity < 0.0
        ):
            raise DegenerateCircuitBreakerException(
                "Component weights must be non-negative: "
                f"weight_entropy={self.weight_entropy}, "
                f"weight_epistemic_ratio={self.weight_epistemic_ratio}, "
                f"weight_ambiguity={self.weight_ambiguity}"
            )

        total_weight = self.weight_entropy + self.weight_epistemic_ratio + self.weight_ambiguity
        if abs(total_weight - 1.0) > 1e-6:
            raise InvalidCircuitBreakerInputException(
                f"Component weights must sum to 1.0 +/- 1e-6, got {total_weight}"
            )

        if not (0.0 < self.haircut_midpoint < 1.0):
            raise InvalidCircuitBreakerInputException(
                f"haircut_midpoint must be in (0.0, 1.0), got {self.haircut_midpoint}"
            )

        if not (0.0 < self.recovery_threshold < self.caution_threshold):
            raise InvalidCircuitBreakerInputException(
                f"recovery_threshold ({self.recovery_threshold}) must be in (0.0, caution_threshold={self.caution_threshold})"
            )

        if not (0.0 < self.caution_threshold < self.derisk_threshold < self.halt_threshold <= 1.0):
            raise InvalidCircuitBreakerInputException(
                "Threshold hierarchy violation: must satisfy 0.0 < caution < derisk < halt <= 1.0. "
                f"Got caution={self.caution_threshold}, derisk={self.derisk_threshold}, halt={self.halt_threshold}"
            )


@dataclass(frozen=True)
class CircuitBreakerState:
    """Immutable state machine snapshot for Circuit Breaker Overlays.

    Enforces Invariants INV-CB-001, INV-CB-002, and INV-CB-004.

    Attributes:
        tier: Current operational risk tier.
        active_bars_in_tier: Number of continuous bars spent in the current tier.
        continuous_haircut: Smooth position sizing multiplier kappa_t in [0.0, 1.0].
        epistemic_entropy: Normalized composite epistemic disagreement entropy in [0.0, 1.0].
        directional_entropy: Normalized Shannon directional consensus entropy in [0.0, 1.0].
        epistemic_ratio: Fraction of predictive variance from model disagreement in [0.0, 1.0].
        composite_shock_score: Thermodynamic shock score Xi_t in [0.0, 1.0].
        directional_probabilities: 3-simplex consensus vector [p_+, p_-, p_0] summing to 1.0 +/- 1e-10.
        step_index: Monotonically increasing bar/event sequence index >= 0.
    """

    tier: CircuitBreakerTier
    active_bars_in_tier: int
    continuous_haircut: float
    epistemic_entropy: float
    directional_entropy: float
    epistemic_ratio: float
    composite_shock_score: float
    directional_probabilities: np.ndarray
    step_index: int

    def __post_init__(self) -> None:
        """Validate invariant contracts on state variables."""
        if not isinstance(self.tier, CircuitBreakerTier):
            raise InvalidCircuitBreakerInputException(
                f"tier must be an instance of CircuitBreakerTier, got {self.tier}"
            )

        if (
            not isinstance(self.active_bars_in_tier, int)
            or isinstance(self.active_bars_in_tier, bool)
            or self.active_bars_in_tier < 0
        ):
            raise InvalidCircuitBreakerInputException(
                f"active_bars_in_tier must be a non-negative integer, got {self.active_bars_in_tier}"
            )

        if (
            not isinstance(self.step_index, int)
            or isinstance(self.step_index, bool)
            or self.step_index < 0
        ):
            raise InvalidCircuitBreakerInputException(
                f"step_index must be a non-negative integer, got {self.step_index}"
            )

        for field_name in (
            "continuous_haircut",
            "epistemic_entropy",
            "directional_entropy",
            "epistemic_ratio",
            "composite_shock_score",
        ):
            val = getattr(self, field_name)
            if not (isinstance(val, (int, float)) and math.isfinite(val)):
                raise DegenerateCircuitBreakerException(
                    f"{field_name} must be a finite float, got {val}"
                )

        if not (0.0 <= self.continuous_haircut <= 1.0):
            raise InvalidCircuitBreakerInputException(
                f"INV-CB-001: continuous_haircut must be in [0.0, 1.0], got {self.continuous_haircut}"
            )

        if not (0.0 <= self.epistemic_entropy <= 1.0):
            raise InvalidCircuitBreakerInputException(
                f"epistemic_entropy must be in [0.0, 1.0], got {self.epistemic_entropy}"
            )

        if not (0.0 <= self.directional_entropy <= 1.0):
            raise InvalidCircuitBreakerInputException(
                f"directional_entropy must be in [0.0, 1.0], got {self.directional_entropy}"
            )

        if not (0.0 <= self.epistemic_ratio <= 1.0):
            raise InvalidCircuitBreakerInputException(
                f"epistemic_ratio must be in [0.0, 1.0], got {self.epistemic_ratio}"
            )

        if not (0.0 <= self.composite_shock_score <= 1.0):
            raise InvalidCircuitBreakerInputException(
                f"composite_shock_score must be in [0.0, 1.0], got {self.composite_shock_score}"
            )

        # INV-CB-004 validation
        if not isinstance(self.directional_probabilities, np.ndarray):
            raise InvalidCircuitBreakerInputException(
                f"directional_probabilities must be a NumPy array, got {type(self.directional_probabilities)}"
            )

        if self.directional_probabilities.shape != (3,):
            raise InvalidCircuitBreakerInputException(
                f"directional_probabilities must have shape (3,), got {self.directional_probabilities.shape}"
            )

        if not np.all(np.isfinite(self.directional_probabilities)):
            raise DegenerateCircuitBreakerException(
                "directional_probabilities contains non-finite values (NaN or Inf)"
            )

        if np.any(self.directional_probabilities < 0.0):
            raise InvalidCircuitBreakerInputException(
                f"directional_probabilities components must be non-negative, got {self.directional_probabilities}"
            )

        prob_sum = float(np.sum(self.directional_probabilities))
        if abs(prob_sum - 1.0) > 1e-10:
            raise InvalidCircuitBreakerInputException(
                f"directional_probabilities must sum to 1.0 +/- 1e-10, got {prob_sum}"
            )

        # Protect underlying NumPy array against external mutation
        self.directional_probabilities.flags.writeable = False


@dataclass(frozen=True)
class CircuitBreakerDecision:
    """Execution decision emitted to the trading engine.

    Enforces Invariant INV-CB-001 and consistency between tier and status booleans.

    Attributes:
        action_tier: Prescribed circuit breaker action tier.
        execution_haircut: Multiplier in [0.0, 1.0] applied to target position size.
        is_halted: True if action_tier is HALT.
        is_derisking: True if action_tier is DERISK.
        is_throttled: True if action_tier is CAUTION.
        state: Full CircuitBreakerState snapshot.
    """

    action_tier: CircuitBreakerTier
    execution_haircut: float
    is_halted: bool
    is_derisking: bool
    is_throttled: bool
    state: CircuitBreakerState

    def __post_init__(self) -> None:
        """Validate invariant contracts on execution decision."""
        if not isinstance(self.action_tier, CircuitBreakerTier):
            raise InvalidCircuitBreakerInputException(
                f"action_tier must be an instance of CircuitBreakerTier, got {self.action_tier}"
            )

        if not (
            isinstance(self.execution_haircut, (int, float))
            and math.isfinite(self.execution_haircut)
        ):
            raise DegenerateCircuitBreakerException(
                f"execution_haircut must be a finite float, got {self.execution_haircut}"
            )

        if not (0.0 <= self.execution_haircut <= 1.0):
            raise InvalidCircuitBreakerInputException(
                f"INV-CB-001: execution_haircut must be in [0.0, 1.0], got {self.execution_haircut}"
            )

        if not isinstance(self.state, CircuitBreakerState):
            raise InvalidCircuitBreakerInputException(
                f"state must be an instance of CircuitBreakerState, got {type(self.state)}"
            )

        expected_halt = self.action_tier == CircuitBreakerTier.HALT
        expected_derisk = self.action_tier == CircuitBreakerTier.DERISK
        expected_throttled = self.action_tier == CircuitBreakerTier.CAUTION

        if self.is_halted != expected_halt:
            raise InvalidCircuitBreakerInputException(
                f"is_halted ({self.is_halted}) must match action_tier ({self.action_tier})"
            )

        if self.is_derisking != expected_derisk:
            raise InvalidCircuitBreakerInputException(
                f"is_derisking ({self.is_derisking}) must match action_tier ({self.action_tier})"
            )

        if self.is_throttled != expected_throttled:
            raise InvalidCircuitBreakerInputException(
                f"is_throttled ({self.is_throttled}) must match action_tier ({self.action_tier})"
            )

    @classmethod
    def from_state(
        cls,
        action_tier: CircuitBreakerTier,
        execution_haircut: float,
        state: CircuitBreakerState,
    ) -> CircuitBreakerDecision:
        """Factory constructor ensuring consistent status booleans."""
        return cls(
            action_tier=action_tier,
            execution_haircut=execution_haircut,
            is_halted=(action_tier == CircuitBreakerTier.HALT),
            is_derisking=(action_tier == CircuitBreakerTier.DERISK),
            is_throttled=(action_tier == CircuitBreakerTier.CAUTION),
            state=state,
        )

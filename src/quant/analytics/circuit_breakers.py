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

from quant.analytics.ensemble import EnsemblePrediction


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


class EpistemicEntropyCalculator:
    """Calculates directional consensus, normalized Shannon entropy, and epistemic uncertainty ratio.

    Implements:
    - 3-simplex directional consensus probabilities (p_+, p_-, p_0)
    - Normalized Shannon directional entropy:
        H_dir = -sum_{s in {+, -, 0}} p_s ln(p_s + epsilon)
        H_dir_tilde = clip(H_dir / ln(3), 0.0, 1.0)
    - Epistemic uncertainty ratio:
        rho_epistemic = sigma^2_epistemic / (sigma^2_aleatoric + sigma^2_epistemic)
    - Composite epistemic entropy:
        H_epistemic = H_dir_tilde * sqrt(rho_epistemic)

    Enforces Invariants:
    - INV-CB-004: Strict simplex conservation on directional probabilities.
    - INV-CB-005: Immediate defensive failure on non-finite data (NaN/Inf).
    """

    def __init__(self, sign_threshold: float = 1e-4, epsilon_log: float = 1e-30) -> None:
        """Initialize EpistemicEntropyCalculator with defensive hyperparameter validation.

        Args:
            sign_threshold: Directional deadband threshold delta_sign >= 0.0 (default 1e-4).
            epsilon_log: Numerical regularization constant epsilon_log > 0.0 (default 1e-30).
        """
        if not isinstance(sign_threshold, (int, float)) or isinstance(sign_threshold, bool):
            raise InvalidCircuitBreakerInputException(
                f"sign_threshold must be a float, got {type(sign_threshold)}"
            )
        if not math.isfinite(sign_threshold):
            raise DegenerateCircuitBreakerException(
                f"sign_threshold must be a finite float, got {sign_threshold}"
            )
        if sign_threshold < 0.0:
            raise InvalidCircuitBreakerInputException(
                f"sign_threshold must be non-negative (>= 0.0), got {sign_threshold}"
            )

        if not isinstance(epsilon_log, (int, float)) or isinstance(epsilon_log, bool):
            raise InvalidCircuitBreakerInputException(
                f"epsilon_log must be a float, got {type(epsilon_log)}"
            )
        if not math.isfinite(epsilon_log):
            raise DegenerateCircuitBreakerException(
                f"epsilon_log must be a finite float, got {epsilon_log}"
            )
        if epsilon_log <= 0.0:
            raise InvalidCircuitBreakerInputException(
                f"epsilon_log must be strictly positive (> 0.0), got {epsilon_log}"
            )

        self._sign_threshold: float = float(sign_threshold)
        self._epsilon_log: float = float(epsilon_log)

    @property
    def sign_threshold(self) -> float:
        """Directional deadband threshold delta_sign."""
        return self._sign_threshold

    @property
    def epsilon_log(self) -> float:
        """Logarithmic numerical regularization epsilon_log."""
        return self._epsilon_log

    @classmethod
    def from_config(
        cls, config: CircuitBreakerConfig, epsilon_log: float = 1e-30
    ) -> EpistemicEntropyCalculator:
        """Create an EpistemicEntropyCalculator from a CircuitBreakerConfig instance."""
        if not isinstance(config, CircuitBreakerConfig):
            raise InvalidCircuitBreakerInputException(
                f"config must be an instance of CircuitBreakerConfig, got {type(config)}"
            )
        return cls(sign_threshold=config.sign_threshold, epsilon_log=epsilon_log)

    def compute_directional_consensus(
        self, predictions: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Vectorized evaluation of model predictions partitioned into 3 directional buckets.

        Positive: y_k > delta_sign ==> p_+ = sum_{k in pos} w_k
        Negative: y_k < -delta_sign ==> p_- = sum_{k in neg} w_k
        Neutral: |y_k| <= delta_sign ==> p_0 = sum_{k in neu} w_k

        Guarantees INV-CB-004:
            Returns a 1D float64 array of shape (3,), non-negative, summing to 1.0 +/- 1e-10.
        """
        if not isinstance(predictions, np.ndarray):
            raise InvalidCircuitBreakerInputException(
                f"predictions must be a NumPy array, got {type(predictions)}"
            )
        if not isinstance(weights, np.ndarray):
            raise InvalidCircuitBreakerInputException(
                f"weights must be a NumPy array, got {type(weights)}"
            )

        if predictions.ndim != 1:
            raise InvalidCircuitBreakerInputException(
                f"predictions must be 1D, got ndim={predictions.ndim}"
            )
        if weights.ndim != 1:
            raise InvalidCircuitBreakerInputException(
                f"weights must be 1D, got ndim={weights.ndim}"
            )

        if len(predictions) != len(weights):
            raise InvalidCircuitBreakerInputException(
                f"predictions length ({len(predictions)}) must match weights length ({len(weights)})"
            )
        if len(predictions) == 0:
            raise InvalidCircuitBreakerInputException(
                "predictions and weights cannot be empty (K >= 1 required)"
            )

        # INV-CB-005 non-finite validation
        if not np.all(np.isfinite(predictions)):
            raise DegenerateCircuitBreakerException(
                "predictions contains non-finite values (NaN or Inf)"
            )
        if not np.all(np.isfinite(weights)):
            raise DegenerateCircuitBreakerException(
                "weights contains non-finite values (NaN or Inf)"
            )

        if np.any(weights < 0.0):
            raise InvalidCircuitBreakerInputException("weights must be non-negative")

        weights_sum = float(np.sum(weights))
        if abs(weights_sum - 1.0) > 1e-10:
            raise InvalidCircuitBreakerInputException(
                f"weights must sum to 1.0 +/- 1e-10, got {weights_sum}"
            )

        pos_mask = predictions > self._sign_threshold
        neg_mask = predictions < -self._sign_threshold
        neu_mask = np.abs(predictions) <= self._sign_threshold

        p_pos = max(0.0, float(np.sum(weights[pos_mask])))
        p_neg = max(0.0, float(np.sum(weights[neg_mask])))
        p_neu = max(0.0, float(np.sum(weights[neu_mask])))

        return np.array([p_pos, p_neg, p_neu], dtype=np.float64)

    def compute_directional_entropy(self, directional_probs: np.ndarray) -> float:
        """Calculates normalized Shannon directional entropy in [0.0, 1.0].

        H_dir = -sum_{s in {+, -, 0}} p_s ln(p_s + epsilon_log)
        H_dir_tilde = clip(H_dir / ln(3), 0.0, 1.0)

        Guarantees:
            Unanimous consensus [1, 0, 0] ==> 0.0
            Maximum confusion [1/3, 1/3, 1/3] ==> 1.0
            50/50 polarization [0.5, 0.5, 0.0] ==> ln(2)/ln(3) ~= 0.6309
        """
        if not isinstance(directional_probs, np.ndarray):
            raise InvalidCircuitBreakerInputException(
                f"directional_probs must be a NumPy array, got {type(directional_probs)}"
            )
        if directional_probs.shape != (3,):
            raise InvalidCircuitBreakerInputException(
                f"directional_probs must have shape (3,), got {directional_probs.shape}"
            )
        if not np.all(np.isfinite(directional_probs)):
            raise DegenerateCircuitBreakerException(
                "directional_probs contains non-finite values (NaN or Inf)"
            )
        if np.any(directional_probs < 0.0):
            raise InvalidCircuitBreakerInputException(
                "directional_probs components must be non-negative"
            )
        prob_sum = float(np.sum(directional_probs))
        if abs(prob_sum - 1.0) > 1e-10:
            raise InvalidCircuitBreakerInputException(
                f"directional_probs must sum to 1.0 +/- 1e-10, got {prob_sum}"
            )

        raw_entropy = -float(
            np.sum(directional_probs * np.log(directional_probs + self._epsilon_log))
        )
        normalized_entropy = raw_entropy / math.log(3.0)
        return float(np.clip(normalized_entropy, 0.0, 1.0)) + 0.0

    def compute_epistemic_ratio(
        self, aleatoric_variance: float, epistemic_variance: float
    ) -> float:
        """Calculates fraction of predictive variance from epistemic uncertainty in [0.0, 1.0).

        rho_epistemic = sigma^2_epistemic / (sigma^2_aleatoric + sigma^2_epistemic)

        Guarantees:
            When sigma^2_epistemic == 0.0 ==> rho_epistemic == 0.0
            As sigma^2_epistemic >> sigma^2_aleatoric ==> rho_epistemic -> 1.0
        """
        if not isinstance(aleatoric_variance, (int, float)) or isinstance(aleatoric_variance, bool):
            raise InvalidCircuitBreakerInputException(
                f"aleatoric_variance must be a float, got {type(aleatoric_variance)}"
            )
        if not isinstance(epistemic_variance, (int, float)) or isinstance(epistemic_variance, bool):
            raise InvalidCircuitBreakerInputException(
                f"epistemic_variance must be a float, got {type(epistemic_variance)}"
            )

        if not (math.isfinite(aleatoric_variance) and math.isfinite(epistemic_variance)):
            raise DegenerateCircuitBreakerException(
                f"Variances must be finite floats, got aleatoric={aleatoric_variance}, epistemic={epistemic_variance}"
            )

        if aleatoric_variance <= 0.0:
            raise InvalidCircuitBreakerInputException(
                f"aleatoric_variance must be strictly positive (> 0.0), got {aleatoric_variance}"
            )
        if epistemic_variance < 0.0:
            raise InvalidCircuitBreakerInputException(
                f"epistemic_variance must be non-negative (>= 0.0), got {epistemic_variance}"
            )

        total_var = aleatoric_variance + epistemic_variance
        ratio = epistemic_variance / total_var
        return float(np.clip(ratio, 0.0, 1.0)) + 0.0

    def compute_epistemic_entropy(
        self,
        predictions: np.ndarray,
        weights: np.ndarray,
        aleatoric_variance: float,
        epistemic_variance: float,
    ) -> tuple[float, float, float, np.ndarray]:
        """Calculates composite epistemic entropy combining directional consensus and epistemic ratio.

        H_epistemic = H_dir_tilde * sqrt(rho_epistemic) in [0.0, 1.0]

        Returns:
            Tuple of:
            - composite_epistemic_entropy: float in [0.0, 1.0]
            - directional_entropy: float in [0.0, 1.0]
            - epistemic_ratio: float in [0.0, 1.0)
            - directional_probabilities: np.ndarray of shape (3,) summing to 1.0 +/- 1e-10
        """
        directional_probs = self.compute_directional_consensus(predictions, weights)
        directional_entropy = self.compute_directional_entropy(directional_probs)
        epistemic_ratio = self.compute_epistemic_ratio(aleatoric_variance, epistemic_variance)

        composite_entropy = directional_entropy * math.sqrt(epistemic_ratio)
        bounded_composite = float(np.clip(composite_entropy, 0.0, 1.0)) + 0.0

        return bounded_composite, directional_entropy, epistemic_ratio, directional_probs


class ContinuousHaircutCalculator:
    """Calculates continuous soft haircut and thermodynamic composite shock score.

    Implements:
    - Normalized thermodynamic ambiguity:
        beta_tilde = clip((beta - beta_min) / (beta_max - beta_min), 0.0, 1.0)
    - Weighted composite shock score:
        Xi_t = omega_H * H_epistemic + omega_rho * rho_epistemic + omega_beta * beta_tilde in [0.0, 1.0]
    - Normalized logistic sigmoid haircut:
        kappa_raw = 1 / (1 + exp(k_steep * (Xi_t - Xi_mid)))
        kappa_t = clip((kappa_raw - kappa_1) / (kappa_0 - kappa_1), 0.0, 1.0)

    Enforces Invariants:
    - INV-CB-001: Bounded Continuous Haircut (kappa_t in [0.0, 1.0]), kappa(0.0) == 1.0, kappa(1.0) == 0.0.
    - INV-CB-005: Immediate defensive failure on non-finite data (NaN/Inf).
    """

    def __init__(
        self,
        steepness: float = 10.0,
        midpoint: float = 0.50,
        weight_entropy: float = 0.50,
        weight_epistemic_ratio: float = 0.30,
        weight_ambiguity: float = 0.20,
        beta_min: float = 1.0,
        beta_max: float = 5.0,
    ) -> None:
        """Initialize ContinuousHaircutCalculator with defensive parameter validation.

        Args:
            steepness: Logistic sigmoid steepness coefficient k_steep > 0.0 (default 10.0).
            midpoint: Logistic sigmoid midpoint inflection Xi_mid in (0.0, 1.0) (default 0.50).
            weight_entropy: Composite shock weight for epistemic entropy omega_H >= 0.0 (default 0.50).
            weight_epistemic_ratio: Composite shock weight for epistemic ratio omega_rho >= 0.0 (default 0.30).
            weight_ambiguity: Composite shock weight for thermodynamic ambiguity omega_beta >= 0.0 (default 0.20).
            beta_min: Macroeconomic ambiguity lower normalization bound beta_min > 0.0 (default 1.0).
            beta_max: Macroeconomic ambiguity upper normalization bound beta_max > beta_min (default 5.0).
        """
        for param_name, val in (
            ("steepness", steepness),
            ("midpoint", midpoint),
            ("weight_entropy", weight_entropy),
            ("weight_epistemic_ratio", weight_epistemic_ratio),
            ("weight_ambiguity", weight_ambiguity),
            ("beta_min", beta_min),
            ("beta_max", beta_max),
        ):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise InvalidCircuitBreakerInputException(
                    f"{param_name} must be a float, got {type(val)}"
                )
            if not math.isfinite(val):
                raise DegenerateCircuitBreakerException(
                    f"INV-CB-005: {param_name} must be a finite float, got {val}"
                )

        if steepness <= 0.0:
            raise InvalidCircuitBreakerInputException(
                f"steepness must be strictly positive (> 0.0), got {steepness}"
            )

        if not (0.0 < midpoint < 1.0):
            raise InvalidCircuitBreakerInputException(
                f"midpoint must be in (0.0, 1.0), got {midpoint}"
            )

        if weight_entropy < 0.0 or weight_epistemic_ratio < 0.0 or weight_ambiguity < 0.0:
            raise InvalidCircuitBreakerInputException(
                "Component weights must be non-negative: "
                f"weight_entropy={weight_entropy}, "
                f"weight_epistemic_ratio={weight_epistemic_ratio}, "
                f"weight_ambiguity={weight_ambiguity}"
            )

        total_weight = weight_entropy + weight_epistemic_ratio + weight_ambiguity
        if abs(total_weight - 1.0) > 1e-6:
            raise InvalidCircuitBreakerInputException(
                f"Component weights must sum to 1.0 +/- 1e-6, got {total_weight}"
            )

        if beta_min <= 0.0:
            raise InvalidCircuitBreakerInputException(
                f"beta_min must be strictly positive (> 0.0), got {beta_min}"
            )

        if beta_max <= beta_min:
            raise InvalidCircuitBreakerInputException(
                f"beta_max ({beta_max}) must be strictly greater than beta_min ({beta_min})"
            )

        self._steepness: float = float(steepness)
        self._midpoint: float = float(midpoint)
        self._weight_entropy: float = float(weight_entropy)
        self._weight_epistemic_ratio: float = float(weight_epistemic_ratio)
        self._weight_ambiguity: float = float(weight_ambiguity)
        self._beta_min: float = float(beta_min)
        self._beta_max: float = float(beta_max)

        self._kappa_0: float = self._raw_sigmoid(0.0, self._steepness, self._midpoint)
        self._kappa_1: float = self._raw_sigmoid(1.0, self._steepness, self._midpoint)
        self._kappa_range: float = self._kappa_0 - self._kappa_1

    @staticmethod
    def _raw_sigmoid(shock: float, steepness: float, midpoint: float) -> float:
        """Evaluates numerically stable logistic sigmoid function."""
        x = steepness * (shock - midpoint)
        if x > 500.0:
            return 0.0
        if x < -500.0:
            return 1.0
        return 1.0 / (1.0 + math.exp(x))

    @property
    def steepness(self) -> float:
        """Logistic sigmoid steepness coefficient k_steep."""
        return self._steepness

    @property
    def midpoint(self) -> float:
        """Logistic sigmoid midpoint inflection Xi_mid."""
        return self._midpoint

    @property
    def weight_entropy(self) -> float:
        """Composite shock weight for epistemic entropy omega_H."""
        return self._weight_entropy

    @property
    def weight_epistemic_ratio(self) -> float:
        """Composite shock weight for epistemic ratio omega_rho."""
        return self._weight_epistemic_ratio

    @property
    def weight_ambiguity(self) -> float:
        """Composite shock weight for thermodynamic ambiguity omega_beta."""
        return self._weight_ambiguity

    @property
    def beta_min(self) -> float:
        """Macroeconomic ambiguity lower normalization bound beta_min."""
        return self._beta_min

    @property
    def beta_max(self) -> float:
        """Macroeconomic ambiguity upper normalization bound beta_max."""
        return self._beta_max

    @classmethod
    def from_config(cls, config: CircuitBreakerConfig) -> ContinuousHaircutCalculator:
        """Create a ContinuousHaircutCalculator from a CircuitBreakerConfig instance."""
        if not isinstance(config, CircuitBreakerConfig):
            raise InvalidCircuitBreakerInputException(
                f"config must be an instance of CircuitBreakerConfig, got {type(config)}"
            )
        return cls(
            steepness=config.haircut_steepness,
            midpoint=config.haircut_midpoint,
            weight_entropy=config.weight_entropy,
            weight_epistemic_ratio=config.weight_epistemic_ratio,
            weight_ambiguity=config.weight_ambiguity,
            beta_min=config.beta_min,
            beta_max=config.beta_max,
        )

    def compute_composite_shock(
        self,
        epistemic_entropy: float,
        epistemic_ratio: float,
        ambiguity_beta: float,
    ) -> float:
        """Calculates normalized thermodynamic ambiguity and weighted composite shock score.

        tilde_beta = clip((beta - beta_min) / (beta_max - beta_min), 0.0, 1.0)
        Xi_t = omega_H * H_epistemic + omega_rho * rho_epistemic + omega_beta * tilde_beta

        Returns:
            Xi_t: float in [0.0, 1.0]
        """
        for param_name, val in (
            ("epistemic_entropy", epistemic_entropy),
            ("epistemic_ratio", epistemic_ratio),
            ("ambiguity_beta", ambiguity_beta),
        ):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise InvalidCircuitBreakerInputException(
                    f"{param_name} must be a float, got {type(val)}"
                )
            if not math.isfinite(val):
                raise DegenerateCircuitBreakerException(
                    f"INV-CB-005: {param_name} must be a finite float, got {val}"
                )

        if not (0.0 <= epistemic_entropy <= 1.0):
            raise InvalidCircuitBreakerInputException(
                f"epistemic_entropy must be in [0.0, 1.0], got {epistemic_entropy}"
            )

        if not (0.0 <= epistemic_ratio <= 1.0):
            raise InvalidCircuitBreakerInputException(
                f"epistemic_ratio must be in [0.0, 1.0], got {epistemic_ratio}"
            )

        if ambiguity_beta <= 0.0:
            raise InvalidCircuitBreakerInputException(
                f"ambiguity_beta must be strictly positive (> 0.0), got {ambiguity_beta}"
            )

        normalized_beta = min(
            max((ambiguity_beta - self._beta_min) / (self._beta_max - self._beta_min), 0.0),
            1.0,
        )

        raw_shock = (
            self._weight_entropy * epistemic_entropy
            + self._weight_epistemic_ratio * epistemic_ratio
            + self._weight_ambiguity * normalized_beta
        )
        return float(min(max(raw_shock, 0.0), 1.0)) + 0.0

    def compute_haircut(self, composite_shock: float) -> float:
        """Calculates normalized logistic sigmoid haircut multiplier kappa_t in [0.0, 1.0].

        Guarantees INV-CB-001:
        - kappa_t(0.0) == 1.0000
        - kappa_t(1.0) == 0.0000
        - Strictly monotonically decreasing with respect to composite_shock.
        - Continuous and smooth.

        Returns:
            kappa_t: float in [0.0, 1.0]
        """
        if not isinstance(composite_shock, (int, float)) or isinstance(composite_shock, bool):
            raise InvalidCircuitBreakerInputException(
                f"composite_shock must be a float, got {type(composite_shock)}"
            )
        if not math.isfinite(composite_shock):
            raise DegenerateCircuitBreakerException(
                f"INV-CB-005: composite_shock must be a finite float, got {composite_shock}"
            )
        if not (0.0 <= composite_shock <= 1.0):
            raise InvalidCircuitBreakerInputException(
                f"composite_shock must be in [0.0, 1.0], got {composite_shock}"
            )

        if composite_shock == 0.0:
            return 1.0
        if composite_shock == 1.0:
            return 0.0

        kappa_raw = self._raw_sigmoid(composite_shock, self._steepness, self._midpoint)
        normalized_haircut = (kappa_raw - self._kappa_1) / self._kappa_range
        return float(min(max(normalized_haircut, 0.0), 1.0)) + 0.0


class CircuitBreakerOverlayEngine:
    """Stateful orchestrator for epistemic circuit breakers and multi-tier hysteresis.

    Integrates:
    - EpistemicEntropyCalculator for directional consensus and uncertainty decomposition.
    - ContinuousHaircutCalculator for thermodynamic composite shock and soft sizing haircut.
    - Multi-tier discrete state machine (NORMAL, CAUTION, DERISK, HALT).
    - Hysteresis anti-chattering lockout and dwell-time cooling rules (INV-CB-003).
    - Exogenous CUSUM panic shock triggers.

    Enforces Invariants:
    - INV-CB-001: Bounded execution haircut in [0.0, 1.0].
    - INV-CB-002: Action tier is valid CircuitBreakerTier.
    - INV-CB-003: Anti-chattering hysteresis guarantee (dwell lockout, recovery barrier).
    - INV-CB-005: Immediate defensive failure on non-finite data (NaN/Inf).
    """

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        """Initialize CircuitBreakerOverlayEngine.

        Args:
            config: Optional CircuitBreakerConfig instance. If None, defaults to CircuitBreakerConfig().
        """
        if config is not None and not isinstance(config, CircuitBreakerConfig):
            raise InvalidCircuitBreakerInputException(
                f"config must be an instance of CircuitBreakerConfig or None, got {type(config)}"
            )

        self._config: CircuitBreakerConfig = (
            config if config is not None else CircuitBreakerConfig()
        )
        self._entropy_calculator: EpistemicEntropyCalculator = (
            EpistemicEntropyCalculator.from_config(self._config)
        )
        self._haircut_calculator: ContinuousHaircutCalculator = (
            ContinuousHaircutCalculator.from_config(self._config)
        )

    @property
    def config(self) -> CircuitBreakerConfig:
        """Active CircuitBreakerConfig."""
        return self._config

    @property
    def entropy_calculator(self) -> EpistemicEntropyCalculator:
        """Subordinate EpistemicEntropyCalculator."""
        return self._entropy_calculator

    @property
    def haircut_calculator(self) -> ContinuousHaircutCalculator:
        """Subordinate ContinuousHaircutCalculator."""
        return self._haircut_calculator

    def initialize_state(self) -> CircuitBreakerState:
        """Returns clean initial state at step_index = 0 in NORMAL tier.

        Guarantees:
            - tier = CircuitBreakerTier.NORMAL
            - active_bars_in_tier = 0
            - continuous_haircut = 1.0
            - epistemic_entropy = 0.0
            - directional_entropy = 0.0
            - epistemic_ratio = 0.0
            - composite_shock_score = 0.0
            - directional_probabilities = [1/3, 1/3, 1/3]
            - step_index = 0
        """
        return CircuitBreakerState(
            tier=CircuitBreakerTier.NORMAL,
            active_bars_in_tier=0,
            continuous_haircut=1.0,
            epistemic_entropy=0.0,
            directional_entropy=0.0,
            epistemic_ratio=0.0,
            composite_shock_score=0.0,
            directional_probabilities=np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64),
            step_index=0,
        )

    def evaluate(
        self,
        predictions: np.ndarray,
        weights: np.ndarray,
        aleatoric_variance: float,
        epistemic_variance: float,
        ambiguity_beta: float,
        state: CircuitBreakerState,
        cusum_shock: bool = False,
        regime_is_panic: bool = False,
    ) -> tuple[CircuitBreakerDecision, CircuitBreakerState]:
        """Evaluates model disagreement, updates discrete tier with hysteresis, and emits execution decision.

        Algorithm Steps:
        1. Compute directional probabilities, Shannon directional entropy, epistemic ratio,
           and composite epistemic entropy via subordinate EpistemicEntropyCalculator.
        2. Compute composite shock score Xi_t and continuous haircut multiplier kappa_t
           via subordinate ContinuousHaircutCalculator.
        3. Determine Candidate Target Tier:
           - If cusum_shock and regime_is_panic: Candidate is CircuitBreakerTier.HALT.
           - Else if Xi_t >= halt_threshold (0.90): Candidate is CircuitBreakerTier.HALT.
           - Else if Xi_t >= derisk_threshold (0.70): Candidate is CircuitBreakerTier.DERISK.
           - Else if Xi_t >= caution_threshold (0.45): Candidate is CircuitBreakerTier.CAUTION.
           - Else: Candidate is CircuitBreakerTier.NORMAL.
        4. Apply Hysteresis & Anti-Chattering State Transitions (INV-CB-003):
           - Instantaneous Escalation: If candidate tier is strictly more severe than current state.tier,
             transition immediately and reset active_bars_in_tier = 1.
           - Sustain Current Tier: If candidate tier equals state.tier, hold current tier and increment
             active_bars_in_tier = state.active_bars_in_tier + 1.
           - Hysteresis Recovery (De-escalation): If candidate tier is less severe:
             * HALT / DERISK: permitted only if active_bars_in_tier >= dwell_time_bars AND Xi_t < recovery_threshold.
               If met, step down one tier (HALT -> DERISK, DERISK -> CAUTION) and reset active_bars_in_tier = 1.
               Otherwise, hold current tier and increment active_bars_in_tier.
             * CAUTION: de-escalates to NORMAL if Xi_t < recovery_threshold. Otherwise holds CAUTION.
        5. Compute Effective Execution Haircut:
           - HALT: 0.0
           - DERISK: 0.0
           - CAUTION: min(0.50, kappa_t) (throttled cap)
           - NORMAL: kappa_t
        6. Construct updated CircuitBreakerState at step_index = state.step_index + 1.
        7. Construct CircuitBreakerDecision via from_state.
        8. Return (decision, new_state).

        Args:
            predictions: Model forecast vector of shape (K,).
            weights: Non-negative model weights summing to 1.0 of shape (K,).
            aleatoric_variance: Strictly positive aleatoric variance sigma^2_aleatoric > 0.0.
            epistemic_variance: Non-negative epistemic variance sigma^2_epistemic >= 0.0.
            ambiguity_beta: Thermodynamic macroeconomic ambiguity beta > 0.0.
            state: Preceding CircuitBreakerState snapshot.
            cusum_shock: Boolean flag indicating exogenous CUSUM jump detection.
            regime_is_panic: Boolean flag indicating panic volatility regime.

        Returns:
            Tuple of (CircuitBreakerDecision, updated CircuitBreakerState).

        Raises:
            InvalidCircuitBreakerInputException: On schema or type mismatches.
            DegenerateCircuitBreakerException: On non-finite values (INV-CB-005).
        """
        # Defensive parameter and type validation
        if not isinstance(state, CircuitBreakerState):
            raise InvalidCircuitBreakerInputException(
                f"state must be an instance of CircuitBreakerState, got {type(state)}"
            )

        if not isinstance(cusum_shock, bool) or type(cusum_shock) is not bool:
            raise InvalidCircuitBreakerInputException(
                f"cusum_shock must be a boolean, got {type(cusum_shock)}"
            )

        if not isinstance(regime_is_panic, bool) or type(regime_is_panic) is not bool:
            raise InvalidCircuitBreakerInputException(
                f"regime_is_panic must be a boolean, got {type(regime_is_panic)}"
            )

        if not isinstance(ambiguity_beta, (int, float)) or isinstance(ambiguity_beta, bool):
            raise InvalidCircuitBreakerInputException(
                f"ambiguity_beta must be a float, got {type(ambiguity_beta)}"
            )
        if not math.isfinite(ambiguity_beta):
            raise DegenerateCircuitBreakerException(
                f"INV-CB-005: ambiguity_beta must be a finite float, got {ambiguity_beta}"
            )
        if ambiguity_beta <= 0.0:
            raise InvalidCircuitBreakerInputException(
                f"ambiguity_beta must be strictly positive (> 0.0), got {ambiguity_beta}"
            )

        # 1. Epistemic entropy and uncertainty decomposition
        (
            h_epi,
            h_dir,
            rho,
            probs,
        ) = self._entropy_calculator.compute_epistemic_entropy(
            predictions=predictions,
            weights=weights,
            aleatoric_variance=aleatoric_variance,
            epistemic_variance=epistemic_variance,
        )

        # 2. Composite shock score and continuous haircut
        shock_score = self._haircut_calculator.compute_composite_shock(
            epistemic_entropy=h_epi,
            epistemic_ratio=rho,
            ambiguity_beta=ambiguity_beta,
        )
        continuous_haircut = self._haircut_calculator.compute_haircut(shock_score)

        # 3. Determine Candidate Target Tier
        if (cusum_shock and regime_is_panic) or shock_score >= self._config.halt_threshold:
            candidate_tier = CircuitBreakerTier.HALT
        elif shock_score >= self._config.derisk_threshold:
            candidate_tier = CircuitBreakerTier.DERISK
        elif shock_score >= self._config.caution_threshold:
            candidate_tier = CircuitBreakerTier.CAUTION
        else:
            candidate_tier = CircuitBreakerTier.NORMAL

        # 4. Apply Hysteresis & Anti-Chattering State Transitions (INV-CB-003)
        current_tier = state.tier
        new_tier: CircuitBreakerTier
        active_bars: int

        if candidate_tier > current_tier:
            # Instantaneous Escalation
            new_tier = candidate_tier
            active_bars = 1
        elif candidate_tier == current_tier:
            # Sustain Current Tier
            new_tier = current_tier
            active_bars = state.active_bars_in_tier + 1
        else:
            # Hysteresis Recovery (De-escalation: candidate_tier < current_tier)
            if current_tier in (CircuitBreakerTier.HALT, CircuitBreakerTier.DERISK):
                if (
                    state.active_bars_in_tier >= self._config.dwell_time_bars
                    and shock_score < self._config.recovery_threshold
                ):
                    # Step down one tier
                    new_tier = CircuitBreakerTier(current_tier - 1)
                    active_bars = 1
                else:
                    # Hold current tier
                    new_tier = current_tier
                    active_bars = state.active_bars_in_tier + 1
            elif current_tier == CircuitBreakerTier.CAUTION:
                if shock_score < self._config.recovery_threshold:
                    new_tier = CircuitBreakerTier.NORMAL
                    active_bars = 1
                else:
                    new_tier = current_tier
                    active_bars = state.active_bars_in_tier + 1
            else:
                # current_tier is NORMAL, cannot de-escalate further
                new_tier = CircuitBreakerTier.NORMAL
                active_bars = state.active_bars_in_tier + 1

        # 5. Compute Effective Execution Haircut
        execution_haircut: float
        if new_tier in (CircuitBreakerTier.HALT, CircuitBreakerTier.DERISK):
            execution_haircut = 0.0
        elif new_tier == CircuitBreakerTier.CAUTION:
            execution_haircut = min(0.50, continuous_haircut)
        else:  # CircuitBreakerTier.NORMAL
            execution_haircut = continuous_haircut

        # 6. Construct updated CircuitBreakerState at step_index = state.step_index + 1
        new_state = CircuitBreakerState(
            tier=new_tier,
            active_bars_in_tier=active_bars,
            continuous_haircut=continuous_haircut,
            epistemic_entropy=h_epi,
            directional_entropy=h_dir,
            epistemic_ratio=rho,
            composite_shock_score=shock_score,
            directional_probabilities=probs,
            step_index=state.step_index + 1,
        )

        # 7. Construct CircuitBreakerDecision
        decision = CircuitBreakerDecision.from_state(
            action_tier=new_tier,
            execution_haircut=execution_haircut,
            state=new_state,
        )

        # 8. Return (decision, new_state)
        return decision, new_state

    def evaluate_prediction(
        self,
        prediction: EnsemblePrediction,
        predictions: np.ndarray,
        ambiguity_beta: float,
        state: CircuitBreakerState,
        cusum_shock: bool = False,
        regime_is_panic: bool | None = None,
    ) -> tuple[CircuitBreakerDecision, CircuitBreakerState]:
        """Evaluate circuit breaker overlays directly from an upstream EnsemblePrediction payload.

        Consumes an EnsemblePrediction from Phase 5 Step 1 (quant.analytics.ensemble) alongside
        the raw model prediction vector, extracting posterior model weights, aleatoric variance,
        epistemic disagreement variance, and automatically deriving panic regime status if unspecified.

        Args:
            prediction: EnsemblePrediction emitted by RegimeConditionedDMAEngine.
            predictions: Model forecast vector of shape (K,).
            ambiguity_beta: Macroeconomic ambiguity parameter beta > 0.0.
            state: Preceding CircuitBreakerState snapshot.
            cusum_shock: Boolean flag indicating exogenous CUSUM jump detection.
            regime_is_panic: Optional boolean flag indicating panic volatility regime.
                If None (default), derived from prediction.regime_probabilities via
                argmax == 2 (Crisis / Panic regime).

        Returns:
            Tuple of (CircuitBreakerDecision, updated CircuitBreakerState).

        Raises:
            InvalidCircuitBreakerInputException: On schema, type mismatches, or invalid prediction instance.
            DegenerateCircuitBreakerException: On non-finite values (INV-CB-005).
        """
        if not isinstance(prediction, EnsemblePrediction):
            raise InvalidCircuitBreakerInputException(
                f"prediction must be an instance of EnsemblePrediction, got {type(prediction)}"
            )

        if regime_is_panic is None:
            panic_flag = bool(np.argmax(prediction.regime_probabilities) == 2)
        elif isinstance(regime_is_panic, bool) and type(regime_is_panic) is bool:
            panic_flag = regime_is_panic
        else:
            raise InvalidCircuitBreakerInputException(
                f"regime_is_panic must be a boolean or None, got {type(regime_is_panic)}"
            )

        return self.evaluate(
            predictions=predictions,
            weights=prediction.model_weights,
            aleatoric_variance=prediction.aleatoric_variance,
            epistemic_variance=prediction.epistemic_variance,
            ambiguity_beta=ambiguity_beta,
            state=state,
            cusum_shock=cusum_shock,
            regime_is_panic=panic_flag,
        )


__all__ = [
    "CircuitBreakerConfig",
    "CircuitBreakerDecision",
    "CircuitBreakerError",
    "CircuitBreakerOverlayEngine",
    "CircuitBreakerState",
    "CircuitBreakerTier",
    "ContinuousHaircutCalculator",
    "DegenerateCircuitBreakerException",
    "EpistemicEntropyCalculator",
    "InvalidCircuitBreakerInputException",
]

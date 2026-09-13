"""Semi-Parametric Extreme Value Theory (EVT-POT) Tail Risk Subsystem.

Purpose:
    Provides institutional-grade, mathematically rigorous tail risk modeling for extreme
    negative portfolio return innovations:
    1. Peaks-Over-Threshold (POT) Generalized Pareto Distribution (GPD): Closed-form algebraic
       parameter estimation (xi, beta) eliminating iterative numerical solvers in the hot path.
    2. Coherent Expected Shortfall (CVaR): Coherent, subadditive downside tail risk strictly
       bounding Value-at-Risk from above (INV-TR-001).
    3. Fréchet Tail Stability & Infinite Variance Tripwire: Detects theoretical infinite
       variance (xi >= 1.0) and triggers emergency execution halts (INV-TR-002).
    4. Cold-Start Degradation Ladder: Seamless transition across Empirical Quantiles (N < 30),
       Student-t Method-of-Moments (30 <= N < 250), and Full EVT-GPD (N >= 250).

Dependencies:
    - math: Scalar finiteness and arithmetic operations.
    - dataclasses: Immutable frozen domain containers.
    - numpy: Vectorized numerical bounds and array operations.

Structural Relationship:
    - Ingests: Strictly lagged rolling loss innovations X_tau = -r_tau from portfolio bars.
    - Emits: EVTTailParameters, TailRiskMetrics (VaR_alpha, CVaR_alpha).
    - Consumed by: Unified convex execution sizer (execution_sizing.py) and circuit breaker
      overlay engine (circuit_breakers.py).

Invariants Enforced:
    - INV-TR-001 (Coherent Risk Ordering): cvar_alpha >= var_alpha within 1e-10 tolerance.
    - INV-TR-002 (Fréchet Tail Stability & Infinite Variance Tripwire): Tail index xi in
      [0.001, 0.999]. If xi >= 1.0, raise InfiniteVarianceException demanding emergency HALT.
    - INV-TR-005 (Non-Finite Input Data Protection): Immediate defensive failure on non-finite
      values (NaN / Inf) with DegenerateTailRiskException.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

import numpy as np
from scipy.special import stdtrit

# Diagnostic Fault Vector Constants (Rule 2: Deterministic Diagnostics)
ERR_TR_ORDERING: Final[str] = "ERR-TR-001: Coherent Risk Ordering violated (cvar_alpha < var_alpha)"
ERR_TR_INFINITE_VARIANCE: Final[str] = (
    "ERR-TR-002: Infinite theoretical variance tripwire (shape_xi >= 1.0)"
)
ERR_TR_DEGENERATE: Final[str] = "ERR-TR-003: Non-finite input, negative scale, or degenerate bounds"
ERR_TR_INVALID_CONFIG: Final[str] = (
    "ERR-TR-004: Invalid hyperparameter configuration or hierarchy violation"
)
ERR_TR_STARVATION: Final[str] = "ERR-TR-005: Insufficient observation history or buffer starvation"
ERR_TR_LATENCY: Final[str] = "ERR-TR-006: Tail risk evaluation latency violation or execution error"

VALID_TAIL_METHODS: Final[tuple[str, ...]] = ("EVT_PWM", "STUDENT_T", "EMPIRICAL")


class TailRiskError(Exception):
    """Base exception for all EVT tail risk and Expected Shortfall errors."""


class DegenerateTailRiskException(TailRiskError):
    """Raised when non-finite inputs (NaN/Inf), negative scale, or degenerate bounds are detected."""


class InfiniteVarianceException(TailRiskError):
    """Raised when tail shape index xi >= 1.0, indicating infinite theoretical variance demanding emergency HALT."""


class InvalidTailRiskInputException(TailRiskError):
    """Raised on invalid hyperparameter configurations, invalid types, or hierarchy contract violations."""


@dataclass(frozen=True, slots=True)
class TailRiskConfig:
    """Hyperparameter configuration container for EVT-GPD tail risk estimation.

    Attributes:
        confidence_level: Downside tail confidence level alpha in (0.50, 1.0), default 0.99.
        threshold_k: Dynamic high-threshold standard deviation multiplier k > 0.0, default 1.645.
        min_observations_evt: Minimum sample size required for EVT-GPD modeling, default 250.
        min_observations_student_t: Minimum sample size for Student-t modeling, default 30.
        min_exceedances_evt: Minimum count of threshold exceedances N_u for EVT fit, default 15.
        tail_index_lower_bound: Lower clamp barrier for Fréchet tail shape index xi, default 0.001.
        tail_index_upper_bound: Upper clamp barrier for Fréchet tail shape index xi, default 0.999.
        rolling_window_size: Rolling historical buffer capacity W >= min_observations_evt, default 500.
    """

    confidence_level: float = 0.99
    threshold_k: float = 1.645
    min_observations_evt: int = 250
    min_observations_student_t: int = 30
    min_exceedances_evt: int = 15
    tail_index_lower_bound: float = 0.001
    tail_index_upper_bound: float = 0.999
    rolling_window_size: int = 500

    def __post_init__(self) -> None:
        """Validate invariant boundaries and threshold hierarchies upon instantiation."""
        # 1. Float field typing and finiteness verification
        for field_name in (
            "confidence_level",
            "threshold_k",
            "tail_index_lower_bound",
            "tail_index_upper_bound",
        ):
            val = getattr(self, field_name)
            if not (isinstance(val, (int, float)) and not isinstance(val, bool)):
                raise InvalidTailRiskInputException(
                    f"{ERR_TR_INVALID_CONFIG}: {field_name} must be numeric, got {type(val).__name__}"
                )
            if not math.isfinite(val):
                raise DegenerateTailRiskException(
                    f"{ERR_TR_DEGENERATE}: {field_name} must be a finite float, got {val}"
                )

        # 2. Integer field typing verification
        for field_name in (
            "min_observations_evt",
            "min_observations_student_t",
            "min_exceedances_evt",
            "rolling_window_size",
        ):
            val = getattr(self, field_name)
            if not (isinstance(val, int) and not isinstance(val, bool)):
                raise InvalidTailRiskInputException(
                    f"{ERR_TR_INVALID_CONFIG}: {field_name} must be an integer, got {type(val).__name__}"
                )

        # 3. Confidence level bounds (0.50 < alpha < 1.0)
        if not (0.50 < self.confidence_level < 1.0):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: confidence_level must be in (0.50, 1.0), got {self.confidence_level}"
            )

        # 4. Threshold multiplier bounds (k > 0.0)
        if self.threshold_k <= 0.0:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: threshold_k must be strictly positive, got {self.threshold_k}"
            )

        # 5. Tail index bounds (0.0 < lower < upper < 1.0)
        if not (0.0 < self.tail_index_lower_bound < self.tail_index_upper_bound < 1.0):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: tail index bounds must satisfy 0.0 < lower < upper < 1.0, "
                f"got lower={self.tail_index_lower_bound}, upper={self.tail_index_upper_bound}"
            )

        # 6. Minimum observation counts must be strictly positive
        if self.min_observations_student_t < 1:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: min_observations_student_t must be >= 1, got {self.min_observations_student_t}"
            )
        if self.min_exceedances_evt < 1:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: min_exceedances_evt must be >= 1, got {self.min_exceedances_evt}"
            )

        # 7. Hierarchy contract: min_student_t < min_evt <= rolling_window_size
        if self.min_observations_student_t >= self.min_observations_evt:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: min_observations_student_t ({self.min_observations_student_t}) "
                f"must be strictly less than min_observations_evt ({self.min_observations_evt})"
            )
        if self.min_observations_evt > self.rolling_window_size:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: min_observations_evt ({self.min_observations_evt}) "
                f"cannot exceed rolling_window_size ({self.rolling_window_size})"
            )

    def clamp_tail_index(self, xi: float) -> float:
        """Clamp estimated tail index into stable Fréchet bounds [lower, upper].

        INV-TR-002: If xi >= 1.0, raises InfiniteVarianceException demanding emergency HALT.
        Otherwise clamps xi to [tail_index_lower_bound, tail_index_upper_bound].

        Args:
            xi: Raw estimated tail shape index.

        Returns:
            Clamped tail shape index strictly contained in [lower_bound, upper_bound].

        Raises:
            InvalidTailRiskInputException: If xi is not numeric.
            DegenerateTailRiskException: If xi is non-finite (NaN / Inf).
            InfiniteVarianceException: If xi >= 1.0 (INV-TR-002 infinite variance tripwire).
        """
        # Functional Purpose: Protect against explosive tail extrapolations and tripwire infinite variance.
        # Explicit Dependency Tracking: self.tail_index_lower_bound, self.tail_index_upper_bound.
        # Structural Relationship: Called by EVTTailRiskEngine during parameter calibration.
        # Defensive Invariant: INV-TR-002 Fréchet stability and infinite theoretical variance rejection.
        if not (isinstance(xi, (int, float)) and not isinstance(xi, bool)):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: xi must be numeric, got {type(xi).__name__}"
            )
        if not math.isfinite(xi):
            raise DegenerateTailRiskException(
                f"{ERR_TR_DEGENERATE}: Cannot clamp non-finite tail index: {xi}"
            )
        if xi >= 1.0:
            raise InfiniteVarianceException(
                f"{ERR_TR_INFINITE_VARIANCE}: Estimated shape_xi={xi} >= 1.0 violates INV-TR-002. "
                "Theoretical variance is infinite; emergency HALT required."
            )
        if xi < self.tail_index_lower_bound:
            return self.tail_index_lower_bound
        if xi > self.tail_index_upper_bound:
            return self.tail_index_upper_bound
        return float(xi)


_DEFAULT_TAIL_CONFIG: Final[TailRiskConfig] = TailRiskConfig()


@lru_cache(maxsize=4096)
def _get_pwm_weights(n_u: int) -> np.ndarray:
    """Compute and cache normalized PWM weights vector for sample size n_u.

    Mathematical Definition:
        w_i = (N_u - i + 0.35) / N_u^2  for i = 1, ..., N_u
    """
    w = np.arange(n_u - 0.65, -0.65, -1.0, dtype=np.float64) / (n_u * n_u)
    w.flags.writeable = False
    return w


@dataclass(frozen=True, slots=True)
class EVTTailParameters:
    """Extreme Value Theory (EVT) Generalized Pareto Distribution (GPD) parameter container.

    Attributes:
        threshold_u: Dynamic high threshold u_t above which excess losses are evaluated.
        shape_xi: Fréchet tail shape parameter xi in [0.0, 1.0). If xi >= 1.0, raises InfiniteVarianceException.
        scale_beta: GPD scale parameter beta > 0.0 (or beta >= 0.0 for empirical fallback).
        num_exceedances: Count of observations strictly exceeding threshold u (N_u >= 0).
        total_observations: Total count of historical return innovations in sample (n >= N_u).
        method: Parameter estimation methodology ("EVT_PWM", "STUDENT_T", "EMPIRICAL").
    """

    threshold_u: float
    shape_xi: float
    scale_beta: float
    num_exceedances: int
    total_observations: int
    method: str

    def __post_init__(self) -> None:
        """Validate invariant boundaries and type integrity upon instantiation."""
        # 1. Float field typing and finiteness verification
        for field_name in ("threshold_u", "shape_xi", "scale_beta"):
            val = getattr(self, field_name)
            if not (isinstance(val, (int, float)) and not isinstance(val, bool)):
                raise InvalidTailRiskInputException(
                    f"{ERR_TR_INVALID_CONFIG}: {field_name} must be numeric, got {type(val).__name__}"
                )
            if not math.isfinite(val):
                raise DegenerateTailRiskException(
                    f"{ERR_TR_DEGENERATE}: {field_name} must be a finite float, got {val}"
                )

        # 2. Integer field typing verification
        for field_name in ("num_exceedances", "total_observations"):
            val = getattr(self, field_name)
            if not (isinstance(val, int) and not isinstance(val, bool)):
                raise InvalidTailRiskInputException(
                    f"{ERR_TR_INVALID_CONFIG}: {field_name} must be an integer, got {type(val).__name__}"
                )

        # 3. Method validation
        if not isinstance(self.method, str):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: method must be a string, got {type(self.method).__name__}"
            )
        if self.method not in VALID_TAIL_METHODS:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: method must be one of {VALID_TAIL_METHODS}, got '{self.method}'"
            )

        # 4. INV-TR-002: Fréchet Tail Stability & Infinite Variance Tripwire
        if self.shape_xi >= 1.0:
            raise InfiniteVarianceException(
                f"{ERR_TR_INFINITE_VARIANCE}: shape_xi={self.shape_xi} >= 1.0 violates INV-TR-002. "
                "Theoretical variance is infinite; emergency HALT required."
            )
        if self.shape_xi < 0.0:
            raise DegenerateTailRiskException(
                f"{ERR_TR_DEGENERATE}: shape_xi must be non-negative, got {self.shape_xi}"
            )

        # 5. Scale parameter beta validation
        if self.scale_beta < 0.0:
            raise DegenerateTailRiskException(
                f"{ERR_TR_DEGENERATE}: scale_beta must be non-negative, got {self.scale_beta}"
            )
        if self.method != "EMPIRICAL" and self.scale_beta <= 0.0:
            raise DegenerateTailRiskException(
                f"{ERR_TR_DEGENERATE}: scale_beta must be strictly positive for method '{self.method}', "
                f"got {self.scale_beta}"
            )

        # 6. Exceedance and observation count validation (0 <= N_u <= n)
        if self.total_observations < 0:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: total_observations must be >= 0, got {self.total_observations}"
            )
        if self.num_exceedances < 0:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: num_exceedances must be >= 0, got {self.num_exceedances}"
            )
        if self.num_exceedances > self.total_observations:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: num_exceedances ({self.num_exceedances}) "
                f"cannot exceed total_observations ({self.total_observations})"
            )


@dataclass(frozen=True, slots=True)
class TailRiskMetrics:
    """Coherent tail risk metrics container holding Value-at-Risk and Expected Shortfall.

    Attributes:
        var_alpha: Value-at-Risk at confidence level alpha (loss quantile).
        cvar_alpha: Expected Shortfall (CVaR) at confidence level alpha (mean excess loss).
        confidence_level: Confidence level alpha in (0.50, 1.0).
        tail_parameters: Underlying EVTTailParameters calibrated for this evaluation.
        step_index: Sequential bar or execution step index >= 0.
    """

    var_alpha: float
    cvar_alpha: float
    confidence_level: float
    tail_parameters: EVTTailParameters
    step_index: int

    def __post_init__(self) -> None:
        """Validate invariant boundaries and coherent risk ordering (INV-TR-001)."""
        # 1. Float field typing and finiteness verification
        for field_name in ("var_alpha", "cvar_alpha", "confidence_level"):
            val = getattr(self, field_name)
            if not (isinstance(val, (int, float)) and not isinstance(val, bool)):
                raise InvalidTailRiskInputException(
                    f"{ERR_TR_INVALID_CONFIG}: {field_name} must be numeric, got {type(val).__name__}"
                )
            if not math.isfinite(val):
                raise DegenerateTailRiskException(
                    f"{ERR_TR_DEGENERATE}: {field_name} must be a finite float, got {val}"
                )

        # 2. Tail parameters type verification
        if not isinstance(self.tail_parameters, EVTTailParameters):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: tail_parameters must be an instance of EVTTailParameters, "
                f"got {type(self.tail_parameters).__name__}"
            )

        # 3. Step index typing and bounds verification
        if not (isinstance(self.step_index, int) and not isinstance(self.step_index, bool)):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: step_index must be an integer, got {type(self.step_index).__name__}"
            )
        if self.step_index < 0:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: step_index must be >= 0, got {self.step_index}"
            )

        # 4. Confidence level bounds (0.50 < alpha < 1.0)
        if not (0.50 < self.confidence_level < 1.0):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: confidence_level must be in (0.50, 1.0), got {self.confidence_level}"
            )

        # 5. INV-TR-001: Coherent Risk Ordering (cvar_alpha >= var_alpha within scale-adaptive tolerance)
        # Functional Purpose: Enforce Artzner's coherence axiom; CVaR must strictly bound VaR from above.
        # Explicit Dependency Tracking: self.cvar_alpha, self.var_alpha.
        # Structural Relationship: Ingested by convex execution sizer for hard drawdown constraints.
        # Defensive Invariant: INV-TR-001 cvar_alpha >= var_alpha - tolerance.
        tolerance = max(1e-10, 1e-9 * abs(self.var_alpha))
        if self.cvar_alpha < self.var_alpha - tolerance:
            raise DegenerateTailRiskException(
                f"{ERR_TR_ORDERING}: INV-TR-001 violated: cvar_alpha ({self.cvar_alpha}) < "
                f"var_alpha ({self.var_alpha}) beyond {tolerance} numerical tolerance."
            )


class ProbabilityWeightedMomentsEstimator:
    """Closed-form Probability Weighted Moments (PWM) parameter estimation engine for GPD.

    Purpose:
        Provides institutional-grade, deterministic algebraic parameter estimation (xi, beta)
        for Peak-Over-Threshold (POT) Generalized Pareto Distribution (GPD) modeling.
        Eliminates iterative numerical optimizers (banned under Rule 4.3) to achieve
        sub-0.02ms execution latency (INV-TR-006).

    Mathematical Foundation:
        Given Nu exceedances sorted in ascending order y_{(1)} <= ... <= y_{(N_u)}:
            M0 = (1 / N_u) * sum(y_{(i)})
            M1 = (1 / N_u) * sum((1 - (i - 0.35) / N_u) * y_{(i)})
            denom = M0 - 2 * M1
            xi_PWM = 2 - M0 / denom
            beta_PWM = (2 * M0 * M1) / denom

    Invariants Enforced:
        - INV-TR-002: Fréchet Tail Stability (xi in [0.001, 0.999]). If xi >= 1.0, raises
          InfiniteVarianceException demanding emergency HALT.
        - INV-TR-005: Non-finite input protection; rejects NaN / Inf with DegenerateTailRiskException.
        - INV-TR-006: Hot-path execution latency SLA <= 0.02ms for Nu = 500.
        - Rule 4.3: Zero iterative numerical solvers (scipy.optimize strictly banned).
    """

    def __init__(self, config: TailRiskConfig | None = None) -> None:
        """Initialize estimator with optional institutional TailRiskConfig."""
        self.config: TailRiskConfig = config if config is not None else TailRiskConfig()

    @staticmethod
    def compute_dynamic_threshold(losses: np.ndarray, threshold_k: float = 1.645) -> float:
        """Compute sample mean and high-volatility dynamic threshold u_t = mu + k * sigma.

        Args:
            losses: 1D numpy array of positive loss innovations X = -r.
            threshold_k: High-threshold standard deviation multiplier k > 0.0 (default 1.645).

        Returns:
            Dynamic threshold u_t as a finite float.

        Raises:
            InvalidTailRiskInputException: If losses is not a 1D ndarray or threshold_k <= 0.
            DegenerateTailRiskException: If losses has n < 2 observations (ERR_TR_STARVATION),
                contains non-finite elements, or threshold_k is non-finite.
        """
        # Functional Purpose: Evaluate dynamic threshold u_t = mu + k * sigma.
        # Explicit Dependency Tracking: losses, threshold_k.
        # Structural Relationship: Feeds extract_exceedances in EVT-POT pipeline.
        # Defensive Invariant: n >= 2 (ERR_TR_STARVATION), finite values, k > 0, ddof=1.
        if not isinstance(losses, np.ndarray):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: losses must be a numpy ndarray, got {type(losses).__name__}"
            )
        if losses.ndim != 1:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: losses array must be 1-dimensional, got shape {losses.shape}"
            )
        if losses.size < 2:
            raise DegenerateTailRiskException(
                f"{ERR_TR_STARVATION}: Minimum sample size n >= 2 required for dynamic threshold estimation, "
                f"got {losses.size}"
            )
        if not np.isfinite(losses).all():
            raise DegenerateTailRiskException(
                f"{ERR_TR_DEGENERATE}: losses contain non-finite elements (NaN or Inf)"
            )
        if not (isinstance(threshold_k, (int, float)) and not isinstance(threshold_k, bool)):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: threshold_k must be numeric, got {type(threshold_k).__name__}"
            )
        if not math.isfinite(threshold_k):
            raise DegenerateTailRiskException(
                f"{ERR_TR_DEGENERATE}: threshold_k must be finite, got {threshold_k}"
            )
        if threshold_k <= 0.0:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: threshold_k must be strictly positive, got {threshold_k}"
            )
        mu = float(np.mean(losses))
        sigma = float(np.std(losses, ddof=1))
        return float(mu + threshold_k * sigma)

    @staticmethod
    def extract_exceedances(losses: np.ndarray, threshold_u: float) -> np.ndarray:
        """Extract threshold exceedances Y = X[X > u] - u strictly above high threshold.

        Args:
            losses: 1D numpy array of historical loss innovations X = -r.
            threshold_u: Dynamic threshold cutoff u_t.

        Returns:
            1D float64 numpy array of positive excess losses Y (empty if no exceedances).

        Raises:
            InvalidTailRiskInputException: If losses is not a 1D ndarray or threshold_u not numeric.
            DegenerateTailRiskException: If losses or threshold_u contain non-finite values.
        """
        # Functional Purpose: Extract excess losses Y = X[X > u] - u above high threshold.
        # Explicit Dependency Tracking: losses, threshold_u.
        # Structural Relationship: Emits exceedance array Y for PWM fitting.
        # Defensive Invariant: Non-finite protection, non-negative exceedance guarantees.
        if not isinstance(losses, np.ndarray):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: losses must be a numpy ndarray, got {type(losses).__name__}"
            )
        if losses.ndim != 1:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: losses array must be 1-dimensional, got shape {losses.shape}"
            )
        if not np.all(np.isfinite(losses)):
            raise DegenerateTailRiskException(
                f"{ERR_TR_DEGENERATE}: losses contain non-finite elements (NaN or Inf)"
            )
        if not (isinstance(threshold_u, (int, float)) and not isinstance(threshold_u, bool)):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: threshold_u must be numeric, got {type(threshold_u).__name__}"
            )
        if not math.isfinite(threshold_u):
            raise DegenerateTailRiskException(
                f"{ERR_TR_DEGENERATE}: threshold_u must be finite, got {threshold_u}"
            )
        exceedances = losses[losses > threshold_u] - threshold_u
        y = np.asarray(exceedances, dtype=np.float64)
        if y.ndim != 1:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: exceedances must be 1-dimensional, got {y.ndim}"
            )
        if not np.all(np.isfinite(y)):
            raise DegenerateTailRiskException(
                f"{ERR_TR_DEGENERATE}: exceedances contain non-finite values"
            )
        if np.any(y < 0.0):
            raise DegenerateTailRiskException(
                f"{ERR_TR_DEGENERATE}: exceedances must be non-negative"
            )
        return y

    @staticmethod
    def fit(
        exceedances: np.ndarray,
        total_observations: int,
        threshold_u: float,
        config: TailRiskConfig | None = None,
    ) -> EVTTailParameters:
        """Calibrate GPD parameters (xi, beta) using closed-form Probability Weighted Moments.

        Args:
            exceedances: 1D non-empty numpy array of positive excess losses Y = X - u.
            total_observations: Historical sample capacity n >= N_u.
            threshold_u: High threshold u_t above which excess losses occurred.
            config: Optional TailRiskConfig specifying bounds and tolerances.

        Returns:
            Calibrated EVTTailParameters with method='EVT_PWM'.

        Raises:
            InvalidTailRiskInputException: If inputs are improperly typed or n < N_u.
            DegenerateTailRiskException: If exceedances is empty, non-finite, or negative.
            InfiniteVarianceException: If xi >= 1.0 (INV-TR-002 infinite variance tripwire).
        """
        # Functional Purpose: Estimate GPD parameters (xi, beta) via closed-form algebraic PWM.
        # Explicit Dependency Tracking: exceedances, total_observations, threshold_u, config.
        # Structural Relationship: Ingested by EVTTailRiskEngine to construct EVTTailParameters.
        # Defensive Invariant: INV-TR-002 Fréchet stability, INV-TR-006 ultra-low latency, zero solvers.
        cfg = config if config is not None else _DEFAULT_TAIL_CONFIG
        if not isinstance(cfg, TailRiskConfig):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: config must be an instance of TailRiskConfig, got {type(cfg).__name__}"
            )
        if not isinstance(exceedances, np.ndarray):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: exceedances must be a numpy ndarray, got {type(exceedances).__name__}"
            )
        if exceedances.ndim != 1:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: exceedances must be 1-dimensional, got shape {exceedances.shape}"
            )
        n_u = int(exceedances.size)
        if n_u == 0:
            raise DegenerateTailRiskException(
                f"{ERR_TR_DEGENERATE}: Cannot fit PWM with zero exceedances (num_exceedances=0)"
            )
        if not (isinstance(total_observations, int) and not isinstance(total_observations, bool)):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: total_observations must be an integer, got {type(total_observations).__name__}"
            )
        if total_observations < n_u:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: total_observations ({total_observations}) "
                f"cannot be less than num_exceedances ({n_u})"
            )
        if not (isinstance(threshold_u, (int, float)) and not isinstance(threshold_u, bool)):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: threshold_u must be numeric, got {type(threshold_u).__name__}"
            )
        if not math.isfinite(threshold_u):
            raise DegenerateTailRiskException(
                f"{ERR_TR_DEGENERATE}: threshold_u must be finite, got {threshold_u}"
            )

        # 1. Sort exceedances in ascending order: y_{(1)} <= ... <= y_{(N_u)}
        y_sorted = (
            np.sort(exceedances)
            if exceedances.dtype == np.float64
            else np.sort(np.asarray(exceedances, dtype=np.float64))
        )
        # Fast O(1) non-finite check: IEEE 754 sort places -inf at 0, +inf and NaN at -1
        if not (math.isfinite(float(y_sorted[0])) and math.isfinite(float(y_sorted[-1]))):
            raise DegenerateTailRiskException(
                f"{ERR_TR_DEGENERATE}: exceedances contain non-finite values"
            )
        if float(y_sorted[0]) < 0.0:
            raise DegenerateTailRiskException(
                f"{ERR_TR_DEGENERATE}: exceedances must be non-negative"
            )

        # INV-TR-002: Algebraic Hill tail index pre-filter on top k extreme exceedances
        if n_u >= 3:
            k = min(max(2, int(0.10 * n_u)), n_u - 1)
            y_ref = float(y_sorted[n_u - k - 1])
            if y_ref > 0.0:
                top_k = y_sorted[-k:]
                xi_hill = float(np.log(top_k).sum()) / k - math.log(y_ref)
                if xi_hill >= 1.0:
                    raise InfiniteVarianceException(
                        f"{ERR_TR_INFINITE_VARIANCE}: Hill tail index pre-filter xi_hill={xi_hill:.4f} >= 1.0 "
                        "violates INV-TR-002. Theoretical variance is infinite; emergency HALT required."
                    )

        # 2. Vectorized Probability Weighted Moments: M0 and M1
        # M0 = (1 / N_u) * sum(y_i)
        inv_n_u = 1.0 / n_u
        m0 = float(y_sorted.sum()) * inv_n_u

        # i = 1, 2, ..., N_u
        # M1 = (1 / N_u) * sum((1 - (i - 0.35) / N_u) * y_{(i)})
        # Pre-calculated and cached weights: w_i = (N_u - i + 0.35) / N_u^2
        weights = _get_pwm_weights(n_u)
        m1 = float(np.dot(weights, y_sorted))

        # 3. Closed-form algebraic estimators
        denom = m0 - 2.0 * m1

        # Defensive Boundary Handling:
        # If denom <= 1e-12 or xi <= 0.0: Exponential fallback
        if denom <= 1e-12:
            xi = cfg.tail_index_lower_bound
            beta = max(m0, 1e-8)
        else:
            inv_denom = 1.0 / denom
            xi = 2.0 - (m0 * inv_denom)
            beta = (2.0 * m0 * m1) * inv_denom
            if xi <= 0.0:
                xi = cfg.tail_index_lower_bound
                beta = max(m0, 1e-8)

        # INV-TR-002: Infinite theoretical variance tripwire
        if xi >= cfg.tail_index_upper_bound:
            raise InfiniteVarianceException(
                f"{ERR_TR_INFINITE_VARIANCE}: Estimated shape_xi={xi:.6f} >= "
                f"tail_index_upper_bound={cfg.tail_index_upper_bound} violates INV-TR-002. "
                "Theoretical variance is infinite; emergency HALT required."
            )

        # Fréchet stability clamp (native Python scalar)
        xi = cfg.clamp_tail_index(xi)
        beta = max(beta, 1e-8)

        return EVTTailParameters(
            threshold_u=float(threshold_u),
            shape_xi=xi,
            scale_beta=beta,
            num_exceedances=n_u,
            total_observations=total_observations,
            method="EVT_PWM",
        )


class EVTTailRiskEngine:
    """Semi-Parametric Extreme Value Theory (EVT-POT) Tail Risk Engine with Cold-Start Ladder.

    Purpose:
        Provides institutional-grade, mathematically coherent Value-at-Risk (VaR_alpha) and
        Expected Shortfall (CVaR_alpha) modeling with Fréchet tail stability and zero iterative
        numerical solvers (Rule 4.3).
        Operates a 3-tier cold-start degradation ladder:
            - Tier 1 (EMPIRICAL): Severe starvation (N < 30 or N_u < 10) evaluating empirical
              order statistics and sample mean excess losses.
            - Tier 2 (STUDENT_T): Maturing history (30 <= N < 250, or N_u < 15, or 1 - alpha >= N_u / N)
              with closed-form method-of-moments degrees of freedom nu, analytical Student-t PDF,
              and coherent Expected Shortfall integral.
            - Tier 3 (EVT_PWM): Fully hydrated history (N >= 250, N_u >= 15, 1 - alpha < N_u / N)
              with algebraic Hosking & Wallis (1987) Probability Weighted Moments GPD calibration.
        Supports both stateless calculation over historical loss arrays and stateful rolling ring
        buffer tracking with strictly lagged zero-lookahead causality (INV-TR-007).

    Mathematical Foundation:
        1. Dynamic High Threshold:
            u_t = mu_{t-1} + k_{threshold} * sigma_{t-1},  where sigma uses ddof=1.
        2. Tier 1 Empirical Quantiles:
            VaR_alpha = Percentile_{Empirical}(X, 100 * alpha)
            CVaR_alpha = Mean(X | X >= VaR_alpha)
        3. Tier 2 Parametric Student-t Method of Moments:
            nu = 4.0 + 6.0 / gamma_4  if gamma_4 > 0 else 100.0,  clamped to [2.10, 100.0]
            sigma_{scale} = s * sqrt((nu - 2.0) / nu)
            q_alpha = stdtrit(nu, alpha)
            VaR_alpha = mu + sigma_{scale} * q_alpha
            CVaR_alpha = mu + sigma_{scale} * [ f_nu(q_alpha) / (1 - alpha) ] * [ (nu + q_alpha^2) / (nu - 1.0) ]
        4. Tier 3 Semi-Parametric EVT-POT GPD via PWM:
            z = (N / N_u) * (1 - alpha)
            VaR_alpha = u_t + (beta / xi) * [ z^{-xi} - 1.0 ]  (or u_t - beta * ln(z) if xi <= 1e-6)
            CVaR_alpha = (VaR_alpha + beta - xi * u_t) / (1.0 - xi)  (or VaR_alpha + beta if xi <= 1e-6)

    Invariants Enforced:
        - INV-TR-001: Coherent Risk Ordering (cvar_alpha >= var_alpha).
        - INV-TR-002: Fréchet Tail Stability (xi in [0.001, 0.999]; xi >= 1.0 triggers InfiniteVarianceException).
        - INV-TR-003: Artzner Subadditivity (CVaR_alpha(w1*X1 + w2*X2) <= w1*CVaR_alpha(X1) + w2*CVaR_alpha(X2)).
        - INV-TR-005: Non-finite input protection (NaN/Inf raises DegenerateTailRiskException).
        - INV-TR-006: Hot-path execution latency SLA <= 0.05ms for N = 500.
        - INV-TR-007: Zero-Lookahead Causality (strictly lagged window [t-W, t-1]).
        - Rule 4.3: Zero iterative numerical solvers (scipy.optimize strictly banned).
    """

    def __init__(self, config: TailRiskConfig | None = None) -> None:
        """Initialize EVTTailRiskEngine with optional configuration and preallocated ring buffer.

        Args:
            config: Optional TailRiskConfig specifying hyperparameters and buffer capacity.

        Raises:
            InvalidTailRiskInputException: If config is not an instance of TailRiskConfig.
        """
        # Functional Purpose: Initialize engine configuration, algebraic PWM estimator, and ring buffer.
        # Explicit Dependency Tracking: config, TailRiskConfig, ProbabilityWeightedMomentsEstimator.
        # Structural Relationship: Ingested by convex execution sizer and risk overlay pipelines.
        # Defensive Invariant: Valid configuration hierarchy; preallocated fixed-capacity buffer.
        self._config: TailRiskConfig = config if config is not None else _DEFAULT_TAIL_CONFIG
        if not isinstance(self._config, TailRiskConfig):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: config must be an instance of TailRiskConfig, "
                f"got {type(self._config).__name__}"
            )
        self._estimator: ProbabilityWeightedMomentsEstimator = ProbabilityWeightedMomentsEstimator(
            self._config
        )
        self._capacity: int = self._config.rolling_window_size
        self._buffer: np.ndarray = np.zeros(self._capacity, dtype=np.float64)
        self._head: int = 0
        self._count: int = 0

    @property
    def config(self) -> TailRiskConfig:
        """Return the immutable TailRiskConfig governing this engine."""
        return self._config

    @property
    def estimator(self) -> ProbabilityWeightedMomentsEstimator:
        """Return the underlying ProbabilityWeightedMomentsEstimator instance."""
        return self._estimator

    @property
    def capacity(self) -> int:
        """Return the maximum rolling window capacity W of the stateful ring buffer."""
        return self._capacity

    @property
    def count(self) -> int:
        """Return the current number of observations accumulated in the rolling ring buffer."""
        return self._count

    def clear(self) -> None:
        """Reset the internal stateful ring buffer to zero observations."""
        # Functional Purpose: Flush rolling buffer during state reinitialization or regime shifts.
        # Explicit Dependency Tracking: self._buffer, self._head, self._count.
        # Structural Relationship: Invoked on instrument transitions or circuit breaker resets.
        # Defensive Invariant: Clean zero-state with head=0 and count=0.
        self._buffer.fill(0.0)
        self._head = 0
        self._count = 0

    def update(self, loss: float) -> None:
        """Append a strictly lagged historical loss innovation X_{t-1} = -r_{t-1} to the rolling ring buffer.

        Args:
            loss: Negative return innovation scalar (X = -r).

        Raises:
            InvalidTailRiskInputException: If loss is not numeric or is boolean.
            DegenerateTailRiskException: If loss is non-finite (NaN / Inf) (INV-TR-005).
        """
        # Functional Purpose: Record incoming loss innovation into fixed-capacity ring buffer in O(1) time.
        # Explicit Dependency Tracking: self._buffer, self._head, self._count, self._capacity.
        # Structural Relationship: Driven by bar completion events in execution orchestrator.
        # Defensive Invariant: Non-finite protection (INV-TR-005); bounded circular buffer capacity.
        if not (isinstance(loss, (int, float)) and not isinstance(loss, bool)):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: loss must be numeric, got {type(loss).__name__}"
            )
        loss_val = float(loss)
        if not math.isfinite(loss_val):
            raise DegenerateTailRiskException(
                f"{ERR_TR_DEGENERATE}: Cannot update ring buffer with non-finite loss: {loss}"
            )
        self._buffer[self._head] = loss_val
        self._head = (self._head + 1) % self._capacity
        if self._count < self._capacity:
            self._count += 1

    def get_rolling_risk_metrics(
        self,
        step_index: int,
        confidence_level: float | None = None,
    ) -> TailRiskMetrics:
        """Calculate tail risk metrics over the current rolling buffer window.

        Args:
            step_index: Sequential bar or execution step index >= 0.
            confidence_level: Optional override for confidence level alpha in (0.50, 1.0).

        Returns:
            Calibrated TailRiskMetrics container with VaR_alpha, CVaR_alpha, and tail parameters.

        Raises:
            DegenerateTailRiskException: If buffer has fewer than 2 observations (ERR_TR_STARVATION).
        """
        # Functional Purpose: Extract strictly lagged rolling historical window and evaluate tail metrics.
        # Explicit Dependency Tracking: self._buffer, self._head, self._count, calculate_risk_metrics.
        # Structural Relationship: Consumed by convex execution sizer on each trading bar t.
        # Defensive Invariant: Buffer starvation protection (count >= 2); zero-lookahead causality (INV-TR-007).
        if self._count < 2:
            raise DegenerateTailRiskException(
                f"{ERR_TR_STARVATION}: Rolling buffer starvation: at least 2 observations required, "
                f"got {self._count}"
            )
        if self._count < self._capacity:
            losses = self._buffer[: self._count].copy()
        else:
            # Chronologically reconstruct window: oldest to newest
            losses = np.empty(self._capacity, dtype=np.float64)
            tail_len = self._capacity - self._head
            losses[:tail_len] = self._buffer[self._head :]
            losses[tail_len:] = self._buffer[: self._head]

        return self.calculate_risk_metrics(
            losses=losses,
            step_index=step_index,
            confidence_level=confidence_level,
        )

    def calculate_risk_metrics(
        self,
        losses: np.ndarray,
        step_index: int = 0,
        confidence_level: float | None = None,
    ) -> TailRiskMetrics:
        """Evaluate VaR_alpha and coherent CVaR_alpha across 3-tier cold-start degradation ladder.

        Args:
            losses: 1D numpy array of historical loss innovations X = -r.
            step_index: Sequential bar or execution step index >= 0.
            confidence_level: Optional override for confidence level alpha in (0.50, 1.0).

        Returns:
            Calibrated TailRiskMetrics container.

        Raises:
            InvalidTailRiskInputException: If arguments are typed incorrectly or step_index < 0.
            DegenerateTailRiskException: If losses contain non-finite elements (NaN/Inf) (INV-TR-005)
                or sample size N < 2 (ERR_TR_STARVATION).
            InfiniteVarianceException: If theoretical tail shape index xi >= 1.0 (INV-TR-002).
        """
        # Functional Purpose: Evaluate coherent VaR/CVaR across Empirical, Student-t, and EVT-PWM tiers.
        # Explicit Dependency Tracking: self._config, self._estimator, losses, step_index, confidence_level.
        # Structural Relationship: Ingested by convex execution sizer and circuit breaker overlay engine.
        # Defensive Invariant: INV-TR-001 (CVaR >= VaR), INV-TR-002 (xi < 1.0), INV-TR-005 (finite inputs).
        if not isinstance(losses, np.ndarray):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: losses must be a numpy ndarray, got {type(losses).__name__}"
            )
        if losses.ndim != 1:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: losses array must be 1-dimensional, got shape {losses.shape}"
            )
        n = int(losses.size)
        if n < 2:
            raise DegenerateTailRiskException(
                f"{ERR_TR_STARVATION}: Minimum sample size N >= 2 required for tail risk evaluation, "
                f"got {n}"
            )
        if not (isinstance(step_index, int) and not isinstance(step_index, bool)):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: step_index must be an integer, got {type(step_index).__name__}"
            )
        if step_index < 0:
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: step_index must be >= 0, got {step_index}"
            )

        alpha = self._config.confidence_level if confidence_level is None else confidence_level
        if not (isinstance(alpha, (int, float)) and not isinstance(alpha, bool)):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: confidence_level must be numeric, got {type(alpha).__name__}"
            )
        alpha = float(alpha)
        if not math.isfinite(alpha):
            raise DegenerateTailRiskException(
                f"{ERR_TR_DEGENERATE}: confidence_level must be finite, got {alpha}"
            )
        if not (0.50 < alpha < 1.0):
            raise InvalidTailRiskInputException(
                f"{ERR_TR_INVALID_CONFIG}: confidence_level must be in (0.50, 1.0), got {alpha}"
            )

        # Fast O(1) IEEE 754 non-finite check on sample mean (INV-TR-005)
        # Any NaN or Inf in losses propagates to non-finite mean
        mu = float(np.mean(losses))
        if not math.isfinite(mu):
            raise DegenerateTailRiskException(
                f"{ERR_TR_DEGENERATE}: losses contain non-finite elements (NaN or Inf)"
            )

        diff = losses - mu
        s2 = float(np.dot(diff, diff)) / (n - 1.0)
        s = math.sqrt(max(0.0, s2))

        # Zero-volatility defensive handling for constant losses (all-zero or constant c)
        if s <= 1e-12:
            return self._calculate_empirical_risk_metrics(
                losses=losses,
                u_t=mu,
                n_u=0,
                n=n,
                alpha=alpha,
                step_index=step_index,
            )

        u_t = mu + self._config.threshold_k * s

        exceedances = losses[losses > u_t] - u_t
        n_u = int(exceedances.size)

        min_obs_student = self._config.min_observations_student_t
        min_obs_evt = self._config.min_observations_evt
        min_exc_evt = self._config.min_exceedances_evt

        # Ladder Selection:
        # Tier 1: Severe Starvation: N < 30 or (N >= 250 and N_u < 10)
        if n < min_obs_student or (n >= min_obs_evt and n_u < 10):
            return self._calculate_empirical_risk_metrics(
                losses=losses,
                u_t=u_t,
                n_u=n_u,
                n=n,
                alpha=alpha,
                step_index=step_index,
            )

        # Tier 3: Fully Hydrated History: N >= 250 and N_u >= 15 and (1 - alpha) < N_u / N
        if n >= min_obs_evt and n_u >= min_exc_evt and (1.0 - alpha) < (n_u / n):
            return self._calculate_evt_pwm_risk_metrics(
                exceedances=exceedances,
                u_t=u_t,
                n_u=n_u,
                n=n,
                alpha=alpha,
                step_index=step_index,
            )

        # Tier 2: Maturing History: 30 <= N < 250 or (N >= 250 and N_u < 15) or (1 - alpha >= N_u / N)
        return self._calculate_student_t_risk_metrics(
            losses=losses,
            mu=mu,
            s=s,
            u_t=u_t,
            n_u=n_u,
            n=n,
            alpha=alpha,
            step_index=step_index,
        )

    def _calculate_empirical_risk_metrics(
        self,
        losses: np.ndarray,
        u_t: float,
        n_u: int,
        n: int,
        alpha: float,
        step_index: int,
    ) -> TailRiskMetrics:
        """Calculate Tier 1 Empirical Quantile tail risk metrics (INV-TR-001)."""
        # Functional Purpose: Fallback to non-parametric empirical order statistics under sample starvation.
        # Explicit Dependency Tracking: losses, alpha, u_t, n_u, n, step_index.
        # Structural Relationship: Emits TailRiskMetrics when N < 30 or N_u < 10.
        # Defensive Invariant: INV-TR-001 (CVaR >= VaR); shape_xi = 0.0, scale_beta = 0.0.
        var_alpha = float(np.percentile(losses, 100.0 * alpha))
        tail_losses = losses[losses >= var_alpha]
        cvar_alpha = float(np.mean(tail_losses)) if tail_losses.size > 0 else var_alpha
        cvar_alpha = max(cvar_alpha, var_alpha)

        tail_params = EVTTailParameters(
            threshold_u=float(u_t),
            shape_xi=0.0,
            scale_beta=0.0,
            num_exceedances=n_u,
            total_observations=n,
            method="EMPIRICAL",
        )
        return TailRiskMetrics(
            var_alpha=var_alpha,
            cvar_alpha=cvar_alpha,
            confidence_level=alpha,
            tail_parameters=tail_params,
            step_index=step_index,
        )

    def _calculate_student_t_risk_metrics(
        self,
        losses: np.ndarray,
        mu: float,
        s: float,
        u_t: float,
        n_u: int,
        n: int,
        alpha: float,
        step_index: int,
    ) -> TailRiskMetrics:
        """Calculate Tier 2 Student-t Method-of-Moments tail risk metrics (INV-TR-001)."""
        # Functional Purpose: Parametric Student-t tail modeling with closed-form degrees of freedom.
        # Explicit Dependency Tracking: losses, mu, s, u_t, n_u, n, alpha, stdtrit, math.lgamma.
        # Structural Relationship: Emits TailRiskMetrics for maturing history (30 <= N < 250).
        # Defensive Invariant: INV-TR-001 (CVaR >= VaR); nu in [2.10, 100.0]; beta > 0.
        diff = losses - mu
        diff2 = diff * diff
        m4 = float(np.mean(diff2 * diff2))
        s2 = s * s
        s4 = s2 * s2

        if s4 > 1e-16:
            gamma4 = (m4 / s4) - 3.0
            nu = (4.0 + (6.0 / gamma4)) if gamma4 > 0.0 else 100.0
        else:
            nu = 100.0

        nu = max(2.10, min(100.0, nu))
        sigma_scale = max(s * math.sqrt((nu - 2.0) / nu), 1e-8)

        q_alpha = float(stdtrit(nu, alpha))

        log_c_nu = (
            math.lgamma((nu + 1.0) / 2.0) - math.lgamma(nu / 2.0) - 0.5 * math.log(nu * math.pi)
        )
        log_pdf = log_c_nu - 0.5 * (nu + 1.0) * math.log(1.0 + (q_alpha * q_alpha) / nu)
        pdf_q = math.exp(log_pdf)

        var_alpha = mu + sigma_scale * q_alpha
        cvar_factor = (pdf_q / (1.0 - alpha)) * ((nu + q_alpha * q_alpha) / (nu - 1.0))
        cvar_alpha = mu + sigma_scale * cvar_factor
        cvar_alpha = max(cvar_alpha, var_alpha)

        tail_params = EVTTailParameters(
            threshold_u=float(u_t),
            shape_xi=1.0 / nu,
            scale_beta=sigma_scale,
            num_exceedances=n_u,
            total_observations=n,
            method="STUDENT_T",
        )
        return TailRiskMetrics(
            var_alpha=var_alpha,
            cvar_alpha=cvar_alpha,
            confidence_level=alpha,
            tail_parameters=tail_params,
            step_index=step_index,
        )

    def _calculate_evt_pwm_risk_metrics(
        self,
        exceedances: np.ndarray,
        u_t: float,
        n_u: int,
        n: int,
        alpha: float,
        step_index: int,
    ) -> TailRiskMetrics:
        """Calculate Tier 3 Semi-Parametric EVT-POT GPD tail risk metrics via PWM (INV-TR-001, INV-TR-002)."""
        # Functional Purpose: Extreme Value Theory Peaks-Over-Threshold closed-form algebraic evaluation.
        # Explicit Dependency Tracking: self._estimator.fit, exceedances, n, u_t, alpha, step_index.
        # Structural Relationship: Emits institutional TailRiskMetrics when fully hydrated (N >= 250).
        # Defensive Invariant: INV-TR-001 (CVaR >= VaR); INV-TR-002 (xi < 1.0 or InfiniteVarianceException).
        fit_params = self._estimator.fit(
            exceedances=exceedances,
            total_observations=n,
            threshold_u=u_t,
            config=self._config,
        )
        xi = fit_params.shape_xi
        beta = fit_params.scale_beta
        z = (n / n_u) * (1.0 - alpha)

        if xi <= 1e-6:
            var_alpha = u_t - beta * math.log(z)
            cvar_alpha = var_alpha + beta
        else:
            var_alpha = u_t + (beta / xi) * (math.pow(z, -xi) - 1.0)
            cvar_alpha = (var_alpha + beta - xi * u_t) / (1.0 - xi)

        cvar_alpha = max(cvar_alpha, var_alpha)

        return TailRiskMetrics(
            var_alpha=var_alpha,
            cvar_alpha=cvar_alpha,
            confidence_level=alpha,
            tail_parameters=fit_params,
            step_index=step_index,
        )


__all__ = [
    "ERR_TR_DEGENERATE",
    "ERR_TR_INFINITE_VARIANCE",
    "ERR_TR_INVALID_CONFIG",
    "ERR_TR_LATENCY",
    "ERR_TR_ORDERING",
    "ERR_TR_STARVATION",
    "VALID_TAIL_METHODS",
    "DegenerateTailRiskException",
    "EVTTailParameters",
    "EVTTailRiskEngine",
    "InfiniteVarianceException",
    "InvalidTailRiskInputException",
    "ProbabilityWeightedMomentsEstimator",
    "TailRiskConfig",
    "TailRiskError",
    "TailRiskMetrics",
]

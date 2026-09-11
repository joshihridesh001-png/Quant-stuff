"""Regime-Conditioned Dynamic Model Averaging (RD-DMA) subsystem.

Provides domain entities, configuration, state representations, and invariant contracts
for institutional-grade ensemble aggregation of quantitative strategies.

Invariants Enforced:
- INV-ENS-001 (Strict Simplex Conservation): Model weights must sum to 1.0 +/- 1e-10
  with all components strictly positive.
- INV-ENS-002 (Variance Positivity & Additivity): sigma^2_total = sigma^2_aleatoric + sigma^2_epistemic
  with sigma^2_aleatoric > 0 and sigma^2_epistemic >= 0.
- INV-ENS-003 (Bounded Adaptive Forgetting): alpha_t in [alpha_min, alpha_max] subset of (0, 1).
"""

import math
from dataclasses import dataclass

import numpy as np


class EnsembleError(Exception):
    """Base exception for all ensemble modeling and invariant violations."""


class DegenerateEnsembleException(EnsembleError):
    """Raised when numerical collapse, NaN/Inf, variance breach, or degeneracy occurs."""


class InvalidPredictionException(EnsembleError):
    """Raised when prediction payloads or dimension alignments are invalid."""


@dataclass(frozen=True)
class EnsembleConfig:
    """Hyperparameter configuration for Regime-Conditioned Dynamic Model Averaging.

    Enforces Invariant INV-ENS-003 and defensive boundary validations.

    Attributes:
        base_forgetting_factor: Baseline memory forgetting factor alpha_0 in (0.80, 0.999).
        volatility_sensitivity: Sensitivity coefficient kappa_alpha >= 0.0.
        min_forgetting_factor: Lower forgetting bound alpha_min in (0.50, base_forgetting_factor).
        max_forgetting_factor: Upper forgetting bound alpha_max in (base_forgetting_factor, 1.0).
        orthogonality_penalty: SVD subspace correlation penalty lambda_ortho >= 0.0.
        downside_penalty: Asymmetric downside loss multiplier gamma_down >= 0.0.
        turnover_damping: L1 turnover regularization weight lambda_churn in [0.0, 1.0).
        ridge_shrinkage: Tikhonov correlation shrinkage delta_ridge in (0.0, 1.0).
        ambiguity_shrinkage_cap: Thermodynamic shrinkage upper bound kappa_shrink in [0.0, 1.0].
        min_weight_floor: Minimum Laplace probability floor eps_floor > 0.0.
        temperature: Optimization entropy temperature tau > 0.0.
        mirror_descent_lr: Entropic Mirror Descent step size eta > 0.0.
        mirror_descent_max_iter: Maximum mirror descent iterations >= 1.
        mirror_descent_tol: Convergence tolerance for mirror descent > 0.0.
    """

    base_forgetting_factor: float = 0.96
    volatility_sensitivity: float = 0.50
    min_forgetting_factor: float = 0.85
    max_forgetting_factor: float = 0.99
    orthogonality_penalty: float = 0.25
    downside_penalty: float = 2.50
    turnover_damping: float = 0.15
    ridge_shrinkage: float = 0.05
    ambiguity_shrinkage_cap: float = 0.50
    min_weight_floor: float = 1e-5
    temperature: float = 1.0
    mirror_descent_lr: float = 0.50
    mirror_descent_max_iter: int = 10
    mirror_descent_tol: float = 1e-6

    def __post_init__(self) -> None:
        """Validate invariant boundary contracts on hyperparameters."""
        # Finiteness check on all float attributes
        for field_name in (
            "base_forgetting_factor",
            "volatility_sensitivity",
            "min_forgetting_factor",
            "max_forgetting_factor",
            "orthogonality_penalty",
            "downside_penalty",
            "turnover_damping",
            "ridge_shrinkage",
            "ambiguity_shrinkage_cap",
            "min_weight_floor",
            "temperature",
            "mirror_descent_lr",
            "mirror_descent_tol",
        ):
            val = getattr(self, field_name)
            if not (isinstance(val, (int, float)) and math.isfinite(val)):
                raise EnsembleError(f"{field_name} must be a finite float, got {val}")

        # INV-ENS-003 bounds & relations
        if self.min_forgetting_factor >= self.max_forgetting_factor:
            raise EnsembleError(
                f"min_forgetting_factor ({self.min_forgetting_factor}) must be strictly less than "
                f"max_forgetting_factor ({self.max_forgetting_factor})"
            )
        if not (0.80 < self.base_forgetting_factor < 0.999):
            raise EnsembleError(
                f"base_forgetting_factor must be in (0.80, 0.999), got {self.base_forgetting_factor}"
            )
        if not (0.50 < self.min_forgetting_factor < self.base_forgetting_factor):
            raise EnsembleError(
                "min_forgetting_factor must be in (0.50, base_forgetting_factor), "
                f"got min={self.min_forgetting_factor}, base={self.base_forgetting_factor}"
            )
        if not (self.base_forgetting_factor < self.max_forgetting_factor < 1.0):
            raise EnsembleError(
                "max_forgetting_factor must be in (base_forgetting_factor, 1.0), "
                f"got max={self.max_forgetting_factor}, base={self.base_forgetting_factor}"
            )

        # Non-negative penalties
        if self.volatility_sensitivity < 0.0:
            raise EnsembleError(
                f"volatility_sensitivity must be >= 0.0, got {self.volatility_sensitivity}"
            )
        if self.orthogonality_penalty < 0.0:
            raise EnsembleError(
                f"orthogonality_penalty must be >= 0.0, got {self.orthogonality_penalty}"
            )
        if self.downside_penalty < 0.0:
            raise EnsembleError(f"downside_penalty must be >= 0.0, got {self.downside_penalty}")

        # Turnover damping in [0.0, 1.0)
        if not (0.0 <= self.turnover_damping < 1.0):
            raise EnsembleError(
                f"turnover_damping must be in [0.0, 1.0), got {self.turnover_damping}"
            )

        # Ridge shrinkage in (0.0, 1.0)
        if not (0.0 < self.ridge_shrinkage < 1.0):
            raise EnsembleError(
                f"ridge_shrinkage must be in (0.0, 1.0), got {self.ridge_shrinkage}"
            )

        # Ambiguity shrinkage cap in [0.0, 1.0]
        if not (0.0 <= self.ambiguity_shrinkage_cap <= 1.0):
            raise EnsembleError(
                f"ambiguity_shrinkage_cap must be in [0.0, 1.0], got {self.ambiguity_shrinkage_cap}"
            )

        # Strictly positive parameters
        if self.min_weight_floor <= 0.0:
            raise EnsembleError(f"min_weight_floor must be > 0.0, got {self.min_weight_floor}")
        if self.temperature <= 0.0:
            raise EnsembleError(f"temperature must be > 0.0, got {self.temperature}")
        if self.mirror_descent_lr <= 0.0:
            raise EnsembleError(f"mirror_descent_lr must be > 0.0, got {self.mirror_descent_lr}")
        if self.mirror_descent_max_iter < 1:
            raise EnsembleError(
                f"mirror_descent_max_iter must be >= 1, got {self.mirror_descent_max_iter}"
            )
        if self.mirror_descent_tol <= 0.0:
            raise EnsembleError(f"mirror_descent_tol must be > 0.0, got {self.mirror_descent_tol}")


@dataclass(frozen=True)
class EnsembleState:
    """Immutable state snapshot tracking online dynamic model averaging across time.

    Enforces Invariant INV-ENS-001 (Strict Simplex Conservation) and dimensional alignment.

    Attributes:
        step_index: Monotonically increasing bar counter t >= 0.
        weights: Active composite model weights vector w_t (shape: (K,), sum=1.0, w_k > 0).
        regime_conditional_posteriors: Matrix of regime-conditioned weights (shape: (3, K), rows sum=1.0).
        cumulative_losses: Historical accumulated asymmetric downside losses (shape: (K,)).
        effective_models: Effective number of constituent models K_eff in [1.0, K].
        mean_realized_volatility: Exponential moving average baseline volatility sigma_bar > 0.0.
        last_ambiguity_temperature: Most recent thermodynamic temperature beta_t > 0.0.
    """

    step_index: int
    weights: np.ndarray
    regime_conditional_posteriors: np.ndarray
    cumulative_losses: np.ndarray
    effective_models: float
    mean_realized_volatility: float
    last_ambiguity_temperature: float

    def __post_init__(self) -> None:
        """Validate invariant contracts on state variables."""
        if self.step_index < 0:
            raise EnsembleError(f"step_index must be >= 0, got {self.step_index}")

        # Weights validation: INV-ENS-001
        if not isinstance(self.weights, np.ndarray) or self.weights.ndim != 1:
            raise EnsembleError("weights must be a 1D numpy array")
        k_models = len(self.weights)
        if k_models < 1:
            raise EnsembleError("weights must contain at least 1 model")
        if not np.all(np.isfinite(self.weights)):
            raise DegenerateEnsembleException("weights must contain only finite values")
        if np.any(self.weights <= 0.0):
            raise EnsembleError("INV-ENS-001 violation: weights must be strictly positive")
        weights_sum = float(np.sum(self.weights))
        if not math.isclose(weights_sum, 1.0, abs_tol=1e-10):
            raise EnsembleError(
                f"INV-ENS-001 violation: weights sum to {weights_sum}, expected 1.0 +/- 1e-10"
            )

        # Regime conditional posteriors validation
        if not isinstance(self.regime_conditional_posteriors, np.ndarray):
            raise EnsembleError("regime_conditional_posteriors must be a numpy array")
        if self.regime_conditional_posteriors.shape != (3, k_models):
            raise EnsembleError(
                f"regime_conditional_posteriors shape must be (3, {k_models}), "
                f"got {self.regime_conditional_posteriors.shape}"
            )
        if not np.all(np.isfinite(self.regime_conditional_posteriors)):
            raise DegenerateEnsembleException(
                "regime_conditional_posteriors must contain only finite values"
            )
        if np.any(self.regime_conditional_posteriors < 0.0):
            raise EnsembleError("regime_conditional_posteriors elements must be non-negative")
        for row_idx in range(3):
            row_sum = float(np.sum(self.regime_conditional_posteriors[row_idx, :]))
            if not math.isclose(row_sum, 1.0, abs_tol=1e-10):
                raise EnsembleError(
                    f"regime_conditional_posteriors row {row_idx} must sum to 1.0 +/- 1e-10, got {row_sum}"
                )

        # Cumulative losses validation
        if not isinstance(self.cumulative_losses, np.ndarray):
            raise EnsembleError("cumulative_losses must be a numpy array")
        if self.cumulative_losses.shape != (k_models,):
            raise EnsembleError(
                f"cumulative_losses shape must be ({k_models},), got {self.cumulative_losses.shape}"
            )
        if not np.all(np.isfinite(self.cumulative_losses)):
            raise DegenerateEnsembleException("cumulative_losses must contain only finite values")

        # Effective models bounds check [1.0, K]
        if not math.isfinite(self.effective_models):
            raise DegenerateEnsembleException("effective_models must be finite")
        if self.effective_models < 1.0 - 1e-6 or self.effective_models > float(k_models) + 1e-6:
            raise EnsembleError(
                f"effective_models must be in [1.0, {k_models}], got {self.effective_models}"
            )

        # Volatility baseline check
        if not (
            math.isfinite(self.mean_realized_volatility) and self.mean_realized_volatility > 0.0
        ):
            raise EnsembleError(
                f"mean_realized_volatility must be > 0.0, got {self.mean_realized_volatility}"
            )

        # Ambiguity temperature check
        if not (
            math.isfinite(self.last_ambiguity_temperature) and self.last_ambiguity_temperature > 0.0
        ):
            raise EnsembleError(
                f"last_ambiguity_temperature must be > 0.0, got {self.last_ambiguity_temperature}"
            )


@dataclass(frozen=True)
class EnsemblePrediction:
    """Probabilistic prediction payload emitted by the RD-DMA engine at bar t.

    Enforces Invariant INV-ENS-002 (Law of Total Variance decomposition).

    Attributes:
        point_prediction: Aggregated return prediction mu_hat_t.
        aleatoric_variance: Expected downside process variance sigma^2_aleatoric > 0.0.
        epistemic_variance: Strategy disagreement variance sigma^2_epistemic >= 0.0.
        total_variance: Total predictive variance sigma^2_total = aleatoric + epistemic.
        model_weights: Posterior model allocation vector w_t (shape: (K,), sum=1.0).
        regime_probabilities: Predictive forward regime distribution p_{t+1|t} (shape: (3,), sum=1.0).
        effective_models: Effective number of models K_eff in [1.0, K].
        volatility_forgetting_factor: Applied volatility-adaptive forgetting factor alpha_t.
        ambiguity_shrinkage_weight: Applied thermodynamic shrinkage weight lambda_beta.
    """

    point_prediction: float
    aleatoric_variance: float
    epistemic_variance: float
    total_variance: float
    model_weights: np.ndarray
    regime_probabilities: np.ndarray
    effective_models: float
    volatility_forgetting_factor: float
    ambiguity_shrinkage_weight: float

    def __post_init__(self) -> None:
        """Validate invariant contracts on predictive emission."""
        # Finiteness checks on scalar attributes
        for field_name in (
            "point_prediction",
            "aleatoric_variance",
            "epistemic_variance",
            "total_variance",
            "effective_models",
            "volatility_forgetting_factor",
            "ambiguity_shrinkage_weight",
        ):
            val = getattr(self, field_name)
            if not (isinstance(val, (int, float)) and math.isfinite(val)):
                raise DegenerateEnsembleException(f"{field_name} must be a finite float, got {val}")

        # Invariant INV-ENS-002: Variance positivity and additivity
        if self.aleatoric_variance <= 0.0:
            raise DegenerateEnsembleException(
                f"INV-ENS-002 violation: aleatoric_variance must be > 0.0, got {self.aleatoric_variance}"
            )
        if self.epistemic_variance < 0.0:
            raise DegenerateEnsembleException(
                f"INV-ENS-002 violation: epistemic_variance must be >= 0.0, got {self.epistemic_variance}"
            )
        expected_total = self.aleatoric_variance + self.epistemic_variance
        if not math.isclose(self.total_variance, expected_total, rel_tol=1e-7, abs_tol=1e-10):
            raise DegenerateEnsembleException(
                f"INV-ENS-002 violation: total_variance ({self.total_variance}) != "
                f"aleatoric ({self.aleatoric_variance}) + epistemic ({self.epistemic_variance})"
            )

        # Model weights validation
        if not isinstance(self.model_weights, np.ndarray) or self.model_weights.ndim != 1:
            raise DegenerateEnsembleException("model_weights must be a 1D numpy array")
        k_models = len(self.model_weights)
        if k_models < 1:
            raise DegenerateEnsembleException("model_weights must have length >= 1")
        if not np.all(np.isfinite(self.model_weights)):
            raise DegenerateEnsembleException("model_weights must contain only finite values")
        if np.any(self.model_weights <= 0.0):
            raise DegenerateEnsembleException(
                "INV-ENS-001 violation: model_weights must be strictly positive"
            )
        weights_sum = float(np.sum(self.model_weights))
        if not math.isclose(weights_sum, 1.0, abs_tol=1e-10):
            raise DegenerateEnsembleException(
                f"INV-ENS-001 violation: model_weights sum to {weights_sum}, expected 1.0 +/- 1e-10"
            )

        # Regime probabilities validation
        if not isinstance(
            self.regime_probabilities, np.ndarray
        ) or self.regime_probabilities.shape != (3,):
            raise DegenerateEnsembleException("regime_probabilities must have shape (3,)")
        if not np.all(np.isfinite(self.regime_probabilities)):
            raise DegenerateEnsembleException(
                "regime_probabilities must contain only finite values"
            )
        if np.any(self.regime_probabilities < 0.0):
            raise DegenerateEnsembleException("regime_probabilities elements must be non-negative")
        regime_sum = float(np.sum(self.regime_probabilities))
        if not math.isclose(regime_sum, 1.0, abs_tol=1e-10):
            raise DegenerateEnsembleException(
                f"regime_probabilities sum to {regime_sum}, expected 1.0 +/- 1e-10"
            )

        # Effective models bounds check
        if self.effective_models < 1.0 - 1e-6 or self.effective_models > float(k_models) + 1e-6:
            raise DegenerateEnsembleException(
                f"effective_models must be in [1.0, {k_models}], got {self.effective_models}"
            )

        # Adaptive forgetting factor bounds check
        if not (0.0 < self.volatility_forgetting_factor < 1.0):
            raise DegenerateEnsembleException(
                f"volatility_forgetting_factor must be in (0.0, 1.0), got {self.volatility_forgetting_factor}"
            )

        # Ambiguity shrinkage weight bounds check
        if not (0.0 <= self.ambiguity_shrinkage_weight <= 1.0):
            raise DegenerateEnsembleException(
                f"ambiguity_shrinkage_weight must be in [0.0, 1.0], got {self.ambiguity_shrinkage_weight}"
            )

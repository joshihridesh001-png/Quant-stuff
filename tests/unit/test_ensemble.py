"""Unit tests for Regime-Conditioned Dynamic Model Averaging (RD-DMA) domain entities.

Validates Invariants:
- INV-ENS-001: Strict Simplex Conservation on weights.
- INV-ENS-002: Variance Positivity and Additivity.
- INV-ENS-003: Bounded Adaptive Forgetting hyperparameter bounds.
"""

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from quant.analytics.ensemble import (
    AsymmetricDownsideLossScorer,
    DegenerateEnsembleException,
    EnsembleConfig,
    EnsembleError,
    EnsemblePrediction,
    EnsembleState,
    InvalidPredictionException,
    VolatilityAdaptiveForgetting,
    predict_forward_regime_prior,
    update_regime_posteriors,
)


class TestDomainEntitiesAndInvariants:
    """Tests for EnsembleConfig, EnsembleState, EnsemblePrediction, and exceptions."""

    def test_custom_exception_hierarchy(self) -> None:
        """Verify exception inheritance tree."""
        assert issubclass(EnsembleError, Exception)
        assert issubclass(DegenerateEnsembleException, EnsembleError)
        assert issubclass(InvalidPredictionException, EnsembleError)

        err = DegenerateEnsembleException("variance breach")
        assert isinstance(err, EnsembleError)

    def test_ensemble_config_defaults_and_validation(self) -> None:
        """Verify EnsembleConfig defaults and invariant validation in __post_init__."""
        cfg = EnsembleConfig()
        assert cfg.base_forgetting_factor == 0.96
        assert cfg.volatility_sensitivity == 0.50
        assert cfg.min_forgetting_factor == 0.85
        assert cfg.max_forgetting_factor == 0.99
        assert cfg.orthogonality_penalty == 0.25
        assert cfg.downside_penalty == 2.50
        assert cfg.turnover_damping == 0.15
        assert cfg.ridge_shrinkage == 0.05
        assert cfg.ambiguity_shrinkage_cap == 0.50
        assert cfg.min_weight_floor == 1e-5
        assert cfg.temperature == 1.0
        assert cfg.mirror_descent_lr == 0.50
        assert cfg.mirror_descent_max_iter == 10
        assert cfg.mirror_descent_tol == 1e-6

        # Invariant INV-ENS-003 & boundary checks
        # base_forgetting_factor in (0.80, 0.999)
        with pytest.raises(EnsembleError, match="base_forgetting_factor"):
            EnsembleConfig(base_forgetting_factor=1.5)
        with pytest.raises(EnsembleError, match="base_forgetting_factor"):
            EnsembleConfig(base_forgetting_factor=0.75)
        with pytest.raises(EnsembleError, match="base_forgetting_factor"):
            EnsembleConfig(base_forgetting_factor=0.80)
        with pytest.raises(EnsembleError, match="base_forgetting_factor"):
            EnsembleConfig(base_forgetting_factor=0.999)

        # min_forgetting_factor > 0.50 and < base
        with pytest.raises(EnsembleError, match="min_forgetting_factor"):
            EnsembleConfig(min_forgetting_factor=0.40)
        with pytest.raises(EnsembleError, match="min_forgetting_factor"):
            EnsembleConfig(min_forgetting_factor=0.98, base_forgetting_factor=0.96)

        # max_forgetting_factor > base and < 1.0
        with pytest.raises(EnsembleError, match="max_forgetting_factor"):
            EnsembleConfig(max_forgetting_factor=0.90, base_forgetting_factor=0.96)
        with pytest.raises(EnsembleError, match="max_forgetting_factor"):
            EnsembleConfig(max_forgetting_factor=1.0)

        # min >= max
        with pytest.raises(EnsembleError):
            EnsembleConfig(min_forgetting_factor=0.98, max_forgetting_factor=0.90)

        # volatility_sensitivity >= 0.0
        with pytest.raises(EnsembleError, match="volatility_sensitivity"):
            EnsembleConfig(volatility_sensitivity=-0.1)

        # orthogonality_penalty >= 0.0
        with pytest.raises(EnsembleError, match="orthogonality_penalty"):
            EnsembleConfig(orthogonality_penalty=-0.1)

        # downside_penalty >= 0.0
        with pytest.raises(EnsembleError, match="downside_penalty"):
            EnsembleConfig(downside_penalty=-0.5)

        # turnover_damping in [0.0, 1.0)
        with pytest.raises(EnsembleError, match="turnover_damping"):
            EnsembleConfig(turnover_damping=-0.01)
        with pytest.raises(EnsembleError, match="turnover_damping"):
            EnsembleConfig(turnover_damping=1.0)
        with pytest.raises(EnsembleError, match="turnover_damping"):
            EnsembleConfig(turnover_damping=1.2)

        # ridge_shrinkage in (0.0, 1.0)
        with pytest.raises(EnsembleError, match="ridge_shrinkage"):
            EnsembleConfig(ridge_shrinkage=0.0)
        with pytest.raises(EnsembleError, match="ridge_shrinkage"):
            EnsembleConfig(ridge_shrinkage=1.0)

        # ambiguity_shrinkage_cap in [0.0, 1.0]
        with pytest.raises(EnsembleError, match="ambiguity_shrinkage_cap"):
            EnsembleConfig(ambiguity_shrinkage_cap=-0.1)
        with pytest.raises(EnsembleError, match="ambiguity_shrinkage_cap"):
            EnsembleConfig(ambiguity_shrinkage_cap=1.05)

        # min_weight_floor > 0.0
        with pytest.raises(EnsembleError, match="min_weight_floor"):
            EnsembleConfig(min_weight_floor=0.0)

        # temperature > 0.0
        with pytest.raises(EnsembleError, match="temperature"):
            EnsembleConfig(temperature=-1.0)

        # mirror_descent_lr > 0.0
        with pytest.raises(EnsembleError, match="mirror_descent_lr"):
            EnsembleConfig(mirror_descent_lr=0.0)

        # mirror_descent_max_iter >= 1
        with pytest.raises(EnsembleError, match="mirror_descent_max_iter"):
            EnsembleConfig(mirror_descent_max_iter=0)

        # mirror_descent_tol > 0.0
        with pytest.raises(EnsembleError, match="mirror_descent_tol"):
            EnsembleConfig(mirror_descent_tol=0.0)

        # Non-finite and type validation
        with pytest.raises(EnsembleError):
            EnsembleConfig(base_forgetting_factor=float("nan"))
        with pytest.raises(EnsembleError):
            EnsembleConfig(volatility_sensitivity=float("inf"))

    def test_ensemble_state_immutability_and_validation(self) -> None:
        """Verify EnsembleState initialization, immutability, and INV-ENS-001."""
        K = 5
        weights = np.full(K, 1.0 / K)
        posteriors = np.full((3, K), 1.0 / K)
        losses = np.zeros(K)
        state = EnsembleState(
            step_index=0,
            weights=weights,
            regime_conditional_posteriors=posteriors,
            cumulative_losses=losses,
            effective_models=float(K),
            mean_realized_volatility=0.015,
            last_ambiguity_temperature=1.0,
        )
        assert state.step_index == 0
        assert state.effective_models == pytest.approx(float(K))
        assert np.isclose(np.sum(state.weights), 1.0)
        assert state.weights.shape == (K,)
        assert state.regime_conditional_posteriors.shape == (3, K)
        assert state.cumulative_losses.shape == (K,)

        # Immutability: frozen dataclass
        with pytest.raises(FrozenInstanceError):
            state.step_index = 1  # type: ignore[misc]

        # Invariant INV-ENS-001: simplex sum breach
        with pytest.raises(EnsembleError):
            EnsembleState(
                step_index=0,
                weights=np.array([0.5, 0.2]),
                regime_conditional_posteriors=np.full((3, 2), 0.5),
                cumulative_losses=np.zeros(2),
                effective_models=2.0,
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        # Invariant INV-ENS-001: weights strictly positive
        with pytest.raises(EnsembleError):
            EnsembleState(
                step_index=0,
                weights=np.array([1.0, 0.0]),
                regime_conditional_posteriors=np.full((3, 2), 0.5),
                cumulative_losses=np.zeros(2),
                effective_models=1.0,
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        with pytest.raises(EnsembleError):
            EnsembleState(
                step_index=0,
                weights=np.array([1.1, -0.1]),
                regime_conditional_posteriors=np.full((3, 2), 0.5),
                cumulative_losses=np.zeros(2),
                effective_models=1.0,
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        # Non-array or bad dimension weights
        with pytest.raises(EnsembleError, match="1D numpy array"):
            EnsembleState(
                step_index=0,
                weights=[0.5, 0.5],  # type: ignore[arg-type]
                regime_conditional_posteriors=np.full((3, 2), 0.5),
                cumulative_losses=np.zeros(2),
                effective_models=2.0,
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )
        with pytest.raises(EnsembleError, match="at least 1 model"):
            EnsembleState(
                step_index=0,
                weights=np.array([]),
                regime_conditional_posteriors=np.zeros((3, 0)),
                cumulative_losses=np.zeros(0),
                effective_models=1.0,
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )
        with pytest.raises(DegenerateEnsembleException, match="finite"):
            EnsembleState(
                step_index=0,
                weights=np.array([np.nan, 1.0]),
                regime_conditional_posteriors=np.full((3, 2), 0.5),
                cumulative_losses=np.zeros(2),
                effective_models=2.0,
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        # step_index >= 0
        with pytest.raises(EnsembleError):
            EnsembleState(
                step_index=-1,
                weights=weights,
                regime_conditional_posteriors=posteriors,
                cumulative_losses=losses,
                effective_models=float(K),
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        # Dimension mismatch between posteriors and weights
        with pytest.raises(EnsembleError):
            EnsembleState(
                step_index=0,
                weights=weights,
                regime_conditional_posteriors=np.full((3, K - 1), 1.0 / (K - 1)),
                cumulative_losses=losses,
                effective_models=float(K),
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        # Non-array posteriors
        with pytest.raises(EnsembleError, match="numpy array"):
            EnsembleState(
                step_index=0,
                weights=weights,
                regime_conditional_posteriors=[[1.0 / K] * K] * 3,  # type: ignore[arg-type]
                cumulative_losses=losses,
                effective_models=float(K),
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        # Non-finite posteriors
        bad_nan_posteriors = np.copy(posteriors)
        bad_nan_posteriors[0, 0] = np.nan
        with pytest.raises(DegenerateEnsembleException, match="finite"):
            EnsembleState(
                step_index=0,
                weights=weights,
                regime_conditional_posteriors=bad_nan_posteriors,
                cumulative_losses=losses,
                effective_models=float(K),
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        # Negative posteriors
        bad_neg_posteriors = np.copy(posteriors)
        bad_neg_posteriors[0, 0] = -0.1
        bad_neg_posteriors[0, 1] += 0.1 + (1.0 / K)
        with pytest.raises(EnsembleError, match="non-negative"):
            EnsembleState(
                step_index=0,
                weights=weights,
                regime_conditional_posteriors=bad_neg_posteriors,
                cumulative_losses=losses,
                effective_models=float(K),
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        # Posterior rows must sum to 1.0
        bad_posteriors = np.copy(posteriors)
        bad_posteriors[0, :] = 0.5
        with pytest.raises(EnsembleError):
            EnsembleState(
                step_index=0,
                weights=weights,
                regime_conditional_posteriors=bad_posteriors,
                cumulative_losses=losses,
                effective_models=float(K),
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        # Dimension mismatch: cumulative_losses
        with pytest.raises(EnsembleError):
            EnsembleState(
                step_index=0,
                weights=weights,
                regime_conditional_posteriors=posteriors,
                cumulative_losses=np.zeros(K + 1),
                effective_models=float(K),
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        # Non-array cumulative_losses
        with pytest.raises(EnsembleError, match="numpy array"):
            EnsembleState(
                step_index=0,
                weights=weights,
                regime_conditional_posteriors=posteriors,
                cumulative_losses=list(losses),  # type: ignore[arg-type]
                effective_models=float(K),
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        # Non-finite cumulative_losses
        with pytest.raises(DegenerateEnsembleException, match="finite"):
            EnsembleState(
                step_index=0,
                weights=weights,
                regime_conditional_posteriors=posteriors,
                cumulative_losses=np.array([np.inf] + [0.0] * (K - 1)),
                effective_models=float(K),
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        # effective_models bounds [1.0, K]
        with pytest.raises(EnsembleError):
            EnsembleState(
                step_index=0,
                weights=weights,
                regime_conditional_posteriors=posteriors,
                cumulative_losses=losses,
                effective_models=0.5,
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )
        with pytest.raises(EnsembleError):
            EnsembleState(
                step_index=0,
                weights=weights,
                regime_conditional_posteriors=posteriors,
                cumulative_losses=losses,
                effective_models=float(K + 1),
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )
        with pytest.raises(DegenerateEnsembleException, match="finite"):
            EnsembleState(
                step_index=0,
                weights=weights,
                regime_conditional_posteriors=posteriors,
                cumulative_losses=losses,
                effective_models=float("nan"),
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

        # mean_realized_volatility > 0.0
        with pytest.raises(EnsembleError):
            EnsembleState(
                step_index=0,
                weights=weights,
                regime_conditional_posteriors=posteriors,
                cumulative_losses=losses,
                effective_models=float(K),
                mean_realized_volatility=0.0,
                last_ambiguity_temperature=1.0,
            )

        # last_ambiguity_temperature > 0.0
        with pytest.raises(EnsembleError):
            EnsembleState(
                step_index=0,
                weights=weights,
                regime_conditional_posteriors=posteriors,
                cumulative_losses=losses,
                effective_models=float(K),
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=-0.5,
            )

    def test_ensemble_prediction_variance_additivity(self) -> None:
        """Verify EnsemblePrediction and invariant INV-ENS-002 (variance additivity)."""
        pred = EnsemblePrediction(
            point_prediction=0.0025,
            aleatoric_variance=0.0004,
            epistemic_variance=0.0001,
            total_variance=0.0005,
            model_weights=np.array([0.5, 0.5]),
            regime_probabilities=np.array([0.7, 0.2, 0.1]),
            effective_models=2.0,
            volatility_forgetting_factor=0.96,
            ambiguity_shrinkage_weight=0.0,
        )
        # Invariant INV-ENS-002: total == aleatoric + epistemic
        assert pred.total_variance == pytest.approx(
            pred.aleatoric_variance + pred.epistemic_variance
        )
        assert math.isclose(
            pred.total_variance,
            pred.aleatoric_variance + pred.epistemic_variance,
            rel_tol=1e-7,
            abs_tol=1e-10,
        )

        # NaN rejection
        with pytest.raises(DegenerateEnsembleException):
            EnsemblePrediction(
                point_prediction=np.nan,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )

        # Inf rejection
        with pytest.raises(DegenerateEnsembleException):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=float("inf"),
                epistemic_variance=0.0001,
                total_variance=float("inf"),
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )

        # Invariant INV-ENS-002: aleatoric_variance must be > 0
        with pytest.raises(DegenerateEnsembleException):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0,
                epistemic_variance=0.0001,
                total_variance=0.0001,
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )

        # Invariant INV-ENS-002: epistemic_variance must be >= 0
        with pytest.raises(DegenerateEnsembleException):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=-0.0001,
                total_variance=0.0003,
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )

        # Invariant INV-ENS-002: variance additivity breach
        with pytest.raises(DegenerateEnsembleException):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0010,  # Should be 0.0005
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )

        # Model weights type, empty, non-finite
        with pytest.raises(DegenerateEnsembleException, match="1D numpy array"):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=[0.5, 0.5],  # type: ignore[arg-type]
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )
        with pytest.raises(DegenerateEnsembleException, match="length >= 1"):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=1.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )
        with pytest.raises(DegenerateEnsembleException, match="finite"):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([np.nan, 1.0]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )

        # Model weights simplex breach
        with pytest.raises(DegenerateEnsembleException):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([0.6, 0.6]),  # sum != 1.0
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )

        # Model weights negative component breach
        with pytest.raises(DegenerateEnsembleException):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([1.1, -0.1]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )

        # Regime probabilities shape, non-finite, negative, sum breach
        with pytest.raises(DegenerateEnsembleException):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([0.5, 0.5]),  # length != 3
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )
        with pytest.raises(DegenerateEnsembleException, match="finite"):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([np.nan, 0.5, 0.5]),
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )
        with pytest.raises(DegenerateEnsembleException, match="non-negative"):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([-0.1, 0.6, 0.5]),
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )
        with pytest.raises(DegenerateEnsembleException):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([0.5, 0.3, 0.1]),  # sum != 1.0
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )

        # Effective models bounds check
        with pytest.raises(DegenerateEnsembleException, match="effective_models"):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=0.5,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )
        with pytest.raises(DegenerateEnsembleException, match="effective_models"):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=3.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=0.0,
            )

        # volatility_forgetting_factor in (0.0, 1.0)
        with pytest.raises(DegenerateEnsembleException, match="volatility_forgetting_factor"):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=2.0,
                volatility_forgetting_factor=0.0,
                ambiguity_shrinkage_weight=0.0,
            )
        with pytest.raises(DegenerateEnsembleException, match="volatility_forgetting_factor"):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=2.0,
                volatility_forgetting_factor=1.0,
                ambiguity_shrinkage_weight=0.0,
            )

        # ambiguity_shrinkage_weight in [0.0, 1.0]
        with pytest.raises(DegenerateEnsembleException, match="ambiguity_shrinkage_weight"):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=-0.1,
            )
        with pytest.raises(DegenerateEnsembleException, match="ambiguity_shrinkage_weight"):
            EnsemblePrediction(
                point_prediction=0.0025,
                aleatoric_variance=0.0004,
                epistemic_variance=0.0001,
                total_variance=0.0005,
                model_weights=np.array([0.5, 0.5]),
                regime_probabilities=np.array([0.7, 0.2, 0.1]),
                effective_models=2.0,
                volatility_forgetting_factor=0.96,
                ambiguity_shrinkage_weight=1.1,
            )


class TestAdaptiveForgetting:
    """Unit tests for VolatilityAdaptiveForgetting enforcing Invariant INV-ENS-003."""

    def test_init_validation(self) -> None:
        """Verify constructor validates initial volatility and smoothing parameter."""
        cfg = EnsembleConfig()

        # Valid initialization
        vaf = VolatilityAdaptiveForgetting(cfg, initial_mean_vol=0.015, vol_smoothing=0.10)
        assert vaf.mean_realized_volatility == pytest.approx(0.015)

        # Invalid initial_mean_vol <= 0 or non-finite
        with pytest.raises(EnsembleError, match="initial_mean_vol"):
            VolatilityAdaptiveForgetting(cfg, initial_mean_vol=0.0)
        with pytest.raises(EnsembleError, match="initial_mean_vol"):
            VolatilityAdaptiveForgetting(cfg, initial_mean_vol=-0.01)
        with pytest.raises(EnsembleError, match="initial_mean_vol"):
            VolatilityAdaptiveForgetting(cfg, initial_mean_vol=float("nan"))
        with pytest.raises(EnsembleError, match="initial_mean_vol"):
            VolatilityAdaptiveForgetting(cfg, initial_mean_vol=float("inf"))

        # Invalid vol_smoothing not in (0, 1] or non-finite
        with pytest.raises(EnsembleError, match="vol_smoothing"):
            VolatilityAdaptiveForgetting(cfg, vol_smoothing=0.0)
        with pytest.raises(EnsembleError, match="vol_smoothing"):
            VolatilityAdaptiveForgetting(cfg, vol_smoothing=-0.1)
        with pytest.raises(EnsembleError, match="vol_smoothing"):
            VolatilityAdaptiveForgetting(cfg, vol_smoothing=1.05)
        with pytest.raises(EnsembleError, match="vol_smoothing"):
            VolatilityAdaptiveForgetting(cfg, vol_smoothing=float("nan"))

    def test_alpha_expansion_in_calm_markets(self) -> None:
        """Verify memory expands (alpha -> alpha_max) when realized vol is below baseline."""
        cfg = EnsembleConfig(
            base_forgetting_factor=0.96,
            volatility_sensitivity=0.50,
            min_forgetting_factor=0.85,
            max_forgetting_factor=0.99,
        )
        vaf = VolatilityAdaptiveForgetting(cfg, initial_mean_vol=0.02, vol_smoothing=0.10)

        # Calm market: realized_vol = 0.01 < baseline 0.02
        # delta_sigma = (0.01 - 0.02) / 0.02 = -0.50
        # raw_alpha = 0.96 - 0.50 * (-0.50) = 0.96 + 0.25 = 1.21 -> clamped to 0.99
        alpha = vaf.compute_alpha(realized_vol=0.01)
        assert alpha > cfg.base_forgetting_factor
        assert alpha == pytest.approx(cfg.max_forgetting_factor)

    def test_alpha_contraction_during_panic_shocks(self) -> None:
        """Verify memory contracts (alpha -> alpha_min) during volatility panic spikes."""
        cfg = EnsembleConfig(
            base_forgetting_factor=0.96,
            volatility_sensitivity=0.50,
            min_forgetting_factor=0.85,
            max_forgetting_factor=0.99,
        )
        vaf = VolatilityAdaptiveForgetting(cfg, initial_mean_vol=0.01, vol_smoothing=0.10)

        # Panic shock: realized_vol = 0.04 >> baseline 0.01
        # delta_sigma = (0.04 - 0.01) / 0.01 = 3.0
        # raw_alpha = 0.96 - 0.50 * 3.0 = 0.96 - 1.50 = -0.54 -> clamped to 0.85
        alpha = vaf.compute_alpha(realized_vol=0.04)
        assert alpha < cfg.base_forgetting_factor
        assert alpha == pytest.approx(cfg.min_forgetting_factor)

    def test_exact_invariant_clamping_inv_ens_003(self) -> None:
        """Verify INV-ENS-003: alpha strictly remains in [alpha_min, alpha_max] under extreme inputs."""
        cfg = EnsembleConfig(
            base_forgetting_factor=0.96,
            volatility_sensitivity=0.50,
            min_forgetting_factor=0.85,
            max_forgetting_factor=0.99,
        )
        vaf = VolatilityAdaptiveForgetting(cfg, initial_mean_vol=0.01)

        # Extremely low vol: 1e-8
        alpha_low = vaf.compute_alpha(1e-8)
        assert alpha_low == pytest.approx(cfg.max_forgetting_factor)

        # Extremely high vol: 100.0
        alpha_high = vaf.compute_alpha(100.0)
        assert alpha_high == pytest.approx(cfg.min_forgetting_factor)

        # Exactly at baseline: delta_sigma = 0 -> alpha == base_forgetting_factor
        current_baseline = vaf.mean_realized_volatility
        alpha_base = vaf.compute_alpha(current_baseline)
        assert alpha_base == pytest.approx(cfg.base_forgetting_factor)

    def test_running_volatility_baseline_exponential_smoothing(self) -> None:
        """Verify baseline volatility updates via exponential smoothing after alpha computation."""
        cfg = EnsembleConfig(
            base_forgetting_factor=0.96,
            volatility_sensitivity=0.50,
            min_forgetting_factor=0.85,
            max_forgetting_factor=0.99,
        )
        beta = 0.20
        vaf = VolatilityAdaptiveForgetting(cfg, initial_mean_vol=0.010, vol_smoothing=beta)

        # Step 1: realized_vol = 0.015
        # Expected new baseline: (1 - 0.20) * 0.010 + 0.20 * 0.015 = 0.008 + 0.003 = 0.011
        vaf.compute_alpha(0.015)
        assert vaf.mean_realized_volatility == pytest.approx(0.011)

        # Step 2: realized_vol = 0.021
        # Expected new baseline: (1 - 0.20) * 0.011 + 0.20 * 0.021 = 0.0088 + 0.0042 = 0.013
        vaf.compute_alpha(0.021)
        assert vaf.mean_realized_volatility == pytest.approx(0.013)

    def test_compute_alpha_invalid_inputs(self) -> None:
        """Verify defensive handling of invalid realized volatility inputs."""
        cfg = EnsembleConfig()
        vaf = VolatilityAdaptiveForgetting(cfg, initial_mean_vol=0.01)

        # Realized vol <= 0
        with pytest.raises(EnsembleError, match="realized_vol"):
            vaf.compute_alpha(0.0)
        with pytest.raises(EnsembleError, match="realized_vol"):
            vaf.compute_alpha(-0.05)

        # Realized vol NaN or Inf
        with pytest.raises(DegenerateEnsembleException, match="realized_vol"):
            vaf.compute_alpha(float("nan"))
        with pytest.raises(DegenerateEnsembleException, match="realized_vol"):
            vaf.compute_alpha(float("inf"))


class TestPredictiveForwardMarkovTransitions:
    """Unit tests for predict_forward_regime_prior and simplex conservation."""

    def test_predictive_forward_markov_projection(self) -> None:
        """Verify forward transition calculation p_{t+1|t} = P_trans^T * p_t."""
        # Row-stochastic transition matrix: P_trans[i, j] = P(S_{t+1}=j | S_t=i)
        P_trans = np.array(
            [
                [0.80, 0.15, 0.05],
                [0.10, 0.70, 0.20],
                [0.05, 0.05, 0.90],
            ]
        )
        p_current = np.array([0.10, 0.80, 0.10])  # Mostly Momentum (regime 1)

        # Expected:
        # p_next[0] = 0.80 * 0.10 + 0.10 * 0.80 + 0.05 * 0.10 = 0.165
        # p_next[1] = 0.15 * 0.10 + 0.70 * 0.80 + 0.05 * 0.10 = 0.580
        # p_next[2] = 0.05 * 0.10 + 0.20 * 0.80 + 0.90 * 0.10 = 0.255
        p_next = predict_forward_regime_prior(p_current, P_trans)

        assert p_next.shape == (3,)
        assert np.isclose(np.sum(p_next), 1.0, atol=1e-10)
        assert p_next[0] == pytest.approx(0.165)
        assert p_next[1] == pytest.approx(0.580)
        assert p_next[2] == pytest.approx(0.255)
        # Momentum regime transition risk increases panic probability
        assert p_next[2] > p_current[2]

    def test_identity_transition_preserves_distribution(self) -> None:
        """Verify identity transition matrix leaves regime distribution unchanged."""
        P_identity = np.eye(3)
        p_current = np.array([0.25, 0.50, 0.25])
        p_next = predict_forward_regime_prior(p_current, P_identity)

        assert np.allclose(p_next, p_current)
        assert np.isclose(np.sum(p_next), 1.0)

    def test_predict_forward_regime_prior_validation(self) -> None:
        """Verify defensive validation on current_regime_probs and transition_matrix."""
        valid_p = np.array([0.70, 0.20, 0.10])
        valid_P = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.1, 0.8],
            ]
        )

        # Invalid type or shape for current_regime_probs
        with pytest.raises(InvalidPredictionException, match="current_regime_probs"):
            predict_forward_regime_prior([0.7, 0.2, 0.1], valid_P)  # type: ignore[arg-type]
        with pytest.raises(InvalidPredictionException, match="current_regime_probs"):
            predict_forward_regime_prior(np.array([0.5, 0.5]), valid_P)
        with pytest.raises(InvalidPredictionException, match="current_regime_probs"):
            predict_forward_regime_prior(np.zeros((3, 1)), valid_P)

        # Non-finite or negative current_regime_probs
        with pytest.raises(DegenerateEnsembleException, match="current_regime_probs"):
            predict_forward_regime_prior(np.array([np.nan, 0.5, 0.5]), valid_P)
        with pytest.raises(InvalidPredictionException, match="current_regime_probs"):
            predict_forward_regime_prior(np.array([-0.1, 0.6, 0.5]), valid_P)

        # Sum != 1.0 for current_regime_probs
        with pytest.raises(InvalidPredictionException, match="current_regime_probs"):
            predict_forward_regime_prior(np.array([0.5, 0.5, 0.5]), valid_P)

        # Invalid type or shape for transition_matrix
        with pytest.raises(InvalidPredictionException, match="transition_matrix"):
            predict_forward_regime_prior(valid_p, valid_P.tolist())  # type: ignore[arg-type]
        with pytest.raises(InvalidPredictionException, match="transition_matrix"):
            predict_forward_regime_prior(valid_p, np.eye(2))
        with pytest.raises(InvalidPredictionException, match="transition_matrix"):
            predict_forward_regime_prior(valid_p, np.zeros((3, 3, 1)))

        # Non-finite or negative transition_matrix
        with pytest.raises(DegenerateEnsembleException, match="transition_matrix"):
            bad_nan_P = np.copy(valid_P)
            bad_nan_P[0, 0] = np.nan
            predict_forward_regime_prior(valid_p, bad_nan_P)

        with pytest.raises(InvalidPredictionException, match="transition_matrix"):
            bad_neg_P = np.copy(valid_P)
            bad_neg_P[0, 0] = -0.1
            bad_neg_P[0, 1] += 0.1
            predict_forward_regime_prior(valid_p, bad_neg_P)

        # Row sum != 1.0 for transition_matrix
        with pytest.raises(InvalidPredictionException, match="transition_matrix"):
            bad_sum_P = np.copy(valid_P)
            bad_sum_P[0, :] = [0.5, 0.1, 0.1]
            predict_forward_regime_prior(valid_p, bad_sum_P)


class TestUpdateRegimePosteriors:
    """Unit tests for update_regime_posteriors enforcing tempered update and INV-ENS-001."""

    def test_update_regime_posteriors_tempering_and_composition(self) -> None:
        """Verify tempered posteriors update and composite prior calculation."""
        K = 4
        # 3 regimes x 4 models
        posteriors = np.array(
            [
                [0.40, 0.30, 0.20, 0.10],  # Regime 0 (Absorption)
                [0.10, 0.50, 0.30, 0.10],  # Regime 1 (Momentum)
                [0.10, 0.10, 0.20, 0.60],  # Regime 2 (Panic)
            ]
        )
        forward_p = np.array([0.20, 0.30, 0.50])
        alpha_t = 0.95

        updated_posteriors, composite_prior = update_regime_posteriors(
            posteriors, alpha_t, forward_p
        )

        # Validate shapes
        assert updated_posteriors.shape == (3, K)
        assert composite_prior.shape == (K,)

        # Validate INV-ENS-001 (Simplex conservation on each row and composite prior)
        for row_idx in range(3):
            row = updated_posteriors[row_idx, :]
            assert np.all(row > 0.0)
            assert np.isclose(np.sum(row), 1.0, atol=1e-10)

        assert np.all(composite_prior > 0.0)
        assert np.isclose(np.sum(composite_prior), 1.0, atol=1e-10)

        # In Panic regime (weight 0.50), model 3 is dominant (0.60 prior)
        # Therefore model 3 should have a significant share in the composite prior
        assert composite_prior[3] > composite_prior[0]

    def test_alpha_tempering_entropy_effect(self) -> None:
        """Verify lower alpha (higher forgetting) flattens distribution toward uniform (increases entropy)."""
        K = 3
        # Skewed posteriors
        posteriors = np.array(
            [
                [0.80, 0.15, 0.05],
                [0.80, 0.15, 0.05],
                [0.80, 0.15, 0.05],
            ]
        )
        forward_p = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

        # High alpha (memory retention, minimal flattening)
        up_high, comp_high = update_regime_posteriors(
            posteriors, alpha_t=0.99, forward_regime_probs=forward_p
        )
        assert up_high.shape == (3, K)
        assert comp_high.shape == (K,)

        # Low alpha (rapid adaptation, significant flattening / tempering)
        up_low, comp_low = update_regime_posteriors(
            posteriors, alpha_t=0.85, forward_regime_probs=forward_p
        )
        assert up_low.shape == (3, K)
        assert comp_low.shape == (K,)

        # Top model weight should be higher with alpha=0.99 than with alpha=0.85
        assert comp_high[0] > comp_low[0]
        # Trailing model weight should be higher under lower alpha (tempered toward uniform)
        assert comp_low[2] > comp_high[2]

    def test_numerical_stability_with_extreme_likelihoods(self) -> None:
        """Verify log-space max-shift prevents underflow/overflow with near-zero priors."""
        K = 3
        posteriors = np.array(
            [
                [1.0 - 1e-15, 5e-16, 5e-16],
                [1e-25, 1.0 - 2e-25, 1e-25],
                [1e-30, 1e-30, 1.0 - 2e-30],
            ]
        )
        forward_p = np.array([0.33, 0.33, 0.34])
        alpha_t = 0.90

        updated_posteriors, composite_prior = update_regime_posteriors(
            posteriors, alpha_t, forward_p
        )

        assert updated_posteriors.shape == (3, K)
        assert composite_prior.shape == (K,)
        assert np.all(np.isfinite(updated_posteriors))
        assert np.all(np.isfinite(composite_prior))
        assert np.all(updated_posteriors >= 0.0)
        assert np.all(composite_prior >= 0.0)
        for r in range(3):
            assert np.isclose(np.sum(updated_posteriors[r, :]), 1.0, atol=1e-10)
        assert np.isclose(np.sum(composite_prior), 1.0, atol=1e-10)

    def test_update_regime_posteriors_validation(self) -> None:
        """Verify defensive checks on posteriors, alpha_t, and forward_regime_probs."""
        valid_post = np.full((3, 4), 0.25)
        valid_fp = np.array([0.5, 0.3, 0.2])
        valid_alpha = 0.96

        # Invalid posteriors type / dimensions
        with pytest.raises(InvalidPredictionException, match="posteriors"):
            update_regime_posteriors([[0.25] * 4] * 3, valid_alpha, valid_fp)  # type: ignore[arg-type]
        with pytest.raises(InvalidPredictionException, match="posteriors"):
            update_regime_posteriors(np.full((2, 4), 0.25), valid_alpha, valid_fp)
        with pytest.raises(InvalidPredictionException, match="posteriors"):
            update_regime_posteriors(np.zeros((3, 0)), valid_alpha, valid_fp)

        # Non-finite posteriors
        with pytest.raises(DegenerateEnsembleException, match="posteriors"):
            bad_post = np.copy(valid_post)
            bad_post[0, 0] = np.nan
            update_regime_posteriors(bad_post, valid_alpha, valid_fp)

        # Negative posteriors
        with pytest.raises(InvalidPredictionException, match="posteriors"):
            bad_post = np.copy(valid_post)
            bad_post[0, 0] = -0.1
            bad_post[0, 1] += 0.1
            update_regime_posteriors(bad_post, valid_alpha, valid_fp)

        # Row sum != 1.0 in posteriors
        with pytest.raises(InvalidPredictionException, match="posteriors"):
            bad_post = np.copy(valid_post)
            bad_post[0, :] = 0.20
            update_regime_posteriors(bad_post, valid_alpha, valid_fp)

        # Invalid alpha_t
        with pytest.raises(DegenerateEnsembleException, match="alpha_t"):
            update_regime_posteriors(valid_post, float("nan"), valid_fp)
        with pytest.raises(DegenerateEnsembleException, match="alpha_t"):
            update_regime_posteriors(valid_post, 0.0, valid_fp)
        with pytest.raises(DegenerateEnsembleException, match="alpha_t"):
            update_regime_posteriors(valid_post, 1.2, valid_fp)
        with pytest.raises(DegenerateEnsembleException, match="alpha_t"):
            update_regime_posteriors(valid_post, 0.40, valid_fp, alpha_min=0.50)

        # Invalid forward_regime_probs
        with pytest.raises(InvalidPredictionException, match="forward_regime_probs"):
            update_regime_posteriors(valid_post, valid_alpha, np.array([0.5, 0.5]))
        with pytest.raises(DegenerateEnsembleException, match="forward_regime_probs"):
            update_regime_posteriors(valid_post, valid_alpha, np.array([np.nan, 0.5, 0.5]))
        with pytest.raises(InvalidPredictionException, match="forward_regime_probs"):
            update_regime_posteriors(valid_post, valid_alpha, np.array([-0.1, 0.6, 0.5]))
        with pytest.raises(InvalidPredictionException, match="forward_regime_probs"):
            update_regime_posteriors(valid_post, valid_alpha, np.array([0.4, 0.4, 0.4]))


class TestAsymmetricDownsideLossScorer:
    """Unit tests for AsymmetricDownsideLossScorer and downside semi-variance."""

    def test_init_validation(self) -> None:
        """Verify default and custom initialization contracts for AsymmetricDownsideLossScorer."""
        scorer = AsymmetricDownsideLossScorer()
        assert scorer.downside_penalty == 2.50

        custom_scorer = AsymmetricDownsideLossScorer(downside_penalty=1.75)
        assert custom_scorer.downside_penalty == 1.75

        # Non-negative validation
        with pytest.raises(EnsembleError, match="downside_penalty"):
            AsymmetricDownsideLossScorer(downside_penalty=-0.1)

        # Finiteness validation
        with pytest.raises(EnsembleError, match="downside_penalty"):
            AsymmetricDownsideLossScorer(downside_penalty=float("nan"))
        with pytest.raises(EnsembleError, match="downside_penalty"):
            AsymmetricDownsideLossScorer(downside_penalty=float("inf"))
        with pytest.raises(EnsembleError, match="downside_penalty"):
            AsymmetricDownsideLossScorer(downside_penalty="not_a_number")  # type: ignore[arg-type]

    def test_compute_losses_symmetric_when_signs_agree(self) -> None:
        """Verify when signs agree (y_t * y_tilde >= 0), loss is exact squared error without downside penalty."""
        scorer = AsymmetricDownsideLossScorer(downside_penalty=2.50)

        # Case 1: Both positive
        y_realized = 0.04
        predictions = np.array([0.01, 0.04, 0.08])
        losses = scorer.compute_losses(predictions, y_realized)
        expected = (y_realized - predictions) ** 2
        assert np.allclose(losses, expected)

        # Case 2: Both negative
        y_realized_neg = -0.03
        predictions_neg = np.array([-0.01, -0.03, -0.05])
        losses_neg = scorer.compute_losses(predictions_neg, y_realized_neg)
        expected_neg = (y_realized_neg - predictions_neg) ** 2
        assert np.allclose(losses_neg, expected_neg)

        # Case 3: Zero boundary
        losses_zero = scorer.compute_losses(np.array([0.0, 0.05]), realized_return=0.0)
        assert np.allclose(losses_zero, np.array([0.0, 0.0025]))

    def test_compute_losses_asymmetric_downside_penalty(self) -> None:
        """Verify directional drawdown penalty amplifies loss when y_t * y_tilde < 0."""
        scorer = AsymmetricDownsideLossScorer(downside_penalty=2.50)

        y_realized = -0.05
        # Model 0: False long (+0.05)
        # Model 1: Correct short (-0.05)
        # Model 2: Short with same absolute error as Model 0 (-0.15, diff = 0.10)
        predictions = np.array([0.05, -0.05, -0.15])
        losses = scorer.compute_losses(predictions, y_realized)

        # Model 0:
        # squared_error = (-0.05 - 0.05)^2 = 0.01
        # asymmetric penalty = 2.50 * max(0, -(-0.05) * 0.05) = 2.50 * 0.0025 = 0.00625
        # total = 0.01625
        assert losses[0] == pytest.approx(0.01625)

        # Model 1:
        # squared_error = (-0.05 - (-0.05))^2 = 0.0
        # asymmetric penalty = 0.0
        assert losses[1] == pytest.approx(0.0)

        # Model 2:
        # squared_error = (-0.05 - (-0.15))^2 = (0.10)^2 = 0.01
        # asymmetric penalty = 0.0 (both signs negative)
        assert losses[2] == pytest.approx(0.01)

        # Crucial Invariant: Directionally wrong prediction (Model 0) receives strictly higher loss
        # than sign-agreeing prediction with same error magnitude (Model 2)
        assert losses[0] > losses[2]
        assert losses[0] - losses[2] == pytest.approx(0.00625)

    def test_compute_losses_multi_horizon_scaling(self) -> None:
        """Verify forecast horizon scaling y_tilde = y_hat / sqrt(H) normalizes predictions."""
        scorer = AsymmetricDownsideLossScorer(downside_penalty=2.50)

        # Horizons: H = [1, 4, 9], sqrt(H) = [1, 2, 3]
        horizons = np.array([1.0, 4.0, 9.0])
        nominal_predictions = np.array([0.04, 0.08, 0.12])
        # Normalized predictions: [0.04/1, 0.08/2, 0.12/3] = [0.04, 0.04, 0.04]
        y_realized = 0.04

        losses = scorer.compute_losses(
            nominal_predictions, y_realized, forecast_horizons=horizons
        )
        assert losses.shape == (3,)
        # All models have scaled prediction matching realized return exactly -> 0 loss
        assert np.allclose(losses, np.zeros(3))

        # Realized return = -0.04 with horizons
        losses_drawdown = scorer.compute_losses(
            nominal_predictions, -0.04, forecast_horizons=horizons
        )
        # All models should produce identical normalized loss
        assert math.isclose(losses_drawdown[0], losses_drawdown[1])
        assert math.isclose(losses_drawdown[1], losses_drawdown[2])

    def test_compute_losses_non_negative_invariant(self) -> None:
        """Verify Invariant: loss l_{t, k} >= 0.0 across random realizations."""
        scorer = AsymmetricDownsideLossScorer(downside_penalty=5.0)
        rng = np.random.default_rng(42)

        for _ in range(20):
            preds = rng.normal(0.0, 0.05, size=10)
            y_realized = float(rng.normal(0.0, 0.05))
            horizons = rng.uniform(1.0, 20.0, size=10)

            losses = scorer.compute_losses(preds, y_realized, forecast_horizons=horizons)
            assert losses.shape == (10,)
            assert np.all(losses >= 0.0)
            assert np.all(np.isfinite(losses))

    def test_compute_losses_validation(self) -> None:
        """Verify defensive input validation on compute_losses."""
        scorer = AsymmetricDownsideLossScorer()
        valid_preds = np.array([0.01, 0.02, 0.03])
        valid_y = 0.015

        # Invalid predictions
        with pytest.raises(InvalidPredictionException, match="predictions"):
            scorer.compute_losses([0.01, 0.02], valid_y)  # type: ignore[arg-type]
        with pytest.raises(InvalidPredictionException, match="predictions"):
            scorer.compute_losses(np.array([[0.01, 0.02]]), valid_y)
        with pytest.raises(InvalidPredictionException, match="predictions"):
            scorer.compute_losses(np.array([]), valid_y)
        with pytest.raises(DegenerateEnsembleException, match="predictions"):
            scorer.compute_losses(np.array([0.01, np.nan]), valid_y)
        with pytest.raises(DegenerateEnsembleException, match="predictions"):
            scorer.compute_losses(np.array([0.01, float("inf")]), valid_y)

        # Invalid realized_return
        with pytest.raises(DegenerateEnsembleException, match="realized_return"):
            scorer.compute_losses(valid_preds, float("nan"))
        with pytest.raises(DegenerateEnsembleException, match="realized_return"):
            scorer.compute_losses(valid_preds, float("inf"))

        # Invalid forecast_horizons
        with pytest.raises(InvalidPredictionException, match="forecast_horizons"):
            scorer.compute_losses(valid_preds, valid_y, forecast_horizons=[1.0, 1.0, 1.0])  # type: ignore[arg-type]
        with pytest.raises(InvalidPredictionException, match="forecast_horizons"):
            scorer.compute_losses(
                valid_preds, valid_y, forecast_horizons=np.array([1.0, 2.0])
            )
        with pytest.raises(InvalidPredictionException, match="forecast_horizons"):
            scorer.compute_losses(
                valid_preds, valid_y, forecast_horizons=np.array([1.0, 0.5, 2.0])
            )
        with pytest.raises(DegenerateEnsembleException, match="forecast_horizons"):
            scorer.compute_losses(
                valid_preds, valid_y, forecast_horizons=np.array([1.0, np.nan, 2.0])
            )

    def test_compute_downside_semi_variance(self) -> None:
        """Verify empirical downside semi-variance isolates negative deviations."""
        scorer = AsymmetricDownsideLossScorer()

        # Returns: [-0.02, 0.04, -0.04, 0.06]
        # Target = 0.0
        # Deviations: min(0, r - 0) = [-0.02, 0.0, -0.04, 0.0]
        # Squared: [0.0004, 0.0, 0.0016, 0.0] -> sum = 0.0020
        # Mean = 0.0020 / 4 = 0.0005
        returns = np.array([-0.02, 0.04, -0.04, 0.06])
        semi_var = scorer.compute_downside_semi_variance(returns, target_return=0.0)
        assert semi_var == pytest.approx(0.0005)

        # Target = 0.01
        # Deviations: [min(0, -0.02 - 0.01), min(0, 0.04 - 0.01), min(0, -0.04 - 0.01), min(0, 0.06 - 0.01)]
        #           = [-0.03, 0.0, -0.05, 0.0]
        # Squared: [0.0009, 0.0, 0.0025, 0.0] -> sum = 0.0034
        # Mean = 0.0034 / 4 = 0.00085
        semi_var_target = scorer.compute_downside_semi_variance(returns, target_return=0.01)
        assert semi_var_target == pytest.approx(0.00085)

    def test_compute_downside_semi_variance_pure_upside_floor(self) -> None:
        """Verify pure upside returns trigger defensive variance floor > 0.0."""
        scorer = AsymmetricDownsideLossScorer()
        pure_upside = np.array([0.02, 0.05, 0.01, 0.08])

        floor = 1e-8
        semi_var = scorer.compute_downside_semi_variance(
            pure_upside, target_return=0.0, min_variance_floor=floor
        )
        assert semi_var == floor
        assert semi_var > 0.0

        # Custom floor
        custom_floor = 1e-5
        semi_var_custom = scorer.compute_downside_semi_variance(
            pure_upside, target_return=0.0, min_variance_floor=custom_floor
        )
        assert semi_var_custom == custom_floor

    def test_compute_downside_semi_variance_validation(self) -> None:
        """Verify defensive validations on compute_downside_semi_variance."""
        scorer = AsymmetricDownsideLossScorer()
        valid_ret = np.array([0.01, -0.02, 0.03])

        # Invalid returns
        with pytest.raises(InvalidPredictionException, match="returns"):
            scorer.compute_downside_semi_variance([0.01, -0.02])  # type: ignore[arg-type]
        with pytest.raises(InvalidPredictionException, match="returns"):
            scorer.compute_downside_semi_variance(np.array([[0.01], [-0.02]]))
        with pytest.raises(InvalidPredictionException, match="returns"):
            scorer.compute_downside_semi_variance(np.array([]))
        with pytest.raises(DegenerateEnsembleException, match="returns"):
            scorer.compute_downside_semi_variance(np.array([0.01, np.nan]))

        # Invalid target_return
        with pytest.raises(DegenerateEnsembleException, match="target_return"):
            scorer.compute_downside_semi_variance(valid_ret, target_return=float("nan"))

        # Invalid min_variance_floor
        with pytest.raises(EnsembleError, match="min_variance_floor"):
            scorer.compute_downside_semi_variance(valid_ret, min_variance_floor=0.0)
        with pytest.raises(EnsembleError, match="min_variance_floor"):
            scorer.compute_downside_semi_variance(valid_ret, min_variance_floor=-1e-5)
        with pytest.raises(DegenerateEnsembleException, match="min_variance_floor"):
            scorer.compute_downside_semi_variance(valid_ret, min_variance_floor=float("inf"))

    def test_compute_cohort_downside_variances(self) -> None:
        """Verify cohort downside variance evaluation across 2D return matrices."""
        scorer = AsymmetricDownsideLossScorer()

        # 4 time steps x 3 models
        # Model 0: mixed [-0.02, 0.04, -0.04, 0.06] -> semi_var = 0.0005
        # Model 1: pure upside [0.01, 0.02, 0.03, 0.04] -> clamped to floor 1e-8
        # Model 2: pure downside [-0.01, -0.02, -0.03, -0.04] -> mean([-0.01^2, -0.02^2, -0.03^2, -0.04^2])
        #          = (0.0001 + 0.0004 + 0.0009 + 0.0016) / 4 = 0.0030 / 4 = 0.00075
        matrix = np.array(
            [
                [-0.02, 0.01, -0.01],
                [0.04, 0.02, -0.02],
                [-0.04, 0.03, -0.03],
                [0.06, 0.04, -0.04],
            ]
        )

        cohort_vars = scorer.compute_cohort_downside_variances(matrix, target_return=0.0)
        assert cohort_vars.shape == (3,)
        assert cohort_vars[0] == pytest.approx(0.0005)
        assert cohort_vars[1] == pytest.approx(1e-8)
        assert cohort_vars[2] == pytest.approx(0.00075)

        # Cross-validate against individual column calls
        for k in range(3):
            single_var = scorer.compute_downside_semi_variance(matrix[:, k], target_return=0.0)
            assert cohort_vars[k] == pytest.approx(single_var)

    def test_compute_cohort_downside_variances_validation(self) -> None:
        """Verify defensive validation on compute_cohort_downside_variances."""
        scorer = AsymmetricDownsideLossScorer()
        valid_mat = np.array([[0.01, -0.01], [0.02, -0.02]])

        # Non-2D
        with pytest.raises(InvalidPredictionException, match="return_matrix"):
            scorer.compute_cohort_downside_variances(np.array([0.01, -0.01]))
        with pytest.raises(InvalidPredictionException, match="return_matrix"):
            scorer.compute_cohort_downside_variances([[0.01, -0.01]])  # type: ignore[arg-type]

        # Empty matrix
        with pytest.raises(InvalidPredictionException, match="return_matrix"):
            scorer.compute_cohort_downside_variances(np.zeros((0, 3)))
        with pytest.raises(InvalidPredictionException, match="return_matrix"):
            scorer.compute_cohort_downside_variances(np.zeros((3, 0)))

        # Non-finite matrix
        with pytest.raises(DegenerateEnsembleException, match="return_matrix"):
            bad_mat = np.copy(valid_mat)
            bad_mat[0, 0] = np.nan
            scorer.compute_cohort_downside_variances(bad_mat)


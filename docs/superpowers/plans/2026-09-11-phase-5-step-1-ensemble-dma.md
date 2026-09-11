# Implementation Plan: Phase 5, Step 1 — Regime-Conditioned Dynamic Model Averaging (RD-DMA)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the institutional-grade Ensemble Alpha Aggregator subsystem (`src/quant/analytics/ensemble.py`), featuring volatility-adaptive forgetting, predictive forward-Markov regime transitions, asymmetric downside loss scoring, Tikhonov-regularized correlation, Entropic Mirror Descent on the simplex, thermodynamic ambiguity shrinkage, and Law-of-Total-Variance risk decomposition.

**Architecture:** Unifies Phase 2 (DSR warm-start), Phase 3 (CUSUM regime filter and thermodynamic ambiguity temperature $\beta_t$), and Phase 4 (SVD subspace orthogonal basis) into an online Dynamic Model Averaging engine that combines Pareto-optimal strategy forecasts while eliminating single-model collapse, collinear echo chambers, and drawdown blindness.

**Tech Stack:** Python 3.13, NumPy (vectorized linear algebra), SciPy (matrix decomposition and optimization), Pytest, Mypy (strict typing), Ruff.

**Spec:** `docs/superpowers/specs/2026-09-11-phase-5-step-1-ensemble-dma-design.md`

## Global Constraints
- Python 3.13 strict static typing (`mypy src --strict` with zero errors).
- Zero lint/format deviations (`ruff check .`, `ruff format --check .`).
- Test suite passing 100% with $\ge 90\%$ line coverage on `ensemble.py`.
- Strict compliance with invariant contracts `INV-ENS-001` through `INV-ENS-006`.
- Execution SLA: $\le 2.0\text{ms}$ per update/prediction cycle for $K=100$.

---

### Task 1: Domain Entities, Invariant Contracts, and Configuration

**Files:**
- Create: `src/quant/analytics/ensemble.py`
- Create: `tests/unit/test_ensemble.py`

**Interfaces:**
- Produces: `EnsembleConfig`, `EnsembleState`, `EnsemblePrediction`, `EnsembleError`, `DegenerateEnsembleException`, `InvalidPredictionException`.

- [ ] **Step 1: Write the failing tests for domain entities and invariants**

```python
# tests/unit/test_ensemble.py
import pytest
import numpy as np
from quant.analytics.ensemble import (
    EnsembleConfig,
    EnsembleState,
    EnsemblePrediction,
    EnsembleError,
    DegenerateEnsembleException,
    InvalidPredictionException,
)


class TestDomainEntitiesAndInvariants:
    def test_ensemble_config_defaults_and_validation(self) -> None:
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

        # Boundary checks
        with pytest.raises(EnsembleError):
            EnsembleConfig(base_forgetting_factor=1.5)
        with pytest.raises(EnsembleError):
            EnsembleConfig(min_forgetting_factor=0.98, max_forgetting_factor=0.90)
        with pytest.raises(EnsembleError):
            EnsembleConfig(turnover_damping=1.2)

    def test_ensemble_state_immutability_and_validation(self) -> None:
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

        # Invariant INV-ENS-001: simplex sum
        with pytest.raises(EnsembleError):
            EnsembleState(
                step_index=0,
                weights=np.array([0.5, 0.2]),
                regime_conditional_posteriors=posteriors,
                cumulative_losses=losses,
                effective_models=2.0,
                mean_realized_volatility=0.015,
                last_ambiguity_temperature=1.0,
            )

    def test_ensemble_prediction_variance_additivity(self) -> None:
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
```

- [ ] **Step 2: Run test to verify failure**

Run: `python -m pytest tests/unit/test_ensemble.py`
Expected: FAIL with ModuleNotFoundError: No module named 'quant.analytics.ensemble'.

- [ ] **Step 3: Implement domain models and exceptions in `src/quant/analytics/ensemble.py`**

Define `EnsembleError`, `DegenerateEnsembleException`, `InvalidPredictionException`, `EnsembleConfig`, `EnsembleState`, and `EnsemblePrediction` with complete invariant validation.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_ensemble.py`
Expected: PASS (3/3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/quant/analytics/ensemble.py tests/unit/test_ensemble.py
git commit -m "feat(ensemble): define domain entities, configuration, and invariants for RD-DMA (Phase 5 Step 1 Task 1)"
```

---

### Task 2: Volatility-Adaptive Forgetting & Predictive Forward-Markov Transitions

**Files:**
- Modify: `src/quant/analytics/ensemble.py`
- Modify: `tests/unit/test_ensemble.py`

**Interfaces:**
- Produces: `VolatilityAdaptiveForgetting`, `predict_forward_regime_prior()`.
- Implements: Invariant `INV-ENS-003` ($\alpha_t \in [\alpha_{\min}, \alpha_{\max}]$).

- [ ] **Step 1: Write the failing tests for forgetting factor and Markov transition**

```python
# In tests/unit/test_ensemble.py
class TestAdaptiveForgettingAndMarkovTransitions:
    def test_adaptive_forgetting_scales_with_volatility(self) -> None:
        cfg = EnsembleConfig()
        vaf = VolatilityAdaptiveForgetting(cfg, initial_mean_vol=0.01)

        # In calm market (sigma < mean): alpha expands toward alpha_max
        alpha_calm = vaf.compute_alpha(realized_vol=0.005)
        assert alpha_calm > cfg.base_forgetting_factor
        assert alpha_calm <= cfg.max_forgetting_factor

        # In panic shock (sigma >> mean): alpha contracts toward alpha_min
        alpha_shock = vaf.compute_alpha(realized_vol=0.030)
        assert alpha_shock < cfg.base_forgetting_factor
        assert alpha_shock >= cfg.min_forgetting_factor

    def test_predictive_forward_markov_transitions(self) -> None:
        # P_trans: 3x3 row-stochastic matrix
        P_trans = np.array(
            [
                [0.80, 0.15, 0.05],
                [0.10, 0.70, 0.20],
                [0.05, 0.05, 0.90],
            ]
        )
        p_current = np.array([0.1, 0.8, 0.1])  # Mostly Momentum
        # Forward prediction: p_{t+1|t} = P_trans^T * p_current
        p_next = predict_forward_regime_prior(p_current, P_trans)
        assert np.isclose(np.sum(p_next), 1.0)
        assert (
            p_next[2] > p_current[2]
        )  # Panic probability increased due to momentum transition risk
```

- [ ] **Step 2: Run test to verify failure**

Run: `python -m pytest tests/unit/test_ensemble.py -k TestAdaptiveForgettingAndMarkovTransitions`
Expected: FAIL with NameError: name 'VolatilityAdaptiveForgetting' is not defined.

- [ ] **Step 3: Implement `VolatilityAdaptiveForgetting` and `predict_forward_regime_prior`**

Implement exponential moving average updates on $\bar{\sigma}$, bounded $\alpha_t$ clamping, and vector-matrix Markov projection $\mathbf{p}_{t+1|t} = \mathbf{P}_{\text{trans}}^T \mathbf{p}_t$ with simplex normalization.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_ensemble.py -k TestAdaptiveForgettingAndMarkovTransitions`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quant/analytics/ensemble.py tests/unit/test_ensemble.py
git commit -m "feat(ensemble): implement volatility-adaptive forgetting and forward-Markov regime transitions (Phase 5 Step 1 Task 2)"
```

---

### Task 3: Asymmetric Downside Loss Scorer & Downside Semi-Variance

**Files:**
- Modify: `src/quant/analytics/ensemble.py`
- Modify: `tests/unit/test_ensemble.py`

**Interfaces:**
- Produces: `AsymmetricDownsideLossScorer` evaluating $\ell_{t, k}$ and $\sigma^2_{k, \text{down}}$.

- [ ] **Step 1: Write the failing tests for asymmetric downside loss and downside semi-variance**

```python
# In tests/unit/test_ensemble.py
class TestAsymmetricDownsideLossScorer:
    def test_asymmetric_loss_penalizes_drawdowns_heavily(self) -> None:
        scorer = AsymmetricDownsideLossScorer(downside_penalty=2.50)

        # Realized return is negative: market crashed -3%
        realized_return = -0.03

        # Model 1 predicted positive: +2% (false long during crash)
        pred_false_long = np.array([0.02])
        loss_false_long = scorer.compute_losses(pred_false_long, realized_return)[0]

        # Model 2 predicted negative: -2% (correct directional hedge)
        pred_hedge = np.array([-0.02])
        loss_hedge = scorer.compute_losses(pred_hedge, realized_return)[0]

        # Model 1 must be penalized exponentially worse due to gamma_down cross-term
        assert loss_false_long > loss_hedge
        # Explicit formula verification: (y - y_hat)^2 + gamma * max(0, -y * y_hat)
        expected_loss = (-0.03 - 0.02) ** 2 + 2.50 * (-(-0.03 * 0.02))
        assert loss_false_long == pytest.approx(expected_loss)

    def test_downside_semi_variance_computation(self) -> None:
        scorer = AsymmetricDownsideLossScorer()
        # Strategy return series with both large upside and occasional downside
        returns = np.array([0.01, 0.02, 0.015, -0.03, 0.01, -0.02, 0.005])
        semi_var = scorer.compute_downside_semi_variance(returns)
        assert semi_var > 0.0
        # Only negative deviations contribute
        expected = np.mean(np.minimum(0.0, returns) ** 2)
        assert semi_var == pytest.approx(expected)
```

- [ ] **Step 2: Run test to verify failure**

Run: `python -m pytest tests/unit/test_ensemble.py -k TestAsymmetricDownsideLossScorer`
Expected: FAIL with NameError: name 'AsymmetricDownsideLossScorer' is not defined.

- [ ] **Step 3: Implement `AsymmetricDownsideLossScorer`**

Vectorized NumPy computation of $\ell_{t, k}$ and empirical downside semi-variance with horizon normalization support.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_ensemble.py -k TestAsymmetricDownsideLossScorer`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quant/analytics/ensemble.py tests/unit/test_ensemble.py
git commit -m "feat(ensemble): implement asymmetric downside loss scorer and downside semi-variance (Phase 5 Step 1 Task 3)"
```

---

### Task 4: Tikhonov-Regularized Correlation & Entropic Mirror Descent Optimizer

**Files:**
- Modify: `src/quant/analytics/ensemble.py`
- Modify: `tests/unit/test_ensemble.py`

**Interfaces:**
- Produces: `TikhonovCorrelationEstimator`, `OrthogonalityRegularizedSolver`.
- Implements: Invariant `INV-ENS-001` (Strict Simplex Conservation) and clone penalization.

- [ ] **Step 1: Write the failing tests for correlation regularizer and mirror descent**

```python
# In tests/unit/test_ensemble.py
class TestCorrelationAndMirrorDescentSolver:
    def test_tikhonov_correlation_regularizer_with_zero_variance_model(self) -> None:
        estimator = TikhonovCorrelationEstimator(ridge_shrinkage=0.05)
        # Model 0 and Model 1 normal, Model 2 flat zeros (inactive)
        preds = np.array(
            [
                [0.01, -0.01, 0.0],
                [0.02, -0.02, 0.0],
                [0.00, 0.01, 0.0],
                [0.03, -0.03, 0.0],
            ]
        )
        C = estimator.compute_correlation_matrix(preds)
        assert C.shape == (3, 3)
        assert not np.isnan(C).any()
        # Eigenvalues must be >= ridge_shrinkage (strictly positive definite)
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals >= 0.049)

    def test_mirror_descent_solver_penalizes_clones(self) -> None:
        solver = OrthogonalityRegularizedSolver(
            orthogonality_penalty=0.50,
            temperature=1.0,
            min_weight_floor=1e-4,
        )
        K = 3
        # Model 0 and Model 1 are identical clones (corr = 1.0)
        # Model 2 is an independent orthogonal model (corr = 0.0 with both)
        C = np.array(
            [
                [1.0, 0.95, 0.0],
                [0.95, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        # Equal loss scores for all three models
        scores = np.array([1.0, 1.0, 1.0])
        initial_w = np.full(K, 1.0 / K)

        w_star = solver.solve(scores=scores, correlation_matrix=C, current_weights=initial_w)
        assert np.isclose(np.sum(w_star), 1.0)
        assert np.all(w_star > 0.0)
        # Model 2 (orthogonal) must receive higher weight than either clone
        assert w_star[2] > w_star[0]
        assert w_star[2] > w_star[1]
        assert w_star[0] == pytest.approx(w_star[1], rel=1e-3)
```

- [ ] **Step 2: Run test to verify failure**

Run: `python -m pytest tests/unit/test_ensemble.py -k TestCorrelationAndMirrorDescentSolver`
Expected: FAIL with NameError: name 'TikhonovCorrelationEstimator' is not defined.

- [ ] **Step 3: Implement `TikhonovCorrelationEstimator` and `OrthogonalityRegularizedSolver`**

Vectorized Entropic Mirror Descent loop with max-shifted Log-Sum-Exp normalization and Laplace floor smoothing.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_ensemble.py -k TestCorrelationAndMirrorDescentSolver`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quant/analytics/ensemble.py tests/unit/test_ensemble.py
git commit -m "feat(ensemble): implement Tikhonov correlation and Entropic Mirror Descent solver (Phase 5 Step 1 Task 4)"
```

---

### Task 5: Master `RegimeConditionedDMAEngine` Facade, Multi-Bar Lifecycle & Benchmark SLA

**Files:**
- Modify: `src/quant/analytics/ensemble.py`
- Modify: `src/quant/analytics/__init__.py`
- Modify: `tests/unit/test_ensemble.py`

**Interfaces:**
- Produces: `RegimeConditionedDMAEngine` master facade.
- Implements: Invariants `INV-ENS-001` through `INV-ENS-006` (execution SLA $\le 2.0\text{ms}$).

- [ ] **Step 1: Write the failing tests for master engine, multi-bar simulation, and performance SLA**

```python
# In tests/unit/test_ensemble.py
class TestRegimeConditionedDMAEngine:
    def test_initialize_state_with_warm_start_dsr(self) -> None:
        engine = RegimeConditionedDMAEngine()
        K = 10
        dsr_scores = np.linspace(0.8, 1.5, K)
        state = engine.initialize_state(n_models=K, initial_dsr=dsr_scores)
        assert state.step_index == 0
        assert np.isclose(np.sum(state.weights), 1.0)
        # Top DSR strategy must have higher initial prior weight than lowest
        assert state.weights[-1] > state.weights[0]

    def test_multi_bar_online_lifecycle_and_turnover_damping(self) -> None:
        engine = RegimeConditionedDMAEngine()
        K = 5
        state = engine.initialize_state(n_models=K)
        P_trans = np.eye(3) * 0.8 + 0.0667

        # Run 20 online steps
        for step in range(20):
            preds = np.random.normal(0.001, 0.005, K)
            variances = np.full(K, 0.0004)
            realized_return = 0.002
            realized_vol = 0.015
            regime_probs = np.array([0.6, 0.3, 0.1])

            prediction, state = engine.predict_and_update(
                predictions=preds,
                variances=variances,
                realized_return=realized_return,
                realized_vol=realized_vol,
                regime_probs=regime_probs,
                transition_matrix=P_trans,
                ambiguity_beta=1.0,
                state=state,
            )
            assert prediction.total_variance == pytest.approx(
                prediction.aleatoric_variance + prediction.epistemic_variance
            )
            assert np.isclose(np.sum(prediction.model_weights), 1.0)
            assert state.step_index == step + 1

    def test_performance_benchmark_sub_2ms(self) -> None:
        engine = RegimeConditionedDMAEngine()
        K = 100
        state = engine.initialize_state(n_models=K)
        P_trans = np.eye(3) * 0.8 + 0.0667

        preds = np.random.normal(0.001, 0.005, K)
        variances = np.full(K, 0.0004)
        regime_probs = np.array([0.5, 0.3, 0.2])

        import time

        # Warmup
        engine.predict_and_update(preds, variances, 0.001, 0.015, regime_probs, P_trans, 1.0, state)

        # Timed benchmark
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            _, state = engine.predict_and_update(
                preds, variances, 0.001, 0.015, regime_probs, P_trans, 1.0, state
            )
            times.append(time.perf_counter() - t0)

        mean_time_ms = np.mean(times) * 1000.0
        assert mean_time_ms <= 2.0, f"Benchmark SLA violated: {mean_time_ms:.3f}ms > 2.0ms"
```

- [ ] **Step 2: Run test to verify failure**

Run: `python -m pytest tests/unit/test_ensemble.py -k TestRegimeConditionedDMAEngine`
Expected: FAIL with NameError: name 'RegimeConditionedDMAEngine' is not defined.

- [ ] **Step 3: Implement `RegimeConditionedDMAEngine` and export in `__init__.py`**

Implement `initialize_state()`, `predict_and_update()`, ambiguity shrinkage, turnover damping, variance decomposition, and export all symbols in `src/quant/analytics/__init__.py`.

- [ ] **Step 4: Run full ensemble unit test suite**

Run: `python -m pytest tests/unit/test_ensemble.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quant/analytics/ensemble.py src/quant/analytics/__init__.py tests/unit/test_ensemble.py
git commit -m "feat(ensemble): implement master RegimeConditionedDMAEngine facade and performance benchmark (Phase 5 Step 1 Task 5)"
```

---

### Task 6: Documentation Synchronization & Quality Verification

**Files:**
- Modify: `Memory.md`
- Modify: `Phases.md`
- Modify: `Architecture.md`
- Modify: `PRD.md`
- Modify: `walkthrough.md`
- Modify: `implementation_plan.md`

- [ ] **Step 1: Update `Memory.md`**
  - Record ADR-018: Regime-Conditioned Dynamic Model Averaging Architecture.
  - Record Fault Vectors `ERR-ENS-001` through `ERR-ENS-005`.
  - Add Phase 26 Audit Trail entry.

- [ ] **Step 2: Update `Phases.md`**
  - In Section 1 Gantt chart: mark Phase 5 Step 1 as `:done`.
  - In Section 2 Phase 5: record Step 1 deliverables, contracts, and test metrics.

- [ ] **Step 3: Update `Architecture.md`**
  - In Section 3: Add `src/quant/analytics/ensemble.py`.
  - In Section 4: Add Section 4.16 describing the RD-DMA aggregation pipeline.

- [ ] **Step 4: Update `PRD.md`**
  - Update Feature Requirements Matrix for Ensemble Aggregator (Phase 5 Step 1 Complete).

- [ ] **Step 5: Run full project verification suite**

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy src --strict
python -m pytest tests/unit -v
```

- [ ] **Step 6: Commit documentation updates**

```bash
git add Memory.md Phases.md Architecture.md PRD.md
git commit -m "docs(ensemble): synchronize architecture, memory, and phase documentation for RD-DMA (Phase 5 Step 1 Task 6)"
```

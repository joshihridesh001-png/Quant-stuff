# Design Specification: Phase 5, Step 1 — Elite Regime-Conditioned Dynamic Model Averaging (RD-DMA)

**Date**: 2026-09-11  
**Status**: APPROVED / FINAL  
**Subsystem**: Ensemble Alpha Aggregator (`src/quant/analytics/ensemble.py`)  
**Target Milestone**: Phase 5, Step 1  

---

## 1. Executive Summary & Problem Statement

In Phase 4, the Evolutionary Strategy Search Engine cultivated a diverse Pareto frontier of $K=100$ champion strategies balancing Deflated Sharpe Ratio ($\text{DSR}$), Minimax Regret ($\Psi^*$), and SVD Subspace Orthogonal Novelty ($\rho_{\text{ortho}}$).

Traditional ensemble methods in quantitative finance suffer from severe structural flaws:
1. **Winner-Takes-All (WTA) Collapse**: Standard Bayesian Model Averaging (BMA) compounds marginal likelihoods over sample size $T$, collapsing into single-model selection and destroying ensemble diversity.
2. **Stationary Fallacy in Non-Stationary Markets**: Fixed historical weights lag behind macro regime shifts (e.g., low-volatility absorption transitioning into violent panic cascades).
3. **Collinear Echo Chambers**: Correlated strategies that share the same signal family dominate the vote by sheer numbers, eliminating portfolio diversification.
4. **Symmetric Loss Blindness**: Standard Mean Squared Error (MSE) penalizes missed upside gains and capital-destroying drawdown crashes equally.
5. **Point-Estimate Blindness**: Blended predictions ($\hat{\mu} = \sum w_k \hat{y}_k$) discard model disagreement, blinding risk systems to epistemic uncertainty.
6. **Market Chaos Overconfidence**: Ensembles maintain 100% full model conviction even during out-of-distribution macro shocks.
7. **Singularity & Division-by-Zero**: Inactive or flatline strategy predictions cause zero variance and NaN crashes in correlation calculations.

Phase 5 Step 1 resolves all seven structural flaws by implementing **Elite Regime-Conditioned Dynamic Model Averaging (RD-DMA)** with:
- **Predictive Forward-Markov Regime Transitions** ($\mathbf{p}_{t+1|t} = \mathbf{P}_{\text{trans}}^T \mathbf{p}_t$).
- **Volatility-Adaptive Memory Depth** ($\alpha_t \in [\alpha_{\min}, \alpha_{\max}]$).
- **Asymmetric Downside Loss** ($\ell_t$) and **Downside Semi-Variance** ($\sigma^2_{k, \text{down}}$).
- **Thermodynamic Ambiguity Shrinkage** linking to Phase 3 temperature ($\beta_t$).
- **Tikhonov-OAS Ridge Regularized Correlation Matrix** ($\mathbf{C}_t \succ 0, \lambda_{\min} \ge 0.05$).
- **Entropic Mirror Descent Optimizer** ($< 0.15\text{ms}$ on simplex).
- **Law-of-Total-Variance Decomposition** (Aleatoric vs. Epistemic).

---

## 2. Invariant Contracts

All components within `src/quant/analytics/ensemble.py` must strictly uphold the following invariant contracts:

| Invariant ID | Contract Name | Formal Definition | Defensive Enforcement |
| :--- | :--- | :--- | :--- |
| **`INV-ENS-001`** | **Strict Simplex Conservation** | $\sum_{k=1}^K w_k \equiv 1.0 \pm 10^{-12} \land \forall k, w_k \ge \epsilon_{\text{floor}} > 0$. | Softmax normalization with immutable Laplace floor $\epsilon_{\text{floor}} = 0.001 / K$. |
| **`INV-ENS-002`** | **Variance Positivity & Additivity** | $\sigma^2_{\text{total}} \equiv \sigma^2_{\text{aleatoric}} + \sigma^2_{\text{epistemic}}$ where $\sigma^2_{\text{aleatoric}} > 0 \land \sigma^2_{\text{epistemic}} \ge 0$. | Verified algebraically via the Law of Total Variance on every emission. |
| **`INV-ENS-003`** | **Bounded Adaptive Forgetting** | $\alpha_t \in [\alpha_{\min}, \alpha_{\max}] \subset (0, 1)$. | Strict numerical clamping preventing explosive memory erasure or static freeze. |
| **`INV-ENS-004`** | **Strict Causal Information Flow** | Prediction $\hat{\mu}_t$ and weights $\mathbf{w}_t$ depend strictly on $\mathcal{F}_{t-1}$ observations and bar $t$ candidate feature forecasts. | Strict separation of prior scoring from contemporaneous return realizations. |
| **`INV-ENS-005`** | **Bounded Weight Turnover** | $\|\mathbf{w}_t - \mathbf{w}_{t-1}\|_1 \le 2(1 - \lambda_{\text{churn}})$. | Convex L1 turnover damping: $\mathbf{w}_t = (1 - \lambda) \mathbf{w}_t^* + \lambda \mathbf{w}_{t-1}$. |
| **`INV-ENS-006`** | **Execution Latency SLA** | Total update and prediction cycle $\le 2.0\text{ms}$ for $K=100$. | Benchmarked with vectorization and Entropic Mirror Descent. |

---

## 3. Mathematical Formulations

### 3.1 Warm-Start Informed Prior Initialization (Bar 0)
To eliminate cold-start lag, initial model priors are seeded from Phase 4's backtested Deflated Sharpe Ratio ($\text{DSR}$) and Angle-Penalized Distance ($\text{APD}$):
$$\pi_{0, k}^{(j)} = \frac{\exp\left(\frac{\text{DSR}_k - \min_m \text{DSR}_m}{\tau_{\text{prior}}}\right)}{\sum_{l=1}^K \exp\left(\frac{\text{DSR}_l - \min_m \text{DSR}_m}{\tau_{\text{prior}}}\right)}, \quad \forall j \in \{1, 2, 3\}$$
where $\tau_{\text{prior}} = 1.0$. If DSR scores are unavailable, defaults to uniform Dirichlet prior $\pi_{0, k}^{(j)} = 1/K$.

### 3.2 Volatility-Adaptive Forgetting Factor ($\alpha_t$)
Memory depth dynamically scales inversely with market realized volatility:
$$\alpha_t = \text{clip}\left(\alpha_0 - \kappa_\alpha \cdot \frac{\sigma_t - \bar{\sigma}}{\bar{\sigma}}, \, \alpha_{\min}, \, \alpha_{\max}\right)$$
* $\alpha_0 = 0.96$: Base forgetting factor ($\approx 25$-bar effective memory).
* In calm regimes ($\sigma_t < \bar{\sigma}$): $\alpha_t \to \alpha_{\max} = 0.99$ ($\approx 100$-bar memory), filtering out transient market noise.
* In panic shocks ($\sigma_t \ge 2\bar{\sigma}$): $\alpha_t \to \alpha_{\min} = 0.85$ ($\approx 6.7$-bar memory), accelerating adaptation to structural breaks within $1\text{--}2$ candles.

### 3.3 Asymmetric Downside Loss Scoring ($\ell_{t, k}$)
Candidate strategy $k$ predicting standardized return $\tilde{y}_{t, k} = \hat{y}_{t, k} / \sqrt{H_k}$ against realized return $y_t$ is scored via:
$$\ell_{t, k} = (y_t - \tilde{y}_{t, k})^2 + \gamma_{\text{down}} \cdot \max\left(0, -y_t \cdot \tilde{y}_{t, k}\right)$$
where $\gamma_{\text{down}} = 2.50$. False positive signals that induce capital drawdowns are penalized quadratically-linearly.

### 3.4 Predictive Forward-Markov Regime Transitions
For each regime $j \in \{1, 2, 3\}$ ($1=\text{Absorption}, 2=\text{Momentum}, 3=\text{Panic}$):
$$\pi_{t|t-1, k}^{(j)} = \frac{\left(\pi_{t-1|t-1, k}^{(j)}\right)^{\alpha_t}}{\sum_{m=1}^K \left(\pi_{t-1|t-1, m}^{(j)}\right)^{\alpha_t}}$$
Given contemporaneous regime filter probabilities $\mathbf{p}_t$ and Phase 3 Markov transition matrix $\mathbf{P}_{\text{trans}} \in \mathbb{R}^{3 \times 3}$, we calculate the **predictive forward regime distribution**:
$$\mathbf{p}_{t+1|t} = \mathbf{P}_{\text{trans}}^T \mathbf{p}_t$$
The composite prior blends with predictive foresight:
$$\bar{\pi}_{t|t-1, k} = \sum_{j=1}^3 p_{t+1|t, j} \cdot \pi_{t|t-1, k}^{(j)}$$

### 3.5 Tikhonov Ridge-Regularized Correlation Matrix ($\mathbf{C}_t$)
To eliminate $0/0$ division by zero and singularity from inactive strategies, prediction correlation is regularized:
$$\mathbf{C}_t = (1 - \delta_{\text{ridge}}) \widehat{\mathbf{C}}_t + \delta_{\text{ridge}} \mathbf{I}, \quad \delta_{\text{ridge}} = 0.05$$
where $\widehat{\mathbf{C}}_t$ enforces a standard deviation floor $\sigma_{\min} = 10^{-8}$. This guarantees $\mathbf{C}_t \succ 0$ with $\lambda_{\min} \ge 0.05$.

### 3.6 SVD Orthogonality Penalization & Entropic Mirror Descent
Target weights solve the entropy-regularized orthogonality objective on the simplex $\Delta^K$:
$$\mathbf{w}_t^* = \arg\min_{\mathbf{w} \in \Delta^K} \left\{ \mathbf{w}^T \mathbf{s}_t + \frac{\lambda_{\text{ortho}}}{2} \mathbf{w}^T \mathbf{C}_t \mathbf{w} - \tau \mathcal{H}(\mathbf{w}) \right\}$$
where $\mathbf{s}_t = \boldsymbol{\ell}_{t-1} - \ln \bar{\boldsymbol{\pi}}_{t|t-1}$ and $\mathcal{H}(\mathbf{w}) = -\sum w_k \ln w_k$.

Solved via **Entropic Mirror Descent** starting from $\mathbf{w}^{(0)} = \mathbf{w}_{t-1}$:
$$w_i^{(m+1)} = \frac{w_i^{(m)} \exp\left(-\frac{\eta}{\tau} \left(s_{t, i} + \lambda_{\text{ortho}} [\mathbf{C}_t \mathbf{w}^{(m)}]_i\right)\right)}{\sum_{j=1}^K w_j^{(m)} \exp\left(-\frac{\eta}{\tau} \left(s_{t, j} + \lambda_{\text{ortho}} [\mathbf{C}_t \mathbf{w}^{(m)}]_j\right)\right)}$$
Terminates when $\|\mathbf{w}^{(m+1)} - \mathbf{w}^{(m)}\|_\infty < 10^{-6}$ or $m = 10$ iterations ($< 0.15\text{ms}$).
Followed by Laplace floor regularization:
$$w_{t, k}^* \leftarrow (1 - K \epsilon_{\text{floor}}) w_{t, k}^* + \epsilon_{\text{floor}}, \quad \epsilon_{\text{floor}} = \frac{0.001}{K}$$

### 3.7 Thermodynamic Ambiguity Shrinkage
When Phase 3 thermodynamic ambiguity $\beta_t$ is elevated, the ensemble shrinks weights toward the entropy-maximizing uniform prior:
$$\lambda_\beta(\beta_t) = \text{clip}\left(\frac{\beta_t - \beta_{\min}}{\beta_{\max} - \beta_{\min}}, 0.0, 1.0\right) \cdot \kappa_{\text{shrink}}$$
$$\mathbf{w}_t^{\text{shrunk}} = (1 - \lambda_\beta(\beta_t)) \mathbf{w}_t^* + \lambda_\beta(\beta_t) \mathbf{w}_{\text{uniform}}, \quad \kappa_{\text{shrink}} = 0.50$$

### 3.8 Turnover Damping & Total Variance Decomposition
To protect net returns against transaction fee erosion:
$$\mathbf{w}_t^{\text{final}} = (1 - \lambda_{\text{churn}}) \mathbf{w}_t^{\text{shrunk}} + \lambda_{\text{churn}} \mathbf{w}_{t-1}^{\text{final}}, \quad \lambda_{\text{churn}} = 0.15$$

The ensemble emits a complete probabilistic prediction:
$$\hat{\mu}_t = \sum_{k=1}^K w_k^{\text{final}} \tilde{y}_{t, k}$$
$$\sigma^2_{\text{aleatoric}} = \sum_{k=1}^K w_k^{\text{final}} \sigma_{k, \text{down}}^2$$
$$\sigma^2_{\text{epistemic}} = \sum_{k=1}^K w_k^{\text{final}} \left(\tilde{y}_{t, k} - \hat{\mu}_t\right)^2$$
$$\sigma^2_{\text{total}} = \sigma^2_{\text{aleatoric}} + \sigma^2_{\text{epistemic}}$$
$$K_{\text{eff}} = \frac{1}{\sum_{k=1}^K \left(w_k^{\text{final}}\right)^2} \in [1.0, K]$$

---

## 4. Component Architecture & Class Specifications

### 4.1 Domain Entities (`src/quant/analytics/ensemble.py`)

```python
@dataclass(frozen=True)
class EnsembleConfig:
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


@dataclass(frozen=True)
class EnsembleState:
    step_index: int
    weights: np.ndarray  # shape (K,), sum=1.0, w_k >= eps
    regime_conditional_posteriors: np.ndarray  # shape (3, K)
    cumulative_losses: np.ndarray  # shape (K,)
    effective_models: float  # K_eff in [1, K]
    mean_realized_volatility: float  # rolling bar baseline sigma_bar
    last_ambiguity_temperature: float  # beta_t


@dataclass(frozen=True)
class EnsemblePrediction:
    point_prediction: float  # mu_t
    aleatoric_variance: float  # sigma^2_aleatoric (downside adjusted)
    epistemic_variance: float  # sigma^2_epistemic (model disagreement)
    total_variance: float  # sigma^2_total
    model_weights: np.ndarray  # shape (K,)
    regime_probabilities: np.ndarray  # shape (3,)
    effective_models: float
    volatility_forgetting_factor: float  # alpha_t applied
    ambiguity_shrinkage_weight: float  # lambda_beta applied
```

### 4.2 Core Analytics Classes

1. **`VolatilityAdaptiveForgetting`**:
   - Computes dynamic $\alpha_t$ from realized $\sigma_t$ and rolling baseline $\bar{\sigma}$.
   - Updates exponential moving average $\bar{\sigma} \leftarrow (1 - \beta) \bar{\sigma} + \beta \sigma_t$.
2. **`AsymmetricDownsideLossScorer`**:
   - Evaluates $(y_t - \tilde{y}_{t, k})^2 + \gamma_{\text{down}} \max(0, -y_t \tilde{y}_{t, k})$.
   - Computes empirical downside semi-variance $\sigma^2_{k, \text{down}}$.
3. **`TikhonovCorrelationEstimator`**:
   - Evaluates pairwise correlation with $\sigma_{\min} = 10^{-8}$ floor and $\delta_{\text{ridge}} = 0.05$ shrinkage.
4. **`OrthogonalityRegularizedSolver`**:
   - Solves the constrained simplex objective via Entropic Mirror Descent.
   - Applies Laplace floor and returns strictly normalized $\mathbf{w}_t^* \in \Delta^K$.
5. **`RegimeConditionedDMAEngine` (Master Facade)**:
   - `initialize_state(n_models, initial_dsr=None, initial_vol=0.01) -> EnsembleState`: Creates warm-started initial state.
   - `predict_and_update(predictions, variances, realized_return, realized_vol, regime_probs, transition_matrix, ambiguity_beta, state, correlation_matrix=None) -> tuple[EnsemblePrediction, EnsembleState]`: Executes complete online cycle.

---

## 5. Deterministic Diagnostics Failure Matrix

| Code | Location | Description & Root Cause | Prevention & Safe Action |
| :--- | :--- | :--- | :--- |
| **`ERR-ENS-DIM`** | `DMAEngine.predict_and_update` | Dimensionality mismatch between predictions ($K$), variances ($K$), or state weights. | Strictly check `len(predictions) == len(variances) == len(state.weights) == K`. Raise `InvalidPredictionException`. |
| **`ERR-ENS-NAN`** | `EnsemblePrediction.__post_init__` | Non-finite values (NaN, Inf) detected in predictions, variances, or market returns. | Validate all float scalar and array inputs via `np.isfinite()`. Raise `DegenerateEnsembleException`. |
| **`ERR-ENS-SIMPLEX`** | `DMAEngine.predict_and_update` | Model weights violate unit simplex constraint: $\|\sum w_k - 1.0\| > 10^{-10}$ or $w_k \le 0$. | Defensive L1 normalization after mirror descent and Laplace floor addition. |
| **`ERR-ENS-REGIME`** | `DMAEngine.predict_and_update` | Regime probability vector does not match $M=3$ or fails simplex sum. | Validate `len(regime_probs) == 3` and `np.isclose(np.sum(regime_probs), 1.0)`. |
| **`ERR-ENS-TRANS`** | `DMAEngine.predict_and_update` | Transition matrix shape mismatch $\ne (3, 3)$ or row stochastic sum $\ne 1.0$. | Validate `transition_matrix.shape == (3, 3)` and rows sum to $1.0$. |

---

## 6. Testing Strategy & Verification Plan

### 6.1 Unit Test Coverage Target ($\ge 90\%$)
Test suite: `tests/unit/test_ensemble.py`
1. **Domain Entities & Invariants**:
   - `test_config_validation`: Bounds on hyperparameters.
   - `test_simplex_invariant_INV_ENS_001`: Weights sum to 1.0, strictly positive.
   - `test_variance_additivity_INV_ENS_002`: $\sigma^2_{\text{total}} \equiv \sigma^2_{\text{aleatoric}} + \sigma^2_{\text{epistemic}}$.
2. **Adaptive Forgetting Dynamics (`INV-ENS-003`)**:
   - Calm markets expand memory toward $\alpha_{\max} = 0.99$.
   - Panic shocks contract memory toward $\alpha_{\min} = 0.85$.
3. **Asymmetric Downside Loss & Downside Semi-Variance**:
   - Directionally wrong sell-off predictions penalized heavily.
   - Downside semi-variance excludes upside variance spikes.
4. **Predictive Forward-Markov Transition**:
   - High transition probability to Panic immediately boosts defensive models.
5. **Tikhonov Regularized Correlation**:
   - Zero-variance flatline models do not produce NaN; matrix has $\lambda_{\min} \ge 0.05$.
6. **Thermodynamic Ambiguity Shrinkage**:
   - High $\beta_t$ shrinks weights toward uniform distribution; normal $\beta_t$ preserves alpha concentration.
7. **Entropic Mirror Descent Solver**:
   - Numerical convergence in $\le 5$ iterations, $< 0.15\text{ms}$.
8. **Multi-Bar Online Lifecycle (`INV-ENS-004`, `INV-ENS-005`)**:
   - 50-bar rolling sequence without lookahead leakage.
   - Turnover damping limits one-bar weight churn.
9. **Performance Benchmark (`INV-ENS-006`)**:
   - $K=100$ models full update and prediction cycle completes in $\le 2.0\text{ms}$.

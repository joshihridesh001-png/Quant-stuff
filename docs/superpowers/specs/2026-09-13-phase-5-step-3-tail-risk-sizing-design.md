# Design Specification: Phase 5, Step 3 — Semi-Parametric EVT Tail Risk & Unified Convex Execution Sizing Calibration

- **Phase:** 5 (Institutional Strategy Suite & Online DMA Aggregator)
- **Step:** 3 (Extreme Tail VaR / Expected Shortfall & Unified Sizing Calibration)
- **Target Files:**
  - `src/quant/analytics/tail_risk.py`
  - `src/quant/analytics/execution_sizing.py`
- **Test Files:**
  - `tests/unit/test_tail_risk.py`
  - `tests/unit/test_execution_sizing.py`
- **Parent Subsystems:** RD-DMA Alpha Aggregator (`ensemble.py`), Circuit Breaker Overlays (`circuit_breakers.py`), Market Impact Propagator (`market_impact.py`), Continuous Meta-Labeling (`meta_labeling.py`).
- **Governing Rule:** `Rules.md` (Rule 4: Mandatory Adversarial Red-Teaming, Superior Alternatives, Anti-Shortcut Blacklist).

---

## 1. Executive Summary & Problem Formulation

In Phase 5 Steps 1 & 2, the quantitative engine established:
1. **Regime-Conditioned Dynamic Model Averaging (RD-DMA)**: Online Entropic Mirror Descent on the simplex combining $K=100$ models, decomposing predictive variance into aleatoric process noise $\sigma^2_{\text{aleatoric}}$ and epistemic model disagreement $\sigma^2_{\text{epistemic}}$.
2. **Epistemic Disagreement Entropy & Circuit Breakers**: Measuring 3-simplex directional consensus $\mathbf{p}_{\text{dir}} \in \Delta^3$, normalized Shannon entropy $\widetilde{\mathcal{H}}_{\text{dir}}$, thermodynamic composite shock score $\Xi_t \in [0, 1]$, continuous logistic haircut $\kappa_t \in [0, 1]$, and discrete institutional tiers (`NORMAL`, `CAUTION`, `DERISK`, `HALT`).

However, turning predictive signals and risk haircuts into actual dollar order allocations $\boldsymbol{\nu}_t \in \mathbb{R}^N$ reveals four fatal industry failure modes (codified in `Rules.md` Rule 4):
- **The Cornish-Fisher Collapse**: Polynomial quantile approximations invert and become non-monotonic when excess kurtosis $\gamma_4 > 3$, predicting lower risk during extreme market panics.
- **The $\sqrt{H}$ Time-Scaling Fallacy**: Assuming Brownian diffusion understates tail risk by orders of magnitude when markets undergo Poisson jump-diffusions and power-law cascades.
- **Value-at-Risk Non-Subadditivity**: VaR violates Artzner's coherence axioms, penalizes diversification, and remains completely blind to loss severity beyond the quantile threshold.
- **The Kelly vs. 3/2-Power Market Impact Collision**: Unconstrained or heuristic Half-Kelly sizing ignores non-linear execution slippage ($\psi_{3/2}(u)$), causing block orders to destroy their own expected alpha.

Phase 5 Step 3 implements **Semi-Parametric Extreme Value Theory (POT-GPD) Tail Risk & Unified Convex Execution Sizing Calibration**:
1. **Closed-Form EVT Peaks-Over-Threshold (POT) Engine**: Estimates Generalized Pareto Distribution (GPD) parameters $(\xi, \beta)$ via algebraic **Probability Weighted Moments (PWM)** in $O(N_u \log N_u)$ time, strictly eliminating iterative numerical optimizers (`scipy.optimize`) from the $< 0.20\text{ms}$ hot path.
2. **Coherent Expected Shortfall (CVaR)**: Evaluates subadditive, coherent tail risk with a theoretical infinite-variance tripwire ($\xi \ge 1.0 \implies \text{HALT}$).
3. **Cold-Start Degradation Hierarchy**: Dynamically transitions across Empirical Quantiles ($N < 30$), Student-t Method-of-Moments ($30 \le N < 250$), and Full EVT-GPD ($N \ge 250$).
4. **Unified Convex Risk-Constrained Sizing Optimizer**: Simultaneously optimizes uncertainty-shrunk Kelly utility, 3/2-power Generalized Pseudo-Huber market impact, circuit breaker continuous haircut regularization ($\frac{1}{2\kappa_t} \|\boldsymbol{\nu}\|^2$), and a hard Rockafellar-Uryasev CVaR Maximum Drawdown (MDD) budget.
5. **Microstructural Lot Discretization**: Conserves risk budgets through randomized lot rounding and enforces zero-lookahead causality via strictly lagged frozen ring buffers $[t-W, t-1]$.

---

## 2. Invariant Contracts

| Invariant ID | Formal Definition | Description |
| :--- | :--- | :--- |
| **`INV-TR-001`** | $\forall \alpha \in (0, 1): \text{CVaR}_\alpha \ge \text{VaR}_\alpha$ | Coherent Risk Ordering: Expected Shortfall strictly bounds Value-at-Risk from above. |
| **`INV-TR-002`** | $\xi \in [0.001, 0.999] \land (\xi \ge 1.0 \implies \text{HALT})$ | Fréchet Tail Stability: Tail shape index $\xi < 1.0$ guarantees finite theoretical mean. If $\xi \ge 1.0$, trigger emergency HALT. |
| **`INV-TR-003`** | $\text{CVaR}_\alpha(\sum w_i r_i) \le \sum w_i \text{CVaR}_\alpha(r_i)$ | Artzner Subadditivity: Portfolio Expected Shortfall respects diversification benefits without artifactual risk inflation. |
| **`INV-TR-004`** | $\nabla^2 U_{\text{Sizer}}(\boldsymbol{\nu}) \prec 0$ | Strict Global Concavity: Sizing objective is strictly concave everywhere, guaranteeing a unique global allocation. |
| **`INV-TR-005`** | $\text{CVaR}_\alpha(\boldsymbol{\nu}) \le \text{MDD}_{\text{budget}} \cdot W_t \land \|\boldsymbol{\nu}\|_1 \le L_{\max} \cdot W_t$ | Hard Drawdown & Leverage Budget: Allocated capital never exceeds predefined risk capacity or gross leverage ceilings. |
| **`INV-TR-006`** | $\text{Latency}_{\text{eval}} \le 0.20\text{ms}$ for $N=10, K=100$ | Hot-Path Execution SLA: Vectorized closed-form PWM tail risk and convex sizing complete within $0.20\text{ms}$ median. |
| **`INV-TR-007`** | $\mathcal{I}_t \subset \sigma(\{X_\tau\}_{\tau \le t-1})$ | Zero-Lookahead Causality: All rolling tail moments, dynamic thresholds $u_t$, and quantiles are computed on strictly lagged windows $[t-W, t-1]$. |

---

## 3. Mathematical Specifications

### 3.1 Semi-Parametric Peaks-Over-Threshold (POT) & Generalized Pareto Distribution (GPD)

Let $X_1, X_2, \dots, X_n$ denote historical negative log-returns (loss innovations) observed over a strictly lagged rolling window of length $W$ ($X_\tau = -r_\tau$).
We establish a dynamic high threshold:
$$u_t = \mu_{t-1} + k_{\text{threshold}} \cdot \sigma_{t-1}$$
where $k_{\text{threshold}} \approx 1.645$ (the nominal 95th percentile).

By the Pickands-Balkema-de Haan Theorem, the distribution of excess losses $Y = X - u_t$ conditional on $X > u_t$ converges asymptotically to the Generalized Pareto Distribution:
$$G_{\xi, \beta}(y) = \mathbb{P}(X - u_t \le y \mid X > u_t) = 1 - \left( 1 + \frac{\xi y}{\beta} \right)^{-1/\xi}$$
where:
- $\xi \in (0, 1)$ is the **tail shape index** (Fréchet heavy-tail parameter).
- $\beta > 0$ is the **scale parameter**.

### 3.2 Closed-Form Probability Weighted Moments (PWM)
To guarantee deterministic execution under the $< 0.20\text{ms}$ SLA and eliminate non-convergence failures from iterative numerical solvers (banned under Rule 4.3), parameters are estimated via algebraic Probability Weighted Moments.

Given $N_u$ exceedances sorted in ascending order $y_{(1)} \le y_{(2)} \le \dots \le y_{(N_u)}$:
$$M_0 = \frac{1}{N_u} \sum_{i=1}^{N_u} y_{(i)}$$
$$M_1 = \frac{1}{N_u} \sum_{i=1}^{N_u} \left( 1 - \frac{i - 0.35}{N_u} \right) y_{(i)}$$

The closed-form analytical PWM estimators are:
$$\xi_{\text{PWM}} = 2 - \frac{M_0}{M_0 - 2M_1}$$
$$\beta_{\text{PWM}} = \frac{2 M_0 M_1}{M_0 - 2M_1}$$

- If $M_0 - 2M_1 \le 0$ or $\xi_{\text{PWM}} \le 0$: Fall back to exponential distribution ($\xi = 0, \beta = M_0$).
- If $\xi_{\text{PWM}} \ge 1.0$: Infinite variance detected; clamp $\xi = 0.999$ and trigger emergency `CircuitBreakerTier.HALT` (`INV-TR-002`).

### 3.3 Closed-Form Value-at-Risk (VaR) & Expected Shortfall (CVaR)
For confidence level $\alpha \in (0.90, 0.999)$ (default $\alpha = 0.99$):

$$\text{VaR}_\alpha = u_t + \frac{\beta}{\xi} \left[ \left( \frac{n}{N_u} (1 - \alpha) \right)^{-\xi} - 1 \right]$$

The coherent **Expected Shortfall (CVaR)** is:
$$\text{CVaR}_\alpha = \mathbb{E}[X \mid X > \text{VaR}_\alpha] = \frac{\text{VaR}_\alpha + \beta - \xi u_t}{1 - \xi}$$

### 3.4 Cold-Start Degradation Hierarchy
To prevent buffer starvation during new asset listings or connection drops:
1. **Severe Starvation ($N_u < 10$ or $n < 30$)**:
   $$\text{VaR}_\alpha = \text{Percentile}_{\text{Empirical}}(X, 100\alpha), \quad \text{CVaR}_\alpha = \text{Mean}(X \mid X \ge \text{VaR}_\alpha)$$
2. **Maturing History ($30 \le n < 250$)**:
   Parametric Student-t tail fitting via method of moments with degrees of freedom $\nu = \frac{4 \gamma_4 - 6}{\gamma_4 - 3}$.
3. **Fully Hydrated History ($n \ge 250, N_u \ge 15$)**:
   Full Semi-Parametric EVT-POT GPD with closed-form PWM.

---

### 3.5 Unified Convex Risk-Constrained Execution Sizer

Given:
- Target asset universe $i \in \{1, \dots, N\}$.
- Expected return vector $\hat{\boldsymbol{\mu}} \in \mathbb{R}^N$ from RD-DMA.
- Epistemic uncertainty vector $\boldsymbol{\sigma}^2_{\text{epistemic}} \in \mathbb{R}^N$.
- Process covariance matrix $\boldsymbol{\Sigma}_{\text{aleatoric}} \in \mathbb{R}^{N \times N}$.
- Cross-impact tensor $\mathbf{\Lambda}_{\text{cross}} \succ 0$ and 3/2-power pseudo-Huber parameters $(\eta, \delta)$.
- Active circuit breaker continuous haircut $\kappa_t \in [0.0, 1.0]$.
- Total portfolio capital $W_t$, max leverage $L_{\max}$, and hard drawdown budget $\text{MDD}_{\text{budget}}$.

We formulate the dollar allocation $\boldsymbol{\nu}^* \in \mathbb{R}^N$ as a **Strictly Concave Maximization**:

$$\max_{\boldsymbol{\nu}} \mathcal{L}(\boldsymbol{\nu}) = U_{\text{Kelly}}(\boldsymbol{\nu}) - \mathcal{C}_{\text{Impact}}(\boldsymbol{\nu}) - \mathcal{R}_{\text{Epistemic}}(\boldsymbol{\nu})$$

#### A. Uncertainty-Shrunk Kelly Utility:
$$U_{\text{Kelly}}(\boldsymbol{\nu}) = \boldsymbol{\nu}^T \left( \hat{\boldsymbol{\mu}} - \lambda_{\text{shrink}} \boldsymbol{\sigma}^2_{\text{epistemic}} \right) - \frac{\gamma_{\text{risk}}}{2 W_t} \boldsymbol{\nu}^T \boldsymbol{\Sigma}_{\text{aleatoric}} \boldsymbol{\nu}$$
*Dynamically shrinks expected edge on strategies with high epistemic disagreement.*

#### B. Generalized Pseudo-Huber Market Impact Friction:
$$\mathcal{C}_{\text{Impact}}(\boldsymbol{\nu}) = \frac{1}{2 W_t} \boldsymbol{\nu}^T \mathbf{\Lambda}_{\text{cross}} \boldsymbol{\nu} + \eta \sum_{i=1}^N \sigma_i \left[ (\nu_i^2 + \delta^2)^{3/4} - \delta^{1.5} \right] \cdot \left(1 + \kappa_{\text{panic}} \pi_{\text{panic}} \sigma_{\text{asym}}(\nu_i)\right)$$

#### C. Circuit Breaker Regularization:
$$\mathcal{R}_{\text{Epistemic}}(\boldsymbol{\nu}) = \frac{1}{2 \kappa_t(\Xi_t) \cdot W_t} \|\boldsymbol{\nu}\|^2_2$$
*As composite shock $\Xi_t \to 1.0$, haircut $\kappa_t \to 0.0$, driving the quadratic regularization penalty to $+\infty$ and smoothly pinning target allocations to zero.*

#### D. Hard Constraints:
1. **Rockafellar-Uryasev CVaR Drawdown Budget**:
   $$\text{CVaR}_\alpha(\boldsymbol{\nu}) \le \text{MDD}_{\text{budget}} \cdot W_t$$
2. **Gross Leverage Ceiling**:
   $$\|\boldsymbol{\nu}\|_1 \le L_{\max} \cdot W_t$$

#### E. Microstructural Lot Discretization:
Given exchange minimum contract lot sizes $\Delta \nu_i$:
$$\tilde{\nu}_i = \left\lfloor \frac{\nu_i^*}{\Delta \nu_i} \right\rfloor \Delta \nu_i + B_i \Delta \nu_i, \quad B_i \sim \text{Bernoulli}\left( \frac{\nu_i^* \bmod \Delta \nu_i}{\Delta \nu_i} \right)$$
*Randomized rounding preserves the portfolio risk budget in expectation without rounding bias.*

---

## 4. Architecture & Component Blueprint

### 4.1 Module Structure

```
src/quant/analytics/
├── tail_risk.py               # EVT-POT, PWM estimator, CVaR/VaR, Cold-Start Ladder
└── execution_sizing.py        # Unified Convex Sizer, Lot Discretizer, KKT Solver
```

### 4.2 Core Data Structures & Signatures

```python
@dataclass(frozen=True)
class EVTTailParameters:
    threshold_u: float
    shape_xi: float
    scale_beta: float
    num_exceedances: int
    total_observations: int
    method: str  # "EVT_PWM", "STUDENT_T", "EMPIRICAL"


@dataclass(frozen=True)
class TailRiskMetrics:
    var_alpha: float
    cvar_alpha: float
    confidence_level: float
    tail_parameters: EVTTailParameters
    step_index: int


@dataclass(frozen=True)
class SizingConfig:
    confidence_level: float = 0.99
    mdd_budget: float = 0.15  # 15% maximum portfolio drawdown budget
    max_leverage: float = 2.0  # 2x gross leverage cap
    epistemic_shrinkage_lambda: float = 1.0
    risk_aversion_gamma: float = 1.0
    impact_penalty_eta: float = 0.10
    lot_sizes: np.ndarray | None = None  # (N,) array of contract lot sizes


@dataclass(frozen=True)
class SizingDecision:
    target_allocations: np.ndarray  # Raw continuous dollar sizing (N,)
    discretized_allocations: np.ndarray  # Microstructure lot-discretized sizing (N,)
    effective_leverage: float
    expected_shortfall: float
    estimated_impact_cost: float
    circuit_breaker_haircut: float
    is_drawdown_constrained: bool
    is_leverage_constrained: bool
```

---

## 5. Verification & Testing Plan

1. **Unit Tests (`tests/unit/test_tail_risk.py`)**:
   - `INV-TR-001`: Assert $\text{CVaR}_\alpha \ge \text{VaR}_\alpha$ across synthetic, normal, Student-t, and heavy-tailed Fréchet draws.
   - `INV-TR-002`: Verify PWM parameter estimation accuracy against known GPD parameters. Assert tripwire triggers emergency `HALT` when $\xi \ge 1.0$.
   - `INV-TR-003`: Verify subadditivity on correlated two-asset portfolios ($\text{CVaR}(A+B) \le \text{CVaR}(A) + \text{CVaR}(B)$).
   - Cold-Start degradation ladder verification: $N < 30 \implies \text{Empirical}$, $30 \le N < 250 \implies \text{Student-t}$, $N \ge 250 \implies \text{EVT-GPD}$.
   - Performance SLA: PWM EVT parameter fit and CVaR calculation completes in $\le 0.05\text{ms}$.

2. **Unit Tests (`tests/unit/test_execution_sizing.py`)**:
   - `INV-TR-004`: Verify strict convexity and Hessian negative definiteness of sizing objective.
   - `INV-TR-005`: Assert hard MDD budget and leverage bounds are never breached even under extreme alpha predictions.
   - Circuit breaker coupling: Verify that when $\kappa_t \to 0.0$, allocations smoothly converge to $\mathbf{0}$.
   - 3/2-power impact coupling: Verify sizing does not explode when order size increases.
   - Lot discretization: Verify randomized rounding expectation equals continuous sizing.
   - Benchmark SLA: Full sizing solve completes in $\le 0.15\text{ms}$ for $N=10$.

# Phase 5 Step 4 Design Specification: End-to-End Live Replay Simulator & Institutional Benchmarking

## 1. Executive Overview & Problem Statement

Phase 5 Steps 1 through 3 constructed the analytical pillars of the institutional execution pipeline:
- **Phase 5 Step 1 (`ensemble.py`)**: Regime-Conditioned Dynamic Model Averaging (RD-DMA) with volatility-adaptive forgetting ($\alpha_t \in [0.85, 0.99]$), forward-Markov regime projection, asymmetric downside loss scoring ($\ell_{t, k}$), Tikhonov ridge correlation ($\mathbf{C}_t \succ 0$), Entropic Mirror Descent on the simplex, thermodynamic ambiguity shrinkage ($\beta_t$), turnover damping ($\lambda_{\text{churn}}$), and Law-of-Total-Variance risk decomposition ($\sigma^2_{\text{total}} = \sigma^2_{\text{aleatoric}} + \sigma^2_{\text{epistemic}}$).
- **Phase 5 Step 2 (`circuit_breakers.py`)**: Epistemic Disagreement Entropy & Circuit Breaker Overlays with 3-simplex directional consensus probabilities $\mathbf{p} \in \Delta^3$, normalized Shannon entropy $\widetilde{H}_{\text{dir}}$, epistemic uncertainty ratio $\rho_{\text{epistemic}}$, continuous logistic haircut $\kappa_t \in [0.0, 1.0]$, 4-tier discrete risk hierarchy (`NORMAL`, `CAUTION`, `DERISK`, `HALT`), and anti-chattering hysteresis state machine with dwell-time cooling lockouts ($\tau_{\text{dwell}} \ge 5$ bars in `HALT`/`DERISK`).
- **Phase 5 Step 3 (`tail_risk.py`, `execution_sizing.py`)**: Semi-Parametric Peaks-Over-Threshold Extreme Value Theory (EVT-POT) with closed-form Probability Weighted Moments (PWM), coherent Expected Shortfall ($\text{CVaR}_\alpha \ge \text{VaR}_\alpha$), Fréchet stability ($\xi \in [0.001, 0.999]$), infinite variance tripwire ($\xi \ge 1.0 \implies \text{HALT}$), 3-tier cold-start degradation ladder, strictly concave sizing objective ($\nabla^2 \mathcal{L} \prec 0$), directional epistemic shrinkage, 3/2-power Pseudo-Huber friction, and exact $O(N \log N)$ dual projection onto gross leverage and CVaR drawdown budgets with randomized lot discretization ($\mathbb{E}[\tilde{\boldsymbol{\nu}}] = \boldsymbol{\nu}^*$).

**Phase 5 Step 4** provides the capstone institutional integration: the **End-to-End Live Replay Simulator & Institutional Benchmarking System** (`src/quant/analytics/simulation.py`).

Traditional backtesters exhibit four fatal flaws that inflate paper performance:
1. **Lookahead Leakage & Temporal Contamination**: Using contemporaneous prices $p_t$ or volume to determine signals that are executed at $t$, introducing predictive leak.
2. **Frictionless Fantasy Execution**: Neglecting non-linear price impact ($\psi'_{3/2}(u)$), bid-ask spread crossing friction, and exchange fee schedules, causing high-turnover strategies to appear profitable when they are net negative.
3. **Selection Bias & Non-Normality Blindness**: Reporting standard Sharpe ratios without adjusting for skewness, kurtosis, and the multiple testing hurdle across candidate models, violating statistical discovery principles.
4. **Tangled Monolithic Architectures**: Conflating market data iteration, order matching, accounting ledgers, and reporting into a single messy loop that cannot be isolated or unit-tested.

Phase 5 Step 4 addresses these failures by implementing a **Modular State-Machine Simulation Rig** with decoupled components, event listener hooks, non-linear market friction, strict zero-lookahead causality (`INV-SIM-001`), capital conservation (`INV-SIM-002`), and institutional statistical certification via the Deflated Sharpe Ratio (DSR) and Minimum Backtest Length (MinBTL) from Phase 2 Step 6.

---

## 2. Mathematical Formulation & Architecture

### 2.1 Decoupled Modular Architecture
The simulation rig is partitioned into four independent classes:
```
+---------------------------------------------------------------------------------------------------------+
|                                    LIVE REPLAY SIMULATION ARCHITECTURE                                  |
+---------------------------------------------------------------------------------------------------------+
| [ReplayEngine] (Master Causal Orchestrator)                                                             |
|   ├── Inputs: Multi-Asset Returns R in R^{T x N}, Candidate Model Forecasts, Regime & Volatility Series  |
|   ├── Event Hooks: on_bar_start, on_decision, on_fill, on_bar_end via SimulationListener interface      |
|   │                                                                                                     |
|   ├── Causal Pipeline per Bar t (Zero Lookahead INV-SIM-001):                                           |
|   │     1. DMA Aggregator          ──> EnsemblePrediction (mu_t, aleatoric, epistemic)                  |
|   │     2. Circuit Breaker Engine   ──> Hysteresis State & Continuous Haircut kappa_t                    |
|   │     3. EVT Tail Risk Engine     ──> Coherent Asset Expected Shortfalls c_i                          |
|   │     4. Convex Execution Sizer   ──> Target Allocation nu* & Discretized Lots nu~                    |
|   │                                                                                                     |
|   ├── [ExecutionCostModel]                                                                              |
|   │     └── Calculates: 3/2-power Kyle-Obizhaeva market impact + half-spread slippage + exchange fees   |
|   │                                                                                                     |
|   ├── [PortfolioLedger]                                                                                 |
|   │     └── Updates: cash balance, positions, mark-to-market equity W_t, leverage, running drawdown DD_t|
|   │                                                                                                     |
|   └── [BenchmarkAuditor] (Vectorized Post-Simulation Analysis)                                          |
|         ├── Performance: CAGR, Sharpe, Sortino, Calmar, Max Drawdown, Realized VaR/CVaR 95/99           |
|         ├── Statistical Significance: Deflated Sharpe Ratio (DSR >= 0.95), MinBTL, FDR via Phase 2 Step 6|
|         └── Institutional Comparison: Alpha, Beta, Information Ratio vs Equal Weight / Risk Parity / Cash |
+---------------------------------------------------------------------------------------------------------+
```

### 2.2 Execution Cost Model (`ExecutionCostModel`)
For a target position vector adjustment $\Delta \boldsymbol{\nu}_t = \tilde{\boldsymbol{\nu}}_t - \tilde{\boldsymbol{\nu}}_{t-1} \in \mathbb{R}^N$, the execution friction $\mathcal{C}(\Delta \boldsymbol{\nu}_t)$ combines three orthogonal components:
$$\mathcal{C}(\Delta \boldsymbol{\nu}_t) = \mathcal{C}_{\text{fee}}(\Delta \boldsymbol{\nu}_t) + \mathcal{C}_{\text{spread}}(\Delta \boldsymbol{\nu}_t) + \mathcal{C}_{\text{impact}}(\Delta \boldsymbol{\nu}_t)$$

1. **Exchange Fee Cost**:
   $$\mathcal{C}_{\text{fee}}(\Delta \boldsymbol{\nu}_t) = \text{fee}_{\text{bps}} \cdot 10^{-4} \cdot \|\Delta \boldsymbol{\nu}_t\|_1$$
2. **Bid-Ask Spread Slippage**:
   $$\mathcal{C}_{\text{spread}}(\Delta \boldsymbol{\nu}_t) = \frac{\text{spread}_{\text{bps}} \cdot 10^{-4}}{2} \cdot \|\Delta \boldsymbol{\nu}_t\|_1$$
3. **3/2-Power Non-Linear Market Impact**:
   From Phase 3 Step 2 Kyle-Obizhaeva generalized Pseudo-Huber formulation:
   $$\mathcal{C}_{\text{impact}}(\Delta \boldsymbol{\nu}_t) = \sum_{i=1}^N \lambda_{\text{impact}, i} \cdot \sigma_i \cdot |\Delta \nu_{i, t}|^{3/2}$$
   where $\lambda_{\text{impact}, i} = \frac{\eta_{\text{base}}}{\text{ADV}_i^{1/2}}$ scales inversely with the square root of average daily volume ($\text{ADV}_i$).

**Invariance**: $\mathcal{C}(\Delta \boldsymbol{\nu}_t) \ge 0.0$ everywhere (`INV-SIM-004`). If $\Delta \boldsymbol{\nu}_t = \mathbf{0}$, $\mathcal{C} \equiv 0.0$.

### 2.3 Portfolio Accounting Ledger (`PortfolioLedger`)
At each bar $t \in [0, T-1]$, the ledger maintains:
- Cash Balance: $\text{cash}_t \in \mathbb{R}$
- Position Holdings: $\boldsymbol{\nu}_t = (\nu_{1, t}, \dots, \nu_{N, t})^T \in \mathbb{R}^N$
- Mark-to-Market Wealth (Equity):
  $$W_t = \text{cash}_t + \sum_{i=1}^N \nu_{i, t}$$
- Gross Leverage:
  $$L_t = \frac{\|\boldsymbol{\nu}_t\|_1}{W_t}$$
- High-Water Mark:
  $$\text{HWM}_t = \max_{0 \le s \le t} W_s$$
- Peak-to-Trough Drawdown:
  $$\text{DD}_t = \frac{\text{HWM}_t - W_t}{\text{HWM}_t} \in [0.0, 1.0]$$

**Causal PnL Evolution**:
Between bar $t-1$ and bar $t$, the portfolio is invested with positions $\boldsymbol{\nu}_{t-1}$. When the return vector $\mathbf{r}_t \in \mathbb{R}^N$ realizes:
$$\text{PnL}_t^{\text{gross}} = \boldsymbol{\nu}_{t-1}^T \mathbf{r}_t = \sum_{i=1}^N \nu_{i, t-1} \cdot r_{i, t}$$
$$\text{PnL}_t^{\text{net}} = \text{PnL}_t^{\text{gross}} - \mathcal{C}(\Delta \boldsymbol{\nu}_t)$$
$$W_t = W_{t-1} + \text{PnL}_t^{\text{net}}$$
$$\text{cash}_t = W_t - \sum_{i=1}^N \nu_{i, t}$$

**Invariance**: Total wealth strictly satisfies $W_t \equiv \text{cash}_t + \sum \nu_{i, t}$ (`INV-SIM-002`). Ruin condition: If $W_t \le 0$, raises `DegenerateSimulationException` (`ERR-SIM-003`).

### 2.4 Performance & Institutional Benchmarking (`BenchmarkAuditor`)
The return series $r_t^{\text{net}} = \frac{W_t - W_{t-1}}{W_{t-1}}$ is evaluated over $T$ bars:

1. **Compounded Growth & Volatility**:
   - $\text{CAGR} = \left(\prod_{t=1}^T (1 + r_t^{\text{net}})\right)^{252 / T} - 1$
   - $\hat{\sigma}_{\text{ann}} = \sqrt{252} \cdot \sqrt{\frac{1}{T-1} \sum_{t=1}^T (r_t^{\text{net}} - \bar{r})^2}$
2. **Risk-Adjusted Performance**:
   - $\text{Sharpe} = \sqrt{252} \cdot \frac{\bar{r} - r_f}{\hat{\sigma}_{\text{ann}}}$
   - $\text{Sortino} = \sqrt{252} \cdot \frac{\bar{r} - r_f}{\hat{\sigma}_{\text{down}}}$, where $\hat{\sigma}_{\text{down}} = \sqrt{\frac{252}{T} \sum \min(0, r_t^{\text{net}} - r_f)^2}$
   - $\text{Calmar} = \frac{\text{CAGR}}{\text{MDD}}$, where $\text{MDD} = \max_t \text{DD}_t$
3. **Realized Tail Risk**:
   - Empirical $\text{VaR}_{0.95} = -q_{0.05}(\mathbf{r}^{\text{net}})$
   - Empirical $\text{CVaR}_{0.95} = -\frac{1}{\sum \mathbf{1}_{[r_t \le - \text{VaR}]}} \sum_{r_t \le - \text{VaR}} r_t$
   - $\text{Tail Ratio} = \frac{q_{0.95}(\mathbf{r}^{\text{net}})}{|q_{0.05}(\mathbf{r}^{\text{net}})|}$
4. **Deflated Sharpe Ratio (DSR) & Statistical Significance**:
   - Ingests strategy return moments ($\hat{\gamma}_3, \hat{\gamma}_4$) into Phase 2 Step 6 `DeflatedSharpeEngine`.
   - Computes expected maximum Sharpe ratio under selection bias:
     $$\text{E}[\max_{k \le K} \text{SR}_k] \approx (1 - \gamma) \Phi^{-1}\left(1 - \frac{1}{K}\right) + \gamma \Phi^{-1}\left(1 - \frac{1}{K \cdot e}\right)$$
   - Computes probabilistic DSR:
     $$\text{DSR} = \Phi\left(\frac{(\widehat{\text{SR}} - \text{E}[\max \text{SR}]) \sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3 \widehat{\text{SR}} + \frac{\hat{\gamma}_4 - 1}{4} \widehat{\text{SR}}^2}}\right)$$
   - Certifies whether strategy clears institutional hurdle ($\text{DSR} \ge 0.95$) and sample length satisfies $\text{MinBTL}$.
5. **Multi-Benchmark Comparison**:
   - Evaluates parallel institutional baselines: Equal Weight, Risk Parity, Inverse Volatility, and Cash.
   - For each benchmark $b$, computes:
     - Jensen's Alpha: $\alpha = \bar{r}_{\text{strat}} - r_f - \beta (\bar{r}_{\text{bench}} - r_f)$
     - Beta: $\beta = \frac{\text{Cov}(\mathbf{r}_{\text{strat}}, \mathbf{r}_{\text{bench}})}{\text{Var}(\mathbf{r}_{\text{bench}})}$
     - Tracking Error: $\text{TE} = \sqrt{252} \cdot \text{std}(\mathbf{r}_{\text{strat}} - \mathbf{r}_{\text{bench}})$
     - Information Ratio: $\text{IR} = \sqrt{252} \cdot \frac{\bar{r}_{\text{strat}} - \bar{r}_{\text{bench}}}{\text{TE}}$

---

## 3. Invariant Contracts & Defensive Specifications

### 3.1 Invariants Matrix
| Invariant ID | Name | Mathematical Contract | Defensive Enforcement |
| :--- | :--- | :--- | :--- |
| `INV-SIM-001` | Zero-Lookahead Causality | $\boldsymbol{\nu}_t = f(\mathcal{F}_{t-1})$; return $r_{i, t}$ realized on $[t-1, t]$. | Temporal indexing strictly separates decision at $t-1$ from execution at $t$. |
| `INV-SIM-002` | Conservation of Capital | $W_t \equiv \text{cash}_t + \sum_{i=1}^N \nu_{i, t}$ and $W_t - W_{t-1} \equiv \text{PnL}_t^{\text{net}}$. | Validated at every bar $t$; tolerance $< 10^{-6}$. |
| `INV-SIM-003` | Non-Negative Execution Friction | $\mathcal{C}(\Delta \boldsymbol{\nu}_t) \ge 0.0$ for all $\Delta \boldsymbol{\nu}_t$. | Rejects negative fees/slippage with `InfeasibleSimulationException`. |
| `INV-SIM-004` | Risk Budget & Leverage Adherence | $\|\boldsymbol{\nu}_t\|_1 \le L_{\max} W_t + \max_i \Delta \nu_i$; HALT $\implies \boldsymbol{\nu}_t = \mathbf{0}$. | Validates sizer output bounds at each step. |
| `INV-SIM-005` | Statistical Rigor & Non-Finite Protection | $T \ge 30$; all inputs $\in \mathbb{R}$ finite (no NaN/Inf). | Input validation throws `DegenerateSimulationException`. |
| `INV-SIM-006` | Execution Latency SLA | $T=100$ bars, $N=10$ assets completed in $\le 25\text{ms}$. | Automated performance benchmark test with GC controls. |

### 3.2 Diagnostic Fault Codes Matrix
| Fault Code | Exception Class | Cause | Recovery |
| :--- | :--- | :--- | :--- |
| `ERR-SIM-001` | `LookaheadViolationException` | Contemporaneous return or future bar data detected in decision input $t$. | Align time indices so signal uses $t-1$. |
| `ERR-SIM-002` | `DegenerateSimulationException` | Input arrays contain NaN, Inf, or non-finite float values. | Sanitize input tensors; reject non-finite inputs. |
| `ERR-SIM-003` | `DegenerateSimulationException` | Portfolio equity $W_t \le 0.0$ (ruin / bankruptcy). | Terminate replay; log liquidation event. |
| `ERR-SIM-004` | `InfeasibleSimulationException` | Execution friction $\mathcal{C}_t < 0.0$ (spurious negative cost). | Assert non-negative friction parameters. |
| `ERR-SIM-005` | `DegenerateSimulationException` | Sample length $T < 30$ bars (insufficient data for DSR/moments). | Expand historical simulation window $T \ge 30$. |
| `ERR-SIM-006` | `DegenerateSimulationException` | Dimension mismatch between returns ($T \times N$) and asset configs. | Verify universe size $N$ consistency. |

---

## 4. Component Interfaces & Domain Data Classes

### 4.1 Domain Entities (`src/quant/analytics/simulation.py`)
```python
@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Immutable simulation configuration parameters."""

    initial_capital: float = 1_000_000.0
    risk_free_rate: float = 0.02
    fee_bps: float = 2.0
    spread_bps: float = 1.0
    impact_coefficient: float = 0.10
    max_leverage: float = 1.0
    mdd_budget: float = 0.20
    confidence_level: float = 0.95
    annualization_factor: int = 252
    num_trials: int = 100


@dataclass(frozen=True, slots=True)
class BarExecutionRecord:
    """Immutable record of single-bar execution and portfolio state."""

    step_index: int
    timestamp: int
    gross_pnl: float
    net_pnl: float
    friction_cost: float
    portfolio_equity: float
    cash_balance: float
    effective_leverage: float
    drawdown: float
    circuit_breaker_tier: str
    circuit_breaker_haircut: float
    target_allocations: tuple[float, ...]
    discretized_allocations: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    """Performance metrics for a specific benchmark."""

    name: str
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    alpha: float
    beta: float
    tracking_error: float
    information_ratio: float


@dataclass(frozen=True, slots=True)
class BenchmarkAuditReport:
    """Comprehensive institutional tear sheet and statistical significance audit."""

    initial_capital: float
    final_equity: float
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    realized_var_95: float
    realized_cvar_95: float
    realized_var_99: float
    realized_cvar_99: float
    tail_ratio: float
    peak_leverage: float
    deflated_sharpe_ratio: float
    min_backtest_length: float
    is_statistically_significant: bool
    total_friction_cost: float
    circuit_breaker_counts: dict[str, int]
    benchmark_comparisons: dict[str, BenchmarkComparison]
```

### 4.2 Listener Interface
```python
class SimulationListener(Protocol):
    """Protocol for observer event listener hooks."""

    def on_bar_start(self, step: int, timestamp: int) -> None: ...
    def on_decision(self, step: int, decision: SizingDecision) -> None: ...
    def on_fill(self, step: int, record: BarExecutionRecord) -> None: ...
    def on_bar_end(self, step: int, record: BarExecutionRecord) -> None: ...
```

---

## 5. Verification & Testing Strategy

1. **Unit Test Suite (`tests/unit/test_simulation.py`)**:
   - `TestExecutionCostModel`: Validate non-negativity (`INV-SIM-004`), 3/2-power scaling, zero cost on zero trade, fee schedule bounds.
   - `TestPortfolioLedger`: Validate $W_t \equiv \text{cash} + \sum \nu_i$ (`INV-SIM-002`), high-water mark, drawdown, bankruptcy detection (`ERR-SIM-003`).
   - `TestBenchmarkAuditor`: Validate analytical accuracy of CAGR, Sharpe, Sortino, Calmar, realized VaR/CVaR, DSR integration with `DeflatedSharpeEngine`, and multi-benchmark Alpha/Beta/IR calculations.
   - `TestReplayEngine`:
     - 100-bar multi-asset replay coupling RD-DMA, Circuit Breakers, EVT Tail Risk, Sizer, Execution Cost, and Ledger.
     - Zero-lookahead causality test (`INV-SIM-001`): assert that shifting future prices does not alter bar $t$ allocations.
     - Circuit breaker coupling test: verify positions collapse to $0.0$ when HALT is triggered.
     - Listener hook test: verify `SimulationListener` receives all expected bar events.
     - Latency SLA benchmark (`INV-SIM-006`): median replay execution $\le 25\text{ms}$ for 100 bars $\times$ 10 assets.
2. **Quality Gates**:
   - All tests passing 100% with $\ge 90\%$ line coverage on `src/quant/analytics/simulation.py`.
   - `mypy src --strict` with zero errors.
   - `ruff check .` and `ruff format --check .` 100% clean.

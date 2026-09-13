# Phase 5 Step 4: End-to-End Live Replay Simulator & Institutional Benchmarking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the institutional End-to-End Live Replay Simulator & Benchmarking Auditor (`src/quant/analytics/simulation.py`), coupling RD-DMA, Circuit Breakers, EVT Tail Risk, Convex Sizing, non-linear market friction, capital ledger conservation, and Deflated Sharpe Ratio (DSR) statistical certification.

**Architecture:** A decoupled, modular state-machine simulation rig comprising `ExecutionCostModel` (3/2-power Kyle impact + spread + fees), `PortfolioLedger` (causal PnL, cash/position accounting, equity, drawdown), `BenchmarkAuditor` (DSR, MinBTL, VaR/CVaR, and multi-benchmark comparison), and `ReplayEngine` (master causal orchestrator with zero-lookahead information barriers and listener hooks).

**Tech Stack:** Python 3.13, NumPy, SciPy (`scipy.special.ndtri`), PyArrow, DeflatedSharpeEngine, RD-DMA, CircuitBreakers, EVTTailRiskEngine, UnifiedConvexExecutionSizer, Pytest, Mypy (strict), Ruff.

**Spec:** `docs/superpowers/specs/2026-09-13-phase-5-step-4-live-replay-simulator-design.md`

## Global Constraints
- Python 3.13 strict static typing (`mypy src --strict` with zero errors across all source files).
- Zero lint/format deviations (`ruff check .`, `ruff format --check .`).
- Test suite passing 100% with $\ge 90\%$ line coverage on `src/quant/analytics/simulation.py`.
- Strict compliance with invariant contracts:
  - `INV-SIM-001`: Zero-Lookahead Causality ($\boldsymbol{\nu}_t = f(\mathcal{F}_{t-1})$; return $r_{i, t}$ realized on $[t-1, t]$).
  - `INV-SIM-002`: Conservation of Capital ($W_t \equiv \text{cash}_t + \sum \nu_{i, t}$ and $W_t - W_{t-1} \equiv \text{PnL}_t^{\text{net}}$).
  - `INV-SIM-003`: Non-Negative Execution Friction ($\mathcal{C}(\Delta \boldsymbol{\nu}_t) \ge 0.0$).
  - `INV-SIM-004`: Risk Budget & Leverage Adherence ($\|\boldsymbol{\nu}_t\|_1 \le L_{\max} W_t + \max_i \Delta \nu_i$; HALT $\implies \boldsymbol{\nu}_t = \mathbf{0}$).
  - `INV-SIM-005`: Statistical Rigor & Non-Finite Protection ($T \ge 30$; all inputs finite, no NaN/Inf).
  - `INV-SIM-006`: Hot-path execution latency SLA ($\le 25\text{ms}$ for $T=100$ bars, $N=10$ assets).
- Strict compliance with `Rules.md`:
  - Rule 1: Line-by-line annotations (Functional Purpose, Explicit Dependency Tracking, Structural Relationship, Defensive Invariants).
  - Rule 2: Diagnostic error codes registered (`ERR-SIM-001` through `ERR-SIM-006`).
  - Rule 3: Quality gates.
  - Rule 4: Mandatory adversarial red-teaming, superior alternatives, anti-shortcut blacklist (zero `scipy.optimize` in hot path).

---

### Task 1: Simulation Domain Entities, Exceptions & Diagnostic Codes

**Files:**
- Create: `src/quant/analytics/simulation.py`
- Test: `tests/unit/test_simulation.py`

**Interfaces:**
- Produces:
  - `ERR_SIM_LOOKAHEAD_VIOLATION = "ERR-SIM-001"`
  - `ERR_SIM_NON_FINITE_INPUT = "ERR-SIM-002"`
  - `ERR_SIM_CAPITAL_RUIN = "ERR-SIM-003"`
  - `ERR_SIM_NEGATIVE_FRICTION = "ERR-SIM-004"`
  - `ERR_SIM_STARVATION = "ERR-SIM-005"`
  - `ERR_SIM_DIMENSION_MISMATCH = "ERR-SIM-006"`
  - `SimulationError(Exception)`
  - `LookaheadViolationException(SimulationError)`
  - `DegenerateSimulationException(SimulationError)`
  - `InfeasibleSimulationException(SimulationError)`
  - `SimulationConfig(initial_capital, risk_free_rate, fee_bps, spread_bps, impact_coefficient, max_leverage, mdd_budget, confidence_level, annualization_factor, num_trials)`
  - `BarExecutionRecord(step_index, timestamp, gross_pnl, net_pnl, friction_cost, portfolio_equity, cash_balance, effective_leverage, drawdown, circuit_breaker_tier, circuit_breaker_haircut, target_allocations, discretized_allocations)`
  - `BenchmarkComparison(name, total_return, annualized_return, annualized_volatility, sharpe_ratio, max_drawdown, alpha, beta, tracking_error, information_ratio)`
  - `BenchmarkAuditReport(initial_capital, final_equity, total_return, cagr, annualized_volatility, sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown, realized_var_95, realized_cvar_95, realized_var_99, realized_cvar_99, tail_ratio, peak_leverage, deflated_sharpe_ratio, min_backtest_length, is_statistically_significant, total_friction_cost, circuit_breaker_counts, benchmark_comparisons)`
  - `SimulationListener(Protocol)`

- [ ] **Step 1: Write failing unit tests for domain entities and exceptions**

```python
# tests/unit/test_simulation.py
import pytest
from quant.analytics.simulation import (
    ERR_SIM_LOOKAHEAD_VIOLATION,
    ERR_SIM_NON_FINITE_INPUT,
    ERR_SIM_CAPITAL_RUIN,
    ERR_SIM_NEGATIVE_FRICTION,
    ERR_SIM_STARVATION,
    ERR_SIM_DIMENSION_MISMATCH,
    SimulationError,
    LookaheadViolationException,
    DegenerateSimulationException,
    InfeasibleSimulationException,
    SimulationConfig,
    BarExecutionRecord,
    BenchmarkComparison,
    BenchmarkAuditReport,
)

class TestSimulationEntitiesAndExceptions:
    def test_diagnostic_codes_and_exception_hierarchy(self) -> None:
        err = LookaheadViolationException("Lookahead", code=ERR_SIM_LOOKAHEAD_VIOLATION)
        assert isinstance(err, SimulationError)
        assert err.code == "ERR-SIM-001"

    def test_simulation_config_validation(self) -> None:
        cfg = SimulationConfig()
        assert cfg.initial_capital == 1_000_000.0
        with pytest.raises(DegenerateSimulationException):
            SimulationConfig(initial_capital=-100.0)
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_simulation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quant.analytics.simulation'`

- [ ] **Step 3: Implement domain entities, exceptions, and diagnostic codes**

Implement in `src/quant/analytics/simulation.py` with full Rule 1 line annotations and validation.

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/unit/test_simulation.py -v`
Expected: PASS

- [ ] **Step 5: Commit Task 1**

```bash
git add src/quant/analytics/simulation.py tests/unit/test_simulation.py
git commit -m "feat(simulation): define simulation domain entities, exceptions, and diagnostic codes (Phase 5 Step 4 Task 1)"
```

---

### Task 2: Execution Cost Model (`ExecutionCostModel`)

**Files:**
- Modify: `src/quant/analytics/simulation.py`
- Test: `tests/unit/test_simulation.py`

**Interfaces:**
- Produces:
  - `ExecutionCostModel`:
    - `__init__(fee_bps: float, spread_bps: float, impact_coefficient: float)`
    - `compute_cost(delta_positions: np.ndarray, asset_volatilities: np.ndarray, advs: np.ndarray | None = None) -> float`

- [ ] **Step 1: Write failing tests for ExecutionCostModel**

```python
def test_execution_cost_model_zero_trade_is_zero() -> None:
    model = ExecutionCostModel(fee_bps=2.0, spread_bps=1.0, impact_coefficient=0.10)
    cost = model.compute_cost(np.zeros(5), np.full(5, 0.02))
    assert cost == 0.0

def test_execution_cost_model_non_negative_invariant_inv_sim_003() -> None:
    model = ExecutionCostModel(fee_bps=2.0, spread_bps=1.0, impact_coefficient=0.10)
    delta = np.array([1000.0, -500.0, 200.0])
    vols = np.array([0.02, 0.03, 0.015])
    cost = model.compute_cost(delta, vols)
    assert cost > 0.0
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_simulation.py -k test_execution_cost_model -v`
Expected: FAIL with `NameError: name 'ExecutionCostModel' is not defined`

- [ ] **Step 3: Implement ExecutionCostModel**

In `src/quant/analytics/simulation.py`:
- Implement fee component, spread crossing component, and 3/2-power Kyle-Obizhaeva non-linear impact component.
- Guard against non-finite inputs (`ERR-SIM-002`), negative parameters (`ERR-SIM-004`).
- Guarantee `INV-SIM-003: cost >= 0.0`.

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/unit/test_simulation.py -k test_execution_cost_model -v`
Expected: PASS

- [ ] **Step 5: Commit Task 2**

```bash
git add src/quant/analytics/simulation.py tests/unit/test_simulation.py
git commit -m "feat(simulation): implement ExecutionCostModel with 3/2-power Kyle impact and fee schedules (Phase 5 Step 4 Task 2)"
```

---

### Task 3: Portfolio Accounting Ledger (`PortfolioLedger`)

**Files:**
- Modify: `src/quant/analytics/simulation.py`
- Test: `tests/unit/test_simulation.py`

**Interfaces:**
- Produces:
  - `PortfolioLedger`:
    - `__init__(initial_capital: float)`
    - `update(step_index: int, timestamp: int, return_vector: np.ndarray, new_positions: np.ndarray, friction_cost: float, circuit_breaker_tier: str, circuit_breaker_haircut: float, target_allocations: np.ndarray) -> BarExecutionRecord`
    - Property: `current_equity: float`
    - Property: `current_cash: float`
    - Property: `current_positions: np.ndarray`
    - Property: `high_water_mark: float`
    - Property: `current_drawdown: float`
    - Property: `history: list[BarExecutionRecord]`

- [ ] **Step 1: Write failing tests for PortfolioLedger**

```python
def test_portfolio_ledger_capital_conservation_inv_sim_002() -> None:
    ledger = PortfolioLedger(initial_capital=100_000.0)
    ret = np.array([0.01, -0.005])
    pos = np.array([20_000.0, 10_000.0])
    record = ledger.update(
        step_index=0, timestamp=1000, return_vector=ret, new_positions=pos, friction_cost=5.0,
        circuit_breaker_tier="NORMAL", circuit_breaker_haircut=1.0, target_allocations=pos
    )
    assert abs(record.portfolio_equity - (record.cash_balance + float(np.sum(pos)))) < 1e-5
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_simulation.py -k test_portfolio_ledger -v`
Expected: FAIL with `NameError: name 'PortfolioLedger' is not defined`

- [ ] **Step 3: Implement PortfolioLedger**

In `src/quant/analytics/simulation.py`:
- Track causal step by step PnL: $\text{PnL}_t^{\text{gross}} = \boldsymbol{\nu}_{t-1}^T \mathbf{r}_t$.
- Deduct friction: $\text{PnL}_t^{\text{net}} = \text{PnL}_t^{\text{gross}} - \mathcal{C}_t$.
- Update wealth: $W_t = W_{t-1} + \text{PnL}_t^{\text{net}}$.
- Update cash: $\text{cash}_t = W_t - \sum \nu_{i, t}$.
- Enforce `INV-SIM-002: W_t == cash_t + sum nu_i`.
- Tripwire: If $W_t \le 0.0$, raise `DegenerateSimulationException(ERR-SIM-003)`.
- Record and append frozen `BarExecutionRecord`.

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/unit/test_simulation.py -k test_portfolio_ledger -v`
Expected: PASS

- [ ] **Step 5: Commit Task 3**

```bash
git add src/quant/analytics/simulation.py tests/unit/test_simulation.py
git commit -m "feat(simulation): implement PortfolioLedger with capital conservation and drawdown tracking (Phase 5 Step 4 Task 3)"
```

---

### Task 4: Institutional Benchmark & Statistical Significance Auditor (`BenchmarkAuditor`)

**Files:**
- Modify: `src/quant/analytics/simulation.py`
- Test: `tests/unit/test_simulation.py`

**Interfaces:**
- Consumes:
  - `quant.analytics.deflated_sharpe.DeflatedSharpeEngine` (Phase 2 Step 6)
  - `quant.analytics.payoff_matrix.InstitutionalBenchmarkUniverse` (Phase 3 Step 3)
- Produces:
  - `BenchmarkAuditor`:
    - `__init__(config: SimulationConfig, dsr_engine: DeflatedSharpeEngine | None = None)`
    - `audit(records: list[BarExecutionRecord], asset_returns: np.ndarray, benchmark_returns: dict[str, np.ndarray] | None = None) -> BenchmarkAuditReport`

- [ ] **Step 1: Write failing tests for BenchmarkAuditor**

```python
def test_benchmark_auditor_statistical_significance_and_ratios() -> None:
    config = SimulationConfig(initial_capital=100_000.0, annualization_factor=252)
    auditor = BenchmarkAuditor(config=config)
    # create synthetic records
    ...
    report = auditor.audit(records, asset_returns)
    assert math.isfinite(report.sharpe_ratio)
    assert math.isfinite(report.deflated_sharpe_ratio)
    assert isinstance(report.is_statistically_significant, bool)
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_simulation.py -k test_benchmark_auditor -v`
Expected: FAIL with `NameError: name 'BenchmarkAuditor' is not defined`

- [ ] **Step 3: Implement BenchmarkAuditor**

In `src/quant/analytics/simulation.py`:
- Compute CAGR, annualized volatility, Sharpe, Sortino, Calmar, Max Drawdown.
- Compute realized empirical VaR (95%, 99%), CVaR (95%, 99%), Tail Ratio, Peak Leverage.
- Integrate with `DeflatedSharpeEngine` to compute DSR, MinBTL, and statistical significance gate.
- Evaluate parallel institutional benchmarks (Equal Weight, Risk Parity, Inverse Vol, Cash) and compute Jensen's Alpha, Beta, Tracking Error, Information Ratio.
- Aggregate circuit breaker operational tier counts and total friction.
- Emit frozen `BenchmarkAuditReport`.

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/unit/test_simulation.py -k test_benchmark_auditor -v`
Expected: PASS

- [ ] **Step 5: Commit Task 4**

```bash
git add src/quant/analytics/simulation.py tests/unit/test_simulation.py
git commit -m "feat(simulation): implement BenchmarkAuditor with Deflated Sharpe and multi-benchmark attribution (Phase 5 Step 4 Task 4)"
```

---

### Task 5: Master Causal Replay Engine (`ReplayEngine`)

**Files:**
- Modify: `src/quant/analytics/simulation.py`
- Test: `tests/unit/test_simulation.py`

**Interfaces:**
- Consumes:
  - `ExecutionCostModel`, `PortfolioLedger`, `BenchmarkAuditor`
  - `quant.analytics.ensemble.RegimeConditionedDMAEngine`
  - `quant.analytics.circuit_breakers.CircuitBreakerOverlayEngine`
  - `quant.analytics.tail_risk.EVTTailRiskEngine`
  - `quant.analytics.execution_sizing.UnifiedConvexExecutionSizer`
- Produces:
  - `ReplayEngine`:
    - `__init__(config: SimulationConfig, cost_model: ExecutionCostModel | None = None, auditor: BenchmarkAuditor | None = None)`
    - `add_listener(listener: SimulationListener) -> None`
    - `run(asset_returns: np.ndarray, asset_volatilities: np.ndarray, candidate_predictions: np.ndarray, regime_probabilities: np.ndarray, ambiguity_betas: np.ndarray, timestamps: np.ndarray | None = None) -> BenchmarkAuditReport`

- [ ] **Step 1: Write failing tests for ReplayEngine**

```python
def test_replay_engine_zero_lookahead_and_full_pipeline_inv_sim_001() -> None:
    engine = ReplayEngine(config=SimulationConfig())
    # Run 50 bars x 3 assets
    report = engine.run(...)
    assert report.final_equity > 0.0
    assert len(report.benchmark_comparisons) >= 1
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_simulation.py -k test_replay_engine -v`
Expected: FAIL with `NameError: name 'ReplayEngine' is not defined`

- [ ] **Step 3: Implement ReplayEngine**

In `src/quant/analytics/simulation.py`:
- Initialize sub-engines (`RegimeConditionedDMAEngine`, `CircuitBreakerOverlayEngine`, `EVTTailRiskEngine`, `UnifiedConvexExecutionSizer`, `PortfolioLedger`, `ExecutionCostModel`).
- Execute strict causal chronological loop $t=0 \dots T-1$:
  - Ensure zero lookahead (`INV-SIM-001`): allocation decision at $t$ strictly conditions on lagged info ($t-1$ returns, volatilities, regimes).
  - Forward-project DMA $\to$ `EnsemblePrediction`.
  - Circuit Breaker $\to$ tier & continuous haircut $\kappa_t$.
  - EVT Tail Risk $\to$ asset Expected Shortfalls $\mathbf{c}_t$.
  - Sizer $\to$ target continuous $\boldsymbol{\nu}_t^*$ and discretized lots $\tilde{\boldsymbol{\nu}}_t$.
  - Cost Model $\to$ friction $\mathcal{C}_t$.
  - Ledger $\to$ update positions and realize market return $\boldsymbol{\nu}_{t-1}^T \mathbf{r}_t$.
  - Dispatch listener events (`on_bar_start`, `on_decision`, `on_fill`, `on_bar_end`).
- Invoke `BenchmarkAuditor.audit()` to return complete `BenchmarkAuditReport`.

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/unit/test_simulation.py -k test_replay_engine -v`
Expected: PASS

- [ ] **Step 5: Commit Task 5**

```bash
git add src/quant/analytics/simulation.py tests/unit/test_simulation.py
git commit -m "feat(simulation): implement ReplayEngine coordinating causal lifecycle and listener hooks (Phase 5 Step 4 Task 5)"
```

---

### Task 6: Master Exports, System Integration & Latency Benchmark SLA

**Files:**
- Modify: `src/quant/analytics/__init__.py`
- Modify: `tests/unit/test_simulation.py`

**Interfaces:**
- Exports all public simulation symbols in `src/quant/analytics/__init__.py`:
  - `SimulationConfig`, `BarExecutionRecord`, `BenchmarkComparison`, `BenchmarkAuditReport`, `SimulationListener`
  - `ExecutionCostModel`, `PortfolioLedger`, `BenchmarkAuditor`, `ReplayEngine`
  - `SimulationError`, `LookaheadViolationException`, `DegenerateSimulationException`, `InfeasibleSimulationException`
  - `ERR_SIM_LOOKAHEAD_VIOLATION`, `ERR_SIM_NON_FINITE_INPUT`, `ERR_SIM_CAPITAL_RUIN`, `ERR_SIM_NEGATIVE_FRICTION`, `ERR_SIM_STARVATION`, `ERR_SIM_DIMENSION_MISMATCH`

- [ ] **Step 1: Export symbols in `src/quant/analytics/__init__.py`**

- [ ] **Step 2: Add comprehensive unit tests in `tests/unit/test_simulation.py`**
  - Verify public exports.
  - Zero-lookahead perturbation test: shifting future prices at $t+10$ does not change allocations at $t$.
  - Circuit Breaker emergency HALT test: verify target allocations crush strictly to $0.0$.
  - Invariant `INV-SIM-002` capital conservation across all bars.
  - Invariant `INV-SIM-006` Latency SLA benchmark: 100 bars $\times$ 10 assets $\le 25\text{ms}$ median execution time.

- [ ] **Step 3: Run full verification suite**

Run: `pytest tests/unit`
Expected: 100% green (all existing + new tests passing).
Run: `mypy src --strict`
Expected: 0 errors across all source files.
Run: `ruff check .` and `ruff format --check .`
Expected: 0 deviations.

- [ ] **Step 4: Commit Task 6**

```bash
git add src/quant/analytics/__init__.py tests/unit/test_simulation.py
git commit -m "feat(simulation): export simulation symbols and verify latency SLA and pipeline invariants (Phase 5 Step 4 Task 6)"
```

---

### Task 7: Documentation, Architecture Synchronization & ADR-021

**Files:**
- Modify: `Memory.md`
- Modify: `Phases.md`
- Modify: `Architecture.md`
- Modify: `PRD.md`

- [ ] **Step 1: Document ADR-021 in `Memory.md`**
  - Document Context, Decision, Architecture, Invariants, Alternatives Evaluated, and Trade-offs for the Live Replay Simulator & Institutional Benchmarking System.
  - Register diagnostic fault codes `ERR-SIM-001` through `ERR-SIM-006`.

- [ ] **Step 2: Update `Phases.md`**
  - Mark Phase 5 Step 4 as COMPLETE with full deliverables and test coverage summary.

- [ ] **Step 3: Update `Architecture.md`**
  - Add Section 4.19 detailing the End-to-End Live Replay Simulation and Institutional Benchmarking pipeline.

- [ ] **Step 4: Update `PRD.md`**
  - Update Feature Requirements Matrix with Phase 5 Step 4 completion status.

- [ ] **Step 5: Run CI gates**

Run: `ruff check . ; ruff format --check . ; mypy src --strict ; pytest tests/unit`

- [ ] **Step 6: Commit Task 7**

```bash
git add Memory.md Phases.md Architecture.md PRD.md
git commit -m "docs: synchronize ADR-021, diagnostic codes, architecture, PRD, and Phase 5 Step 4 roadmap (Phase 5 Step 4 Task 7)"
```

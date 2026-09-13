# Implementation Plan: Phase 5, Step 3 — Semi-Parametric EVT Tail Risk & Unified Convex Execution Sizing Calibration

- **Design Spec:** `docs/superpowers/specs/2026-09-13-phase-5-step-3-tail-risk-sizing-design.md`
- **Governing Rule:** `Rules.md` (Rule 4: Mandatory Adversarial Red-Teaming, Superior Alternatives, Anti-Shortcut Blacklist)
- **Target Files:**
  - `src/quant/analytics/tail_risk.py`
  - `src/quant/analytics/execution_sizing.py`
  - `src/quant/analytics/__init__.py`
- **Test Files:**
  - `tests/unit/test_tail_risk.py`
  - `tests/unit/test_execution_sizing.py`

---

## Tasks & Execution Roadmap

### Task 1: EVT-GPD Tail Risk Domain Entities, Configuration, and Invariants
- **Target File:** `src/quant/analytics/tail_risk.py`
- **Test File:** `tests/unit/test_tail_risk.py`
- **Scope:**
  - Define `EVTTailParameters`, `TailRiskMetrics`, `TailRiskConfig`.
  - Define exception hierarchy: `TailRiskError`, `DegenerateTailRiskException`, `InfiniteVarianceException`.
  - Enforce Invariant `INV-TR-001` ($\text{CVaR}_\alpha \ge \text{VaR}_\alpha$) and `INV-TR-002` ($\xi \in [0.001, 0.999]$, $\xi \ge 1.0 \implies \text{HALT}$).
  - Implement frozen dataclasses with defensive immutability checks.

### Task 2: Closed-Form Probability Weighted Moments (PWM) & GPD Tail Estimator
- **Target File:** `src/quant/analytics/tail_risk.py`
- **Test File:** `tests/unit/test_tail_risk.py`
- **Scope:**
  - Implement `ProbabilityWeightedMomentsEstimator` calculating $M_0, M_1$ and closed-form $(\xi_{\text{PWM}}, \beta_{\text{PWM}})$.
  - Zero iterative numerical optimizers (`scipy.optimize` banned under Rule 4.3).
  - Implement dynamic high threshold $u_t = \mu_{t-1} + k_{\text{threshold}} \sigma_{t-1}$ and exceedance extraction.
  - Robust fallback to exponential distribution on degenerate moments ($M_0 - 2M_1 \le 0$).
  - Infinite variance tripwire ($\xi \ge 1.0 \implies \text{InfiniteVarianceException}$).

### Task 3: Coherent Expected Shortfall Engine & Cold-Start Degradation Ladder
- **Target File:** `src/quant/analytics/tail_risk.py`
- **Test File:** `tests/unit/test_tail_risk.py`
- **Scope:**
  - Implement `EVTTailRiskEngine` evaluating closed-form $\text{VaR}_\alpha$ and $\text{CVaR}_\alpha$.
  - Implement 3-tier Cold-Start Degradation Ladder:
    - Tier 1: $N < 30$ or $N_u < 10 \implies$ Empirical Order Statistics.
    - Tier 2: $30 \le N < 250 \implies$ Student-t Method-of-Moments.
    - Tier 3: $N \ge 250 \implies$ Full EVT-POT GPD with PWM.
  - Enforce subadditivity `INV-TR-003` and zero-lookahead causality `INV-TR-007` with lagged rolling ring buffer $[t-W, t-1]$.
  - Microsecond latency SLA benchmark: $\le 0.05\text{ms}$ execution.

### Task 4: Sizing Domain Entities, Market Impact Penalty & Uncertainty-Shrunk Kelly
- **Target File:** `src/quant/analytics/execution_sizing.py`
- **Test File:** `tests/unit/test_execution_sizing.py`
- **Scope:**
  - Define `SizingConfig`, `SizingDecision`, exceptions (`SizingError`, `InfeasibleSizingException`).
  - Implement `UncertaintyShrunkKellyUtility`: penalizes expected return by epistemic disagreement $\lambda_{\text{shrink}} \boldsymbol{\sigma}^2_{\text{epistemic}}$.
  - Implement `PseudoHuberImpactPenalty`: 3/2-power non-linear execution friction and permanent cross-impact $\mathbf{\Lambda}_{\text{cross}}$.
  - Implement circuit breaker covariance regularizer $\frac{1}{2\kappa_t W_t} \|\boldsymbol{\nu}\|^2$: smoothly contracts sizing to $\mathbf{0}$ as $\kappa_t \to 0$.
  - Enforce strict concavity `INV-TR-004` ($\nabla^2 U \prec 0$).

### Task 5: Unified Convex Execution Sizer, Hard MDD Budget & Lot Discretization
- **Target File:** `src/quant/analytics/execution_sizing.py`
- **Test File:** `tests/unit/test_execution_sizing.py`
- **Scope:**
  - Implement `UnifiedConvexExecutionSizer`: Projected Gradient / KKT solver with Armijo line search for optimal continuous allocation $\boldsymbol{\nu}^*$.
  - Enforce Hard Drawdown Budget via Rockafellar-Uryasev CVaR: $\text{CVaR}_\alpha(\boldsymbol{\nu}) \le \text{MDD}_{\text{budget}} \cdot W_t$ (`INV-TR-005`).
  - Enforce Gross Leverage Cap: $\|\boldsymbol{\nu}\|_1 \le L_{\max} \cdot W_t$.
  - Implement randomized lot rounding for exchange microstructure, preserving risk budget in expectation.

### Task 6: Master Pipeline Integration, Public Symbol Exports & End-to-End Benchmarks
- **Target File:** `src/quant/analytics/__init__.py`
- **Test Files:** `tests/unit/test_tail_risk.py`, `tests/unit/test_execution_sizing.py`
- **Scope:**
  - Export all tail risk and execution sizing symbols in `src/quant/analytics/__init__.py`.
  - Closed-loop 50-bar rolling simulation coupling `RD-DMA` + `CircuitBreakerOverlayEngine` + `EVTTailRiskEngine` + `UnifiedConvexExecutionSizer`.
  - Performance SLA verification (`INV-TR-006`): Full pipeline cycle $\le 0.20\text{ms}$ for $N=10$ assets.

### Task 7: Documentation Synchronization & Whole-Branch Quality Verification
- **Target Files:** `Memory.md`, `Phases.md`, `Architecture.md`, `PRD.md`
- **Scope:**
  - Add ADR-020 to `Memory.md`, Fault Vectors `ERR-TR-001` through `ERR-TR-006`, Phase 28 Audit Trail.
  - Update `Phases.md` (Gantt chart and Phase 5 Step 3 marked complete).
  - Update `Architecture.md` (Section 3 module inventory and Section 4.18 Sizing Pipeline).
  - Update `PRD.md` Feature Requirements Matrix.
  - Run full verification: `pytest tests/unit`, `mypy src --strict`, `ruff check .`, `ruff format --check .`.

---

## Verification Plan

- Unit test suites: `pytest tests/unit/test_tail_risk.py`, `pytest tests/unit/test_execution_sizing.py`
- Subsystem coverage gate: $\ge 90\%$ line coverage on `tail_risk.py` and `execution_sizing.py`.
- Full project test suite: `pytest tests/unit` ($514+$ tests passing with zero regressions).
- Static typing: `mypy src --strict` with zero errors across all source files.
- Code style: `ruff check .` and `ruff format --check .`.
- Benchmark SLA: Sub-$0.20\text{ms}$ execution latency clocked without profiling overhead.

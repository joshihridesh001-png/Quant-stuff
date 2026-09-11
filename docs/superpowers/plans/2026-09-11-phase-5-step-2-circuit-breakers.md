# Implementation Plan: Phase 5, Step 2 — Epistemic Disagreement Entropy & Circuit Breaker Overlays

Implement the institutional-grade **Circuit Breaker Overlays & Epistemic Disagreement Entropy subsystem** (`src/quant/analytics/circuit_breakers.py`) for the Quantitative Prediction Engine.

---

## User Review Required

> [!IMPORTANT]
> - Phase 5 Step 2 introduces continuous position sizing haircuts $\kappa_t \in [0.0, 1.0]$ coupled with hard 4-tier circuit breakers (`NORMAL`, `CAUTION`, `DERISK`, `HALT`).
> - An anti-chattering hysteresis state machine enforces a minimum dwell time ($\tau_{\text{dwell}} = 5$ bars) and asymmetric recovery barrier before releasing emergency halts.
> - High epistemic disagreement between candidate models will automatically de-leverage positions before loss materialization.

---

## Proposed Changes

### Core Analytics

#### [NEW] `src/quant/analytics/circuit_breakers.py`
- Domain Enums & Entities: `CircuitBreakerTier`, `CircuitBreakerConfig`, `CircuitBreakerState`, `CircuitBreakerDecision`, and custom exceptions (`CircuitBreakerError`, `DegenerateCircuitBreakerException`, `InvalidCircuitBreakerInputException`).
- `EpistemicEntropyCalculator`: Evaluates 3-simplex directional consensus probabilities $(p_+, p_-, p_0)$, normalized Shannon directional entropy $\tilde{\mathcal{H}}_{\text{dir}}$, epistemic uncertainty ratio $\rho_{\text{epistemic}}$, and composite epistemic entropy $\mathcal{H}_{\text{epistemic}}$.
- `ContinuousHaircutCalculator`: Evaluates smooth logistic sigmoid haircut multiplier $\kappa_t \in [0.0, 1.0]$ from thermodynamic composite shock score $\Xi_t$.
- `CircuitBreakerOverlayEngine`: Master facade orchestrating instantaneous tier escalation, hysteresis recovery, dwell-time lockouts, and execution decision packaging.

#### [MODIFY] `src/quant/analytics/__init__.py`
- Export all Phase 5 Step 2 classes, enums, exceptions, and engines.

---

### Testing

#### [NEW] `tests/unit/test_circuit_breakers.py`
- Comprehensive unit test suite:
  - Invariants `INV-CB-001` through `INV-CB-006`.
  - Directional consensus 3-simplex conservation and Shannon entropy properties.
  - Epistemic uncertainty ratio dynamics.
  - Continuous haircut monotonicity and smooth scaling.
  - Multi-tier escalation (`NORMAL` $\to$ `CAUTION` $\to$ `DERISK` $\to$ `HALT`).
  - Stateful hysteresis and anti-chattering dwell time validation.
  - Defensive non-finite and anomaly exception handling.
  - Sub-0.20ms execution latency SLA for $K=100$ models.

---

### Documentation

#### [MODIFY] `Memory.md`
- Record ADR-019 (Epistemic Disagreement Entropy & Circuit Breaker Overlays Architecture).
- Record Fault Vectors `ERR-CB-001` through `ERR-CB-005` in Deterministic Diagnostics table.
- Add Phase 27 Audit Trail entry.

#### [MODIFY] `Phases.md`
- Mark Phase 5 Step 2 as `:done` in Section 1 Mermaid Gantt chart and document deliverables.

#### [MODIFY] `Architecture.md`
- Add `src/quant/analytics/circuit_breakers.py` to module table and create Section 4.17 pipeline documentation.

#### [MODIFY] `PRD.md`
- Update Feature Requirements Matrix for Circuit Breakers Overlay.

---

## Tasks & Execution Roadmap

### Task 1: Domain Entities, Enums, Invariant Contracts, and Configuration
- Produces: `CircuitBreakerTier`, `CircuitBreakerConfig`, `CircuitBreakerState`, `CircuitBreakerDecision`, and exception hierarchy.
- Invariants: `INV-CB-001` ($\kappa_t \in [0, 1]$), `INV-CB-002` (valid tiers), `INV-CB-004` (simplex sum $1.0$).
- Tests: Config validation, state immutability, decision formatting.

### Task 2: Directional Consensus, Shannon Entropy & Epistemic Uncertainty Ratio
- Produces: `EpistemicEntropyCalculator`.
- Formulas:
  - $p_+ = \sum_{\tilde{y}_k > \delta} w_k, p_- = \sum_{\tilde{y}_k < -\delta} w_k, p_0 = \sum_{|\tilde{y}_k| \le \delta} w_k$.
  - $\tilde{\mathcal{H}}_{\text{dir}} = -\sum p_s \ln(p_s + \epsilon) / \ln 3$.
  - $\rho_{\text{epistemic}} = \sigma^2_{\text{epistemic}} / (\sigma^2_{\text{aleatoric}} + \sigma^2_{\text{epistemic}})$.
  - $\mathcal{H}_{\text{epistemic}} = \tilde{\mathcal{H}}_{\text{dir}} \cdot \sqrt{\rho_{\text{epistemic}}}$.
- Tests: Unanimous consensus ($\mathcal{H}=0$), 50/50 polarization ($\mathcal{H} \to 1$), noise dominance ($\rho \to 0$).

### Task 3: Continuous Soft Haircut Function & Thermodynamic Composite Shock
- Produces: `ContinuousHaircutCalculator`.
- Formulas:
  - $\Xi_t = \omega_\mathcal{H} \mathcal{H} + \omega_\rho \rho + \omega_\beta \tilde{\beta}_t$.
  - $\kappa_t = \text{clip}\left(\frac{1}{1 + \exp(k (\Xi_t - \Xi_{\text{mid}}))}, 0.0, 1.0\right)$.
- Tests: Monotonicity $\frac{\partial \kappa}{\partial \Xi} \le 0$, boundary calibration ($\kappa(\Xi=0) = 1.0$, $\kappa(\Xi=1) = 0.0$).

### Task 4: Multi-Tier Discrete Circuit Breakers & Hysteresis State Machine
- Produces: `CircuitBreakerOverlayEngine`.
- Features: Instantaneous escalation (`NORMAL` $\to$ `CAUTION` $\to$ `DERISK` $\to$ `HALT`), dwell time locking ($\tau \ge \tau_{\text{dwell}}$), recovery barrier ($\Xi < \theta_{\text{recovery}}$), CUSUM panic shock handling.
- Tests: Anti-chattering hysteresis, multi-bar recovery sequence, emergency halts.

### Task 5: Master Integration, Export Symbols, and Benchmark SLA
- Integration: Full lifecycle consuming `EnsemblePrediction` and Phase 3 regime state.
- Symbols: Export all 8+ symbols in `src/quant/analytics/__init__.py`.
- Benchmark: Sub-0.20ms execution time for $K=100$ models.
- Tests: 50-bar rolling lifecycle with zero lookahead leak.

### Task 6: Documentation Synchronization & Whole-Branch Quality Verification
- Synchronize `Memory.md`, `Phases.md`, `Architecture.md`, `PRD.md`.
- Run full verification: `ruff`, `mypy --strict`, `pytest tests/unit`.
- Commit documentation.

---

## Verification Plan

### Automated Tests
- Unit test suite: `python -m pytest tests/unit/test_circuit_breakers.py -v`
- Coverage gate: `python -m coverage run -m pytest tests/unit/test_circuit_breakers.py && python -m coverage report -m --include="src/quant/analytics/circuit_breakers.py"` ($\ge 90\%$)
- Full project test suite: `python -m pytest tests/unit` (420+ unit tests passing)
- Static typing: `python -m mypy src --strict` (0 errors across all source files)
- Code formatting & linting: `python -m ruff check .` and `python -m ruff format --check .`

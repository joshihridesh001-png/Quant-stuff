# Implementation Plan: Phase 6, Step 3 — Real-Time Risk Monitor, OMS Heartbeats & Emergency Kill Switch

## Overview & Metadata
- **Milestone**: Phase 6 Step 3 (Live Execution Quality & Safety Gateway)
- **Design Spec**: `docs/superpowers/specs/2026-09-18-phase-6-step-3-risk-monitor-kill-switch-design.md`
- **Governing Standard**: `Rules.md` (Rules 1, 2, 3, 4)
- **Mandatory User Directive**: *"tell the sub agents to use best of the best approach; i want every agent to find flaws in the plan explain why it came out with that solution and find the best of the best approach"*
- **Target Branch**: `feat/phase-6-step-3-risk-monitor-kill-switch`
- **Base Commit**: `3c17f83` (`main`)

---

## Proposed Changes & Task Breakdown

### Task 1: Pre-Trade Risk Firewall, Limits & Fault Codes
- **Files**:
  - `src/quant/execution/risk.py` [NEW]
  - `tests/unit/test_pre_trade_risk.py` [NEW]
- **Deliverables**:
  - Diagnostic fault codes: `ERR-RSK-001` through `ERR-RSK-007`.
  - Immutable domain dataclass `RiskLimits`: `max_order_notional`, `max_order_qty`, `max_gross_leverage`, `max_net_leverage`, `max_concentration_nav_pct`, `max_intraday_drawdown_pct`, `min_free_margin`.
  - Stateful `PortfolioRiskState`: tracks current cash, positions per symbol, open leaves per symbol, peak session equity ($W_{\text{peak}}$), current equity ($W_t$), and real-time intraday drawdown.
  - `PreTradeRiskFirewall`:
    - `validate_order(order: Order, state: PortfolioRiskState) -> None`
    - Evaluates Fat-Finger bounds (`INV-RSK-001`), Gross/Net Leverage bounds (`INV-RSK-002`), Single-Asset NAV Concentration ceiling (`INV-RSK-003`), Intraday Drawdown tripwire (`INV-RSK-004`), Margin sufficiency (`INV-RSK-005`), Sub-10$\mu$s hot path (`INV-RSK-006`), Input sanitization (`INV-RSK-007`).
- **Tests**:
  - Exhaustive unit tests in `test_pre_trade_risk.py` covering boundary triggers, exact threshold rejections, boolean rejections, NaN/Inf defenses, and latency benchmarks.

### Task 2: Exchange Heartbeat & Connection Watchdog
- **Files**:
  - `src/quant/execution/heartbeat.py` [NEW]
  - `tests/unit/test_heartbeat_watchdog.py` [NEW]
- **Deliverables**:
  - Enum `ConnectionStatus`: `CONNECTED`, `DEGRADED`, `DISCONNECTED`, `RECONNECTING`.
  - Dataclass `HeartbeatConfig`: `heartbeat_interval_seconds` (e.g. 1.0s), `timeout_seconds` (e.g. 3.0s), `max_consecutive_misses` (e.g. 3), `max_latency_warning_ms` (e.g. 100ms).
  - Protocol / Class `HeartbeatWatchdog`:
    - Tracks last heartbeat timestamp (nanoseconds), current round-trip ping-pong latency, and expected vs received inbound sequence numbers.
    - Method `record_heartbeat(sequence_number: int, latency_ms: float, timestamp_ns: int) -> None`: detects sequence gaps and latency degradation.
    - Method `check_liveness(current_timestamp_ns: int) -> ConnectionStatus`: evaluates elapsed time, transitions state to `DEGRADED` or `DISCONNECTED`.
    - Event listener callbacks: `on_status_change`, `on_sequence_gap`, `on_disconnect`.
- **Tests**:
  - Unit tests covering sequence gaps, missed ping timeouts, state machine transitions, concurrent heartbeats, and reconnection reset.

### Task 3: Emergency Panic Kill Switch & Bulk Cancellation Engine
- **Files**:
  - `src/quant/execution/kill_switch.py` [NEW]
  - `tests/unit/test_emergency_kill_switch.py` [NEW]
- **Deliverables**:
  - Dataclass `KillSwitchEvent`: record of trigger timestamp, trigger reason (`MANUAL`, `DRAWDOWN`, `DISCONNECT`, `ROGUE_FILLS`), trigger source, open orders cancelled count, positions snapshot.
  - Class `EmergencyKillSwitch`:
    - States: `ARMED_STANDBY`, `PANIC_TRIGGERED`, `DISARMED`.
    - `trigger_panic(reason: str, source: str) -> KillSwitchEvent`:
      - Atomically sets kill state to active (`is_active = True`).
      - Calls bulk order cancellation across all registered gateways asynchronously.
      - Dispatches halt signals to parent order algorithmic schedulers.
      - Rejects subsequent order submissions with `KillSwitchActiveException(ERR-RSK-008)`.
    - `reset(admin_token: str) -> None`: requires explicit manual token confirmation to re-arm.
    - Support for `PanicMode`: `CANCEL_ONLY` (default institutional posture) and `CANCEL_AND_FLATTEN` (emergency market liquidation).
- **Tests**:
  - Unit tests covering manual trigger, automated trigger from risk breach / disconnect, bulk cancellation verification, scheduler freezing, re-submission rejection, and admin reset.

### Task 4: Risk Orchestrator & Live Integration
- **Files**:
  - `src/quant/execution/risk_orchestrator.py` [NEW]
  - `tests/unit/test_risk_orchestrator.py` [NEW]
- **Deliverables**:
  - `RiskOrchestrator`: unified façade integrating `PreTradeRiskFirewall`, `HeartbeatWatchdog`, `EmergencyKillSwitch`, `SmartOrderRouter`, and `ExecutionGateway`.
  - Coordinates order pre-trade clearance $\to$ router / gateway dispatch $\to$ fill tracking $\to$ mark-to-market drawdown update.
  - Automatic tripwire linkage: Heartbeat watchdog disconnect or risk firewall drawdown breach automatically fires `EmergencyKillSwitch`.
- **Tests**:
  - End-to-end integration tests verifying order submission clearance, rejection on risk breach, automatic kill switch tripping on gateway disconnect, and post-kill order rejections.

### Task 5: Repository Synchronization, Exports & Canonical Documentation
- **Files**:
  - `src/quant/execution/__init__.py` [MODIFY] - export all Step 3 symbols, models, exceptions, and engines.
  - `Memory.md` [MODIFY] - Register `ADR-024` and `ERR-RSK-001` through `ERR-RSK-008` in Diagnostic Failure Matrix; log Phase 32 entry.
  - `Phases.md` [MODIFY] - Mark Phase 6 Step 3 as **100% COMPLETE** and Phase 6 overall as **100% COMPLETE**.
  - `Architecture.md` [MODIFY] - Add Section 4.22 (*Real-Time Risk Firewall, OMS Heartbeats & Emergency Kill Switch Subsystem*).
  - `PRD.md` [MODIFY] - Update Section 3.1 Feature Requirements Matrix.
  - `README.md` [MODIFY] - Update badges, test counts, execution architecture tree, and roadmap table.

---

## Quality Gates & Verification Standards
- **Rule 1**: Four-tier line annotations on every function, class, and method.
- **Rule 2**: Diagnostic fault codes `ERR-RSK-001` through `ERR-RSK-008` mapped to exact code coordinates.
- **Rule 3**: 100% test pass rate across entire platform ($1,664 + N$ tests), $\ge 90\%$ statement coverage on new modules, strict static typing (`mypy src --strict` 0 errors), zero linter deviations (`ruff check .`, `ruff format --check .`).
- **Rule 4**: Mandatory adversarial red-teaming subagent audits for each task and the whole branch, zero `scipy.optimize` solvers, sub-$10\mu\text{s}$ pre-trade latency SLA.

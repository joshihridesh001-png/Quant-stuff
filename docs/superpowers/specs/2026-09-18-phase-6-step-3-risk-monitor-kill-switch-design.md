# Design Specification: Phase 6, Step 3 — Real-Time Risk Monitor, OMS Heartbeats & Emergency Kill Switch

**Author:** Antigravity / DeepMind Advanced Agentic Coding  
**Date:** 2026-09-18  
**Status:** In Review / Ready for Implementation  
**Target Milestone:** Phase 6 Step 3 (Live Execution Quality & Safety Gateway)  
**Governing Standard:** `Rules.md` (Rules 1, 2, 3, 4)

---

## 1. Executive Summary & Problem Formulation

In Phase 6 Steps 1 and 2, the execution subsystem established:
1. Pure domain order models, deterministic finite state machine, idempotency routing, high-fidelity paper gateway, and non-blocking asynchronous WAL audit logging (Step 1).
2. Multi-venue consolidated quote parsing, anti-gaming Poisson TWAP, Bayesian volume-adaptive VWAP, closed-form Almgren-Chriss Arrival Price, two-phase dark/lit Smart Order Router with toxic markout watchdog, parent order lifecycle management, and exact additive Perold (1988) Implementation Shortfall TCA (Step 2).

While the router can now optimally slice and route orders, **no institutional algorithmic trading system can be connected to real broker transports without an independent, deterministic, low-latency pre-trade risk firewall and an automated emergency kill switch**:

1. **Rogue Algorithmic Behavior & "Fat-Finger" Outliers**: Sizing bugs, erroneous market data, or loop faults can output orders with astronomical notionals or inverted pricing. The system requires an in-memory, deterministic pre-trade firewall that intercepts and rejects rogue orders in $< 10\mu\text{s}$ before any network packet is dispatched.
2. **Portfolio Leverage & Concentration Creep**: Overlapping algorithmic parent orders across multiple symbols can silently breach regulatory or firm-wide gross leverage limits ($L_{\text{gross}} = \sum |V_i| / W$) or over-concentrate exposure in single assets.
3. **Intraday Drawdown Tripwires**: Macro flash crashes or model breakdown during live market sessions must be detected in real time against high-water marks, tripping automated execution lockouts when losses exceed risk budgets.
4. **Exchange Transport Disconnection & Silent Drops**: FIX / WebSocket connections to broker gateways can silently degrade, drop packets, or hang without throwing immediate OS socket errors. The system requires an active heartbeat watchdog tracking ping/pong latencies and sequence numbers.
5. **Emergency Panic Kill Switch (Hard Mass Cancellation)**: In the event of catastrophic failure, broker malfunction, or risk breach, the system must execute an atomic multi-stage panic protocol: cancel 100% of open resting lit and dark orders across all venues in $< 50\text{ms}$, freeze algorithmic schedulers, lock the pre-trade firewall, and optionally liquidate or freeze inventory.

---

## 2. Invariant Contracts & Mathematical Formulations

Phase 6 Step 3 is governed by eight strict institutional invariants:

### Invariant 1: Single-Order Fat-Finger Bounds (`INV-RSK-001`)
- Every outbound order (parent or child) must strictly satisfy single-order size and notional limits:
  $$Q \le Q_{\max}, \quad Q \cdot P_{\text{limit}} \le \text{Notional}_{\max}$$
- Exceeding notional limits raises `FatFingerNotionalException` (`ERR-RSK-001`).
- Exceeding quantity limits raises `FatFingerQuantityException` (`ERR-RSK-002`).

### Invariant 2: Portfolio Gross & Net Leverage Limits (`INV-RSK-002`)
- Given current portfolio positions $\mathbf{w} \in \mathbb{R}^N$, pending open leaves $\mathbf{q}_{\text{leaves}}$, and current equity $W > 0$:
  $$L_{\text{gross}} = \frac{\sum_{i=1}^N (|w_i| + |q_{\text{leaves}, i}|) \cdot P_i}{W} \le L_{\text{gross}}^{\max}$$
  $$L_{\text{net}} = \frac{\left| \sum_{i=1}^N (w_i + q_{\text{leaves}, i}) \cdot P_i \right|}{W} \le L_{\text{net}}^{\max}$$
- Any order whose execution would cause $L_{\text{gross}} > L_{\text{gross}}^{\max}$ or $L_{\text{net}} > L_{\text{net}}^{\max}$ must be rejected immediately with `LeverageLimitExceededException` (`ERR-RSK-003`).

### Invariant 3: Single-Asset NAV Concentration Ceiling (`INV-RSK-003`)
- The aggregate exposure in any single asset $i$ (current inventory plus open working orders) must not exceed a configured percentage of Net Asset Value:
  $$\frac{(|w_i| + |q_{\text{leaves}, i}|) \cdot P_i}{W} \le \omega_{\max}, \quad \omega_{\max} \in (0, 1]$$
- Breaches raise `ConcentrationLimitExceededException` (`ERR-RSK-004`).

### Invariant 4: Intraday Drawdown Circuit Breaker (`INV-RSK-004`)
- Real-time mark-to-market equity $W_t$ is tracked against the intraday session peak $W_{\text{peak}} = \max_{s \le t} W_s$. The intraday drawdown is:
  $$D_{\text{intraday}}(t) = \frac{W_{\text{peak}} - W_t}{W_{\text{peak}}}$$
- If $D_{\text{intraday}}(t) \ge \text{MDD}_{\text{intraday}}^{\max}$ (e.g. $2.5\%$), new order placement is locked out, and the emergency panic protocol is triggered with `DrawdownLimitExceededException` (`ERR-RSK-006`).

### Invariant 5: Margin & Borrow Sufficiency (`INV-RSK-005`)
- Order margin requirement must not exceed free liquid margin:
  $$M_{\text{req}}(Q, P, \text{side}) \le M_{\text{free}}$$
- Short sells require borrow availability verification. Breaches raise `InsufficientMarginRiskException` (`ERR-RSK-005`).

### Invariant 6: Sub-10$\mu$s Pre-Trade Latency SLA (`INV-RSK-006`)
- Pre-trade risk checks sit directly in the hot path. All validations (fat-finger, leverage, concentration, drawdown, margin) must execute in $< 10\mu\text{s}$ using zero-copy, in-memory scalar operations without disk I/O, network queries, or iterative numerical solvers (`Rules.md` 4.3).

### Invariant 7: Strict Input Sanitization (`INV-RSK-007`)
- Complete boundary rejection of booleans masquerading as numerics, $\text{NaN}$, $\pm\infty$, and non-positive prices/quantities with `NonFiniteRiskInputException` (`ERR-RSK-007`).

### Invariant 8: Atomic Kill Switch Mass Cancellation (`INV-RSK-008`)
- Upon activation of the emergency kill switch (manual or automated), the system must execute the panic protocol in $< 50\text{ms}$:
  1. Transition system state to `KILL_SWITCH_ACTIVE`.
  2. Broadcast batch cancellations for 100% of open resting child and parent orders across all registered gateways.
  3. Terminate all active algorithmic schedulers (`PoissonTWAP`, `VolumeAdaptiveVWAP`, `NonlinearArrivalPrice`).
  4. Reject 100% of subsequent order submissions with `KillSwitchActiveException` (`ERR-RSK-008`).

---

## 3. Deterministic Diagnostic Failure Matrix

| Fault Code | Exception Class | Primary Invariant | Operational Trigger | Automated Remediation |
| :--- | :--- | :--- | :--- | :--- |
| **`ERR-RSK-001`** | `FatFingerNotionalException` | `INV-RSK-001` | Order notional value exceeds firm-wide single-order limit ($Q \cdot P > \text{Notional}_{\max}$). | Reject order; log alert; sizer re-scales order size. |
| **`ERR-RSK-002`** | `FatFingerQuantityException` | `INV-RSK-001` | Order share quantity exceeds firm-wide single-order limit ($Q > Q_{\max}$). | Reject order; verify model target units. |
| **`ERR-RSK-003`** | `LeverageLimitExceededException` | `INV-RSK-002` | Order pushes gross leverage $L_{\text{gross}} > L_{\max}$ or net leverage $L_{\text{net}} > L_{\text{net\_max}}$. | Reject order; trigger portfolio rebalance / derisking. |
| **`ERR-RSK-004`** | `ConcentrationLimitExceededException` | `INV-RSK-003` | Asset exposure exceeds single-name concentration cap ($\omega_i > \omega_{\max}$). | Reject order; cap target allocation to allowable headroom. |
| **`ERR-RSK-005`** | `InsufficientMarginRiskException` | `INV-RSK-005` | Free margin is insufficient or short borrow locate is unavailable. | Reject order; request margin deposit or wait for fill settlement. |
| **`ERR-RSK-006`** | `DrawdownLimitExceededException` | `INV-RSK-004` | Intraday peak-to-trough equity drop exceeds drawdown tripwire ($D_t \ge \text{MDD}_{\max}$). | Activate Emergency Kill Switch; cancel all open orders; freeze trading. |
| **`ERR-RSK-007`** | `NonFiniteRiskInputException` | `INV-RSK-007` | Non-finite scalar (NaN, $\pm\infty$), boolean mask, or non-positive value passed into risk engine. | Reject order; inspect upstream pipeline for NaN propagation. |
| **`ERR-RSK-008`** | `KillSwitchActiveException` | `INV-RSK-008` | Order submission attempted while emergency kill switch is armed and active. | Reject order; require manual administrator reset before resuming. |

---

## 4. Architectural Design & Component Topology

```
+---------------------------------------------------------------------------------------------------+
|                           Phase 6 Step 3: Real-Time Risk & Safety Subsystem                       |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  Upstream Order Pipeline (ParentOrder / SOR Child Slice)                                          |
|                                    |                                                              |
|                                    v                                                              |
|              +-------------------------------------------+                                        |
|              |  PreTradeRiskFirewall (INV-RSK-001..007)  |  <-- Sub-10us In-Memory Check          |
|              |   - Fat-Finger Notional & Qty Guards      |                                        |
|              |   - Gross & Net Leverage Checker          |                                        |
|              |   - Single-Asset Concentration Limiter    |                                        |
|              |   - Margin & Borrow Headroom Check        |                                        |
|              |   - Intraday Drawdown Monitor             |                                        |
|              +-------------------------------------------+                                        |
|                     /                             \                                               |
|           [Pass / Approved]                [Breach / Rejected]                                    |
|                   /                                 \                                             |
|                  v                                   v                                            |
|    +---------------------------+              +------------------------------+                    |
|    | Dispatch to SOR / Gateway |              | Emits ERR-RSK-001..007       |                    |
|    +---------------------------+              | Logs to Audit WAL            |                    |
|                  |                            +------------------------------+                    |
|                  v                                                                                |
|    +---------------------------+                                                                  |
|    | Downstream Gateways       |                                                                  |
|    +---------------------------+                                                                  |
|         ^                 ^                                                                       |
|         |                 |                                                                       |
|         v                 v                                                                       |
|  +-------------------------------+                                                                |
|  | HeartbeatWatchdog             |  <-- Interval tau_hb, Timeout tau_timeout                      |
|  |  - FIX/WS Ping-Pong Latency   |  <-- Sequence Gap Detection                                    |
|  |  - Auto-Disconnect Handler    |                                                                |
|  +-------------------------------+                                                                |
|                 |                                                                                 |
|       [Critical Failure]                                                                          |
|                 |                                                                                 |
|                 v                                                                                 |
|  +----------------------------------------------------------------------+                         |
|  | EmergencyKillSwitch (INV-RSK-008)                                    |                         |
|  |  - Triggers: Manual API, Drawdown Tripwire, Gateway Disconnect      |                         |
|  |  - Actions: Bulk Cancel 100% Orders, Terminate Schedulers,           |                         |
|  |    Lock Pre-Trade Firewall, Snapshot Positions to WAL (< 50ms)        |                         |
|  +----------------------------------------------------------------------+                         |
+---------------------------------------------------------------------------------------------------+
```

### 4.1 New Source Modules
- `src/quant/execution/risk.py`: `RiskLimits` configuration, `PreTradeRiskFirewall`, `PortfolioRiskState`, and exception hierarchy (`ERR-RSK-001` through `ERR-RSK-008`).
- `src/quant/execution/heartbeat.py`: `HeartbeatWatchdog`, `ConnectionStatus` (`CONNECTED`, `DEGRADED`, `DISCONNECTED`), sequence tracking, and timeout listeners.
- `src/quant/execution/kill_switch.py`: `EmergencyKillSwitch`, panic execution protocol, bulk cancellation engine, and state lockdown.
- `src/quant/execution/risk_orchestrator.py`: `RiskOrchestrator` unified facade wrapping gateways/router with pre-trade checks and kill switch hooks.

---

## 5. Verification Plan & Test Strategy

1. **Unit Test Coverage ($\ge 90\%$ statement coverage)**:
   - `tests/unit/test_pre_trade_risk.py`: Tests for fat finger notional/quantity, leverage bounds, concentration limits, margin checks, drawdown calculation, boolean rejection, and sub-$10\mu\text{s}$ latency verification.
   - `tests/unit/test_heartbeat_watchdog.py`: Tests for heartbeat intervals, missed ping timeouts, sequence gap detection, degradation escalation, and reconnection recovery.
   - `tests/unit/test_emergency_kill_switch.py`: Tests for manual panic triggers, automated drawdown triggers, bulk multi-venue order cancellation, scheduler freezing, and firewall lockdown.
   - `tests/unit/test_risk_orchestrator.py`: Tests for end-to-end order flow through risk firewall $\to$ router $\to$ gateway, and kill switch intercept.
2. **Quality Gates**:
   - 100% test pass rate across entire platform ($1,664 + N$ tests).
   - Strict static typing (`mypy src --strict` with 0 errors).
   - Zero lint/formatting deviations (`ruff check .`, `ruff format --check .`).

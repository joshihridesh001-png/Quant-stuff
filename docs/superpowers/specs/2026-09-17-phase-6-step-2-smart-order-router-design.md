# Design Specification: Phase 6, Step 2 — Microstructural Smart Order Router (SOR), Dark Pool Routing & Execution Algorithms

**Author:** Antigravity / DeepMind Advanced Agentic Coding  
**Date:** 2026-09-17  
**Status:** Approved by User  
**Target Milestone:** Phase 6 Step 2  
**Governing Standard:** `Rules.md` (Rules 1, 2, 3, 4)

---

## 1. Executive Summary & Problem Formulation

In Phase 6 Step 1, the execution subsystem established the foundational domain layer: deterministic Order State Machine (`fsm.py`), cryptographic idempotency routing (`idempotency.py`), high-fidelity paper gateway (`gateway.py`), and non-blocking asynchronous WAL audit logging (`audit.py`).

Phase 6 Step 2 transitions the engine from single-venue order routing to **institutional microstructural execution intelligence**:
1. **Liquidity Fragmentation Across Lit & Dark Venues**: Modern equity and crypto markets fragment volume across dozens of lit exchanges and Alternative Trading Systems (ATS) / Dark Pools. Slicing orders across venues requires navigating maker rebates vs. taker fees, queue priorities, and non-displayed midpoint matching.
2. **Anti-Gaming Anti-Frontrunning Execution**: Naive clock-periodic TWAP orders create unmistakable footprint signatures that HFT predatory algorithms detect and exploit within 3–5 slices. Slicing algorithms must randomize both timing (Poisson clock) and sizing without violating mass conservation.
3. **Dynamic Volume Participation Caps**: Institutional orders must never move market prices beyond economic viability. Algorithms must enforce real-time participation rate caps ($\le 15\%$ of bar volume), dynamically blending diurnal historical profiles with online volume filtering.
4. **Adverse Selection & The "Winner's Curse"**: Dark pools offer half-spread price improvement, but risk toxic fill flow where fills only occur immediately before adverse price moves. The router must track post-trade markouts ($P_{t+\Delta t} - P_t$) to dynamically down-weight or quarantine toxic venues.
5. **Closed-Form Multi-Venue Waterfilling (Zero Iterative Solvers)**: Strict adherence to Rule 4.3 bans black-box numerical optimizers (`scipy.optimize`) on the execution hot path. Multi-venue liquidity allocation must be solved algebraically via Karush-Kuhn-Tucker (KKT) waterfilling in $< 0.05\text{ms}$.

---

## 2. Invariant Contracts & Mathematical Formulations

Phase 6 Step 2 is governed by six microstructural invariants:

### Invariant 1: Parent-Child Mass Conservation (`INV-SOR-001`)
- The sum of all spawned child order quantities must strictly equal the parent order target quantity:
  $$\sum_{k=1}^K q_{k, \text{target}} \equiv Q_{\text{parent}}$$
- At all times, cumulative filled child quantities plus open child leaves plus unscheduled parent leaves must satisfy:
  $$\left| \sum_{k=1}^K q_{k, \text{filled}} + \sum_{k=1}^K q_{k, \text{leaves}} + Q_{\text{unscheduled}} - Q_{\text{parent}} \right| < 10^{-7}$$
- Any discrepancy exceeding $10^{-7}$ raises `MassConservationException` (`ERR-SOR-004`).

### Invariant 2: Non-Worse-Than-NBBO & Dark Price Improvement Guarantee (`INV-SOR-002`)
- No child order may be routed at an aggressive price crossing the National Best Bid and Offer (NBBO):
  $$P_{\text{buy}} \le P_{\text{ask}}^{\text{NBBO}}, \quad P_{\text{sell}} \ge P_{\text{bid}}^{\text{NBBO}}$$
- Dark pool child orders must be routed strictly at the NBBO midpoint price:
  $$P_{\text{midpoint}} = \frac{P_{\text{bid}}^{\text{NBBO}} + P_{\text{ask}}^{\text{NBBO}}}{2}$$
  guaranteeing half-spread price improvement $\Delta P_{\text{spread}} = \frac{P_{\text{ask}} - P_{\text{bid}}}{2} > 0$.
- Routing through a locked or crossed market ($P_{\text{bid}}^{\text{NBBO}} \ge P_{\text{ask}}^{\text{NBBO}}$) raises `NBBOViolationException` (`ERR-SOR-003`).

### Invariant 3: Hard Volume Participation Rate Cap (`INV-SOR-003`)
- For all volume-tracking execution algorithms (VWAP, Dynamic Participation), child order slice quantity in interval $k$ must not exceed the institutional volume participation ceiling:
  $$q_k \le \rho_{\max} \cdot \widehat{V}_k, \quad \rho_{\max} \le 0.15 \; (15\%)$$
  where $\widehat{V}_k$ is the combined historical and real-time forecasted bar volume.

### Invariant 4: Anti-Gaming Jitter Non-Negative Slices (`INV-SOR-004`)
- For TWAP execution, anti-gaming randomized jitter $\delta_k \sim \text{Uniform}(-\alpha, \alpha)$ must strictly satisfy:
  $$q_k = \frac{Q_{\text{rem}}}{K - k + 1} (1 + \delta_k) \ge 0.0$$
  with $\sum_{k=1}^K q_k \equiv Q_{\text{parent}}$. Interval durations must follow a Poisson inter-arrival process $\Delta t_k \sim \text{Poisson}(\bar{\tau})$ with $\Delta t_k \ge \Delta t_{\min}$.

### Invariant 5: Causal Implementation Shortfall Attribution (`INV-SOR-005`)
- Total execution shortfall against the arrival benchmark $P_0$ must be causally decomposed into exact additive components:
  $$\text{IS} = \text{Delay Cost} + \text{Price Impact} + \text{Spread Slippage} + \text{Exchange Fees} + \text{Opportunity Cost}$$
  where:
  - $\text{Delay Cost} = (P_{\text{decision}} - P_0) \cdot Q_{\text{parent}}$
  - $\text{Price Impact} = \sum_{k=1}^m q_k \cdot (P_k - P_{\text{decision}})$
  - $\text{Exchange Fees} = \sum_{k=1}^m \text{Fee}_k$
  - $\text{Opportunity Cost} = (P_T - P_0) \cdot (Q_{\text{parent}} - Q_{\text{filled}})$

### Invariant 6: Sub-0.10ms Hot-Path Routing Latency SLA (`INV-SOR-006`)
- Multi-venue liquidity aggregation, dark probing, and closed-form KKT waterfilling allocation must execute in:
  $$\tau_{\text{sor}} \le 0.10\text{ms} \quad (100\mu\text{s})$$
  across $M \le 10$ execution venues.

---

## 3. Architecture & Class Hierarchy

```
src/quant/execution/
├── venues.py         # VenueProfile, VenueType, ConsolidatedQuote, NBBOImbalance
├── algorithms.py     # TWAPScheduler (Poisson), VWAPScheduler (Dynamic Cap), ArrivalPriceScheduler (3/2)
├── sor.py            # SmartOrderRouter with Dark Probing, Markout Watchdog, and KKT Waterfilling
└── parent_order.py   # ParentOrder entity, child order tracking, and Implementation Shortfall TCA
```

### 3.1 Venue Profiles & NBBO Models (`src/quant/execution/venues.py`)

#### `VenueType` (StrEnum)
- `LIT_EXCHANGE`: Displayed order book, maker/taker fee schedule.
- `DARK_POOL`: Non-displayed liquidity, midpoint matching, minimum execution size (MES).

#### `VenueProfile` (Dataclass, `slots=True`)
```python
@dataclass(slots=True)
class VenueProfile:
    venue_id: str
    venue_type: VenueType
    maker_fee_bps: float        # Can be negative for maker rebate
    taker_fee_bps: float
    min_order_size: float = 1.0
    lot_size: float = 1.0
    avg_latency_ms: float = 1.0
    dark_fill_probability: float = 0.35  # For dark pool modeling
```

#### `ConsolidatedQuote` (Dataclass, `slots=True`)
```python
@dataclass(slots=True)
class ConsolidatedQuote:
    symbol: str
    bid_price: float
    bid_quantity: float
    ask_price: float
    ask_quantity: float
    timestamp_ns: int
    venue_depths: dict[str, tuple[float, float]]  # venue_id -> (bid_depth, ask_depth)
```
- Derived properties:
  - `midpoint: float`: $(P_{\text{bid}} + P_{\text{ask}}) / 2$
  - `spread: float`: $P_{\text{ask}} - P_{\text{bid}}$
  - `spread_bps: float`: $(P_{\text{ask}} - P_{\text{bid}}) / P_{\text{midpoint}} \cdot 10^4$
  - `order_book_imbalance: float`: $(V_{\text{bid}} - V_{\text{ask}}) / (V_{\text{bid}} + V_{\text{ask}}) \in [-1.0, 1.0]$

---

### 3.2 Execution Algorithms & Meta-Order Schedulers (`src/quant/execution/algorithms.py`)

#### 1. `PoissonTWAPScheduler`
- Slices parent order into $K$ intervals with randomized Poisson clock timing:
  $$\Delta t_k = \max(\Delta t_{\min}, \; \text{round}(\bar{\tau} \cdot (1 + \xi_k))), \quad \xi_k \sim \text{Uniform}(-\alpha_t, \alpha_t)$$
- Slices quantity with anti-gaming jitter:
  $$q_k = \max(0.0, \; \frac{Q_{\text{rem}}}{K - k + 1} (1 + \delta_k)), \quad \delta_k \sim \text{Uniform}(-\alpha_q, \alpha_q)$$
  with final slice $q_K \equiv Q_{\text{rem}}$ strictly guaranteeing mass conservation (`INV-SOR-001`).

#### 2. `VolumeAdaptiveVWAPScheduler`
- Blends historical volume distribution $w_{\text{hist}, k} \in \Delta^K$ with real-time exponentially smoothed volume $\bar{V}_{\text{realtime}}$:
  $$\widehat{V}_k = \omega V_{\text{hist}, k} + (1 - \omega) \bar{V}_{\text{realtime}}$$
- Applies hard participation rate cap:
  $$q_k = \min\left( \rho_{\max} \cdot \widehat{V}_k, \; Q_{\text{rem}} \cdot \frac{w_k}{\sum_{j=k}^K w_j} \right)$$
  ensuring $\rho \le 15\%$ everywhere (`INV-SOR-003`).

#### 3. `NonlinearArrivalPriceScheduler`
- Closed-form hyperbolic Almgren-Chriss inventory trajectory under non-linear market impact:
  $$x_j = X_0 \cdot \frac{\sinh(\kappa (T - t_j))}{\sinh(\kappa T)}, \quad \kappa = \sqrt{\frac{\lambda \sigma_t^2}{\eta}}$$
- Dynamically scales urgency parameter $\lambda$ with market volatility $\sigma_t$:
  $$\kappa_t = \kappa_0 \cdot \left( \frac{\sigma_t}{\sigma_{\text{baseline}}} \right)$$
  accelerating execution in volatile regimes to reduce timing risk and decelerating in tranquil regimes to minimize price impact.

---

### 3.3 Smart Order Router (SOR) Engine (`src/quant/execution/sor.py`)

#### Class: `SmartOrderRouter`
- **Phase 1: Dark Pool Midpoint Probing**:
  - Directs initial slice to dark pools with Midpoint Pegged IOC orders and Minimum Execution Size (`min_order_size`).
  - Dark allocation is bounded by available leaves: never routes more than $Q_{\text{rem}}$.
  - Captures half-spread price improvement with zero market impact (`INV-SOR-002`).
- **Phase 2: Toxic Markout Watchdog**:
  - Tracks post-fill markout over horizon $\tau_{\text{markout}}$ (e.g. 5 seconds):
    $$\text{Markout} = \text{sign}(\text{side}) \cdot (P_{\text{mid}, t+\tau} - P_{\text{fill}})$$
  - If a dark pool consistently generates negative markouts ($\text{Markout} < -\theta_{\text{toxic}}$), the watchdog flags adverse selection and temporarily routes away from that venue.
- **Phase 3: Closed-Form KKT Lit Waterfilling**:
  - Allocates residual unfilled slice across lit venues $v=1 \dots M$ to minimize total marginal cost:
    $$\min_{q_1, \dots, q_M} \sum_{v=1}^M \left[ \text{taker\_fee}_v \cdot q_v + \frac{\text{spread}_v}{2} q_v + \frac{\lambda_v \sigma}{\sqrt{\text{ADV}_v}} q_v^{3/2} \right]$$
    subject to $\sum_{v=1}^M q_v = Q_{\text{slice}}$, $0 \le q_v \le \text{Depth}_v$.
  - Solved algebraically in $O(M \log M)$ via exact KKT active-set waterfilling in $< 0.03\text{ms}$ (`INV-SOR-006`).

---

### 3.4 Parent Order Coordination & Implementation Shortfall (`src/quant/execution/parent_order.py`)

#### Class: `ParentOrder`
- Coordinates parent order lifecycle, tracking spawned child `Order` instances and cumulative fills.
- Computes real-time **Implementation Shortfall (IS)** TCA attribution (`INV-SOR-005`):
  - Delay Cost, Price Impact, Spread Slippage, Exchange Fees, and Opportunity Cost.

---

## 4. Rule 2: Diagnostic Failure Matrix

| Fault Vector ID | Name | Invariant | Malfunction Symptoms | Root Cause Etiology | Zero-Runtime Verification | Remediation Procedure |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`ERR-SOR-001`** | `ERR_SOR_INVALID_SCHEDULE` | `INV-SOR-004` | `InvalidScheduleException` raised during scheduler construction. | Non-positive intervals ($K \le 0$), duration $\le 0$, or negative slice quantities. | Verify scheduler constructor arguments in configuration. | Ensure $K \ge 1$ and $\text{duration} > 0$. |
| **`ERR-SOR-002`** | `ERR_SOR_INSUFFICIENT_LIQUIDITY` | `INV-SOR-002` | `InsufficientLiquidityException` raised by router. | All configured venues report zero depth or market data disconnect. | Inspect `quote.venue_depths` dictionary for zero sums. | Widen participation rate or enable additional fallback venues. |
| **`ERR-SOR-003`** | `ERR_SOR_NBBO_VIOLATION` | `INV-SOR-002` | `NBBOViolationException` raised before routing. | Consolidated quote has locked or crossed market ($P_{\text{bid}} \ge P_{\text{ask}}$). | Check quote timestamps and bid/ask values. | Wait for next uncrossed quote tick before routing child orders. |
| **`ERR-SOR-004`** | `ERR_SOR_MASS_CONSERVATION_BREACH` | `INV-SOR-001` | `MassConservationException` raised during child order generation. | Sum of child slices deviates from parent quantity by $> 10^{-7}$. | Sum child order quantities and compare to `parent.quantity`. | Enforce final-slice residual closure ($q_K = Q_{\text{rem}}$). |
| **`ERR-SOR-005`** | `ERR_SOR_CHILD_ORDER_FAILED` | `INV-SOR-001` | `ChildOrderFailedException` raised upon gateway rejection. | Gateway rejects child slice (e.g. rate limit, margin, or transport disconnect). | Check underlying `ExecutionReport.text` error reason. | Reroute unfilled quantity through alternative venue. |
| **`ERR-SOR-006`** | `ERR_SOR_ALGORITHM_TIMEOUT` | `INV-SOR-005` | `AlgorithmTimeoutException` raised at horizon expiration. | Parent order execution horizon elapsed with remaining unfilled leaves. | Compare `elapsed_time` against `parent.max_duration_seconds`. | Execute aggressive liquidity clean-up slice or cancel remainder. |

---

## 5. Verification Plan

1. **Automated Unit Tests**:
   - `tests/unit/test_venues.py`: `VenueProfile` validation, `ConsolidatedQuote` NBBO properties, imbalance calculations.
   - `tests/unit/test_execution_algorithms.py`: TWAP Poisson jitter, mass conservation, VWAP participation cap, Almgren-Chriss hyperbolic decay.
   - `tests/unit/test_smart_order_router.py`: Dark pool probing, markout watchdog toxic venue isolation, KKT lit waterfilling, latency SLA.
   - `tests/unit/test_parent_order.py`: Parent-child state synchronization, implementation shortfall TCA attribution.
2. **Quality Gates**:
   - `pytest tests/unit` $\ge 90\%$ statement coverage on new execution files.
   - `mypy src --strict` with zero errors.
   - `ruff check .` and `ruff format --check .` with zero deviations.

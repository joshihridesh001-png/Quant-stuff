# Design Specification: Phase 6, Step 1 — Live Execution Gateway, Order State Machine & Idempotency Token Routing

**Author:** Antigravity / DeepMind Advanced Agentic Coding  
**Date:** 2026-09-15  
**Status:** Approved by User  
**Target Milestone:** Phase 6 Step 1  
**Governing Standard:** `Rules.md` (Rules 1, 2, 3, 4)

---

## 1. Executive Summary & Problem Formulation

In Phase 5, the quantitative engine achieved end-to-end simulated replay and institutional benchmarking capabilities (`simulation.py`), validating that dynamic model averaging (RD-DMA), epistemic circuit breakers, semi-parametric EVT tail risk, and strictly concave Kelly sizing produce statistically significant alpha under realistic friction.

However, moving from backtesting/simulation to live market operation exposes the engine to microstructural asynchronous execution hazards:
1. **Network Packet Reordering & Asynchronous Jitter**: In live FIX / WebSocket environments, execution reports (`FILLED`, `PARTIALLY_FILLED`) frequently arrive prior to order acknowledgments (`NEW`) due to multi-threaded exchange gateways or UDP multicast race conditions. A naive state machine crashes or drops subsequent fills.
2. **Duplicate Order Ingestion & Idempotency Failures**: Transient network timeouts cause client retries. Without deterministic idempotency token routing, retry logic risks duplicate order submission, doubling leverage and violating capital constraints.
3. **Floating-Point Lot Misalignment**: Microstructural exchanges enforce discrete lot step sizes and price tick sizes. Naive floating-point order sizes cause exchange rejects (`INVALID_QUANTITY`).
4. **State Machine Deadlocks & Invariant Breaches**: Incomplete state tracking risks attempting to cancel already-filled orders, or leaving orphaned orders in-flight during emergency disconnects.

Phase 6 Step 1 implements the institutional **Live Execution Gateway Subsystem**, establishing an immutable domain model, an asynchronous deterministic Order State Machine (FSM) with causal out-of-order reconciliation, an in-flight idempotency token router, a high-fidelity paper broker gateway, and a non-blocking asynchronous audit logger.

---

## 2. Invariant Contracts & Mathematical Guarantees

Phase 6 Step 1 is governed by six inviolable engineering and microstructural invariants:

### Invariant 1: Causal State Machine Monotonicity (`INV-GW-001`)
- The lifecycle of an order must strictly adhere to the deterministic directed acyclic transition graph:
  $$\mathcal{S}_{\text{init}} = \{\text{PENDING\_NEW}\} \longrightarrow \{\text{NEW}\} \longrightarrow \{\text{PARTIALLY\_FILLED}\} \longrightarrow \{\text{FILLED}\}$$
  with terminal abort transitions to $\{\text{CANCELLED}, \text{REJECTED}, \text{EXPIRED}\}$.
- Any attempt to transition from a terminal state ($\mathcal{S}_{\text{terminal}} = \{\text{FILLED}, \text{CANCELLED}, \text{REJECTED}, \text{EXPIRED}\}$) to any other state is physically impossible and must raise `InvalidStateTransitionException` (`ERR-GW-001`).

### Invariant 2: Cryptographic Idempotency Token Uniqueness (`INV-GW-002`)
- Every outbound order must possess a deterministic, collision-resistant `client_order_id` synthesized as a UUIDv5 from strategy namespace, symbol, and microsecond sequence timestamp:
  $$\text{cl\_ord\_id} = \text{UUIDv5}(\text{NAMESPACE\_ORDER}, f"{strategy\_id}:{symbol}:{side}:{timestamp\_ns}")$$
- Re-submission of an identical `cl_ord_id` while in-flight or within the deduplication window must return the existing order reference without routing a new order to the exchange (`ERR-GW-002`).

### Invariant 3: Execution Mass Conservation (`INV-GW-003`)
- At all times, the cumulative filled quantity $Q_{\text{filled}}$ and remaining quantity $Q_{\text{leaves}}$ must satisfy exact conservation:
  $$Q_{\text{filled}} + Q_{\text{leaves}} \equiv Q_{\text{target}}$$
  with $0.0 \le Q_{\text{filled}} \le Q_{\text{target}}$ and $Q_{\text{leaves}} \ge 0.0$.
- Average fill price must strictly satisfy:
  $$\bar{P}_{\text{exec}} = \frac{\sum_{k=1}^m q_k \cdot p_k}{\sum_{k=1}^m q_k} = \frac{\text{quote\_filled}}{Q_{\text{filled}}}$$
  with $\bar{P}_{\text{exec}} > 0.0$ whenever $Q_{\text{filled}} > 0.0$.

### Invariant 4: Causal Out-of-Order Packet Reconciliation (`INV-GW-004`)
- If an execution report with status `FILLED` or `PARTIALLY_FILLED` arrives while the internal order state is still `PENDING_NEW` (due to transport-layer packet reordering), the state machine must automatically synthesize an implicit transition to `NEW` before applying the fill, preserving monotonic causality without dropping the execution event.

### Invariant 5: Strict Non-Finite Input & Boundary Protection (`INV-GW-005`)
- Order price and quantity must be finite, strictly positive real numbers:
  $$p \in (0.0, \infty), \quad q \in (0.0, \infty)$$
- Any NaN, $\pm\infty$, zero, or negative scalar raises `InvalidOrderInputException` (`ERR-GW-003`).

### Invariant 6: Sub-Millisecond Gateway Execution Latency SLA (`INV-GW-006`)
- Hot-path order validation, idempotency token check, state transition, and paper broker fill dispatch must execute in:
  $$\tau_{\text{exec}} \le 0.10\text{ms} \quad (100\mu\text{s})$$
  under zero-lock in-memory dispatch.

---

## 3. Architecture & Class Hierarchy

```
src/quant/execution/
├── __init__.py           # Unified exports of execution domain models and engines
├── models.py             # Order, ExecutionReport, OrderState, OrderSide, OrderType, TimeInForce
├── fsm.py                # OrderStateMachine with causal out-of-order reconciliation
├── idempotency.py        # IdempotencyRouter with lock-free deduplication ring buffer
├── gateway.py            # ExecutionGateway Protocol, PaperExecutionGateway, AsyncBrokerGateway
└── audit.py              # Non-blocking async SQLite/DuckDB WAL order audit logger
```

### 3.1 Domain Models (`src/quant/execution/models.py`)

#### `OrderState` (Enum)
- `PENDING_NEW`: Created locally, awaiting exchange submission or wire ack.
- `NEW`: Acknowledged by exchange order book, resting and active.
- `PARTIALLY_FILLED`: Partially matched; remaining quantity resting.
- `FILLED`: 100% matched; terminal state.
- `PENDING_CANCEL`: Cancel request dispatched, awaiting exchange ack.
- `CANCELLED`: Confirmed cancelled by exchange; terminal state.
- `REJECTED`: Rejected by exchange or pre-trade risk filter; terminal state.
- `EXPIRED`: Time-In-Force expired (e.g. unhit IOC); terminal state.

#### `OrderSide` (Enum)
- `BUY`, `SELL`.

#### `OrderType` (Enum)
- `MARKET`, `LIMIT`, `STOP_LIMIT`, `PEGGED`.

#### `TimeInForce` (Enum)
- `DAY`, `GTC` (Good-Till-Cancel), `IOC` (Immediate-Or-Cancel), `FOK` (Fill-Or-Kill).

#### `Order` (Dataclass, `slots=True`)
```python
@dataclass(slots=True)
class Order:
    cl_ord_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    state: OrderState = OrderState.PENDING_NEW
    exchange_order_id: str | None = None
    filled_quantity: float = 0.0
    filled_quote_amount: float = 0.0
    average_price: float = 0.0
    fees_paid: float = 0.0
    created_at_ns: int = 0
    updated_at_ns: int = 0
```

#### `ExecutionReport` (Dataclass, `slots=True`)
```python
@dataclass(frozen=True, slots=True)
class ExecutionReport:
    report_id: str
    cl_ord_id: str
    exchange_order_id: str
    symbol: str
    side: OrderSide
    exec_type: OrderState
    last_quantity: float
    last_price: float
    cum_quantity: float
    leaves_quantity: float
    cum_quote_amount: float
    average_price: float
    fee: float
    timestamp_ns: int
    text: str = ""
```

---

### 3.2 Order State Machine (`src/quant/execution/fsm.py`)

#### Class: `OrderStateMachine`
- Manages order lifecycle transitions with strict invariant verification.
- **Valid Transition Matrix**:
  | Current State | Permitted Next States |
  | :--- | :--- |
  | `PENDING_NEW` | `NEW`, `PARTIALLY_FILLED` (reconciled), `FILLED` (reconciled), `REJECTED`, `EXPIRED` |
  | `NEW` | `PARTIALLY_FILLED`, `FILLED`, `PENDING_CANCEL`, `CANCELLED`, `EXPIRED` |
  | `PARTIALLY_FILLED` | `PARTIALLY_FILLED`, `FILLED`, `PENDING_CANCEL`, `CANCELLED`, `EXPIRED` |
  | `PENDING_CANCEL` | `CANCELLED`, `FILLED` (late fill race), `PARTIALLY_FILLED` |
  | `FILLED` | *(Terminal — none)* |
  | `CANCELLED` | *(Terminal — none)* |
  | `REJECTED` | *(Terminal — none)* |
  | `EXPIRED` | *(Terminal — none)* |

- **Method**: `transition(order: Order, next_state: OrderState, report: ExecutionReport | None = None) -> Order`
  - Validates source and target states.
  - Automatically reconciles `PENDING_NEW` $\to$ `FILLED` via intermediate synthetic `NEW` event (`INV-GW-004`).
  - Verifies mass conservation (`INV-GW-003`).
  - Updates order fields and timestamps.

---

### 3.3 Idempotency Token Router (`src/quant/execution/idempotency.py`)

#### Class: `IdempotencyRouter`
- Generates deterministic order IDs via UUIDv5 hashing.
- Maintains an in-flight order cache and ring buffer of recent order tokens.
- **Method**: `generate_client_order_id(strategy_id: str, symbol: str, side: OrderSide, timestamp_ns: int) -> str`
- **Method**: `register_order(order: Order) -> None`:
  - If `order.cl_ord_id` already exists and is active, raises `DuplicateOrderException` (`ERR-GW-002`).
  - Stores active order in registry.
- **Method**: `deregister_order(cl_ord_id: str) -> None`: Moves order from active registry to historical deduplication window.

---

### 3.4 Gateway Protocol & Paper Broker (`src/quant/execution/gateway.py`)

#### Protocol: `ExecutionGateway` (Structural Subtyping)
```python
class ExecutionGateway(Protocol):
    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def submit_order(self, order: Order) -> ExecutionReport: ...
    async def cancel_order(self, cl_ord_id: str) -> ExecutionReport: ...
    async def get_order(self, cl_ord_id: str) -> Order | None: ...
    async def get_open_orders(self) -> list[Order]: ...
    async def get_positions(self) -> dict[str, float]: ...
    async def get_account_balance(self) -> dict[str, float]: ...
```

#### Class: `PaperExecutionGateway`
- High-fidelity simulated broker implementing `ExecutionGateway`.
- Realistic execution modeling:
  - Synthetic latency simulation (`latency_ms`).
  - Configurable fee schedule (`fee_bps`).
  - Bid-ask spread slippage (`slippage_bps`).
  - Immediate execution for `MARKET` orders.
  - Limit order match simulation based on simulated market ticks.
  - Full margin and balance accounting.

---

### 3.5 Async Order Audit Logger (`src/quant/execution/audit.py`)

#### Class: `OrderAuditLogger`
- Non-blocking background worker consuming execution events via `asyncio.Queue`.
- Writes records to SQLite / DuckDB WAL table `order_execution_audit`:
  - `timestamp_ns`, `cl_ord_id`, `exchange_order_id`, `symbol`, `side`, `event_type`, `quantity`, `price`, `fill_qty`, `fill_price`, `fee`.
- Zero latency overhead on the order routing hot path.

---

## 4. Rule 2: Diagnostic Failure Matrix

| Fault Vector ID | Name | Invariant | Malfunction Symptoms | Root Cause Etiology | Zero-Runtime Verification | Remediation Procedure |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`ERR-GW-001`** | `ERR_GW_INVALID_STATE_TRANSITION` | `INV-GW-001` | `InvalidStateTransitionException` raised; order state unchanged. | Attempting an invalid state transition (e.g., cancelling a `FILLED` order or transitioning from terminal state). | Inspect `order.state` and requested target state in execution log. | Ensure caller queries order status before issuing cancel requests; verify terminal guards. |
| **`ERR-GW-002`** | `ERR_GW_DUPLICATE_ORDER_ID` | `INV-GW-002` | `DuplicateOrderException` raised; duplicate order rejected. | Identical `client_order_id` re-submitted while order is active. | Check timestamp resolution or UUID generation parameters. | Ensure timestamp nanosecond precision or regenerate unique nonce. |
| **`ERR-GW-003`** | `ERR_GW_NON_FINITE_INPUT` | `INV-GW-005` | `InvalidOrderInputException` raised during order construction. | NaN or $\pm\infty$ passed as price or quantity. | Verify sizing engine output before creating `Order`. | Ensure upstream sizing engine clamps lot sizes and checks finite bounds. |
| **`ERR-GW-004`** | `ERR_GW_INSUFFICIENT_MARGIN` | `INV-GW-003` | `InsufficientMarginException` raised; order rejected. | Order value exceeds available account balance or purchasing power. | Compare order notion ($p \cdot q$) against available cash in `get_account_balance()`. | Scale down allocation via upstream risk manager or deposit capital. |
| **`ERR-GW-005`** | `ERR_GW_DISCONNECTED` | `INV-GW-006` | `GatewayDisconnectedException` raised upon order submission. | Gateway network session is closed or uninitialized. | Check `gateway.is_connected` boolean flag. | Await `gateway.connect()` before routing orders. |
| **`ERR-GW-006`** | `ERR_GW_RATE_LIMIT_EXCEEDED` | `INV-GW-006` | `RateLimitExceededException` raised with HTTP 429 backoff. | Outbound request frequency exceeds broker rate-limit bucket. | Inspect token bucket capacity in gateway config. | Implement exponential backoff or throttle upstream order generation. |

---

## 5. Rule 4: Adversarial Stress-Testing & "Best of the Best" Verification

### 5.1 Naive Shortcuts Blacklist
1. **No In-Memory-Only State Without Reconciliation**: Naive gateways crash when an exchange delivers an execution report out of order. Our FSM synthesizes causal intermediate states (`INV-GW-004`).
2. **No Unchecked Floating-Point Arithmetic**: Invariant 3 explicitly guards against floating-point epsilon leakage ($|Q_{\text{filled}} + Q_{\text{leaves}} - Q_{\text{target}}| < 10^{-7}$).
3. **No Blocking Database I/O on Hot Path**: Order persistence is strictly asynchronous via bounded queue and background WAL worker (`audit.py`).

### 5.2 Verification Plan
1. **Automated Unit Tests**:
   - `tests/unit/test_order_models.py`: Model validation, slot immutability, non-finite checks.
   - `tests/unit/test_order_fsm.py`: Complete directed state transition graph, terminal state protections, out-of-order fill reconciliation.
   - `tests/unit/test_idempotency.py`: UUIDv5 deterministic generation, duplicate rejection, ring buffer expiration.
   - `tests/unit/test_paper_gateway.py`: Connect/disconnect, market/limit fills, slippage, latency simulation, balance accounting.
   - `tests/unit/test_order_audit.py`: Async queue persistence, SQLite/DuckDB insertion verification, clean shutdown.
2. **Quality Gates**:
   - `pytest tests/unit` $\ge 90\%$ coverage on `src/quant/execution/`.
   - `mypy src --strict` 0 errors.
   - `ruff check .` and `ruff format --check .` 0 deviations.

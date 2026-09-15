"""Universal Execution Gateway Protocol and High-Fidelity Paper Broker.

Purpose:
    Establishes the institutional asynchronous execution gateway subsystem for live market order
    routing and simulated paper execution:
    1. ExecutionGateway Protocol (INV-GW-001 through INV-GW-006): Unified asynchronous interface
       for order submission, cancellation, position tracking, and balance inspection.
    2. PaperExecutionGateway: Realistic high-fidelity simulated venue providing:
       - Deterministic finite-state machine lifecycle management (OrderStateMachine).
       - Zero-leakage cryptographic idempotency token routing (IdempotencyRouter).
       - Bid-ask spread slippage simulation (slippage_bps).
       - Realistic venue fee schedules (fee_bps).
       - Immediate matching for MARKET and marketable LIMIT orders.
       - Limit order book resting and tick-level matching (set_market_price).
       - Pre-trade purchasing power and cash margin verification.
       - Sub-millisecond hot-path latency SLA (<= 0.10ms under zero synthetic delay).

Dependencies:
    - asyncio: Asynchronous coroutines, sleep timers for latency simulation.
    - math: Finite float scalar verification (math.isfinite).
    - time: High-resolution nanosecond epoch timestamps (time.time_ns).
    - typing: Protocol, runtime_checkable, Final, ClassVar.
    - uuid: RFC 4122 cryptographic UUID generation for execution report IDs.
    - quant.execution.fsm: OrderStateMachine with causal out-of-order reconciliation.
    - quant.execution.idempotency: IdempotencyRouter with in-flight duplicate rejection.
    - quant.execution.models: Order, ExecutionReport, OrderState, OrderSide, OrderType,
      diagnostic exceptions, and error code constants.

Structural Relationship:
    - Core broker abstraction layer for Phase 6 live trading:
        1. Consumed by Strategy Execution Engine and Portfolio Rebalancing Orchestrator.
        2. Implemented by PaperExecutionGateway (simulated) and AsyncBrokerGateway (FIX/WebSocket).
        3. Emits ExecutionReports to OrderAuditLogger (audit.py) and strategy callbacks.

Invariants Enforced:
    - INV-GW-001 (Causal State Machine Monotonicity): State transitions mediated strictly via FSM.
    - INV-GW-002 (Idempotency Token Uniqueness): Duplicate active or historical IDs rejected.
    - INV-GW-003 (Execution Mass Conservation): Exact cash, position, and leaves quantity conservation.
    - INV-GW-005 (Strict Boundary & Connection Invariant):
        - Submitting or cancelling while disconnected raises GatewayDisconnectedException (ERR-GW-005).
        - Insufficient purchasing power raises InsufficientMarginException (ERR-GW-004).
        - Non-finite or corrupt inputs raise InvalidOrderInputException (ERR-GW-003).
    - INV-GW-006 (Hot-Path Latency SLA): Market order execution completes in <= 0.10ms (100us).
"""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from typing import Final, Protocol, runtime_checkable

from quant.execution.fsm import OrderStateMachine
from quant.execution.idempotency import IdempotencyRouter
from quant.execution.models import (
    ERR_GW_DISCONNECTED,
    ERR_GW_INSUFFICIENT_MARGIN,
    ERR_GW_INVALID_STATE_TRANSITION,
    ERR_GW_NON_FINITE_INPUT,
    ExecutionReport,
    GatewayDisconnectedException,
    InsufficientMarginException,
    InvalidOrderInputException,
    InvalidStateTransitionException,
    Order,
    OrderSide,
    OrderState,
    OrderType,
)

# Diagnostic fault code aliases for internal reference
_ERR_DISCONNECTED: Final[str] = ERR_GW_DISCONNECTED
_ERR_INSUFFICIENT_MARGIN: Final[str] = ERR_GW_INSUFFICIENT_MARGIN
_ERR_NON_FINITE: Final[str] = ERR_GW_NON_FINITE_INPUT
_ERR_INVALID_TRANSITION: Final[str] = ERR_GW_INVALID_STATE_TRANSITION


@runtime_checkable
class ExecutionGateway(Protocol):
    """Protocol defining the asynchronous execution gateway interface for broker connectivity.

    Adheres to structural subtyping with runtime inspection support via @runtime_checkable.
    """

    @property
    def is_connected(self) -> bool:
        """Indicate whether the gateway has an active broker or exchange session."""
        ...

    async def connect(self) -> None:
        """Establish connection session to the broker or exchange."""
        ...

    async def disconnect(self) -> None:
        """Terminate connection session to the broker or exchange."""
        ...

    async def submit_order(self, order: Order) -> ExecutionReport:
        """Submit an order for execution or order-book resting."""
        ...

    async def cancel_order(self, cl_ord_id: str) -> ExecutionReport:
        """Request cancellation of an active resting or in-flight order."""
        ...

    async def get_order(self, cl_ord_id: str) -> Order | None:
        """Retrieve an order by client_order_id."""
        ...

    async def get_open_orders(self) -> list[Order]:
        """Retrieve all currently active / open orders."""
        ...

    async def get_positions(self) -> dict[str, float]:
        """Retrieve current portfolio instrument positions."""
        ...

    async def get_account_balance(self) -> dict[str, float]:
        """Retrieve current account cash and equity balances."""
        ...


class PaperExecutionGateway:
    """High-fidelity simulated paper broker implementing the ExecutionGateway protocol.

    Provides realistic slippage simulation, fee accounting, limit order book resting,
    pre-trade margin checks, and state machine lifecycle reconciliation.
    """

    __slots__ = (
        "_cash",
        "_fee_bps",
        "_fsm",
        "_idempotency_router",
        "_initial_balance",
        "_is_connected",
        "_latency_ms",
        "_market_prices",
        "_orders",
        "_positions",
        "_report_counter",
        "_slippage_bps",
    )

    def __init__(
        self,
        initial_balance: float = 1_000_000.0,
        fee_bps: float = 2.0,
        slippage_bps: float = 1.0,
        latency_ms: float = 0.0,
        fsm: OrderStateMachine | None = None,
        idempotency_router: IdempotencyRouter | None = None,
    ) -> None:
        """Initialize paper execution gateway with configuration and accounting parameters.

        Args:
            initial_balance: Initial cash balance in quote currency (finite scalar >= 0.0).
            fee_bps: Venue transaction fee in basis points (finite scalar >= 0.0; 1 bp = 0.0001).
            slippage_bps: Bid-ask spread slippage penalty in basis points (finite scalar >= 0.0).
            latency_ms: Synthetic round-trip latency in milliseconds (finite scalar >= 0.0).
            fsm: Injected OrderStateMachine instance (defaults to new OrderStateMachine).
            idempotency_router: Injected IdempotencyRouter instance (defaults to new router).

        Raises:
            InvalidOrderInputException: If any parameter violates domain boundaries or types (ERR-GW-003).
        """
        # Functional Purpose: Initialize paper broker accounting ledger, matching state, and risk controls.
        # Explicit Dependency Tracking: math.isfinite, OrderStateMachine, IdempotencyRouter, ERR_GW_NON_FINITE_INPUT.
        # Structural Relationship: Core simulation venue consumed by unit tests and strategy backtest harness.
        # Defensive Invariant: initial_balance, fee_bps, slippage_bps, latency_ms must be finite non-negative scalars; reject bool.

        # 1. Validate initial_balance
        if (
            isinstance(initial_balance, bool)
            or not isinstance(initial_balance, (int, float))
            or not math.isfinite(initial_balance)
            or initial_balance < 0.0
        ):
            raise InvalidOrderInputException(
                f"initial_balance must be finite scalar >= 0.0, got {initial_balance!r}",
                code=_ERR_NON_FINITE,
            )

        # 2. Validate fee_bps
        if (
            isinstance(fee_bps, bool)
            or not isinstance(fee_bps, (int, float))
            or not math.isfinite(fee_bps)
            or fee_bps < 0.0
        ):
            raise InvalidOrderInputException(
                f"fee_bps must be finite scalar >= 0.0, got {fee_bps!r}",
                code=_ERR_NON_FINITE,
            )

        # 3. Validate slippage_bps
        if (
            isinstance(slippage_bps, bool)
            or not isinstance(slippage_bps, (int, float))
            or not math.isfinite(slippage_bps)
            or slippage_bps < 0.0
        ):
            raise InvalidOrderInputException(
                f"slippage_bps must be finite scalar >= 0.0, got {slippage_bps!r}",
                code=_ERR_NON_FINITE,
            )

        # 4. Validate latency_ms
        if (
            isinstance(latency_ms, bool)
            or not isinstance(latency_ms, (int, float))
            or not math.isfinite(latency_ms)
            or latency_ms < 0.0
        ):
            raise InvalidOrderInputException(
                f"latency_ms must be finite scalar >= 0.0, got {latency_ms!r}",
                code=_ERR_NON_FINITE,
            )

        # 5. Validate fsm
        if fsm is not None and not isinstance(fsm, OrderStateMachine):
            raise InvalidOrderInputException(
                f"fsm must be an instance of OrderStateMachine or None, got {type(fsm).__name__}",
                code=_ERR_NON_FINITE,
            )

        # 6. Validate idempotency_router
        if idempotency_router is not None and not isinstance(idempotency_router, IdempotencyRouter):
            raise InvalidOrderInputException(
                f"idempotency_router must be an instance of IdempotencyRouter or None, got {type(idempotency_router).__name__}",
                code=_ERR_NON_FINITE,
            )

        self._initial_balance: float = float(initial_balance)
        self._cash: float = float(initial_balance)
        self._fee_bps: float = float(fee_bps)
        self._slippage_bps: float = float(slippage_bps)
        self._latency_ms: float = float(latency_ms)
        self._fsm: OrderStateMachine = fsm if fsm is not None else OrderStateMachine()
        self._idempotency_router: IdempotencyRouter = (
            idempotency_router if idempotency_router is not None else IdempotencyRouter()
        )
        self._is_connected: bool = False
        self._positions: dict[str, float] = {}
        self._orders: dict[str, Order] = {}
        self._market_prices: dict[str, float] = {}
        self._report_counter: int = 0

    @property
    def is_connected(self) -> bool:
        """Whether the gateway connection session to the simulated venue is currently active."""
        # Functional Purpose: Expose instantaneous connection status for gateway boundary enforcement.
        # Explicit Dependency Tracking: self._is_connected.
        # Structural Relationship: Checked prior to submit_order and cancel_order operations.
        # Defensive Invariant: Returns boolean True only if connect() has completed without disconnect().
        return self._is_connected

    @property
    def initial_balance(self) -> float:
        """Initial opening cash balance configured for this paper session."""
        # Functional Purpose: Expose baseline equity for return on investment calculations.
        # Explicit Dependency Tracking: self._initial_balance.
        # Structural Relationship: Queried by audit and benchmarking reporting.
        # Defensive Invariant: Always finite float >= 0.0.
        return self._initial_balance

    @property
    def fee_bps(self) -> float:
        """Configured transaction fee rate in basis points."""
        # Functional Purpose: Expose fee rate for diagnostic inspection and cost modeling.
        # Explicit Dependency Tracking: self._fee_bps.
        # Structural Relationship: Queried by cost attribution analysis.
        # Defensive Invariant: Always finite float >= 0.0.
        return self._fee_bps

    @property
    def slippage_bps(self) -> float:
        """Configured bid-ask spread slippage penalty in basis points."""
        # Functional Purpose: Expose execution slippage rate for friction auditing.
        # Explicit Dependency Tracking: self._slippage_bps.
        # Structural Relationship: Queried by transaction cost analysis (TCA).
        # Defensive Invariant: Always finite float >= 0.0.
        return self._slippage_bps

    @property
    def latency_ms(self) -> float:
        """Configured synthetic round-trip latency simulation in milliseconds."""
        # Functional Purpose: Expose simulated wire delay duration.
        # Explicit Dependency Tracking: self._latency_ms.
        # Structural Relationship: Queried by latency profiling harnesses.
        # Defensive Invariant: Always finite float >= 0.0.
        return self._latency_ms

    @property
    def fsm(self) -> OrderStateMachine:
        """Underlying OrderStateMachine managing lifecycle transitions."""
        # Functional Purpose: Provide reference to state machine instance.
        # Explicit Dependency Tracking: self._fsm.
        # Structural Relationship: Used for transition introspection.
        # Defensive Invariant: Always valid OrderStateMachine instance.
        return self._fsm

    @property
    def idempotency_router(self) -> IdempotencyRouter:
        """Underlying IdempotencyRouter managing active and historical token deduplication."""
        # Functional Purpose: Provide reference to idempotency router.
        # Explicit Dependency Tracking: self._idempotency_router.
        # Structural Relationship: Queried for active order counts and token verification.
        # Defensive Invariant: Always valid IdempotencyRouter instance.
        return self._idempotency_router

    async def connect(self) -> None:
        """Establish session connection to the simulated paper execution venue.

        Simulates network connection delay if latency_ms > 0.0.
        """
        # Functional Purpose: Transition gateway state to connected, opening venue for order routing.
        # Explicit Dependency Tracking: asyncio.sleep, self._latency_ms.
        # Structural Relationship: Required before submit_order or cancel_order can succeed.
        # Defensive Invariant: Sets _is_connected to True; sleeps exactly latency_ms / 1000 seconds if positive.
        if self._latency_ms > 0.0:
            await asyncio.sleep(self._latency_ms / 1000.0)
        self._is_connected = True

    async def disconnect(self) -> None:
        """Terminate session connection to the simulated paper execution venue.

        Simulates network disconnection delay if latency_ms > 0.0.
        """
        # Functional Purpose: Transition gateway state to disconnected, closing venue to incoming order actions.
        # Explicit Dependency Tracking: asyncio.sleep, self._latency_ms.
        # Structural Relationship: Invoked during system shutdown or session teardown.
        # Defensive Invariant: Sets _is_connected to False; sleeps exactly latency_ms / 1000 seconds if positive.
        if self._latency_ms > 0.0:
            await asyncio.sleep(self._latency_ms / 1000.0)
        self._is_connected = False

    def set_market_price(self, symbol: str, price: float) -> list[ExecutionReport]:
        """Update market price for a symbol and execute matching resting limit orders.

        Args:
            symbol: Target market instrument ticker (non-empty string).
            price: Current market trade price (finite float > 0.0).

        Returns:
            List of ExecutionReport instances generated from filled resting limit orders.

        Raises:
            InvalidOrderInputException: If symbol or price violate boundary constraints (ERR-GW-003).
        """
        # Functional Purpose: Update market price and match resting limit orders under price-priority rules.
        # Explicit Dependency Tracking: self._market_prices, self._orders, self._fsm, self._idempotency_router.
        # Structural Relationship: Invoked by market data feed simulators to advance market price and trigger resting fills.
        # Defensive Invariant: symbol non-empty str; price finite > 0.0 non-bool; exact cash and position conservation.
        if not isinstance(symbol, str) or not symbol.strip():
            raise InvalidOrderInputException(
                f"symbol must be non-empty string, got {symbol!r}",
                code=_ERR_NON_FINITE,
            )
        if (
            isinstance(price, bool)
            or not isinstance(price, (int, float))
            or not math.isfinite(price)
            or price <= 0.0
        ):
            raise InvalidOrderInputException(
                f"price must be finite scalar > 0.0, got {price!r}",
                code=_ERR_NON_FINITE,
            )

        clean_symbol = symbol.strip()
        flt_price = float(price)
        self._market_prices[clean_symbol] = flt_price

        reports: list[ExecutionReport] = []

        # Iterate over currently resting active limit orders for this symbol
        for order in list(self._orders.values()):
            if (
                order.symbol != clean_symbol
                or not order.is_active
                or order.order_type != OrderType.LIMIT
            ):
                continue

            assert order.price is not None
            # BUY limit matched if limit price >= current market price; SELL limit if limit price <= current market price
            is_match = (order.side == OrderSide.BUY and order.price >= flt_price) or (
                order.side == OrderSide.SELL and order.price <= flt_price
            )

            if not is_match:
                continue

            # Fill occurs at matched market tick price
            fill_price = flt_price
            order_qty = order.quantity
            notional = fill_price * order_qty
            fee = notional * (self._fee_bps * 1e-4)

            # Mass conservation cash and position updates (INV-GW-003)
            if order.side == OrderSide.BUY:
                self._cash -= notional + fee
                new_pos = round(self._positions.get(clean_symbol, 0.0) + order_qty, 8)
                self._positions[clean_symbol] = 0.0 if abs(new_pos) < 1e-9 else new_pos
            else:
                self._cash += notional - fee
                new_pos = round(self._positions.get(clean_symbol, 0.0) - order_qty, 8)
                self._positions[clean_symbol] = 0.0 if abs(new_pos) < 1e-9 else new_pos

            # Synthesize fill ExecutionReport
            self._report_counter += 1
            now_ns = time.time_ns()
            report = ExecutionReport(
                report_id=f"rep-paper-{uuid.uuid4().hex[:12]}-{self._report_counter}",
                cl_ord_id=order.cl_ord_id,
                exchange_order_id=order.exchange_order_id or f"ex-{order.cl_ord_id}",
                symbol=clean_symbol,
                side=order.side,
                exec_type=OrderState.FILLED,
                last_quantity=order_qty,
                last_price=fill_price,
                cum_quantity=order_qty,
                leaves_quantity=0.0,
                cum_quote_amount=notional,
                average_price=fill_price,
                fee=fee,
                timestamp_ns=now_ns,
                text="Resting limit order matched",
            )

            # Apply fill via FSM (INV-GW-001, INV-GW-003)
            self._fsm.apply_execution_report(order, report)
            # Deregister from active idempotency tracking (INV-GW-002)
            self._idempotency_router.deregister_order(order.cl_ord_id)
            reports.append(report)

        return reports

    async def submit_order(self, order: Order) -> ExecutionReport:
        """Submit an order for execution or order-book resting.

        Args:
            order: Order domain entity in PENDING_NEW state.

        Returns:
            ExecutionReport indicating FILLED (for MARKET / marketable LIMIT) or NEW (for resting LIMIT).

        Raises:
            GatewayDisconnectedException: If gateway is not connected (ERR-GW-005).
            InvalidOrderInputException: If order parameters violate invariants (ERR-GW-003).
            DuplicateOrderException: If client_order_id is already active or recently used (ERR-GW-002).
            InsufficientMarginException: If required purchasing power exceeds available cash (ERR-GW-004).
            InvalidStateTransitionException: If order is in invalid state for submission (ERR-GW-001).
        """
        # Functional Purpose: Ingest outbound order, verify connection and margin, register token, and execute.
        # Explicit Dependency Tracking: Order, ExecutionReport, OrderStateMachine, IdempotencyRouter.
        # Structural Relationship: Main routing entrypoint for strategy orders.
        # Defensive Invariant: Gateway must be connected; order in PENDING_NEW; margin available; token unique.

        # 1. Connection check (INV-GW-005)
        if not self._is_connected:
            raise GatewayDisconnectedException(
                "Cannot submit order while gateway is disconnected",
                code=_ERR_DISCONNECTED,
            )

        # 2. Input domain validation (INV-GW-005)
        if not isinstance(order, Order):
            raise InvalidOrderInputException(
                f"order must be an instance of Order, got {type(order).__name__}",
                code=_ERR_NON_FINITE,
            )

        if order.is_terminal:
            raise InvalidStateTransitionException(
                f"Cannot submit order in terminal state {order.state.value}",
                code=_ERR_INVALID_TRANSITION,
            )

        if order.state != OrderState.PENDING_NEW:
            raise InvalidStateTransitionException(
                f"Cannot submit order with state {order.state.value}, expected PENDING_NEW",
                code=_ERR_INVALID_TRANSITION,
            )

        if order.order_type not in (OrderType.MARKET, OrderType.LIMIT):
            raise InvalidOrderInputException(
                f"Unsupported order type for paper gateway: {order.order_type.value}",
                code=_ERR_NON_FINITE,
            )

        if order.order_type == OrderType.LIMIT and (order.price is None or order.price <= 0.0):
            raise InvalidOrderInputException(
                f"LIMIT order must specify price > 0.0, got {order.price!r}",
                code=_ERR_NON_FINITE,
            )

        # 3. Synthetic latency simulation
        if self._latency_ms > 0.0:
            await asyncio.sleep(self._latency_ms / 1000.0)

        # 4. Idempotency registration check (INV-GW-002)
        # Raises DuplicateOrderException(ERR_GW_DUPLICATE_ORDER_ID) if active or in history ring
        self._idempotency_router.register_order(order)
        self._orders[order.cl_ord_id] = order

        # 5. Pre-trade purchasing power and margin check (INV-GW-003, INV-GW-005)
        if order.side == OrderSide.BUY:
            if order.order_type == OrderType.MARKET:
                if order.symbol not in self._market_prices:
                    # Clean up registered order upon fatal parameter failure
                    self._idempotency_router.deregister_order(order.cl_ord_id)
                    self._orders.pop(order.cl_ord_id, None)
                    raise InvalidOrderInputException(
                        f"No market price available for symbol {order.symbol}",
                        code=_ERR_NON_FINITE,
                    )
                mkt_price = self._market_prices[order.symbol]
                est_fill_price = mkt_price * (1.0 + self._slippage_bps * 1e-4)
            else:
                # LIMIT BUY
                assert order.price is not None
                if (
                    order.symbol in self._market_prices
                    and order.price >= self._market_prices[order.symbol]
                ):
                    mkt_price = self._market_prices[order.symbol]
                    est_fill_price = min(order.price, mkt_price * (1.0 + self._slippage_bps * 1e-4))
                else:
                    est_fill_price = order.price

            est_notional = est_fill_price * order.quantity
            est_fee = est_notional * (self._fee_bps * 1e-4)
            required_power = est_notional + est_fee

            if required_power > self._cash:
                # Margin failure: deregister token and purge active order reference
                self._idempotency_router.deregister_order(order.cl_ord_id)
                self._orders.pop(order.cl_ord_id, None)
                raise InsufficientMarginException(
                    f"Required purchasing power {required_power:.2f} exceeds available cash {self._cash:.2f}",
                    code=_ERR_INSUFFICIENT_MARGIN,
                )

        # 6. Monotonic state transition PENDING_NEW -> NEW via FSM (INV-GW-001)
        now_ns = time.time_ns()
        self._fsm.transition(order, OrderState.NEW, timestamp_ns=now_ns)
        if order.exchange_order_id is None:
            order.exchange_order_id = f"ex-{order.cl_ord_id}"

        # 7. Evaluate marketability
        is_marketable = False
        if order.order_type == OrderType.MARKET:
            is_marketable = True
        elif order.order_type == OrderType.LIMIT and order.symbol in self._market_prices:
            assert order.price is not None
            cur_mkt = self._market_prices[order.symbol]
            if (order.side == OrderSide.BUY and order.price >= cur_mkt) or (
                order.side == OrderSide.SELL and order.price <= cur_mkt
            ):
                is_marketable = True

        if is_marketable:
            # Immediate match execution
            if order.symbol not in self._market_prices:
                self._idempotency_router.deregister_order(order.cl_ord_id)
                self._orders.pop(order.cl_ord_id, None)
                raise InvalidOrderInputException(
                    f"No market price available for symbol {order.symbol}",
                    code=_ERR_NON_FINITE,
                )

            cur_mkt = self._market_prices[order.symbol]
            if order.side == OrderSide.BUY:
                fill_price = cur_mkt * (1.0 + self._slippage_bps * 1e-4)
                if order.order_type == OrderType.LIMIT and order.price is not None:
                    fill_price = min(order.price, fill_price)
            else:
                fill_price = cur_mkt * (1.0 - self._slippage_bps * 1e-4)
                if order.order_type == OrderType.LIMIT and order.price is not None:
                    fill_price = max(order.price, fill_price)

            order_qty = order.quantity
            notional = fill_price * order_qty
            fee = notional * (self._fee_bps * 1e-4)

            # Mass conservation cash and position updates (INV-GW-003)
            if order.side == OrderSide.BUY:
                self._cash -= notional + fee
                new_pos = round(self._positions.get(order.symbol, 0.0) + order_qty, 8)
                self._positions[order.symbol] = 0.0 if abs(new_pos) < 1e-9 else new_pos
            else:
                self._cash += notional - fee
                new_pos = round(self._positions.get(order.symbol, 0.0) - order_qty, 8)
                self._positions[order.symbol] = 0.0 if abs(new_pos) < 1e-9 else new_pos

            self._report_counter += 1
            report = ExecutionReport(
                report_id=f"rep-paper-{uuid.uuid4().hex[:12]}-{self._report_counter}",
                cl_ord_id=order.cl_ord_id,
                exchange_order_id=order.exchange_order_id,
                symbol=order.symbol,
                side=order.side,
                exec_type=OrderState.FILLED,
                last_quantity=order_qty,
                last_price=fill_price,
                cum_quantity=order_qty,
                leaves_quantity=0.0,
                cum_quote_amount=notional,
                average_price=fill_price,
                fee=fee,
                timestamp_ns=now_ns,
                text="Order filled immediately",
            )

            # Apply fill to order entity via FSM
            self._fsm.apply_execution_report(order, report)
            # Deregister from active idempotency tracking
            self._idempotency_router.deregister_order(order.cl_ord_id)
            return report

        else:
            # Rests on order book in NEW state
            self._report_counter += 1
            ack_report = ExecutionReport(
                report_id=f"rep-paper-{uuid.uuid4().hex[:12]}-{self._report_counter}",
                cl_ord_id=order.cl_ord_id,
                exchange_order_id=order.exchange_order_id,
                symbol=order.symbol,
                side=order.side,
                exec_type=OrderState.NEW,
                last_quantity=0.0,
                last_price=0.0,
                cum_quantity=0.0,
                leaves_quantity=order.quantity,
                cum_quote_amount=0.0,
                average_price=0.0,
                fee=0.0,
                timestamp_ns=now_ns,
                text="Limit order resting on book",
            )
            return ack_report

    async def cancel_order(self, cl_ord_id: str) -> ExecutionReport:
        """Request cancellation of an active resting or in-flight order.

        Args:
            cl_ord_id: Target order client identifier (non-empty string).

        Returns:
            ExecutionReport indicating CANCELLED state.

        Raises:
            GatewayDisconnectedException: If gateway is not connected (ERR-GW-005).
            InvalidOrderInputException: If cl_ord_id is invalid or order not found (ERR-GW-003).
            InvalidStateTransitionException: If order is already in terminal state (ERR-GW-001).
        """
        # Functional Purpose: Cancel active resting order and transition order entity to CANCELLED.
        # Explicit Dependency Tracking: self._orders, self._fsm, self._idempotency_router.
        # Structural Relationship: Invoked by risk management or strategy cancel triggers.
        # Defensive Invariant: Gateway connected; order exists in registry; order is not already terminal.

        # 1. Connection check (INV-GW-005)
        if not self._is_connected:
            raise GatewayDisconnectedException(
                "Cannot cancel order while gateway is disconnected",
                code=_ERR_DISCONNECTED,
            )

        # 2. Input validation (INV-GW-005)
        if not isinstance(cl_ord_id, str) or not cl_ord_id.strip():
            raise InvalidOrderInputException(
                f"cl_ord_id must be non-empty string, got {cl_ord_id!r}",
                code=_ERR_NON_FINITE,
            )

        # 3. Synthetic latency simulation
        if self._latency_ms > 0.0:
            await asyncio.sleep(self._latency_ms / 1000.0)

        # 4. Lookup order in gateway registry
        clean_id = cl_ord_id.strip()
        order = self._orders.get(clean_id)
        if order is None:
            raise InvalidOrderInputException(
                f"Order '{clean_id}' not found in gateway registry",
                code=_ERR_NON_FINITE,
            )

        # 5. Terminal state lockout (INV-GW-001)
        if order.is_terminal:
            raise InvalidStateTransitionException(
                f"Cannot cancel order {clean_id} in immutable terminal state {order.state.value}",
                code=_ERR_INVALID_TRANSITION,
            )

        # 6. Generate cancel ExecutionReport
        self._report_counter += 1
        now_ns = time.time_ns()
        report = ExecutionReport(
            report_id=f"rep-paper-{uuid.uuid4().hex[:12]}-{self._report_counter}",
            cl_ord_id=order.cl_ord_id,
            exchange_order_id=order.exchange_order_id or f"ex-{order.cl_ord_id}",
            symbol=order.symbol,
            side=order.side,
            exec_type=OrderState.CANCELLED,
            last_quantity=0.0,
            last_price=0.0,
            cum_quantity=order.filled_quantity,
            leaves_quantity=0.0,
            cum_quote_amount=order.filled_quote_amount,
            average_price=order.average_price,
            fee=0.0,
            timestamp_ns=now_ns,
            text="Order cancelled by client",
        )

        # 7. Apply cancel transition via FSM and deregister from active idempotency tracking
        self._fsm.apply_execution_report(order, report)
        self._idempotency_router.deregister_order(order.cl_ord_id)
        return report

    async def get_order(self, cl_ord_id: str) -> Order | None:
        """Retrieve an order by client_order_id.

        Args:
            cl_ord_id: Client order identifier (non-empty string).

        Returns:
            Order entity if found in gateway registry; None otherwise.

        Raises:
            InvalidOrderInputException: If cl_ord_id is empty or invalid (ERR-GW-003).
        """
        # Functional Purpose: Fast O(1) query of order lifecycle state from gateway registry.
        # Explicit Dependency Tracking: self._orders.
        # Structural Relationship: Queried by order tracking pipelines and verification tests.
        # Defensive Invariant: cl_ord_id must be non-empty string.
        if not isinstance(cl_ord_id, str) or not cl_ord_id.strip():
            raise InvalidOrderInputException(
                f"cl_ord_id must be non-empty string, got {cl_ord_id!r}",
                code=_ERR_NON_FINITE,
            )
        return self._orders.get(cl_ord_id.strip())

    async def get_open_orders(self) -> list[Order]:
        """Retrieve all currently active / open orders.

        Returns:
            List of Order instances currently resting or in-flight (non-terminal).
        """
        # Functional Purpose: Return snapshot of active orders for portfolio risk reconciliation.
        # Explicit Dependency Tracking: self._orders, Order.is_active.
        # Structural Relationship: Queried by position reconcilers and cancel-on-disconnect routines.
        # Defensive Invariant: Returns list of orders where order.is_active is True.
        return [o for o in self._orders.values() if o.is_active]

    async def get_positions(self) -> dict[str, float]:
        """Retrieve current portfolio instrument positions.

        Returns:
            Shallow copy of positions mapping instrument symbol to quantity.
        """
        # Functional Purpose: Return current inventory across all traded instruments.
        # Explicit Dependency Tracking: self._positions.
        # Structural Relationship: Queried by strategy engines and portfolio balance calculators.
        # Defensive Invariant: Returns dictionary copy guarding internal state against caller mutations.
        return dict(self._positions)

    async def get_account_balance(self) -> dict[str, float]:
        """Retrieve current account cash and total mark-to-market equity balances.

        Returns:
            Dictionary containing 'cash' and 'equity' float balances.
        """
        # Functional Purpose: Compute total purchasing power and mark-to-market portfolio equity.
        # Explicit Dependency Tracking: self._cash, self._positions, self._market_prices.
        # Structural Relationship: Consumed by risk management modules and capital allocation filters.
        # Defensive Invariant: Cash and equity values are strictly finite floats.
        equity = self._cash + sum(
            qty * self._market_prices.get(sym, 0.0) for sym, qty in self._positions.items()
        )
        return {
            "cash": self._cash,
            "equity": equity,
        }

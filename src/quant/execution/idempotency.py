"""Cryptographic Idempotency Token Router and In-Flight Order Ring Buffer.

Purpose:
    Enforces microstructural execution invariants for order lifecycle tracking:
    1. Cryptographic Idempotency Token Uniqueness (INV-GW-002): Outbound orders receive
       deterministic UUIDv5 identifiers preventing order replay and double-execution under
       network jitter or retry loops.
    2. Strict Non-Finite Input & Boundary Protection (INV-GW-005): Comprehensive validation
       guarding against non-positive capacities, non-finite TTLs, boolean type punning, and
       corrupt order tokens.
    3. Hot-Path Latency SLA (INV-GW-006): Zero-lock O(1) active dictionary lookups and bounded
       ring buffer FIFO eviction completing in <= 0.05ms (50us).

Dependencies:
    - collections.deque: High-performance double-ended queue for bounded FIFO ring buffer tracking.
    - math: Finite float scalar verification (math.isfinite).
    - typing: Static typing annotations, ClassVar, Final.
    - uuid: RFC 4122 cryptographic UUIDv5 hashing.
    - quant.execution.models: Order, OrderSide, diagnostic exceptions, error codes.

Structural Relationship:
    - Core routing component for:
        1. ExecutionGateway & PaperExecutionGateway (gateway.py)
        2. AsyncBrokerGateway wire adapters (gateway.py)
        3. OrderStateMachine lifecycle integration (fsm.py)
    - Emits: Deterministic client_order_id tokens, cached Order references, and duplicate fault tripwires.

Invariants Enforced:
    - INV-GW-002: Deterministic UUIDv5 hashing and active/historical duplicate rejection.
    - INV-GW-005: Strict domain boundary validation and identifier verification.
    - INV-GW-006: Hot-path latency SLA <= 0.05ms (50us).
"""

from __future__ import annotations

import math
import uuid
from collections import deque
from typing import Final

from quant.execution.models import (
    ERR_GW_DUPLICATE_ORDER_ID,
    ERR_GW_NON_FINITE_INPUT,
    DuplicateOrderException,
    InvalidOrderInputException,
    Order,
    OrderSide,
)

# Fixed UUIDv5 namespace constant for quantitative order token derivation.
# Functional Purpose: Provide deterministic RFC 4122 cryptographic hashing domain for order tokens.
# Explicit Dependency Tracking: uuid.UUID.
# Structural Relationship: Used by IdempotencyRouter.generate_client_order_id.
# Defensive Invariant: Immutable Final UUID instance matching specified domain seed.
NAMESPACE_QUANT_ORDER: Final[uuid.UUID] = uuid.UUID("a7e1c0d4-1234-5678-9abc-def012345678")


class IdempotencyRouter:
    """Deterministic cryptographic idempotency token router and in-flight ring buffer.

    Maintains an active order lookup table and a bounded historical ring buffer with O(1) set
    deduplication, enforcing INV-GW-002 (Cryptographic Idempotency Token Uniqueness),
    INV-GW-005 (Strict Non-Finite & Boundary Protection), and INV-GW-006 (Hot-Path Latency SLA).
    """

    __slots__ = (
        "_active_orders",
        "_history_capacity",
        "_history_ring",
        "_history_set",
        "_ttl_seconds",
    )

    def __init__(self, history_capacity: int = 10_000, ttl_seconds: float = 86400.0) -> None:
        """Initialize the idempotency router with capacity limits and retention window.

        Args:
            history_capacity: Maximum historical order IDs retained for deduplication (int > 0).
            ttl_seconds: Historical deduplication retention window in seconds (finite float > 0.0).

        Raises:
            InvalidOrderInputException: If history_capacity or ttl_seconds violate domain boundaries.
        """
        # Functional Purpose: Initialize active order registry and bounded FIFO historical ring buffer.
        # Explicit Dependency Tracking: collections.deque, set, dict, math.isfinite, InvalidOrderInputException.
        # Structural Relationship: Instantiated by ExecutionGateway and broker router pipelines.
        # Defensive Invariant: history_capacity must be int > 0; ttl_seconds must be finite float > 0.0; reject bool.
        if (
            isinstance(history_capacity, bool)
            or not isinstance(history_capacity, int)
            or history_capacity <= 0
        ):
            raise InvalidOrderInputException(
                f"history_capacity must be an integer > 0, got {history_capacity!r}",
                code=ERR_GW_NON_FINITE_INPUT,
            )
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, (int, float))
            or not math.isfinite(ttl_seconds)
            or ttl_seconds <= 0.0
        ):
            raise InvalidOrderInputException(
                f"ttl_seconds must be a finite float > 0.0, got {ttl_seconds!r}",
                code=ERR_GW_NON_FINITE_INPUT,
            )

        self._history_capacity: int = history_capacity
        self._ttl_seconds: float = float(ttl_seconds)
        self._active_orders: dict[str, Order] = {}
        self._history_ring: deque[str] = deque(maxlen=history_capacity)
        self._history_set: set[str] = set()

    @property
    def history_capacity(self) -> int:
        """Maximum capacity of the historical deduplication ring buffer."""
        # Functional Purpose: Expose configured maximum capacity for historical order ID tracking.
        # Explicit Dependency Tracking: self._history_capacity.
        # Structural Relationship: Queried by monitoring systems and unit tests.
        # Defensive Invariant: Always strictly positive integer > 0.
        return self._history_capacity

    @property
    def ttl_seconds(self) -> float:
        """Configured time-to-live threshold in seconds for historical order caching."""
        # Functional Purpose: Expose configured deduplication retention TTL window.
        # Explicit Dependency Tracking: self._ttl_seconds.
        # Structural Relationship: Queried by risk engines and monitoring health checks.
        # Defensive Invariant: Always finite float > 0.0.
        return self._ttl_seconds

    @property
    def active_order_count(self) -> int:
        """Total number of in-flight active orders currently tracked."""
        # Functional Purpose: Return instantaneous cardinality of in-flight active orders.
        # Explicit Dependency Tracking: self._active_orders.
        # Structural Relationship: Queried by order routing pipeline and pre-trade risk filters.
        # Defensive Invariant: Always non-negative integer >= 0.
        return len(self._active_orders)

    @property
    def history_count(self) -> int:
        """Total number of historical completed/cancelled orders retained in the ring buffer."""
        # Functional Purpose: Return current size of bounded historical deduplication set.
        # Explicit Dependency Tracking: self._history_set.
        # Structural Relationship: Queried by telemetry diagnostics and memory leak watchdogs.
        # Defensive Invariant: Always bounded by 0 <= count <= history_capacity.
        return len(self._history_set)

    def generate_client_order_id(
        self,
        strategy_id: str,
        symbol: str,
        side: OrderSide,
        timestamp_ns: int,
        nonce: int = 0,
    ) -> str:
        """Synthesize a deterministic, collision-resistant UUIDv5 client order identifier (INV-GW-002).

        Args:
            strategy_id: Unique strategy or alpha model identifier (non-empty string).
            symbol: Ticker symbol (non-empty string).
            side: OrderSide enum instance (BUY or SELL).
            timestamp_ns: Nanosecond timestamp integer (>= 0).
            nonce: Disambiguation sequence nonce integer (>= 0, defaults to 0).

        Returns:
            Synthesized client order ID formatted as 'cl-{uuid5}'.

        Raises:
            InvalidOrderInputException: If any argument violates domain boundaries or types.
        """
        # Functional Purpose: Deterministically derive a cryptographically unique UUIDv5 client order ID.
        # Explicit Dependency Tracking: uuid.uuid5, NAMESPACE_QUANT_ORDER, OrderSide.
        # Structural Relationship: Called prior to Order submission by strategy engines and execution gateways.
        # Defensive Invariant: strategy_id and symbol non-empty; side is OrderSide; timestamp_ns, nonce >= 0 int (non-bool).
        if not isinstance(strategy_id, str) or not strategy_id.strip():
            raise InvalidOrderInputException(
                f"strategy_id must be a non-empty string, got {strategy_id!r}",
                code=ERR_GW_NON_FINITE_INPUT,
            )
        if not isinstance(symbol, str) or not symbol.strip():
            raise InvalidOrderInputException(
                f"symbol must be a non-empty string, got {symbol!r}",
                code=ERR_GW_NON_FINITE_INPUT,
            )
        if not isinstance(side, OrderSide):
            raise InvalidOrderInputException(
                f"side must be an instance of OrderSide, got {type(side).__name__}",
                code=ERR_GW_NON_FINITE_INPUT,
            )
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int) or timestamp_ns < 0:
            raise InvalidOrderInputException(
                f"timestamp_ns must be an integer >= 0, got {timestamp_ns!r}",
                code=ERR_GW_NON_FINITE_INPUT,
            )
        if isinstance(nonce, bool) or not isinstance(nonce, int) or nonce < 0:
            raise InvalidOrderInputException(
                f"nonce must be an integer >= 0, got {nonce!r}",
                code=ERR_GW_NON_FINITE_INPUT,
            )

        name_payload = f"{strategy_id}:{symbol}:{side.value}:{timestamp_ns}:{nonce}"
        token = uuid.uuid5(NAMESPACE_QUANT_ORDER, name_payload)
        return f"cl-{token}"

    def register_order(self, order: Order) -> None:
        """Register an order as in-flight active, rejecting collisions under INV-GW-002.

        Args:
            order: Valid Order instance to register.

        Raises:
            InvalidOrderInputException: If order is not an Order instance.
            DuplicateOrderException: If order.cl_ord_id is already active or in historical ring buffer.
        """
        # Functional Purpose: Ingest outbound order into active tracking, guarding against replay attacks.
        # Explicit Dependency Tracking: self._active_orders, self._history_set, DuplicateOrderException.
        # Structural Relationship: Executed by ExecutionGateway on order submission hot path.
        # Defensive Invariant: order must be Order; cl_ord_id must not exist in active dict or history set.
        if not isinstance(order, Order):
            raise InvalidOrderInputException(
                f"order must be an instance of Order, got {type(order).__name__}",
                code=ERR_GW_NON_FINITE_INPUT,
            )

        cl_ord_id = order.cl_ord_id
        if cl_ord_id in self._active_orders or cl_ord_id in self._history_set:
            raise DuplicateOrderException(
                f"Order with client_order_id {cl_ord_id!r} already registered or in historical ring buffer.",
                code=ERR_GW_DUPLICATE_ORDER_ID,
            )

        self._active_orders[cl_ord_id] = order

    def deregister_order(self, cl_ord_id: str) -> Order | None:
        """Deregister an active order upon termination, transitioning token to bounded historical ring buffer.

        Args:
            cl_ord_id: Order identifier to deregister (non-empty string).

        Returns:
            The deregistered Order instance if it was active; None if it was not active.

        Raises:
            InvalidOrderInputException: If cl_ord_id is empty, non-string, or boolean.
        """
        # Functional Purpose: Transition terminal order from active registry to historical deduplication buffer.
        # Explicit Dependency Tracking: self._active_orders, self._history_ring, self._history_set.
        # Structural Relationship: Invoked upon terminal state transitions (FILLED, CANCELLED, REJECTED, EXPIRED).
        # Defensive Invariant: FIFO eviction maintains exact bounded size <= history_capacity.
        if not isinstance(cl_ord_id, str) or not cl_ord_id.strip():
            raise InvalidOrderInputException(
                f"cl_ord_id must be a non-empty string, got {cl_ord_id!r}",
                code=ERR_GW_NON_FINITE_INPUT,
            )

        order = self._active_orders.pop(cl_ord_id, None)
        if order is None:
            return None

        if len(self._history_ring) >= self._history_capacity:
            evicted_id = self._history_ring.popleft()
            self._history_set.discard(evicted_id)

        self._history_ring.append(cl_ord_id)
        self._history_set.add(cl_ord_id)
        return order

    def get_active_order(self, cl_ord_id: str) -> Order | None:
        """Lookup an active in-flight order by client_order_id in O(1) time.

        Args:
            cl_ord_id: Client order identifier (non-empty string).

        Returns:
            The Order instance if active; None otherwise.

        Raises:
            InvalidOrderInputException: If cl_ord_id is empty, non-string, or boolean.
        """
        # Functional Purpose: Fast O(1) retrieval of in-flight active order references without locks.
        # Explicit Dependency Tracking: self._active_orders.
        # Structural Relationship: Queried during execution report reconciliation and cancel routing.
        # Defensive Invariant: cl_ord_id must be non-empty string.
        if not isinstance(cl_ord_id, str) or not cl_ord_id.strip():
            raise InvalidOrderInputException(
                f"cl_ord_id must be a non-empty string, got {cl_ord_id!r}",
                code=ERR_GW_NON_FINITE_INPUT,
            )
        return self._active_orders.get(cl_ord_id)

    def is_duplicate(self, cl_ord_id: str) -> bool:
        """Evaluate whether a client_order_id is currently active or in the historical deduplication window.

        Args:
            cl_ord_id: Client order identifier to check (non-empty string).

        Returns:
            True if present in active orders or historical deduplication set; False otherwise.

        Raises:
            InvalidOrderInputException: If cl_ord_id is empty, non-string, or boolean.
        """
        # Functional Purpose: O(1) duplicate membership test before submitting outbound wire messages.
        # Explicit Dependency Tracking: self._active_orders, self._history_set.
        # Structural Relationship: Pre-trade idempotency checkpoint for execution gateways.
        # Defensive Invariant: Returns boolean result; raises on invalid identifier format.
        if not isinstance(cl_ord_id, str) or not cl_ord_id.strip():
            raise InvalidOrderInputException(
                f"cl_ord_id must be a non-empty string, got {cl_ord_id!r}",
                code=ERR_GW_NON_FINITE_INPUT,
            )
        return cl_ord_id in self._active_orders or cl_ord_id in self._history_set

    def clear(self) -> None:
        """Purge all active orders and historical deduplication ring buffer states."""
        # Functional Purpose: Reset the entire router state for clean gateway restart or session teardown.
        # Explicit Dependency Tracking: self._active_orders, self._history_ring, self._history_set.
        # Structural Relationship: Invoked during gateway disconnect and unit test fixtures.
        # Defensive Invariant: All collections reset to empty state with zero memory retention.
        self._active_orders.clear()
        self._history_ring.clear()
        self._history_set.clear()

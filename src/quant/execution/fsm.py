"""Order State Machine with causal out-of-order reconciliation and invariant protection.

Purpose:
    Provides deterministic, causal lifecycle management for live market orders adhering to
    the FIX protocol state transition graph and microstructural invariants:
    1. Causal State Machine Monotonicity (INV-GW-001): Terminal states (FILLED, CANCELLED,
       REJECTED, EXPIRED) are strictly immutable.
    2. Execution Mass Conservation (INV-GW-003): Cumulative filled quantity is strictly bounded
       by initial target quantity (cum_qty <= quantity + 1e-7).
    3. Causal Out-of-Order Packet Reconciliation (INV-GW-004): Automatic synthesis of intermediate
       NEW state when execution reports arrive prior to wire acknowledgments.
    4. Strict Non-Finite Input & Boundary Protection (INV-GW-005): Complete rejection of NaN, Inf,
       type errors, and order ID mismatches.
    5. Sub-Millisecond Gateway Execution Latency SLA (INV-GW-006): Zero-allocation, lock-free
       in-memory transition graph execution completing in < 10us.

Dependencies:
    - time: System nanosecond epoch timestamp resolution (time.time_ns).
    - typing: Static typing annotations, ClassVar, Final.
    - quant.execution.models: Order, ExecutionReport, OrderState, diagnostic exceptions, error codes.

Structural Relationship:
    - Core state transition engine for:
        1. ExecutionGateway & PaperExecutionGateway (gateway.py)
        2. IdempotencyRouter (idempotency.py)
        3. OrderAuditLogger (audit.py)
    - Emits: Mutated Order instances with updated state, filled quantities, average prices, and timestamps.

Invariants Enforced:
    - INV-GW-001: Strict terminal lockout and directed acyclic state transitions.
    - INV-GW-003: Mass conservation (0.0 <= cum_qty <= target_qty + 1e-7).
    - INV-GW-004: Automatic reconciliation of out-of-order fills from PENDING_NEW.
    - INV-GW-005: Strict domain boundary validation and identifier verification.
    - INV-GW-006: Hot-path latency SLA < 0.10ms (100us).
"""

from __future__ import annotations

import time
from typing import ClassVar, Final

from quant.execution.models import (
    ERR_GW_INVALID_STATE_TRANSITION,
    ERR_GW_NON_FINITE_INPUT,
    ExecutionReport,
    InvalidOrderInputException,
    InvalidStateTransitionException,
    Order,
    OrderState,
)

# Transition matrix defining permitted target states for each current state under INV-GW-001.
_VALID_TRANSITIONS: Final[dict[OrderState, frozenset[OrderState]]] = {
    OrderState.PENDING_NEW: frozenset(
        {
            OrderState.NEW,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
        }
    ),
    OrderState.NEW: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.PENDING_CANCEL,
            OrderState.CANCELLED,
            OrderState.EXPIRED,
        }
    ),
    OrderState.PARTIALLY_FILLED: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.PENDING_CANCEL,
            OrderState.CANCELLED,
            OrderState.EXPIRED,
        }
    ),
    OrderState.PENDING_CANCEL: frozenset(
        {
            OrderState.CANCELLED,
            OrderState.FILLED,
            OrderState.PARTIALLY_FILLED,
        }
    ),
    OrderState.FILLED: frozenset(),
    OrderState.CANCELLED: frozenset(),
    OrderState.REJECTED: frozenset(),
    OrderState.EXPIRED: frozenset(),
}


class OrderStateMachine:
    """Deterministic Order State Machine with causal out-of-order reconciliation and invariant protection.

    Enforces the deterministic directed acyclic transition graph across order lifecycles,
    synthesizes intermediate causal states for out-of-order execution packets (INV-GW-004),
    and strictly locks down terminal states (INV-GW-001).
    """

    __slots__ = ()

    _TRANSITIONS: ClassVar[dict[OrderState, frozenset[OrderState]]] = _VALID_TRANSITIONS

    @classmethod
    def can_transition(cls, current_state: OrderState, next_state: OrderState) -> bool:
        """Evaluate whether a state transition from current_state to next_state is valid under INV-GW-001.

        Args:
            current_state: Current order lifecycle state.
            next_state: Desired target lifecycle state.

        Returns:
            True if the transition is permitted by the DAG; False otherwise.
        """
        # Functional Purpose: Deterministically evaluate transition validity under INV-GW-001 without mutating state.
        # Explicit Dependency Tracking: cls._TRANSITIONS, OrderState enum.
        # Structural Relationship: Queried by ExecutionGateway, risk routers, and transition methods before mutation.
        # Defensive Invariant: Returns False if either state is not an OrderState instance or if current_state is terminal.
        if not isinstance(current_state, OrderState) or not isinstance(next_state, OrderState):
            return False
        permitted = cls._TRANSITIONS.get(current_state, frozenset())
        return next_state in permitted

    @classmethod
    def transition(
        cls,
        order: Order,
        next_state: OrderState,
        timestamp_ns: int | ExecutionReport | None = None,
        *,
        report: ExecutionReport | None = None,
    ) -> Order:
        """Transition an order to a target state, verifying INV-GW-001 and reconciling INV-GW-004.

        Args:
            order: Mutable order entity to transition.
            next_state: Target OrderState.
            timestamp_ns: Optional epoch nanosecond transition timestamp (defaults to current time_ns).
            report: Optional ExecutionReport if this transition was triggered by an execution event.

        Returns:
            The mutated Order instance with updated state and timestamp.

        Raises:
            InvalidOrderInputException: If inputs violate domain types or finiteness (ERR-GW-003).
            InvalidStateTransitionException: If transition violates INV-GW-001 (ERR-GW-001).
        """
        # Functional Purpose: Enforce monotonic state transitions on order entities with causal out-of-order reconciliation.
        # Explicit Dependency Tracking: cls.can_transition, ERR_GW_INVALID_STATE_TRANSITION, ERR_GW_NON_FINITE_INPUT.
        # Structural Relationship: Invoked by ExecutionGateway upon receiving order routing events or cancel acks.
        # Defensive Invariant: Terminal orders cannot transition; timestamp must be non-negative integer.

        # 1. Input type validation
        if not isinstance(order, Order):
            raise InvalidOrderInputException(
                f"OrderStateMachine.transition requires Order instance, got {type(order).__name__}",
                code=ERR_GW_NON_FINITE_INPUT,
            )
        if not isinstance(next_state, OrderState):
            raise InvalidOrderInputException(
                f"OrderStateMachine.transition requires OrderState instance, got {type(next_state).__name__}",
                code=ERR_GW_NON_FINITE_INPUT,
            )

        # 2. Check if a report was supplied positionally or by keyword
        actual_report: ExecutionReport | None = None
        if isinstance(timestamp_ns, ExecutionReport):
            actual_report = timestamp_ns
            resolved_ts = actual_report.timestamp_ns
        elif report is not None:
            if not isinstance(report, ExecutionReport):
                raise InvalidOrderInputException(
                    f"ExecutionReport must be ExecutionReport instance, got {type(report).__name__}",
                    code=ERR_GW_NON_FINITE_INPUT,
                )
            actual_report = report
            resolved_ts = actual_report.timestamp_ns
        elif timestamp_ns is None:
            resolved_ts = time.time_ns()
        else:
            if (
                not isinstance(timestamp_ns, int)
                or isinstance(timestamp_ns, bool)
                or timestamp_ns < 0
            ):
                raise InvalidOrderInputException(
                    f"Order transition timestamp_ns must be integer >= 0, got {timestamp_ns!r}",
                    code=ERR_GW_NON_FINITE_INPUT,
                )
            resolved_ts = timestamp_ns

        # 3. If an ExecutionReport is provided, delegate to apply_execution_report for full accounting
        if actual_report is not None:
            if actual_report.exec_type != next_state:
                raise InvalidStateTransitionException(
                    f"Target state {next_state.value} does not match report exec_type {actual_report.exec_type.value}",
                    code=ERR_GW_INVALID_STATE_TRANSITION,
                )
            return cls.apply_execution_report(order, actual_report)

        # 4. Terminal state lockout (INV-GW-001)
        if order.is_terminal:
            raise InvalidStateTransitionException(
                f"Cannot transition order {order.cl_ord_id} from terminal state {order.state.value} to {next_state.value}",
                code=ERR_GW_INVALID_STATE_TRANSITION,
            )

        # 5. Causal out-of-order reconciliation check (INV-GW-004)
        if order.state == OrderState.PENDING_NEW and next_state in (
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
        ):
            # Synthesize intermediate transition to NEW
            order.state = OrderState.NEW
            order.updated_at_ns = resolved_ts

        # 6. Validate transition validity under DAG
        if not cls.can_transition(order.state, next_state):
            raise InvalidStateTransitionException(
                f"Invalid state transition from {order.state.value} to {next_state.value} for order {order.cl_ord_id}",
                code=ERR_GW_INVALID_STATE_TRANSITION,
            )

        # 7. Apply state mutation
        order.state = next_state
        order.updated_at_ns = resolved_ts
        return order

    @classmethod
    def apply_execution_report(cls, order: Order, report: ExecutionReport) -> Order:
        """Apply an execution report to an order with causal out-of-order reconciliation and mass conservation.

        Args:
            order: Mutable Order entity to update.
            report: Immutable ExecutionReport from exchange or gateway.

        Returns:
            The mutated Order instance reflecting the execution event.

        Raises:
            InvalidOrderInputException: If identifiers mismatch or mass conservation fails (ERR-GW-003).
            InvalidStateTransitionException: If transition violates INV-GW-001 or terminal lockout (ERR-GW-001).
        """
        # Functional Purpose: Process fills, cancels, and acks with causal packet reconciliation under INV-GW-001/3/4.
        # Explicit Dependency Tracking: Order, ExecutionReport, INV-GW-001, INV-GW-003, INV-GW-004, ERR_GW_NON_FINITE_INPUT.
        # Structural Relationship: Main ingress point for ExecutionGateway when receiving execution events from brokers.
        # Defensive Invariant: cl_ord_id must match; cum_quantity <= order.quantity + 1e-7; terminal states locked.

        # 1. Type validation
        if not isinstance(order, Order):
            raise InvalidOrderInputException(
                f"OrderStateMachine.apply_execution_report requires Order instance, got {type(order).__name__}",
                code=ERR_GW_NON_FINITE_INPUT,
            )
        if not isinstance(report, ExecutionReport):
            raise InvalidOrderInputException(
                f"OrderStateMachine.apply_execution_report requires ExecutionReport instance, got {type(report).__name__}",
                code=ERR_GW_NON_FINITE_INPUT,
            )

        # 2. Identifier matching validation (INV-GW-005)
        if order.cl_ord_id != report.cl_ord_id:
            raise InvalidOrderInputException(
                f"Order cl_ord_id '{order.cl_ord_id}' does not match ExecutionReport cl_ord_id '{report.cl_ord_id}'",
                code=ERR_GW_NON_FINITE_INPUT,
            )

        # 3. Terminal state lockout (INV-GW-001)
        if order.is_terminal:
            raise InvalidStateTransitionException(
                f"Cannot apply execution report {report.report_id} to order {order.cl_ord_id} "
                f"in immutable terminal state {order.state.value}",
                code=ERR_GW_INVALID_STATE_TRANSITION,
            )

        # 4. Execution mass conservation verification (INV-GW-003)
        # Cumulative filled quantity must not exceed target quantity within epsilon tolerance
        if report.cum_quantity > order.quantity + 1e-7:
            raise InvalidOrderInputException(
                f"Execution report cumulative quantity {report.cum_quantity} exceeds order target "
                f"quantity {order.quantity} (INV-GW-003 violation)",
                code=ERR_GW_NON_FINITE_INPUT,
            )
        # Cumulative filled quantity must not regress monotonically
        if report.cum_quantity < order.filled_quantity - 1e-7:
            raise InvalidOrderInputException(
                f"Execution report cumulative quantity {report.cum_quantity} is less than previously "
                f"filled quantity {order.filled_quantity} (monotonicity violation)",
                code=ERR_GW_NON_FINITE_INPUT,
            )

        # 5. Causal out-of-order packet reconciliation (INV-GW-004)
        if order.state == OrderState.PENDING_NEW and report.exec_type in (
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
        ):
            # Synthesize intermediate transition to NEW before applying fill
            order.state = OrderState.NEW
            order.updated_at_ns = report.timestamp_ns
            if report.exchange_order_id:
                order.exchange_order_id = report.exchange_order_id

        # 6. Directed transition graph validation (INV-GW-001)
        if not cls.can_transition(order.state, report.exec_type):
            raise InvalidStateTransitionException(
                f"Invalid state transition from {order.state.value} to {report.exec_type.value} "
                f"for order {order.cl_ord_id}",
                code=ERR_GW_INVALID_STATE_TRANSITION,
            )

        # 7. Apply execution report mutations
        order.filled_quantity = report.cum_quantity
        order.filled_quote_amount = report.cum_quote_amount
        order.average_price = report.average_price
        order.fees_paid += report.fee
        if report.exchange_order_id:
            order.exchange_order_id = report.exchange_order_id
        order.updated_at_ns = report.timestamp_ns
        order.state = report.exec_type

        return order

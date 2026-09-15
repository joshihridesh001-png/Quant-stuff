"""Unit tests for execution domain models, enums, exceptions, and diagnostic error codes.

Governing Standards:
- Rules.md (Rule 1, Rule 2, Rule 3, Rule 4)
- INV-GW-001: Causal State Machine Monotonicity
- INV-GW-002: Cryptographic Idempotency Token Uniqueness
- INV-GW-003: Execution Mass Conservation
- INV-GW-005: Strict Non-Finite Input & Boundary Protection
"""

import math  # noqa: F401

import pytest

from quant.execution.models import (
    ERR_GW_DISCONNECTED,
    ERR_GW_DUPLICATE_ORDER_ID,
    ERR_GW_INSUFFICIENT_MARGIN,
    ERR_GW_INVALID_STATE_TRANSITION,
    ERR_GW_NON_FINITE_INPUT,
    ERR_GW_RATE_LIMIT_EXCEEDED,
    DuplicateOrderException,
    ExecutionReport,
    GatewayDisconnectedException,
    GatewayError,
    InsufficientMarginException,
    InvalidOrderInputException,
    InvalidStateTransitionException,
    Order,
    OrderSide,
    OrderState,
    OrderType,
    RateLimitExceededException,
    TimeInForce,
)


def test_order_creation_and_slots() -> None:
    """Test valid order instantiation, default field assignments, and slot enforcement."""
    order = Order(
        cl_ord_id="ord-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.STOP_LIMIT,
        quantity=100.0,
        price=150.0,
        stop_price=145.0,
        time_in_force=TimeInForce.GTC,
    )
    assert order.cl_ord_id == "ord-001"
    assert order.symbol == "AAPL"
    assert order.side == OrderSide.BUY
    assert order.order_type == OrderType.STOP_LIMIT
    assert order.quantity == 100.0
    assert order.price == 150.0
    assert order.stop_price == 145.0
    assert order.time_in_force == TimeInForce.GTC
    assert order.state == OrderState.PENDING_NEW
    assert order.exchange_order_id is None
    assert order.filled_quantity == 0.0
    assert order.filled_quote_amount == 0.0
    assert order.average_price == 0.0
    assert order.fees_paid == 0.0
    assert order.created_at_ns == 0
    assert order.updated_at_ns == 0
    assert order.leaves_quantity == 100.0

    # Verify slot enforcement: setting an arbitrary attribute must raise AttributeError
    with pytest.raises(AttributeError):
        order.unrecognized_dynamic_attr = "forbidden"  # type: ignore[attr-defined]


def test_order_properties_lifecycle() -> None:
    """Test leaves_quantity, is_terminal, and is_active across the lifecycle."""
    order = Order(
        cl_ord_id="ord-002",
        symbol="MSFT",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=50.0,
        price=320.0,
    )

    # Initial state: PENDING_NEW
    assert order.is_active is True
    assert order.is_terminal is False
    assert order.leaves_quantity == 50.0

    # State: NEW
    order.state = OrderState.NEW
    assert order.is_active is True
    assert order.is_terminal is False

    # Partial fill
    order.state = OrderState.PARTIALLY_FILLED
    order.filled_quantity = 20.0
    assert order.is_active is True
    assert order.is_terminal is False
    assert order.leaves_quantity == 30.0

    # PENDING_CANCEL
    order.state = OrderState.PENDING_CANCEL
    assert order.is_active is True
    assert order.is_terminal is False

    # Terminal state: FILLED
    order.state = OrderState.FILLED
    order.filled_quantity = 50.0
    assert order.is_active is False
    assert order.is_terminal is True
    assert order.leaves_quantity == 0.0

    # Overfill protection in leaves_quantity (max(0.0, ...))
    order.filled_quantity = 55.0
    assert order.leaves_quantity == 0.0

    # Terminal state: CANCELLED
    order.state = OrderState.CANCELLED
    assert order.is_active is False
    assert order.is_terminal is True

    # Terminal state: REJECTED
    order.state = OrderState.REJECTED
    assert order.is_active is False
    assert order.is_terminal is True

    # Terminal state: EXPIRED
    order.state = OrderState.EXPIRED
    assert order.is_active is False
    assert order.is_terminal is True


@pytest.mark.parametrize(
    "bad_qty",
    [0.0, -1.0, -1e-8, float("nan"), float("inf"), float("-inf"), True, False, "100", None],
)
def test_order_invalid_quantity_rejected(bad_qty: object) -> None:
    """Quantity must be strictly positive, finite, non-boolean float."""
    with pytest.raises(InvalidOrderInputException) as exc:
        Order(
            cl_ord_id="ord-bad-qty",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=bad_qty,  # type: ignore[arg-type]
            price=150.0,
        )
    assert exc.value.code == ERR_GW_NON_FINITE_INPUT


@pytest.mark.parametrize(
    "bad_price",
    [0.0, -1.0, -1e-8, float("nan"), float("inf"), float("-inf"), True, False, "150"],
)
def test_order_invalid_price_rejected(bad_price: object) -> None:
    """Price if specified must be strictly positive, finite, non-boolean float."""
    with pytest.raises(InvalidOrderInputException) as exc:
        Order(
            cl_ord_id="ord-bad-price",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=10.0,
            price=bad_price,  # type: ignore[arg-type]
        )
    assert exc.value.code == ERR_GW_NON_FINITE_INPUT


@pytest.mark.parametrize(
    "bad_stop_price",
    [0.0, -1.0, -1e-8, float("nan"), float("inf"), float("-inf"), True, False, "150"],
)
def test_order_invalid_stop_price_rejected(bad_stop_price: object) -> None:
    """Stop price if specified must be strictly positive, finite, non-boolean float."""
    with pytest.raises(InvalidOrderInputException) as exc:
        Order(
            cl_ord_id="ord-bad-stop",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.STOP_LIMIT,
            quantity=10.0,
            price=150.0,
            stop_price=bad_stop_price,  # type: ignore[arg-type]
        )
    assert exc.value.code == ERR_GW_NON_FINITE_INPUT


@pytest.mark.parametrize(
    ("field_name", "bad_val"),
    [
        ("filled_quantity", -1.0),
        ("filled_quantity", float("nan")),
        ("filled_quantity", float("inf")),
        ("filled_quantity", True),
        ("filled_quote_amount", -1.0),
        ("filled_quote_amount", float("nan")),
        ("filled_quote_amount", float("inf")),
        ("filled_quote_amount", False),
        ("average_price", -1.0),
        ("average_price", float("nan")),
        ("average_price", float("inf")),
        ("average_price", True),
        ("fees_paid", -1.0),
        ("fees_paid", float("nan")),
        ("fees_paid", float("inf")),
        ("fees_paid", False),
        ("created_at_ns", -1),
        ("created_at_ns", True),
        ("created_at_ns", 1.5),
        ("updated_at_ns", -1),
        ("updated_at_ns", False),
        ("updated_at_ns", 1.5),
    ],
)
def test_order_invalid_auxiliary_fields_rejected(field_name: str, bad_val: object) -> None:
    """Non-negative bounds and types for auxiliary fields must be enforced."""
    kwargs: dict[str, object] = {
        "cl_ord_id": "ord-aux",
        "symbol": "AAPL",
        "side": OrderSide.BUY,
        "order_type": OrderType.LIMIT,
        "quantity": 10.0,
        "price": 100.0,
        field_name: bad_val,
    }
    with pytest.raises(InvalidOrderInputException) as exc:
        Order(**kwargs)  # type: ignore[arg-type]
    assert exc.value.code == ERR_GW_NON_FINITE_INPUT


@pytest.mark.parametrize(
    ("field_name", "bad_val"),
    [
        ("cl_ord_id", ""),
        ("cl_ord_id", 123),
        ("symbol", ""),
        ("symbol", None),
        ("side", "INVALID_SIDE"),
        ("order_type", "INVALID_TYPE"),
        ("time_in_force", "INVALID_TIF"),
        ("state", "INVALID_STATE"),
        ("exchange_order_id", 123),
    ],
)
def test_order_invalid_identifiers_and_enums_rejected(field_name: str, bad_val: object) -> None:
    """Identifier strings and enum instances must be strictly validated."""
    kwargs: dict[str, object] = {
        "cl_ord_id": "ord-valid",
        "symbol": "AAPL",
        "side": OrderSide.BUY,
        "order_type": OrderType.LIMIT,
        "quantity": 10.0,
        "price": 100.0,
        field_name: bad_val,
    }
    with pytest.raises(InvalidOrderInputException) as exc:
        Order(**kwargs)  # type: ignore[arg-type]
    assert exc.value.code == ERR_GW_NON_FINITE_INPUT


def test_execution_report_instantiation_and_immutability() -> None:
    """Test valid ExecutionReport instantiation, frozen immutability, and slot enforcement."""
    report = ExecutionReport(
        report_id="rep-001",
        cl_ord_id="ord-001",
        exchange_order_id="ex-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        exec_type=OrderState.NEW,
        last_quantity=0.0,
        last_price=0.0,
        cum_quantity=0.0,
        leaves_quantity=100.0,
        cum_quote_amount=0.0,
        average_price=0.0,
        fee=0.0,
        timestamp_ns=1_000_000,
        text="Order placed",
    )
    assert report.report_id == "rep-001"
    assert report.cl_ord_id == "ord-001"
    assert report.exchange_order_id == "ex-001"
    assert report.symbol == "AAPL"
    assert report.side == OrderSide.BUY
    assert report.exec_type == OrderState.NEW
    assert report.last_quantity == 0.0
    assert report.last_price == 0.0
    assert report.cum_quantity == 0.0
    assert report.leaves_quantity == 100.0
    assert report.cum_quote_amount == 0.0
    assert report.average_price == 0.0
    assert report.fee == 0.0
    assert report.timestamp_ns == 1_000_000
    assert report.text == "Order placed"

    # Verify frozen immutability
    with pytest.raises(AttributeError):
        report.fee = 10.0  # type: ignore[misc]

    # Verify slot enforcement: no __dict__ exists and dynamic fields are blocked
    assert not hasattr(report, "__dict__")
    assert isinstance(ExecutionReport.__slots__, tuple)
    with pytest.raises((AttributeError, TypeError)):
        report.new_dynamic_field = "blocked"  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("field_name", "bad_val"),
    [
        ("last_quantity", -0.1),
        ("last_quantity", float("nan")),
        ("last_quantity", float("inf")),
        ("last_quantity", True),
        ("last_price", -10.0),
        ("last_price", float("nan")),
        ("last_price", False),
        ("cum_quantity", -1.0),
        ("cum_quantity", float("nan")),
        ("leaves_quantity", -0.5),
        ("leaves_quantity", float("nan")),
        ("cum_quote_amount", -100.0),
        ("cum_quote_amount", float("nan")),
        ("average_price", -1.0),
        ("average_price", float("nan")),
        ("fee", -0.01),
        ("fee", float("nan")),
        ("fee", True),
        ("timestamp_ns", -1),
        ("timestamp_ns", True),
        ("timestamp_ns", 1.5),
        ("report_id", ""),
        ("cl_ord_id", ""),
        ("exchange_order_id", 123),
        ("symbol", ""),
        ("text", 123),
        ("side", "BUY"),  # Must be OrderSide enum instance
        ("exec_type", "FILLED"),  # Must be OrderState enum instance
    ],
)
def test_execution_report_invalid_inputs_rejected(field_name: str, bad_val: object) -> None:
    """ExecutionReport must reject non-finite, negative, boolean, and mis-typed fields."""
    kwargs: dict[str, object] = {
        "report_id": "rep-001",
        "cl_ord_id": "ord-001",
        "exchange_order_id": "ex-001",
        "symbol": "AAPL",
        "side": OrderSide.BUY,
        "exec_type": OrderState.NEW,
        "last_quantity": 0.0,
        "last_price": 0.0,
        "cum_quantity": 0.0,
        "leaves_quantity": 100.0,
        "cum_quote_amount": 0.0,
        "average_price": 0.0,
        "fee": 0.0,
        "timestamp_ns": 1_000_000,
    }
    kwargs[field_name] = bad_val
    with pytest.raises(InvalidOrderInputException) as exc:
        ExecutionReport(**kwargs)  # type: ignore[arg-type]
    assert exc.value.code == ERR_GW_NON_FINITE_INPUT


def test_enum_members_and_string_equality() -> None:
    """Verify StrEnum values, string equality, and enum membership."""
    # OrderState
    assert OrderState.PENDING_NEW == "PENDING_NEW"
    assert OrderState.NEW == "NEW"
    assert OrderState.PARTIALLY_FILLED == "PARTIALLY_FILLED"
    assert OrderState.FILLED == "FILLED"
    assert OrderState.PENDING_CANCEL == "PENDING_CANCEL"
    assert OrderState.CANCELLED == "CANCELLED"
    assert OrderState.REJECTED == "REJECTED"
    assert OrderState.EXPIRED == "EXPIRED"

    # OrderSide
    assert OrderSide.BUY == "BUY"
    assert OrderSide.SELL == "SELL"

    # OrderType
    assert OrderType.MARKET == "MARKET"
    assert OrderType.LIMIT == "LIMIT"
    assert OrderType.STOP_LIMIT == "STOP_LIMIT"
    assert OrderType.PEGGED == "PEGGED"

    # TimeInForce
    assert TimeInForce.DAY == "DAY"
    assert TimeInForce.GTC == "GTC"
    assert TimeInForce.IOC == "IOC"
    assert TimeInForce.FOK == "FOK"


def test_exception_hierarchy_and_diagnostic_codes() -> None:
    """Verify all exceptions inherit from GatewayError and have correct default error codes."""
    # GatewayError base
    base_err = GatewayError("Base error", code="ERR-GW-999")
    assert isinstance(base_err, Exception)
    assert base_err.message == "Base error"
    assert base_err.code == "ERR-GW-999"

    # InvalidStateTransitionException
    inv_state = InvalidStateTransitionException("State error")
    assert isinstance(inv_state, GatewayError)
    assert inv_state.code == ERR_GW_INVALID_STATE_TRANSITION
    assert inv_state.code == "ERR-GW-001"

    # DuplicateOrderException
    dup_ord = DuplicateOrderException("Duplicate order")
    assert isinstance(dup_ord, GatewayError)
    assert dup_ord.code == ERR_GW_DUPLICATE_ORDER_ID
    assert dup_ord.code == "ERR-GW-002"

    # InvalidOrderInputException
    inv_input = InvalidOrderInputException("Invalid input")
    assert isinstance(inv_input, GatewayError)
    assert inv_input.code == ERR_GW_NON_FINITE_INPUT
    assert inv_input.code == "ERR-GW-003"

    # InsufficientMarginException
    insuf_margin = InsufficientMarginException("Insufficient margin")
    assert isinstance(insuf_margin, GatewayError)
    assert insuf_margin.code == ERR_GW_INSUFFICIENT_MARGIN
    assert insuf_margin.code == "ERR-GW-004"

    # GatewayDisconnectedException
    disc = GatewayDisconnectedException("Gateway disconnected")
    assert isinstance(disc, GatewayError)
    assert disc.code == ERR_GW_DISCONNECTED
    assert disc.code == "ERR-GW-005"

    # RateLimitExceededException
    rate_lim = RateLimitExceededException("Rate limit exceeded")
    assert isinstance(rate_lim, GatewayError)
    assert rate_lim.code == ERR_GW_RATE_LIMIT_EXCEEDED
    assert rate_lim.code == "ERR-GW-006"

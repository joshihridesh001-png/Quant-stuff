"""Unit tests for IdempotencyRouter and deterministic UUIDv5 order token routing.

Governing Standards:
- Rules.md (Rule 1: Line annotations; Rule 2: Diagnostic error codes; Rule 3: Quality gates; Rule 4: Mandatory adversarial red-teaming)
- INV-GW-002: Cryptographic Idempotency Token Uniqueness
- INV-GW-005: Strict Non-Finite Input & Boundary Protection
- INV-GW-006: Hot-Path Latency SLA (<= 0.05ms / 50us)
"""

from __future__ import annotations

import math
import time
import uuid

import pytest

from quant.execution.idempotency import (
    NAMESPACE_QUANT_ORDER,
    IdempotencyRouter,
)
from quant.execution.models import (
    ERR_GW_DUPLICATE_ORDER_ID,
    ERR_GW_NON_FINITE_INPUT,
    DuplicateOrderException,
    InvalidOrderInputException,
    Order,
    OrderSide,
    OrderState,
    OrderType,
    TimeInForce,
)


def _make_order(
    cl_ord_id: str,
    symbol: str = "AAPL",
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.LIMIT,
    quantity: float = 100.0,
    price: float = 150.0,
) -> Order:
    """Helper factory for creating valid Order instances."""
    return Order(
        cl_ord_id=cl_ord_id,
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=quantity,
        price=price,
        time_in_force=TimeInForce.GTC,
        state=OrderState.PENDING_NEW,
    )


# ============================================================================
# 1. Namespace & Constant Verification
# ============================================================================


def test_namespace_quant_order_constant() -> None:
    """Validate fixed UUIDv5 namespace constant matching specification."""
    expected_uuid = uuid.UUID("a7e1c0d4-1234-5678-9abc-def012345678")
    assert expected_uuid == NAMESPACE_QUANT_ORDER
    assert isinstance(NAMESPACE_QUANT_ORDER, uuid.UUID)


# ============================================================================
# 2. Initialization & Parameter Boundary Protection (INV-GW-005)
# ============================================================================


def test_router_initialization_defaults_and_properties() -> None:
    """Validate default parameter initialization and read-only properties."""
    router = IdempotencyRouter()
    assert router.history_capacity == 10_000
    assert router.ttl_seconds == 86400.0
    assert router.active_order_count == 0
    assert router.history_count == 0


def test_router_initialization_custom_parameters() -> None:
    """Validate custom parameter initialization within valid domain bounds."""
    router = IdempotencyRouter(history_capacity=500, ttl_seconds=3600.0)
    assert router.history_capacity == 500
    assert router.ttl_seconds == 3600.0


@pytest.mark.parametrize(
    "invalid_capacity",
    [
        0,
        -1,
        -10_000,
        True,
        False,
        100.5,
        "1000",
        None,
        math.nan,
        math.inf,
    ],
)
def test_router_init_invalid_history_capacity_rejected(invalid_capacity: object) -> None:
    """Reject non-positive, non-integer, boolean, or non-finite history_capacity."""
    with pytest.raises(InvalidOrderInputException) as exc:
        IdempotencyRouter(history_capacity=invalid_capacity)  # type: ignore[arg-type]
    assert exc.value.code == ERR_GW_NON_FINITE_INPUT


@pytest.mark.parametrize(
    "invalid_ttl",
    [
        0.0,
        0,
        -1.0,
        -86400.0,
        True,
        False,
        math.nan,
        math.inf,
        -math.inf,
        "86400",
        None,
    ],
)
def test_router_init_invalid_ttl_seconds_rejected(invalid_ttl: object) -> None:
    """Reject non-positive, boolean, or non-finite ttl_seconds."""
    with pytest.raises(InvalidOrderInputException) as exc:
        IdempotencyRouter(ttl_seconds=invalid_ttl)  # type: ignore[arg-type]
    assert exc.value.code == ERR_GW_NON_FINITE_INPUT


# ============================================================================
# 3. Deterministic UUIDv5 Token Generation (INV-GW-002)
# ============================================================================


def test_generate_client_order_id_deterministic() -> None:
    """Identical input parameters must yield bit-identical deterministic client_order_id."""
    router = IdempotencyRouter()
    ts = 1_700_000_000_123_456_789
    id1 = router.generate_client_order_id("MOMENTUM_V1", "AAPL", OrderSide.BUY, ts, nonce=0)
    id2 = router.generate_client_order_id("MOMENTUM_V1", "AAPL", OrderSide.BUY, ts, nonce=0)

    assert id1 == id2
    assert id1.startswith("cl-")
    # Verify standard UUID format after "cl-" prefix
    raw_uuid_str = id1[3:]
    parsed_uuid = uuid.UUID(raw_uuid_str)
    assert parsed_uuid.version == 5


def test_generate_client_order_id_distinctness() -> None:
    """Varying any single input parameter must synthesize a globally distinct UUIDv5 token."""
    router = IdempotencyRouter()
    ts = 1_700_000_000_000_000_000

    base = router.generate_client_order_id("STRAT_A", "AAPL", OrderSide.BUY, ts, nonce=0)
    diff_strat = router.generate_client_order_id("STRAT_B", "AAPL", OrderSide.BUY, ts, nonce=0)
    diff_symbol = router.generate_client_order_id("STRAT_A", "MSFT", OrderSide.BUY, ts, nonce=0)
    diff_side = router.generate_client_order_id("STRAT_A", "AAPL", OrderSide.SELL, ts, nonce=0)
    diff_ts = router.generate_client_order_id("STRAT_A", "AAPL", OrderSide.BUY, ts + 1, nonce=0)
    diff_nonce = router.generate_client_order_id("STRAT_A", "AAPL", OrderSide.BUY, ts, nonce=1)

    tokens = {base, diff_strat, diff_symbol, diff_side, diff_ts, diff_nonce}
    assert len(tokens) == 6, "All tokens must be distinct when varying any input vector dimension"


def test_generate_client_order_id_default_nonce() -> None:
    """Calling generate_client_order_id without nonce defaults to nonce=0."""
    router = IdempotencyRouter()
    ts = 1_700_000_000_000_000_000
    id_default = router.generate_client_order_id("STRAT", "AAPL", OrderSide.BUY, ts)
    id_explicit_zero = router.generate_client_order_id("STRAT", "AAPL", OrderSide.BUY, ts, nonce=0)
    assert id_default == id_explicit_zero


@pytest.mark.parametrize(
    ("strat", "symbol", "side", "ts", "nonce"),
    [
        ("", "AAPL", OrderSide.BUY, 1000, 0),
        ("   ", "AAPL", OrderSide.BUY, 1000, 0),
        (None, "AAPL", OrderSide.BUY, 1000, 0),
        (123, "AAPL", OrderSide.BUY, 1000, 0),
        (True, "AAPL", OrderSide.BUY, 1000, 0),
        ("STRAT", "", OrderSide.BUY, 1000, 0),
        ("STRAT", "   ", OrderSide.BUY, 1000, 0),
        ("STRAT", None, OrderSide.BUY, 1000, 0),
        ("STRAT", 456, OrderSide.BUY, 1000, 0),
        ("STRAT", "AAPL", "BUY", 1000, 0),
        ("STRAT", "AAPL", None, 1000, 0),
        ("STRAT", "AAPL", 1, 1000, 0),
        ("STRAT", "AAPL", OrderSide.BUY, -1, 0),
        ("STRAT", "AAPL", OrderSide.BUY, True, 0),
        ("STRAT", "AAPL", OrderSide.BUY, False, 0),
        ("STRAT", "AAPL", OrderSide.BUY, 1.5, 0),
        ("STRAT", "AAPL", OrderSide.BUY, "1000", 0),
        ("STRAT", "AAPL", OrderSide.BUY, 1000, -1),
        ("STRAT", "AAPL", OrderSide.BUY, 1000, True),
        ("STRAT", "AAPL", OrderSide.BUY, 1000, False),
        ("STRAT", "AAPL", OrderSide.BUY, 1000, 2.5),
        ("STRAT", "AAPL", OrderSide.BUY, 1000, "0"),
    ],
)
def test_generate_client_order_id_input_validation(
    strat: object,
    symbol: object,
    side: object,
    ts: object,
    nonce: object,
) -> None:
    """Validate strict rejection of invalid argument types and out-of-bound values."""
    router = IdempotencyRouter()
    with pytest.raises(InvalidOrderInputException) as exc:
        router.generate_client_order_id(
            strategy_id=strat,  # type: ignore[arg-type]
            symbol=symbol,  # type: ignore[arg-type]
            side=side,  # type: ignore[arg-type]
            timestamp_ns=ts,  # type: ignore[arg-type]
            nonce=nonce,  # type: ignore[arg-type]
        )
    assert exc.value.code == ERR_GW_NON_FINITE_INPUT


# ============================================================================
# 4. Order Registration, Lookup & Active Collision Rejection
# ============================================================================


def test_register_and_get_active_order() -> None:
    """Order registration populates active lookup and updates active count."""
    router = IdempotencyRouter()
    order = _make_order("cl-001")

    assert not router.is_duplicate("cl-001")
    assert router.get_active_order("cl-001") is None

    router.register_order(order)

    assert router.active_order_count == 1
    assert router.history_count == 0
    assert router.is_duplicate("cl-001")
    assert router.get_active_order("cl-001") is order


def test_register_order_duplicate_active_raises() -> None:
    """Re-submitting an identical cl_ord_id while order is active raises DuplicateOrderException."""
    router = IdempotencyRouter()
    order1 = _make_order("cl-dup-100")
    order2 = _make_order("cl-dup-100", symbol="MSFT", quantity=200.0)

    router.register_order(order1)
    with pytest.raises(DuplicateOrderException) as exc:
        router.register_order(order2)

    assert exc.value.code == ERR_GW_DUPLICATE_ORDER_ID
    assert "already registered" in exc.value.message
    # Active order remains intact
    assert router.get_active_order("cl-dup-100") is order1
    assert router.active_order_count == 1


@pytest.mark.parametrize("invalid_order", [None, "order-id", 123, True, False, object()])
def test_register_order_invalid_type_raises(invalid_order: object) -> None:
    """Registering non-Order instances raises InvalidOrderInputException."""
    router = IdempotencyRouter()
    with pytest.raises(InvalidOrderInputException) as exc:
        router.register_order(invalid_order)  # type: ignore[arg-type]
    assert exc.value.code == ERR_GW_NON_FINITE_INPUT


# ============================================================================
# 5. Order Deregistration & Historical Deduplication
# ============================================================================


def test_deregister_order_lifecycle() -> None:
    """Deregistering pops active order, adds token to historical ring buffer, and retains duplicate guard."""
    router = IdempotencyRouter(history_capacity=10)
    order = _make_order("cl-term-001")

    router.register_order(order)
    assert router.active_order_count == 1
    assert router.history_count == 0

    popped = router.deregister_order("cl-term-001")
    assert popped is order
    assert router.active_order_count == 0
    assert router.history_count == 1
    assert router.get_active_order("cl-term-001") is None

    # Even though inactive, historical buffer prevents re-submission
    assert router.is_duplicate("cl-term-001")
    with pytest.raises(DuplicateOrderException) as exc:
        router.register_order(order)
    assert exc.value.code == ERR_GW_DUPLICATE_ORDER_ID


def test_deregister_nonexistent_order_returns_none() -> None:
    """Deregistering an unknown order returns None without mutating history or active counts."""
    router = IdempotencyRouter()
    result = router.deregister_order("cl-unknown")
    assert result is None
    assert router.active_order_count == 0
    assert router.history_count == 0


def test_deregister_order_already_deregistered_returns_none() -> None:
    """Subsequent deregister calls for an already-deregistered order return None."""
    router = IdempotencyRouter()
    order = _make_order("cl-once")
    router.register_order(order)

    assert router.deregister_order("cl-once") is order
    assert router.deregister_order("cl-once") is None
    assert router.history_count == 1


@pytest.mark.parametrize("invalid_id", ["", "   ", None, 123, True, False, 3.14])
def test_deregister_order_invalid_id_raises(invalid_id: object) -> None:
    """Deregistering with non-string, empty, or boolean ID raises InvalidOrderInputException."""
    router = IdempotencyRouter()
    with pytest.raises(InvalidOrderInputException) as exc:
        router.deregister_order(invalid_id)  # type: ignore[arg-type]
    assert exc.value.code == ERR_GW_NON_FINITE_INPUT


# ============================================================================
# 6. Ring Buffer Eviction & Re-Registration (FIFO Semantics)
# ============================================================================


def test_ring_buffer_fifo_eviction_and_re_registration() -> None:
    """When historical ring buffer exceeds capacity, oldest entries are evicted from cache and set."""
    capacity = 3
    router = IdempotencyRouter(history_capacity=capacity)

    orders = [_make_order(f"cl-ring-{i}") for i in range(1, 5)]

    # Register and deregister first 3 orders
    for i in range(3):
        router.register_order(orders[i])
        router.deregister_order(orders[i].cl_ord_id)

    assert router.history_count == 3
    assert router.active_order_count == 0
    for i in range(3):
        assert router.is_duplicate(orders[i].cl_ord_id)

    # Register and deregister 4th order -> causes eviction of orders[0] ("cl-ring-1")
    router.register_order(orders[3])
    router.deregister_order(orders[3].cl_ord_id)

    assert router.history_count == 3
    # orders[0] was evicted!
    assert not router.is_duplicate(orders[0].cl_ord_id)
    # orders[1], orders[2], orders[3] remain protected
    assert router.is_duplicate(orders[1].cl_ord_id)
    assert router.is_duplicate(orders[2].cl_ord_id)
    assert router.is_duplicate(orders[3].cl_ord_id)

    # orders[0] can now be re-registered safely without collision error
    router.register_order(orders[0])
    assert router.get_active_order(orders[0].cl_ord_id) is orders[0]
    assert router.active_order_count == 1


def test_ring_buffer_large_scale_continuous_eviction() -> None:
    """Large scale eviction maintains exact capacity bound and set synchronization."""
    capacity = 50
    total_orders = 300
    router = IdempotencyRouter(history_capacity=capacity)

    for i in range(total_orders):
        ord_id = f"cl-seq-{i}"
        order = _make_order(ord_id)
        router.register_order(order)
        router.deregister_order(ord_id)

    assert router.history_count == capacity
    assert router.active_order_count == 0

    # Oldest orders (0 to 249) must be evicted
    for i in range(total_orders - capacity):
        assert not router.is_duplicate(f"cl-seq-{i}")

    # Recent orders (250 to 299) must be retained
    for i in range(total_orders - capacity, total_orders):
        assert router.is_duplicate(f"cl-seq-{i}")


# ============================================================================
# 7. Query Methods & Clear Protocol
# ============================================================================


@pytest.mark.parametrize("invalid_id", ["", "   ", None, 123, True, False, 3.14])
def test_get_active_order_invalid_id_raises(invalid_id: object) -> None:
    """get_active_order with non-string, empty, or boolean ID raises InvalidOrderInputException."""
    router = IdempotencyRouter()
    with pytest.raises(InvalidOrderInputException) as exc:
        router.get_active_order(invalid_id)  # type: ignore[arg-type]
    assert exc.value.code == ERR_GW_NON_FINITE_INPUT


@pytest.mark.parametrize("invalid_id", ["", "   ", None, 123, True, False, 3.14])
def test_is_duplicate_invalid_id_raises(invalid_id: object) -> None:
    """is_duplicate with non-string, empty, or boolean ID raises InvalidOrderInputException."""
    router = IdempotencyRouter()
    with pytest.raises(InvalidOrderInputException) as exc:
        router.is_duplicate(invalid_id)  # type: ignore[arg-type]
    assert exc.value.code == ERR_GW_NON_FINITE_INPUT


def test_clear_resets_all_state() -> None:
    """Clear empties both active order dictionary and historical ring buffer."""
    router = IdempotencyRouter(history_capacity=10)
    o1 = _make_order("cl-clear-1")
    o2 = _make_order("cl-clear-2")

    router.register_order(o1)
    router.register_order(o2)
    router.deregister_order("cl-clear-1")

    assert router.active_order_count == 1
    assert router.history_count == 1

    router.clear()

    assert router.active_order_count == 0
    assert router.history_count == 0
    assert not router.is_duplicate("cl-clear-1")
    assert not router.is_duplicate("cl-clear-2")
    assert router.get_active_order("cl-clear-2") is None


# ============================================================================
# 8. Hot-Path Execution Latency SLA Benchmark (INV-GW-006)
# ============================================================================


def test_hot_path_execution_latency_sla() -> None:
    """Token generation, lookup, and registration must complete in <= 0.05ms (50us) per INV-GW-006."""
    router = IdempotencyRouter(history_capacity=10_000)
    ts_base = 1_700_000_000_000_000_000

    # Warm-up CPU caches and Python bytecode
    for i in range(100):
        token = router.generate_client_order_id(
            "STRAT", "AAPL", OrderSide.BUY, ts_base + i, nonce=0
        )
        order = _make_order(token)
        router.register_order(order)
        router.is_duplicate(token)
        router.deregister_order(token)
    router.clear()

    iterations = 2_000

    # 1. Benchmark generate_client_order_id
    t0 = time.perf_counter()
    tokens = [
        router.generate_client_order_id("STRAT", "AAPL", OrderSide.BUY, ts_base + i, nonce=0)
        for i in range(iterations)
    ]
    t1 = time.perf_counter()
    avg_gen_latency_s = (t1 - t0) / iterations

    # 2. Benchmark register_order
    orders = [_make_order(tok) for tok in tokens]
    t0 = time.perf_counter()
    for o in orders:
        router.register_order(o)
    t1 = time.perf_counter()
    avg_reg_latency_s = (t1 - t0) / iterations

    # 3. Benchmark is_duplicate and get_active_order
    t0 = time.perf_counter()
    for tok in tokens:
        router.is_duplicate(tok)
        router.get_active_order(tok)
    t1 = time.perf_counter()
    avg_lookup_latency_s = (t1 - t0) / (iterations * 2)

    # 4. Benchmark deregister_order
    t0 = time.perf_counter()
    for tok in tokens:
        router.deregister_order(tok)
    t1 = time.perf_counter()
    avg_dereg_latency_s = (t1 - t0) / iterations

    # Invariant 6 SLA: <= 0.05ms (50 microseconds = 5e-5 seconds)
    sla_threshold_s = 5e-5
    assert avg_gen_latency_s < sla_threshold_s, (
        f"generate_client_order_id latency {avg_gen_latency_s * 1e6:.2f}us exceeds 50us SLA"
    )
    assert avg_reg_latency_s < sla_threshold_s, (
        f"register_order latency {avg_reg_latency_s * 1e6:.2f}us exceeds 50us SLA"
    )
    assert avg_lookup_latency_s < sla_threshold_s, (
        f"lookup latency {avg_lookup_latency_s * 1e6:.2f}us exceeds 50us SLA"
    )
    assert avg_dereg_latency_s < sla_threshold_s, (
        f"deregister_order latency {avg_dereg_latency_s * 1e6:.2f}us exceeds 50us SLA"
    )

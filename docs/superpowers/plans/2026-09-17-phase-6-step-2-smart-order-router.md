# Phase 6, Step 2: Microstructural Smart Order Router & Execution Algorithms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the institutional Smart Order Router (SOR) and algorithmic execution suite providing Poisson-jitter anti-gaming TWAP, volume-adaptive dynamic VWAP, non-linear 3/2-power Almgren-Chriss Arrival Price, dark pool midpoint probing with toxic markout detection, and closed-form algebraic KKT lit waterfilling.

**Architecture:** A decoupled execution subsystem where meta-order schedulers decompose parent orders into child slices, which the Smart Order Router probes through dark pools (for midpoint price improvement) and splits across lit exchanges via closed-form KKT waterfilling, with end-to-end Implementation Shortfall TCA attribution.

**Tech Stack:** Python 3.13 strict typing (`mypy --strict`), NumPy, Dataclasses (`slots=True`), Asyncio.

**Spec:** [`docs/superpowers/specs/2026-09-17-phase-6-step-2-smart-order-router-design.md`](file:///c:/Users/jishu/OneDrive/Documents/quant/docs/superpowers/specs/2026-09-17-phase-6-step-2-smart-order-router-design.md)

## Global Constraints

- Python 3.13 strict static typing (`mypy src --strict` with zero errors across all source files).
- Zero lint/format deviations (`ruff check .`, `ruff format --check .` enforcing 100-character line width standards).
- Test suite passing 100% with $\ge 90\%$ line coverage on all new execution files.
- Strict compliance with invariant contracts:
  - `INV-SOR-001`: Parent-Child Mass Conservation ($|\sum q_k - Q_{\text{parent}}| < 10^{-7}$).
  - `INV-SOR-002`: Non-Worse-Than-NBBO & Dark Midpoint Price Improvement Guarantee.
  - `INV-SOR-003`: Hard Volume Participation Rate Cap ($\rho \le 15\%$ of bar volume).
  - `INV-SOR-004`: Anti-Gaming Poisson Clock & Non-Negative Slices ($q_k \ge 0.0$).
  - `INV-SOR-005`: Causal Implementation Shortfall (IS) TCA Attribution.
  - `INV-SOR-006`: Hot-path routing latency SLA ($\tau_{\text{sor}} \le 0.10\text{ms}$ across $M=10$ venues).
- Rule 1 line-by-line annotation standards (Functional Purpose, Explicit Dependency Tracking, Structural Relationship, Defensive Invariant).
- Rule 2 diagnostic fault codes (`ERR-SOR-001` through `ERR-SOR-006`).
- Rule 4 anti-shortcut blacklist: zero black-box numerical optimizers (`scipy.optimize` strictly banned; closed-form algebraic KKT waterfilling required).

---

### Task 1: Venue Profiles, Consolidated Quotes & Microstructural Models

**Files:**
- Create: `src/quant/execution/venues.py`
- Test: `tests/unit/test_venues.py`

**Interfaces:**
- Consumes: Standard library (`enum`, `dataclasses`, `math`, `typing`).
- Produces:
  - Enums: `VenueType` (`LIT_EXCHANGE`, `DARK_POOL`).
  - Dataclasses (`slots=True`): `VenueProfile`, `ConsolidatedQuote`.
  - Fault codes: `ERR_SOR_NBBO_VIOLATION = "ERR-SOR-003"`, `ERR_SOR_INSUFFICIENT_LIQUIDITY = "ERR-SOR-002"`, `ERR_SOR_NON_FINITE_INPUT = "ERR-SOR-007"`.
  - Exceptions: `SORError`, `NBBOViolationException`, `InsufficientLiquidityException`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_venues.py
import pytest
from quant.execution.venues import (
    ERR_SOR_NBBO_VIOLATION,
    ConsolidatedQuote,
    NBBOViolationException,
    VenueProfile,
    VenueType,
)


def test_venue_profile_creation_and_slots() -> None:
    profile = VenueProfile(
        venue_id="NASDAQ",
        venue_type=VenueType.LIT_EXCHANGE,
        maker_fee_bps=-0.5,  # Maker rebate
        taker_fee_bps=1.5,
        min_order_size=1.0,
        lot_size=1.0,
    )
    assert profile.venue_id == "NASDAQ"
    assert profile.maker_fee_bps == -0.5
    assert not hasattr(profile, "__dict__")


def test_consolidated_quote_properties_and_nbbo_validation() -> None:
    quote = ConsolidatedQuote(
        symbol="AAPL",
        bid_price=150.00,
        bid_quantity=500.0,
        ask_price=150.05,
        ask_quantity=300.0,
        timestamp_ns=1_000_000,
        venue_depths={
            "NASDAQ": (300.0, 200.0),
            "NYSE": (200.0, 100.0),
        },
    )
    assert quote.midpoint == 150.025
    assert quote.spread == 0.05
    assert quote.order_book_imbalance == pytest.approx((500.0 - 300.0) / 800.0)


def test_crossed_locked_market_rejected() -> None:
    with pytest.raises(NBBOViolationException) as exc:
        ConsolidatedQuote(
            symbol="AAPL",
            bid_price=150.10,
            bid_quantity=100.0,
            ask_price=150.05,  # Crossed! Bid > Ask
            ask_quantity=100.0,
            timestamp_ns=1_000_000,
            venue_depths={},
        )
    assert exc.value.code == ERR_SOR_NBBO_VIOLATION
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_venues.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'quant.execution.venues'`

- [ ] **Step 3: Write minimal implementation**

Create `src/quant/execution/venues.py`:
Implement `VenueType`, `VenueProfile`, `ConsolidatedQuote`, exceptions, and fault codes. Enforce non-finite validation, reject bools, and guard crossed/locked NBBO (`bid_price >= ask_price` raises `NBBOViolationException`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_venues.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/quant/execution/venues.py tests/unit/test_venues.py
git commit -m "feat(execution): implement VenueProfile, ConsolidatedQuote, and NBBO invariants (Phase 6 Step 2 Task 1)"
```

---

### Task 2: Algorithmic Meta-Order Schedulers (Anti-Gaming TWAP, Adaptive VWAP, Non-Linear Arrival Price)

**Files:**
- Create: `src/quant/execution/algorithms.py`
- Test: `tests/unit/test_execution_algorithms.py`

**Interfaces:**
- Consumes: `src/quant/execution/models.py` (`OrderSide`), `src/quant/execution/venues.py`.
- Produces:
  - `ExecutionScheduler` protocol.
  - `ScheduledSlice` dataclass (`slice_index`, `quantity`, `scheduled_time_ns`, `price_limit`).
  - `PoissonTWAPScheduler`: anti-gaming Poisson clock timing, randomized jitter, exact mass conservation closure (`INV-SOR-001`, `INV-SOR-004`).
  - `VolumeAdaptiveVWAPScheduler`: diurnal historical profile blending, participation rate cap ($\le 15\%$, `INV-SOR-003`).
  - `NonlinearArrivalPriceScheduler`: closed-form hyperbolic Almgren-Chriss inventory trajectory under 3/2-power impact.
  - Fault codes: `ERR_SOR_INVALID_SCHEDULE = "ERR-SOR-001"`, `ERR_SOR_MASS_CONSERVATION_BREACH = "ERR-SOR-004"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_execution_algorithms.py
import pytest
from quant.execution.algorithms import (
    NonlinearArrivalPriceScheduler,
    PoissonTWAPScheduler,
    VolumeAdaptiveVWAPScheduler,
)


def test_poisson_twap_mass_conservation_and_jitter() -> None:
    scheduler = PoissonTWAPScheduler(
        total_quantity=1000.0,
        num_slices=10,
        mean_interval_sec=5.0,
        jitter_ratio=0.15,
        seed=42,
    )
    slices = scheduler.generate_schedule(start_time_ns=1_000_000_000)
    assert len(slices) == 10
    total_scheduled = sum(s.quantity for s in slices)
    assert total_scheduled == pytest.approx(1000.0, abs=1e-7)
    # Check that slices are strictly positive and non-uniform
    assert all(s.quantity > 0 for s in slices)
    assert len(set(s.quantity for s in slices)) > 1


def test_vwap_participation_rate_cap() -> None:
    historical_curve = [0.10] * 10
    scheduler = VolumeAdaptiveVWAPScheduler(
        total_quantity=1000.0,
        historical_volume_curve=historical_curve,
        max_participation_rate=0.15,  # 15% cap
    )
    # If bar volume is only 500, max allowed is 75, not 100
    slices = scheduler.generate_schedule(
        start_time_ns=0,
        expected_bar_volumes=[500.0] * 10,
    )
    for s in slices:
        assert s.quantity <= 75.0 + 1e-7


def test_nonlinear_arrival_price_hyperbolic_decay() -> None:
    scheduler = NonlinearArrivalPriceScheduler(
        total_quantity=1000.0,
        horizon_seconds=60.0,
        volatility=0.015,
        risk_aversion=1e-5,
        impact_coefficient=0.10,
        num_intervals=6,
    )
    trajectory = scheduler.compute_inventory_trajectory()
    assert len(trajectory) == 7
    assert trajectory[0] == pytest.approx(1000.0)
    assert trajectory[-1] == pytest.approx(0.0, abs=1e-7)
    # Inventory must monotonically decrease
    for i in range(len(trajectory) - 1):
        assert trajectory[i] >= trajectory[i + 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_execution_algorithms.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'quant.execution.algorithms'`

- [ ] **Step 3: Write minimal implementation**

Create `src/quant/execution/algorithms.py`:
Implement `ScheduledSlice`, `PoissonTWAPScheduler`, `VolumeAdaptiveVWAPScheduler`, and `NonlinearArrivalPriceScheduler`. Enforce mass conservation closure ($q_K \equiv Q_{\text{rem}}$), non-negative bounds, and Rule 1 annotations.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_execution_algorithms.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/quant/execution/algorithms.py tests/unit/test_execution_algorithms.py
git commit -m "feat(execution): implement Poisson TWAP, Adaptive VWAP, and Almgren-Chriss schedulers (Phase 6 Step 2 Task 2)"
```

---

### Task 3: Smart Order Router (SOR) Engine: Dark Probing, Toxic Markout Watchdog & Closed-Form KKT Waterfilling

**Files:**
- Create: `src/quant/execution/sor.py`
- Test: `tests/unit/test_smart_order_router.py`

**Interfaces:**
- Consumes: `src/quant/execution/venues.py`, `src/quant/execution/models.py`, `src/quant/execution/gateway.py`.
- Produces:
  - `SmartOrderRouter` class with methods `route_slice()`, `record_fill_markout()`, `get_venue_health()`.
  - `RoutedVenueOrder` dataclass.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_smart_order_router.py
import pytest
from quant.execution.models import OrderSide
from quant.execution.sor import SmartOrderRouter
from quant.execution.venues import ConsolidatedQuote, VenueProfile, VenueType


def test_dark_first_probing_and_midpoint_pricing() -> None:
    venues = {
        "DARK_1": VenueProfile("DARK_1", VenueType.DARK_POOL, maker_fee_bps=0.0, taker_fee_bps=0.5),
        "LIT_1": VenueProfile("LIT_1", VenueType.LIT_EXCHANGE, maker_fee_bps=-0.2, taker_fee_bps=1.0),
    }
    router = SmartOrderRouter(venues=venues)
    quote = ConsolidatedQuote(
        symbol="AAPL",
        bid_price=150.00,
        bid_quantity=1000.0,
        ask_price=150.10,
        ask_quantity=1000.0,
        timestamp_ns=1_000_000,
        venue_depths={"LIT_1": (1000.0, 1000.0)},
    )
    allocations = router.compute_routing_allocation(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100.0,
        quote=quote,
        enable_dark=True,
    )
    # Must probe dark pool first with midpoint limit price
    assert "DARK_1" in allocations
    assert allocations["DARK_1"].price == 150.05  # Exact midpoint price improvement


def test_closed_form_kkt_lit_waterfilling() -> None:
    venues = {
        "CHEAP_EXCHANGE": VenueProfile("CHEAP_EXCHANGE", VenueType.LIT_EXCHANGE, maker_fee_bps=0.0, taker_fee_bps=0.5),
        "PRICEY_EXCHANGE": VenueProfile("PRICEY_EXCHANGE", VenueType.LIT_EXCHANGE, maker_fee_bps=0.0, taker_fee_bps=2.0),
    }
    router = SmartOrderRouter(venues=venues)
    quote = ConsolidatedQuote(
        symbol="AAPL",
        bid_price=150.00,
        bid_quantity=1000.0,
        ask_price=150.10,
        ask_quantity=1000.0,
        timestamp_ns=1_000_000,
        venue_depths={
            "CHEAP_EXCHANGE": (500.0, 50.0),   # Only 50 available on cheap venue
            "PRICEY_EXCHANGE": (500.0, 500.0), # Ample depth on pricey venue
        },
    )
    allocations = router.compute_routing_allocation(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100.0,
        quote=quote,
        enable_dark=False,
    )
    # Should fill 50 on cheap exchange, remaining 50 on pricey exchange
    assert allocations["CHEAP_EXCHANGE"].quantity == 50.0
    assert allocations["PRICEY_EXCHANGE"].quantity == 50.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_smart_order_router.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'quant.execution.sor'`

- [ ] **Step 3: Write minimal implementation**

Create `src/quant/execution/sor.py`:
Implement `SmartOrderRouter`:
- Dark-first midpoint probing with Minimum Execution Size.
- Toxic markout watchdog tracking short-term price drift and down-weighting toxic venues.
- Closed-form algebraic KKT waterfilling over lit venues (sorting venues by marginal cost $\text{fee} + \text{spread}/2$, filling up to depth, then spilling over).
- Verified sub-0.10ms latency SLA (`INV-SOR-006`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_smart_order_router.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/quant/execution/sor.py tests/unit/test_smart_order_router.py
git commit -m "feat(execution): implement SmartOrderRouter with dark probing, toxic markout guard, and KKT waterfilling (Phase 6 Step 2 Task 3)"
```

---

### Task 4: Parent Order Coordination & Implementation Shortfall (IS) Attribution

**Files:**
- Create: `src/quant/execution/parent_order.py`
- Test: `tests/unit/test_parent_order.py`

**Interfaces:**
- Consumes: `src/quant/execution/models.py`, `src/quant/execution/algorithms.py`, `src/quant/execution/sor.py`.
- Produces:
  - `ParentOrder` entity.
  - `ImplementationShortfallReport` dataclass with additive cost breakdown: Delay Cost, Price Impact, Spread Slippage, Exchange Fees, Opportunity Cost.
  - Fault codes: `ERR_SOR_CHILD_ORDER_FAILED = "ERR-SOR-005"`, `ERR_SOR_ALGORITHM_TIMEOUT = "ERR-SOR-006"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_parent_order.py
import pytest
from quant.execution.models import OrderSide
from quant.execution.parent_order import ImplementationShortfallReport, ParentOrder


def test_parent_order_child_tracking_and_is_attribution() -> None:
    parent = ParentOrder(
        parent_id="parent-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        total_quantity=100.0,
        arrival_price=150.00,
        decision_price=150.02,
        max_duration_seconds=60.0,
    )
    # Add child execution fills
    parent.record_child_fill(child_id="c-1", quantity=50.0, price=150.10, fee=1.0)
    parent.record_child_fill(child_id="c-2", quantity=50.0, price=150.12, fee=1.0)

    assert parent.is_completed
    assert parent.filled_quantity == 100.0
    assert parent.average_execution_price == 150.11

    is_report = parent.compute_implementation_shortfall(terminal_price=150.15)
    assert isinstance(is_report, ImplementationShortfallReport)
    assert is_report.delay_cost == pytest.approx(100.0 * (150.02 - 150.00))
    assert is_report.price_impact == pytest.approx(100.0 * (150.11 - 150.02))
    assert is_report.fees_paid == 2.0
    assert is_report.opportunity_cost == 0.0  # Fully filled
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_parent_order.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'quant.execution.parent_order'`

- [ ] **Step 3: Write minimal implementation**

Create `src/quant/execution/parent_order.py`:
Implement `ParentOrder` and `ImplementationShortfallReport` adhering to `INV-SOR-005` additive cost decomposition.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_parent_order.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/quant/execution/parent_order.py tests/unit/test_parent_order.py
git commit -m "feat(execution): implement ParentOrder lifecycle and Implementation Shortfall attribution (Phase 6 Step 2 Task 4)"
```

---

### Task 5: Repository Synchronization, Exports & Canonical Documentation

**Files:**
- Modify: `src/quant/execution/__init__.py`
- Modify: `Memory.md`
- Modify: `Phases.md`
- Modify: `Architecture.md`
- Modify: `PRD.md`
- Modify: `README.md`

- [ ] **Step 1: Export all symbols in `src/quant/execution/__init__.py`**
- [ ] **Step 2: Run all quality gates**
```bash
pytest tests/unit
mypy src --strict
ruff check .
ruff format --check .
```
- [ ] **Step 3: Synchronize canonical documentation**
  - Register `ADR-023: Microstructural Smart Order Routing & Algorithmic Slicing` in `Memory.md`.
  - Register fault codes `ERR-SOR-001` through `ERR-SOR-007` in `Memory.md`.
  - Mark Phase 6 Step 2 as **100% COMPLETE** in `Phases.md`.
  - Add Section 4.21 in `Architecture.md`.
  - Update `PRD.md` and `README.md`.
- [ ] **Step 4: Commit and Push**
```bash
git add .
git commit -m "feat(execution): export Phase 6 Step 2 symbols and synchronize documentation (Phase 6 Step 2 Task 5)"
git push origin main
```

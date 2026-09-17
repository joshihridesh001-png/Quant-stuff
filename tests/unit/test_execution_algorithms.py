"""Unit test suite for algorithmic meta-order execution schedulers (Phase 6 Step 2 Task 2).

Purpose:
    Exhaustively stress-tests Poisson-jitter anti-gaming TWAP, volume-adaptive dynamic VWAP,
    closed-form hyperbolic Almgren-Chriss Arrival Price, ScheduledSlice domain invariants,
    mass conservation, participation rate caps, numerical overflow guards, and Rule 2 fault codes.

Dependencies:
    - pytest: Test execution and parametric assertions.
    - numpy: Array comparison and trajectory checks.
    - quant.execution.algorithms: ScheduledSlice, ExecutionScheduler, PoissonTWAPScheduler,
      VolumeAdaptiveVWAPScheduler, NonlinearArrivalPriceScheduler.
    - quant.execution.venues: ERR_SOR_INVALID_SCHEDULE, ERR_SOR_MASS_CONSERVATION_BREACH,
      ERR_SOR_NON_FINITE_INPUT, InvalidScheduleException, MassConservationException,
      NonFiniteInputException.

Invariants Enforced:
    - INV-SOR-001 (Mass Conservation): Slices sum to total target quantity within 1e-7 tolerance.
    - INV-SOR-003 (Hard Volume Participation Rate Cap): VWAP slices do not exceed rho_max * V_bar.
    - INV-SOR-004 (Anti-Gaming Clock & Sizing): Randomized timing/sizing with strictly positive slices.
    - Rule 1: Self-explicating test coverage with boundary validations.
    - Rule 2: Zero-execution diagnostic fault code verification.
    - Rule 4: Mandatory adversarial red-teaming (zero black-box optimizers, numerical overflow guards).
"""

from __future__ import annotations

import numpy as np
import pytest

from quant.execution.algorithms import (
    ExecutionScheduler,
    NonlinearArrivalPriceScheduler,
    PoissonTWAPScheduler,
    ScheduledSlice,
    VolumeAdaptiveVWAPScheduler,
)
from quant.execution.venues import (
    ERR_SOR_INVALID_SCHEDULE,
    ERR_SOR_NON_FINITE_INPUT,
    InvalidScheduleException,
    NonFiniteInputException,
)

# ============================================================================
# ScheduledSlice Dataclass Invariant Tests
# ============================================================================


def test_scheduled_slice_creation_and_slots() -> None:
    """Verify nominal creation of ScheduledSlice with default price_limit and slots."""
    # Functional Purpose: Confirm slotted memory layout and attribute storage.
    # Explicit Dependency Tracking: ScheduledSlice.
    # Structural Relationship: Primary output atomic slice of all meta-order schedulers.
    # Defensive Invariant: No __dict__ attribute; finite, valid inputs preserved.
    slice_obj = ScheduledSlice(
        slice_index=0,
        quantity=150.0,
        scheduled_time_ns=1_000_000_000,
        price_limit=105.50,
    )
    assert slice_obj.slice_index == 0
    assert slice_obj.quantity == 150.0
    assert slice_obj.scheduled_time_ns == 1_000_000_000
    assert slice_obj.price_limit == 105.50
    assert not hasattr(slice_obj, "__dict__")

    # Optional price_limit None
    slice_no_limit = ScheduledSlice(
        slice_index=1,
        quantity=50.0,
        scheduled_time_ns=2_000_000_000,
    )
    assert slice_no_limit.price_limit is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("slice_index", -1),
        ("slice_index", True),
        ("slice_index", 1.5),
        ("slice_index", "0"),
        ("quantity", 0.0),
        ("quantity", -10.0),
        ("quantity", True),
        ("quantity", False),
        ("quantity", float("nan")),
        ("quantity", float("inf")),
        ("scheduled_time_ns", -100),
        ("scheduled_time_ns", True),
        ("scheduled_time_ns", 1.5),
        ("price_limit", 0.0),
        ("price_limit", -5.0),
        ("price_limit", True),
        ("price_limit", float("nan")),
        ("price_limit", float("-inf")),
    ],
)
def test_scheduled_slice_rejections(field: str, value: object) -> None:
    """Verify ScheduledSlice rejects negative, boolean, and non-finite values."""
    kwargs: dict[str, object] = {
        "slice_index": 0,
        "quantity": 100.0,
        "scheduled_time_ns": 1_000_000,
        "price_limit": 100.0,
    }
    kwargs[field] = value

    with pytest.raises(NonFiniteInputException) as exc_info:
        ScheduledSlice(**kwargs)  # type: ignore[arg-type]
    assert exc_info.value.code == ERR_SOR_NON_FINITE_INPUT


# ============================================================================
# Protocol Conformance Tests
# ============================================================================


def test_execution_scheduler_protocol_conformance() -> None:
    """Verify all three schedulers conform to ExecutionScheduler protocol."""
    twap = PoissonTWAPScheduler(total_quantity=100.0, num_slices=5, mean_interval_sec=1.0)
    vwap = VolumeAdaptiveVWAPScheduler(
        total_quantity=100.0, historical_volume_curve=[1.0, 1.0, 1.0]
    )
    arrival = NonlinearArrivalPriceScheduler(
        total_quantity=100.0,
        horizon_seconds=10.0,
        volatility=0.02,
        risk_aversion=1e-4,
        impact_coefficient=0.1,
        num_intervals=5,
    )

    assert isinstance(twap, ExecutionScheduler)
    assert isinstance(vwap, ExecutionScheduler)
    assert isinstance(arrival, ExecutionScheduler)


# ============================================================================
# PoissonTWAPScheduler Tests
# ============================================================================


def test_poisson_twap_mass_conservation_and_jitter() -> None:
    """Verify Poisson TWAP schedule adheres to INV-SOR-001 mass conservation and anti-gaming jitter."""
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

    # Check strictly positive slices and non-uniform jittered sizes
    assert all(s.quantity > 0.0 for s in slices)
    assert len({s.quantity for s in slices}) > 1

    # Check timestamps: monotonically increasing and non-uniform intervals
    times = [s.scheduled_time_ns for s in slices]
    assert times[0] == 1_000_000_000
    for i in range(len(times) - 1):
        assert times[i + 1] > times[i]
        interval_sec = (times[i + 1] - times[i]) / 1e9
        assert interval_sec >= 0.1  # min_interval_sec


def test_poisson_twap_seed_reproducibility() -> None:
    """Verify identical seeds produce byte-for-byte identical schedules."""
    s1 = PoissonTWAPScheduler(
        total_quantity=500.0, num_slices=8, mean_interval_sec=2.0, seed=123
    ).generate_schedule(100)
    s2 = PoissonTWAPScheduler(
        total_quantity=500.0, num_slices=8, mean_interval_sec=2.0, seed=123
    ).generate_schedule(100)
    assert [s.quantity for s in s1] == [s.quantity for s in s2]
    assert [s.scheduled_time_ns for s in s1] == [s.scheduled_time_ns for s in s2]

    # Different seeds produce different schedules
    s3 = PoissonTWAPScheduler(
        total_quantity=500.0, num_slices=8, mean_interval_sec=2.0, seed=999
    ).generate_schedule(100)
    assert [s.quantity for s in s1] != [s.quantity for s in s3]


def test_poisson_twap_single_slice() -> None:
    """Single-slice TWAP must return exactly one slice with total quantity."""
    scheduler = PoissonTWAPScheduler(total_quantity=250.0, num_slices=1, mean_interval_sec=10.0)
    slices = scheduler.generate_schedule(start_time_ns=500_000)
    assert len(slices) == 1
    assert slices[0].slice_index == 0
    assert slices[0].quantity == pytest.approx(250.0)
    assert slices[0].scheduled_time_ns == 500_000


@pytest.mark.parametrize(
    "param,value",
    [
        ("total_quantity", 0.0),
        ("total_quantity", -10.0),
        ("total_quantity", True),
        ("total_quantity", float("nan")),
        ("total_quantity", float("inf")),
        ("num_slices", 0),
        ("num_slices", -5),
        ("num_slices", True),
        ("num_slices", 2.5),
        ("mean_interval_sec", 0.0),
        ("mean_interval_sec", -1.0),
        ("mean_interval_sec", True),
        ("mean_interval_sec", float("nan")),
        ("jitter_ratio", -0.01),
        ("jitter_ratio", 1.0),
        ("jitter_ratio", 1.5),
        ("jitter_ratio", True),
        ("time_jitter_ratio", -0.01),
        ("time_jitter_ratio", 1.0),
        ("time_jitter_ratio", float("nan")),
        ("min_interval_sec", 0.0),
        ("min_interval_sec", -0.5),
        ("min_interval_sec", True),
        ("seed", True),
        ("seed", 1.5),
    ],
)
def test_poisson_twap_invalid_parameters(param: str, value: object) -> None:
    """Verify PoissonTWAPScheduler rejects invalid domain constraints with ERR_SOR_INVALID_SCHEDULE."""
    base_kwargs: dict[str, object] = {
        "total_quantity": 100.0,
        "num_slices": 5,
        "mean_interval_sec": 1.0,
        "jitter_ratio": 0.15,
        "time_jitter_ratio": 0.20,
        "min_interval_sec": 0.1,
        "seed": 42,
    }
    base_kwargs[param] = value

    with pytest.raises(InvalidScheduleException) as exc_info:
        PoissonTWAPScheduler(**base_kwargs)  # type: ignore[arg-type]
    assert exc_info.value.code == ERR_SOR_INVALID_SCHEDULE


def test_poisson_twap_invalid_start_time() -> None:
    """Verify generate_schedule rejects invalid start_time_ns."""
    scheduler = PoissonTWAPScheduler(total_quantity=100.0, num_slices=3, mean_interval_sec=1.0)
    with pytest.raises(InvalidScheduleException) as exc1:
        scheduler.generate_schedule(-1)
    assert exc1.value.code == ERR_SOR_INVALID_SCHEDULE

    with pytest.raises(InvalidScheduleException) as exc2:
        scheduler.generate_schedule(True)  # type: ignore[arg-type]
    assert exc2.value.code == ERR_SOR_INVALID_SCHEDULE


# ============================================================================
# VolumeAdaptiveVWAPScheduler Tests
# ============================================================================


def test_vwap_participation_rate_cap() -> None:
    """Verify VWAP scheduler enforces hard volume participation cap INV-SOR-003."""
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
    assert len(slices) == 10
    for s in slices:
        assert s.quantity <= 75.0 + 1e-7
        assert s.quantity > 0.0


def test_vwap_historical_unconstrained_mass_conservation() -> None:
    """Verify VWAP without expected_bar_volumes adheres to exact mass conservation."""
    historical_curve = [100.0, 250.0, 400.0, 150.0, 100.0]
    scheduler = VolumeAdaptiveVWAPScheduler(
        total_quantity=1500.0,
        historical_volume_curve=historical_curve,
        interval_sec=30.0,
    )
    slices = scheduler.generate_schedule(start_time_ns=1_000_000_000)
    assert len(slices) == 5
    total_scheduled = sum(s.quantity for s in slices)
    assert total_scheduled == pytest.approx(1500.0, abs=1e-7)

    # Check timestamps spacing matches interval_sec
    assert slices[0].scheduled_time_ns == 1_000_000_000
    assert slices[1].scheduled_time_ns == 1_000_000_000 + 30 * 1_000_000_000


def test_vwap_online_realtime_volume_blending() -> None:
    """Verify realtime volume blending formula V_hat = omega * V_exp + (1 - omega) * V_rt."""
    scheduler = VolumeAdaptiveVWAPScheduler(
        total_quantity=1000.0,
        historical_volume_curve=[0.5, 0.5],
        max_participation_rate=0.10,
        smoothing_weight=0.60,
    )
    # V_exp = [1000, 1000], V_rt = [2000, 2000]
    # V_hat = 0.60 * 1000 + 0.40 * 2000 = 600 + 800 = 1400
    # 10% cap of 1400 = 140.0
    slices = scheduler.generate_schedule(
        start_time_ns=0,
        expected_bar_volumes=[1000.0, 1000.0],
        realtime_volume_estimates=[2000.0, 2000.0],
    )
    for s in slices:
        assert s.quantity == pytest.approx(140.0, abs=1e-7)


@pytest.mark.parametrize(
    "param,value",
    [
        ("total_quantity", 0.0),
        ("total_quantity", -50.0),
        ("total_quantity", True),
        ("total_quantity", float("nan")),
        ("historical_volume_curve", []),
        ("historical_volume_curve", [-1.0, 2.0]),
        ("historical_volume_curve", [0.0, 0.0]),
        ("historical_volume_curve", [float("nan"), 1.0]),
        ("historical_volume_curve", [True, 1.0]),
        ("max_participation_rate", 0.0),
        ("max_participation_rate", -0.1),
        ("max_participation_rate", 1.05),
        ("max_participation_rate", True),
        ("smoothing_weight", -0.1),
        ("smoothing_weight", 1.1),
        ("smoothing_weight", float("nan")),
        ("interval_sec", 0.0),
        ("interval_sec", -10.0),
        ("interval_sec", True),
    ],
)
def test_vwap_invalid_parameters(param: str, value: object) -> None:
    """Verify VolumeAdaptiveVWAPScheduler parameter boundary rejections."""
    base_kwargs: dict[str, object] = {
        "total_quantity": 500.0,
        "historical_volume_curve": [1.0, 2.0, 3.0],
        "max_participation_rate": 0.15,
        "smoothing_weight": 0.70,
        "interval_sec": 60.0,
    }
    base_kwargs[param] = value

    with pytest.raises(InvalidScheduleException) as exc_info:
        VolumeAdaptiveVWAPScheduler(**base_kwargs)  # type: ignore[arg-type]
    assert exc_info.value.code == ERR_SOR_INVALID_SCHEDULE


def test_vwap_invalid_generate_schedule_inputs() -> None:
    """Verify generate_schedule validates expected and realtime volume arrays."""
    scheduler = VolumeAdaptiveVWAPScheduler(
        total_quantity=500.0,
        historical_volume_curve=[1.0, 1.0],
    )
    # Mismatched length
    with pytest.raises(InvalidScheduleException):
        scheduler.generate_schedule(0, expected_bar_volumes=[100.0])

    # Negative volume
    with pytest.raises(InvalidScheduleException):
        scheduler.generate_schedule(0, expected_bar_volumes=[-10.0, 100.0])

    # Non-finite volume
    with pytest.raises(InvalidScheduleException):
        scheduler.generate_schedule(0, expected_bar_volumes=[float("nan"), 100.0])

    # Expected volume not a sequence
    with pytest.raises(InvalidScheduleException):
        scheduler.generate_schedule(0, expected_bar_volumes=123)  # type: ignore[arg-type]

    # Realtime volume not a sequence
    with pytest.raises(InvalidScheduleException):
        scheduler.generate_schedule(
            0,
            expected_bar_volumes=[100.0, 100.0],
            realtime_volume_estimates=123,  # type: ignore[arg-type]
        )

    # Realtime non-finite
    with pytest.raises(InvalidScheduleException):
        scheduler.generate_schedule(
            0, expected_bar_volumes=[100.0, 100.0], realtime_volume_estimates=[float("nan"), 100.0]
        )

    # Invalid start_time_ns for VWAP
    with pytest.raises(InvalidScheduleException):
        scheduler.generate_schedule(-1)
    with pytest.raises(InvalidScheduleException):
        scheduler.generate_schedule(True)  # type: ignore[arg-type]


# ============================================================================
# NonlinearArrivalPriceScheduler Tests
# ============================================================================


def test_nonlinear_arrival_price_hyperbolic_decay() -> None:
    """Verify closed-form Almgren-Chriss inventory trajectory and telescoping mass conservation."""
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

    # Generate schedule and verify mass conservation
    slices = scheduler.generate_schedule(start_time_ns=1_000_000_000)
    assert len(slices) == 6
    total_scheduled = sum(s.quantity for s in slices)
    assert total_scheduled == pytest.approx(1000.0, abs=1e-7)
    assert all(s.quantity > 0 for s in slices)

    # Telescoping property: q_j == x_{j-1} - x_j
    for j, s in enumerate(slices):
        expected_q = trajectory[j] - trajectory[j + 1]
        assert s.quantity == pytest.approx(expected_q, abs=1e-7)


def test_nonlinear_arrival_price_volatility_scaling() -> None:
    """Verify baseline_volatility scales kappa_t = kappa_0 * (sigma / sigma_baseline)."""
    s_base = NonlinearArrivalPriceScheduler(
        total_quantity=1000.0,
        horizon_seconds=60.0,
        volatility=0.02,
        risk_aversion=1e-4,
        impact_coefficient=0.1,
        num_intervals=10,
    )
    s_scaled = NonlinearArrivalPriceScheduler(
        total_quantity=1000.0,
        horizon_seconds=60.0,
        volatility=0.04,  # 2x higher volatility
        risk_aversion=1e-4,
        impact_coefficient=0.1,
        num_intervals=10,
        baseline_volatility=0.02,
    )
    # When volatility doubles, kappa_0 doubles (since sigma enters directly as sqrt(sigma^2)),
    # and urgency scaling multiplies by another (sigma / sigma_baseline) = 2x, total 4x kappa
    assert s_scaled.kappa > s_base.kappa

    # Higher urgency discharges inventory faster early on
    traj_base = s_base.compute_inventory_trajectory()
    traj_scaled = s_scaled.compute_inventory_trajectory()
    assert traj_scaled[1] < traj_base[1]


def test_nonlinear_arrival_price_linear_limit() -> None:
    """When risk_aversion is infinitesimally small (kappa*T < 1e-6), trajectory is linear TWAP."""
    scheduler = NonlinearArrivalPriceScheduler(
        total_quantity=1000.0,
        horizon_seconds=100.0,
        volatility=1e-6,
        risk_aversion=1e-12,  # kappa*T << 1e-6
        impact_coefficient=1.0,
        num_intervals=5,
    )
    assert scheduler.kappa * 100.0 < 1e-6
    traj = scheduler.compute_inventory_trajectory()
    # Linear trajectory: [1000, 800, 600, 400, 200, 0]
    expected = np.array([1000.0, 800.0, 600.0, 400.0, 200.0, 0.0])
    np.testing.assert_allclose(traj, expected, atol=1e-5)

    slices = scheduler.generate_schedule(0)
    assert len(slices) == 5
    for s in slices:
        assert s.quantity == pytest.approx(200.0, abs=1e-5)


def test_nonlinear_arrival_price_extreme_urgency_no_overflow() -> None:
    """Verify kappa*T > 50 uses exponential ratio formulation without math overflow."""
    scheduler = NonlinearArrivalPriceScheduler(
        total_quantity=1000.0,
        horizon_seconds=100.0,
        volatility=1.0,
        risk_aversion=100.0,  # Huge urgency: kappa*T >> 50
        impact_coefficient=0.01,
        num_intervals=10,
    )
    assert scheduler.kappa * 100.0 > 50.0

    # Must compute without OverflowError or NaN
    traj = scheduler.compute_inventory_trajectory()
    assert len(traj) == 11
    assert traj[0] == pytest.approx(1000.0)
    assert traj[-1] == pytest.approx(0.0)
    assert not np.isnan(traj).any()
    assert not np.isinf(traj).any()

    # Verify monotonicity
    for i in range(len(traj) - 1):
        assert traj[i] >= traj[i + 1]

    # Verify schedule mass conservation
    slices = scheduler.generate_schedule(0)
    total_scheduled = sum(s.quantity for s in slices)
    assert total_scheduled == pytest.approx(1000.0, abs=1e-7)


@pytest.mark.parametrize(
    "param,value",
    [
        ("total_quantity", 0.0),
        ("total_quantity", -10.0),
        ("total_quantity", True),
        ("horizon_seconds", 0.0),
        ("horizon_seconds", -5.0),
        ("horizon_seconds", True),
        ("volatility", 0.0),
        ("volatility", -0.05),
        ("volatility", True),
        ("risk_aversion", 0.0),
        ("risk_aversion", -1.0),
        ("risk_aversion", True),
        ("impact_coefficient", 0.0),
        ("impact_coefficient", -0.1),
        ("impact_coefficient", False),
        ("num_intervals", 0),
        ("num_intervals", -1),
        ("num_intervals", True),
        ("num_intervals", 4.5),
        ("baseline_volatility", 0.0),
        ("baseline_volatility", -0.01),
        ("baseline_volatility", True),
        ("baseline_volatility", float("nan")),
    ],
)
def test_nonlinear_arrival_price_invalid_parameters(param: str, value: object) -> None:
    """Verify NonlinearArrivalPriceScheduler rejects invalid parameters."""
    base_kwargs: dict[str, object] = {
        "total_quantity": 1000.0,
        "horizon_seconds": 60.0,
        "volatility": 0.02,
        "risk_aversion": 1e-4,
        "impact_coefficient": 0.1,
        "num_intervals": 5,
        "baseline_volatility": 0.02,
    }
    base_kwargs[param] = value

    with pytest.raises(InvalidScheduleException) as exc_info:
        NonlinearArrivalPriceScheduler(**base_kwargs)  # type: ignore[arg-type]
    assert exc_info.value.code == ERR_SOR_INVALID_SCHEDULE


def test_nonlinear_arrival_price_invalid_start_time() -> None:
    """Verify arrival price generate_schedule rejects negative or boolean start_time_ns."""
    scheduler = NonlinearArrivalPriceScheduler(
        total_quantity=100.0,
        horizon_seconds=10.0,
        volatility=0.02,
        risk_aversion=1e-4,
        impact_coefficient=0.1,
        num_intervals=5,
    )
    with pytest.raises(InvalidScheduleException):
        scheduler.generate_schedule(-1)
    with pytest.raises(InvalidScheduleException):
        scheduler.generate_schedule(True)  # type: ignore[arg-type]

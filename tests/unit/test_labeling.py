"""Unit tests for Dynamic Volatility Triple-Barrier Labeling Subsystem.

Purpose: Verifies mathematical invariants, causal volatility lagging, collision policies, and net payoffs.
Dependencies: pytest, numpy, quant.analytics.labeling, quant.domain.models.
"""

import numpy as np
import pytest

from quant.analytics.labeling import (
    BarrierLabel,
    BarrierTouchReason,
    DynamicTripleBarrierLabeler,
    PositionSide,
    TripleBarrierConfig,
    compute_parkinson_volatility,
)
from quant.domain.models import MarketDataBatch, Resolution

# =====================================================================
# 1. Tests for TripleBarrierConfig Invariants
# =====================================================================


def test_triple_barrier_config_defaults() -> None:
    """Default configuration must satisfy all invariants."""
    cfg = TripleBarrierConfig()
    assert cfg.profit_multiplier == 2.0
    assert cfg.stop_multiplier == 1.0
    assert cfg.horizon_bars == 60
    assert cfg.volatility_window == 20
    assert cfg.volatility_floor == 1e-4
    assert cfg.volatility_cap == 0.50
    assert cfg.pessimistic_collision is True
    assert cfg.execution_delay_bars == 1


def test_triple_barrier_config_invalid_multipliers() -> None:
    """Non-positive multipliers must raise ValueError."""
    with pytest.raises(ValueError, match="profit_multiplier must be strictly positive"):
        TripleBarrierConfig(profit_multiplier=0.0)

    with pytest.raises(ValueError, match="profit_multiplier must be strictly positive"):
        TripleBarrierConfig(profit_multiplier=-1.5)

    with pytest.raises(ValueError, match="stop_multiplier must be strictly positive"):
        TripleBarrierConfig(stop_multiplier=0.0)


def test_triple_barrier_config_invalid_bounds_and_windows() -> None:
    """Invalid horizon, window, or volatility bounds must raise ValueError."""
    with pytest.raises(ValueError, match="horizon_bars must be at least 1"):
        TripleBarrierConfig(horizon_bars=0)

    with pytest.raises(ValueError, match="volatility_window must be at least 2"):
        TripleBarrierConfig(volatility_window=1)

    with pytest.raises(ValueError, match="volatility_floor must be strictly positive"):
        TripleBarrierConfig(volatility_floor=0.0)

    with pytest.raises(ValueError, match="volatility_cap .* cannot be less than volatility_floor"):
        TripleBarrierConfig(volatility_floor=0.10, volatility_cap=0.05)


def test_triple_barrier_config_invalid_frictions_and_ticks() -> None:
    """Negative spread, fee, delay, or non-positive tick size must raise ValueError."""
    with pytest.raises(ValueError, match="spread_bps cannot be negative"):
        TripleBarrierConfig(spread_bps=-1.0)

    with pytest.raises(ValueError, match="fee_bps cannot be negative"):
        TripleBarrierConfig(fee_bps=-0.5)

    with pytest.raises(ValueError, match="execution_delay_bars cannot be negative"):
        TripleBarrierConfig(execution_delay_bars=-1)

    with pytest.raises(ValueError, match="tick_size must be strictly positive"):
        TripleBarrierConfig(tick_size=0.0)


# =====================================================================
# 2. Tests for compute_parkinson_volatility
# =====================================================================


def test_parkinson_volatility_length_mismatch() -> None:
    """Highs and Lows dimension mismatch must raise ValueError."""
    with pytest.raises(ValueError, match="must match Lows length"):
        compute_parkinson_volatility(np.ones(10), np.ones(5))


def test_parkinson_volatility_invalid_prices() -> None:
    """Non-positive prices or High < Low must raise ValueError."""
    with pytest.raises(ValueError, match="must be strictly positive"):
        compute_parkinson_volatility(np.array([10.0, 0.0]), np.array([9.0, 0.0]))

    with pytest.raises(ValueError, match="cannot be strictly less than Low"):
        compute_parkinson_volatility(np.array([10.0, 8.0]), np.array([11.0, 7.0]))


def test_parkinson_volatility_causal_lagging() -> None:
    """Volatility at index t must strictly use data up to index t-1."""
    # Synthetic flat series with an extreme volatility shock at index 25
    n = 50
    window = 10
    highs = np.full(n, 101.0)
    lows = np.full(n, 99.0)

    # Insert shock at index 25
    highs[25] = 150.0
    lows[25] = 50.0

    vol = compute_parkinson_volatility(highs, lows, window=window)

    # Index 25 itself must NOT reflect the shock at index 25 (causal lagging)
    # The shock at index 25 should first appear at index 26
    baseline_vol = vol[24]
    assert vol[25] == baseline_vol  # Shocks at t do not affect sigma_t
    assert vol[26] > baseline_vol  # First visible at t+1


def test_parkinson_volatility_magnitude() -> None:
    """Higher price range must produce higher realized volatility."""
    n = 30
    window = 5
    # Low-volatility series
    low_highs = np.full(n, 100.5)
    low_lows = np.full(n, 99.5)
    vol_low = compute_parkinson_volatility(low_highs, low_lows, window=window)

    # High-volatility series
    high_highs = np.full(n, 110.0)
    high_lows = np.full(n, 90.0)
    vol_high = compute_parkinson_volatility(high_highs, high_lows, window=window)

    assert vol_high[20] > vol_low[20]


# =====================================================================
# 3. Tests for DynamicTripleBarrierLabeler Execution
# =====================================================================


def _generate_synthetic_market_data(n: int = 150) -> MarketDataBatch:
    """Generate deterministic synthetic market data batch for labeling tests."""
    timestamps = np.arange(1_000_000, 1_000_000 + n * 60_000, 60_000, dtype=np.int64)
    closes = np.full(n, 100.0, dtype=np.float64)
    opens = np.full(n, 100.0, dtype=np.float64)
    highs = np.full(n, 100.2, dtype=np.float64)
    lows = np.full(n, 99.8, dtype=np.float64)
    volumes = np.full(n, 1000.0, dtype=np.float64)
    vwaps = np.full(n, 100.0, dtype=np.float64)

    return MarketDataBatch(
        asset_id="TEST-ASSET",
        resolution=Resolution.ONE_MINUTE,
        timestamps=timestamps,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        volumes=volumes,
        vwaps=vwaps,
    )


def test_labeler_short_series_rejected() -> None:
    """Series shorter than window + horizon + delay must be rejected."""
    batch = _generate_synthetic_market_data(n=40)
    cfg = TripleBarrierConfig(volatility_window=20, horizon_bars=30, execution_delay_bars=1)
    labeler = DynamicTripleBarrierLabeler(cfg)

    with pytest.raises(ValueError, match="Input series length .* is too short"):
        labeler.label_batch(batch)


def test_labeler_long_position_profit_barrier_hit() -> None:
    """Long trade hitting upper barrier must produce label +1 and UPPER reason."""
    batch = _generate_synthetic_market_data(n=100)
    # Event at index 25, execution at index 26
    # Let's spike high at index 28 to hit upper profit barrier
    batch.highs[28] = 200.0

    cfg = TripleBarrierConfig(
        volatility_window=10,
        horizon_bars=20,
        profit_multiplier=1.5,
        stop_multiplier=1.5,
        spread_bps=0.0,
        fee_bps=0.0,
    )
    labeler = DynamicTripleBarrierLabeler(cfg)
    labels = labeler.label_batch(batch, event_indices=[25], sides=[PositionSide.LONG])

    assert len(labels) == 1
    res: BarrierLabel = labels[0]
    assert res.label == 1
    assert res.touch_reason == BarrierTouchReason.UPPER
    assert res.exit_timestamp == batch.timestamps[28]
    assert res.realized_return > 0.0
    assert res.holding_period_bars == 2  # 28 - 26 = 2


def test_labeler_long_position_stop_barrier_hit() -> None:
    """Long trade hitting lower barrier must produce label -1 and LOWER reason."""
    batch = _generate_synthetic_market_data(n=100)
    # Event at index 25, execution at index 26
    # Let's crash low at index 27 to hit stop-loss
    batch.lows[27] = 50.0

    cfg = TripleBarrierConfig(
        volatility_window=10,
        horizon_bars=20,
        profit_multiplier=1.5,
        stop_multiplier=1.5,
        spread_bps=0.0,
        fee_bps=0.0,
    )
    labeler = DynamicTripleBarrierLabeler(cfg)
    labels = labeler.label_batch(batch, event_indices=[25], sides=[PositionSide.LONG])

    assert len(labels) == 1
    res = labels[0]
    assert res.label == -1
    assert res.touch_reason == BarrierTouchReason.LOWER
    assert res.exit_timestamp == batch.timestamps[27]
    assert res.realized_return < 0.0
    assert res.holding_period_bars == 1  # 27 - 26 = 1


def test_labeler_short_position_profit_and_stop() -> None:
    """Short trade: price drop is profit (+1); price rally is stop loss (-1)."""
    batch = _generate_synthetic_market_data(n=100)
    cfg = TripleBarrierConfig(
        volatility_window=10,
        horizon_bars=20,
        profit_multiplier=2.0,
        stop_multiplier=2.0,
        spread_bps=0.0,
        fee_bps=0.0,
    )
    labeler = DynamicTripleBarrierLabeler(cfg)

    # 1. Price drops -> Short profits (label = +1, touch LOWER)
    batch.lows[30] = 50.0
    labels_profit = labeler.label_batch(batch, event_indices=[25], sides=[PositionSide.SHORT])
    assert labels_profit[0].label == 1
    assert labels_profit[0].touch_reason == BarrierTouchReason.LOWER
    assert labels_profit[0].realized_return > 0.0

    # 2. Price rallies -> Short loses (label = -1, touch UPPER)
    batch.lows[30] = 99.8  # restore
    batch.highs[30] = 200.0  # rally
    labels_stop = labeler.label_batch(batch, event_indices=[25], sides=[PositionSide.SHORT])
    assert labels_stop[0].label == -1
    assert labels_stop[0].touch_reason == BarrierTouchReason.UPPER
    assert labels_stop[0].realized_return < 0.0


def test_labeler_vertical_barrier_expiration() -> None:
    """Neither barrier touched within horizon must trigger vertical timeout (label = 0)."""
    batch = _generate_synthetic_market_data(n=100)
    # Keep prices very flat to guarantee no barrier touch
    batch.opens[:] = 100.0
    batch.highs[:] = 100.01
    batch.lows[:] = 99.99
    batch.closes[:] = 100.0

    cfg = TripleBarrierConfig(
        volatility_window=10,
        horizon_bars=15,
        profit_multiplier=5.0,
        stop_multiplier=5.0,
        sign_expiration_returns=False,
    )
    labeler = DynamicTripleBarrierLabeler(cfg)
    labels = labeler.label_batch(batch, event_indices=[20], sides=[PositionSide.LONG])

    assert len(labels) == 1
    res = labels[0]
    assert res.label == 0
    assert res.touch_reason == BarrierTouchReason.VERTICAL
    assert res.holding_period_bars == 15
    assert res.exit_timestamp == batch.timestamps[20 + 1 + 15]


def test_labeler_pessimistic_collision_policy() -> None:
    """Dual intra-bar barrier breach must trigger stop-loss under pessimistic policy."""
    batch = _generate_synthetic_market_data(n=100)
    entry_idx = 25
    collision_bar = 28

    # Bar 28 has both huge High and tiny Low
    batch.highs[collision_bar] = 300.0
    batch.lows[collision_bar] = 10.0

    # Test with pessimistic_collision=True (default)
    cfg_pessimistic = TripleBarrierConfig(
        volatility_window=10,
        horizon_bars=20,
        pessimistic_collision=True,
    )
    labeler_pess = DynamicTripleBarrierLabeler(cfg_pessimistic)
    labels_pess = labeler_pess.label_batch(
        batch, event_indices=[entry_idx], sides=[PositionSide.LONG]
    )

    assert labels_pess[0].touch_reason == BarrierTouchReason.COLLISION_STOP
    assert labels_pess[0].label == -1

    # Test with pessimistic_collision=False
    cfg_optimistic = TripleBarrierConfig(
        volatility_window=10,
        horizon_bars=20,
        pessimistic_collision=False,
    )
    labeler_opt = DynamicTripleBarrierLabeler(cfg_optimistic)
    labels_opt = labeler_opt.label_batch(
        batch, event_indices=[entry_idx], sides=[PositionSide.LONG]
    )

    assert labels_opt[0].touch_reason == BarrierTouchReason.UPPER
    assert labels_opt[0].label == 1


def test_labeler_discontinuous_gap_honors_open_price() -> None:
    """Market opening gap beyond barrier must record open price as exit, not theoretical barrier."""
    batch = _generate_synthetic_market_data(n=100)
    entry_idx = 25
    gap_bar = 27

    # Upper barrier is around 102. Let's gap open to 115.0!
    batch.opens[gap_bar] = 115.0
    batch.highs[gap_bar] = 116.0
    batch.lows[gap_bar] = 114.0
    batch.closes[gap_bar] = 115.5

    cfg = TripleBarrierConfig(
        volatility_window=10,
        horizon_bars=20,
        profit_multiplier=2.0,
        stop_multiplier=2.0,
        spread_bps=0.0,
        fee_bps=0.0,
    )
    labeler = DynamicTripleBarrierLabeler(cfg)
    labels = labeler.label_batch(batch, event_indices=[entry_idx], sides=[PositionSide.LONG])

    assert labels[0].touch_reason == BarrierTouchReason.UPPER
    # Exit price MUST be the actual gap open (115.0), not the theoretical ~102 barrier
    assert labels[0].exit_price == 115.0


def test_labeler_friction_deduction() -> None:
    """Realized return must accurately deduct round-trip spread and fees."""
    batch = _generate_synthetic_market_data(n=100)
    batch.opens[:] = 100.0
    batch.highs[:] = 100.01
    batch.lows[:] = 99.99
    batch.closes[:] = 100.0

    # With 5 bps spread and 10 bps fee -> round-trip friction is 2 * 15 bps = 30 bps (0.0030)
    cfg = TripleBarrierConfig(
        volatility_window=10,
        horizon_bars=10,
        profit_multiplier=10.0,
        stop_multiplier=10.0,
        spread_bps=5.0,
        fee_bps=10.0,
    )
    labeler = DynamicTripleBarrierLabeler(cfg)
    labels = labeler.label_batch(batch, event_indices=[20], sides=[PositionSide.LONG])

    res = labels[0]
    expected_friction = 2.0 * (5.0 + 10.0) * 1e-4  # 0.0030
    assert abs(res.realized_return - (-expected_friction)) < 1e-6


def test_labeler_tick_size_quantization() -> None:
    """When tick_size is specified, barriers must be exact integer multiples of tick_size."""
    batch = _generate_synthetic_market_data(n=100)
    tick = 0.05
    cfg = TripleBarrierConfig(
        volatility_window=10,
        horizon_bars=10,
        tick_size=tick,
    )
    labeler = DynamicTripleBarrierLabeler(cfg)
    labels = labeler.label_batch(batch, event_indices=[25], sides=[PositionSide.LONG])

    res = labels[0]
    # Verify barriers are integer multiples of tick size within machine precision
    rem_upper = (res.upper_barrier / tick) % 1
    rem_lower = (res.lower_barrier / tick) % 1
    assert rem_upper < 1e-9 or abs(rem_upper - 1.0) < 1e-9
    assert rem_lower < 1e-9 or abs(rem_lower - 1.0) < 1e-9


def test_parkinson_volatility_window_less_than_two() -> None:
    """Window < 2 must raise ValueError."""
    with pytest.raises(ValueError, match="Rolling window must be >= 2"):
        compute_parkinson_volatility(np.ones(10), np.ones(10), window=1)


def test_labeler_dimension_mismatch() -> None:
    """Mismatched arrays in label_arrays must raise ValueError."""
    cfg = TripleBarrierConfig()
    labeler = DynamicTripleBarrierLabeler(cfg)
    with pytest.raises(ValueError, match="Dimension mismatch in input arrays"):
        labeler.label_arrays(
            timestamps=np.ones(10),
            opens=np.ones(9),
            highs=np.ones(10),
            lows=np.ones(10),
            closes=np.ones(10),
        )


def test_labeler_unsigned_position() -> None:
    """Unsigned position side must evaluate breakout magnitude."""
    batch = _generate_synthetic_market_data(n=100)
    batch.highs[28] = 110.0  # breakout
    cfg = TripleBarrierConfig(
        volatility_window=10,
        horizon_bars=20,
        spread_bps=0.0,
        fee_bps=0.0,
    )
    labeler = DynamicTripleBarrierLabeler(cfg)
    labels = labeler.label_batch(batch, event_indices=[25], sides=[PositionSide.UNSIGNED])
    assert labels[0].side == PositionSide.UNSIGNED
    assert labels[0].label == 1
    assert labels[0].realized_return > 0.0


def test_labeler_linear_barriers_and_sqrt_scaling() -> None:
    """Linear barrier calculation and sqrt horizon scaling must execute properly."""
    batch = _generate_synthetic_market_data(n=100)
    cfg = TripleBarrierConfig(
        volatility_window=10,
        horizon_bars=15,
        use_log_barriers=False,
        scale_by_sqrt_horizon=True,
    )
    labeler = DynamicTripleBarrierLabeler(cfg)
    labels_long = labeler.label_batch(batch, event_indices=[25], sides=[PositionSide.LONG])
    labels_short = labeler.label_batch(batch, event_indices=[25], sides=[PositionSide.SHORT])

    assert len(labels_long) == 1
    assert len(labels_short) == 1
    assert labels_long[0].upper_barrier > labels_long[0].lower_barrier
    assert labels_short[0].upper_barrier > labels_short[0].lower_barrier


def test_labeler_sign_expiration_returns() -> None:
    """sign_expiration_returns=True must set label to sign(net_return)."""
    batch = _generate_synthetic_market_data(n=100)
    # Drift slightly up without touching barriers
    batch.closes[26 + 10] = 100.10

    cfg = TripleBarrierConfig(
        volatility_window=10,
        horizon_bars=10,
        profit_multiplier=10.0,
        stop_multiplier=10.0,
        spread_bps=0.0,
        fee_bps=0.0,
        sign_expiration_returns=True,
    )
    labeler = DynamicTripleBarrierLabeler(cfg)
    labels = labeler.label_batch(batch, event_indices=[25], sides=[PositionSide.LONG])
    assert labels[0].touch_reason == BarrierTouchReason.VERTICAL
    assert labels[0].label == 1  # positive drift gives +1


def test_labeler_full_series_sweep_without_explicit_events() -> None:
    """Invoking label_batch without event_indices must sweep all valid bars."""
    batch = _generate_synthetic_market_data(n=100)
    cfg = TripleBarrierConfig(
        volatility_window=10,
        horizon_bars=15,
        execution_delay_bars=1,
    )
    labeler = DynamicTripleBarrierLabeler(cfg)
    labels = labeler.label_batch(batch)
    assert len(labels) > 0


def test_barrier_label_invariants() -> None:
    """BarrierLabel must reject exit_timestamp < entry_timestamp, invalid labels, negative holding bars."""
    with pytest.raises(ValueError, match="cannot be strictly greater than exit_timestamp"):
        BarrierLabel(
            event_timestamp=100,
            entry_timestamp=200,
            exit_timestamp=150,
            entry_price=100.0,
            exit_price=101.0,
            side=PositionSide.LONG,
            label=1,
            realized_return=0.01,
            touch_reason=BarrierTouchReason.UPPER,
            holding_period_bars=5,
            volatility_at_entry=0.01,
            upper_barrier=102.0,
            lower_barrier=98.0,
        )

    with pytest.raises(ValueError, match=r"label must be in \{-1, 0, 1\}"):
        BarrierLabel(
            event_timestamp=100,
            entry_timestamp=200,
            exit_timestamp=250,
            entry_price=100.0,
            exit_price=101.0,
            side=PositionSide.LONG,
            label=2,
            realized_return=0.01,
            touch_reason=BarrierTouchReason.UPPER,
            holding_period_bars=5,
            volatility_at_entry=0.01,
            upper_barrier=102.0,
            lower_barrier=98.0,
        )

    with pytest.raises(ValueError, match="holding_period_bars cannot be negative"):
        BarrierLabel(
            event_timestamp=100,
            entry_timestamp=200,
            exit_timestamp=250,
            entry_price=100.0,
            exit_price=101.0,
            side=PositionSide.LONG,
            label=1,
            realized_return=0.01,
            touch_reason=BarrierTouchReason.UPPER,
            holding_period_bars=-1,
            volatility_at_entry=0.01,
            upper_barrier=102.0,
            lower_barrier=98.0,
        )

"""Unit tests for Two-Stage Continuous-Payoff Kelly Meta-Labeling Subsystem.

Purpose: Verifies payoff-aware meta-label generation, trade concurrency tracking,
         regularized Platt probability calibration, and continuous Kelly bet sizing.
Dependencies: pytest, numpy, quant.analytics.meta_labeling, quant.analytics.labeling.
Relationship: Validates Step 5 against mathematical invariants and CI quality gates.
Invariants:
    - Negative expectancy trades always yield 0.0 allocation.
    - Sizing monotonically scales with conviction and payoff ratio.
    - Concurrency and duration penalties appropriately discount position size.
    - Calibrated probabilities are strictly bounded and monotonic.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant.analytics.labeling import BarrierLabel, BarrierTouchReason, PositionSide
from quant.analytics.meta_labeling import (
    ContinuousKellySizer,
    MetaLabel,
    MetaLabelConfig,
    ProbabilityCalibrator,
    TwoStageMetaLabeler,
)


def test_meta_label_config_validation() -> None:
    """Verify MetaLabelConfig default values and invariant constraint enforcement."""
    config = MetaLabelConfig()
    assert config.kelly_fraction == 0.50
    assert config.max_leverage == 1.0
    assert config.min_edge_hurdle == 0.0
    assert config.concurrency_penalty is True
    assert config.duration_discounting is True
    assert config.reference_duration_bars == 10.0
    assert config.brier_improvement_threshold == 0.0

    # Test invalid kelly_fraction
    with pytest.raises(ValueError, match="kelly_fraction must be in"):
        MetaLabelConfig(kelly_fraction=0.0)
    with pytest.raises(ValueError, match="kelly_fraction must be in"):
        MetaLabelConfig(kelly_fraction=1.5)

    # Test invalid max_leverage
    with pytest.raises(ValueError, match="max_leverage must be strictly positive"):
        MetaLabelConfig(max_leverage=0.0)

    # Test invalid min_edge_hurdle
    with pytest.raises(ValueError, match="min_edge_hurdle cannot be negative"):
        MetaLabelConfig(min_edge_hurdle=-0.1)

    # Test invalid reference_duration_bars
    with pytest.raises(ValueError, match="reference_duration_bars must be strictly positive"):
        MetaLabelConfig(reference_duration_bars=0.0)

    # Test invalid brier_improvement_threshold
    with pytest.raises(ValueError, match="brier_improvement_threshold cannot be negative"):
        MetaLabelConfig(brier_improvement_threshold=-0.05)


def test_meta_label_data_invariants() -> None:
    """Verify MetaLabel data container validation and boundary invariants."""
    valid_ml = MetaLabel(
        event_timestamp=1000,
        entry_timestamp=1000,
        exit_timestamp=2000,
        primary_direction=1,
        realized_payoff=0.02,
        meta_label=1,
        holding_period_bars=10,
        payoff_odds=2.0,
    )
    assert valid_ml.meta_label == 1
    assert valid_ml.realized_payoff == 0.02

    # Invalid primary direction
    with pytest.raises(ValueError, match="primary_direction must be -1 or 1"):
        MetaLabel(
            event_timestamp=1000,
            entry_timestamp=1000,
            exit_timestamp=2000,
            primary_direction=0,
            realized_payoff=0.02,
            meta_label=1,
            holding_period_bars=10,
            payoff_odds=2.0,
        )

    # Invalid meta_label value
    with pytest.raises(ValueError, match="meta_label must be 0 or 1"):
        MetaLabel(
            event_timestamp=1000,
            entry_timestamp=1000,
            exit_timestamp=2000,
            primary_direction=1,
            realized_payoff=0.02,
            meta_label=2,
            holding_period_bars=10,
            payoff_odds=2.0,
        )

    # Negative holding period
    with pytest.raises(ValueError, match="holding_period_bars cannot be negative"):
        MetaLabel(
            event_timestamp=1000,
            entry_timestamp=1000,
            exit_timestamp=2000,
            primary_direction=1,
            realized_payoff=0.02,
            meta_label=1,
            holding_period_bars=-1,
            payoff_odds=2.0,
        )

    # Non-positive payoff odds
    with pytest.raises(ValueError, match="payoff_odds must be strictly positive"):
        MetaLabel(
            event_timestamp=1000,
            entry_timestamp=1000,
            exit_timestamp=2000,
            primary_direction=1,
            realized_payoff=0.02,
            meta_label=1,
            holding_period_bars=10,
            payoff_odds=0.0,
        )


def test_generate_meta_labels_payoff_alignment() -> None:
    """Verify payoff alignment: Long/Short, Take-Profit/Stop-Loss, and profitable timeouts."""
    labels = [
        # 1. Long signal + Take Profit hit -> Win (meta_label = 1)
        BarrierLabel(
            event_timestamp=100,
            entry_timestamp=100,
            exit_timestamp=200,
            entry_price=100.0,
            exit_price=102.0,
            side=PositionSide.LONG,
            label=1,
            realized_return=0.02,
            touch_reason=BarrierTouchReason.UPPER,
            holding_period_bars=10,
            volatility_at_entry=0.01,
            upper_barrier=102.0,
            lower_barrier=99.0,
        ),
        # 2. Long signal + Stop Loss hit -> Loss (meta_label = 0)
        BarrierLabel(
            event_timestamp=200,
            entry_timestamp=200,
            exit_timestamp=250,
            entry_price=100.0,
            exit_price=99.0,
            side=PositionSide.LONG,
            label=-1,
            realized_return=-0.01,
            touch_reason=BarrierTouchReason.LOWER,
            holding_period_bars=5,
            volatility_at_entry=0.01,
            upper_barrier=102.0,
            lower_barrier=99.0,
        ),
        # 3. Short signal + Lower Barrier hit (Price dropped, short made profit) -> Win (meta_label = 1)
        BarrierLabel(
            event_timestamp=300,
            entry_timestamp=300,
            exit_timestamp=400,
            entry_price=100.0,
            exit_price=98.0,
            side=PositionSide.SHORT,
            label=1,
            realized_return=0.02,
            touch_reason=BarrierTouchReason.LOWER,
            holding_period_bars=10,
            volatility_at_entry=0.01,
            upper_barrier=101.0,
            lower_barrier=98.0,
        ),
        # 4. Short signal + Upper Barrier hit (Price rose, short lost) -> Loss (meta_label = 0)
        BarrierLabel(
            event_timestamp=400,
            entry_timestamp=400,
            exit_timestamp=450,
            entry_price=100.0,
            exit_price=101.0,
            side=PositionSide.SHORT,
            label=-1,
            realized_return=-0.01,
            touch_reason=BarrierTouchReason.UPPER,
            holding_period_bars=5,
            volatility_at_entry=0.01,
            upper_barrier=101.0,
            lower_barrier=98.0,
        ),
        # 5. Long signal + Vertical Timeout with net positive gain (+1.5%) -> Win (meta_label = 1)
        BarrierLabel(
            event_timestamp=500,
            entry_timestamp=500,
            exit_timestamp=600,
            entry_price=100.0,
            exit_price=101.5,
            side=PositionSide.LONG,
            label=0,
            realized_return=0.015,
            touch_reason=BarrierTouchReason.VERTICAL,
            holding_period_bars=60,
            volatility_at_entry=0.01,
            upper_barrier=102.0,
            lower_barrier=99.0,
        ),
        # 6. Long signal + Vertical Timeout with net negative return (-0.5%) -> Loss (meta_label = 0)
        BarrierLabel(
            event_timestamp=600,
            entry_timestamp=600,
            exit_timestamp=700,
            entry_price=100.0,
            exit_price=99.5,
            side=PositionSide.LONG,
            label=0,
            realized_return=-0.005,
            touch_reason=BarrierTouchReason.VERTICAL,
            holding_period_bars=60,
            volatility_at_entry=0.01,
            upper_barrier=102.0,
            lower_barrier=99.0,
        ),
    ]

    primary_signals = [1, 1, -1, -1, 1, 1]
    meta_labels = TwoStageMetaLabeler.generate_meta_labels(primary_signals, labels)

    assert len(meta_labels) == 6
    assert [ml.meta_label for ml in meta_labels] == [1, 0, 1, 0, 1, 0]

    # Verify payoff odds calculation: Upper distance = 2, Lower distance = 1 -> odds = 2.0
    assert meta_labels[0].payoff_odds == pytest.approx(2.0)

    # Test dimension mismatch error
    with pytest.raises(ValueError, match="Dimension mismatch"):
        TwoStageMetaLabeler.generate_meta_labels([1, 1], labels)

    # Test invalid signal value
    with pytest.raises(ValueError, match="Primary signal must be -1 or 1"):
        TwoStageMetaLabeler.generate_meta_labels([0] * 6, labels)


def test_compute_concurrency() -> None:
    """Verify calculation of concurrent active overlapping trades."""
    labels = [
        # Trade 0: [100, 500)
        BarrierLabel(
            event_timestamp=100,
            entry_timestamp=100,
            exit_timestamp=500,
            entry_price=100.0,
            exit_price=102.0,
            side=PositionSide.LONG,
            label=1,
            realized_return=0.02,
            touch_reason=BarrierTouchReason.UPPER,
            holding_period_bars=4,
            volatility_at_entry=0.01,
            upper_barrier=102.0,
            lower_barrier=99.0,
        ),
        # Trade 1: [200, 700) -> overlaps with Trade 0
        BarrierLabel(
            event_timestamp=200,
            entry_timestamp=200,
            exit_timestamp=700,
            entry_price=100.0,
            exit_price=102.0,
            side=PositionSide.LONG,
            label=1,
            realized_return=0.02,
            touch_reason=BarrierTouchReason.UPPER,
            holding_period_bars=5,
            volatility_at_entry=0.01,
            upper_barrier=102.0,
            lower_barrier=99.0,
        ),
        # Trade 2: [300, 800) -> overlaps with Trade 0 and Trade 1
        BarrierLabel(
            event_timestamp=300,
            entry_timestamp=300,
            exit_timestamp=800,
            entry_price=100.0,
            exit_price=102.0,
            side=PositionSide.LONG,
            label=1,
            realized_return=0.02,
            touch_reason=BarrierTouchReason.UPPER,
            holding_period_bars=5,
            volatility_at_entry=0.01,
            upper_barrier=102.0,
            lower_barrier=99.0,
        ),
        # Trade 3: [600, 900) -> Trade 0 is closed (500), but Trade 1 (700) and Trade 2 (800) active
        BarrierLabel(
            event_timestamp=600,
            entry_timestamp=600,
            exit_timestamp=900,
            entry_price=100.0,
            exit_price=102.0,
            side=PositionSide.LONG,
            label=1,
            realized_return=0.02,
            touch_reason=BarrierTouchReason.UPPER,
            holding_period_bars=3,
            volatility_at_entry=0.01,
            upper_barrier=102.0,
            lower_barrier=99.0,
        ),
    ]

    concurrencies = TwoStageMetaLabeler.compute_concurrency(labels)
    # Trade 0 at 100: only Trade 0 active -> 1
    # Trade 1 at 200: Trade 0 and 1 active -> 2
    # Trade 2 at 300: Trade 0, 1, 2 active -> 3
    # Trade 3 at 600: Trade 1, 2, 3 active (Trade 0 closed at 500) -> 3
    assert np.array_equal(concurrencies, np.array([1, 2, 3, 3]))

    # Empty labels test
    assert len(TwoStageMetaLabeler.compute_concurrency([])) == 0


def test_probability_calibrator_fitting_and_properties() -> None:
    """Verify regularized Platt probability calibrator fitting and validation."""
    calibrator = ProbabilityCalibrator(l2_penalty=1e-2)
    assert not calibrator.is_fitted

    with pytest.raises(RuntimeError, match="must be fitted"):
        _ = calibrator.params

    with pytest.raises(RuntimeError, match="must be fitted"):
        calibrator.predict_proba([0.5])

    # Test input validation
    with pytest.raises(ValueError, match="Dimension mismatch"):
        calibrator.fit([1.0, 2.0], [1])

    with pytest.raises(ValueError, match="Insufficient samples"):
        calibrator.fit([1.0, 2.0, 3.0], [1, 0, 1])

    with pytest.raises(ValueError, match="y_true must contain only binary values"):
        calibrator.fit([1.0, 2.0, 3.0, 4.0, 5.0], [0, 1, 2, 1, 0])

    # Fit on synthetic signal: scores linearly correlated with binary outcome
    np.random.seed(42)
    scores = np.linspace(-3.0, 3.0, 100)
    # True probability of 1 is sigmoid(scores)
    true_probs = 1.0 / (1.0 + np.exp(-scores))
    y_true = (np.random.rand(100) < true_probs).astype(int)

    calibrator.fit(scores, y_true)
    assert calibrator.is_fitted

    a, b = calibrator.params
    # A must be negative (higher score -> higher probability)
    assert a < 0.0

    # Verify predict_proba monotonicity
    predicted_probs = calibrator.predict_proba(scores)
    assert len(predicted_probs) == 100
    assert np.all(predicted_probs >= 0.0)
    assert np.all(predicted_probs <= 1.0)
    # Must be strictly increasing with score
    assert np.all(np.diff(predicted_probs) > 0.0)

    # Verify Brier score calculation
    brier = ProbabilityCalibrator.brier_score(y_true, predicted_probs)
    assert 0.0 < brier < 0.25

    # Verify validate_calibration
    cal_brier, base_brier, is_valid = calibrator.validate_calibration(scores, y_true)
    assert cal_brier < base_brier
    assert is_valid is True


def test_continuous_kelly_sizer_single_trade() -> None:
    """Verify ContinuousKellySizer mathematical invariants for single trade allocations."""
    sizer = ContinuousKellySizer(
        MetaLabelConfig(
            kelly_fraction=0.50,
            max_leverage=1.0,
            reference_duration_bars=10.0,
        )
    )

    # 1. Parameter boundaries
    with pytest.raises(ValueError, match="prob must be in"):
        sizer.compute_bet_size(prob=0.0, payoff_ratio=2.0)
    with pytest.raises(ValueError, match="prob must be in"):
        sizer.compute_bet_size(prob=1.0, payoff_ratio=2.0)
    with pytest.raises(ValueError, match="payoff_ratio must be strictly positive"):
        sizer.compute_bet_size(prob=0.6, payoff_ratio=0.0)
    with pytest.raises(ValueError, match="duration_bars cannot be negative"):
        sizer.compute_bet_size(prob=0.6, payoff_ratio=2.0, duration_bars=-1.0)
    with pytest.raises(ValueError, match="concurrency must be at least 1"):
        sizer.compute_bet_size(prob=0.6, payoff_ratio=2.0, concurrency=0)

    # 2. Negative expected value -> Size MUST be 0.0
    # For b=1.0 (1:1 odds), breakeven is 50%. At p=0.40, edge is negative -> size = 0.0
    size_neg = sizer.compute_bet_size(prob=0.40, payoff_ratio=1.0)
    assert size_neg == 0.0

    # 3. Positive expected value -> Size > 0.0
    # For b=2.0, breakeven is 33.3%. At p=0.40, edge is POSITIVE!
    # f* = (0.40 * 2.0 - 0.60) / 2.0 = (0.80 - 0.60) / 2.0 = 0.10
    # Half-Kelly (lambda=0.50) -> raw size = 0.05
    size_pos = sizer.compute_bet_size(prob=0.40, payoff_ratio=2.0, duration_bars=10.0)
    assert size_pos == pytest.approx(0.05)

    # 4. Monotonicity in win probability: higher prob -> higher size
    size_p50 = sizer.compute_bet_size(prob=0.50, payoff_ratio=2.0, duration_bars=10.0)
    size_p60 = sizer.compute_bet_size(prob=0.60, payoff_ratio=2.0, duration_bars=10.0)
    size_p70 = sizer.compute_bet_size(prob=0.70, payoff_ratio=2.0, duration_bars=10.0)
    assert 0.0 < size_p50 < size_p60 < size_p70

    # 5. Monotonicity in payoff odds: higher b -> higher size
    size_b1 = sizer.compute_bet_size(prob=0.60, payoff_ratio=1.0, duration_bars=10.0)
    size_b2 = sizer.compute_bet_size(prob=0.60, payoff_ratio=2.0, duration_bars=10.0)
    size_b3 = sizer.compute_bet_size(prob=0.60, payoff_ratio=3.0, duration_bars=10.0)
    assert size_b1 < size_b2 < size_b3

    # 6. Duration discounting: longer holding period -> lower allocation
    size_fast = sizer.compute_bet_size(prob=0.60, payoff_ratio=2.0, duration_bars=2.5)
    size_base = sizer.compute_bet_size(prob=0.60, payoff_ratio=2.0, duration_bars=10.0)
    size_slow = sizer.compute_bet_size(prob=0.60, payoff_ratio=2.0, duration_bars=40.0)
    assert size_fast > size_base > size_slow
    # Duration ratio 40 / 10 = 4 -> sqrt(4) = 2 -> size should be exactly halved
    assert size_slow == pytest.approx(size_base / 2.0)

    # 7. Concurrency throttling: 2 overlapping trades -> size halved
    size_c1 = sizer.compute_bet_size(prob=0.60, payoff_ratio=2.0, duration_bars=10.0, concurrency=1)
    size_c2 = sizer.compute_bet_size(prob=0.60, payoff_ratio=2.0, duration_bars=10.0, concurrency=2)
    assert size_c2 == pytest.approx(size_c1 / 2.0)

    # 8. Maximum leverage ceiling clamping
    huge_edge_size = sizer.compute_bet_size(prob=0.99, payoff_ratio=10.0, duration_bars=1.0)
    assert huge_edge_size <= 1.0


def test_continuous_kelly_sizer_batch() -> None:
    """Verify batch position sizing with direction signing and dimension validation."""
    sizer = ContinuousKellySizer()

    primary_signals = [1, -1, 0, 1]
    probs = [0.60, 0.60, 0.60, 0.30]  # last one has negative expectancy
    payoff_ratios = [2.0, 2.0, 2.0, 1.0]

    sizes = sizer.size_batch(primary_signals, probs, payoff_ratios)
    assert len(sizes) == 4

    # Long signal with positive edge -> positive allocation
    assert sizes[0] > 0.0
    # Short signal with positive edge -> negative allocation
    assert sizes[1] < 0.0
    assert abs(sizes[0]) == pytest.approx(abs(sizes[1]))
    # Neutral signal (0) -> exactly 0.0
    assert sizes[2] == 0.0
    # Long signal with negative edge (p=0.30, b=1.0) -> exactly 0.0
    assert sizes[3] == 0.0

    # Dimension mismatch
    with pytest.raises(ValueError, match="Dimension mismatch"):
        sizer.size_batch([1, -1], [0.6], [2.0])


def test_two_stage_meta_labeler_end_to_end() -> None:
    """Verify complete TwoStageMetaLabeler workflow consuming Step 3 labels."""
    np.random.seed(42)
    n = 50
    primary_signals = np.random.choice([1, -1], size=n)

    # Synthetic BarrierLabel objects
    barrier_labels: list[BarrierLabel] = []
    for i in range(n):
        sig = primary_signals[i]
        # Make ~60% of signals profitable
        is_win = np.random.rand() < 0.60
        ret = 0.02 if is_win else -0.01
        reason = BarrierTouchReason.UPPER if (sig == 1 and is_win) else BarrierTouchReason.LOWER
        barrier_labels.append(
            BarrierLabel(
                event_timestamp=i * 1000,
                entry_timestamp=i * 1000,
                exit_timestamp=(i + 5) * 1000,
                entry_price=100.0,
                exit_price=102.0 if is_win else 99.0,
                side=PositionSide.LONG if sig == 1 else PositionSide.SHORT,
                label=1 if is_win else -1,
                realized_return=ret,
                touch_reason=reason,
                holding_period_bars=5,
                volatility_at_entry=0.01,
                upper_barrier=102.0,
                lower_barrier=99.0,
            )
        )

    pipeline = TwoStageMetaLabeler()

    # 1. Generate ground truth meta labels
    meta_labels = pipeline.generate_meta_labels(primary_signals, barrier_labels)
    assert len(meta_labels) == n
    assert all(ml.meta_label in (0, 1) for ml in meta_labels)

    # 2. Concurrency calculation
    concurrencies = pipeline.compute_concurrency(barrier_labels)
    assert len(concurrencies) == n
    assert np.all(concurrencies >= 1)

    # 3. Fit probability calibrator on model margins
    # Synthetic model decision margins correlated with true meta labels
    z_true = np.array([ml.meta_label for ml in meta_labels])
    scores = np.where(z_true == 1, 1.5, -1.5) + np.random.normal(0, 0.5, n)

    calibrator = pipeline.fit_calibrator(scores, meta_labels)
    assert calibrator.is_fitted

    # 4. Predict sizes
    odds = np.array([ml.payoff_odds for ml in meta_labels])
    durations = np.array([ml.holding_period_bars for ml in meta_labels], dtype=float)

    sizes = pipeline.predict_sizes(
        primary_signals=primary_signals,
        scores=scores,
        payoff_ratios=odds,
        durations=durations,
        concurrencies=concurrencies,
    )

    assert len(sizes) == n
    assert np.all(np.abs(sizes) <= 1.0)

    # Verify signs match primary signals for active bets
    for i in range(n):
        if sizes[i] != 0.0:
            assert np.sign(sizes[i]) == primary_signals[i]

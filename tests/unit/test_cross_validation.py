"""Unit tests for Combinatorial Purged Cross-Validation (CPCV) Subsystem.

Purpose: Verifies combinatorial splitting, interval purging, post-test embargoing,
         starvation bounds, and out-of-sample path reconstruction.
Dependencies: pytest, numpy, quant.analytics.cross_validation, quant.analytics.labeling.
Relationship: Validates Step 4 against architectural invariants and CI quality gates.
Invariants:
    - Purged observations never appear in train sets.
    - Embargoed observations never appear in train sets.
    - Reconstructed paths are continuous and NaN-free.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant.analytics.cross_validation import (
    CombinatorialPurgedCV,
    CPCVConfig,
    PurgedSplit,
)
from quant.analytics.labeling import BarrierLabel, BarrierTouchReason, PositionSide


def test_cpcv_config_defaults_and_validation() -> None:
    """Verify CPCVConfig default values and invariant constraint checks."""
    config = CPCVConfig()
    assert config.n_splits == 6
    assert config.n_test_splits == 2
    assert config.embargo_pct == 0.01
    assert config.embargo_bars is None
    assert config.max_splits == 50
    assert config.min_train_ratio == 0.20
    assert not config.forward_chaining

    # Test invalid n_splits
    with pytest.raises(ValueError, match="n_splits must be at least 2"):
        CPCVConfig(n_splits=1)

    # Test invalid n_test_splits
    with pytest.raises(ValueError, match="n_test_splits must be in"):
        CPCVConfig(n_splits=6, n_test_splits=0)
    with pytest.raises(ValueError, match="n_test_splits must be in"):
        CPCVConfig(n_splits=6, n_test_splits=6)

    # Test invalid embargo_pct
    with pytest.raises(ValueError, match="embargo_pct must be in"):
        CPCVConfig(embargo_pct=-0.01)
    with pytest.raises(ValueError, match="embargo_pct must be in"):
        CPCVConfig(embargo_pct=1.0)

    # Test invalid embargo_bars
    with pytest.raises(ValueError, match="embargo_bars cannot be negative"):
        CPCVConfig(embargo_bars=-5)

    # Test invalid max_splits
    with pytest.raises(ValueError, match="max_splits must be at least 1"):
        CPCVConfig(max_splits=0)

    # Test invalid min_train_ratio
    with pytest.raises(ValueError, match="min_train_ratio must be in"):
        CPCVConfig(min_train_ratio=0.0)
    with pytest.raises(ValueError, match="min_train_ratio must be in"):
        CPCVConfig(min_train_ratio=1.0)


def test_partition_groups_exact_partitioning() -> None:
    """Verify that _partition_groups creates balanced, non-overlapping contiguous slices."""
    # 100 samples into 6 groups: sizes will be 17, 17, 17, 17, 16, 16
    groups = CombinatorialPurgedCV._partition_groups(100, 6)
    assert len(groups) == 6

    all_indices = np.concatenate(groups)
    assert len(all_indices) == 100
    assert np.array_equal(all_indices, np.arange(100))

    sizes = [len(g) for g in groups]
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == 100


def test_cpcv_split_counts_and_combinations() -> None:
    """Verify combinatorial fold generation matches C(N, k)."""
    # N=6, k=2 -> C(6, 2) = 15
    config = CPCVConfig(n_splits=6, n_test_splits=2, max_splits=None)
    cv = CombinatorialPurgedCV(config)

    dummy_data = np.arange(120)
    assert cv.get_n_splits() == 15

    splits = list(cv.split(dummy_data))
    assert len(splits) == 15

    # Verify each split has correct structure
    for idx, split in enumerate(splits):
        assert isinstance(split, PurgedSplit)
        assert split.split_idx == idx
        assert len(split.test_groups) == 2
        assert len(split.train_groups) == 4
        # Test and train indices must be completely disjoint
        intersection = np.intersect1d(split.train_indices, split.test_indices)
        assert len(intersection) == 0


def test_cpcv_max_splits_budget_bounding() -> None:
    """Verify that max_splits caps large combinatorial explosions deterministically."""
    # N=8, k=4 -> C(8, 4) = 70 combinations
    config = CPCVConfig(n_splits=8, n_test_splits=4, max_splits=20)
    cv = CombinatorialPurgedCV(config)

    dummy_data = np.arange(160)
    assert cv.get_n_splits() == 20

    splits = list(cv.split(dummy_data))
    assert len(splits) == 20


def test_cpcv_forward_chaining() -> None:
    """Verify forward-chaining mode strictly enforces past-to-future temporal causality."""
    config = CPCVConfig(n_splits=6, n_test_splits=2, forward_chaining=True, min_train_ratio=0.10)
    cv = CombinatorialPurgedCV(config)

    dummy_data = np.arange(120)
    splits = list(cv.split(dummy_data))
    # For N=6, k=2, forward chaining produces test blocks starting at 1, 2, 3, 4 -> 4 splits
    assert len(splits) == 4

    for split in splits:
        # Every training index must be strictly less than every test index
        assert np.max(split.train_indices) < np.min(split.test_indices)


def test_cpcv_scikit_learn_unpacking_and_indexing() -> None:
    """Verify PurgedSplit supports scikit-learn (train_idx, test_idx) 2-tuple protocol."""
    config = CPCVConfig(n_splits=4, n_test_splits=1)
    cv = CombinatorialPurgedCV(config)

    dummy_data = np.arange(40)
    splits = list(cv.split(dummy_data))
    assert len(splits) == 4

    for split in splits:
        # 2-tuple unpacking
        train_idx, test_idx = split
        assert np.array_equal(train_idx, split.train_indices)
        assert np.array_equal(test_idx, split.test_indices)

        # Indexing
        assert np.array_equal(split[0], split.train_indices)
        assert np.array_equal(split[1], split.test_indices)
        assert len(split) == 2

        with pytest.raises(IndexError):
            _ = split[2]


def test_cpcv_purging_mechanism() -> None:
    """Verify that trades whose holding period overlaps test intervals are purged from train."""
    # 4 blocks of 25 bars: [0..24], [25..49], [50..74], [75..99]
    config = CPCVConfig(n_splits=4, n_test_splits=1, embargo_pct=0.0, min_train_ratio=0.10)
    cv = CombinatorialPurgedCV(config)

    n_samples = 100
    pred_times = np.arange(n_samples)
    eval_times = pred_times.copy()

    # Create a trade at index 20 that extends into block 1 ([25..49])
    # Entry at 20, Exit at 30
    eval_times[20] = 30

    # Create a trade at index 10 that ends cleanly before block 1
    # Entry at 10, Exit at 15
    eval_times[10] = 15

    splits = list(cv.split(np.arange(n_samples), pred_times=pred_times, eval_times=eval_times))

    # Split where test_groups == (1,) -> test interval is [25, 49]
    split_block_1 = next(s for s in splits if s.test_groups == (1,))

    # Sample 20's interval [20, 30] overlaps [25, 49] -> MUST BE PURGED
    assert 20 in split_block_1.purged_indices
    assert 20 not in split_block_1.train_indices

    # Sample 10's interval [10, 15] does not overlap [25, 49] -> MUST BE RETAINED
    assert 10 not in split_block_1.purged_indices
    assert 10 in split_block_1.train_indices


def test_cpcv_embargo_mechanism() -> None:
    """Verify that training samples immediately following test intervals are embargoed."""
    # 4 blocks of 25 bars: [0..24], [25..49], [50..74], [75..99]
    # Set explicit embargo of 5 bars
    config = CPCVConfig(n_splits=4, n_test_splits=1, embargo_bars=5, min_train_ratio=0.10)
    cv = CombinatorialPurgedCV(config)

    n_samples = 100
    splits = list(cv.split(np.arange(n_samples)))

    # Look at split where test block is group 1: [25..49]
    split_1 = next(s for s in splits if s.test_groups == (1,))

    # Post-test window is (49, 49 + 5] = [50, 51, 52, 53, 54]
    expected_embargoed = np.array([50, 51, 52, 53, 54])
    for idx in expected_embargoed:
        assert idx in split_1.embargoed_indices
        assert idx not in split_1.train_indices

    # Sample 55 is beyond embargo -> must be retained in training
    assert 55 in split_1.train_indices
    assert 55 not in split_1.embargoed_indices


def test_cpcv_starvation_guard_rejection() -> None:
    """Verify that excessive purging/embargoing violating min_train_ratio raises ValueError."""
    # Require 80% retained data in train set, but test set alone takes 50% (2 out of 4 splits)
    config = CPCVConfig(n_splits=4, n_test_splits=2, min_train_ratio=0.80)
    cv = CombinatorialPurgedCV(config)

    with pytest.raises(ValueError, match="violated min_train_ratio"):
        list(cv.split(np.arange(100)))


def test_cpcv_input_validation_errors() -> None:
    """Verify input validation errors for improper sample sizes and causality violations."""
    config = CPCVConfig(n_splits=6, n_test_splits=2)
    cv = CombinatorialPurgedCV(config)

    # 1. Samples less than splits
    with pytest.raises(ValueError, match="cannot be smaller than n_splits"):
        list(cv.split(np.arange(4)))

    # 2. Incomplete times (pred_times given without eval_times)
    with pytest.raises(ValueError, match="eval_times must be provided if pred_times is provided"):
        list(cv.split(np.arange(20), pred_times=np.arange(20)))

    # 3. Dimension mismatch between pred_times and n_samples
    with pytest.raises(ValueError, match="Dimension mismatch"):
        list(cv.split(np.arange(20), pred_times=np.arange(10), eval_times=np.arange(10)))

    # 4. Temporal causality violation: pred_time > eval_time
    pred = np.arange(20)
    ev = pred.copy()
    ev[5] = 2  # pred[5] == 5 > ev[5] == 2
    with pytest.raises(ValueError, match="Temporal causality violated"):
        list(cv.split(np.arange(20), pred_times=pred, eval_times=ev))


def test_cpcv_path_reconstruction() -> None:
    """Verify reconstruction of continuous, NaN-free backtest paths."""
    # N=4, k=2 -> C(4, 2) = 6 folds. Number of paths = C(3, 1) = 3.
    config = CPCVConfig(n_splits=4, n_test_splits=2, max_splits=None)
    cv = CombinatorialPurgedCV(config)

    n_samples = 40  # 4 groups of 10 samples
    splits = list(cv.split(np.arange(n_samples)))
    assert len(splits) == 6

    # Synthesize predictions for each fold:
    # Set each prediction equal to the fold index float
    fold_predictions = [
        np.full(len(split.test_indices), float(split.split_idx)) for split in splits
    ]

    paths = cv.reconstruct_paths(fold_predictions, n_samples)

    # Must produce shape (n_paths=3, n_samples=40)
    assert paths.shape == (3, 40)
    # Zero NaNs
    assert not np.any(np.isnan(paths))

    # Test error handling on fold prediction count mismatch
    with pytest.raises(ValueError, match="Fold count mismatch"):
        cv.reconstruct_paths(fold_predictions[:4], n_samples)

    # Test error handling on fold prediction length mismatch
    bad_preds = [fold_predictions[0][:-1]] + fold_predictions[1:]
    with pytest.raises(ValueError, match="prediction length"):
        cv.reconstruct_paths(bad_preds, n_samples)


def test_cpcv_path_sharpe_distribution() -> None:
    """Verify compute_path_sharpe_distribution calculates empirical Sharpe vector and variance."""
    # 3 paths of 100 bars
    np.random.seed(42)
    paths = np.ones((3, 100), dtype=float)
    # Make path 1 neutral (0), path 2 positive (1), path 3 negative (-1)
    paths[0] = 1.0
    paths[1] = 0.0
    paths[2] = -1.0

    noise = np.random.normal(0.0, 0.005, 100)
    returns = 0.01 + noise  # strictly positive mean returns (mean ≈ 0.01)

    sr_dist, mean_sr, var_sr = CombinatorialPurgedCV.compute_path_sharpe_distribution(
        paths, returns, annualization_factor=252.0
    )

    assert len(sr_dist) == 3
    assert sr_dist[0] > 0.0  # long positive drift
    assert sr_dist[1] == 0.0  # zero signal
    assert sr_dist[2] < 0.0  # short positive drift
    assert var_sr > 0.0
    assert mean_sr == pytest.approx(float(np.mean(sr_dist)))

    # Verify dimension mismatch error
    with pytest.raises(ValueError, match="Dimension mismatch"):
        CombinatorialPurgedCV.compute_path_sharpe_distribution(paths, returns[:50])


def test_cpcv_with_barrier_labels() -> None:
    """Verify integration consuming BarrierLabel sequences from Step 3."""
    config = CPCVConfig(n_splits=4, n_test_splits=1, embargo_pct=0.0)
    cv = CombinatorialPurgedCV(config)

    n_samples = 40
    # Construct synthetic BarrierLabel objects
    labels: list[BarrierLabel] = []
    for i in range(n_samples):
        # Even samples exit at i+5; odd samples exit at i
        exit_idx = i + 5 if i % 2 == 0 else i
        labels.append(
            BarrierLabel(
                event_timestamp=i * 1000,
                entry_timestamp=i * 1000,
                exit_timestamp=exit_idx * 1000,
                entry_price=100.0,
                exit_price=101.0,
                side=PositionSide.LONG,
                label=1,
                realized_return=0.01,
                touch_reason=BarrierTouchReason.UPPER,
                holding_period_bars=exit_idx - i,
                volatility_at_entry=0.015,
                upper_barrier=102.0,
                lower_barrier=98.0,
            )
        )

    splits = list(cv.split(np.arange(n_samples), pred_times=labels, eval_times=labels))
    assert len(splits) == 4

    # Group 1 is test: [10..19] (timestamps 10000..19000)
    split_1 = next(s for s in splits if s.test_groups == (1,))

    # Sample 8 has entry 8000 and exit 13000 (overlaps [10000..19000]) -> Purged!
    assert 8 in split_1.purged_indices
    assert 8 not in split_1.train_indices

    # Sample 9 has entry 9000 and exit 9000 (before 10000) -> Retained!
    assert 9 in split_1.train_indices
    assert 9 not in split_1.purged_indices

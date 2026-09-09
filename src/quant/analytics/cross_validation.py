"""Combinatorial Purged Cross-Validation (CPCV) Subsystem.

Purpose: Implements leakage-free combinatorial cross-validation with interval purging,
         adaptive autoregressive embargoing, and continuous out-of-sample path reconstruction.
Dependencies: numpy, dataclasses, itertools, math, logging.
Relationship: Consumes BarrierLabel from Step 3 (or temporal timestamps); feeds Meta-Labeling (Step 5)
              and supplies empirical Sharpe variance V[{SR_k}] to Deflated Sharpe Ratio (Step 6).
Invariants: Train-test temporal separation; exact interval intersection purging;
            strict post-test embargo buffer; zero lookahead bias.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Any

import numpy as np

# Structured application logger
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CPCVConfig:
    """Hyperparameter configuration for Combinatorial Purged Cross-Validation.

    Purpose: Encapsulates partition dimensions, embargo bounds, computational limits,
             and starvation protections.
    Invariants:
        - n_splits >= 2
        - 1 <= n_test_splits < n_splits
        - 0.0 <= embargo_pct < 1.0
        - embargo_bars is None or embargo_bars >= 0
        - max_splits is None or max_splits >= 1
        - 0.0 < min_train_ratio < 1.0
    """

    # Total number of contiguous chronological group blocks (N)
    n_splits: int = 6

    # Number of group blocks allocated to the test set per fold (k)
    n_test_splits: int = 2

    # Fraction of total sample length to embargo immediately post-test (h_embargo)
    embargo_pct: float = 0.01

    # Explicit integer count of bars to embargo post-test (overrides embargo_pct if set)
    embargo_bars: int | None = None

    # Computational budget cap on total combinatorial splits to prevent factorial explosion
    max_splits: int | None = 50

    # Minimum fraction of total observations required in train set after purging and embargoing
    min_train_ratio: float = 0.20

    # Enforce strict forward causality (test blocks must strictly succeed train blocks)
    forward_chaining: bool = False

    def __post_init__(self) -> None:
        """Validate configuration hyperparameters and boundary invariants."""
        if self.n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {self.n_splits}")
        if self.n_test_splits < 1 or self.n_test_splits >= self.n_splits:
            raise ValueError(
                f"n_test_splits must be in [1, {self.n_splits - 1}], got {self.n_test_splits}"
            )
        if self.embargo_pct < 0.0 or self.embargo_pct >= 1.0:
            raise ValueError(f"embargo_pct must be in [0.0, 1.0), got {self.embargo_pct}")
        if self.embargo_bars is not None and self.embargo_bars < 0:
            raise ValueError(f"embargo_bars cannot be negative, got {self.embargo_bars}")
        if self.max_splits is not None and self.max_splits < 1:
            raise ValueError(f"max_splits must be at least 1, got {self.max_splits}")
        if self.min_train_ratio <= 0.0 or self.min_train_ratio >= 1.0:
            raise ValueError(f"min_train_ratio must be in (0.0, 1.0), got {self.min_train_ratio}")


@dataclass(frozen=True)
class PurgedSplit:
    """Container for a single Combinatorial Purged Cross-Validation fold.

    Purpose: Holds partitioned training and testing index vectors alongside comprehensive
             audit indices for purged and embargoed observations.
    Invariants:
        - train_indices and test_indices are mutually disjoint.
        - purged_indices and embargoed_indices are mutually disjoint.
        - Supports 2-tuple unpacking (train_idx, test_idx = split) for scikit-learn compliance.
    """

    # Index of this combinatorial fold (0-indexed)
    split_idx: int

    # Retained training observation indices after purging and embargoing
    train_indices: np.ndarray

    # Test observation indices for this fold
    test_indices: np.ndarray

    # Indices dropped from training due to label holding period overlap with test intervals
    purged_indices: np.ndarray

    # Indices dropped from training due to the post-test embargo window
    embargoed_indices: np.ndarray

    # Group block indices assigned to the test set for this fold
    test_groups: tuple[int, ...]

    # Group block indices assigned to the candidate train set before purge/embargo
    train_groups: tuple[int, ...]

    def __iter__(self) -> Iterator[np.ndarray]:
        """Support 2-tuple unpacking: train_idx, test_idx = split."""
        yield self.train_indices
        yield self.test_indices

    def __getitem__(self, index: int) -> np.ndarray:
        """Support indexing: split[0] -> train_indices, split[1] -> test_indices."""
        if index == 0:
            return self.train_indices
        if index == 1:
            return self.test_indices
        raise IndexError(f"PurgedSplit index out of range: {index} (valid indices are 0 and 1)")

    def __len__(self) -> int:
        """Return length 2 to match 2-tuple protocol."""
        return 2


class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation (CPCV) Engine.

    Purpose: Partitions time-series data into N contiguous chronological blocks and generates
             combinatorial splits while purging overlapping trade lifespans and embargoing
             autoregressive post-test buffers. Reconstructs continuous out-of-sample backtest paths.
    Dependencies: CPCVConfig, PurgedSplit, numpy, math.comb, itertools.combinations.
    Relationship: Intermediates between Step 3 (TripleBarrierLabeler) and Step 5 (Meta-Labeling);
                  supplies backtest path Sharpe variance to Step 6 (Deflated Sharpe Ratio).
    Invariants:
        - No sample in train_indices has a trade lifespan overlapping any test block.
        - No sample in train_indices falls within the post-test embargo window.
        - Reconstructed paths cover all N group blocks without gaps or NaNs.
    """

    def __init__(self, config: CPCVConfig | None = None) -> None:
        """Initialize the CPCV engine with configuration.

        Args:
            config: CPCVConfig instance. If None, default hyperparameters are used.
        """
        self._config = config if config is not None else CPCVConfig()

    @property
    def config(self) -> CPCVConfig:
        """Return the immutable configuration object."""
        return self._config

    @staticmethod
    def _partition_groups(n_samples: int, n_splits: int) -> list[np.ndarray]:
        """Divide n_samples into n_splits contiguous chronological group blocks.

        Purpose: Establishes balanced, non-overlapping contiguous time blocks.
        Invariants:
            - Sum of block lengths == n_samples.
            - Blocks differ in size by at most 1 sample.
        """
        group_size, remainder = divmod(n_samples, n_splits)
        groups: list[np.ndarray] = []
        cursor = 0
        for i in range(n_splits):
            sz = group_size + (1 if i < remainder else 0)
            groups.append(np.arange(cursor, cursor + sz, dtype=np.int64))
            cursor += sz
        return groups

    def _generate_test_combinations(self) -> list[tuple[int, ...]]:
        """Generate test group combinations according to config parameters.

        Purpose: Produces combinatorial combinations or forward-chained splits with budget capping.
        Invariants:
            - If forward_chaining=True, test blocks strictly succeed train blocks.
            - Returned list length <= max_splits (if max_splits is configured).
        """
        n = self._config.n_splits
        k = self._config.n_test_splits

        if self._config.forward_chaining:
            # Forward-chaining: test blocks must strictly follow training blocks.
            # E.g. train on [0..t-1], test on [t..t+k-1]
            valid_combos: list[tuple[int, ...]] = []
            for start_test in range(1, n - k + 1):
                test_combo = tuple(range(start_test, start_test + k))
                valid_combos.append(test_combo)
            all_combos = valid_combos
        else:
            all_combos = list(combinations(range(n), k))

        # Apply computational budget cap if combinations exceed max_splits
        if self._config.max_splits is not None and len(all_combos) > self._config.max_splits:
            # Deterministically select evenly-spaced subset of combinations
            stride_indices = np.round(
                np.linspace(0, len(all_combos) - 1, self._config.max_splits)
            ).astype(int)
            all_combos = [all_combos[idx] for idx in stride_indices]

        return all_combos

    def get_n_splits(
        self,
        X: Any = None,
        y: Any = None,
        groups: Any = None,
    ) -> int:
        """Return the number of cross-validation splits generated under current config.

        Args:
            X: Ignored (scikit-learn interface compatibility).
            y: Ignored (scikit-learn interface compatibility).
            groups: Ignored (scikit-learn interface compatibility).

        Returns:
            Total count of combinatorial folds.
        """
        return len(self._generate_test_combinations())

    @staticmethod
    def _coerce_time_arrays(
        n_samples: int,
        pred_times: Sequence[Any] | np.ndarray | None,
        eval_times: Sequence[Any] | np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Coerce and validate prediction and evaluation timestamp arrays.

        Purpose: Ensures consistent 1D arrays for temporal interval overlap calculations.
        Invariants:
            - len(start_arr) == len(end_arr) == n_samples.
            - start_arr[i] <= end_arr[i] for all i.
        """
        if pred_times is None and eval_times is None:
            # Default to index-based times [0, 1, ..., n_samples-1]
            default_times = np.arange(n_samples, dtype=np.int64)
            return default_times, default_times.copy()

        if pred_times is None:
            raise ValueError("pred_times must be provided if eval_times is provided")
        if eval_times is None:
            raise ValueError("eval_times must be provided if pred_times is provided")

        # Handle Sequence of BarrierLabel objects if passed
        if len(pred_times) > 0 and hasattr(pred_times[0], "entry_timestamp"):
            start_arr = np.array([item.entry_timestamp for item in pred_times], dtype=np.int64)
        else:
            start_arr = np.asarray(pred_times)

        if len(eval_times) > 0 and hasattr(eval_times[0], "exit_timestamp"):
            end_arr = np.array([item.exit_timestamp for item in eval_times], dtype=np.int64)
        else:
            end_arr = np.asarray(eval_times)

        if len(start_arr) != n_samples or len(end_arr) != n_samples:
            raise ValueError(
                f"Dimension mismatch: pred_times ({len(start_arr)}) and eval_times ({len(end_arr)}) "
                f"must match n_samples ({n_samples})"
            )

        if np.any(start_arr > end_arr):
            raise ValueError("Temporal causality violated: pred_times cannot exceed eval_times")

        return start_arr, end_arr

    def split(
        self,
        X: Any,
        y: Any = None,
        pred_times: Sequence[Any] | np.ndarray | None = None,
        eval_times: Sequence[Any] | np.ndarray | None = None,
        groups: Any = None,
    ) -> Iterator[PurgedSplit]:
        """Generate combinatorial train/test splits with interval purging and post-test embargo.

        Args:
            X: Input features or data container (must support len(X)).
            y: Target labels (optional).
            pred_times: Event entry timestamps / indices. If None, default to sample indices.
            eval_times: Event exit timestamps / indices. If None, default to sample indices.
            groups: Group labels (ignored; contiguous blocks are derived from config.n_splits).

        Yields:
            PurgedSplit instances containing train_indices, test_indices, purged_indices,
            and embargoed_indices. Unpackable as (train_idx, test_idx).
        """
        n_samples = len(X)
        if n_samples < self._config.n_splits:
            raise ValueError(
                f"n_samples ({n_samples}) cannot be smaller than n_splits ({self._config.n_splits})"
            )

        start_times, end_times = self._coerce_time_arrays(n_samples, pred_times, eval_times)
        group_blocks = self._partition_groups(n_samples, self._config.n_splits)
        test_combos = self._generate_test_combinations()

        # Compute embargo duration in time units or index units
        embargo_duration: float
        if self._config.embargo_bars is not None:
            embargo_duration = float(self._config.embargo_bars)
            use_bar_embargo = True
        else:
            # If using timestamp differences, calculate time-based embargo
            if np.issubdtype(start_times.dtype, np.integer) and start_times[-1] != n_samples - 1:
                # Timestamps (e.g. nanoseconds): scale by time span
                total_time_span = float(end_times[-1] - start_times[0])
                embargo_duration = total_time_span * self._config.embargo_pct
                use_bar_embargo = False
            else:
                # Bar count index-based embargo
                embargo_duration = float(max(1, int(round(n_samples * self._config.embargo_pct))))
                use_bar_embargo = True

        for split_idx, test_group_indices in enumerate(test_combos):
            # 1. Identify test indices and test time intervals
            test_indices_list: list[np.ndarray] = []
            test_intervals: list[tuple[Any, Any]] = []

            for g_idx in test_group_indices:
                g_indices = group_blocks[g_idx]
                test_indices_list.append(g_indices)
                # Test interval boundaries for this group block
                test_intervals.append((start_times[g_indices[0]], end_times[g_indices[-1]]))

            test_indices = np.concatenate(test_indices_list)

            # 2. Candidate training groups
            if self._config.forward_chaining:
                # Strictly candidate groups preceding the first test block
                min_test_g = min(test_group_indices)
                train_group_indices = tuple(range(0, min_test_g))
            else:
                train_group_indices = tuple(
                    g for g in range(self._config.n_splits) if g not in test_group_indices
                )

            if len(train_group_indices) == 0:
                raise ValueError(
                    f"Split {split_idx}: No training groups available before test groups in forward_chaining mode"
                )

            candidate_train_indices = np.concatenate([group_blocks[g] for g in train_group_indices])

            # 3. Purging: Drop training observations whose trade interval intersects any test interval
            # Interval overlap: [t_start, t_end] overlaps [T_test_start, T_test_end]
            # iff t_start <= T_test_end and t_end >= T_start
            train_starts = start_times[candidate_train_indices]
            train_ends = end_times[candidate_train_indices]

            purged_mask = np.zeros(len(candidate_train_indices), dtype=bool)
            for t_start, t_end in test_intervals:
                overlaps = (train_starts <= t_end) & (train_ends >= t_start)
                purged_mask |= overlaps

            purged_indices = candidate_train_indices[purged_mask]
            remaining_train_indices = candidate_train_indices[~purged_mask]

            # 4. Embargoing: Drop remaining training observations immediately succeeding test intervals
            embargoed_mask = np.zeros(len(remaining_train_indices), dtype=bool)
            rem_starts = start_times[remaining_train_indices]

            for _, t_end in test_intervals:
                if use_bar_embargo:
                    # In index space, embargo applies to samples immediately following test
                    # Find candidate indices that are greater than t_end up to t_end + embargo_duration
                    embargo_cutoff = t_end + embargo_duration
                    in_embargo = (rem_starts > t_end) & (rem_starts <= embargo_cutoff)
                else:
                    embargo_cutoff = t_end + embargo_duration
                    in_embargo = (rem_starts > t_end) & (rem_starts <= embargo_cutoff)
                embargoed_mask |= in_embargo

            embargoed_indices = remaining_train_indices[embargoed_mask]
            final_train_indices = remaining_train_indices[~embargoed_mask]

            # 5. Starvation Guard: verify retained training observations satisfy min_train_ratio
            retained_ratio = len(final_train_indices) / n_samples
            if retained_ratio < self._config.min_train_ratio:
                raise ValueError(
                    f"CPCV split {split_idx} violated min_train_ratio: "
                    f"retained {retained_ratio:.2%} ({len(final_train_indices)}/{n_samples}) "
                    f"< threshold {self._config.min_train_ratio:.2%}. "
                    "Reduce embargo_pct, increase n_splits, or lower min_train_ratio."
                )

            yield PurgedSplit(
                split_idx=split_idx,
                train_indices=final_train_indices,
                test_indices=test_indices,
                purged_indices=purged_indices,
                embargoed_indices=embargoed_indices,
                test_groups=test_group_indices,
                train_groups=train_group_indices,
            )

    def reconstruct_paths(
        self,
        fold_predictions: Sequence[np.ndarray],
        n_samples: int,
    ) -> np.ndarray:
        """Assemble fold out-of-sample predictions into continuous backtest paths.

        Purpose: Stitches combinatorial test folds into phi = C(N-1, k-1) complete out-of-sample
                 prediction series covering all N time blocks.
        Invariants:
            - Returned paths array has shape (n_paths, n_samples).
            - No NaN values in any path.
            - Uses canonical greedy positional assignment: group g's p-th fold occurrence -> path p.
        """
        if self._config.forward_chaining:
            raise NotImplementedError(
                "Path reconstruction is defined for full combinatorial CPCV, not forward-chaining."
            )

        n = self._config.n_splits
        k = self._config.n_test_splits
        all_combos = self._generate_test_combinations()

        if len(fold_predictions) != len(all_combos):
            raise ValueError(
                f"Fold count mismatch: expected {len(all_combos)} folds, got {len(fold_predictions)}"
            )

        group_blocks = self._partition_groups(n_samples, n)

        # Validate that each fold prediction vector matches its test indices length
        for f_idx, (combo, preds) in enumerate(zip(all_combos, fold_predictions, strict=True)):
            expected_len = sum(len(group_blocks[g]) for g in combo)
            if len(preds) != expected_len:
                raise ValueError(
                    f"Fold {f_idx}: prediction length ({len(preds)}) does not match "
                    f"expected test length ({expected_len})"
                )

        # Map each group g to the list of prediction slices in fold-iteration order
        group_to_fold_slices: dict[int, list[np.ndarray]] = {g: [] for g in range(n)}
        for combo, preds in zip(all_combos, fold_predictions, strict=True):
            cursor = 0
            for g in combo:
                g_len = len(group_blocks[g])
                group_to_fold_slices[g].append(preds[cursor : cursor + g_len])
                cursor += g_len

        # Determine number of paths: C(N-1, k-1) if unconstrained
        # Each group must appear an identical number of times
        path_counts = [len(group_to_fold_slices[g]) for g in range(n)]
        min_paths = min(path_counts)
        if min_paths == 0:
            raise ValueError(
                "CPCV configuration resulted in unrepresented groups; cannot reconstruct paths."
            )

        # If combinations were capped, use min_paths
        n_paths = min_paths if self._config.max_splits is not None else comb(n - 1, k - 1)

        paths = np.full((n_paths, n_samples), np.nan, dtype=float)
        for g in range(n):
            for path_idx in range(n_paths):
                slice_preds = group_to_fold_slices[g][path_idx]
                paths[path_idx, group_blocks[g]] = slice_preds

        if np.any(np.isnan(paths)):
            raise RuntimeError(
                "Path reconstruction invariant violated: NaN detected in reconstructed paths"
            )

        return paths

    @staticmethod
    def compute_path_sharpe_distribution(
        paths: np.ndarray,
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        annualization_factor: float = 252.0,
    ) -> tuple[np.ndarray, float, float]:
        """Compute the empirical distribution of Sharpe ratios across all reconstructed CPCV paths.

        Purpose: Evaluates strategy payoff across diverse out-of-sample paths, generating
                 empirical Sharpe variance V[{SR_k}] directly consumed by Step 6 (Deflated Sharpe Ratio).
        Invariants:
            - len(returns) == paths.shape[1].
            - Returned variance is strictly non-negative.
        """
        returns_arr = np.asarray(returns, dtype=float)
        if paths.ndim != 2:
            raise ValueError(
                f"paths must be 2D array (n_paths, n_samples), got shape {paths.shape}"
            )
        if paths.shape[1] != len(returns_arr):
            raise ValueError(
                f"Dimension mismatch: paths ({paths.shape[1]}) must match returns ({len(returns_arr)})"
            )

        n_paths = paths.shape[0]
        sharpe_ratios = np.zeros(n_paths, dtype=float)
        rf_per_bar = risk_free_rate / annualization_factor

        for p in range(n_paths):
            path_signal = paths[p]
            # Strategy bar return = signal * asset_return
            strat_returns = path_signal * returns_arr
            excess_returns = strat_returns - rf_per_bar

            std_dev = np.std(excess_returns, ddof=1) if len(excess_returns) > 1 else 0.0
            if std_dev > 1e-12:
                sr = (np.mean(excess_returns) / std_dev) * np.sqrt(annualization_factor)
            else:
                sr = 0.0
            sharpe_ratios[p] = sr

        mean_sr = float(np.mean(sharpe_ratios))
        var_sr = float(np.var(sharpe_ratios, ddof=1)) if n_paths > 1 else 0.0

        return sharpe_ratios, mean_sr, var_sr

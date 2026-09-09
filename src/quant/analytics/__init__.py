"""Econometric and quantitative analytical modeling subpackage.

Purpose: Contains advanced quantitative transformations, labeling algorithms, and statistical tests.
Dependencies: numpy, scipy, statsmodels, pyarrow, domain models.
Relationship: Consumed by application services and research pipelines.
"""

from quant.analytics.cross_validation import (
    CombinatorialPurgedCV,
    CPCVConfig,
    PurgedSplit,
)
from quant.analytics.deflated_sharpe import (
    DSRConfig,
    DSRResult,
    DeflatedSharpeEngine,
    adjust_p_values_fdr,
    compute_effective_trials,
    compute_expected_max_sharpe,
    compute_min_backtest_length,
    compute_moments,
    compute_probabilistic_sharpe_ratio,
)
from quant.analytics.fractional_diff import (
    FractionalDifferentiator,
    StreamingFracDiffBuffer,
    compute_fractional_weights,
)
from quant.analytics.labeling import (
    BarrierLabel,
    BarrierTouchReason,
    DynamicTripleBarrierLabeler,
    PositionSide,
    TripleBarrierConfig,
    compute_parkinson_volatility,
)
from quant.analytics.meta_labeling import (
    ContinuousKellySizer,
    MetaLabel,
    MetaLabelConfig,
    ProbabilityCalibrator,
    TwoStageMetaLabeler,
)

__all__ = [
    "BarrierLabel",
    "BarrierTouchReason",
    "CPCVConfig",
    "CombinatorialPurgedCV",
    "ContinuousKellySizer",
    "DSRConfig",
    "DSRResult",
    "DeflatedSharpeEngine",
    "DynamicTripleBarrierLabeler",
    "FractionalDifferentiator",
    "MetaLabel",
    "MetaLabelConfig",
    "PositionSide",
    "ProbabilityCalibrator",
    "PurgedSplit",
    "StreamingFracDiffBuffer",
    "TripleBarrierConfig",
    "TwoStageMetaLabeler",
    "adjust_p_values_fdr",
    "compute_effective_trials",
    "compute_expected_max_sharpe",
    "compute_fractional_weights",
    "compute_min_backtest_length",
    "compute_moments",
    "compute_parkinson_volatility",
    "compute_probabilistic_sharpe_ratio",
]

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
    "compute_fractional_weights",
    "compute_parkinson_volatility",
]

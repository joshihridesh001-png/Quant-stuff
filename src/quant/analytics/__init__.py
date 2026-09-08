"""Econometric and quantitative analytical modeling subpackage.

Purpose: Contains advanced quantitative transformations, labeling algorithms, and statistical tests.
Dependencies: numpy, scipy, statsmodels, pyarrow, domain models.
Relationship: Consumed by application services and research pipelines.
"""

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

__all__ = [
    "BarrierLabel",
    "BarrierTouchReason",
    "DynamicTripleBarrierLabeler",
    "FractionalDifferentiator",
    "PositionSide",
    "StreamingFracDiffBuffer",
    "TripleBarrierConfig",
    "compute_fractional_weights",
    "compute_parkinson_volatility",
]

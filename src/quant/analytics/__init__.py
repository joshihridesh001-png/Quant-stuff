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

__all__ = [
    "FractionalDifferentiator",
    "StreamingFracDiffBuffer",
    "compute_fractional_weights",
]

"""Econometric and quantitative analytical modeling subpackage.

Purpose: Contains advanced quantitative transformations, labeling algorithms, and statistical tests.
Dependencies: numpy, scipy, statsmodels, pyarrow, domain models.
Relationship: Consumed by application services and research pipelines.
"""

from quant.analytics.chromosomes import (
    GENE_REGISTRY,
    ChromosomeVectorCodec,
    GameTheoryChromosome,
    GeneSpec,
    InferenceChromosome,
    RepresentationChromosome,
    RiskChromosome,
    ScaleType,
    StrategyChromosome,
)
from quant.analytics.cross_validation import (
    CombinatorialPurgedCV,
    CPCVConfig,
    PurgedSplit,
)
from quant.analytics.deflated_sharpe import (
    DeflatedSharpeEngine,
    DSRConfig,
    DSRResult,
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
from quant.analytics.market_impact import (
    DepletionState,
    GeneralizedPseudoHuber,
    HubermanStanzlCrossImpact,
    MarketImpactConfig,
    MarketImpactResult,
    MultiAssetMarketImpactEngine,
)
from quant.analytics.meta_labeling import (
    ContinuousKellySizer,
    MetaLabel,
    MetaLabelConfig,
    ProbabilityCalibrator,
    TwoStageMetaLabeler,
)
from quant.analytics.minimax_regret import (
    EntropicBoltzmannPotential,
    LatentSoftmaxTransform,
    MinimaxRegretConfig,
    OptimizationResult,
    VectorizedNewtonSolver,
)
from quant.analytics.payoff_matrix import (
    BenchmarkEvaluationResult,
    DiscreteHyperbolicPropagator,
    InstitutionalBenchmarkUniverse,
    PayoffTensorResult,
    StackelbergConfig,
    StackelbergPayoffEngine,
    StackelbergPayoffTensorConstructor,
)
from quant.analytics.regimes import (
    CausalBayesianRegimeFilter,
    CUSUMJumpDetector,
    OASCovarianceEstimator,
    RegimeConfig,
    RegimeEstimationResult,
    RegimeState,
)

__all__ = [
    "BarrierLabel",
    "BarrierTouchReason",
    "BenchmarkEvaluationResult",
    "CPCVConfig",
    "CUSUMJumpDetector",
    "CausalBayesianRegimeFilter",
    "ChromosomeVectorCodec",
    "CombinatorialPurgedCV",
    "ContinuousKellySizer",
    "DSRConfig",
    "DSRResult",
    "DeflatedSharpeEngine",
    "DepletionState",
    "DiscreteHyperbolicPropagator",
    "DynamicTripleBarrierLabeler",
    "EntropicBoltzmannPotential",
    "FractionalDifferentiator",
    "GENE_REGISTRY",
    "GameTheoryChromosome",
    "GeneSpec",
    "GeneralizedPseudoHuber",
    "HubermanStanzlCrossImpact",
    "InferenceChromosome",
    "InstitutionalBenchmarkUniverse",
    "LatentSoftmaxTransform",
    "MarketImpactConfig",
    "MarketImpactResult",
    "MetaLabel",
    "MetaLabelConfig",
    "MinimaxRegretConfig",
    "MultiAssetMarketImpactEngine",
    "OASCovarianceEstimator",
    "OptimizationResult",
    "PayoffTensorResult",
    "PositionSide",
    "ProbabilityCalibrator",
    "PurgedSplit",
    "RegimeConfig",
    "RegimeEstimationResult",
    "RegimeState",
    "RepresentationChromosome",
    "RiskChromosome",
    "ScaleType",
    "StackelbergConfig",
    "StackelbergPayoffEngine",
    "StackelbergPayoffTensorConstructor",
    "StrategyChromosome",
    "StreamingFracDiffBuffer",
    "TripleBarrierConfig",
    "TwoStageMetaLabeler",
    "VectorizedNewtonSolver",
    "adjust_p_values_fdr",
    "compute_effective_trials",
    "compute_expected_max_sharpe",
    "compute_fractional_weights",
    "compute_min_backtest_length",
    "compute_moments",
    "compute_parkinson_volatility",
    "compute_probabilistic_sharpe_ratio",
]

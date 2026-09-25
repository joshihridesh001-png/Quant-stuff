"""Modular Alpha Strategy Library and Cross-Asset Signal Generators."""

from quant.analytics.strategies.base import (
    ERR_STRAT_INSUFFICIENT_WARMUP,
    ERR_STRAT_INVALID_LEVERAGE,
    ERR_STRAT_INVALID_SIGNAL,
    ERR_STRAT_NON_FINITE_WEIGHT,
    ERR_STRAT_SINGULAR_COVARIANCE,
    BarHistoryWindow,
    BaseAlphaStrategy,
    IAlphaStrategy,
    InsufficientWarmupError,
    InvalidSignalError,
    NonFiniteWeightError,
    SignalDirection,
    SingularCovarianceError,
    StrategyContext,
    StrategyError,
    StrategySignal,
)
from quant.analytics.strategies.momentum import (
    ERR_STRAT_MOMENTUM_INVALID_PARAMS,
    FracDiffMomentumStrategy,
)
from quant.analytics.strategies.sentiment import (
    ERR_STRAT_SENTIMENT_INVALID_INPUT,
    LoughranMcDonaldSentimentStrategy,
    NewsSentimentEvent,
)
from quant.analytics.strategies.stat_arb import (
    ERR_STRAT_STATARB_INVALID_PAIR,
    ERR_STRAT_STATARB_SINGULAR_KALMAN,
    KalmanFilterState,
    KalmanPairsTradingStrategy,
)
from quant.analytics.strategies.swarm_meta import (
    ERR_STRAT_SWARM_EMPTY_SUBSTRATEGIES,
    ERR_STRAT_SWARM_INVALID_REGIME,
    REGIME_PRIORS,
    SwarmMetaStrategy,
)
from quant.analytics.strategies.volatility import (
    ERR_STRAT_VOL_INVALID_PARAMS,
    VolatilityBreakoutStrategy,
    compute_garman_klass_volatility,
    compute_parkinson_volatility,
)

__all__ = [
    "ERR_STRAT_INSUFFICIENT_WARMUP",
    "ERR_STRAT_INVALID_LEVERAGE",
    "ERR_STRAT_INVALID_SIGNAL",
    "ERR_STRAT_MOMENTUM_INVALID_PARAMS",
    "ERR_STRAT_NON_FINITE_WEIGHT",
    "ERR_STRAT_SENTIMENT_INVALID_INPUT",
    "ERR_STRAT_SINGULAR_COVARIANCE",
    "ERR_STRAT_STATARB_INVALID_PAIR",
    "ERR_STRAT_STATARB_SINGULAR_KALMAN",
    "ERR_STRAT_SWARM_EMPTY_SUBSTRATEGIES",
    "ERR_STRAT_SWARM_INVALID_REGIME",
    "ERR_STRAT_VOL_INVALID_PARAMS",
    "BarHistoryWindow",
    "BaseAlphaStrategy",
    "FracDiffMomentumStrategy",
    "IAlphaStrategy",
    "InsufficientWarmupError",
    "InvalidSignalError",
    "KalmanFilterState",
    "KalmanPairsTradingStrategy",
    "LoughranMcDonaldSentimentStrategy",
    "NewsSentimentEvent",
    "NonFiniteWeightError",
    "REGIME_PRIORS",
    "SignalDirection",
    "SingularCovarianceError",
    "StrategyContext",
    "StrategyError",
    "StrategySignal",
    "SwarmMetaStrategy",
    "VolatilityBreakoutStrategy",
    "compute_garman_klass_volatility",
    "compute_parkinson_volatility",
]

"""Domain Strategy Protocol (IAlphaStrategy), Signal DTOs, and Invariants.

Functional Purpose:
    Defines uniform, decoupled interfaces and immutable signal contracts for all alpha strategies
    in the quantitative library. Strategy implementations emit mathematically bounded target weights
    w_i in [-1, 1], conviction scores c_i in [0, 1], and causal risk diagnostics.

Explicit Dependency Tracking:
    - numpy: Vectorized price and volume array representations.
    - quant.domain.historical: HistoricalDataError base exception.

Structural Relationship:
    Root contract module for Phase 14 strategy implementations:
    stat_arb.py, momentum.py, sentiment.py, volatility.py, and swarm_meta.py.
    Consumed by BacktestRunner, ReplayEngine, and LiveSessionOrchestrator.

Defensive Invariants:
    - INV-STRAT-001: Target weight boundary w_i in [-1.0, 1.0] and strictly finite.
    - INV-STRAT-002: Conviction boundary c_i in [0.0, 1.0] and strictly finite.
    - INV-STRAT-003: Directional consistency: LONG -> w > 0, SHORT -> w < 0, FLAT -> w == 0.
    - INV-STRAT-004: Strict warmup requirement: T >= min_warmup_bars before signal emission.
    - INV-STRAT-005: Strict non-finite and boolean rejection on all inputs and outputs.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np

# ============================================================================
# Diagnostic Fault Codes (Rule 2: Diagnostic Error Taxonomy)
# ============================================================================

ERR_STRAT_INSUFFICIENT_WARMUP: Final[str] = "ERR-STRAT-001"
ERR_STRAT_NON_FINITE_WEIGHT: Final[str] = "ERR-STRAT-002"
ERR_STRAT_SINGULAR_COVARIANCE: Final[str] = "ERR-STRAT-003"
ERR_STRAT_INVALID_LEVERAGE: Final[str] = "ERR-STRAT-004"
ERR_STRAT_INVALID_SIGNAL: Final[str] = "ERR-STRAT-005"


# ============================================================================
# Exception Taxonomy
# ============================================================================


class StrategyError(Exception):
    """Base exception for all strategy analytical and execution failures."""

    def __init__(self, message: str, code: str = "ERR-STRAT-000") -> None:
        super().__init__(f"[{code}] {message}")
        self.message = message
        self.code = code


class InsufficientWarmupError(StrategyError):
    """Raised when bar history is shorter than the strategy minimum warmup horizon."""

    def __init__(self, message: str, code: str = ERR_STRAT_INSUFFICIENT_WARMUP) -> None:
        super().__init__(message, code=code)


class NonFiniteWeightError(StrategyError):
    """Raised when a strategy emits non-finite target weights or convictions."""

    def __init__(self, message: str, code: str = ERR_STRAT_NON_FINITE_WEIGHT) -> None:
        super().__init__(message, code=code)


class SingularCovarianceError(StrategyError):
    """Raised when covariance or regression matrices are singular / ill-conditioned."""

    def __init__(self, message: str, code: str = ERR_STRAT_SINGULAR_COVARIANCE) -> None:
        super().__init__(message, code=code)


class InvalidSignalError(StrategyError):
    """Raised when emitted strategy signal violates physical or directional invariants."""

    def __init__(self, message: str, code: str = ERR_STRAT_INVALID_SIGNAL) -> None:
        super().__init__(message, code=code)


# ============================================================================
# Signal Enums and Data Transfer Objects
# ============================================================================


class SignalDirection(StrEnum):
    """Directional classification for generated alpha signals."""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass(frozen=True, slots=True)
class StrategySignal:
    """Immutable quantitative trading signal emitted by an alpha strategy.

    Functional Purpose:
        Captures target portfolio weight w_i in [-1, 1], model conviction c_i in [0, 1],
        expected holding horizon, and model-specific diagnostic metadata.

    Explicit Dependency Tracking:
        SignalDirection enum.

    Structural Relationship:
        Produced by IAlphaStrategy implementations; ingested by SOR, RiskOrchestrator,
        and ExecutionSizing.

    Defensive Invariants:
        - INV-STRAT-001: target_weight in [-1.0, 1.0] and strictly finite.
        - INV-STRAT-002: conviction in [0.0, 1.0] and strictly finite.
        - INV-STRAT-003: LONG -> target_weight > 0, SHORT -> target_weight < 0, FLAT -> target_weight == 0.
        - INV-STRAT-005: Reject booleans and non-finite numbers.
    """

    symbol: str
    direction: SignalDirection
    target_weight: float
    conviction: float
    target_horizon_bars: int
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    diagnostics: dict[str, float | str | int] = field(default_factory=dict)
    timestamp: int = 0

    def __post_init__(self) -> None:
        # Functional Purpose: Verify signal geometric boundaries and directional invariants.
        # Explicit Dependency Tracking: math.isfinite, SignalDirection.
        # Defensive Invariant: INV-STRAT-001, INV-STRAT-002, INV-STRAT-003, INV-STRAT-005.
        if not self.symbol or not isinstance(self.symbol, str):
            raise InvalidSignalError(f"Symbol must be non-empty string, got {self.symbol!r}")

        if isinstance(self.target_weight, bool) or not math.isfinite(self.target_weight):
            raise NonFiniteWeightError(
                f"Target weight must be finite float, got {self.target_weight!r}"
            )

        if not -1.0 <= self.target_weight <= 1.0:
            raise InvalidSignalError(
                f"Target weight must be in [-1.0, 1.0], got {self.target_weight}"
            )

        if isinstance(self.conviction, bool) or not math.isfinite(self.conviction):
            raise InvalidSignalError(f"Conviction must be finite float, got {self.conviction!r}")

        if not 0.0 <= self.conviction <= 1.0:
            raise InvalidSignalError(f"Conviction must be in [0.0, 1.0], got {self.conviction}")

        if (
            isinstance(self.target_horizon_bars, bool)
            or not isinstance(self.target_horizon_bars, int)
            or self.target_horizon_bars <= 0
        ):
            raise InvalidSignalError(
                f"target_horizon_bars must be integer > 0, got {self.target_horizon_bars!r}"
            )

        # Directional consistency check (INV-STRAT-003)
        if self.direction == SignalDirection.LONG and self.target_weight <= 0.0:
            raise InvalidSignalError(
                f"LONG signal requires target_weight > 0.0, got {self.target_weight}"
            )
        if self.direction == SignalDirection.SHORT and self.target_weight >= 0.0:
            raise InvalidSignalError(
                f"SHORT signal requires target_weight < 0.0, got {self.target_weight}"
            )
        if self.direction == SignalDirection.FLAT and not math.isclose(
            self.target_weight, 0.0, abs_tol=1e-7
        ):
            raise InvalidSignalError(
                f"FLAT signal requires target_weight == 0.0, got {self.target_weight}"
            )

        # Optional stop-loss & take-profit bounds
        if self.stop_loss_pct is not None and (
            isinstance(self.stop_loss_pct, bool)
            or not math.isfinite(self.stop_loss_pct)
            or self.stop_loss_pct <= 0.0
        ):
            raise InvalidSignalError(
                f"stop_loss_pct must be finite positive float, got {self.stop_loss_pct!r}"
            )

        if self.take_profit_pct is not None and (
            isinstance(self.take_profit_pct, bool)
            or not math.isfinite(self.take_profit_pct)
            or self.take_profit_pct <= 0.0
        ):
            raise InvalidSignalError(
                f"take_profit_pct must be finite positive float, got {self.take_profit_pct!r}"
            )


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Runtime execution and portfolio state context supplied to strategies during evaluation.

    Functional Purpose:
        Supplies current portfolio equity, unencumbered cash, prevailing regime classification,
        and current asset positions to support state-dependent strategy decisions.

    Explicit Dependency Tracking:
        Dictionary mappings of current prices and positions.

    Defensive Invariants:
        total_equity > 0, cash and positions are finite.
    """

    current_timestamp: int
    current_prices: dict[str, float]
    current_positions: dict[str, float]
    total_equity: float
    unencumbered_cash: float
    regime_label: str = "NORMAL"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.total_equity, bool) or not math.isfinite(self.total_equity):
            raise NonFiniteWeightError(
                f"total_equity must be finite float, got {self.total_equity!r}"
            )
        if isinstance(self.unencumbered_cash, bool) or not math.isfinite(self.unencumbered_cash):
            raise NonFiniteWeightError(
                f"unencumbered_cash must be finite float, got {self.unencumbered_cash!r}"
            )


@dataclass(frozen=True, slots=True)
class BarHistoryWindow:
    """Multi-asset columnar time series window supplied to alpha strategies.

    Functional Purpose:
        Encapsulates historical OHLCV and VWAP arrays for monitored assets, providing
        uniform vectorized accessors and verifying observation count.

    Explicit Dependency Tracking:
        numpy.ndarray for columnar price and volume series.

    Defensive Invariants:
        Zero forward lookahead: Contains only past observations up to current evaluation bar.
    """

    data: dict[str, dict[str, np.ndarray]]
    symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.symbols:
            raise StrategyError("BarHistoryWindow must contain at least one symbol")

    def bar_count(self, symbol: str) -> int:
        """Return the number of historical observations for a symbol."""
        sym_data = self.data.get(symbol)
        if not sym_data or "close" not in sym_data:
            return 0
        return len(sym_data["close"])

    def closes(self, symbol: str) -> np.ndarray:
        """Extract closing price vector for symbol."""
        return self._get_series(symbol, "close")

    def opens(self, symbol: str) -> np.ndarray:
        """Extract open price vector for symbol."""
        return self._get_series(symbol, "open")

    def highs(self, symbol: str) -> np.ndarray:
        """Extract high price vector for symbol."""
        return self._get_series(symbol, "high")

    def lows(self, symbol: str) -> np.ndarray:
        """Extract low price vector for symbol."""
        return self._get_series(symbol, "low")

    def volumes(self, symbol: str) -> np.ndarray:
        """Extract volume vector for symbol."""
        return self._get_series(symbol, "volume")

    def vwaps(self, symbol: str) -> np.ndarray:
        """Extract vwap vector for symbol."""
        return self._get_series(symbol, "vwap")

    def timestamps(self, symbol: str) -> np.ndarray:
        """Extract timestamp vector for symbol."""
        return self._get_series(symbol, "timestamp")

    def _get_series(self, symbol: str, field_name: str) -> np.ndarray:
        sym_dict = self.data.get(symbol)
        if not sym_dict or field_name not in sym_dict:
            return np.empty(0, dtype=np.float64)
        return sym_dict[field_name]


# ============================================================================
# Strategy Protocol & Abstract Base Class
# ============================================================================


@runtime_checkable
class IAlphaStrategy(Protocol):
    """Protocol defining the uniform contract for modular alpha strategies."""

    @property
    def strategy_id(self) -> str:
        """Unique strategy identifier (e.g. 'stat_arb_kalman', 'momentum_fracdiff')."""
        ...

    @property
    def min_warmup_bars(self) -> int:
        """Minimum bar observation count required before computing valid signals."""
        ...

    @property
    def monitored_symbols(self) -> tuple[str, ...]:
        """Sequence of ticker symbols analyzed by this strategy."""
        ...

    def compute_signals(
        self,
        history: BarHistoryWindow,
        context: StrategyContext | None = None,
    ) -> dict[str, StrategySignal]:
        """Compute target weights and conviction scores across the monitored universe.

        Functional Purpose:
            Analyzes historical bar series and produces asset allocations.
        Defensive Invariants:
            INV-STRAT-004: history.bar_count(s) >= min_warmup_bars for all analyzed symbols.
        """
        ...


class BaseAlphaStrategy(ABC, IAlphaStrategy):
    """Abstract base class providing parameter management and boundary sanitization.

    Functional Purpose:
        Enforces warmup validation, non-finite checks, and parameter encapsulation across
        concrete alpha strategies.

    Defensive Invariants:
        Verifies history length >= min_warmup_bars before delegating to subclass logic.
    """

    def __init__(
        self,
        strategy_id: str,
        monitored_symbols: Sequence[str],
        min_warmup_bars: int = 50,
        max_leverage: float = 1.0,
    ) -> None:
        if not strategy_id or not isinstance(strategy_id, str):
            raise StrategyError("strategy_id must be non-empty string")
        if not monitored_symbols:
            raise StrategyError("monitored_symbols must be non-empty sequence")
        if (
            isinstance(min_warmup_bars, bool)
            or not isinstance(min_warmup_bars, int)
            or min_warmup_bars <= 0
        ):
            raise StrategyError("min_warmup_bars must be positive integer")
        if isinstance(max_leverage, bool) or not math.isfinite(max_leverage) or max_leverage <= 0.0:
            raise StrategyError("max_leverage must be positive finite float")

        self._strategy_id = strategy_id
        self._monitored_symbols = tuple(monitored_symbols)
        self._min_warmup_bars = min_warmup_bars
        self._max_leverage = max_leverage

    @property
    def strategy_id(self) -> str:
        return self._strategy_id

    @property
    def min_warmup_bars(self) -> int:
        return self._min_warmup_bars

    @property
    def monitored_symbols(self) -> tuple[str, ...]:
        return self._monitored_symbols

    @property
    def max_leverage(self) -> float:
        return self._max_leverage

    def validate_warmup(self, history: BarHistoryWindow) -> None:
        """Validate that historical window satisfies minimum warmup horizon."""
        for sym in self._monitored_symbols:
            count = history.bar_count(sym)
            if count < self._min_warmup_bars:
                raise InsufficientWarmupError(
                    f"Strategy '{self.strategy_id}' requires at least {self._min_warmup_bars} bars "
                    f"for symbol '{sym}', but received only {count} observations."
                )

    def compute_signals(
        self,
        history: BarHistoryWindow,
        context: StrategyContext | None = None,
    ) -> dict[str, StrategySignal]:
        """Template method validating warmup and enforcing signal constraints."""
        self.validate_warmup(history)
        signals = self._generate_signals(history, context)

        # Enforce maximum leverage budget across emitted signals
        total_abs_weight = sum(abs(sig.target_weight) for sig in signals.values())
        if total_abs_weight > self._max_leverage + 1e-6:
            # Scale down proportionally to respect leverage ceiling
            scale = self._max_leverage / total_abs_weight
            scaled_signals: dict[str, StrategySignal] = {}
            for sym, sig in signals.items():
                scaled_signals[sym] = StrategySignal(
                    symbol=sig.symbol,
                    direction=sig.direction,
                    target_weight=sig.target_weight * scale,
                    conviction=sig.conviction,
                    target_horizon_bars=sig.target_horizon_bars,
                    stop_loss_pct=sig.stop_loss_pct,
                    take_profit_pct=sig.take_profit_pct,
                    diagnostics=sig.diagnostics,
                    timestamp=sig.timestamp,
                )
            return scaled_signals

        return signals

    @abstractmethod
    def _generate_signals(
        self,
        history: BarHistoryWindow,
        context: StrategyContext | None = None,
    ) -> dict[str, StrategySignal]:
        """Subclass implementation generating alpha signals post-warmup."""
        ...

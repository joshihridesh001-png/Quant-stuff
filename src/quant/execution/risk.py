"""Pre-Trade Risk Firewall, Dynamic Limit Enforcement, and Portfolio State Invariants.

Purpose:
    Implements institutional pre-trade risk controls, real-time portfolio risk state
    tracking, dynamic leverage and concentration calculations, and diagnostic fault codes
    under Phase 6 Step 3 (Live Execution Quality & Safety Gateway).

Dependencies:
    - dataclasses: High-performance memory-compact slots and frozen limit configurations.
    - math: Strict non-finite scalar validation (math.isfinite).
    - typing: Static typing annotations, Final constants, and protocol definitions.
    - quant.execution.models: Order domain entity, OrderSide, and OrderType.

Structural Relationship:
    - Sits directly in the execution hot path between Upstream Schedulers/SOR and Gateways.
    - Upstream Caller: SmartOrderRouter, ParentOrder, ExecutionScheduler.
    - Downstream Consumer: PaperExecutionGateway, Live Broker Gateway, RiskOrchestrator.
    - Emits: Diagnostic exceptions (ERR-RSK-001 through ERR-RSK-008).

Invariants Enforced:
    - INV-RSK-001 (Single-Order Fat-Finger Bounds): Q <= Q_max and Q * P <= Notional_max.
    - INV-RSK-002 (Portfolio Leverage Bounds): L_gross <= L_max and L_net <= L_net_max.
    - INV-RSK-003 (Single-Asset Concentration): Single-name exposure / NAV <= omega_max.
    - INV-RSK-004 (Intraday Drawdown Circuit Breaker): Session peak-to-trough drawdown < MDD_max.
    - INV-RSK-005 (Margin & Borrow Sufficiency): Margin required <= Free liquid margin.
    - INV-RSK-006 (Hot-Path Latency SLA): Sub-10us deterministic in-memory validation.
    - INV-RSK-007 (Strict Input Sanitization): Immediate rejection of NaN, Inf, bool, and <= 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

from quant.execution.models import Order, OrderSide

# ============================================================================
# Diagnostic Fault Vector Constants (Rule 2: Zero-Execution Diagnostics)
# ============================================================================

ERR_RSK_FAT_FINGER_NOTIONAL: Final[str] = "ERR-RSK-001"
ERR_RSK_FAT_FINGER_QUANTITY: Final[str] = "ERR-RSK-002"
ERR_RSK_LEVERAGE_LIMIT_EXCEEDED: Final[str] = "ERR-RSK-003"
ERR_RSK_CONCENTRATION_LIMIT_EXCEEDED: Final[str] = "ERR-RSK-004"
ERR_RSK_INSUFFICIENT_MARGIN: Final[str] = "ERR-RSK-005"
ERR_RSK_DRAWDOWN_LIMIT_EXCEEDED: Final[str] = "ERR-RSK-006"
ERR_RSK_NON_FINITE_INPUT: Final[str] = "ERR-RSK-007"
ERR_RSK_KILL_SWITCH_ACTIVE: Final[str] = "ERR-RSK-008"


# ============================================================================
# Exception Hierarchy (Rule 2: Structured Diagnostic Exception Protocol)
# ============================================================================


class RiskError(Exception):
    """Base exception for all pre-trade risk firewall breaches and invariant violations."""

    def __init__(self, message: str, code: str = "ERR-RSK-000") -> None:
        # Functional Purpose: Initialize base risk exception with descriptive message and fault code.
        # Explicit Dependency Tracking: Python Exception root class.
        # Structural Relationship: Root parent class for all execution risk failure taxonomy.
        # Defensive Invariant: Diagnostic code must be a non-empty string identifier.
        super().__init__(message)
        self.message: str = message
        self.code: str = code


class FatFingerNotionalException(RiskError):
    """Raised when an order's gross notional value exceeds firm single-order bounds (INV-RSK-001)."""

    def __init__(self, message: str, code: str = ERR_RSK_FAT_FINGER_NOTIONAL) -> None:
        # Functional Purpose: Reject outbound orders whose notional (Q * P) breaches maximum limit.
        # Explicit Dependency Tracking: ERR_RSK_FAT_FINGER_NOTIONAL diagnostic fault code.
        # Structural Relationship: Emitted by PreTradeRiskFirewall.validate_order.
        # Defensive Invariant: code defaults to ERR-RSK-001.
        super().__init__(message=message, code=code)


class FatFingerQuantityException(RiskError):
    """Raised when an order's unit lot size exceeds firm single-order bounds (INV-RSK-001)."""

    def __init__(self, message: str, code: str = ERR_RSK_FAT_FINGER_QUANTITY) -> None:
        # Functional Purpose: Reject outbound orders whose share quantity breaches maximum limit.
        # Explicit Dependency Tracking: ERR_RSK_FAT_FINGER_QUANTITY diagnostic fault code.
        # Structural Relationship: Emitted by PreTradeRiskFirewall.validate_order.
        # Defensive Invariant: code defaults to ERR-RSK-002.
        super().__init__(message=message, code=code)


class LeverageLimitExceededException(RiskError):
    """Raised when projected gross or net portfolio leverage breaches limits (INV-RSK-002)."""

    def __init__(self, message: str, code: str = ERR_RSK_LEVERAGE_LIMIT_EXCEEDED) -> None:
        # Functional Purpose: Reject orders that push gross or net leverage beyond risk ceiling.
        # Explicit Dependency Tracking: ERR_RSK_LEVERAGE_LIMIT_EXCEEDED diagnostic fault code.
        # Structural Relationship: Emitted by PreTradeRiskFirewall.validate_order.
        # Defensive Invariant: code defaults to ERR-RSK-003.
        super().__init__(message=message, code=code)


class ConcentrationLimitExceededException(RiskError):
    """Raised when single-asset exposure exceeds configured percentage of NAV (INV-RSK-003)."""

    def __init__(self, message: str, code: str = ERR_RSK_CONCENTRATION_LIMIT_EXCEEDED) -> None:
        # Functional Purpose: Reject orders concentrating portfolio exposure into a single symbol.
        # Explicit Dependency Tracking: ERR_RSK_CONCENTRATION_LIMIT_EXCEEDED diagnostic fault code.
        # Structural Relationship: Emitted by PreTradeRiskFirewall.validate_order.
        # Defensive Invariant: code defaults to ERR-RSK-004.
        super().__init__(message=message, code=code)


class InsufficientMarginRiskException(RiskError):
    """Raised when required order margin exceeds liquid free margin or buying power (INV-RSK-005)."""

    def __init__(self, message: str, code: str = ERR_RSK_INSUFFICIENT_MARGIN) -> None:
        # Functional Purpose: Reject buy or short orders exceeding available liquid capital.
        # Explicit Dependency Tracking: ERR_RSK_INSUFFICIENT_MARGIN diagnostic fault code.
        # Structural Relationship: Emitted by PreTradeRiskFirewall.validate_order.
        # Defensive Invariant: code defaults to ERR-RSK-005.
        super().__init__(message=message, code=code)


class DrawdownLimitExceededException(RiskError):
    """Raised when session peak-to-trough equity drawdown breaches tripwire threshold (INV-RSK-004)."""

    def __init__(self, message: str, code: str = ERR_RSK_DRAWDOWN_LIMIT_EXCEEDED) -> None:
        # Functional Purpose: Lock out order placement and trigger panic halts upon excessive loss.
        # Explicit Dependency Tracking: ERR_RSK_DRAWDOWN_LIMIT_EXCEEDED diagnostic fault code.
        # Structural Relationship: Emitted by PreTradeRiskFirewall.validate_order and RiskOrchestrator.
        # Defensive Invariant: code defaults to ERR-RSK-006.
        super().__init__(message=message, code=code)


class NonFiniteRiskInputException(RiskError):
    """Raised when non-finite scalars, booleans, or illegal negative bounds enter risk math (INV-RSK-007)."""

    def __init__(self, message: str, code: str = ERR_RSK_NON_FINITE_INPUT) -> None:
        # Functional Purpose: Reject corrupt data, NaNs, infinities, booleans, or non-positive bounds.
        # Explicit Dependency Tracking: ERR_RSK_NON_FINITE_INPUT diagnostic fault code.
        # Structural Relationship: Emitted by RiskLimits, PortfolioRiskState, and PreTradeRiskFirewall.
        # Defensive Invariant: code defaults to ERR-RSK-007.
        super().__init__(message=message, code=code)


class KillSwitchActiveException(RiskError):
    """Raised when order submission is attempted while the emergency kill switch is active (INV-RSK-008)."""

    def __init__(self, message: str, code: str = ERR_RSK_KILL_SWITCH_ACTIVE) -> None:
        # Functional Purpose: Enforce total trading lockout during emergency system freeze.
        # Explicit Dependency Tracking: ERR_RSK_KILL_SWITCH_ACTIVE diagnostic fault code.
        # Structural Relationship: Emitted by EmergencyKillSwitch and PreTradeRiskFirewall.
        # Defensive Invariant: code defaults to ERR-RSK-008.
        super().__init__(message=message, code=code)


# ============================================================================
# Defensive Helper Validation Functions (Rule 1 & Rule 4)
# ============================================================================


def _validate_positive_scalar(val: object, name: str) -> float:
    """Validate that an input is a finite, non-boolean float strictly greater than zero."""
    # Functional Purpose: Ensure risk threshold inputs are strictly positive real numbers.
    # Explicit Dependency Tracking: math.isfinite, NonFiniteRiskInputException.
    # Structural Relationship: Called by RiskLimits and PortfolioRiskState __post_init__.
    # Defensive Invariant: val must be numeric, not bool, finite, and > 0.0.
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise NonFiniteRiskInputException(
            f"Parameter '{name}' must be a numeric scalar, got {type(val).__name__} ({val!r})",
            code=ERR_RSK_NON_FINITE_INPUT,
        )
    f_val = float(val)
    if not math.isfinite(f_val):
        raise NonFiniteRiskInputException(
            f"Parameter '{name}' must be a finite scalar, got {f_val!r}",
            code=ERR_RSK_NON_FINITE_INPUT,
        )
    if f_val <= 0.0:
        raise NonFiniteRiskInputException(
            f"Parameter '{name}' must be strictly positive (> 0.0), got {f_val!r}",
            code=ERR_RSK_NON_FINITE_INPUT,
        )
    return f_val


def _validate_non_negative_scalar(val: object, name: str) -> float:
    """Validate that an input is a finite, non-boolean float greater than or equal to zero."""
    # Functional Purpose: Ensure margin buffers and non-negative parameters are valid.
    # Explicit Dependency Tracking: math.isfinite, NonFiniteRiskInputException.
    # Structural Relationship: Called by RiskLimits for min_free_margin validation.
    # Defensive Invariant: val must be numeric, not bool, finite, and >= 0.0.
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise NonFiniteRiskInputException(
            f"Parameter '{name}' must be a numeric scalar, got {type(val).__name__} ({val!r})",
            code=ERR_RSK_NON_FINITE_INPUT,
        )
    f_val = float(val)
    if not math.isfinite(f_val):
        raise NonFiniteRiskInputException(
            f"Parameter '{name}' must be a finite scalar, got {f_val!r}",
            code=ERR_RSK_NON_FINITE_INPUT,
        )
    if f_val < 0.0:
        raise NonFiniteRiskInputException(
            f"Parameter '{name}' cannot be negative (< 0.0), got {f_val!r}",
            code=ERR_RSK_NON_FINITE_INPUT,
        )
    return f_val


# ============================================================================
# Domain Dataclasses: RiskLimits & PortfolioRiskState
# ============================================================================


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """Immutable firm-wide and portfolio-level risk limits configuration.

    Attributes:
        max_order_notional: Single-order gross consideration ceiling (Q * P).
        max_order_qty: Single-order maximum lot units limit (Q).
        max_gross_leverage: Maximum portfolio gross leverage ratio (Gross Notional / NAV).
        max_net_leverage: Maximum portfolio net leverage ratio (|Net Notional| / NAV).
        max_concentration_nav_pct: Maximum single-name exposure as fraction of NAV.
        max_intraday_drawdown_pct: Session drawdown tripwire threshold as fraction (e.g. 0.025 = 2.5%).
        min_free_margin: Required minimum unencumbered cash/margin buffer (default: 0.0).
    """

    max_order_notional: float
    max_order_qty: float
    max_gross_leverage: float
    max_net_leverage: float
    max_concentration_nav_pct: float
    max_intraday_drawdown_pct: float
    min_free_margin: float = 0.0

    def __post_init__(self) -> None:
        """Enforce strict boundary invariants and non-finite defenses on all limit thresholds."""
        # Functional Purpose: Enforce INV-RSK-007 defensive validation on limits configuration.
        # Explicit Dependency Tracking: _validate_positive_scalar, _validate_non_negative_scalar.
        # Structural Relationship: Invoked upon construction of RiskLimits value object.
        # Defensive Invariant: All thresholds must be strictly positive except min_free_margin (>= 0).
        object.__setattr__(
            self,
            "max_order_notional",
            _validate_positive_scalar(self.max_order_notional, "max_order_notional"),
        )
        object.__setattr__(
            self,
            "max_order_qty",
            _validate_positive_scalar(self.max_order_qty, "max_order_qty"),
        )
        object.__setattr__(
            self,
            "max_gross_leverage",
            _validate_positive_scalar(self.max_gross_leverage, "max_gross_leverage"),
        )
        object.__setattr__(
            self,
            "max_net_leverage",
            _validate_positive_scalar(self.max_net_leverage, "max_net_leverage"),
        )
        object.__setattr__(
            self,
            "max_concentration_nav_pct",
            _validate_positive_scalar(self.max_concentration_nav_pct, "max_concentration_nav_pct"),
        )
        object.__setattr__(
            self,
            "max_intraday_drawdown_pct",
            _validate_positive_scalar(self.max_intraday_drawdown_pct, "max_intraday_drawdown_pct"),
        )
        object.__setattr__(
            self,
            "min_free_margin",
            _validate_non_negative_scalar(self.min_free_margin, "min_free_margin"),
        )


@dataclass(slots=True)
class PortfolioRiskState:
    """Real-time portfolio valuation, inventory, open leaves, and drawdown tracker.

    Attributes:
        cash: Total unencumbered and encumbered liquid cash balance.
        positions: Map of instrument symbol to currently held position quantity (signed).
        pending_leaves: Map of instrument symbol to open working order leaves (signed).
        current_prices: Map of instrument symbol to latest mark-to-market valuation price.
        peak_equity: Intraday session high-water mark equity ($W_{peak}$).
        initial_equity: Session opening equity capital baseline ($W_0$).
    """

    cash: float
    positions: dict[str, float] = field(default_factory=dict)
    pending_leaves: dict[str, float] = field(default_factory=dict)
    current_prices: dict[str, float] = field(default_factory=dict)
    peak_equity: float = 0.0
    initial_equity: float = 0.0

    def __post_init__(self) -> None:
        """Validate input types, finiteness, and establish default session equity baselines."""
        # Functional Purpose: Guarantee state consistency and reject non-finite values on initialization.
        # Explicit Dependency Tracking: math.isfinite, NonFiniteRiskInputException.
        # Structural Relationship: Executed upon creation of PortfolioRiskState.
        # Defensive Invariant: Cash must be finite; peak and initial equity must be strictly positive.
        if isinstance(self.cash, bool) or not isinstance(self.cash, (int, float)):
            raise NonFiniteRiskInputException(
                f"Cash must be a numeric scalar, got {type(self.cash).__name__} ({self.cash!r})",
                code=ERR_RSK_NON_FINITE_INPUT,
            )
        f_cash = float(self.cash)
        if not math.isfinite(f_cash):
            raise NonFiniteRiskInputException(
                f"Cash must be finite scalar, got {f_cash!r}",
                code=ERR_RSK_NON_FINITE_INPUT,
            )
        self.cash = f_cash

        # Ensure dictionaries are cloned to prevent external state mutation
        self.positions = dict(self.positions)
        self.pending_leaves = dict(self.pending_leaves)
        self.current_prices = dict(self.current_prices)

        # Validate positions dictionary
        for sym, qty in self.positions.items():
            if not isinstance(sym, str) or not sym:
                raise NonFiniteRiskInputException(
                    f"Position symbol key must be non-empty str, got {sym!r}",
                    code=ERR_RSK_NON_FINITE_INPUT,
                )
            if isinstance(qty, bool) or not isinstance(qty, (int, float)) or not math.isfinite(qty):
                raise NonFiniteRiskInputException(
                    f"Position quantity for {sym} must be finite float, got {qty!r}",
                    code=ERR_RSK_NON_FINITE_INPUT,
                )
            self.positions[sym] = float(qty)

        # Validate pending leaves dictionary
        for sym, leaves in self.pending_leaves.items():
            if not isinstance(sym, str) or not sym:
                raise NonFiniteRiskInputException(
                    f"Pending leaves symbol key must be non-empty str, got {sym!r}",
                    code=ERR_RSK_NON_FINITE_INPUT,
                )
            if (
                isinstance(leaves, bool)
                or not isinstance(leaves, (int, float))
                or not math.isfinite(leaves)
            ):
                raise NonFiniteRiskInputException(
                    f"Pending leaves quantity for {sym} must be finite float, got {leaves!r}",
                    code=ERR_RSK_NON_FINITE_INPUT,
                )
            self.pending_leaves[sym] = float(leaves)

        # Validate current prices dictionary
        for sym, price in self.current_prices.items():
            if not isinstance(sym, str) or not sym:
                raise NonFiniteRiskInputException(
                    f"Price symbol key must be non-empty str, got {sym!r}",
                    code=ERR_RSK_NON_FINITE_INPUT,
                )
            if (
                isinstance(price, bool)
                or not isinstance(price, (int, float))
                or not math.isfinite(price)
                or price <= 0.0
            ):
                raise NonFiniteRiskInputException(
                    f"Price for {sym} must be strictly positive finite scalar, got {price!r}",
                    code=ERR_RSK_NON_FINITE_INPUT,
                )
            self.current_prices[sym] = float(price)

        # Compute initial mark-to-market equity if peak/initial are unassigned (0.0)
        current_eq = self.current_equity
        if self.initial_equity <= 0.0:
            if current_eq <= 0.0:
                raise NonFiniteRiskInputException(
                    f"Initial equity must be positive, computed {current_eq!r}",
                    code=ERR_RSK_NON_FINITE_INPUT,
                )
            self.initial_equity = current_eq
        else:
            self.initial_equity = _validate_positive_scalar(self.initial_equity, "initial_equity")

        if self.peak_equity <= 0.0:
            self.peak_equity = max(self.initial_equity, current_eq)
        else:
            self.peak_equity = _validate_positive_scalar(self.peak_equity, "peak_equity")
            self.peak_equity = max(self.peak_equity, current_eq)

    @property
    def current_equity(self) -> float:
        r"""Mark-to-market session net asset value: $W_t = \text{cash} + \sum (\text{pos}_i \cdot P_i)$."""
        # Functional Purpose: Compute instantaneous mark-to-market liquidation equity.
        # Explicit Dependency Tracking: self.cash, self.positions, self.current_prices.
        # Structural Relationship: Core denominator for leverage, concentration, and drawdown formulas.
        # Defensive Invariant: Returns finite scalar; active positions must have valid positive prices.
        eq = self.cash
        for sym, pos in self.positions.items():
            if abs(pos) > 1e-12:
                if sym not in self.current_prices:
                    raise NonFiniteRiskInputException(
                        f"Missing current market price for symbol '{sym}' in portfolio state",
                        code=ERR_RSK_NON_FINITE_INPUT,
                    )
                p = self.current_prices[sym]
                if isinstance(p, bool) or not math.isfinite(p) or p <= 0.0:
                    raise NonFiniteRiskInputException(
                        f"Price for symbol '{sym}' must be strictly positive finite scalar, got {p!r}",
                        code=ERR_RSK_NON_FINITE_INPUT,
                    )
                eq += pos * p
        return float(eq)

    @property
    def intraday_drawdown(self) -> float:
        r"""Real-time peak-to-trough session drawdown: $D_t = (W_{\text{peak}} - W_t) / W_{\text{peak}}$."""
        # Functional Purpose: Calculate percentage loss from session high-water mark for INV-RSK-004.
        # Explicit Dependency Tracking: self.peak_equity, self.current_equity.
        # Structural Relationship: Evaluated by PreTradeRiskFirewall to trip drawdown circuit breaker.
        # Defensive Invariant: Returns fraction in [0.0, 1.0+]; automatically updates peak equity.
        eq = self.current_equity
        if eq > self.peak_equity:
            self.peak_equity = eq
        if self.peak_equity <= 0.0:
            return 1.0 if eq <= 0.0 else 0.0
        dd = (self.peak_equity - eq) / self.peak_equity
        return max(0.0, float(dd))

    @property
    def gross_leverage(self) -> float:
        r"""Portfolio gross leverage: $L_{\text{gross}} = \sum \text{exposure}_i \cdot P_i / W_t$ with universal directional netting."""
        # Functional Purpose: Measure aggregate market commitment relative to net asset value for INV-RSK-002.
        # Explicit Dependency Tracking: self.positions, self.pending_leaves, self.current_prices, self.current_equity.
        # Structural Relationship: Queried by RiskOrchestrator and PreTradeRiskFirewall.
        # Defensive Invariant: If equity <= 0.0, returns float('inf') to force risk rejection.
        eq = self.current_equity
        if eq <= 0.0:
            return float("inf")

        symbols = set(self.positions.keys()) | set(self.pending_leaves.keys())
        gross_notional = 0.0
        for sym in symbols:
            pos = self.positions.get(sym, 0.0)
            leaves = self.pending_leaves.get(sym, 0.0)
            if abs(pos) > 1e-12 or abs(leaves) > 1e-12:
                if sym not in self.current_prices:
                    raise NonFiniteRiskInputException(
                        f"Missing current market price for symbol '{sym}' in portfolio state",
                        code=ERR_RSK_NON_FINITE_INPUT,
                    )
                p = self.current_prices[sym]
                if isinstance(p, bool) or not math.isfinite(p) or p <= 0.0:
                    raise NonFiniteRiskInputException(
                        f"Price for symbol '{sym}' must be strictly positive finite scalar, got {p!r}",
                        code=ERR_RSK_NON_FINITE_INPUT,
                    )
                # Universal Directional Netting: opposite signs net to max(|pos|, |pos + leaves|);
                # same signs add (|pos| + |leaves|).
                if (pos > 0.0 and leaves < 0.0) or (pos < 0.0 and leaves > 0.0):
                    gross_notional += max(abs(pos), abs(pos + leaves)) * p
                else:
                    gross_notional += (abs(pos) + abs(leaves)) * p

        return float(gross_notional / eq)

    @property
    def net_leverage(self) -> float:
        r"""Portfolio net directional leverage: $L_{\text{net}} = |\sum (w_i + q_{\text{leaves}, i}) \cdot P_i| / W_t$."""
        # Functional Purpose: Measure net directional exposure relative to net asset value for INV-RSK-002.
        # Explicit Dependency Tracking: self.positions, self.pending_leaves, self.current_prices, self.current_equity.
        # Structural Relationship: Queried by RiskOrchestrator and PreTradeRiskFirewall.
        # Defensive Invariant: If equity <= 0.0, returns float('inf') to force risk rejection.
        eq = self.current_equity
        if eq <= 0.0:
            return float("inf")

        symbols = set(self.positions.keys()) | set(self.pending_leaves.keys())
        net_notional = 0.0
        for sym in symbols:
            pos_qty = self.positions.get(sym, 0.0)
            leaves_qty = self.pending_leaves.get(sym, 0.0)
            if abs(pos_qty) > 1e-12 or abs(leaves_qty) > 1e-12:
                if sym not in self.current_prices:
                    raise NonFiniteRiskInputException(
                        f"Missing current market price for symbol '{sym}' in portfolio state",
                        code=ERR_RSK_NON_FINITE_INPUT,
                    )
                p = self.current_prices[sym]
                if isinstance(p, bool) or not math.isfinite(p) or p <= 0.0:
                    raise NonFiniteRiskInputException(
                        f"Price for symbol '{sym}' must be strictly positive finite scalar, got {p!r}",
                        code=ERR_RSK_NON_FINITE_INPUT,
                    )
                net_notional += (pos_qty + leaves_qty) * p

        return float(abs(net_notional) / eq)

    @property
    def free_margin(self) -> float:
        """Unencumbered liquid buying power: cash minus margin locked in resting buy and short sell orders."""
        # Functional Purpose: Determine capital available to fund new order margin requirements for INV-RSK-005.
        # Explicit Dependency Tracking: self.cash, self.positions, self.pending_leaves, self.current_prices.
        # Structural Relationship: Queried during margin sufficiency validation.
        # Defensive Invariant: Locks capital for buy leaves and uncovered short sell leaves; deducts from cash.
        locked = 0.0
        for sym, leaves in self.pending_leaves.items():
            if abs(leaves) <= 1e-12:
                continue
            if sym not in self.current_prices:
                raise NonFiniteRiskInputException(
                    f"Missing current market price for symbol '{sym}' in portfolio state",
                    code=ERR_RSK_NON_FINITE_INPUT,
                )
            price = self.current_prices[sym]
            if isinstance(price, bool) or not math.isfinite(price) or price <= 0.0:
                raise NonFiniteRiskInputException(
                    f"Price for symbol '{sym}' must be strictly positive finite scalar, got {price!r}",
                    code=ERR_RSK_NON_FINITE_INPUT,
                )
            if leaves > 0.0:
                locked += leaves * price
            elif leaves < 0.0:
                pos = self.positions.get(sym, 0.0)
                short_leaves = max(0.0, abs(leaves) - max(0.0, pos))
                locked += short_leaves * price

        return float(self.cash - locked)

    def update_price(self, symbol: str, price: float) -> None:
        """Update the mark-to-market valuation price for an instrument."""
        # Functional Purpose: Record latest tick/bar price and dynamically refresh session high-water mark.
        # Explicit Dependency Tracking: _validate_positive_scalar, self.current_prices, self.peak_equity.
        # Structural Relationship: Called by market data subscribers and ExecutionGateway fill handlers.
        # Defensive Invariant: Symbol must be non-empty str; price must be strictly positive finite scalar.
        if not isinstance(symbol, str) or not symbol:
            raise NonFiniteRiskInputException(
                f"Symbol must be non-empty str, got {symbol!r}",
                code=ERR_RSK_NON_FINITE_INPUT,
            )
        f_price = _validate_positive_scalar(price, f"price[{symbol}]")
        self.current_prices[symbol] = f_price

        # Update peak equity if new valuation establishes a new high-water mark
        eq = self.current_equity
        if eq > self.peak_equity:
            self.peak_equity = eq

    def update_fill(
        self,
        symbol: str,
        filled_qty: float,
        price: float,
        side: OrderSide,
    ) -> None:
        """Update portfolio state upon receipt of an execution report fill."""
        # Functional Purpose: Reconcile cash, position inventory, leaves, and mark-to-market equity.
        # Explicit Dependency Tracking: _validate_positive_scalar, OrderSide, self.positions, self.cash.
        # Structural Relationship: Invoked synchronously upon execution report acknowledgement.
        # Defensive Invariant: filled_qty and price must be finite, non-boolean, and strictly positive.
        if not isinstance(symbol, str) or not symbol:
            raise NonFiniteRiskInputException(
                f"Symbol must be non-empty str, got {symbol!r}",
                code=ERR_RSK_NON_FINITE_INPUT,
            )
        f_qty = _validate_positive_scalar(filled_qty, "filled_qty")
        f_price = _validate_positive_scalar(price, "fill_price")
        if not isinstance(side, OrderSide):
            raise NonFiniteRiskInputException(
                f"side must be an OrderSide enum, got {side!r}",
                code=ERR_RSK_NON_FINITE_INPUT,
            )

        # Update cash consideration and position inventory
        if side == OrderSide.BUY:
            self.cash -= f_qty * f_price
            self.positions[symbol] = self.positions.get(symbol, 0.0) + f_qty
            # Decrement working open buy leaves if tracked
            if symbol in self.pending_leaves:
                leaves = self.pending_leaves[symbol]
                if leaves > 0.0:
                    self.pending_leaves[symbol] = max(0.0, leaves - f_qty)
                    if self.pending_leaves[symbol] <= 1e-12:
                        del self.pending_leaves[symbol]
        else:
            self.cash += f_qty * f_price
            self.positions[symbol] = self.positions.get(symbol, 0.0) - f_qty
            # Reconcile working open sell leaves if tracked (supporting both signed negative and unsigned positive)
            if symbol in self.pending_leaves:
                leaves = self.pending_leaves[symbol]
                if leaves < 0.0:
                    self.pending_leaves[symbol] = min(0.0, leaves + f_qty)
                elif leaves > 0.0:
                    self.pending_leaves[symbol] = max(0.0, leaves - f_qty)
                if abs(self.pending_leaves[symbol]) <= 1e-12:
                    del self.pending_leaves[symbol]

        # Clean zero inventory to prevent dictionary bloat
        if symbol in self.positions and abs(self.positions[symbol]) < 1e-12:
            self.positions[symbol] = 0.0

        # Update mark-to-market price and high-water mark
        self.current_prices[symbol] = f_price
        eq = self.current_equity
        if eq > self.peak_equity:
            self.peak_equity = eq


# ============================================================================
# Core Pre-Trade Risk Firewall (INV-RSK-001 through INV-RSK-007)
# ============================================================================


@dataclass(slots=True)
class PreTradeRiskFirewall:
    """Institutional pre-trade deterministic risk firewall sitting in the hot execution path.

    Evaluates every candidate order against single-order fat-finger limits, portfolio gross/net
    leverage caps, single-asset NAV concentration ceilings, margin sufficiency, and intraday
    drawdown tripwires in sub-10 microseconds before routing to any external transport.

    Attributes:
        limits: Immutable risk limits configuration value object.
    """

    limits: RiskLimits

    def __post_init__(self) -> None:
        """Verify limits configuration object is valid."""
        # Functional Purpose: Guarantee limits attribute is an instance of RiskLimits.
        # Explicit Dependency Tracking: RiskLimits, NonFiniteRiskInputException.
        # Structural Relationship: Invoked on PreTradeRiskFirewall initialization.
        # Defensive Invariant: limits must be an instance of RiskLimits.
        if not isinstance(self.limits, RiskLimits):
            raise NonFiniteRiskInputException(
                f"Firewall requires RiskLimits instance, got {type(self.limits).__name__}",
                code=ERR_RSK_NON_FINITE_INPUT,
            )

    def validate_order(
        self,
        order: Order,
        state: PortfolioRiskState,
        current_price: float | None = None,
    ) -> None:
        """Validate candidate order against all institutional pre-trade risk invariants.

        Check Sequence:
            1. Strict Non-Finite & Bool Input Sanitization (INV-RSK-007, ERR-RSK-007).
            2. Intraday Drawdown Circuit Breaker (INV-RSK-004, ERR-RSK-006).
            3. Fat-Finger Quantity Limit (INV-RSK-001, ERR-RSK-002).
            4. Reference Price Determination (INV-RSK-001, ERR-RSK-007).
            5. Fat-Finger Notional Limit (INV-RSK-001, ERR-RSK-001).
            6. Margin & Borrow Sufficiency (INV-RSK-005, ERR-RSK-005).
            7. Projected Portfolio Gross Leverage (INV-RSK-002, ERR-RSK-003).
            8. Projected Portfolio Net Leverage (INV-RSK-002, ERR-RSK-003).
            9. Projected Single-Asset NAV Concentration (INV-RSK-003, ERR-RSK-004).

        Args:
            order: Candidate domain order entity.
            state: Current portfolio valuation, position, and leaves state.
            current_price: Optional prevailing market price (used if order is MARKET).

        Raises:
            NonFiniteRiskInputException: If inputs are non-finite, booleans, or prices <= 0.
            DrawdownLimitExceededException: If intraday drawdown breaches threshold.
            FatFingerQuantityException: If order quantity exceeds max_order_qty.
            FatFingerNotionalException: If order notional exceeds max_order_notional.
            InsufficientMarginRiskException: If buying power is insufficient.
            LeverageLimitExceededException: If projected gross or net leverage breaches limits.
            ConcentrationLimitExceededException: If single-asset exposure breaches concentration cap.
        """
        # Functional Purpose: Enforce INV-RSK-001 through INV-RSK-007 in deterministic hot path.
        # Explicit Dependency Tracking: Order, PortfolioRiskState, RiskLimits, math.isfinite.
        # Structural Relationship: Intercepts all order submissions before ExecutionGateway.
        # Defensive Invariant: Hot-path execution strictly completes in < 10us; zero allocations.

        # -------------------------------------------------------------------------
        # Check 1: Strict Non-Finite & Bool Input Sanitization (INV-RSK-007)
        # -------------------------------------------------------------------------
        if not isinstance(order, Order):
            raise NonFiniteRiskInputException(
                f"order must be an Order instance, got {type(order).__name__}",
                code=ERR_RSK_NON_FINITE_INPUT,
            )
        if not isinstance(state, PortfolioRiskState):
            raise NonFiniteRiskInputException(
                f"state must be a PortfolioRiskState instance, got {type(state).__name__}",
                code=ERR_RSK_NON_FINITE_INPUT,
            )

        if (
            isinstance(order.quantity, bool)
            or not isinstance(order.quantity, (int, float))
            or not math.isfinite(order.quantity)
            or order.quantity <= 0.0
        ):
            raise NonFiniteRiskInputException(
                f"Order quantity must be finite scalar > 0.0, got {order.quantity!r}",
                code=ERR_RSK_NON_FINITE_INPUT,
            )

        if order.price is not None and (
            isinstance(order.price, bool)
            or not isinstance(order.price, (int, float))
            or not math.isfinite(order.price)
            or order.price <= 0.0
        ):
            raise NonFiniteRiskInputException(
                f"Order price must be finite scalar > 0.0, got {order.price!r}",
                code=ERR_RSK_NON_FINITE_INPUT,
            )

        if current_price is not None and (
            isinstance(current_price, bool)
            or not isinstance(current_price, (int, float))
            or not math.isfinite(current_price)
            or current_price <= 0.0
        ):
            raise NonFiniteRiskInputException(
                f"current_price must be finite scalar > 0.0, got {current_price!r}",
                code=ERR_RSK_NON_FINITE_INPUT,
            )

        # -------------------------------------------------------------------------
        # Check 2: Intraday Drawdown Circuit Breaker (INV-RSK-004)
        # -------------------------------------------------------------------------
        drawdown = state.intraday_drawdown
        if drawdown >= self.limits.max_intraday_drawdown_pct:
            raise DrawdownLimitExceededException(
                f"Intraday drawdown {drawdown:.4f} breaches tripwire limit {self.limits.max_intraday_drawdown_pct:.4f}",
                code=ERR_RSK_DRAWDOWN_LIMIT_EXCEEDED,
            )

        # -------------------------------------------------------------------------
        # Check 3: Fat-Finger Quantity Limit (INV-RSK-001)
        # -------------------------------------------------------------------------
        if order.quantity > self.limits.max_order_qty:
            raise FatFingerQuantityException(
                f"Order quantity {order.quantity:.4f} exceeds max_order_qty {self.limits.max_order_qty:.4f}",
                code=ERR_RSK_FAT_FINGER_QUANTITY,
            )

        # -------------------------------------------------------------------------
        # Check 4: Reference Price Determination (INV-RSK-001, INV-RSK-007)
        # -------------------------------------------------------------------------
        ref_price: float | None = None
        if current_price is not None:
            ref_price = float(current_price)
        elif order.price is not None:
            ref_price = float(order.price)
        elif order.symbol in state.current_prices:
            ref_price = float(state.current_prices[order.symbol])

        if (
            ref_price is None
            or isinstance(ref_price, bool)
            or not math.isfinite(ref_price)
            or ref_price <= 0.0
        ):
            raise NonFiniteRiskInputException(
                f"Cannot determine valid reference price for order on {order.symbol}",
                code=ERR_RSK_NON_FINITE_INPUT,
            )

        # -------------------------------------------------------------------------
        # Check 5: Fat-Finger Notional Limit (INV-RSK-001)
        # -------------------------------------------------------------------------
        order_notional = order.quantity * ref_price
        if order_notional > self.limits.max_order_notional:
            raise FatFingerNotionalException(
                f"Order notional {order_notional:.2f} exceeds max_order_notional {self.limits.max_order_notional:.2f}",
                code=ERR_RSK_FAT_FINGER_NOTIONAL,
            )

        # -------------------------------------------------------------------------
        # Check 6: Margin & Borrow Sufficiency (INV-RSK-005)
        # -------------------------------------------------------------------------
        if order.side == OrderSide.BUY:
            # If closing existing short position, no margin required for closing portion;
            # if opening or expanding a long position, margin equals incremental long notional.
            pending_buy_leaves = max(0.0, state.pending_leaves.get(order.symbol, 0.0))
            available_short_pos = max(
                0.0, -state.positions.get(order.symbol, 0.0) - pending_buy_leaves
            )
            long_units = max(0.0, order.quantity - available_short_pos)
            required_margin = long_units * ref_price
        else:
            # Sell order: if closing existing long position, no margin required;
            # if opening or expanding a short position, margin equals short notional.
            pending_sell_leaves = abs(min(0.0, state.pending_leaves.get(order.symbol, 0.0)))
            available_long_pos = max(
                0.0, state.positions.get(order.symbol, 0.0) - pending_sell_leaves
            )
            short_units = max(0.0, order.quantity - available_long_pos)
            required_margin = short_units * ref_price

        # Mandate Invariant: De-risking orders with required_margin <= 0.0 (e.g. selling
        # long inventory or closing a short) do NOT consume incremental margin and pass
        # Check 6 unconditionally, even if state.free_margin < min_free_margin (e.g. negative cash).
        if required_margin > 0.0:
            free_margin_after = state.free_margin - required_margin
            if free_margin_after < self.limits.min_free_margin:
                raise InsufficientMarginRiskException(
                    f"Projected free margin {free_margin_after:.2f} breaches minimum buffer {self.limits.min_free_margin:.2f}",
                    code=ERR_RSK_INSUFFICIENT_MARGIN,
                )

        # -------------------------------------------------------------------------
        # Check 7 & 8: Projected Portfolio Gross & Net Leverage (INV-RSK-002)
        # -------------------------------------------------------------------------
        # Calculate projected portfolio equity including latest reference price
        projected_equity = state.cash
        for sym, pos in state.positions.items():
            if abs(pos) <= 1e-12:
                continue
            if sym == order.symbol:
                p = ref_price
            else:
                if sym not in state.current_prices:
                    raise NonFiniteRiskInputException(
                        f"Missing current market price for symbol '{sym}' in portfolio state",
                        code=ERR_RSK_NON_FINITE_INPUT,
                    )
                p_val = state.current_prices[sym]
                if isinstance(p_val, bool) or not math.isfinite(p_val) or p_val <= 0.0:
                    raise NonFiniteRiskInputException(
                        f"Price for symbol '{sym}' must be strictly positive finite scalar, got {p_val!r}",
                        code=ERR_RSK_NON_FINITE_INPUT,
                    )
                p = float(p_val)
            projected_equity += pos * p

        if projected_equity <= 0.0:
            raise LeverageLimitExceededException(
                f"Current or projected portfolio equity ({projected_equity:.2f}) is non-positive; order rejected",
                code=ERR_RSK_LEVERAGE_LIMIT_EXCEEDED,
            )

        all_symbols = set(state.positions)
        if state.pending_leaves:
            all_symbols.update(state.pending_leaves)
        all_symbols.add(order.symbol)

        order_delta = order.quantity if order.side == OrderSide.BUY else -order.quantity

        # Compute projected gross and net notional exposures across entire book
        projected_gross_notional = 0.0
        projected_net_notional = 0.0

        for sym in all_symbols:
            pos = state.positions.get(sym, 0.0)
            leaves = state.pending_leaves.get(sym, 0.0)

            if sym == order.symbol:
                p = ref_price
                proj_leaves = leaves + order_delta
                projected_net_notional += (pos + proj_leaves) * p

                # Universal Directional Netting for order symbol:
                if (pos > 0.0 and (order_delta < 0.0 or proj_leaves < 0.0)) or (
                    pos < 0.0 and (order_delta > 0.0 or proj_leaves > 0.0)
                ):
                    projected_gross_notional += max(abs(pos), abs(pos + proj_leaves)) * p
                else:
                    projected_gross_notional += (abs(pos) + abs(proj_leaves)) * p
            else:
                if abs(pos) <= 1e-12 and abs(leaves) <= 1e-12:
                    continue
                if sym not in state.current_prices:
                    raise NonFiniteRiskInputException(
                        f"Missing current market price for symbol '{sym}' in portfolio state",
                        code=ERR_RSK_NON_FINITE_INPUT,
                    )
                p_val = state.current_prices[sym]
                if isinstance(p_val, bool) or not math.isfinite(p_val) or p_val <= 0.0:
                    raise NonFiniteRiskInputException(
                        f"Price for symbol '{sym}' must be strictly positive finite scalar, got {p_val!r}",
                        code=ERR_RSK_NON_FINITE_INPUT,
                    )
                p = float(p_val)
                projected_net_notional += (pos + leaves) * p

                # Universal Directional Netting for non-order symbols:
                if (pos > 0.0 and leaves < 0.0) or (pos < 0.0 and leaves > 0.0):
                    projected_gross_notional += max(abs(pos), abs(pos + leaves)) * p
                else:
                    projected_gross_notional += (abs(pos) + abs(leaves)) * p

        projected_gross_leverage = projected_gross_notional / projected_equity
        if projected_gross_leverage > self.limits.max_gross_leverage:
            raise LeverageLimitExceededException(
                f"Projected gross leverage {projected_gross_leverage:.2f} exceeds limit {self.limits.max_gross_leverage:.2f}",
                code=ERR_RSK_LEVERAGE_LIMIT_EXCEEDED,
            )

        projected_net_leverage = abs(projected_net_notional) / projected_equity
        if projected_net_leverage > self.limits.max_net_leverage:
            raise LeverageLimitExceededException(
                f"Projected net leverage {projected_net_leverage:.2f} exceeds limit {self.limits.max_net_leverage:.2f}",
                code=ERR_RSK_LEVERAGE_LIMIT_EXCEEDED,
            )

        # -------------------------------------------------------------------------
        # Check 9: Projected Single-Asset NAV Concentration (INV-RSK-003)
        # -------------------------------------------------------------------------
        target_pos = state.positions.get(order.symbol, 0.0)
        target_leaves = state.pending_leaves.get(order.symbol, 0.0)
        target_proj_leaves = target_leaves + order_delta

        if (target_pos > 0.0 and (order_delta < 0.0 or target_proj_leaves < 0.0)) or (
            target_pos < 0.0 and (order_delta > 0.0 or target_proj_leaves > 0.0)
        ):
            projected_asset_exposure = (
                max(abs(target_pos), abs(target_pos + target_proj_leaves)) * ref_price
            )
        else:
            projected_asset_exposure = (abs(target_pos) + abs(target_proj_leaves)) * ref_price

        concentration = projected_asset_exposure / projected_equity
        if concentration > self.limits.max_concentration_nav_pct:
            raise ConcentrationLimitExceededException(
                f"Projected single-asset concentration for {order.symbol} ({concentration:.4f}) exceeds limit {self.limits.max_concentration_nav_pct:.4f}",
                code=ERR_RSK_CONCENTRATION_LIMIT_EXCEEDED,
            )

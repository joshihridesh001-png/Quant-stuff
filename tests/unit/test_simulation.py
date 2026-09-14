"""Unit tests for Simulation Domain Entities, Exceptions & Diagnostic Codes (Phase 5 Step 4 Task 1).

Tests the foundational domain models, frozen value objects, protocol contracts, exception
classes, and diagnostic fault codes in quant.analytics.simulation.
"""

from __future__ import annotations

import dataclasses
import gc
import math
import sys
import time
from typing import Any

import numpy as np
import pytest

from quant.analytics.circuit_breakers import (
    CircuitBreakerConfig,
    CircuitBreakerOverlayEngine,
)
from quant.analytics.ensemble import EnsembleConfig, RegimeConditionedDMAEngine
from quant.analytics.execution_sizing import (
    SizingConfig,
    SizingDecision,
    UnifiedConvexExecutionSizer,
)
from quant.analytics.simulation import (
    ERR_SIM_CAPITAL_RUIN,
    ERR_SIM_DIMENSION_MISMATCH,
    ERR_SIM_LOOKAHEAD_VIOLATION,
    ERR_SIM_NEGATIVE_FRICTION,
    ERR_SIM_NON_FINITE_INPUT,
    ERR_SIM_STARVATION,
    BarExecutionRecord,
    BenchmarkAuditor,
    BenchmarkAuditReport,
    BenchmarkComparison,
    DegenerateSimulationException,
    ExecutionCostModel,
    InfeasibleSimulationException,
    LookaheadViolationException,
    PortfolioLedger,
    ReplayEngine,
    SimulationConfig,
    SimulationError,
    SimulationListener,
)


class TestDiagnosticCodesAndExceptions:
    """Test suite verifying diagnostic codes and exception hierarchy."""

    def test_diagnostic_code_constants(self) -> None:
        """Verify diagnostic fault vector constants match specification verbatim."""
        assert ERR_SIM_LOOKAHEAD_VIOLATION == "ERR-SIM-001"
        assert ERR_SIM_NON_FINITE_INPUT == "ERR-SIM-002"
        assert ERR_SIM_CAPITAL_RUIN == "ERR-SIM-003"
        assert ERR_SIM_NEGATIVE_FRICTION == "ERR-SIM-004"
        assert ERR_SIM_STARVATION == "ERR-SIM-005"
        assert ERR_SIM_DIMENSION_MISMATCH == "ERR-SIM-006"

    def test_simulation_error_base(self) -> None:
        """Verify SimulationError base class properties."""
        err = SimulationError("Base error", code="ERR-SIM-000")
        assert isinstance(err, Exception)
        assert str(err) == "Base error"
        assert err.message == "Base error"
        assert err.code == "ERR-SIM-000"

    def test_lookahead_violation_exception(self) -> None:
        """Verify LookaheadViolationException defaults to ERR_SIM_LOOKAHEAD_VIOLATION."""
        err = LookaheadViolationException("Lookahead detected")
        assert isinstance(err, SimulationError)
        assert err.code == ERR_SIM_LOOKAHEAD_VIOLATION
        assert str(err) == "Lookahead detected"
        assert err.message == "Lookahead detected"

        # Explicit code override
        err_custom = LookaheadViolationException("Custom lookahead", code="ERR-CUSTOM")
        assert err_custom.code == "ERR-CUSTOM"

    def test_degenerate_simulation_exception(self) -> None:
        """Verify DegenerateSimulationException defaults to ERR_SIM_NON_FINITE_INPUT."""
        err = DegenerateSimulationException("Non-finite input encountered")
        assert isinstance(err, SimulationError)
        assert err.code == ERR_SIM_NON_FINITE_INPUT
        assert err.message == "Non-finite input encountered"

        # Ruin code override
        err_ruin = DegenerateSimulationException("Portfolio ruined", code=ERR_SIM_CAPITAL_RUIN)
        assert err_ruin.code == ERR_SIM_CAPITAL_RUIN

        # Starvation code override
        err_starve = DegenerateSimulationException("Starvation", code=ERR_SIM_STARVATION)
        assert err_starve.code == ERR_SIM_STARVATION

        # Dimension mismatch code override
        err_dim = DegenerateSimulationException("Mismatch", code=ERR_SIM_DIMENSION_MISMATCH)
        assert err_dim.code == ERR_SIM_DIMENSION_MISMATCH

    def test_infeasible_simulation_exception(self) -> None:
        """Verify InfeasibleSimulationException defaults to ERR_SIM_NEGATIVE_FRICTION."""
        err = InfeasibleSimulationException("Negative friction cost")
        assert isinstance(err, SimulationError)
        assert err.code == ERR_SIM_NEGATIVE_FRICTION
        assert err.message == "Negative friction cost"


class TestSimulationConfig:
    """Test suite for SimulationConfig domain value container."""

    def test_default_instantiation(self) -> None:
        """Verify default configuration values match institutional specification."""
        cfg = SimulationConfig()
        assert cfg.initial_capital == 1_000_000.0
        assert cfg.risk_free_rate == 0.02
        assert cfg.fee_bps == 2.0
        assert cfg.spread_bps == 1.0
        assert cfg.impact_coefficient == 0.10
        assert cfg.max_leverage == 1.0
        assert cfg.mdd_budget == 0.20
        assert cfg.confidence_level == 0.95
        assert cfg.annualization_factor == 252
        assert cfg.num_trials == 100

    def test_custom_valid_instantiation(self) -> None:
        """Verify custom valid configuration parameters are accepted."""
        cfg = SimulationConfig(
            initial_capital=500_000.0,
            risk_free_rate=0.03,
            fee_bps=1.5,
            spread_bps=0.5,
            impact_coefficient=0.05,
            max_leverage=2.5,
            mdd_budget=0.15,
            confidence_level=0.99,
            annualization_factor=365,
            num_trials=500,
        )
        assert cfg.initial_capital == 500_000.0
        assert cfg.risk_free_rate == 0.03
        assert cfg.fee_bps == 1.5
        assert cfg.spread_bps == 0.5
        assert cfg.impact_coefficient == 0.05
        assert cfg.max_leverage == 2.5
        assert cfg.mdd_budget == 0.15
        assert cfg.confidence_level == 0.99
        assert cfg.annualization_factor == 365
        assert cfg.num_trials == 500

    def test_frozen_immutability(self) -> None:
        """Verify SimulationConfig is immutable and enforces slots."""
        cfg = SimulationConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.initial_capital = 2_000_000.0  # type: ignore[misc]

    @pytest.mark.parametrize("invalid_capital", [0.0, -1000.0, -1e-6])
    def test_initial_capital_boundary_rejection(self, invalid_capital: float) -> None:
        """Verify non-positive initial capital is rejected with ERR_SIM_CAPITAL_RUIN."""
        with pytest.raises(DegenerateSimulationException) as exc_info:
            SimulationConfig(initial_capital=invalid_capital)
        assert exc_info.value.code == ERR_SIM_CAPITAL_RUIN

    @pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_float_rejection(self, non_finite: float) -> None:
        """Verify non-finite float fields are rejected with ERR_SIM_NON_FINITE_INPUT."""
        with pytest.raises(DegenerateSimulationException) as exc_info:
            SimulationConfig(initial_capital=non_finite)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        with pytest.raises(DegenerateSimulationException) as exc_info:
            SimulationConfig(risk_free_rate=non_finite)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        with pytest.raises(DegenerateSimulationException) as exc_info:
            SimulationConfig(fee_bps=non_finite)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        with pytest.raises(DegenerateSimulationException) as exc_info:
            SimulationConfig(spread_bps=non_finite)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        with pytest.raises(DegenerateSimulationException) as exc_info:
            SimulationConfig(impact_coefficient=non_finite)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        with pytest.raises(DegenerateSimulationException) as exc_info:
            SimulationConfig(max_leverage=non_finite)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        with pytest.raises(DegenerateSimulationException) as exc_info:
            SimulationConfig(mdd_budget=non_finite)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        with pytest.raises(DegenerateSimulationException) as exc_info:
            SimulationConfig(confidence_level=non_finite)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

    @pytest.mark.parametrize(
        ("field", "bad_val"),
        [
            ("fee_bps", -0.01),
            ("spread_bps", -1.0),
            ("impact_coefficient", -0.5),
        ],
    )
    def test_negative_friction_rejection(self, field: str, bad_val: float) -> None:
        """Verify negative friction parameters raise InfeasibleSimulationException (ERR-SIM-004)."""
        kwargs = {field: bad_val}
        with pytest.raises(InfeasibleSimulationException) as exc_info:
            SimulationConfig(**kwargs)
        assert exc_info.value.code == ERR_SIM_NEGATIVE_FRICTION

    @pytest.mark.parametrize("bad_rfr", [-0.01, -1.0])
    def test_negative_risk_free_rate_rejection(self, bad_rfr: float) -> None:
        """Verify negative risk free rate is rejected."""
        with pytest.raises(DegenerateSimulationException):
            SimulationConfig(risk_free_rate=bad_rfr)

    @pytest.mark.parametrize("bad_leverage", [0.0, -1.0, -0.5])
    def test_max_leverage_boundary_rejection(self, bad_leverage: float) -> None:
        """Verify non-positive max leverage is rejected."""
        with pytest.raises(DegenerateSimulationException):
            SimulationConfig(max_leverage=bad_leverage)

    @pytest.mark.parametrize("bad_mdd", [0.0, -0.1, 1.01, 2.0])
    def test_mdd_budget_bounds_rejection(self, bad_mdd: float) -> None:
        """Verify mdd_budget must be in (0.0, 1.0]."""
        with pytest.raises(DegenerateSimulationException):
            SimulationConfig(mdd_budget=bad_mdd)

    @pytest.mark.parametrize("bad_conf", [0.50, 0.49, 0.0, 1.0, 1.05])
    def test_confidence_level_bounds_rejection(self, bad_conf: float) -> None:
        """Verify confidence_level must be in (0.50, 1.0)."""
        with pytest.raises(DegenerateSimulationException):
            SimulationConfig(confidence_level=bad_conf)

    @pytest.mark.parametrize("bad_ann", [0, -1, -252])
    def test_annualization_factor_bounds_rejection(self, bad_ann: int) -> None:
        """Verify annualization_factor must be > 0."""
        with pytest.raises(DegenerateSimulationException):
            SimulationConfig(annualization_factor=bad_ann)

    @pytest.mark.parametrize("bad_trials", [0, -1, -100])
    def test_num_trials_bounds_rejection(self, bad_trials: int) -> None:
        """Verify num_trials must be >= 1."""
        with pytest.raises(DegenerateSimulationException):
            SimulationConfig(num_trials=bad_trials)

    @pytest.mark.parametrize(
        ("field", "bad_type_val"),
        [
            ("initial_capital", "1000000"),
            ("initial_capital", True),
            ("risk_free_rate", False),
            ("fee_bps", [1.0]),
            ("annualization_factor", 252.0),
            ("annualization_factor", True),
            ("num_trials", 10.5),
            ("num_trials", False),
        ],
    )
    def test_type_enforcement(self, field: str, bad_type_val: object) -> None:
        """Verify strict type enforcement guards against booleans and non-numerics."""
        kwargs = {field: bad_type_val}
        with pytest.raises(DegenerateSimulationException):
            SimulationConfig(**kwargs)


class TestBarExecutionRecord:
    """Test suite for BarExecutionRecord immutable domain record."""

    @pytest.fixture
    def valid_record_kwargs(self) -> dict[str, object]:
        """Provides valid baseline parameters for BarExecutionRecord."""
        return {
            "step_index": 0,
            "timestamp": 1_700_000_000_000_000_000,
            "gross_pnl": 500.0,
            "net_pnl": 480.0,
            "friction_cost": 20.0,
            "portfolio_equity": 1_000_480.0,
            "cash_balance": 900_000.0,
            "effective_leverage": 0.85,
            "drawdown": 0.05,
            "circuit_breaker_tier": "NORMAL",
            "circuit_breaker_haircut": 1.0,
            "target_allocations": (0.10, 0.20, 0.30),
            "discretized_allocations": (0.10, 0.20, 0.30),
        }

    def test_valid_instantiation(self, valid_record_kwargs: dict[str, object]) -> None:
        """Verify valid instantiation and property access."""
        rec = BarExecutionRecord(**valid_record_kwargs)  # type: ignore[arg-type]
        assert rec.step_index == 0
        assert rec.timestamp == 1_700_000_000_000_000_000
        assert rec.gross_pnl == 500.0
        assert rec.net_pnl == 480.0
        assert rec.friction_cost == 20.0
        assert rec.portfolio_equity == 1_000_480.0
        assert rec.cash_balance == 900_000.0
        assert rec.effective_leverage == 0.85
        assert rec.drawdown == 0.05
        assert rec.circuit_breaker_tier == "NORMAL"
        assert rec.circuit_breaker_haircut == 1.0
        assert rec.target_allocations == (0.10, 0.20, 0.30)
        assert rec.discretized_allocations == (0.10, 0.20, 0.30)

    def test_frozen_immutability(self, valid_record_kwargs: dict[str, object]) -> None:
        """Verify BarExecutionRecord is strictly frozen."""
        rec = BarExecutionRecord(**valid_record_kwargs)  # type: ignore[arg-type]
        with pytest.raises(dataclasses.FrozenInstanceError):
            rec.net_pnl = 1000.0  # type: ignore[misc]

    def test_negative_friction_rejection_inv_sim_003(
        self, valid_record_kwargs: dict[str, object]
    ) -> None:
        """Verify negative friction cost is rejected with ERR-SIM-004 (INV-SIM-003)."""
        valid_record_kwargs["friction_cost"] = -1e-5
        with pytest.raises(InfeasibleSimulationException) as exc_info:
            BarExecutionRecord(**valid_record_kwargs)  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_NEGATIVE_FRICTION

    @pytest.mark.parametrize("bad_drawdown", [-0.01, 1.01, -1.0, 2.0])
    def test_drawdown_bounds_rejection(
        self, valid_record_kwargs: dict[str, object], bad_drawdown: float
    ) -> None:
        """Verify drawdown must be in [0.0, 1.0]."""
        valid_record_kwargs["drawdown"] = bad_drawdown
        with pytest.raises(DegenerateSimulationException):
            BarExecutionRecord(**valid_record_kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_haircut", [-0.01, 1.01, -0.5, 1.5])
    def test_haircut_bounds_rejection(
        self, valid_record_kwargs: dict[str, object], bad_haircut: float
    ) -> None:
        """Verify circuit_breaker_haircut must be in [0.0, 1.0]."""
        valid_record_kwargs["circuit_breaker_haircut"] = bad_haircut
        with pytest.raises(DegenerateSimulationException):
            BarExecutionRecord(**valid_record_kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_leverage", [-0.01, -1.0])
    def test_effective_leverage_non_negative(
        self, valid_record_kwargs: dict[str, object], bad_leverage: float
    ) -> None:
        """Verify effective_leverage must be >= 0.0."""
        valid_record_kwargs["effective_leverage"] = bad_leverage
        with pytest.raises(DegenerateSimulationException):
            BarExecutionRecord(**valid_record_kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_step", [-1, -10])
    def test_step_index_non_negative(
        self, valid_record_kwargs: dict[str, object], bad_step: int
    ) -> None:
        """Verify step_index must be >= 0."""
        valid_record_kwargs["step_index"] = bad_step
        with pytest.raises(DegenerateSimulationException):
            BarExecutionRecord(**valid_record_kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_step_type", ["0", True, 1.5, None])
    def test_step_index_type_enforcement(
        self, valid_record_kwargs: dict[str, object], bad_step_type: object
    ) -> None:
        """Verify step_index strictly enforces integer type."""
        valid_record_kwargs["step_index"] = bad_step_type
        with pytest.raises(DegenerateSimulationException):
            BarExecutionRecord(**valid_record_kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_timestamp", [-1, -100])
    def test_timestamp_non_negative(
        self, valid_record_kwargs: dict[str, object], bad_timestamp: int
    ) -> None:
        """Verify timestamp must be >= 0."""
        valid_record_kwargs["timestamp"] = bad_timestamp
        with pytest.raises(DegenerateSimulationException):
            BarExecutionRecord(**valid_record_kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_timestamp_type", ["1000", False, 1000.5, None])
    def test_timestamp_type_enforcement(
        self, valid_record_kwargs: dict[str, object], bad_timestamp_type: object
    ) -> None:
        """Verify timestamp strictly enforces integer type."""
        valid_record_kwargs["timestamp"] = bad_timestamp_type
        with pytest.raises(DegenerateSimulationException):
            BarExecutionRecord(**valid_record_kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_tier", ["", "   ", None])
    def test_circuit_breaker_tier_non_empty(
        self, valid_record_kwargs: dict[str, object], bad_tier: object
    ) -> None:
        """Verify circuit_breaker_tier must be non-empty string."""
        valid_record_kwargs["circuit_breaker_tier"] = bad_tier
        with pytest.raises(DegenerateSimulationException):
            BarExecutionRecord(**valid_record_kwargs)  # type: ignore[arg-type]

    def test_allocation_dimension_mismatch_rejection(
        self, valid_record_kwargs: dict[str, object]
    ) -> None:
        """Verify target and discretized allocation dimension mismatch raises ERR-SIM-006."""
        valid_record_kwargs["target_allocations"] = (0.1, 0.2, 0.3)
        valid_record_kwargs["discretized_allocations"] = (0.1, 0.2)
        with pytest.raises(DegenerateSimulationException) as exc_info:
            BarExecutionRecord(**valid_record_kwargs)  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_DIMENSION_MISMATCH

    @pytest.mark.parametrize(
        "float_field",
        [
            "gross_pnl",
            "net_pnl",
            "friction_cost",
            "portfolio_equity",
            "cash_balance",
            "effective_leverage",
            "drawdown",
            "circuit_breaker_haircut",
        ],
    )
    def test_non_finite_scalars_rejection(
        self, valid_record_kwargs: dict[str, object], float_field: str
    ) -> None:
        """Verify NaN and Inf are rejected across all scalar float fields."""
        for bad_val in (float("nan"), float("inf"), float("-inf")):
            valid_record_kwargs[float_field] = bad_val
            with pytest.raises(DegenerateSimulationException) as exc_info:
                BarExecutionRecord(**valid_record_kwargs)  # type: ignore[arg-type]
            assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

    @pytest.mark.parametrize(
        "float_field",
        [
            "gross_pnl",
            "net_pnl",
            "friction_cost",
            "portfolio_equity",
            "cash_balance",
            "effective_leverage",
            "drawdown",
            "circuit_breaker_haircut",
        ],
    )
    def test_scalar_float_type_enforcement(
        self, valid_record_kwargs: dict[str, object], float_field: str
    ) -> None:
        """Verify non-numeric and boolean types are rejected for scalar float fields."""
        for bad_val in ("100.0", True, False, [1.0]):
            valid_record_kwargs[float_field] = bad_val
            with pytest.raises(DegenerateSimulationException):
                BarExecutionRecord(**valid_record_kwargs)  # type: ignore[arg-type]

    def test_allocation_container_type_rejection(
        self, valid_record_kwargs: dict[str, object]
    ) -> None:
        """Verify allocations must strictly be tuples, not lists."""
        valid_record_kwargs["target_allocations"] = [0.1, 0.2, 0.3]
        with pytest.raises(DegenerateSimulationException):
            BarExecutionRecord(**valid_record_kwargs)  # type: ignore[arg-type]

        valid_record_kwargs["target_allocations"] = (0.1, 0.2, 0.3)
        valid_record_kwargs["discretized_allocations"] = [0.1, 0.2, 0.3]
        with pytest.raises(DegenerateSimulationException):
            BarExecutionRecord(**valid_record_kwargs)  # type: ignore[arg-type]

    def test_non_finite_in_allocations_rejection(
        self, valid_record_kwargs: dict[str, object]
    ) -> None:
        """Verify non-finite numbers inside allocation tuples are rejected."""
        valid_record_kwargs["target_allocations"] = (0.1, float("nan"))
        valid_record_kwargs["discretized_allocations"] = (0.1, 0.2)
        with pytest.raises(DegenerateSimulationException) as exc_info:
            BarExecutionRecord(**valid_record_kwargs)  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        valid_record_kwargs["target_allocations"] = (0.1, 0.2)
        valid_record_kwargs["discretized_allocations"] = (0.1, float("inf"))
        with pytest.raises(DegenerateSimulationException) as exc_info:
            BarExecutionRecord(**valid_record_kwargs)  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT


class TestBenchmarkComparison:
    """Test suite for BenchmarkComparison domain entity."""

    @pytest.fixture
    def valid_comparison_kwargs(self) -> dict[str, object]:
        """Provides valid baseline parameters for BenchmarkComparison."""
        return {
            "name": "SPY_EqualWeight",
            "total_return": 0.15,
            "annualized_return": 0.14,
            "annualized_volatility": 0.18,
            "sharpe_ratio": 0.78,
            "max_drawdown": 0.12,
            "alpha": 0.04,
            "beta": 0.92,
            "tracking_error": 0.06,
            "information_ratio": 0.67,
        }

    def test_valid_instantiation(self, valid_comparison_kwargs: dict[str, object]) -> None:
        """Verify valid instantiation and property access."""
        comp = BenchmarkComparison(**valid_comparison_kwargs)  # type: ignore[arg-type]
        assert comp.name == "SPY_EqualWeight"
        assert comp.total_return == 0.15
        assert comp.annualized_volatility == 0.18
        assert comp.max_drawdown == 0.12

    def test_frozen_immutability(self, valid_comparison_kwargs: dict[str, object]) -> None:
        """Verify BenchmarkComparison is frozen."""
        comp = BenchmarkComparison(**valid_comparison_kwargs)  # type: ignore[arg-type]
        with pytest.raises(dataclasses.FrozenInstanceError):
            comp.alpha = 0.10  # type: ignore[misc]

    @pytest.mark.parametrize("bad_name", ["", "   ", None])
    def test_name_non_empty(
        self, valid_comparison_kwargs: dict[str, object], bad_name: object
    ) -> None:
        """Verify name must be a non-empty string."""
        valid_comparison_kwargs["name"] = bad_name
        with pytest.raises(DegenerateSimulationException):
            BenchmarkComparison(**valid_comparison_kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_vol", [-0.01, -1.0])
    def test_volatility_non_negative(
        self, valid_comparison_kwargs: dict[str, object], bad_vol: float
    ) -> None:
        """Verify annualized volatility must be >= 0.0."""
        valid_comparison_kwargs["annualized_volatility"] = bad_vol
        with pytest.raises(DegenerateSimulationException):
            BenchmarkComparison(**valid_comparison_kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_mdd", [-0.01, 1.01, -0.5, 2.0])
    def test_max_drawdown_bounds(
        self, valid_comparison_kwargs: dict[str, object], bad_mdd: float
    ) -> None:
        """Verify max drawdown must be in [0.0, 1.0]."""
        valid_comparison_kwargs["max_drawdown"] = bad_mdd
        with pytest.raises(DegenerateSimulationException):
            BenchmarkComparison(**valid_comparison_kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_te", [-0.01, -0.5])
    def test_tracking_error_non_negative(
        self, valid_comparison_kwargs: dict[str, object], bad_te: float
    ) -> None:
        """Verify tracking error must be >= 0.0."""
        valid_comparison_kwargs["tracking_error"] = bad_te
        with pytest.raises(DegenerateSimulationException):
            BenchmarkComparison(**valid_comparison_kwargs)  # type: ignore[arg-type]

    def test_non_finite_rejection(self, valid_comparison_kwargs: dict[str, object]) -> None:
        """Verify non-finite scalars are rejected with ERR-SIM-002."""
        valid_comparison_kwargs["sharpe_ratio"] = float("nan")
        with pytest.raises(DegenerateSimulationException) as exc_info:
            BenchmarkComparison(**valid_comparison_kwargs)  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

    @pytest.mark.parametrize(
        "field",
        [
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "alpha",
            "beta",
            "tracking_error",
            "information_ratio",
        ],
    )
    def test_type_enforcement(self, valid_comparison_kwargs: dict[str, object], field: str) -> None:
        """Verify invalid types are rejected."""
        valid_comparison_kwargs[field] = "0.5"
        with pytest.raises(DegenerateSimulationException):
            BenchmarkComparison(**valid_comparison_kwargs)  # type: ignore[arg-type]


class TestBenchmarkAuditReport:
    """Test suite for BenchmarkAuditReport tear sheet container."""

    @pytest.fixture
    def valid_report_kwargs(self) -> dict[str, object]:
        """Provides valid baseline parameters for BenchmarkAuditReport."""
        comp = BenchmarkComparison(
            name="Cash",
            total_return=0.02,
            annualized_return=0.02,
            annualized_volatility=0.001,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            alpha=0.0,
            beta=0.0,
            tracking_error=0.15,
            information_ratio=0.8,
        )
        return {
            "initial_capital": 1_000_000.0,
            "final_equity": 1_120_000.0,
            "total_return": 0.12,
            "cagr": 0.125,
            "annualized_volatility": 0.10,
            "sharpe_ratio": 1.25,
            "sortino_ratio": 1.80,
            "calmar_ratio": 2.50,
            "max_drawdown": 0.05,
            "realized_var_95": 0.015,
            "realized_cvar_95": 0.022,
            "realized_var_99": 0.025,
            "realized_cvar_99": 0.035,
            "tail_ratio": 1.15,
            "peak_leverage": 0.95,
            "deflated_sharpe_ratio": 0.965,
            "min_backtest_length": 180.0,
            "is_statistically_significant": True,
            "total_friction_cost": 2500.0,
            "circuit_breaker_counts": {"NORMAL": 95, "CAUTION": 5, "DERISK": 0, "HALT": 0},
            "benchmark_comparisons": {"Cash": comp},
        }

    def test_valid_instantiation(self, valid_report_kwargs: dict[str, object]) -> None:
        """Verify valid report instantiation and fields."""
        report = BenchmarkAuditReport(**valid_report_kwargs)  # type: ignore[arg-type]
        assert report.initial_capital == 1_000_000.0
        assert report.final_equity == 1_120_000.0
        assert report.deflated_sharpe_ratio == 0.965
        assert report.is_statistically_significant is True
        assert report.total_friction_cost == 2500.0
        assert "Cash" in report.benchmark_comparisons
        assert report.circuit_breaker_counts["NORMAL"] == 95

    def test_frozen_immutability(self, valid_report_kwargs: dict[str, object]) -> None:
        """Verify BenchmarkAuditReport is frozen."""
        report = BenchmarkAuditReport(**valid_report_kwargs)  # type: ignore[arg-type]
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.final_equity = 1_500_000.0  # type: ignore[misc]

    @pytest.mark.parametrize("bad_capital", [0.0, -100.0, -1e-5])
    def test_initial_capital_rejection(
        self, valid_report_kwargs: dict[str, object], bad_capital: float
    ) -> None:
        """Verify non-positive initial capital raises DegenerateSimulationException."""
        valid_report_kwargs["initial_capital"] = bad_capital
        with pytest.raises(DegenerateSimulationException) as exc_info:
            BenchmarkAuditReport(**valid_report_kwargs)  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_CAPITAL_RUIN

    @pytest.mark.parametrize("bad_cap_type", ["1000000", True, [1_000_000.0]])
    def test_initial_capital_type_enforcement(
        self, valid_report_kwargs: dict[str, object], bad_cap_type: object
    ) -> None:
        """Verify initial_capital type enforcement."""
        valid_report_kwargs["initial_capital"] = bad_cap_type
        with pytest.raises(DegenerateSimulationException):
            BenchmarkAuditReport(**valid_report_kwargs)  # type: ignore[arg-type]

    def test_initial_capital_non_finite(self, valid_report_kwargs: dict[str, object]) -> None:
        """Verify non-finite initial capital is rejected."""
        valid_report_kwargs["initial_capital"] = float("nan")
        with pytest.raises(DegenerateSimulationException):
            BenchmarkAuditReport(**valid_report_kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "float_field",
        [
            "final_equity",
            "total_return",
            "cagr",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "max_drawdown",
            "realized_var_95",
            "realized_cvar_95",
            "realized_var_99",
            "realized_cvar_99",
            "tail_ratio",
            "peak_leverage",
            "deflated_sharpe_ratio",
            "min_backtest_length",
            "total_friction_cost",
        ],
    )
    def test_report_float_fields_finiteness_and_type(
        self, valid_report_kwargs: dict[str, object], float_field: str
    ) -> None:
        """Verify non-finite and bad types in report float fields are rejected."""
        valid_report_kwargs[float_field] = float("nan")
        with pytest.raises(DegenerateSimulationException):
            BenchmarkAuditReport(**valid_report_kwargs)  # type: ignore[arg-type]

        valid_report_kwargs[float_field] = "0.5"
        with pytest.raises(DegenerateSimulationException):
            BenchmarkAuditReport(**valid_report_kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_dsr", [-0.01, 1.01, -0.5, 1.5])
    def test_deflated_sharpe_ratio_bounds(
        self, valid_report_kwargs: dict[str, object], bad_dsr: float
    ) -> None:
        """Verify deflated_sharpe_ratio must be in [0.0, 1.0]."""
        valid_report_kwargs["deflated_sharpe_ratio"] = bad_dsr
        with pytest.raises(DegenerateSimulationException):
            BenchmarkAuditReport(**valid_report_kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_mdd", [-0.01, 1.01])
    def test_max_drawdown_bounds(
        self, valid_report_kwargs: dict[str, object], bad_mdd: float
    ) -> None:
        """Verify max_drawdown must be in [0.0, 1.0]."""
        valid_report_kwargs["max_drawdown"] = bad_mdd
        with pytest.raises(DegenerateSimulationException):
            BenchmarkAuditReport(**valid_report_kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_vol", [-0.01, -1.0])
    def test_annualized_volatility_bounds(
        self, valid_report_kwargs: dict[str, object], bad_vol: float
    ) -> None:
        """Verify annualized_volatility must be non-negative."""
        valid_report_kwargs["annualized_volatility"] = bad_vol
        with pytest.raises(DegenerateSimulationException):
            BenchmarkAuditReport(**valid_report_kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_tail", [-0.01, -1.0])
    def test_tail_ratio_bounds(
        self, valid_report_kwargs: dict[str, object], bad_tail: float
    ) -> None:
        """Verify tail_ratio must be non-negative."""
        valid_report_kwargs["tail_ratio"] = bad_tail
        with pytest.raises(DegenerateSimulationException):
            BenchmarkAuditReport(**valid_report_kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_peak_lev", [-0.01, -1.0])
    def test_peak_leverage_bounds(
        self, valid_report_kwargs: dict[str, object], bad_peak_lev: float
    ) -> None:
        """Verify peak_leverage must be non-negative."""
        valid_report_kwargs["peak_leverage"] = bad_peak_lev
        with pytest.raises(DegenerateSimulationException):
            BenchmarkAuditReport(**valid_report_kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_min_btl", [-0.01, -1.0])
    def test_min_backtest_length_bounds(
        self, valid_report_kwargs: dict[str, object], bad_min_btl: float
    ) -> None:
        """Verify min_backtest_length must be non-negative."""
        valid_report_kwargs["min_backtest_length"] = bad_min_btl
        with pytest.raises(DegenerateSimulationException):
            BenchmarkAuditReport(**valid_report_kwargs)  # type: ignore[arg-type]

    def test_min_backtest_length_permits_infinity(
        self, valid_report_kwargs: dict[str, object]
    ) -> None:
        """Verify min_backtest_length permits float('inf') for sub-benchmark strategies."""
        valid_report_kwargs["min_backtest_length"] = float("inf")
        report = BenchmarkAuditReport(**valid_report_kwargs)  # type: ignore[arg-type]
        assert report.min_backtest_length == float("inf")

    @pytest.mark.parametrize("bad_friction", [-0.01, -100.0])
    def test_total_friction_cost_non_negative(
        self, valid_report_kwargs: dict[str, object], bad_friction: float
    ) -> None:
        """Verify total_friction_cost must be >= 0.0 (ERR-SIM-004)."""
        valid_report_kwargs["total_friction_cost"] = bad_friction
        with pytest.raises(InfeasibleSimulationException) as exc_info:
            BenchmarkAuditReport(**valid_report_kwargs)  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_NEGATIVE_FRICTION

    def test_statistically_significant_type(self, valid_report_kwargs: dict[str, object]) -> None:
        """Verify is_statistically_significant must be strictly a bool."""
        valid_report_kwargs["is_statistically_significant"] = "True"
        with pytest.raises(DegenerateSimulationException):
            BenchmarkAuditReport(**valid_report_kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_cb_counts",
        [
            [1, 2, 3],
            {"": 10},
            {"NORMAL": -5},
            {"NORMAL": True},
            {"NORMAL": "5"},
        ],
    )
    def test_circuit_breaker_counts_validation(
        self, valid_report_kwargs: dict[str, object], bad_cb_counts: object
    ) -> None:
        """Verify circuit_breaker_counts must be dict of non-empty strings to non-negative ints."""
        valid_report_kwargs["circuit_breaker_counts"] = bad_cb_counts
        with pytest.raises(DegenerateSimulationException):
            BenchmarkAuditReport(**valid_report_kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_benchmarks",
        [
            [1, 2, 3],
            {"": None},
            {"Cash": "NotABenchmarkComparison"},
        ],
    )
    def test_benchmark_comparisons_validation(
        self, valid_report_kwargs: dict[str, object], bad_benchmarks: object
    ) -> None:
        """Verify benchmark_comparisons must be dict of non-empty strings to BenchmarkComparison."""
        valid_report_kwargs["benchmark_comparisons"] = bad_benchmarks
        with pytest.raises(DegenerateSimulationException):
            BenchmarkAuditReport(**valid_report_kwargs)  # type: ignore[arg-type]

    def test_defensive_dict_copying(self, valid_report_kwargs: dict[str, object]) -> None:
        """Verify mutating caller dictionaries does not mutate internal report state (ISSUE-SIM-002)."""
        cb_counts = {"NORMAL": 95, "CAUTION": 5}
        benchmarks = {
            "Cash": BenchmarkComparison(
                name="Cash",
                total_return=0.02,
                annualized_return=0.02,
                annualized_volatility=0.001,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                alpha=0.0,
                beta=0.0,
                tracking_error=0.15,
                information_ratio=0.8,
            )
        }
        valid_report_kwargs["circuit_breaker_counts"] = cb_counts
        valid_report_kwargs["benchmark_comparisons"] = benchmarks

        report = BenchmarkAuditReport(**valid_report_kwargs)  # type: ignore[arg-type]

        # Mutate external dicts
        cb_counts["NORMAL"] = 0
        cb_counts["HALT"] = 100
        benchmarks["New"] = benchmarks["Cash"]

        # Assert internal dicts remain intact
        assert report.circuit_breaker_counts["NORMAL"] == 95
        assert "HALT" not in report.circuit_breaker_counts
        assert "New" not in report.benchmark_comparisons


class TestSimulationListenerProtocol:
    """Test suite verifying SimulationListener protocol definition and structural subtyping."""

    def test_listener_protocol_conformance(self) -> None:
        """Verify a conforming observer class satisfies SimulationListener via runtime protocol."""

        class ConformingListener:
            def __init__(self) -> None:
                self.bar_starts: list[tuple[int, int]] = []
                self.decisions: list[tuple[int, object]] = []
                self.fills: list[tuple[int, BarExecutionRecord]] = []
                self.bar_ends: list[tuple[int, BarExecutionRecord]] = []

            def on_bar_start(self, step: int, timestamp: int) -> None:
                self.bar_starts.append((step, timestamp))

            def on_decision(self, step: int, decision: object) -> None:
                self.decisions.append((step, decision))

            def on_fill(self, step: int, record: BarExecutionRecord) -> None:
                self.fills.append((step, record))

            def on_bar_end(self, step: int, record: BarExecutionRecord) -> None:
                self.bar_ends.append((step, record))

        listener = ConformingListener()
        assert isinstance(listener, SimulationListener)

    def test_non_conforming_listener_fails_protocol(self) -> None:
        """Verify an object missing protocol methods is not recognized as SimulationListener."""

        class IncompleteListener:
            def on_bar_start(self, step: int, timestamp: int) -> None:
                pass

        incomplete = IncompleteListener()
        assert not isinstance(incomplete, SimulationListener)


class TestExecutionCostModel:
    """Comprehensive test suite for ExecutionCostModel (Phase 5 Step 4 Task 2)."""

    def test_default_parameters(self) -> None:
        """Verify default fee, spread, and impact parameters match institutional specification."""
        model = ExecutionCostModel()
        assert model.fee_bps == 2.0
        assert model.spread_bps == 1.0
        assert model.impact_coefficient == 0.10

    def test_custom_valid_parameters(self) -> None:
        """Verify custom valid configuration parameters are accepted and stored."""
        model = ExecutionCostModel(fee_bps=1.5, spread_bps=0.8, impact_coefficient=0.05)
        assert model.fee_bps == 1.5
        assert model.spread_bps == 0.8
        assert model.impact_coefficient == 0.05

    @pytest.mark.parametrize(
        ("param_name", "bad_val"),
        [
            ("fee_bps", -0.01),
            ("fee_bps", -1.0),
            ("spread_bps", -0.01),
            ("spread_bps", -1.0),
            ("impact_coefficient", -0.01),
            ("impact_coefficient", -0.5),
        ],
    )
    def test_constructor_negative_parameter_rejection(
        self, param_name: str, bad_val: float
    ) -> None:
        """Verify negative friction parameters raise InfeasibleSimulationException (INV-SIM-003 / ERR-SIM-004)."""
        kwargs = {param_name: bad_val}
        with pytest.raises(InfeasibleSimulationException) as exc_info:
            ExecutionCostModel(**kwargs)
        assert exc_info.value.code == ERR_SIM_NEGATIVE_FRICTION

    @pytest.mark.parametrize(
        ("param_name", "non_finite"),
        [
            ("fee_bps", float("nan")),
            ("fee_bps", float("inf")),
            ("fee_bps", float("-inf")),
            ("spread_bps", float("nan")),
            ("spread_bps", float("inf")),
            ("spread_bps", float("-inf")),
            ("impact_coefficient", float("nan")),
            ("impact_coefficient", float("inf")),
            ("impact_coefficient", float("-inf")),
        ],
    )
    def test_constructor_non_finite_rejection(self, param_name: str, non_finite: float) -> None:
        """Verify non-finite parameters raise DegenerateSimulationException (ERR-SIM-002)."""
        kwargs = {param_name: non_finite}
        with pytest.raises(DegenerateSimulationException) as exc_info:
            ExecutionCostModel(**kwargs)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

    @pytest.mark.parametrize(
        ("param_name", "bad_type"),
        [
            ("fee_bps", True),
            ("fee_bps", False),
            ("fee_bps", "2.0"),
            ("spread_bps", True),
            ("spread_bps", "1.0"),
            ("impact_coefficient", False),
            ("impact_coefficient", [0.10]),
        ],
    )
    def test_constructor_type_rejection(self, param_name: str, bad_type: object) -> None:
        """Verify boolean and non-numeric types are rejected in constructor (ERR-SIM-002)."""
        kwargs = {param_name: bad_type}
        with pytest.raises(DegenerateSimulationException) as exc_info:
            ExecutionCostModel(**kwargs)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

    def test_zero_trade_returns_zero(self) -> None:
        """Verify zero trade delta returns exactly 0.0 total cost and (0.0, 0.0, 0.0) breakdown."""
        model = ExecutionCostModel(fee_bps=2.0, spread_bps=1.0, impact_coefficient=0.10)
        delta = np.zeros(5, dtype=np.float64)
        vols = np.full(5, 0.02, dtype=np.float64)
        advs = np.full(5, 1_000_000.0, dtype=np.float64)

        cost = model.compute_cost(delta, vols, advs)
        assert cost == 0.0
        assert isinstance(cost, float)

        breakdown = model.compute_cost_breakdown(delta, vols, advs)
        assert breakdown == (0.0, 0.0, 0.0)
        assert all(isinstance(c, float) for c in breakdown)

        # Without ADV
        cost_no_adv = model.compute_cost(delta, vols)
        assert cost_no_adv == 0.0
        assert model.compute_cost_breakdown(delta, vols) == (0.0, 0.0, 0.0)

    def test_empty_arrays_returns_zero(self) -> None:
        """Verify empty trade vector (N=0) returns exactly 0.0 cost and (0.0, 0.0, 0.0) breakdown."""
        model = ExecutionCostModel()
        delta = np.array([], dtype=np.float64)
        vols = np.array([], dtype=np.float64)
        advs = np.array([], dtype=np.float64)

        assert model.compute_cost(delta, vols, advs) == 0.0
        assert model.compute_cost_breakdown(delta, vols, advs) == (0.0, 0.0, 0.0)

    def test_non_negative_friction_invariant_inv_sim_003(self) -> None:
        """Verify INV-SIM-003: cost >= 0.0 and all breakdown components >= 0.0."""
        model = ExecutionCostModel(fee_bps=2.0, spread_bps=1.0, impact_coefficient=0.10)
        rng = np.random.default_rng(seed=42)

        for _ in range(50):
            n = rng.integers(1, 20)
            delta = rng.normal(0.0, 10_000.0, size=n)
            vols = rng.uniform(0.005, 0.08, size=n)
            advs = rng.uniform(100_000.0, 10_000_000.0, size=n)

            cost = model.compute_cost(delta, vols, advs)
            assert cost >= 0.0

            fee, spread, impact = model.compute_cost_breakdown(delta, vols, advs)
            assert fee >= 0.0
            assert spread >= 0.0
            assert impact >= 0.0
            assert math.isclose(cost, fee + spread + impact, rel_tol=1e-12, abs_tol=1e-12)

    def test_symmetry(self) -> None:
        """Verify positive and negative trade deltas of identical magnitude produce identical friction."""
        model = ExecutionCostModel(fee_bps=2.0, spread_bps=1.0, impact_coefficient=0.10)
        delta_pos = np.array([500.0, 1200.0, 300.0], dtype=np.float64)
        delta_neg = -delta_pos
        vols = np.array([0.02, 0.03, 0.015], dtype=np.float64)
        advs = np.array([1_000_000.0, 2_000_000.0, 500_000.0], dtype=np.float64)

        cost_pos = model.compute_cost(delta_pos, vols, advs)
        cost_neg = model.compute_cost(delta_neg, vols, advs)
        assert math.isclose(cost_pos, cost_neg, rel_tol=1e-14, abs_tol=1e-14)

        bd_pos = model.compute_cost_breakdown(delta_pos, vols, advs)
        bd_neg = model.compute_cost_breakdown(delta_neg, vols, advs)
        for c1, c2 in zip(bd_pos, bd_neg, strict=True):
            assert math.isclose(c1, c2, rel_tol=1e-14, abs_tol=1e-14)

    def test_monotonicity(self) -> None:
        """Verify strictly larger trade positions produce strictly larger execution costs."""
        model = ExecutionCostModel(fee_bps=2.0, spread_bps=1.0, impact_coefficient=0.10)
        delta_small = np.array([100.0, 200.0], dtype=np.float64)
        delta_medium = np.array([200.0, 400.0], dtype=np.float64)
        delta_large = np.array([500.0, 1000.0], dtype=np.float64)
        vols = np.array([0.02, 0.025], dtype=np.float64)

        cost_small = model.compute_cost(delta_small, vols)
        cost_medium = model.compute_cost(delta_medium, vols)
        cost_large = model.compute_cost(delta_large, vols)

        assert cost_small < cost_medium < cost_large

    def test_3_2_power_nonlinear_impact_scaling(self) -> None:
        """Verify 3/2-power non-linear impact scaling: f(4x) = 8 f(x) > 4 f(x)."""
        model = ExecutionCostModel(fee_bps=2.0, spread_bps=1.0, impact_coefficient=0.10)
        delta = np.array([100.0, 250.0], dtype=np.float64)
        vols = np.array([0.02, 0.03], dtype=np.float64)

        _, _, impact_1 = model.compute_cost_breakdown(delta, vols)
        _, _, impact_4 = model.compute_cost_breakdown(4.0 * delta, vols)

        # 4^(1.5) = (sqrt(4))^3 = 2^3 = 8.0
        expected_ratio = 8.0
        actual_ratio = impact_4 / impact_1
        assert math.isclose(actual_ratio, expected_ratio, rel_tol=1e-12)
        assert impact_4 > 4.0 * impact_1

    def test_adv_scaling_dampening(self) -> None:
        """Verify inverse square root ADV scaling: quadrupling ADV halves the impact cost."""
        model = ExecutionCostModel(fee_bps=2.0, spread_bps=1.0, impact_coefficient=0.10)
        delta = np.array([500.0, 1000.0], dtype=np.float64)
        vols = np.array([0.02, 0.03], dtype=np.float64)
        advs_1 = np.array([1_000_000.0, 4_000_000.0], dtype=np.float64)
        advs_4 = 4.0 * advs_1

        _, _, impact_1 = model.compute_cost_breakdown(delta, vols, advs_1)
        _, _, impact_4 = model.compute_cost_breakdown(delta, vols, advs_4)

        # lambda scales as 1 / sqrt(ADV), so 1 / sqrt(4 * ADV) = 0.5 * (1 / sqrt(ADV))
        expected_ratio = 0.5
        actual_ratio = impact_4 / impact_1
        assert math.isclose(actual_ratio, expected_ratio, rel_tol=1e-12)

    def test_impact_without_adv(self) -> None:
        """Verify impact calculation when advs is None uses impact_coefficient directly."""
        model = ExecutionCostModel(fee_bps=0.0, spread_bps=0.0, impact_coefficient=0.10)
        delta = np.array([100.0], dtype=np.float64)
        vols = np.array([0.02], dtype=np.float64)

        # cost = 0.10 * 0.02 * (100^(1.5)) = 0.002 * 1000 = 2.0
        cost = model.compute_cost(delta, vols)
        assert math.isclose(cost, 2.0, rel_tol=1e-12)

    def test_exact_hand_calculated_breakdown(self) -> None:
        """Verify exact analytical values against manual arithmetic derivation."""
        # Setup: fee_bps = 2.0 (0.0002), spread_bps = 1.0 (0.00005 half spread), impact = 0.10
        model = ExecutionCostModel(fee_bps=2.0, spread_bps=1.0, impact_coefficient=0.10)
        delta = np.array([100.0], dtype=np.float64)
        vols = np.array([0.02], dtype=np.float64)
        advs = np.array([10_000.0], dtype=np.float64)

        # L1 norm = 100.0
        # fee = 2.0 * 1e-4 * 100.0 = 0.02
        # spread = 0.5 * 1.0 * 1e-4 * 100.0 = 0.005
        # lambda = 0.10 / sqrt(10000.0) = 0.10 / 100.0 = 0.001
        # |delta|^(3/2) = 100.0 * 10.0 = 1000.0
        # impact = 0.001 * 0.02 * 1000.0 = 0.02
        # total = 0.02 + 0.005 + 0.02 = 0.045
        fee, spread, impact = model.compute_cost_breakdown(delta, vols, advs)
        assert math.isclose(fee, 0.02, rel_tol=1e-12)
        assert math.isclose(spread, 0.005, rel_tol=1e-12)
        assert math.isclose(impact, 0.02, rel_tol=1e-12)

        total = model.compute_cost(delta, vols, advs)
        assert math.isclose(total, 0.045, rel_tol=1e-12)

    def test_dimension_mismatch_rejection(self) -> None:
        """Verify dimension mismatches between delta, vols, and advs raise ERR-SIM-006."""
        model = ExecutionCostModel()
        delta = np.ones(3, dtype=np.float64)
        vols_bad = np.ones(4, dtype=np.float64)
        advs_bad = np.ones(2, dtype=np.float64)
        vols_good = np.ones(3, dtype=np.float64)

        # delta vs vols mismatch
        with pytest.raises(DegenerateSimulationException) as exc_info:
            model.compute_cost(delta, vols_bad)
        assert exc_info.value.code == ERR_SIM_DIMENSION_MISMATCH

        with pytest.raises(DegenerateSimulationException) as exc_info:
            model.compute_cost_breakdown(delta, vols_bad)
        assert exc_info.value.code == ERR_SIM_DIMENSION_MISMATCH

        # delta vs advs mismatch
        with pytest.raises(DegenerateSimulationException) as exc_info:
            model.compute_cost(delta, vols_good, advs_bad)
        assert exc_info.value.code == ERR_SIM_DIMENSION_MISMATCH

        with pytest.raises(DegenerateSimulationException) as exc_info:
            model.compute_cost_breakdown(delta, vols_good, advs_bad)
        assert exc_info.value.code == ERR_SIM_DIMENSION_MISMATCH

    def test_multidimensional_arrays_rejection(self) -> None:
        """Verify non-1D arrays raise ERR-SIM-006."""
        model = ExecutionCostModel()
        delta_2d = np.ones((3, 1), dtype=np.float64)
        vols_1d = np.ones(3, dtype=np.float64)

        with pytest.raises(DegenerateSimulationException) as exc_info:
            model.compute_cost(delta_2d, vols_1d)
        assert exc_info.value.code == ERR_SIM_DIMENSION_MISMATCH

        with pytest.raises(DegenerateSimulationException) as exc_info:
            model.compute_cost(vols_1d, delta_2d)
        assert exc_info.value.code == ERR_SIM_DIMENSION_MISMATCH

        with pytest.raises(DegenerateSimulationException) as exc_info:
            model.compute_cost(vols_1d, vols_1d, delta_2d)
        assert exc_info.value.code == ERR_SIM_DIMENSION_MISMATCH

    def test_input_type_rejection(self) -> None:
        """Verify non-numpy ndarrays and invalid dtypes raise ERR-SIM-002."""
        model = ExecutionCostModel()
        vols = np.full(3, 0.02, dtype=np.float64)

        # List instead of ndarray
        with pytest.raises(DegenerateSimulationException) as exc_info:
            model.compute_cost([1.0, 2.0, 3.0], vols)  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # Boolean array in delta
        bool_arr = np.array([True, False, True])
        with pytest.raises(DegenerateSimulationException) as exc_info:
            model.compute_cost(bool_arr, vols)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # String / object array in delta
        obj_arr = np.array(["a", "b", "c"], dtype=object)
        with pytest.raises(DegenerateSimulationException) as exc_info:
            model.compute_cost(obj_arr, vols)  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # List instead of ndarray for asset_volatilities
        delta = np.ones(3, dtype=np.float64)
        with pytest.raises(DegenerateSimulationException) as exc_info:
            model.compute_cost(delta, [0.02, 0.02, 0.02])  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # Boolean array in asset_volatilities
        with pytest.raises(DegenerateSimulationException) as exc_info:
            model.compute_cost(delta, bool_arr)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # List instead of ndarray for advs
        with pytest.raises(DegenerateSimulationException) as exc_info:
            model.compute_cost(delta, vols, [1e6, 1e6, 1e6])  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # Boolean array in advs
        with pytest.raises(DegenerateSimulationException) as exc_info:
            model.compute_cost(delta, vols, bool_arr)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

    @pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_inputs_rejection(self, bad_val: float) -> None:
        """Verify NaN and Inf in delta, vols, or advs raise ERR-SIM-002."""
        model = ExecutionCostModel()
        vols = np.full(3, 0.02, dtype=np.float64)
        advs = np.full(3, 1_000_000.0, dtype=np.float64)

        # Non-finite in delta
        delta_bad = np.array([100.0, bad_val, 200.0], dtype=np.float64)
        with pytest.raises(DegenerateSimulationException) as exc_info:
            model.compute_cost(delta_bad, vols, advs)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # Non-finite in vols
        vols_bad = np.array([0.02, bad_val, 0.01], dtype=np.float64)
        with pytest.raises(DegenerateSimulationException) as exc_info:
            model.compute_cost(np.ones(3), vols_bad, advs)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # Non-finite in advs
        advs_bad = np.array([1e6, bad_val, 2e6], dtype=np.float64)
        with pytest.raises(DegenerateSimulationException) as exc_info:
            model.compute_cost(np.ones(3), vols, advs_bad)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

    def test_negative_volatility_rejection(self) -> None:
        """Verify negative volatility raises DegenerateSimulationException (ERR-SIM-002)."""
        model = ExecutionCostModel()
        delta = np.array([100.0, 200.0], dtype=np.float64)
        vols = np.array([0.02, -0.01], dtype=np.float64)

        with pytest.raises(DegenerateSimulationException) as exc_info:
            model.compute_cost(delta, vols)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        with pytest.raises(DegenerateSimulationException) as exc_info:
            model.compute_cost_breakdown(delta, vols)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

    @pytest.mark.parametrize("bad_adv", [0.0, -1.0, -1000.0])
    def test_zero_or_negative_adv_rejection(self, bad_adv: float) -> None:
        """Verify non-positive ADV (ADV_i <= 0.0) raises DegenerateSimulationException (ERR-SIM-002)."""
        model = ExecutionCostModel()
        delta = np.array([100.0, 200.0], dtype=np.float64)
        vols = np.array([0.02, 0.01], dtype=np.float64)
        advs = np.array([1_000_000.0, bad_adv], dtype=np.float64)

        with pytest.raises(DegenerateSimulationException) as exc_info:
            model.compute_cost(delta, vols, advs)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        with pytest.raises(DegenerateSimulationException) as exc_info:
            model.compute_cost_breakdown(delta, vols, advs)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

    def test_inv_sim_006_execution_cost_latency_sla(self) -> None:
        """Verify INV-SIM-006: Hot-path execution cost evaluation SLA for N=10 assets."""
        model = ExecutionCostModel(fee_bps=2.0, spread_bps=1.0, impact_coefficient=0.10)
        n = 10
        delta = np.ones(n, dtype=np.float64) * 5_000.0
        vols = np.full(n, 0.02, dtype=np.float64)
        advs = np.full(n, 1_000_000.0, dtype=np.float64)

        # Warm-up JIT and caches
        for _ in range(50):
            model.compute_cost(delta, vols, advs)
            model.compute_cost_breakdown(delta, vols, advs)

        gc_was_enabled = gc.isenabled()
        gc.collect()
        gc.disable()
        old_trace = sys.gettrace()
        num_batches = 10
        batch_size = 100
        latencies_ms: list[float] = []
        try:
            sys.settrace(None)
            for _ in range(num_batches):
                t0 = time.perf_counter()
                for _ in range(batch_size):
                    model.compute_cost(delta, vols, advs)
                latencies_ms.append(((time.perf_counter() - t0) / batch_size) * 1000.0)
        finally:
            sys.settrace(old_trace)
            if gc_was_enabled:
                gc.enable()

        is_traced = (
            old_trace is not None
            or "coverage" in sys.modules
            or "pytest_cov" in sys.modules
            or (
                hasattr(sys, "monitoring")
                and any(sys.monitoring.get_tool(i) is not None for i in range(6))
            )
        )
        # Hot-path SLA threshold: <= 0.150ms (150 microseconds) under tracing, <= 0.025ms without tracing
        threshold_ms = 0.150 if is_traced else 0.025
        min_latency = float(np.min(latencies_ms))
        assert min_latency <= threshold_ms, (
            f"INV-SIM-006 SLA breached: min evaluation took {min_latency:.5f}ms > {threshold_ms}ms"
        )


class TestPortfolioLedger:
    """Comprehensive test suite for PortfolioLedger (Phase 5 Step 4 Task 3).

    Verifies causal portfolio accounting, capital conservation (INV-SIM-002),
    ruin detection (ERR-SIM-003), drawdown & HWM tracking, out-of-order rejection (ERR-SIM-001),
    dimension consistency (ERR-SIM-006), non-finite protection (ERR-SIM-002), and hot-path latency SLA (INV-SIM-006).
    """

    def test_constructor_default_initial_capital(self) -> None:
        """Verify default initial capital is 1,000,000.0 and properties are properly initialized."""
        ledger = PortfolioLedger()
        assert ledger.initial_capital == 1_000_000.0
        assert ledger.current_equity == 1_000_000.0
        assert ledger.current_cash == 1_000_000.0
        assert len(ledger.current_positions) == 0
        assert ledger.high_water_mark == 1_000_000.0
        assert ledger.current_drawdown == 0.0
        assert len(ledger.history) == 0
        assert len(ledger.get_equity_curve()) == 0
        assert len(ledger.get_net_returns()) == 0

    def test_constructor_custom_initial_capital(self) -> None:
        """Verify custom positive initial capital is accepted."""
        ledger = PortfolioLedger(initial_capital=250_000.0)
        assert ledger.initial_capital == 250_000.0
        assert ledger.current_equity == 250_000.0
        assert ledger.current_cash == 250_000.0
        assert ledger.high_water_mark == 250_000.0
        assert ledger.current_drawdown == 0.0

    @pytest.mark.parametrize("bad_cap", [0.0, -1.0, -100_000.0])
    def test_constructor_non_positive_capital_rejection(self, bad_cap: float) -> None:
        """Verify initial capital <= 0 raises DegenerateSimulationException (ERR-SIM-003)."""
        with pytest.raises(DegenerateSimulationException) as exc_info:
            PortfolioLedger(initial_capital=bad_cap)
        assert exc_info.value.code == ERR_SIM_CAPITAL_RUIN

    @pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
    def test_constructor_non_finite_capital_rejection(self, bad_val: float) -> None:
        """Verify non-finite initial capital raises DegenerateSimulationException (ERR-SIM-002)."""
        with pytest.raises(DegenerateSimulationException) as exc_info:
            PortfolioLedger(initial_capital=bad_val)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

    @pytest.mark.parametrize("bad_type", [True, False, "1000000", [1_000_000.0], None])
    def test_constructor_type_rejection(self, bad_type: Any) -> None:
        """Verify non-numeric or bool initial capital raises DegenerateSimulationException (ERR-SIM-002)."""
        with pytest.raises(DegenerateSimulationException) as exc_info:
            PortfolioLedger(initial_capital=bad_type)  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

    def test_step_0_execution_mechanics_and_friction_deduction(self) -> None:
        """Verify bar 0 causal mechanics: gross PnL is 0.0, friction deducted from cash and equity."""
        init_cap = 1_000_000.0
        ledger = PortfolioLedger(initial_capital=init_cap)
        ret_0 = np.array([0.05, -0.02, 0.01], dtype=np.float64)
        pos_0 = np.array([200_000.0, 100_000.0, 50_000.0], dtype=np.float64)
        target_0 = np.array([200_000.0, 100_000.0, 50_000.0], dtype=np.float64)
        friction_0 = 125.50

        record = ledger.update(
            step_index=0,
            timestamp=1_700_000_000_000,
            return_vector=ret_0,
            new_positions=pos_0,
            friction_cost=friction_0,
            circuit_breaker_tier="NORMAL",
            circuit_breaker_haircut=1.0,
            target_allocations=target_0,
        )

        # Causal property: gross PnL must be bit-exact 0.0 because prior positions were 0
        assert record.gross_pnl == 0.0
        assert record.net_pnl == -friction_0
        expected_equity = init_cap - friction_0
        assert record.portfolio_equity == expected_equity
        assert ledger.current_equity == expected_equity

        # Conservation of capital INV-SIM-002
        expected_cash = expected_equity - float(np.sum(pos_0))
        assert record.cash_balance == expected_cash
        assert ledger.current_cash == expected_cash
        assert abs(record.portfolio_equity - (record.cash_balance + float(np.sum(pos_0)))) < 1e-5

        # Effective leverage = ||pos||_1 / equity
        expected_leverage = float(np.sum(np.abs(pos_0))) / expected_equity
        assert abs(record.effective_leverage - expected_leverage) < 1e-6

        # HWM and Drawdown: HWM remains init_cap, drawdown = friction / init_cap
        assert ledger.high_water_mark == init_cap
        expected_dd = friction_0 / init_cap
        assert abs(record.drawdown - expected_dd) < 1e-6
        assert abs(ledger.current_drawdown - expected_dd) < 1e-6

        # Positions and History
        assert np.array_equal(ledger.current_positions, pos_0)
        assert len(ledger.history) == 1
        assert ledger.history[0] == record

        # Equity curve and returns
        curve = ledger.get_equity_curve()
        assert len(curve) == 1
        assert curve[0] == expected_equity
        rets = ledger.get_net_returns()
        assert len(rets) == 1
        assert abs(rets[0] - (-friction_0 / init_cap)) < 1e-6

    def test_step_0_ruin_from_friction_rejection(self) -> None:
        """Verify step 0 friction exceeding initial capital causes ruin (ERR-SIM-003)."""
        ledger = PortfolioLedger(initial_capital=100.0)
        ret = np.zeros(2, dtype=np.float64)
        pos = np.zeros(2, dtype=np.float64)

        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update(
                step_index=0,
                timestamp=1000,
                return_vector=ret,
                new_positions=pos,
                friction_cost=150.0,
                circuit_breaker_tier="NORMAL",
                circuit_breaker_haircut=1.0,
                target_allocations=pos,
            )
        assert exc_info.value.code == ERR_SIM_CAPITAL_RUIN

    def test_causal_pnl_lag_and_return_compounding(self) -> None:
        """Verify return r_t at bar t compounds strictly on positions held at bar t-1 (nu_{t-1})."""
        init_cap = 100_000.0
        ledger = PortfolioLedger(initial_capital=init_cap)

        # Bar 0: Enter position [50_000, 30_000]. Returns at bar 0 are NOT earned.
        pos_0 = np.array([50_000.0, 30_000.0], dtype=np.float64)
        ret_0 = np.array([0.10, -0.05], dtype=np.float64)
        f_0 = 50.0
        rec_0 = ledger.update(
            step_index=0,
            timestamp=1000,
            return_vector=ret_0,
            new_positions=pos_0,
            friction_cost=f_0,
            circuit_breaker_tier="NORMAL",
            circuit_breaker_haircut=1.0,
            target_allocations=pos_0,
        )
        assert rec_0.gross_pnl == 0.0
        w_0 = init_cap - f_0
        assert rec_0.portfolio_equity == w_0

        # Bar 1: Return vector r_1 = [0.02, 0.01]. Earned on pos_0!
        # gross_pnl_1 = 50_000 * 0.02 + 30_000 * 0.01 = 1_000 + 300 = 1_300.0
        pos_1 = np.array([40_000.0, 40_000.0], dtype=np.float64)
        ret_1 = np.array([0.02, 0.01], dtype=np.float64)
        f_1 = 20.0
        rec_1 = ledger.update(
            step_index=1,
            timestamp=2000,
            return_vector=ret_1,
            new_positions=pos_1,
            friction_cost=f_1,
            circuit_breaker_tier="NORMAL",
            circuit_breaker_haircut=1.0,
            target_allocations=pos_1,
        )
        expected_gross_1 = 50_000.0 * 0.02 + 30_000.0 * 0.01
        expected_net_1 = expected_gross_1 - f_1
        assert abs(rec_1.gross_pnl - expected_gross_1) < 1e-6
        assert abs(rec_1.net_pnl - expected_net_1) < 1e-6
        w_1 = w_0 + expected_net_1
        assert abs(rec_1.portfolio_equity - w_1) < 1e-6

        # Bar 2: Return vector r_2 = [-0.03, 0.05]. Earned on pos_1!
        # gross_pnl_2 = 40_000 * (-0.03) + 40_000 * (0.05) = -1200 + 2000 = 800.0
        pos_2 = np.array([0.0, 0.0], dtype=np.float64)
        ret_2 = np.array([-0.03, 0.05], dtype=np.float64)
        f_2 = 30.0
        rec_2 = ledger.update(
            step_index=2,
            timestamp=3000,
            return_vector=ret_2,
            new_positions=pos_2,
            friction_cost=f_2,
            circuit_breaker_tier="NORMAL",
            circuit_breaker_haircut=1.0,
            target_allocations=pos_2,
        )
        expected_gross_2 = 40_000.0 * (-0.03) + 40_000.0 * 0.05
        expected_net_2 = expected_gross_2 - f_2
        assert abs(rec_2.gross_pnl - expected_gross_2) < 1e-6
        assert abs(rec_2.net_pnl - expected_net_2) < 1e-6
        w_2 = w_1 + expected_net_2
        assert abs(rec_2.portfolio_equity - w_2) < 1e-6

        # Verify compounding matches telescopic product: prod(1 + r_t) == W_T / W_init
        rets = ledger.get_net_returns()
        compounded = float(np.prod(1.0 + rets))
        assert abs(compounded - (w_2 / init_cap)) < 1e-5

    def test_capital_conservation_inv_sim_002_multi_asset(self) -> None:
        """Verify capital conservation identity W_t == cash_t + sum(nu_t) across 100 bars with long and short positions."""
        np.random.seed(42)
        n_assets = 5
        n_bars = 100
        init_cap = 500_000.0
        ledger = PortfolioLedger(initial_capital=init_cap)

        for t in range(n_bars):
            # Returns random normal ~ N(0.0005, 0.015)
            r_t = np.random.normal(0.0005, 0.015, size=n_assets)
            # Mixed long and short positions bounded by 60% of equity
            target_p = np.random.uniform(-0.3, 0.3, size=n_assets) * ledger.current_equity
            f_t = float(np.random.uniform(5.0, 25.0))

            rec = ledger.update(
                step_index=t,
                timestamp=1000 * (t + 1),
                return_vector=r_t,
                new_positions=target_p,
                friction_cost=f_t,
                circuit_breaker_tier="NORMAL" if t < 80 else "MODERATE",
                circuit_breaker_haircut=1.0 if t < 80 else 0.75,
                target_allocations=target_p,
            )

            # INV-SIM-002: W_t == cash_t + sum(nu_i, t)
            pos_sum = float(np.sum(target_p))
            assert abs(rec.portfolio_equity - (rec.cash_balance + pos_sum)) < 1e-5
            assert abs(ledger.current_equity - (ledger.current_cash + pos_sum)) < 1e-5

    def test_drawdown_and_hwm_tracking(self) -> None:
        """Verify high-water mark monotonically expands on rallies and drawdown calculates peak-to-trough drop."""
        ledger = PortfolioLedger(initial_capital=100_000.0)
        pos = np.array([100_000.0], dtype=np.float64)

        # Bar 0: Enter position with 0 friction
        ledger.update(0, 1000, np.array([0.0]), pos, 0.0, "NORMAL", 1.0, pos)
        assert ledger.high_water_mark == 100_000.0
        assert ledger.current_drawdown == 0.0

        # Bar 1: Gain 20% -> equity = 120,000, HWM = 120,000, DD = 0
        ledger.update(1, 2000, np.array([0.20]), pos, 0.0, "NORMAL", 1.0, pos)
        assert ledger.current_equity == 120_000.0
        assert ledger.high_water_mark == 120_000.0
        assert ledger.current_drawdown == 0.0

        # Bar 2: Lose 10% (on 100k pos) -> gross PnL = -10,000 -> equity = 110,000
        # HWM stays 120,000, DD = (120k - 110k) / 120k = 10k / 120k = 1/12
        ledger.update(2, 3000, np.array([-0.10]), pos, 0.0, "NORMAL", 1.0, pos)
        assert ledger.current_equity == 110_000.0
        assert ledger.high_water_mark == 120_000.0
        assert abs(ledger.current_drawdown - (10_000.0 / 120_000.0)) < 1e-6

        # Bar 3: Lose another 20,000 -> equity = 90,000
        # HWM stays 120,000, DD = 30k / 120k = 0.25
        ledger.update(3, 4000, np.array([-0.20]), pos, 0.0, "NORMAL", 1.0, pos)
        assert ledger.current_equity == 90_000.0
        assert ledger.high_water_mark == 120_000.0
        assert abs(ledger.current_drawdown - 0.25) < 1e-6

        # Bar 4: Massive rally +40k -> equity = 130,000
        # HWM = 130,000, DD = 0.0
        ledger.update(4, 5000, np.array([0.40]), pos, 0.0, "NORMAL", 1.0, pos)
        assert ledger.current_equity == 130_000.0
        assert ledger.high_water_mark == 130_000.0
        assert ledger.current_drawdown == 0.0

    def test_total_ruin_detection_at_step_k(self) -> None:
        """Verify capital ruin (W_t <= 0.0) raises DegenerateSimulationException (ERR-SIM-003)."""
        ledger = PortfolioLedger(initial_capital=100_000.0)
        pos = np.array([100_000.0], dtype=np.float64)

        # Bar 0: Enter position
        ledger.update(0, 1000, np.array([0.0]), pos, 0.0, "NORMAL", 1.0, pos)

        # Bar 1: Return of -1.05 (wipeout) -> equity = 100k - 105k = -5k <= 0
        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update(1, 2000, np.array([-1.05]), pos, 0.0, "NORMAL", 1.0, pos)
        assert exc_info.value.code == ERR_SIM_CAPITAL_RUIN

    def test_strict_causal_sequencing_out_of_order_rejection(self) -> None:
        """Verify non-sequential step index raises LookaheadViolationException (ERR-SIM-001)."""
        ledger = PortfolioLedger(initial_capital=100_000.0)
        pos = np.array([10_000.0], dtype=np.float64)
        ret = np.array([0.01], dtype=np.float64)

        # Step 0 succeeds
        ledger.update(0, 1000, ret, pos, 0.0, "NORMAL", 1.0, pos)

        # Step 2 instead of Step 1 (temporal jump) raises ERR-SIM-001
        with pytest.raises(LookaheadViolationException) as exc_info:
            ledger.update(2, 2000, ret, pos, 0.0, "NORMAL", 1.0, pos)
        assert exc_info.value.code == ERR_SIM_LOOKAHEAD_VIOLATION

        # Step 0 repeated raises ERR-SIM-001
        with pytest.raises(LookaheadViolationException) as exc_info:
            ledger.update(0, 2000, ret, pos, 0.0, "NORMAL", 1.0, pos)
        assert exc_info.value.code == ERR_SIM_LOOKAHEAD_VIOLATION

    def test_temporal_jump_initial_step_must_be_zero(self) -> None:
        """Verify starting simulation at step != 0 raises LookaheadViolationException (ERR-SIM-001)."""
        ledger = PortfolioLedger(initial_capital=100_000.0)
        pos = np.array([10_000.0], dtype=np.float64)
        ret = np.array([0.01], dtype=np.float64)

        with pytest.raises(LookaheadViolationException) as exc_info:
            ledger.update(1, 1000, ret, pos, 0.0, "NORMAL", 1.0, pos)
        assert exc_info.value.code == ERR_SIM_LOOKAHEAD_VIOLATION

    def test_dimension_mismatch_rejection(self) -> None:
        """Verify mismatched vector dimensions raise DegenerateSimulationException (ERR-SIM-006)."""
        ledger = PortfolioLedger()
        ret = np.array([0.01, 0.02])
        pos_wrong = np.array([10_000.0, 20_000.0, 30_000.0])
        target = np.array([10_000.0, 20_000.0])

        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update(0, 1000, ret, pos_wrong, 0.0, "NORMAL", 1.0, target)
        assert exc_info.value.code == ERR_SIM_DIMENSION_MISMATCH

        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update(0, 1000, ret, target, 0.0, "NORMAL", 1.0, pos_wrong)
        assert exc_info.value.code == ERR_SIM_DIMENSION_MISMATCH

    def test_position_dimension_change_across_bars_rejection(self) -> None:
        """Verify asset universe size cannot change mid-simulation (ERR-SIM-006)."""
        ledger = PortfolioLedger()
        ret_2 = np.array([0.01, 0.02])
        pos_2 = np.array([10_000.0, 20_000.0])

        ledger.update(0, 1000, ret_2, pos_2, 0.0, "NORMAL", 1.0, pos_2)

        # Bar 1 with 3 assets
        ret_3 = np.array([0.01, 0.02, 0.03])
        pos_3 = np.array([10_000.0, 20_000.0, 30_000.0])
        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update(1, 2000, ret_3, pos_3, 0.0, "NORMAL", 1.0, pos_3)
        assert exc_info.value.code == ERR_SIM_DIMENSION_MISMATCH

    def test_multidimensional_array_rejection(self) -> None:
        """Verify 2D or higher dimensional arrays are rejected with ERR-SIM-006."""
        ledger = PortfolioLedger()
        ret_2d = np.array([[0.01], [0.02]])
        pos_1d = np.array([10_000.0, 20_000.0])

        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update(0, 1000, ret_2d, pos_1d, 0.0, "NORMAL", 1.0, pos_1d)
        assert exc_info.value.code == ERR_SIM_DIMENSION_MISMATCH

    @pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_inputs_rejection(self, bad_val: float) -> None:
        """Verify NaN/Inf in returns, positions, targets, or friction raises ERR-SIM-002."""
        ledger = PortfolioLedger()
        pos = np.array([10_000.0, 20_000.0])
        ret = np.array([0.01, 0.02])

        # Bad in returns
        ret_bad = np.array([0.01, bad_val])
        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update(0, 1000, ret_bad, pos, 0.0, "NORMAL", 1.0, pos)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # Bad in positions
        pos_bad = np.array([bad_val, 20_000.0])
        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update(0, 1000, ret, pos_bad, 0.0, "NORMAL", 1.0, pos)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # Bad in target allocations
        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update(0, 1000, ret, pos, 0.0, "NORMAL", 1.0, pos_bad)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # Bad in friction
        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update(0, 1000, ret, pos, bad_val, "NORMAL", 1.0, pos)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

    def test_negative_friction_cost_rejection(self) -> None:
        """Verify negative friction cost raises InfeasibleSimulationException (ERR-SIM-004)."""
        ledger = PortfolioLedger()
        pos = np.array([10_000.0, 20_000.0])
        ret = np.array([0.01, 0.02])

        with pytest.raises(InfeasibleSimulationException) as exc_info:
            ledger.update(0, 1000, ret, pos, -5.0, "NORMAL", 1.0, pos)
        assert exc_info.value.code == ERR_SIM_NEGATIVE_FRICTION

    def test_circuit_breaker_inputs_validation(self) -> None:
        """Verify invalid circuit breaker tier or haircut raises ERR-SIM-002."""
        ledger = PortfolioLedger()
        pos = np.array([10_000.0])
        ret = np.array([0.01])

        # Empty tier string
        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update(0, 1000, ret, pos, 0.0, "", 1.0, pos)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # Haircut > 1.0
        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update(0, 1000, ret, pos, 0.0, "NORMAL", 1.5, pos)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # Haircut < 0.0
        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update(0, 1000, ret, pos, 0.0, "NORMAL", -0.1, pos)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

    def test_defensive_copies_prevent_state_corruption(self) -> None:
        """Verify mutating arrays returned by properties does not corrupt ledger internal state."""
        ledger = PortfolioLedger(initial_capital=100_000.0)
        pos = np.array([10_000.0, 20_000.0])
        ret = np.array([0.01, 0.02])

        ledger.update(0, 1000, ret, pos, 0.0, "NORMAL", 1.0, pos)

        # Mutate current_positions copy
        positions_copy = ledger.current_positions
        positions_copy[0] = 999_999.0
        assert ledger.current_positions[0] == 10_000.0

        # Mutate history copy
        hist_copy = ledger.history
        hist_copy.clear()
        assert len(ledger.history) == 1

        # Mutate equity curve copy
        eq_curve = ledger.get_equity_curve()
        eq_curve[0] = 0.0
        assert ledger.get_equity_curve()[0] == 100_000.0

    def test_empty_asset_universe_support(self) -> None:
        """Verify ledger correctly handles empty asset universe (N=0 assets)."""
        ledger = PortfolioLedger(initial_capital=50_000.0)
        ret_0 = np.empty(0, dtype=np.float64)
        pos_0 = np.empty(0, dtype=np.float64)

        rec = ledger.update(
            step_index=0,
            timestamp=1000,
            return_vector=ret_0,
            new_positions=pos_0,
            friction_cost=0.0,
            circuit_breaker_tier="NORMAL",
            circuit_breaker_haircut=1.0,
            target_allocations=pos_0,
        )
        assert rec.gross_pnl == 0.0
        assert rec.net_pnl == 0.0
        assert rec.portfolio_equity == 50_000.0
        assert rec.cash_balance == 50_000.0
        assert rec.effective_leverage == 0.0
        assert rec.target_allocations == ()
        assert rec.discretized_allocations == ()

    def test_step_index_type_and_negative_rejection(self) -> None:
        """Verify invalid step_index types and negative step_index raise ERR-SIM-002."""
        ledger = PortfolioLedger()
        pos = np.array([10_000.0])
        ret = np.array([0.01])

        # Boolean step_index
        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update(True, 1000, ret, pos, 0.0, "NORMAL", 1.0, pos)  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # String step_index
        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update("0", 1000, ret, pos, 0.0, "NORMAL", 1.0, pos)  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # Negative step_index
        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update(-1, 1000, ret, pos, 0.0, "NORMAL", 1.0, pos)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

    def test_timestamp_type_and_negative_rejection(self) -> None:
        """Verify invalid timestamp types and negative timestamp raise ERR-SIM-002."""
        ledger = PortfolioLedger()
        pos = np.array([10_000.0])
        ret = np.array([0.01])

        # Boolean timestamp
        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update(0, False, ret, pos, 0.0, "NORMAL", 1.0, pos)  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # String timestamp
        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update(0, "1000", ret, pos, 0.0, "NORMAL", 1.0, pos)  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # Negative timestamp
        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update(0, -500, ret, pos, 0.0, "NORMAL", 1.0, pos)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

    def test_friction_cost_type_rejection(self) -> None:
        """Verify non-numeric or boolean friction cost raises ERR-SIM-002."""
        ledger = PortfolioLedger()
        pos = np.array([10_000.0])
        ret = np.array([0.01])

        # Boolean friction
        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update(0, 1000, ret, pos, True, "NORMAL", 1.0, pos)  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # String friction
        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update(0, 1000, ret, pos, "5.0", "NORMAL", 1.0, pos)  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

    def test_circuit_breaker_haircut_type_rejection(self) -> None:
        """Verify boolean or non-numeric circuit breaker haircut raises ERR-SIM-002."""
        ledger = PortfolioLedger()
        pos = np.array([10_000.0])
        ret = np.array([0.01])

        # Boolean haircut
        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update(0, 1000, ret, pos, 0.0, "NORMAL", True, pos)  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # String haircut
        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update(0, 1000, ret, pos, 0.0, "NORMAL", "0.5", pos)  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

    def test_array_input_type_and_dtype_rejection(self) -> None:
        """Verify list or non-numeric dtype array inputs raise ERR-SIM-002."""
        ledger = PortfolioLedger()
        pos = np.array([10_000.0])
        ret = np.array([0.01])

        # List instead of ndarray for return_vector
        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update(0, 1000, [0.01], pos, 0.0, "NORMAL", 1.0, pos)  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # List instead of ndarray for new_positions
        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update(0, 1000, ret, [10_000.0], 0.0, "NORMAL", 1.0, pos)  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # List instead of ndarray for target_allocations
        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update(0, 1000, ret, pos, 0.0, "NORMAL", 1.0, [10_000.0])  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # Boolean array for returns
        bool_arr = np.array([True])
        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update(0, 1000, bool_arr, pos, 0.0, "NORMAL", 1.0, pos)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # Object array with strings for positions
        str_arr = np.array(["10000.0"], dtype=object)
        with pytest.raises(DegenerateSimulationException) as exc_info:
            ledger.update(0, 1000, ret, str_arr, 0.0, "NORMAL", 1.0, pos)  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

    def test_inv_sim_006_portfolio_ledger_latency_sla(self) -> None:
        """Verify INV-SIM-006: Hot-path per-bar ledger update latency SLA <= 0.050ms (50 microseconds)."""
        n_assets = 10
        ledger = PortfolioLedger(initial_capital=1_000_000.0)
        ret = np.full(n_assets, 0.001, dtype=np.float64)
        pos = np.full(n_assets, 50_000.0, dtype=np.float64)

        # Warm up
        for s in range(20):
            ledger.update(s, 1000 + s, ret, pos, 1.0, "NORMAL", 1.0, pos)

        gc_was_enabled = gc.isenabled()
        gc.collect()
        gc.disable()
        old_trace = sys.gettrace()
        num_batches = 10
        batch_size = 100
        latencies_ms: list[float] = []

        fresh_ledger = PortfolioLedger(initial_capital=1_000_000.0)
        step = 0
        try:
            sys.settrace(None)
            for _ in range(num_batches):
                t0 = time.perf_counter()
                for _ in range(batch_size):
                    fresh_ledger.update(step, 1000 + step, ret, pos, 0.5, "NORMAL", 1.0, pos)
                    step += 1
                latencies_ms.append(((time.perf_counter() - t0) / batch_size) * 1000.0)
        finally:
            sys.settrace(old_trace)
            if gc_was_enabled:
                gc.enable()

        is_traced = (
            old_trace is not None
            or "coverage" in sys.modules
            or "pytest_cov" in sys.modules
            or (
                hasattr(sys, "monitoring")
                and any(sys.monitoring.get_tool(i) is not None for i in range(6))
            )
        )
        threshold_ms = 0.150 if is_traced else 0.050
        min_latency = float(np.min(latencies_ms))
        assert min_latency <= threshold_ms, (
            f"INV-SIM-006 SLA breached: min ledger update took {min_latency:.5f}ms > {threshold_ms}ms"
        )


class TestBenchmarkAuditor:
    """Test suite for BenchmarkAuditor institutional benchmarking & DSR certification engine."""

    def _create_synthetic_records(
        self,
        n_bars: int,
        initial_capital: float = 1_000_000.0,
        returns: np.ndarray | None = None,
        friction_per_bar: float = 10.0,
        circuit_breaker_tier: str = "NORMAL",
        leverage: float = 0.8,
    ) -> list[BarExecutionRecord]:
        """Helper to generate a sequence of valid BarExecutionRecord objects."""
        if returns is None:
            returns = np.full(n_bars, 0.001, dtype=np.float64)

        records: list[BarExecutionRecord] = []
        equity = initial_capital
        hwm = initial_capital
        for step in range(n_bars):
            ret = float(returns[step])
            gross_pnl = equity * ret
            net_pnl = gross_pnl - friction_per_bar
            equity += net_pnl
            hwm = max(hwm, equity)
            dd = max(0.0, min(1.0, (hwm - equity) / hwm))
            rec = BarExecutionRecord(
                step_index=step,
                timestamp=1_700_000_000_000_000_000 + step * 60_000_000_000,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                friction_cost=friction_per_bar,
                portfolio_equity=equity,
                cash_balance=equity * 0.2,
                effective_leverage=leverage,
                drawdown=dd,
                circuit_breaker_tier=circuit_breaker_tier,
                circuit_breaker_haircut=1.0,
                target_allocations=(equity * 0.4, equity * 0.4),
                discretized_allocations=(equity * 0.4, equity * 0.4),
            )
            records.append(rec)
        return records

    def test_constructor_valid(self) -> None:
        """Verify valid constructor instantiation with default and custom DSR engines."""
        cfg = SimulationConfig(initial_capital=500_000.0, confidence_level=0.99)
        auditor = BenchmarkAuditor(config=cfg)
        assert auditor.config == cfg
        assert auditor._dsr_engine is not None

        # Custom mock DSR engine
        class DummyDSREngine:
            def evaluate_strategy(self, *args: Any, **kwargs: Any) -> Any:
                pass

        dummy = DummyDSREngine()
        auditor_custom = BenchmarkAuditor(config=cfg, dsr_engine=dummy)
        assert auditor_custom._dsr_engine is dummy

    @pytest.mark.parametrize("bad_cfg", [None, "config", 100_000.0, {"initial_capital": 100_000.0}])
    def test_constructor_config_type_rejection(self, bad_cfg: object) -> None:
        """Verify non-SimulationConfig raises ERR-SIM-002."""
        with pytest.raises(DegenerateSimulationException) as exc_info:
            BenchmarkAuditor(config=bad_cfg)  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

    def test_constructor_invalid_dsr_engine_rejection(self) -> None:
        """Verify dsr_engine without evaluate_strategy raises ERR-SIM-002."""
        cfg = SimulationConfig()
        with pytest.raises(DegenerateSimulationException) as exc_info:
            BenchmarkAuditor(config=cfg, dsr_engine="not_an_engine")
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        class UncallableDSREngine:
            evaluate_strategy = "not_callable"

        with pytest.raises(DegenerateSimulationException) as exc_info:
            BenchmarkAuditor(config=cfg, dsr_engine=UncallableDSREngine())
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

    def test_audit_sample_starvation_rejection_err_sim_005(self) -> None:
        """Verify len(records) < 30 raises ERR-SIM-005 (INV-SIM-005)."""
        cfg = SimulationConfig()
        auditor = BenchmarkAuditor(config=cfg)

        # 29 records < 30
        records_29 = self._create_synthetic_records(n_bars=29)
        asset_returns = np.zeros((29, 3), dtype=np.float64)
        with pytest.raises(DegenerateSimulationException) as exc_info:
            auditor.audit(records=records_29, asset_returns=asset_returns)
        assert exc_info.value.code == ERR_SIM_STARVATION

        # Empty records
        with pytest.raises(DegenerateSimulationException) as exc_info:
            auditor.audit(records=[], asset_returns=np.zeros((0, 3), dtype=np.float64))
        assert exc_info.value.code == ERR_SIM_STARVATION

    def test_audit_records_container_and_element_type_rejection(self) -> None:
        """Verify records must be a list of BarExecutionRecord instances."""
        cfg = SimulationConfig()
        auditor = BenchmarkAuditor(config=cfg)
        records = self._create_synthetic_records(n_bars=35)
        asset_returns = np.zeros((35, 2), dtype=np.float64)

        # Tuple instead of list
        with pytest.raises(DegenerateSimulationException) as exc_info:
            auditor.audit(records=tuple(records), asset_returns=asset_returns)  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # Bad element inside records
        corrupt_records = list(records)
        corrupt_records[5] = "NotARecord"  # type: ignore[assignment]
        with pytest.raises(DegenerateSimulationException) as exc_info:
            auditor.audit(records=corrupt_records, asset_returns=asset_returns)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

    def test_audit_dimension_mismatch_rejection_err_sim_006(self) -> None:
        """Verify dimension mismatch between records and asset_returns raises ERR-SIM-006."""
        cfg = SimulationConfig()
        auditor = BenchmarkAuditor(config=cfg)
        records_35 = self._create_synthetic_records(n_bars=35)

        # Shape mismatch: 34 rows != 35 bars
        returns_34 = np.zeros((34, 2), dtype=np.float64)
        with pytest.raises(DegenerateSimulationException) as exc_info:
            auditor.audit(records=records_35, asset_returns=returns_34)
        assert exc_info.value.code == ERR_SIM_DIMENSION_MISMATCH

        # 1D array instead of 2D
        returns_1d = np.zeros(35, dtype=np.float64)
        with pytest.raises(DegenerateSimulationException) as exc_info:
            auditor.audit(records=records_35, asset_returns=returns_1d)
        assert exc_info.value.code == ERR_SIM_DIMENSION_MISMATCH

        # 3D array instead of 2D
        returns_3d = np.zeros((35, 2, 1), dtype=np.float64)
        with pytest.raises(DegenerateSimulationException) as exc_info:
            auditor.audit(records=records_35, asset_returns=returns_3d)
        assert exc_info.value.code == ERR_SIM_DIMENSION_MISMATCH

    def test_audit_asset_returns_type_and_non_finite_rejection(self) -> None:
        """Verify non-ndarray, boolean dtype, and NaN/Inf in asset_returns raise ERR-SIM-002."""
        cfg = SimulationConfig()
        auditor = BenchmarkAuditor(config=cfg)
        records = self._create_synthetic_records(n_bars=35)

        # List instead of ndarray
        with pytest.raises(DegenerateSimulationException) as exc_info:
            auditor.audit(records=records, asset_returns=[[0.01, 0.02]] * 35)  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # Boolean ndarray
        bool_returns = np.ones((35, 2), dtype=bool)
        with pytest.raises(DegenerateSimulationException) as exc_info:
            auditor.audit(records=records, asset_returns=bool_returns)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # NaN in asset returns
        nan_returns = np.zeros((35, 2), dtype=np.float64)
        nan_returns[10, 0] = float("nan")
        with pytest.raises(DegenerateSimulationException) as exc_info:
            auditor.audit(records=records, asset_returns=nan_returns)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # Inf in asset returns
        inf_returns = np.zeros((35, 2), dtype=np.float64)
        inf_returns[10, 0] = float("inf")
        with pytest.raises(DegenerateSimulationException) as exc_info:
            auditor.audit(records=records, asset_returns=inf_returns)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

    def test_audit_benchmark_returns_validation(self) -> None:
        """Verify custom benchmark_returns mapping validation and error handling."""
        cfg = SimulationConfig()
        auditor = BenchmarkAuditor(config=cfg)
        records = self._create_synthetic_records(n_bars=35)
        asset_returns = np.zeros((35, 2), dtype=np.float64)

        # benchmark_returns not a dict
        with pytest.raises(DegenerateSimulationException) as exc_info:
            auditor.audit(
                records=records,
                asset_returns=asset_returns,
                benchmark_returns=[np.zeros(35)],  # type: ignore[arg-type]
            )
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # Empty string benchmark name
        with pytest.raises(DegenerateSimulationException) as exc_info:
            auditor.audit(
                records=records,
                asset_returns=asset_returns,
                benchmark_returns={"": np.zeros(35)},
            )
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # Non-ndarray benchmark value
        with pytest.raises(DegenerateSimulationException) as exc_info:
            auditor.audit(
                records=records,
                asset_returns=asset_returns,
                benchmark_returns={"SPY": list(np.zeros(35))},  # type: ignore[arg-type]
            )
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # Length mismatch in benchmark series
        with pytest.raises(DegenerateSimulationException) as exc_info:
            auditor.audit(
                records=records,
                asset_returns=asset_returns,
                benchmark_returns={"SPY": np.zeros(30)},
            )
        assert exc_info.value.code == ERR_SIM_DIMENSION_MISMATCH

        # 2D benchmark series
        with pytest.raises(DegenerateSimulationException) as exc_info:
            auditor.audit(
                records=records,
                asset_returns=asset_returns,
                benchmark_returns={"SPY": np.zeros((35, 1))},
            )
        assert exc_info.value.code == ERR_SIM_DIMENSION_MISMATCH

        # Non-finite value in benchmark series
        nan_bench = np.zeros(35, dtype=np.float64)
        nan_bench[5] = float("nan")
        with pytest.raises(DegenerateSimulationException) as exc_info:
            auditor.audit(
                records=records,
                asset_returns=asset_returns,
                benchmark_returns={"SPY": nan_bench},
            )
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

    def test_audit_capital_ruin_in_records_err_sim_003(self) -> None:
        """Verify non-positive equity in records raises ERR-SIM-003."""
        cfg = SimulationConfig(initial_capital=100_000.0)
        auditor = BenchmarkAuditor(config=cfg)
        records = self._create_synthetic_records(n_bars=35, initial_capital=100_000.0)

        # Corrupt one record with 0.0 equity
        corrupt_records = list(records)
        bad_rec = dataclasses.replace(corrupt_records[20], portfolio_equity=0.0)
        corrupt_records[20] = bad_rec

        with pytest.raises(DegenerateSimulationException) as exc_info:
            auditor.audit(records=corrupt_records, asset_returns=np.zeros((35, 2)))
        assert exc_info.value.code == ERR_SIM_CAPITAL_RUIN

    def test_exact_mathematical_accuracy_hand_computed(self) -> None:
        """Verify exact mathematical accuracy of Sharpe, Sortino, Calmar, VaR/CVaR, and Tail Ratio."""
        w0 = 100_000.0
        ann = 252
        rf = 0.02
        cfg = SimulationConfig(
            initial_capital=w0,
            annualization_factor=ann,
            risk_free_rate=rf,
            confidence_level=0.95,
            num_trials=1,
        )
        auditor = BenchmarkAuditor(config=cfg)

        # Construct a 40-bar deterministic return pattern
        n_bars = 40
        # Deterministic fractional net returns:
        # 30 positive returns (+0.01) and 10 negative returns (-0.02)
        r_net = np.array([0.01] * 30 + [-0.02] * 10, dtype=np.float64)
        equities = [w0]
        for r in r_net:
            equities.append(equities[-1] * (1.0 + r))

        records: list[BarExecutionRecord] = []
        hwm = w0
        for step in range(n_bars):
            eq = equities[step + 1]
            hwm = max(hwm, eq)
            dd = (hwm - eq) / hwm
            tier = "CAUTION" if dd > 0.05 else "NORMAL"
            rec = BarExecutionRecord(
                step_index=step,
                timestamp=1000 + step,
                gross_pnl=equities[step] * r_net[step] + 5.0,
                net_pnl=equities[step] * r_net[step],
                friction_cost=5.0,
                portfolio_equity=eq,
                cash_balance=eq * 0.1,
                effective_leverage=0.90,
                drawdown=dd,
                circuit_breaker_tier=tier,
                circuit_breaker_haircut=1.0,
                target_allocations=(eq * 0.9,),
                discretized_allocations=(eq * 0.9,),
            )
            records.append(rec)

        asset_returns = np.column_stack([r_net, r_net * 0.5])
        report = auditor.audit(records=records, asset_returns=asset_returns)

        # 1. Total Return
        expected_w_t = equities[-1]
        expected_total_return = (expected_w_t - w0) / w0
        assert np.isclose(report.total_return, expected_total_return, atol=1e-10)

        # 2. CAGR
        expected_cagr = (expected_w_t / w0) ** (ann / n_bars) - 1.0
        assert np.isclose(report.cagr, expected_cagr, atol=1e-10)

        # 3. Annualized Volatility
        expected_std = float(np.std(r_net, ddof=1))
        expected_ann_vol = float(np.sqrt(ann) * expected_std)
        assert np.isclose(report.annualized_volatility, expected_ann_vol, atol=1e-10)

        # 4. Sharpe Ratio
        rf_bar = rf / ann
        expected_mean = float(np.mean(r_net))
        expected_sharpe = float(np.sqrt(ann) * (expected_mean - rf_bar) / expected_std)
        assert np.isclose(report.sharpe_ratio, expected_sharpe, atol=1e-10)

        # 5. Sortino Ratio
        downside_diff = np.minimum(0.0, r_net - rf_bar)
        downside_dev = float(np.sqrt(np.mean(downside_diff**2)))
        expected_sortino = float(np.sqrt(ann) * (expected_mean - rf_bar) / downside_dev)
        assert np.isclose(report.sortino_ratio, expected_sortino, atol=1e-10)

        # 6. Max Drawdown and Calmar Ratio
        expected_mdd = float(max(rec.drawdown for rec in records))
        expected_calmar = expected_cagr / expected_mdd
        assert np.isclose(report.max_drawdown, expected_mdd, atol=1e-10)
        assert np.isclose(report.calmar_ratio, expected_calmar, atol=1e-10)

        # 7. Realized VaR & CVaR (in loss space, positive)
        expected_q05 = float(np.quantile(r_net, 0.05))
        expected_var_95 = -expected_q05
        tail_95 = r_net[r_net <= expected_q05]
        expected_cvar_95 = float(-np.mean(tail_95))
        assert np.isclose(report.realized_var_95, expected_var_95, atol=1e-10)
        assert np.isclose(report.realized_cvar_95, expected_cvar_95, atol=1e-10)
        assert report.realized_cvar_95 >= report.realized_var_95

        expected_q01 = float(np.quantile(r_net, 0.01))
        expected_var_99 = -expected_q01
        tail_99 = r_net[r_net <= expected_q01]
        expected_cvar_99 = float(-np.mean(tail_99))
        assert np.isclose(report.realized_var_99, expected_var_99, atol=1e-10)
        assert np.isclose(report.realized_cvar_99, expected_cvar_99, atol=1e-10)
        assert report.realized_cvar_99 >= report.realized_var_99

        # 8. Tail Ratio
        expected_q95 = float(np.quantile(r_net, 0.95))
        expected_tail_ratio = expected_q95 / abs(expected_q05)
        assert np.isclose(report.tail_ratio, expected_tail_ratio, atol=1e-10)

        # 9. Friction, Leverage, and Circuit Breakers
        assert report.total_friction_cost == 5.0 * n_bars
        assert report.peak_leverage == 0.90
        assert sum(report.circuit_breaker_counts.values()) == n_bars

    def test_all_default_benchmarks_and_custom_benchmarks(self) -> None:
        """Verify EqualWeight, RiskParity, InverseVolatility, Cash, and custom benchmarks."""
        cfg = SimulationConfig(
            initial_capital=100_000.0, annualization_factor=252, risk_free_rate=0.02
        )
        auditor = BenchmarkAuditor(config=cfg)

        t_bars = 50
        n_assets = 3
        rng = np.random.default_rng(42)
        asset_returns = rng.normal(loc=0.001, scale=0.015, size=(t_bars, n_assets))
        records = self._create_synthetic_records(n_bars=t_bars, returns=asset_returns[:, 0])

        custom_spy = rng.normal(loc=0.0008, scale=0.012, size=t_bars)
        report = auditor.audit(
            records=records,
            asset_returns=asset_returns,
            benchmark_returns={"SPY_Benchmark": custom_spy},
        )

        comparisons = report.benchmark_comparisons
        # All 4 default benchmarks must be present
        assert "EqualWeight" in comparisons
        assert "RiskParity" in comparisons
        assert "InverseVolatility" in comparisons
        assert "Cash" in comparisons
        # Custom benchmark must be present
        assert "SPY_Benchmark" in comparisons

        # Verify Cash properties
        cash_comp = comparisons["Cash"]
        assert cash_comp.beta == 0.0
        assert np.isclose(cash_comp.annualized_volatility, 0.0, atol=1e-10)
        assert cash_comp.max_drawdown == 0.0
        assert np.isclose(cash_comp.annualized_return, 0.02, atol=1e-3)

        # Verify OLS consistency: alpha + beta * mean(b - rf) == mean(strat - rf)
        rf_bar = 0.02 / 252.0
        equities = [100_000.0] + [r.portfolio_equity for r in records]
        eq_arr = np.array(equities)
        strat_r = (eq_arr[1:] - eq_arr[:-1]) / eq_arr[:-1]
        y_mean = float(np.mean(strat_r - rf_bar))

        for b_name, comp in comparisons.items():
            assert comp.name == b_name
            assert comp.annualized_volatility >= 0.0
            assert 0.0 <= comp.max_drawdown <= 1.0
            assert comp.tracking_error >= 0.0
            if b_name != "Cash":
                b_ret = (
                    custom_spy
                    if b_name == "SPY_Benchmark"
                    else (np.mean(asset_returns, axis=1) if b_name == "EqualWeight" else None)
                )
                if b_ret is not None:
                    x_mean = float(np.mean(b_ret - rf_bar))
                    reconstructed_y_mean = (comp.alpha / 252.0) + comp.beta * x_mean
                    assert np.isclose(y_mean, reconstructed_y_mean, atol=1e-10)

    def test_dsr_and_min_btl_integration(self) -> None:
        """Verify DSR statistical certification and sub-benchmark infinite min_backtest_length."""
        cfg = SimulationConfig(
            initial_capital=100_000.0, annualization_factor=252, confidence_level=0.95
        )
        auditor = BenchmarkAuditor(config=cfg)

        # Sub-benchmark losing strategy (SR < 0.0)
        losing_returns = np.full(50, -0.005, dtype=np.float64)
        losing_records = self._create_synthetic_records(
            n_bars=50,
            initial_capital=cfg.initial_capital,
            returns=losing_returns,
        )
        losing_assets = np.column_stack([losing_returns, losing_returns])

        losing_report = auditor.audit(records=losing_records, asset_returns=losing_assets)
        assert losing_report.sharpe_ratio < 0.0
        assert losing_report.min_backtest_length == float("inf")
        assert losing_report.is_statistically_significant is False

        # Outstanding genuine strategy (SR ~ 3.0 over 252 bars)
        rng = np.random.default_rng(123)
        winning_returns = rng.normal(loc=0.002, scale=0.01, size=252)
        winning_records = self._create_synthetic_records(
            n_bars=252,
            initial_capital=cfg.initial_capital,
            returns=winning_returns,
        )
        winning_assets = np.column_stack([winning_returns, winning_returns])

        winning_report = auditor.audit(records=winning_records, asset_returns=winning_assets)
        assert winning_report.sharpe_ratio > 2.0
        assert winning_report.deflated_sharpe_ratio >= 0.95
        assert winning_report.min_backtest_length < 252.0
        assert winning_report.is_statistically_significant is True

    def test_zero_variance_flatline_and_edge_cases(self) -> None:
        """Verify edge cases: flatline equity, single asset universe, empty universe."""
        cfg = SimulationConfig(
            initial_capital=100_000.0, annualization_factor=252, risk_free_rate=0.0
        )
        auditor = BenchmarkAuditor(config=cfg)

        # 1. Zero-variance flatline (identical equity on every bar)
        flat_records = self._create_synthetic_records(
            n_bars=35,
            initial_capital=cfg.initial_capital,
            returns=np.zeros(35),
            friction_per_bar=0.0,
        )
        flat_assets = np.zeros((35, 2), dtype=np.float64)
        report_flat = auditor.audit(records=flat_records, asset_returns=flat_assets)
        assert report_flat.annualized_volatility == 0.0
        assert report_flat.sharpe_ratio == 0.0
        assert report_flat.sortino_ratio == 0.0
        assert report_flat.max_drawdown == 0.0
        assert report_flat.calmar_ratio == 0.0
        assert report_flat.tail_ratio == 0.0

        # 2. Single asset universe (N=1)
        single_asset = np.full((35, 1), 0.001, dtype=np.float64)
        single_records = self._create_synthetic_records(
            n_bars=35,
            initial_capital=cfg.initial_capital,
            returns=single_asset[:, 0],
        )
        report_single = auditor.audit(records=single_records, asset_returns=single_asset)
        assert "EqualWeight" in report_single.benchmark_comparisons
        assert "RiskParity" in report_single.benchmark_comparisons
        assert "InverseVolatility" in report_single.benchmark_comparisons

        # 3. Empty asset universe (N=0)
        empty_asset = np.empty((35, 0), dtype=np.float64)
        empty_records = self._create_synthetic_records(
            n_bars=35,
            initial_capital=cfg.initial_capital,
            returns=np.zeros(35),
        )
        report_empty = auditor.audit(records=empty_records, asset_returns=empty_asset)
        assert len(report_empty.benchmark_comparisons) == 4

    def test_inv_sim_006_benchmark_auditor_latency_sla(self) -> None:
        """Verify INV-SIM-006: Hot-path auditing 1000 bars executes in < 5.0ms."""
        cfg = SimulationConfig(initial_capital=1_000_000.0, annualization_factor=252)
        auditor = BenchmarkAuditor(config=cfg)

        n_bars = 1000
        n_assets = 10
        rng = np.random.default_rng(42)
        asset_returns = rng.normal(loc=0.0005, scale=0.01, size=(n_bars, n_assets))
        records = self._create_synthetic_records(n_bars=n_bars, returns=asset_returns[:, 0])

        # Warm-up run
        auditor.audit(records=records, asset_returns=asset_returns)

        gc_was_enabled = gc.isenabled()
        gc.collect()
        gc.disable()
        old_trace = sys.gettrace()
        num_runs = 10
        latencies_ms: list[float] = []

        try:
            sys.settrace(None)
            for _ in range(num_runs):
                t0 = time.perf_counter()
                auditor.audit(records=records, asset_returns=asset_returns)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)
        finally:
            sys.settrace(old_trace)
            if gc_was_enabled:
                gc.enable()

        is_traced = (
            old_trace is not None
            or "coverage" in sys.modules
            or "pytest_cov" in sys.modules
            or (
                hasattr(sys, "monitoring")
                and any(sys.monitoring.get_tool(i) is not None for i in range(6))
            )
        )
        threshold_ms = 15.0 if is_traced else 5.0
        min_latency = float(np.min(latencies_ms))
        assert min_latency <= threshold_ms, (
            f"INV-SIM-006 SLA breached: min auditor time was {min_latency:.3f}ms > {threshold_ms}ms"
        )


class TestReplayEngine:
    """Test suite for ReplayEngine master causal live replay simulation orchestrator."""

    class MockTrackingListener:
        """Observer listener capturing lifecycle events for verification."""

        def __init__(self) -> None:
            self.events: list[tuple[str, int, Any]] = []

        def on_bar_start(self, step: int, timestamp: int) -> None:
            self.events.append(("start", step, timestamp))

        def on_decision(self, step: int, decision: Any) -> None:
            self.events.append(("decision", step, decision))

        def on_fill(self, step: int, record: BarExecutionRecord) -> None:
            self.events.append(("fill", step, record))

        def on_bar_end(self, step: int, record: BarExecutionRecord) -> None:
            self.events.append(("end", step, record))

    @staticmethod
    def _generate_synthetic_inputs(
        t_bars: int = 50,
        n_assets: int = 3,
        k_models: int = 5,
        is_3d: bool = True,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate statistically consistent, finite synthetic test arrays."""
        rng = np.random.default_rng(seed)
        returns = rng.normal(loc=0.0005, scale=0.01, size=(t_bars, n_assets)).astype(np.float64)
        vols = np.full((t_bars, n_assets), 0.01, dtype=np.float64)
        if is_3d:
            preds = rng.normal(loc=0.0008, scale=0.005, size=(t_bars, n_assets, k_models)).astype(
                np.float64
            )
        else:
            preds = rng.normal(loc=0.0008, scale=0.005, size=(t_bars, n_assets)).astype(np.float64)
        regimes = np.full((t_bars, 3), [0.7, 0.2, 0.1], dtype=np.float64)
        betas = np.full(t_bars, 1.5, dtype=np.float64)
        return returns, vols, preds, regimes, betas

    def test_replay_engine_instantiation(self) -> None:
        """Verify ReplayEngine initializes with default components."""
        engine = ReplayEngine()
        assert isinstance(engine.config, SimulationConfig)
        assert isinstance(engine.cost_model, ExecutionCostModel)
        assert isinstance(engine.auditor, BenchmarkAuditor)
        assert isinstance(engine.sizer, UnifiedConvexExecutionSizer)
        assert isinstance(engine.cb_engine, CircuitBreakerOverlayEngine)
        assert engine.dma_engine is None
        assert engine.listeners == []

    def test_replay_engine_custom_component_injection(self) -> None:
        """Verify ReplayEngine accepts custom-injected sub-engines."""
        cfg = SimulationConfig(initial_capital=500_000.0, fee_bps=1.0)
        cost_model = ExecutionCostModel(fee_bps=1.0, spread_bps=0.5, impact_coefficient=0.05)
        auditor = BenchmarkAuditor(config=cfg)
        sizer = UnifiedConvexExecutionSizer(config=SizingConfig())
        cb_engine = CircuitBreakerOverlayEngine(config=CircuitBreakerConfig())
        dma_engine = RegimeConditionedDMAEngine(config=EnsembleConfig())

        engine = ReplayEngine(
            config=cfg,
            cost_model=cost_model,
            auditor=auditor,
            sizer=sizer,
            cb_engine=cb_engine,
            dma_engine=dma_engine,
        )

        assert engine.config is cfg
        assert engine.cost_model is cost_model
        assert engine.auditor is auditor
        assert engine.sizer is sizer
        assert engine.cb_engine is cb_engine
        assert engine.dma_engine is dma_engine

    @pytest.mark.parametrize(
        ("param_name", "bad_val"),
        [
            ("config", "not_a_config"),
            ("cost_model", 12345),
            ("auditor", [1, 2, 3]),
            ("sizer", {"a": 1}),
            ("cb_engine", object()),
            ("dma_engine", "invalid_dma"),
        ],
    )
    def test_replay_engine_invalid_component_rejection(self, param_name: str, bad_val: Any) -> None:
        """Verify invalid component types raise DegenerateSimulationException(ERR-SIM-002)."""
        kwargs: dict[str, Any] = {param_name: bad_val}
        with pytest.raises(DegenerateSimulationException) as exc_info:
            ReplayEngine(**kwargs)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

    def test_add_listener_and_listeners_property(self) -> None:
        """Verify observer listener registration, validation, and defensive isolation."""
        engine = ReplayEngine()
        listener = self.MockTrackingListener()

        # Reject invalid listener
        with pytest.raises(DegenerateSimulationException) as exc_info:
            engine.add_listener("not_a_listener")  # type: ignore[arg-type]
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # Register valid listener
        engine.add_listener(listener)
        listeners = engine.listeners
        assert len(listeners) == 1
        assert listeners[0] is listener

        # Verify defensive isolation: modifying returned list does not mutate internal registry
        listeners.clear()
        assert len(engine.listeners) == 1

    def test_observer_listener_event_sequence_tracking(self) -> None:
        """Verify all 4 observer listener hooks receive events in strict chronological order."""
        engine = ReplayEngine()
        listener = self.MockTrackingListener()
        engine.add_listener(listener)

        t_bars = 50
        returns, vols, preds, regimes, betas = self._generate_synthetic_inputs(t_bars=t_bars)
        report = engine.run(
            asset_returns=returns,
            asset_volatilities=vols,
            candidate_predictions=preds,
            regime_probabilities=regimes,
            ambiguity_betas=betas,
        )

        assert report.final_equity > 0.0
        # 4 events per bar across 50 bars = 200 events
        assert len(listener.events) == 4 * t_bars

        for t in range(t_bars):
            ev_start = listener.events[4 * t]
            ev_decision = listener.events[4 * t + 1]
            ev_fill = listener.events[4 * t + 2]
            ev_end = listener.events[4 * t + 3]

            assert ev_start[0] == "start"
            assert ev_start[1] == t
            assert ev_start[2] == t * 86_400_000_000_000

            assert ev_decision[0] == "decision"
            assert ev_decision[1] == t
            assert isinstance(ev_decision[2], SizingDecision)

            assert ev_fill[0] == "fill"
            assert ev_fill[1] == t
            assert isinstance(ev_fill[2], BarExecutionRecord)

            assert ev_end[0] == "end"
            assert ev_end[1] == t
            assert isinstance(ev_end[2], BarExecutionRecord)
            assert ev_fill[2] is ev_end[2]

    def test_multi_asset_live_replay_50_bars_n3(self) -> None:
        """Verify 50-bar, 3-asset live replay with 3D model predictions."""
        engine = ReplayEngine()
        t_bars, n_assets = 50, 3
        returns, vols, preds, regimes, betas = self._generate_synthetic_inputs(
            t_bars=t_bars, n_assets=n_assets, k_models=5, is_3d=True
        )
        report = engine.run(
            asset_returns=returns,
            asset_volatilities=vols,
            candidate_predictions=preds,
            regime_probabilities=regimes,
            ambiguity_betas=betas,
        )

        assert report.initial_capital == 1_000_000.0
        assert report.final_equity > 0.0
        assert math.isfinite(report.total_return)
        assert math.isfinite(report.cagr)
        assert math.isfinite(report.annualized_volatility)
        assert math.isfinite(report.sharpe_ratio)
        assert 0.0 <= report.max_drawdown <= 1.0
        assert 0.0 <= report.deflated_sharpe_ratio <= 1.0
        assert report.total_friction_cost >= 0.0
        assert len(report.benchmark_comparisons) == 4

    def test_multi_asset_live_replay_100_bars_n5(self) -> None:
        """Verify 100-bar, 5-asset live replay with 2D candidate predictions."""
        engine = ReplayEngine()
        t_bars, n_assets = 100, 5
        returns, vols, preds, regimes, betas = self._generate_synthetic_inputs(
            t_bars=t_bars, n_assets=n_assets, is_3d=False
        )
        report = engine.run(
            asset_returns=returns,
            asset_volatilities=vols,
            candidate_predictions=preds,
            regime_probabilities=regimes,
            ambiguity_betas=betas,
        )

        assert report.final_equity > 0.0
        assert math.isfinite(report.sharpe_ratio)
        assert 0.0 <= report.max_drawdown <= 1.0
        assert len(report.benchmark_comparisons) == 4

    def test_multi_asset_live_replay_100_bars_n10_with_all_options(self) -> None:
        """Verify 100-bar, 10-asset simulation with timestamps, 1D and 2D ADVs, and custom benchmark."""
        engine = ReplayEngine()
        t_bars, n_assets = 100, 10
        returns, vols, preds, regimes, betas = self._generate_synthetic_inputs(
            t_bars=t_bars, n_assets=n_assets, k_models=4, is_3d=True
        )
        timestamps = np.arange(1000, 1000 + t_bars, dtype=np.int64) * 1_000_000_000
        advs_1d = np.full(n_assets, 5_000_000.0, dtype=np.float64)
        custom_bm = np.full(t_bars, 0.0003, dtype=np.float64)

        # 1. Run with 1D ADVs
        report_1d = engine.run(
            asset_returns=returns,
            asset_volatilities=vols,
            candidate_predictions=preds,
            regime_probabilities=regimes,
            ambiguity_betas=betas,
            timestamps=timestamps,
            advs=advs_1d,
            benchmark_returns={"CustomIndex": custom_bm},
        )
        assert "CustomIndex" in report_1d.benchmark_comparisons
        assert report_1d.final_equity > 0.0

        # 2. Run with 2D ADVs
        advs_2d = np.full((t_bars, n_assets), 5_000_000.0, dtype=np.float64)
        report_2d = engine.run(
            asset_returns=returns,
            asset_volatilities=vols,
            candidate_predictions=preds,
            regime_probabilities=regimes,
            ambiguity_betas=betas,
            timestamps=timestamps,
            advs=advs_2d,
            benchmark_returns={"CustomIndex": custom_bm},
        )
        assert np.isclose(report_1d.total_return, report_2d.total_return, atol=1e-8)

    def test_inv_sim_001_zero_lookahead_perturbation(self) -> None:
        """Verify INV-SIM-001: Perturbing return at t+5 does not alter allocations at t < t+5."""
        engine = ReplayEngine()
        t_bars, n_assets = 50, 3
        returns_base, vols, preds, regimes, betas = self._generate_synthetic_inputs(
            t_bars=t_bars, n_assets=n_assets
        )

        listener_base = self.MockTrackingListener()
        engine.add_listener(listener_base)
        engine.run(
            asset_returns=returns_base,
            asset_volatilities=vols,
            candidate_predictions=preds,
            regime_probabilities=regimes,
            ambiguity_betas=betas,
        )

        # Extract executed allocations at each bar t for baseline
        allocations_base = [
            ev[2].discretized_allocations for ev in listener_base.events if ev[0] == "fill"
        ]

        # Perturb future return at step 25 (e.g. huge shock +0.10)
        returns_perturbed = returns_base.copy()
        perturb_step = 25
        returns_perturbed[perturb_step, :] += 0.10

        engine_perturbed = ReplayEngine()
        listener_perturbed = self.MockTrackingListener()
        engine_perturbed.add_listener(listener_perturbed)
        engine_perturbed.run(
            asset_returns=returns_perturbed,
            asset_volatilities=vols,
            candidate_predictions=preds,
            regime_probabilities=regimes,
            ambiguity_betas=betas,
        )

        allocations_perturbed = [
            ev[2].discretized_allocations for ev in listener_perturbed.events if ev[0] == "fill"
        ]

        # Assert zero-lookahead: for all steps t < 25, allocations are bit-exact identical!
        for t in range(perturb_step):
            assert allocations_base[t] == allocations_perturbed[t], (
                f"INV-SIM-001 Lookahead violation at bar {t} < {perturb_step}: "
                f"base={allocations_base[t]} vs perturbed={allocations_perturbed[t]}"
            )

    def test_inv_sim_002_capital_conservation_across_all_bars(self) -> None:
        """Verify INV-SIM-002: W_t == cash_t + sum nu_i and W_t - W_{t-1} == PnL_t^{net} everywhere."""
        engine = ReplayEngine()
        listener = self.MockTrackingListener()
        engine.add_listener(listener)

        t_bars, n_assets = 100, 5
        returns, vols, preds, regimes, betas = self._generate_synthetic_inputs(
            t_bars=t_bars, n_assets=n_assets
        )
        report = engine.run(
            asset_returns=returns,
            asset_volatilities=vols,
            candidate_predictions=preds,
            regime_probabilities=regimes,
            ambiguity_betas=betas,
        )

        records = [ev[2] for ev in listener.events if ev[0] == "fill"]
        assert len(records) == t_bars

        prev_equity = engine.config.initial_capital
        for rec in records:
            # 1. Capital balance identity: W_t == cash_t + sum(nu_{i, t})
            pos_sum = sum(rec.discretized_allocations)
            discrepancy = abs(rec.portfolio_equity - (rec.cash_balance + pos_sum))
            assert discrepancy < 1e-5, (
                f"INV-SIM-002 Capital discrepancy at bar {rec.step_index}: {discrepancy} >= 1e-5"
            )

            # 2. Net wealth increment identity: W_t - W_{t-1} == PnL_t^{net}
            pnl_discrepancy = abs((rec.portfolio_equity - prev_equity) - rec.net_pnl)
            assert pnl_discrepancy < 1e-5, (
                f"INV-SIM-002 PnL discrepancy at bar {rec.step_index}: {pnl_discrepancy} >= 1e-5"
            )
            prev_equity = rec.portfolio_equity

        assert np.isclose(report.final_equity, records[-1].portfolio_equity, atol=1e-8)

    def test_inv_sim_003_non_negative_friction_cost(self) -> None:
        """Verify INV-SIM-003: Friction cost is non-negative on every bar and in aggregate."""
        engine = ReplayEngine()
        listener = self.MockTrackingListener()
        engine.add_listener(listener)

        returns, vols, preds, regimes, betas = self._generate_synthetic_inputs(
            t_bars=50, n_assets=3
        )
        report = engine.run(
            asset_returns=returns,
            asset_volatilities=vols,
            candidate_predictions=preds,
            regime_probabilities=regimes,
            ambiguity_betas=betas,
        )

        records = [ev[2] for ev in listener.events if ev[0] == "fill"]
        for rec in records:
            assert rec.friction_cost >= 0.0
            assert math.isfinite(rec.friction_cost)

        assert report.total_friction_cost >= 0.0
        assert np.isclose(
            report.total_friction_cost, sum(r.friction_cost for r in records), atol=1e-8
        )

    def test_inv_sim_004_circuit_breaker_halt_collapses_positions(self) -> None:
        """Verify INV-SIM-004: Circuit breaker HALT forces allocations strictly to 0.0."""
        engine = ReplayEngine()
        listener = self.MockTrackingListener()
        engine.add_listener(listener)

        t_bars, n_assets = 50, 3
        returns, vols, preds, regimes, betas = self._generate_synthetic_inputs(
            t_bars=t_bars, n_assets=n_assets
        )

        # Force extreme crisis / panic regime from step 20 onward
        regimes[20:, :] = [0.01, 0.01, 0.98]
        # Inject extreme disagreement across model forecasts and max thermodynamic ambiguity
        preds[20:, :, 0] = 0.10
        preds[20:, :, 1] = -0.10
        preds[20:, :, 2] = 0.0
        preds[20:, :, 3] = 0.10
        preds[20:, :, 4] = -0.10
        betas[20:] = 10.0
        vols[20:, :] = 1e-4

        report = engine.run(
            asset_returns=returns,
            asset_volatilities=vols,
            candidate_predictions=preds,
            regime_probabilities=regimes,
            ambiguity_betas=betas,
        )

        records = [ev[2] for ev in listener.events if ev[0] == "fill"]
        halt_found = False

        for rec in records:
            if rec.circuit_breaker_tier == "HALT" or rec.circuit_breaker_haircut == 0.0:
                halt_found = True
                assert all(x == 0.0 for x in rec.target_allocations), (
                    f"INV-SIM-004 violation: target_allocations={rec.target_allocations} in HALT"
                )
                assert all(x == 0.0 for x in rec.discretized_allocations), (
                    f"INV-SIM-004 violation: discretized_allocations={rec.discretized_allocations} in HALT"
                )
                assert rec.effective_leverage == 0.0

        assert halt_found is True
        assert report.circuit_breaker_counts.get("HALT", 0) > 0

    def test_inv_sim_005_sample_starvation_rejection(self) -> None:
        """Verify INV-SIM-005: T < 30 bars raises DegenerateSimulationException(ERR-SIM-005)."""
        engine = ReplayEngine()
        returns, vols, preds, regimes, betas = self._generate_synthetic_inputs(t_bars=29)

        with pytest.raises(DegenerateSimulationException) as exc_info:
            engine.run(
                asset_returns=returns,
                asset_volatilities=vols,
                candidate_predictions=preds,
                regime_probabilities=regimes,
                ambiguity_betas=betas,
            )
        assert exc_info.value.code == ERR_SIM_STARVATION

    @pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
    def test_inv_sim_005_non_finite_input_rejections(self, bad_val: float) -> None:
        """Verify INV-SIM-005: NaN and Inf in any input array raise ERR-SIM-002."""
        engine = ReplayEngine()
        r, v, p, reg, b = self._generate_synthetic_inputs(t_bars=35)

        # 1. Non-finite in asset_returns
        r_bad = r.copy()
        r_bad[10, 0] = bad_val
        with pytest.raises(DegenerateSimulationException) as exc_info:
            engine.run(r_bad, v, p, reg, b)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # 2. Non-finite in asset_volatilities
        v_bad = v.copy()
        v_bad[5, 1] = bad_val
        with pytest.raises(DegenerateSimulationException) as exc_info:
            engine.run(r, v_bad, p, reg, b)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # 3. Non-finite in candidate_predictions
        p_bad = p.copy()
        p_bad[15, 0, 0] = bad_val
        with pytest.raises(DegenerateSimulationException) as exc_info:
            engine.run(r, v, p_bad, reg, b)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # 4. Non-finite in regime_probabilities
        reg_bad = reg.copy()
        reg_bad[20, 0] = bad_val
        with pytest.raises(DegenerateSimulationException) as exc_info:
            engine.run(r, v, p, reg_bad, b)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # 5. Non-finite in ambiguity_betas
        b_bad = b.copy()
        b_bad[8] = bad_val
        with pytest.raises(DegenerateSimulationException) as exc_info:
            engine.run(r, v, p, reg, b_bad)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # 6. Non-finite in timestamps
        ts_bad = np.arange(35, dtype=np.float64)
        ts_bad[2] = bad_val
        with pytest.raises(DegenerateSimulationException) as exc_info:
            engine.run(r, v, p, reg, b, timestamps=ts_bad)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # 7. Non-finite in advs
        adv_bad = np.full(3, 1_000_000.0, dtype=np.float64)
        adv_bad[1] = bad_val
        with pytest.raises(DegenerateSimulationException) as exc_info:
            engine.run(r, v, p, reg, b, advs=adv_bad)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

    def test_dimension_mismatch_rejections(self) -> None:
        """Verify dimension mismatches raise DegenerateSimulationException(ERR-SIM-006)."""
        engine = ReplayEngine()
        r, v, p, reg, b = self._generate_synthetic_inputs(t_bars=40, n_assets=3)

        # 1. 1D or 3D asset_returns
        with pytest.raises(DegenerateSimulationException) as exc_info:
            engine.run(r[:, 0], v, p, reg, b)
        assert exc_info.value.code == ERR_SIM_DIMENSION_MISMATCH

        # 2. Volatilities length mismatch
        with pytest.raises(DegenerateSimulationException) as exc_info:
            engine.run(r, v[:35], p, reg, b)
        assert exc_info.value.code == ERR_SIM_DIMENSION_MISMATCH

        # 3. Volatilities asset count mismatch
        with pytest.raises(DegenerateSimulationException) as exc_info:
            engine.run(r, v[:, :2], p, reg, b)
        assert exc_info.value.code == ERR_SIM_DIMENSION_MISMATCH

        # 4. Predictions length mismatch
        with pytest.raises(DegenerateSimulationException) as exc_info:
            engine.run(r, v, p[:30], reg, b)
        assert exc_info.value.code == ERR_SIM_DIMENSION_MISMATCH

        # 5. Predictions asset count mismatch
        with pytest.raises(DegenerateSimulationException) as exc_info:
            engine.run(r, v, p[:, :1], reg, b)
        assert exc_info.value.code == ERR_SIM_DIMENSION_MISMATCH

        # 6. Predictions 4D array
        p_4d = np.zeros((40, 3, 2, 2), dtype=np.float64)
        with pytest.raises(DegenerateSimulationException) as exc_info:
            engine.run(r, v, p_4d, reg, b)
        assert exc_info.value.code == ERR_SIM_DIMENSION_MISMATCH

        # 7. Regime probabilities length mismatch
        with pytest.raises(DegenerateSimulationException) as exc_info:
            engine.run(r, v, p, reg[:20], b)
        assert exc_info.value.code == ERR_SIM_DIMENSION_MISMATCH

        # 8. Ambiguity betas length mismatch
        with pytest.raises(DegenerateSimulationException) as exc_info:
            engine.run(r, v, p, reg, b[:15])
        assert exc_info.value.code == ERR_SIM_DIMENSION_MISMATCH

        # 9. Timestamps length mismatch
        with pytest.raises(DegenerateSimulationException) as exc_info:
            engine.run(r, v, p, reg, b, timestamps=np.arange(10))
        assert exc_info.value.code == ERR_SIM_DIMENSION_MISMATCH

        # 10. ADVs dimension mismatch
        with pytest.raises(DegenerateSimulationException) as exc_info:
            engine.run(r, v, p, reg, b, advs=np.full(5, 1e6))
        assert exc_info.value.code == ERR_SIM_DIMENSION_MISMATCH

    def test_domain_boundary_rejections(self) -> None:
        """Verify negative volatility, non-positive ADV/beta, and negative timestamps raise ERR-SIM-002."""
        engine = ReplayEngine()
        r, v, p, reg, b = self._generate_synthetic_inputs(t_bars=35, n_assets=3)

        # 1. Negative volatility
        v_neg = v.copy()
        v_neg[0, 0] = -0.01
        with pytest.raises(DegenerateSimulationException) as exc_info:
            engine.run(r, v_neg, p, reg, b)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # 2. Non-positive ADV
        with pytest.raises(DegenerateSimulationException) as exc_info:
            engine.run(r, v, p, reg, b, advs=np.array([1e6, 0.0, 1e6]))
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # 3. Non-positive ambiguity beta
        b_neg = b.copy()
        b_neg[5] = 0.0
        with pytest.raises(DegenerateSimulationException) as exc_info:
            engine.run(r, v, p, reg, b_neg)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # 4. Negative timestamp
        with pytest.raises(DegenerateSimulationException) as exc_info:
            engine.run(r, v, p, reg, b, timestamps=np.array([-1] + list(range(1, 35))))
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

        # 5. Negative regime probability
        reg_neg = reg.copy()
        reg_neg[0, 0] = -0.1
        with pytest.raises(DegenerateSimulationException) as exc_info:
            engine.run(r, v, p, reg_neg, b)
        assert exc_info.value.code == ERR_SIM_NON_FINITE_INPUT

    def test_single_asset_universe_n1(self) -> None:
        """Verify single asset universe (N=1) executes successfully."""
        engine = ReplayEngine()
        t_bars, n_assets = 35, 1
        returns, vols, preds, regimes, betas = self._generate_synthetic_inputs(
            t_bars=t_bars, n_assets=n_assets, is_3d=False
        )
        report = engine.run(returns, vols, preds, regimes, betas)
        assert report.final_equity > 0.0
        assert math.isfinite(report.sharpe_ratio)
        assert "EqualWeight" in report.benchmark_comparisons

    def test_empty_asset_universe_n0(self) -> None:
        """Verify empty asset universe (N=0) executes without crash."""
        engine = ReplayEngine()
        t_bars, n_assets = 35, 0
        returns, vols, preds, regimes, betas = self._generate_synthetic_inputs(
            t_bars=t_bars, n_assets=n_assets, is_3d=False
        )
        report = engine.run(returns, vols, preds, regimes, betas)
        assert np.isclose(report.final_equity, engine.config.initial_capital, atol=1e-8)
        assert report.total_return == 0.0
        assert report.total_friction_cost == 0.0

    def test_capital_ruin_tripping(self) -> None:
        """Verify catastrophic capital ruin raises DegenerateSimulationException(ERR-SIM-003)."""
        cfg = SimulationConfig(initial_capital=100.0)
        engine = ReplayEngine(config=cfg)
        t_bars, n_assets = 40, 2
        returns, vols, preds, regimes, betas = self._generate_synthetic_inputs(
            t_bars=t_bars, n_assets=n_assets
        )

        # Huge negative return innovation that will wipe out capital
        returns[1, :] = -50.0
        preds[:] = 0.05

        with pytest.raises(DegenerateSimulationException) as exc_info:
            engine.run(returns, vols, preds, regimes, betas)
        assert exc_info.value.code == ERR_SIM_CAPITAL_RUIN

    def test_circuit_breaker_exogenous_cusum_shock_coupling(self) -> None:
        """Verify exogenous CUSUM jump + panic regime triggers instantaneous single-bar HALT."""
        engine = ReplayEngine()
        listener = self.MockTrackingListener()
        engine.add_listener(listener)

        t_bars, n_assets = 40, 2
        returns, vols, preds, regimes, betas = self._generate_synthetic_inputs(
            t_bars=t_bars, n_assets=n_assets
        )
        # Bar 15 has panic regime and exogenous CUSUM jump shock
        regimes[15, :] = [0.0, 0.0, 1.0]
        cusum_shocks = np.zeros(t_bars, dtype=bool)
        cusum_shocks[15] = True

        report = engine.run(returns, vols, preds, regimes, betas, cusum_shocks=cusum_shocks)
        records = [ev[2] for ev in listener.events if ev[0] == "fill"]
        assert records[15].circuit_breaker_tier == "HALT"
        assert records[15].circuit_breaker_haircut == 0.0
        assert all(x == 0.0 for x in records[15].target_allocations)
        assert all(x == 0.0 for x in records[15].discretized_allocations)
        assert report.circuit_breaker_counts.get("HALT", 0) > 0

    def test_invalid_cusum_shocks_rejections(self) -> None:
        """Verify invalid cusum_shocks array triggers defensive validation errors."""
        engine = ReplayEngine()
        t_bars, n_assets = 40, 2
        r, v, p, reg, b = self._generate_synthetic_inputs(t_bars=t_bars, n_assets=n_assets)

        with pytest.raises(DegenerateSimulationException) as exc1:
            engine.run(r, v, p, reg, b, cusum_shocks="not_an_array")  # type: ignore[arg-type]
        assert exc1.value.code == ERR_SIM_NON_FINITE_INPUT

        with pytest.raises(DegenerateSimulationException) as exc2:
            engine.run(r, v, p, reg, b, cusum_shocks=np.array([float("nan")] * t_bars))
        assert exc2.value.code == ERR_SIM_NON_FINITE_INPUT

        with pytest.raises(DegenerateSimulationException) as exc3:
            engine.run(r, v, p, reg, b, cusum_shocks=np.zeros(20, dtype=bool))
        assert exc3.value.code == ERR_SIM_DIMENSION_MISMATCH

    def test_inv_sim_006_latency_sla_benchmark(self) -> None:
        """Verify INV-SIM-006: Hot-path replay of 100 bars x 10 assets completes within SLA."""
        engine = ReplayEngine()
        t_bars, n_assets = 100, 10
        returns, vols, preds, regimes, betas = self._generate_synthetic_inputs(
            t_bars=t_bars, n_assets=n_assets, k_models=5, is_3d=True
        )

        # Warm-up JIT, Python bytecode, caches
        engine.run(returns, vols, preds, regimes, betas)

        gc_was_enabled = gc.isenabled()
        gc.collect()
        gc.disable()
        old_trace = sys.gettrace()
        num_runs = 10
        latencies_ms: list[float] = []

        try:
            sys.settrace(None)
            for _ in range(num_runs):
                t0 = time.perf_counter()
                engine.run(returns, vols, preds, regimes, betas)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)
        finally:
            sys.settrace(old_trace)
            if gc_was_enabled:
                gc.enable()

        is_traced = (
            old_trace is not None
            or "coverage" in sys.modules
            or "pytest_cov" in sys.modules
            or (
                hasattr(sys, "monitoring")
                and any(sys.monitoring.get_tool(i) is not None for i in range(6))
            )
        )
        # Latency threshold: <= 60.0ms under tracing/coverage, <= 35.0ms on untraced hardware
        threshold_ms = 60.0 if is_traced else 35.0
        min_latency = float(np.min(latencies_ms))
        assert min_latency <= threshold_ms, (
            f"INV-SIM-006 SLA breached: min replay time was {min_latency:.3f}ms > {threshold_ms}ms"
        )


class TestSimulationMasterExports:
    """Test suite verifying public exports of simulation symbols in quant.analytics."""

    def test_analytics_simulation_symbols_exported(self) -> None:
        """Verify all simulation domain entities, engines, protocols, and fault codes are exported."""
        import quant.analytics as qa

        expected_symbols = [
            "BarExecutionRecord",
            "BenchmarkAuditReport",
            "BenchmarkAuditor",
            "BenchmarkComparison",
            "DegenerateSimulationException",
            "ERR_SIM_CAPITAL_RUIN",
            "ERR_SIM_DIMENSION_MISMATCH",
            "ERR_SIM_LOOKAHEAD_VIOLATION",
            "ERR_SIM_NEGATIVE_FRICTION",
            "ERR_SIM_NON_FINITE_INPUT",
            "ERR_SIM_STARVATION",
            "ExecutionCostModel",
            "InfeasibleSimulationException",
            "LookaheadViolationException",
            "PortfolioLedger",
            "ReplayEngine",
            "SimulationConfig",
            "SimulationError",
            "SimulationListener",
        ]

        for sym in expected_symbols:
            assert hasattr(qa, sym), f"quant.analytics must export {sym}"
            assert sym in qa.__all__, f"{sym} must be listed in quant.analytics.__all__"
            obj = getattr(qa, sym)
            assert obj is not None

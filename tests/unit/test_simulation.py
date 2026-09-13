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

import numpy as np
import pytest

from quant.analytics.simulation import (
    ERR_SIM_CAPITAL_RUIN,
    ERR_SIM_DIMENSION_MISMATCH,
    ERR_SIM_LOOKAHEAD_VIOLATION,
    ERR_SIM_NEGATIVE_FRICTION,
    ERR_SIM_NON_FINITE_INPUT,
    ERR_SIM_STARVATION,
    BarExecutionRecord,
    BenchmarkAuditReport,
    BenchmarkComparison,
    DegenerateSimulationException,
    ExecutionCostModel,
    InfeasibleSimulationException,
    LookaheadViolationException,
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

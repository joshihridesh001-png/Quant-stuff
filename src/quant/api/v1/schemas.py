"""Pydantic request and response schemas (DTOs) for API v1."""

from datetime import datetime
from typing import Any, Self
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


# Event Schemas
class EventIngestRequest(BaseModel):
    headline: str = Field(..., min_length=3, max_length=500)
    raw_text: str = Field(..., min_length=10)
    timestamp: datetime
    ticker_weights: dict[str, float] = Field(
        ..., description="Mapping of ticker to centrality weight c_{i,k}"
    )
    dense_embedding: list[float] | None = Field(
        default=None, description="Optional dense Transformer vector"
    )
    sentiment_polarity: float = Field(0.0, ge=-1.0, le=1.0)
    sentiment_subjectivity: float = Field(0.0, ge=0.0, le=1.0)
    sentiment_novelty: float = Field(0.0, ge=0.0, le=1.0)
    urgency: float = Field(0.5, ge=0.0, le=1.0)
    source: str = Field("GENERIC", max_length=50)


class EventResponse(BaseModel):
    id: UUID
    headline: str
    timestamp: datetime
    sentiment_polarity: float
    sentiment_subjectivity: float
    sentiment_novelty: float
    urgency: float
    source: str
    created_at: datetime


class BatchEventIngestRequest(BaseModel):
    events: list[EventIngestRequest] = Field(
        ..., min_length=1, max_length=500, description="Batch of news events to ingest atomically"
    )


class BatchEventIngestResponse(BaseModel):
    ingested_count: int
    events: list[EventResponse]


class ActiveStateResponse(BaseModel):
    ticker: str
    as_of_time: datetime
    state_vector: list[float]
    event_count: int
    use_projected_subspace: bool = False


# Genotype Schemas
class GenotypeCreateRequest(BaseModel):
    generation: int = Field(0, ge=0)
    cohort: str = Field("ASPIRANT", pattern="^(ALPHA|ASPIRANT)$")
    chromosome_repr: dict[str, Any]
    chromosome_game: dict[str, Any]
    chromosome_infer: dict[str, Any]
    chromosome_risk: dict[str, Any]


class GenotypeEvaluateRequest(BaseModel):
    deflated_sharpe: float = Field(..., description="Deflated Sharpe Ratio (DSR)")
    max_drawdown: float = Field(..., ge=0.0, description="Historical Max Drawdown")
    regret_score: float = Field(0.0, description="Minimax Regret metric")
    novelty_score: float = Field(0.0, description="Phenotypic novelty distance")


class GenotypeResponse(BaseModel):
    id: UUID
    generation: int
    cohort: str
    chromosome_repr: dict[str, Any]
    chromosome_game: dict[str, Any]
    chromosome_infer: dict[str, Any]
    chromosome_risk: dict[str, Any]
    fitness_score: float | None = None
    deflated_sharpe: float | None = None
    max_drawdown: float | None = None
    regret_score: float | None = None
    novelty_score: float | None = None
    created_at: datetime


class ParetoRankedGenotypeResponse(BaseModel):
    genotype: GenotypeResponse
    pareto_rank: int
    crowding_distance: float


class PopulationSeedRequest(BaseModel):
    population_size: int = Field(20, ge=4, le=500)


# Auth Schemas
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# Market Data Schemas
class PriceBarDTO(BaseModel):
    """Data transfer object for a single discrete price bar."""

    asset_id: str = Field(..., min_length=1, max_length=30, description="Asset ticker symbol")
    timestamp: int = Field(..., gt=0, description="Unix epoch nanoseconds timestamp")
    open: float = Field(..., gt=0.0, description="Opening price")
    high: float = Field(..., gt=0.0, description="Highest price in interval")
    low: float = Field(..., gt=0.0, description="Lowest price in interval")
    close: float = Field(..., gt=0.0, description="Closing price")
    volume: float = Field(..., ge=0.0, description="Traded volume")
    vwap: float = Field(..., ge=0.0, description="Volume-Weighted Average Price")
    resolution: str = Field("1m", description="Bar sampling resolution (e.g. 1m, 5m, 1h, 1d)")


class BatchPriceBarIngestRequest(BaseModel):
    """Batch ingestion request for high-throughput price bars."""

    resolution: str = Field("1m", description="Resolution applied to incoming batch")
    bars: list[PriceBarDTO] = Field(
        ..., min_length=1, max_length=10000, description="Chronological list of price bars"
    )


class BatchPriceBarIngestResponse(BaseModel):
    """Batch ingestion response reporting processed and persisted bar counts."""

    processed_count: int
    persisted_count: int
    resolution: str


class MarketDataBatchSummaryResponse(BaseModel):
    """Summary response for historical bar range queries."""

    asset_id: str
    resolution: str
    count: int
    start_time: int | None
    end_time: int | None
    latest_close: float | None
    realized_volatility_latest: float | None


# ============================================================================
# Live Execution & SOR Schemas
# ============================================================================


class ParentOrderCreateRequest(BaseModel):
    """Request DTO to initiate an algorithmic parent order execution."""

    symbol: str = Field(..., min_length=1, max_length=30, description="Asset ticker symbol")
    side: str = Field(..., pattern="^(BUY|SELL)$", description="Order execution side (BUY or SELL)")
    order_type: str = Field(
        "LIMIT", pattern="^(LIMIT|MARKET)$", description="Order price matching type"
    )
    quantity: float = Field(..., gt=0.0, description="Total target share/contract quantity")
    price: float | None = Field(
        None, gt=0.0, description="Limit price ceiling/floor (required for LIMIT orders)"
    )
    price_limit: float | None = Field(None, gt=0.0, description="Limit price ceiling/floor alias")
    algorithm: str = Field(
        "POISSON_TWAP",
        pattern="^(POISSON_TWAP|VOLUME_ADAPTIVE_VWAP|ARRIVAL_PRICE|DIRECT_MARKET)$",
        description="Algorithmic execution strategy",
    )
    horizon_seconds: float = Field(
        60.0, gt=0.0, le=86400.0, description="Execution horizon duration in seconds"
    )
    num_slices: int = Field(5, ge=1, le=100, description="Number of child order slices")
    algo_params: dict[str, Any] = Field(
        default_factory=dict, description="Algorithm-specific tuning parameters"
    )

    @model_validator(mode="after")
    def _reconcile_price(self) -> Self:
        if self.price is None and self.price_limit is not None:
            self.price = self.price_limit
        elif self.price_limit is None and self.price is not None:
            self.price_limit = self.price
        return self


class ChildOrderDTO(BaseModel):
    """Data transfer object for child order slices."""

    child_id: str
    quantity: float
    price: float
    fee: float
    timestamp_ns: int
    spread_slippage: float = 0.0


class ParentOrderResponse(BaseModel):
    """Response DTO representing parent order lifecycle and fill progression."""

    order_id: str
    symbol: str
    side: str
    total_quantity: float
    filled_quantity: float
    leaves_quantity: float
    arrival_price: float
    decision_price: float
    vwap_execution_price: float
    total_fees_paid: float
    max_duration_seconds: float
    start_time_ns: int
    is_closed: bool
    child_fills: list[ChildOrderDTO] = Field(default_factory=list)


class ImplementationShortfallResponse(BaseModel):
    """Response DTO for Perold (1988) implementation shortfall transaction cost analysis."""

    order_id: str
    symbol: str
    side: str
    total_quantity: float
    filled_quantity: float
    decision_price: float
    arrival_price: float
    execution_vwap: float
    terminal_price: float
    delay_cost: float
    price_impact: float
    spread_slippage: float
    fees_paid: float
    opportunity_cost: float
    total_shortfall: float
    total_shortfall_bps: float
    is_additive_conserved: bool


# ============================================================================
# Real-Time Risk Monitor & Kill Switch Schemas
# ============================================================================


class RiskStatusResponse(BaseModel):
    """Response DTO reporting firm-wide real-time risk metrics and firewall state."""

    nav: float
    peak_nav: float
    cash: float
    margin_used: float
    free_margin: float
    gross_notional: float
    net_notional: float
    gross_leverage: float
    net_leverage: float
    intraday_drawdown_pct: float
    is_kill_switch_active: bool
    kill_switch_trigger: str | None = None
    open_leaves_count: int


class RiskLimitsDTO(BaseModel):
    """Data transfer object for pre-trade risk boundaries."""

    max_order_notional: float
    max_order_qty: float
    max_gross_leverage: float
    max_net_leverage: float
    max_concentration_nav_pct: float
    max_intraday_drawdown_pct: float
    min_free_margin: float


class RiskLimitsUpdateRequest(BaseModel):
    """Request DTO to dynamically update pre-trade risk boundaries."""

    max_order_notional: float | None = Field(None, gt=0.0)
    max_order_qty: float | None = Field(None, gt=0.0)
    max_gross_leverage: float | None = Field(None, gt=0.0, le=20.0)
    max_net_leverage: float | None = Field(None, gt=0.0, le=10.0)
    max_concentration_nav_pct: float | None = Field(None, gt=0.0, le=1.0)
    max_intraday_drawdown_pct: float | None = Field(None, gt=0.0, le=1.0)
    min_free_margin: float | None = Field(None, ge=0.0)


class PanicTriggerRequest(BaseModel):
    """Request DTO to manually trigger emergency panic kill switch."""

    reason: str = Field(
        ..., min_length=3, max_length=200, description="Operator justification for triggering panic"
    )
    details: dict[str, Any] = Field(
        default_factory=dict, description="Diagnostic contextual metadata"
    )


class KillSwitchResetRequest(BaseModel):
    """Request DTO for authenticated kill switch disarm and reset."""

    admin_token: str = Field(..., min_length=8, description="Cryptographic administrative secret")


class GatewayHealthDTO(BaseModel):
    """Response DTO reporting broker gateway connection and watchdog health."""

    gateway_id: str
    status: str
    last_heartbeat_timestamp: int | None = None
    last_latency_ms: float = 0.0
    missed_sequence_count: int = 0
    is_connected: bool


class HeartbeatPingRequest(BaseModel):
    """Request DTO for inbound gateway heartbeat pulse."""

    sequence_number: int = Field(
        ..., ge=1, description="Inbound transport heartbeat sequence number"
    )
    latency_ms: float = Field(
        0.0, ge=0.0, description="Round-trip ping-pong latency in milliseconds"
    )
    timestamp_ns: int | None = Field(
        None, ge=0, description="Optional explicit epoch nanoseconds timestamp"
    )


class AutonomousStatusDTO(BaseModel):
    """Response DTO reporting the operational state and telemetry of the autonomous trading engine."""

    state: str = Field(
        ..., description="Operational state: IDLE, RUNNING, PAUSED, STOPPED, or ERROR"
    )
    iteration: int = Field(..., description="Total completed rebalancing cycle iterations")
    universe: list[str] = Field(..., description="Active trading universe symbols")
    target_allocations: dict[str, float] = Field(
        default_factory=dict, description="Last target dollar allocations"
    )
    last_step: dict[str, Any] | None = Field(
        None, description="Telemetry from the most recent rebalance step"
    )


class AutonomousStepReportDTO(BaseModel):
    """Response DTO reporting execution of a single discrete autonomous trading iteration."""

    iteration: int = Field(..., description="Completed iteration counter")
    timestamp_ns: int = Field(..., description="Execution epoch nanoseconds")
    universe: list[str] = Field(..., description="Monitored universe symbols")
    target_allocations: dict[str, float] = Field(
        ..., description="Calculated target dollar allocations"
    )
    current_positions: dict[str, float] = Field(
        ..., description="Broker share positions at cycle execution"
    )
    orders_dispatched: list[str] = Field(
        ..., description="IDs of parent orders submitted in this cycle"
    )
    duration_ms: float = Field(..., description="Cycle compute duration in milliseconds")
    haircut: float = Field(..., description="Risk circuit-breaker multiplier applied")
    is_kill_switch_active: bool = Field(
        ..., description="Whether emergency panic lockout was active"
    )


# Pre-Trade Decision Gate Schemas
class PreTradeEvaluateRequest(BaseModel):
    """Request payload to test or dry-run an order through the Bayesian Pre-Trade Decision Gate."""

    symbol: str = Field(..., description="Target ticker symbol", min_length=1)
    action: str = Field("BUY", description="Proposed action: BUY, SELL, CLOSE, or LIQUIDATE")
    quantity: float = Field(..., gt=0.0, description="Order quantity strictly positive")
    reference_price: float = Field(
        ..., gt=0.0, description="Current market reference price strictly positive"
    )
    is_position_exit: bool = Field(
        False, description="Whether this order reduces or closes an existing position"
    )
    market_spread_bps: float = Field(
        5.0, ge=0.0, description="Observed bid-ask spread in basis points"
    )
    order_book_imbalance: float = Field(
        0.0, ge=-1.0, le=1.0, description="Level 2 queue imbalance in [-1.0, 1.0]"
    )
    macro_yield_spread: float = Field(0.18, description="FRED T10Y2Y spread in percentage points")
    trailing_volatility_pct: float = Field(
        1.2, ge=0.0, description="Trailing asset volatility percentage"
    )
    current_bar_return_pct: float = Field(0.05, description="Current bar price return percentage")
    consecutive_losses: int = Field(0, ge=0, description="Consecutive losing trades count")
    current_drawdown_pct: float = Field(
        0.0, ge=0.0, description="Current portfolio drawdown percentage"
    )
    cvar_95_pct: float = Field(
        2.5, ge=0.0, description="Coherent 95% Expected Shortfall percentage"
    )
    data_age_seconds: float = Field(1.0, ge=0.0, description="Age of market quote in seconds")


class DimensionCheckDTO(BaseModel):
    """Individual analytical evaluation dimension record."""

    name: str
    passed: bool
    score: float
    details: str


class PreTradeDecisionDTO(BaseModel):
    """Audit verdict generated by the Pre-Trade Decision Gate."""

    decision_id: str
    order_id: str
    symbol: str
    action: str
    allowed: bool
    decision: str
    toxicity_probability: float
    confidence: float
    primary_code: str
    reason: str
    checks: list[DimensionCheckDTO]
    latency_us: float


class PreTradeStatusDTO(BaseModel):
    """Configuration thresholds and operational status of the Pre-Trade Decision Gate."""

    max_toxic_probability_threshold: float
    max_data_age_seconds: float
    max_consecutive_losses: int
    max_drawdown_threshold_pct: float
    prior_toxic_prob: float
    prior_log_odds: float
    decisions_count: int


# ============================================================================
# Replay Simulation & Institutional Backtesting Schemas
# ============================================================================


class SimulationRunRequest(BaseModel):
    """Execution parameters for institutional live replay simulation backtesting."""

    asset_id: str = Field("NVDA", description="Primary asset ticker symbol for backtesting")
    bar_count: int = Field(120, ge=30, le=5000, description="Number of bars to replay (minimum 30)")
    initial_capital: float = Field(10_000.0, gt=0.0, description="Initial portfolio endowment ($)")
    fee_bps: float = Field(2.0, ge=0.0, description="Exchange clearing fee in bps")
    spread_bps: float = Field(1.0, ge=0.0, description="Average half-spread in bps")
    impact_coefficient: float = Field(
        0.10, ge=0.0, description="Kyle-Obizhaeva market impact coefficient"
    )


class SimulationBenchmarkDTO(BaseModel):
    """Institutional benchmark comparison performance record."""

    name: str
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    alpha: float
    beta: float
    information_ratio: float


class SimulationRunResponse(BaseModel):
    """Institutional audit tear sheet generated by ReplayEngine and BenchmarkAuditor."""

    asset_id: str
    bar_count: int
    initial_capital: float
    final_equity: float
    total_return_pct: float
    cagr_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    realized_cvar_95_pct: float
    deflated_sharpe_ratio: float
    is_statistically_significant: bool
    total_friction_cost: float
    equity_curve: list[float]
    benchmarks: list[SimulationBenchmarkDTO]


# ============================================================================
# Econometric Stationarity & Volatility Rig Schemas
# ============================================================================


class FFDScanPointDTO(BaseModel):
    """Evaluation point along fractional differentiation spectrum d in [0.0, 1.0]."""

    d: float
    adf_stat: float
    adf_pvalue: float
    correlation: float
    is_stationary: bool


class FFDScanResponse(BaseModel):
    """Response DTO for memory-preserving fractional differentiation stationarity scan."""

    symbol: str
    optimal_d: float
    threshold_pvalue: float
    points: list[FFDScanPointDTO]


class VolatilitySeriesPointDTO(BaseModel):
    """Realized volatility time-series record."""

    timestamp_ns: int
    close: float
    parkinson_vol: float
    garman_klass_vol: float


class VolatilityResponse(BaseModel):
    """Response DTO for Parkinson and Garman-Klass realized volatility estimators."""

    symbol: str
    window: int
    latest_close: float
    annualized_parkinson_pct: float
    annualized_garman_klass_pct: float
    series: list[VolatilitySeriesPointDTO]


class TripleBarrierSimulateRequest(BaseModel):
    """Request parameters to simulate dynamic volatility triple-barrier labeling."""

    symbol: str = Field("NVDA", description="Asset ticker symbol")
    profit_multiplier: float = Field(2.0, gt=0.0, description="Take-profit barrier multiplier c1")
    stop_multiplier: float = Field(1.0, gt=0.0, description="Stop-loss barrier multiplier c2")
    horizon_bars: int = Field(30, ge=5, le=120, description="Vertical holding barrier in bars")
    volatility_window: int = Field(
        20, ge=5, le=60, description="Parkinson volatility rolling window"
    )
    side: int = Field(1, description="Trade direction (+1 Long, -1 Short)")


class BarrierTrajectoryPointDTO(BaseModel):
    """Sample bar point within triple-barrier evaluation window."""

    bar_index: int
    close_price: float
    upper_barrier: float
    lower_barrier: float
    event_type: str | None = None


class TripleBarrierSimulateResponse(BaseModel):
    """Response DTO for simulated triple-barrier labeling outcome distributions."""

    symbol: str
    total_events: int
    take_profit_hits: int
    stop_loss_hits: int
    vertical_expiration_hits: int
    take_profit_pct: float
    stop_loss_pct: float
    vertical_expiration_pct: float
    average_holding_bars: float
    average_net_return_pct: float
    sample_trajectory: list[BarrierTrajectoryPointDTO]


# ============================================================================
# Bayesian Game Theory & Jump Regimes Schemas
# ============================================================================


class CUSUMPointDTO(BaseModel):
    """Time-series observation of two-sided CUSUM detector."""

    timestamp_ns: int
    bar_index: int
    s_pos: float
    s_neg: float
    return_pct: float
    is_shock: bool


class RegimeHistoryPointDTO(BaseModel):
    """Historical point in 3-simplex regime evolution."""

    timestamp_ns: int
    bar_index: int
    p_absorption: float
    p_momentum: float
    p_panic: float


class RegimeStatusResponse(BaseModel):
    """Response DTO for Bayesian jump regime estimation."""

    symbol: str
    current_regime: str
    p_absorption: float
    p_momentum: float
    p_panic: float
    dirichlet_alphas: list[float]
    confidence_pct: float
    thermodynamic_beta: float
    cusum_alarm: bool
    cusum_threshold_h: float
    cusum_s_pos: float
    cusum_s_neg: float
    cusum_series: list[CUSUMPointDTO]
    regime_history: list[RegimeHistoryPointDTO]


class PayoffMatrixRequest(BaseModel):
    """Request DTO to evaluate minimax regret game."""

    ambiguity_beta: float = Field(
        1.5, ge=0.01, le=20.0, description="Thermodynamic ambiguity temperature"
    )
    risk_aversion: float = Field(
        2.0, ge=0.1, le=10.0, description="Arrow-Pratt risk aversion parameter"
    )


class PayoffMatrixResponse(BaseModel):
    """Response DTO for Stackelberg minimax regret matrix."""

    actions: list[str]
    counterparties: list[str]
    payoff_matrix: list[list[float]]
    regret_matrix: list[list[float]]
    worst_case_regrets: list[float]
    optimal_action: str
    worst_case_counterparty_probs: list[float]


# Backtest Studio Schemas (Phase 15 Step 15.3)
class BacktestRunRequest(BaseModel):
    """Request DTO to initiate a historical backtest run."""

    strategy_type: str = Field(
        "FracDiff_Swarm",
        description="Alpha strategy identifier: FracDiff_Swarm, Kalman_StatArb, FracDiff_Momentum, Loughran_Sentiment, Vol_Breakout, Composite_Meta",
    )
    symbols: list[str] = Field(
        default_factory=lambda: ["SPY", "QQQ", "AAPL", "NVDA", "MSFT"],
        min_length=1,
        max_length=20,
        description="Asset symbols for the backtest universe",
    )
    start_date: str | None = Field(None, description="ISO start date YYYY-MM-DD")
    end_date: str | None = Field(None, description="ISO end date YYYY-MM-DD")
    initial_cash: float = Field(100_000.0, gt=0.0, description="Initial portfolio cash")
    benchmark_symbol: str = Field("SPY", description="Benchmark symbol for Alpha/Beta")
    cost_bps: float = Field(2.0, ge=0.0, le=100.0, description="Microstructure cost in bps")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Custom strategy hyperparameters"
    )


class BacktestSummaryMetrics(BaseModel):
    """Institutional CFA performance summary metrics."""

    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    annualized_return: float
    annualized_volatility: float
    win_rate: float
    profit_factor: float
    deflated_sharpe_ratio: float | None = None
    alpha: float | None = None
    beta: float | None = None
    total_trades: int
    initial_capital: float
    final_equity: float
    net_pnl: float


class BacktestStatusResponse(BaseModel):
    """Status and full attribution report for a historical backtest."""

    backtest_id: str
    status: str = Field(..., description="QUEUED | RUNNING | COMPLETED | FAILED")
    progress: float = Field(..., ge=0.0, le=1.0)
    message: str
    strategy_type: str
    symbols: list[str]
    metrics: BacktestSummaryMetrics | None = None
    equity_curve: list[float] | None = None
    benchmark_equity_curve: list[float] | None = None
    drawdown_series: list[float] | None = None
    monthly_matrix: dict[str, dict[str, float]] | None = None
    created_at: str
    completed_at: str | None = None
    html_report_path: str | None = None


class BacktestRunResponse(BaseModel):
    """Response DTO confirming backtest job dispatch."""

    backtest_id: str
    status: str
    message: str
    estimated_duration_sec: float

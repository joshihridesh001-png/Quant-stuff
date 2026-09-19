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

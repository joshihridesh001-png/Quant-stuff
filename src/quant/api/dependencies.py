"""FastAPI dependency injection providers for database, repositories, services, and security."""

from collections.abc import Callable
from typing import Any

from fastapi import Depends, Header, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from quant.core.config import get_settings
from quant.core.security import decode_access_token, verify_api_key
from quant.data.alpaca_feed import AlpacaMarketDataFeed
from quant.data.external_providers import ExternalProviderManager
from quant.domain.interfaces import (
    IAssetRepository,
    IEventRepository,
    IGenotypeRepository,
    IMarketDataRepository,
)
from quant.execution.alpaca_gateway import AlpacaExecutionGateway
from quant.execution.gateway import ExecutionGateway, PaperExecutionGateway
from quant.execution.heartbeat import HeartbeatWatchdog
from quant.execution.kill_switch import EmergencyKillSwitch
from quant.execution.risk import (
    PortfolioRiskState,
    PreTradeRiskFirewall,
    RiskLimits,
)
from quant.execution.risk_orchestrator import RiskOrchestrator
from quant.infrastructure.database.duckdb_session import DuckDBManager
from quant.infrastructure.database.session import get_db
from quant.infrastructure.repositories.asset_repository import SqlAlchemyAssetRepository
from quant.infrastructure.repositories.duckdb_market_data_repository import (
    DuckDBMarketDataRepository,
)
from quant.infrastructure.repositories.event_repository import SqlAlchemyEventRepository
from quant.infrastructure.repositories.genotype_repository import SqlAlchemyGenotypeRepository
from quant.services.autonomous_trader import AutonomousTradingEngine
from quant.services.event_service import EventService
from quant.services.execution_service import ExecutionService
from quant.services.genotype_service import GenotypeService
from quant.services.market_data_service import MarketDataService
from quant.services.news_prediction_service import NewsPredictionService
from quant.services.risk_service import RiskService

settings = get_settings()
bearer_scheme = HTTPBearer(auto_error=False)

# Module-level singleton handle for embedded DuckDB
_duckdb_manager: DuckDBManager | None = None


def get_duckdb_manager() -> DuckDBManager:
    """Return singleton instance of embedded DuckDB session manager."""
    global _duckdb_manager
    if _duckdb_manager is None:
        _duckdb_manager = DuckDBManager()
    return _duckdb_manager


# Repository & Service Dependencies
def get_asset_repo(session: AsyncSession = Depends(get_db)) -> IAssetRepository:
    return SqlAlchemyAssetRepository(session)


def get_event_repo(session: AsyncSession = Depends(get_db)) -> IEventRepository:
    return SqlAlchemyEventRepository(session)


def get_genotype_repo(session: AsyncSession = Depends(get_db)) -> IGenotypeRepository:
    return SqlAlchemyGenotypeRepository(session)


def get_event_service(
    event_repo: IEventRepository = Depends(get_event_repo),
    asset_repo: IAssetRepository = Depends(get_asset_repo),
) -> EventService:
    return EventService(event_repo, asset_repo)


def get_genotype_service(
    genotype_repo: IGenotypeRepository = Depends(get_genotype_repo),
) -> GenotypeService:
    return GenotypeService(genotype_repo)


def get_market_data_repo(
    manager: DuckDBManager = Depends(get_duckdb_manager),
) -> IMarketDataRepository:
    """Provide DuckDB market data persistence adapter."""
    return DuckDBMarketDataRepository(manager)


def get_market_data_service(
    market_repo: IMarketDataRepository = Depends(get_market_data_repo),
    asset_repo: IAssetRepository = Depends(get_asset_repo),
) -> MarketDataService:
    """Provide market data application service."""
    return MarketDataService(market_repo, asset_repo)


# Execution & Risk Singletons & Dependencies
_risk_orchestrator: RiskOrchestrator | None = None
_paper_gateway: PaperExecutionGateway | None = None
_execution_gateway: ExecutionGateway | None = None
_execution_service: ExecutionService | None = None
_risk_service: RiskService | None = None
_alpaca_feed: AlpacaMarketDataFeed | None = None
_autonomous_trader: AutonomousTradingEngine | None = None


def get_paper_gateway() -> PaperExecutionGateway:
    """Provide singleton PaperExecutionGateway instance."""
    global _paper_gateway
    if _paper_gateway is None:
        _paper_gateway = PaperExecutionGateway()
        _paper_gateway._is_connected = True
        _paper_gateway.set_market_price("NVDA", 125.0)
        _paper_gateway.set_market_price("AAPL", 185.0)
        _paper_gateway.set_market_price("MSFT", 420.0)
        _paper_gateway.set_market_price("SPY", 510.0)
    return _paper_gateway


def get_execution_gateway() -> ExecutionGateway:
    """Provide singleton pluggable ExecutionGateway (Paper or Alpaca)."""
    global _execution_gateway
    if _execution_gateway is None:
        if (
            settings.BROKER_TYPE.lower() == "alpaca"
            and settings.ALPACA_API_KEY
            and settings.ALPACA_SECRET_KEY
        ):
            _execution_gateway = AlpacaExecutionGateway(
                api_key=settings.ALPACA_API_KEY,
                secret_key=settings.ALPACA_SECRET_KEY,
                base_url=settings.ALPACA_BASE_URL,
            )
        else:
            _execution_gateway = get_paper_gateway()
    return _execution_gateway


def get_risk_orchestrator() -> RiskOrchestrator:
    """Provide singleton live RiskOrchestrator instance configured with default institutional safety limits."""
    global _risk_orchestrator
    if _risk_orchestrator is None:
        limits = RiskLimits(
            max_order_notional=500_000.0,
            max_order_qty=50_000.0,
            max_gross_leverage=2.0,
            max_net_leverage=1.0,
            max_concentration_nav_pct=0.25,
            max_intraday_drawdown_pct=0.05,
            min_free_margin=100_000.0,
        )
        state = PortfolioRiskState(cash=1_000_000.0, initial_equity=1_000_000.0)
        firewall = PreTradeRiskFirewall(limits=limits)
        kill_switch = EmergencyKillSwitch(admin_token="DEFAULT_ADMIN_TOKEN")
        _risk_orchestrator = RiskOrchestrator(
            firewall=firewall,
            kill_switch=kill_switch,
            state=state,
            admin_token="DEFAULT_ADMIN_TOKEN",
        )
        # Register default paper broker and watchdog
        paper = get_paper_gateway()
        paper_watchdog = HeartbeatWatchdog(gateway_id="PAPER_BROKER")
        _risk_orchestrator.register_gateway(
            paper, watchdog=paper_watchdog, gateway_id="PAPER_BROKER", is_default=True
        )
        # If live execution gateway is separate (e.g. Alpaca), register and set as default
        live_gateway = get_execution_gateway()
        if live_gateway is not paper:
            alpaca_watchdog = HeartbeatWatchdog(gateway_id="ALPACA_BROKER")
            _risk_orchestrator.register_gateway(
                live_gateway, watchdog=alpaca_watchdog, gateway_id="ALPACA_BROKER", is_default=True
            )
    return _risk_orchestrator


def get_execution_service(
    orchestrator: RiskOrchestrator = Depends(get_risk_orchestrator),
    gateway: ExecutionGateway = Depends(get_execution_gateway),
) -> ExecutionService:
    """Provide live ExecutionService application coordinator."""
    global _execution_service
    if _execution_service is None:
        _execution_service = ExecutionService(orchestrator=orchestrator, gateway=gateway)
    return _execution_service


def get_risk_service(
    orchestrator: RiskOrchestrator = Depends(get_risk_orchestrator),
) -> RiskService:
    """Provide live RiskService application coordinator."""
    global _risk_service
    if _risk_service is None:
        _risk_service = RiskService(orchestrator=orchestrator)
    return _risk_service


def get_market_feed(
    manager: DuckDBManager = Depends(get_duckdb_manager),
) -> AlpacaMarketDataFeed:
    """Provide singleton AlpacaMarketDataFeed instance bound to active DuckDBManager."""
    global _alpaca_feed
    current_manager = (
        getattr(_alpaca_feed._repository, "_manager", None) if _alpaca_feed is not None else None
    )
    if _alpaca_feed is None or current_manager is not manager:
        repo = DuckDBMarketDataRepository(manager)
        _alpaca_feed = AlpacaMarketDataFeed(
            repository=repo,
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
            data_url=settings.ALPACA_DATA_URL,
        )
    return _alpaca_feed


def get_autonomous_trader(
    execution_service: ExecutionService = Depends(get_execution_service),
    gateway: ExecutionGateway = Depends(get_execution_gateway),
    market_feed: AlpacaMarketDataFeed = Depends(get_market_feed),
) -> AutonomousTradingEngine:
    """Provide singleton AutonomousTradingEngine daemon bound to active gateway and feed."""
    global _autonomous_trader
    if (
        _autonomous_trader is None
        or getattr(_autonomous_trader, "_market_feed", None) is not market_feed
        or getattr(_autonomous_trader, "_gateway", None) is not gateway
    ):
        _autonomous_trader = AutonomousTradingEngine(
            execution_service=execution_service,
            gateway=gateway,
            market_feed=market_feed,
            universe=settings.TRADING_UNIVERSE,
            interval_sec=settings.AUTONOMOUS_LOOP_INTERVAL_SEC,
            min_trade_notional=settings.MIN_TRADE_NOTIONAL,
        )
    return _autonomous_trader


_external_provider_manager: ExternalProviderManager | None = None


def get_external_provider_manager() -> ExternalProviderManager:
    """Provide singleton ExternalProviderManager instance."""
    global _external_provider_manager
    if _external_provider_manager is None:
        _external_provider_manager = ExternalProviderManager(settings=settings)
    return _external_provider_manager


_news_prediction_service: NewsPredictionService | None = None


def get_news_prediction_service() -> NewsPredictionService:
    """Provide singleton NewsPredictionService instance."""
    global _news_prediction_service
    if _news_prediction_service is None:
        autonomous_engine = get_autonomous_trader()
        provider_manager = get_external_provider_manager()
        _news_prediction_service = NewsPredictionService(
            autonomous_engine=autonomous_engine,
            provider_manager=provider_manager,
        )
    return _news_prediction_service


# Security & Role Dependencies
async def require_api_key(x_api_key: str | None = Header(None)) -> str:
    """Validate machine-to-machine ingestion API key."""
    if not x_api_key or not verify_api_key(x_api_key, settings.API_KEY_SECRET):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return x_api_key


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> dict[str, Any]:
    """Validate Bearer JWT and extract user claims."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization Bearer header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = decode_access_token(credentials.credentials)
        return payload
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {str(exc)}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def require_role(allowed_roles: list[str]) -> Callable[..., Any]:
    """Enforce Role-Based Access Control on authenticated user claims."""

    async def role_checker(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        user_role = user.get("role", "GUEST")
        if user_role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required one of: {allowed_roles}",
            )
        return user

    return role_checker

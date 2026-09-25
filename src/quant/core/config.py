"""Application configuration and environment settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration management for the quant engine."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    PROJECT_NAME: str = "News-Driven Quantitative Prediction Engine"
    API_V1_PREFIX: str = "/api/v1"

    # Relational Database (Transactional Metadata: Assets, News, Genotypes, Auth)
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/quant.db"
    DATABASE_ECHO: bool = False

    # Embedded Columnar Storage (High-Throughput Time-Series Market Data)
    # Purpose: Designates the local filesystem path for the embedded DuckDB database file
    # Dependencies: Used by DuckDBMarketDataRepository for ultra-low latency bar queries
    # Invariant: Must point to a writable directory or ':memory:' for transient test execution
    DUCKDB_PATH: str = "data/market_data.duckdb"

    # Security & Authentication
    API_KEY_SECRET: str = "dev-secret-api-key-992384918237192837"
    JWT_SECRET_KEY: str = "dev-jwt-super-secret-key-change-in-production-12983719283"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Cross-Origin Resource Sharing
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    @model_validator(mode="after")
    def validate_production_security(self) -> Settings:
        """Enforce strict cryptographic secret standards and prevent wildcard CORS in production."""
        if self.ENVIRONMENT.lower() == "production":
            if "dev-" in self.JWT_SECRET_KEY or len(self.JWT_SECRET_KEY) < 32:
                raise ValueError(
                    "Production mode requires a secure, non-default JWT_SECRET_KEY (at least 32 characters)"
                )
            if "dev-" in self.API_KEY_SECRET or len(self.API_KEY_SECRET) < 32:
                raise ValueError(
                    "Production mode requires a secure, non-default API_KEY_SECRET (at least 32 characters)"
                )
            if "*" in self.CORS_ORIGINS:
                raise ValueError(
                    "Wildcard CORS_ORIGINS ['*'] is strictly prohibited in production mode"
                )
        return self

    # Execution Broker & Gateway Configuration (Paper or Live Alpaca Markets)
    # Purpose: Configures target broker connectivity, credentials, and endpoints
    # Invariant: Defaults to "paper" with zero external network requirement
    BROKER_TYPE: str = "paper"  # "paper" | "alpaca"
    ALPACA_API_KEY: str = ""
    ALPACA_SECRET_KEY: str = ""
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"
    ALPACA_DATA_URL: str = "https://data.alpaca.markets"

    # Autonomous Swarm Live Trading Loop Configuration
    # Purpose: Sets monitored universe and clock interval for autonomous rebalancing
    TRADING_UNIVERSE: list[str] = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT"]
    AUTONOMOUS_LOOP_INTERVAL_SEC: float = 4.0
    MIN_TRADE_NOTIONAL: float = 100.0

    # External Quantitative Data Providers (Selected from public-apis)
    # Purpose: Configure external macroeconomic, news, and market data API credentials
    # Invariant: Empty strings denote disabled or mock/fallback mode; safe zero-secret defaults
    FRED_API_KEY: str = ""
    FINNHUB_API_KEY: str = ""
    NEWS_API_KEY: str = ""
    POLYGON_API_KEY: str = ""
    ALPHA_VANTAGE_API_KEY: str = ""
    TWELVE_DATA_API_KEY: str = ""
    FMP_API_KEY: str = ""
    OPENFIGI_API_KEY: str = ""
    SEC_EDGAR_USER_AGENT: str = "QuantEngine Research Contact@quantplatform.internal"
    BINANCE_API_KEY: str = ""


@lru_cache
def get_settings() -> Settings:
    """Return cached singleton instance of system settings."""
    return Settings()

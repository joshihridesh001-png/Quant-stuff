"""Application configuration and environment settings."""

from functools import lru_cache

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

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./quant.db"
    DATABASE_ECHO: bool = False

    # Security & Authentication
    API_KEY_SECRET: str = "dev-secret-api-key-992384918237192837"
    JWT_SECRET_KEY: str = "dev-jwt-super-secret-key-change-in-production-12983719283"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Cross-Origin Resource Sharing
    CORS_ORIGINS: list[str] = ["*"]


@lru_cache
def get_settings() -> Settings:
    """Return cached singleton instance of system settings."""
    return Settings()

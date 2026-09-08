"""Pytest configuration and global asynchronous fixtures."""

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from quant.api.dependencies import get_db
from quant.core.config import get_settings
from quant.core.security import create_access_token
from quant.infrastructure.database.models import Base
from quant.main import app

settings = get_settings()


@pytest_asyncio.fixture(scope="function")
async def test_engine() -> AsyncGenerator[AsyncEngine]:
    """Provide an isolated in-memory SQLite async database engine per test."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        future=True,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(test_engine: AsyncEngine) -> AsyncGenerator[AsyncSession]:
    """Provide a clean transactional async session bound to the test database."""
    session_maker = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    async with session_maker() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient]:
    """Provide an authenticated asynchronous HTTP test client with database override."""

    async def override_get_db() -> AsyncGenerator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    # Provide isolated in-memory DuckDB instance per test client session
    from quant.api.dependencies import get_duckdb_manager
    from quant.infrastructure.database.duckdb_session import DuckDBManager

    test_duckdb = DuckDBManager(":memory:")
    app.dependency_overrides[get_duckdb_manager] = lambda: test_duckdb

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac

    test_duckdb.close()
    app.dependency_overrides.clear()


@pytest.fixture
def api_key_headers() -> dict[str, str]:
    """Headers containing valid machine-to-machine API key."""
    return {"X-API-Key": settings.API_KEY_SECRET}


@pytest.fixture
def admin_jwt_headers() -> dict[str, str]:
    """Headers containing valid ADMIN Bearer JWT."""
    token = create_access_token(payload={"sub": "test-admin", "role": "ADMIN"})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def researcher_jwt_headers() -> dict[str, str]:
    """Headers containing valid RESEARCHER Bearer JWT."""
    token = create_access_token(payload={"sub": "test-researcher", "role": "RESEARCHER"})
    return {"Authorization": f"Bearer {token}"}

"""FastAPI dependency injection providers for database, repositories, services, and security."""

from collections.abc import Callable
from typing import Any

from fastapi import Depends, Header, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from quant.core.config import get_settings
from quant.core.security import decode_access_token, verify_api_key
from quant.domain.interfaces import IAssetRepository, IEventRepository, IGenotypeRepository
from quant.infrastructure.database.session import get_db
from quant.infrastructure.repositories.asset_repository import SqlAlchemyAssetRepository
from quant.infrastructure.repositories.event_repository import SqlAlchemyEventRepository
from quant.infrastructure.repositories.genotype_repository import SqlAlchemyGenotypeRepository
from quant.services.event_service import EventService
from quant.services.genotype_service import GenotypeService

settings = get_settings()
bearer_scheme = HTTPBearer(auto_error=False)


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

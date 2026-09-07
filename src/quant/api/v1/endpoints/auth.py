"""Authentication endpoint issuing JWT tokens for API clients."""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from quant.api.v1.schemas import TokenResponse
from quant.core.config import get_settings
from quant.core.security import create_access_token

router = APIRouter(prefix="/auth", tags=["Authentication"])
settings = get_settings()


class LoginRequest(BaseModel):
    username: str
    password: str
    role: str = "RESEARCHER"


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="Acquire JWT Bearer token",
)
async def login_for_access_token(payload: LoginRequest) -> TokenResponse:
    """Issue a signed JWT for researcher or admin credentials."""
    # In production, verify user credentials against a users table.
    # For Sprint 1 development/testing, validate against dev credentials.
    if payload.password != "quant-secret-pass":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    token = create_access_token(payload={"sub": payload.username, "role": payload.role.upper()})
    return TokenResponse(access_token=token)

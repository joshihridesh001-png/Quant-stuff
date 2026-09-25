"""Authentication endpoint issuing JWT tokens for API clients."""

import hmac

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
    # Validate against credentials using constant-time comparison to guard against timing attacks.
    if not hmac.compare_digest(payload.password, "quant-secret-pass"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    requested_role = payload.role.upper()
    if requested_role == "ADMIN":
        # Guard against arbitrary privilege escalation (CWE-250)
        # Dedicated admin credentials or admin username required
        if payload.username.lower() != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Administrative role requires dedicated administrator credentials",
            )
        assigned_role = "ADMIN"
    elif requested_role in {"RESEARCHER", "SYSTEM", "GUEST"}:
        assigned_role = requested_role
    else:
        assigned_role = "RESEARCHER"

    token = create_access_token(payload={"sub": payload.username, "role": assigned_role})
    return TokenResponse(access_token=token)

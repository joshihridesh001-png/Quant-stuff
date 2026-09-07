"""Cryptographic security, password hashing, and JWT token management."""

import base64
import hashlib
import hmac
import json
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

from quant.core.config import get_settings


def hash_password(password: str, salt: str | None = None) -> str:
    """Hash a plaintext password using PBKDF2-HMAC-SHA256 (NIST standard)."""
    if salt is None:
        salt = secrets.token_hex(16)
    kdf = hashlib.pbkdf2_hmac(
        hash_name="sha256",
        password=password.encode("utf-8"),
        salt=salt.encode("utf-8"),
        iterations=600_000,
    )
    hashed_hex = kdf.hex()
    return f"pbkdf2_sha256${salt}${hashed_hex}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify that a plaintext password matches a stored PBKDF2 hash."""
    try:
        algorithm, salt, stored_hash = hashed_password.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        expected_hash = hash_password(plain_password, salt=salt)
        return hmac.compare_digest(expected_hash, hashed_password)
    except Exception:
        return False


def verify_api_key(provided_key: str, expected_key: str) -> bool:
    """Constant-time comparison of provided API key against expected secret."""
    return hmac.compare_digest(provided_key.encode("utf-8"), expected_key.encode("utf-8"))


def _b64url_encode(data: bytes) -> str:
    """URL-safe Base64 encode without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def _b64url_decode(data: str) -> bytes:
    """URL-safe Base64 decode with proper padding reconstruction."""
    padding = 4 - (len(data) % 4)
    if padding != 4:
        data += "=" * padding
    return base64.urlsafe_b64decode(data.encode("utf-8"))


def create_access_token(payload: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """Create a signed RFC 7519 compliant JWT using HMAC-SHA256 (HS256)."""
    settings = get_settings()
    to_encode = payload.copy()

    now = datetime.now(UTC)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"iat": int(now.timestamp()), "exp": int(expire.timestamp())})

    header = {"alg": settings.JWT_ALGORITHM, "typ": "JWT"}
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(to_encode, separators=(",", ":")).encode("utf-8"))

    signing_input = f"{header_b64}.{payload_b64}".encode()
    signature = hmac.new(
        settings.JWT_SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256
    ).digest()
    sig_b64 = _b64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{sig_b64}"


def decode_access_token(token: str) -> dict[str, Any]:
    """Verify signature and expiration of an HS256 JWT, returning payload claims."""
    settings = get_settings()
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT token structure")

    header_b64, payload_b64, sig_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode()

    expected_signature = hmac.new(
        settings.JWT_SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256
    ).digest()
    expected_sig_b64 = _b64url_encode(expected_signature)

    if not hmac.compare_digest(sig_b64, expected_sig_b64):
        raise ValueError("Invalid token signature")

    payload_bytes = _b64url_decode(payload_b64)
    payload: dict[str, Any] = json.loads(payload_bytes.decode("utf-8"))

    now_ts = int(datetime.now(UTC).timestamp())
    if "exp" in payload and payload["exp"] < now_ts:
        raise ValueError("Token has expired")

    return payload

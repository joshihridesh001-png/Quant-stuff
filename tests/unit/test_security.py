"""Unit tests for cryptographic primitives and token security."""

from datetime import timedelta

import pytest

from quant.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_api_key,
    verify_password,
)


def test_password_hashing_and_verification() -> None:
    secret = "SuperQuantitativePassw0rd!"
    hashed = hash_password(secret)

    assert hashed.startswith("pbkdf2_sha256$")
    assert verify_password(secret, hashed) is True
    assert verify_password("WrongPassword123", hashed) is False


def test_tampered_password_hash() -> None:
    assert verify_password("pass", "malformed_hash") is False
    assert verify_password("pass", "unknown_alg$salt$hash") is False


def test_api_key_verification() -> None:
    valid_key = "my-secret-key-123456"
    assert verify_api_key(valid_key, valid_key) is True
    assert verify_api_key("wrong-key", valid_key) is False


def test_jwt_token_lifecycle() -> None:
    claims = {"sub": "trader-42", "role": "RESEARCHER"}
    token = create_access_token(payload=claims, expires_delta=timedelta(minutes=15))

    decoded = decode_access_token(token)
    assert decoded["sub"] == "trader-42"
    assert decoded["role"] == "RESEARCHER"
    assert "exp" in decoded
    assert "iat" in decoded


def test_jwt_expired_token() -> None:
    claims = {"sub": "trader-42"}
    token = create_access_token(payload=claims, expires_delta=timedelta(seconds=-10))

    with pytest.raises(ValueError, match="Token has expired"):
        decode_access_token(token)


def test_jwt_tampered_signature() -> None:
    token = create_access_token(payload={"sub": "trader-42"})
    parts = token.split(".")
    # Tamper with signature
    tampered = f"{parts[0]}.{parts[1]}.tampered_signature"

    with pytest.raises(ValueError, match="Invalid token signature"):
        decode_access_token(tampered)

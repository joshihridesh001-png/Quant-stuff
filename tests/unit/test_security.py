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


def test_jwt_missing_exp_claim() -> None:
    import hashlib
    import hmac
    import json

    from quant.core.config import get_settings
    from quant.core.security import _b64url_encode

    settings = get_settings()
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"sub": "trader-42"}  # missing "exp"
    h_b64 = _b64url_encode(json.dumps(header).encode())
    p_b64 = _b64url_encode(json.dumps(payload).encode())
    sig = hmac.new(
        settings.JWT_SECRET_KEY.encode(), f"{h_b64}.{p_b64}".encode(), hashlib.sha256
    ).digest()
    token = f"{h_b64}.{p_b64}.{_b64url_encode(sig)}"

    with pytest.raises(ValueError, match="Token missing mandatory 'exp' claim"):
        decode_access_token(token)


def test_jwt_mismatched_alg_in_header() -> None:
    import hashlib
    import hmac
    import json

    from quant.core.config import get_settings
    from quant.core.security import _b64url_encode

    settings = get_settings()
    header = {"alg": "none", "typ": "JWT"}
    payload = {"sub": "trader-42", "exp": 9999999999}
    h_b64 = _b64url_encode(json.dumps(header).encode())
    p_b64 = _b64url_encode(json.dumps(payload).encode())
    sig = hmac.new(
        settings.JWT_SECRET_KEY.encode(), f"{h_b64}.{p_b64}".encode(), hashlib.sha256
    ).digest()
    token = f"{h_b64}.{p_b64}.{_b64url_encode(sig)}"

    with pytest.raises(ValueError, match="Unsupported or mismatched JWT algorithm"):
        decode_access_token(token)


def test_argon2id_hashing_and_verification() -> None:
    try:
        import argon2  # noqa: F401
    except ImportError:
        with pytest.raises(NotImplementedError):
            hash_password("test_pass", algorithm="argon2id")
        return

    pwd = "InstitutionalArgon2Password#1"
    hashed = hash_password(pwd, algorithm="argon2id")
    assert hashed.startswith("argon2id$")
    assert verify_password(pwd, hashed) is True
    assert verify_password("WrongPassword", hashed) is False


def test_production_settings_validation() -> None:
    from quant.core.config import Settings

    # Default dev secrets in production must raise ValueError
    with pytest.raises(ValueError, match="secure, non-default JWT_SECRET_KEY"):
        Settings(ENVIRONMENT="production")

    # Providing secure JWT secret but dev API key secret must raise ValueError
    with pytest.raises(ValueError, match="secure, non-default API_KEY_SECRET"):
        Settings(
            ENVIRONMENT="production",
            JWT_SECRET_KEY="A_Very_Long_And_Super_Secure_JWT_Key_For_Production_12345",
        )

    # Providing wildcard CORS in production must raise ValueError
    with pytest.raises(ValueError, match="Wildcard CORS_ORIGINS"):
        Settings(
            ENVIRONMENT="production",
            JWT_SECRET_KEY="A_Very_Long_And_Super_Secure_JWT_Key_For_Production_12345",
            API_KEY_SECRET="A_Very_Long_And_Super_Secure_API_Key_For_Production_12345",
            CORS_ORIGINS=["*"],
        )

    # Compliant production settings must succeed
    prod_settings = Settings(
        ENVIRONMENT="production",
        JWT_SECRET_KEY="A_Very_Long_And_Super_Secure_JWT_Key_For_Production_12345",
        API_KEY_SECRET="A_Very_Long_And_Super_Secure_API_Key_For_Production_12345",
        CORS_ORIGINS=["https://quant.internal.firm"],
    )
    assert prod_settings.ENVIRONMENT == "production"

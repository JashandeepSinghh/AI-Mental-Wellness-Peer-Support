
"""
Authentication Token Security Module
=====================================

Responsibilities:
- Create signed JWT access tokens.
- Decode and validate JWT access tokens.
- Generate cryptographically secure refresh tokens.
- Hash refresh tokens before database storage.
- Verify refresh tokens against stored hashes.

Token architecture:
- Access tokens: Signed JWT using HS256.
- Refresh tokens: Opaque random tokens.
- Refresh-token storage: SHA-256 hashes only.

Important:
- Never store raw refresh tokens in the database.
- Never log access or refresh tokens.
- Refresh-token rotation is handled by the authentication service.
"""

import hashlib
import hmac
import secrets

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import jwt

from backend.app.core.config import get_settings


# ============================================================
# 1. TOKEN CONFIGURATION
# ============================================================

settings = get_settings()

JWT_ALGORITHM = "HS256"

# Fixed application identifiers prevent tokens issued for
# another application or intended for another audience
# from being accepted here.

JWT_ISSUER = "ai-mental-wellness-api"
JWT_AUDIENCE = "ai-mental-wellness-client"

ACCESS_TOKEN_EXPIRE_MINUTES = (
    settings.ACCESS_TOKEN_EXPIRE_MINUTES
)

REFRESH_TOKEN_EXPIRE_DAYS = (
    settings.REFRESH_TOKEN_EXPIRE_DAYS
)


# ============================================================
# 2. SECRET KEY VALIDATION
# ============================================================

def _get_secret_key() -> str:
    """
    Retrieve and validate the JWT signing secret.

    Raises:
        RuntimeError: If the secret is missing or too short.
    """

    secret_key = settings.SECRET_KEY

    if not secret_key or len(secret_key) < 32:
        raise RuntimeError(
            "A valid SECRET_KEY containing at least "
            "32 characters is required for JWT operations."
        )

    return secret_key


# ============================================================
# 3. ACCESS TOKEN CREATION
# ============================================================

def create_access_token(
    user_id: UUID | str,
) -> str:
    """
    Create a signed JWT access token.

    Args:
        user_id: UUID or string representation of the user ID.

    Returns:
        str: Encoded JWT access token.

    Claims:
        sub: User UUID.
        type: access.
        jti: Unique token identifier.
        iat: Issued-at timestamp.
        exp: Expiration timestamp.
        iss: Trusted token issuer.
        aud: Intended token audience.
    """

    try:
        normalized_user_id = str(UUID(str(user_id)))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(
            "A valid user UUID is required."
        ) from exc

    now = datetime.now(timezone.utc)

    expiration = now + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": normalized_user_id,
        "type": "access",
        "jti": str(uuid4()),
        "iat": now,
        "exp": expiration,
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
    }

    return jwt.encode(
        payload,
        _get_secret_key(),
        algorithm=JWT_ALGORITHM,
    )


# ============================================================
# 4. ACCESS TOKEN DECODING AND VALIDATION
# ============================================================

def decode_access_token(
    token: str,
) -> dict[str, Any]:
    """
    Decode and validate a JWT access token.

    Validates:
        - Signature.
        - Signing algorithm.
        - Expiration.
        - Issuer.
        - Audience.
        - Required claims.
        - Access-token type.
        - User UUID.

    Args:
        token: Encoded JWT access token.

    Returns:
        dict: Validated JWT claims.

    Raises:
        jwt.InvalidTokenError: If the token is invalid.
    """

    if not isinstance(token, str) or not token:
        raise jwt.InvalidTokenError(
            "Invalid access token."
        )

    payload = jwt.decode(
        token,
        _get_secret_key(),
        algorithms=[JWT_ALGORITHM],
        issuer=JWT_ISSUER,
        audience=JWT_AUDIENCE,
        options={
            "require": [
                "sub",
                "type",
                "jti",
                "iat",
                "exp",
                "iss",
                "aud",
            ],
        },
    )

    if payload.get("type") != "access":
        raise jwt.InvalidTokenError(
            "Token is not an access token."
        )

    try:
        UUID(payload["sub"])
        UUID(payload["jti"])
    except (ValueError, TypeError, AttributeError) as exc:
        raise jwt.InvalidTokenError(
            "Invalid token identity claims."
        ) from exc

    return payload


# ============================================================
# 5. REFRESH TOKEN GENERATION
# ============================================================

def create_refresh_token() -> str:
    """
    Generate a cryptographically secure opaque refresh token.

    Returns:
        str: Random URL-safe refresh token.

    Important:
        Store only its SHA-256 hash in the database.
        Return the raw token to the client only when issuing
        or rotating the session.
    """

    return secrets.token_urlsafe(64)


# ============================================================
# 6. REFRESH TOKEN HASHING
# ============================================================

def hash_refresh_token(
    refresh_token: str,
) -> str:
    """
    Calculate the SHA-256 hash of a refresh token.

    Args:
        refresh_token: Raw refresh token.

    Returns:
        str: Hexadecimal SHA-256 digest.

    Important:
        Never store the raw refresh token in the database.
    """

    if not isinstance(refresh_token, str):
        raise TypeError(
            "Refresh token must be a string."
        )

    if not refresh_token:
        raise ValueError(
            "Refresh token cannot be empty."
        )

    return hashlib.sha256(
        refresh_token.encode("utf-8")
    ).hexdigest()


# ============================================================
# 7. REFRESH TOKEN VERIFICATION
# ============================================================

def verify_refresh_token(
    refresh_token: str,
    stored_hash: str,
) -> bool:
    """
    Verify a raw refresh token against its stored hash.

    Args:
        refresh_token: Raw token supplied by the client.
        stored_hash: SHA-256 digest stored in the database.

    Returns:
        bool: True if the token matches, otherwise False.
    """

    if not isinstance(refresh_token, str):
        return False

    if not isinstance(stored_hash, str):
        return False

    if not refresh_token or not stored_hash:
        return False

    calculated_hash = hash_refresh_token(
        refresh_token
    )

    return hmac.compare_digest(
        calculated_hash,
        stored_hash,
    )


# ============================================================
# 8. REFRESH TOKEN EXPIRATION
# ============================================================

def get_refresh_token_expiration() -> datetime:
    """
    Calculate the expiration time for a new refresh session.

    Returns:
        datetime: Timezone-aware UTC expiration timestamp.
    """

    return datetime.now(timezone.utc) + timedelta(
        days=REFRESH_TOKEN_EXPIRE_DAYS
    )

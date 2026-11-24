
"""
Email Verification Token Utilities
===================================

Responsibilities:
- Generate cryptographically secure verification tokens.
- Hash tokens before database storage.
- Construct email verification URLs.
- Validate token format.

Security:
- Raw tokens must never be stored in the database.
- Only SHA-256 token hashes should be persisted.
- Verification tokens must be single-use.
- Token expiration is controlled by application configuration.
"""

import hashlib
import re
import secrets
from urllib.parse import urlencode

from backend.app.core.config import get_settings


# ============================================================
# 1. TOKEN CONFIGURATION
# ============================================================

TOKEN_BYTE_LENGTH = 32

TOKEN_HASH_LENGTH = 64

TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{40,100}$")


# ============================================================
# 2. TOKEN GENERATION
# ============================================================

def generate_verification_token() -> str:
    """
    Generate a cryptographically secure random token.

    The returned raw token is intended to be sent to the
    user's email address.

    Never store this raw token in the database.
    """

    return secrets.token_urlsafe(TOKEN_BYTE_LENGTH)


# ============================================================
# 3. TOKEN HASHING
# ============================================================

def hash_verification_token(token: str) -> str:
    """
    Generate a SHA-256 hash of a verification token.

    The hash is safe to store in the database because the
    original token has high cryptographic randomness.

    Args:
        token: Original verification token.

    Returns:
        64-character hexadecimal SHA-256 hash.
    """

    if not isinstance(token, str):
        raise TypeError("Verification token must be a string.")

    if not TOKEN_PATTERN.fullmatch(token):
        raise ValueError("Invalid verification token format.")

    return hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()


# ============================================================
# 4. TOKEN HASH COMPARISON
# ============================================================

def verify_token_hash(
    token: str,
    stored_hash: str,
) -> bool:
    """
    Compare a raw token against its stored SHA-256 hash.

    Uses constant-time comparison to reduce timing
    information leakage.
    """

    if not isinstance(stored_hash, str):
        return False

    if len(stored_hash) != TOKEN_HASH_LENGTH:
        return False

    try:

        calculated_hash = hash_verification_token(token)

    except (TypeError, ValueError):
        return False

    return secrets.compare_digest(
        calculated_hash,
        stored_hash,
    )


# ============================================================
# 5. VERIFICATION URL GENERATION
# ============================================================

def build_verification_url(token: str) -> str:
    """
    Construct the frontend email verification URL.

    The raw token is included only in the email link.

    Example:
        http://localhost:5173/verify-email?token=...

    The frontend verification page will later submit this
    token to the backend verification endpoint.
    """

    if not isinstance(token, str):
        raise TypeError("Verification token must be a string.")

    if not TOKEN_PATTERN.fullmatch(token):
        raise ValueError("Invalid verification token format.")

    settings = get_settings()

    base_url = settings.FRONTEND_BASE_URL.rstrip("/")

    query_string = urlencode({
        "token": token,
    })

    return (
        f"{base_url}/verify-email"
        f"?{query_string}"
    )
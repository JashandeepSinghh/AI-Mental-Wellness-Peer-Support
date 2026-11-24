"""
API Authentication Dependencies
================================

Provides reusable authentication dependencies for protected
API endpoints.

Security:
- Accepts JWT access tokens through Bearer authentication.
- Validates tokens using the existing token security module.
- Retrieves users using the validated JWT subject.
- Allows access only to active user accounts.
- Never trusts user IDs supplied by the client.
"""

from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import InvalidTokenError
from sqlalchemy.orm import Session

from backend.app.auth.tokens import decode_access_token
from backend.app.database.models import User
from backend.app.database.session import get_db


# ============================================================
# 1. BEARER AUTHENTICATION CONFIGURATION
# ============================================================

bearer_scheme = HTTPBearer(
    auto_error=False,
)


# ============================================================
# 2. AUTHENTICATED USER DEPENDENCY
# ============================================================

def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        bearer_scheme
    ),
    db: Session = Depends(get_db),
) -> User:

    """
    Validate the access token and return the authenticated user.

    Raises:
        HTTPException: If authentication fails or the account
        is not active.
    """

    # --------------------------------------------------------
    # A. Check whether a Bearer token was provided.
    # --------------------------------------------------------

    if credentials is None:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # --------------------------------------------------------
    # B. Decode and validate the access token.
    # --------------------------------------------------------

    try:

        claims = decode_access_token(
            credentials.credentials
        )

    except InvalidTokenError:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication is required.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None

    # --------------------------------------------------------
    # C. Extract the authenticated user's UUID.
    # --------------------------------------------------------

    user_id = UUID(claims["sub"])

    # --------------------------------------------------------
    # D. Retrieve the user from the database.
    # --------------------------------------------------------

    user = (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )

    if user is None:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # --------------------------------------------------------
    # E. Verify that the account is active.
    # --------------------------------------------------------

    if user.account_status != "active":

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # --------------------------------------------------------
    # F. Return the authenticated user.
    # --------------------------------------------------------

    return user

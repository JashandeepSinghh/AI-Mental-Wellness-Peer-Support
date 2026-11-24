"""
Refresh Token Management Service
=================================

Responsibilities:
- Validate refresh tokens against stored hashes.
- Verify session expiration and revocation status.
- Rotate refresh tokens after successful validation.
- Revoke replaced sessions.
- Detect reuse of previously rotated refresh tokens.
- Revoke active sessions after detected token reuse.

Security:
- Never store plaintext refresh tokens.
- Never return refresh tokens in exception messages.
- Reject expired and revoked sessions.
- Revoke active sessions when rotated-token reuse is detected.
- Keep session rotation inside a database transaction.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from backend.app.auth.tokens import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    create_refresh_token,
    get_refresh_token_expiration,
    hash_refresh_token,
    verify_refresh_token,
)

from backend.app.database.models.audit_log import AuditLog
from backend.app.database.models.user import User
from backend.app.database.models.user_session import UserSession


# ============================================================
# 1. REFRESH TOKEN EXCEPTION
# ============================================================

class RefreshTokenError(Exception):
    """Raised when refresh-token authentication fails."""


# ============================================================
# 2. REFRESH RESULT
# ============================================================

@dataclass(frozen=True)
class RefreshResult:
    """Contains newly issued authentication tokens."""

    user_id: UUID
    role: str

    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)

    token_type: str = "bearer"

    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60


# ============================================================
# 3. UTC DATETIME NORMALIZATION
# ============================================================

def _as_utc(value: datetime) -> datetime:
    """
    Normalize database datetimes to timezone-aware UTC.

    SQLite may return naive datetimes even when the
    SQLAlchemy column uses timezone=True.
    """

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


def _add_refresh_audit(
    db: Session,
    *,
    actor_user_id: UUID,
    action: str,
    outcome: str,
    reason: str,
    session_id: UUID,
    category: str = "authentication",
) -> None:
    """Add a non-sensitive refresh/logout audit event."""

    db.add(
        AuditLog(
            actor_user_id=actor_user_id,
            category=category,
            action=action,
            outcome=outcome,
            target_type="session",
            target_id=str(session_id),
            metadata_json={"reason": reason},
        )
    )


# ============================================================
# 4. REVOKE ALL ACTIVE USER SESSIONS
# ============================================================

def _revoke_all_active_sessions(
    db: Session,
    user_id: UUID,
    now: datetime,
) -> None:
    """
    Revoke every currently active session belonging to a user.
    """

    db.execute(
        update(UserSession)
        .where(
            UserSession.user_id == user_id,
            UserSession.revoked_at.is_(None),
        )
        .values(
            revoked_at=now,
        )
    )


# ============================================================
# 5. REFRESH TOKEN ROTATION
# ============================================================

def refresh_user_session(
    db: Session,
    refresh_token: str,
) -> RefreshResult:
    """
    Validate and rotate an authenticated user session.

    Workflow:
        1. Hash the supplied refresh token.
        2. Find its associated session.
        3. Detect previously rotated-token reuse.
        4. Validate session expiration.
        5. Validate account availability.
        6. Generate replacement tokens.
        7. Create a replacement session.
        8. Revoke and link the old session.
        9. Commit the rotation transaction.

    Raises:
        RefreshTokenError:
            If the token is invalid, expired, revoked,
            reused, or associated with an unavailable account.
    """

    # --------------------------------------------------------
    # A. Validate input
    # --------------------------------------------------------

    if not isinstance(refresh_token, str) or not refresh_token:
        raise RefreshTokenError(
            "Refresh authentication failed."
        )

    now = datetime.now(timezone.utc)

    try:

        # ----------------------------------------------------
        # B. Find the session using the token hash
        # ----------------------------------------------------

        supplied_hash = hash_refresh_token(refresh_token)

        statement = (
            select(UserSession)
            .where(
                UserSession.refresh_token_hash == supplied_hash
            )
            .with_for_update()
        )

        user_session = db.execute(
            statement
        ).scalar_one_or_none()

        if user_session is None:
            raise RefreshTokenError(
                "Refresh authentication failed."
            )

        # ----------------------------------------------------
        # C. Verify supplied token against stored hash
        # ----------------------------------------------------

        if not verify_refresh_token(
            refresh_token,
            user_session.refresh_token_hash,
        ):
            raise RefreshTokenError(
                "Refresh authentication failed."
            )

        # ----------------------------------------------------
        # D. Detect reuse of a rotated refresh token
        # ----------------------------------------------------

        if user_session.revoked_at is not None:

            if user_session.replaced_by_session_id is not None:

                _revoke_all_active_sessions(
                    db,
                    user_session.user_id,
                    now,
                )

                _add_refresh_audit(
                    db,
                    actor_user_id=user_session.user_id,
                    action="refresh_token_reuse",
                    outcome="denied",
                    reason="rotated_token_reuse",
                    session_id=user_session.id,
                    category="security",
                )

                db.commit()

            raise RefreshTokenError(
                "Refresh authentication failed."
            )

        # ----------------------------------------------------
        # E. Check session expiration
        # ----------------------------------------------------

        if _as_utc(user_session.expires_at) <= now:

            user_session.revoked_at = now

            _add_refresh_audit(
                db,
                actor_user_id=user_session.user_id,
                action="refresh",
                outcome="denied",
                reason="expired",
                session_id=user_session.id,
            )

            db.commit()

            raise RefreshTokenError(
                "Refresh authentication failed."
            )

        # ----------------------------------------------------
        # F. Verify account availability
        # ----------------------------------------------------

        user = db.get(
            User,
            user_session.user_id,
        )

        if user is None or user.account_status != "active":

            _revoke_all_active_sessions(
                db,
                user_session.user_id,
                now,
            )

            _add_refresh_audit(
                db,
                actor_user_id=user_session.user_id,
                action="refresh",
                outcome="denied",
                reason="account_unavailable",
                session_id=user_session.id,
            )

            db.commit()

            raise RefreshTokenError(
                "Refresh authentication failed."
            )

        # ----------------------------------------------------
        # G. Generate replacement tokens
        # ----------------------------------------------------

        new_access_token = create_access_token(
            user_id=user.id,
        )

        new_refresh_token = create_refresh_token()

        new_refresh_token_hash = hash_refresh_token(
            new_refresh_token
        )

        new_expiration = get_refresh_token_expiration()

        # ----------------------------------------------------
        # H. Create replacement session
        # ----------------------------------------------------

        replacement_session = UserSession(
            user_id=user.id,
            refresh_token_hash=new_refresh_token_hash,
            expires_at=new_expiration,
        )

        db.add(replacement_session)

        # Obtain replacement session UUID before linking.
        db.flush()

        # ----------------------------------------------------
        # I. Revoke and link previous session
        # ----------------------------------------------------

        user_session.revoked_at = now

        user_session.last_used_at = now

        user_session.replaced_by_session_id = (
            replacement_session.id
        )

        _add_refresh_audit(
            db,
            actor_user_id=user.id,
            action="refresh",
            outcome="success",
            reason="token_rotated",
            session_id=user_session.id,
        )

        # ----------------------------------------------------
        # J. Commit rotation transaction
        # ----------------------------------------------------

        db.commit()

        # ----------------------------------------------------
        # K. Return new authentication credentials
        # ----------------------------------------------------

        return RefreshResult(
            user_id=user.id,
            role=user.role,
            access_token=new_access_token,
            refresh_token=new_refresh_token,
        )

    except Exception:
        db.rollback()
        raise

# ============================================================
# 6. REVOKE A SINGLE USER SESSION (LOGOUT)
# ============================================================

def revoke_user_session(
    db: Session,
    refresh_token: str,
) -> bool:
    """
    Revoke only the session associated with the supplied token.

    Returns:
        True:
            An active session was successfully revoked.

        False:
            The token is invalid, unknown, or already revoked.

    Security:
        - Never stores or logs plaintext refresh tokens.
        - Does not revoke sessions on other devices.
        - Does not mark a normal logout as token rotation.
        - Does not trigger the refresh-token replay mechanism.
    """

    if not isinstance(refresh_token, str) or not refresh_token:
        return False

    now = datetime.now(timezone.utc)

    try:
        supplied_hash = hash_refresh_token(refresh_token)

        statement = (
            select(UserSession)
            .where(
                UserSession.refresh_token_hash == supplied_hash
            )
            .with_for_update()
        )

        user_session = db.execute(
            statement
        ).scalar_one_or_none()

        if user_session is None:
            db.rollback()
            return False

        if not verify_refresh_token(
            refresh_token,
            user_session.refresh_token_hash,
        ):
            db.rollback()
            return False

        # An already-revoked session remains unchanged.
        if user_session.revoked_at is not None:
            db.rollback()
            return False

        # Revoke only this particular session.
        user_session.revoked_at = now

        _add_refresh_audit(
            db,
            actor_user_id=user_session.user_id,
            action="logout",
            outcome="success",
            reason="session_revoked",
            session_id=user_session.id,
        )

        # Do not set replaced_by_session_id:
        # logout is not refresh-token rotation.
        db.commit()

        return True

    except Exception:
        db.rollback()
        raise


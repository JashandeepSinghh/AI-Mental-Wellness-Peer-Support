"""
User Login Authentication Service
==================================

Responsibilities:
- Authenticate users using normalized email and password.
- Enforce account verification and active account status.
- Track failed login attempts.
- Apply temporary account lockouts.
- Upgrade outdated password hashes.
- Generate access and refresh tokens.
- Create persistent authenticated sessions.
- Record privacy-safe authentication audit events.

Security:
- Never store plaintext passwords.
- Never store plaintext refresh tokens.
- Never store submitted email addresses in audit metadata.
- Return generic authentication failures.
- Roll back unsuccessful database transactions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.auth.password import verify_and_update_password
from backend.app.auth.tokens import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    create_refresh_token,
    get_refresh_token_expiration,
    hash_refresh_token,
)
from backend.app.database.models.user import User
from backend.app.database.models.user_credential import UserCredential
from backend.app.database.models.user_session import UserSession
from backend.app.database.models.audit_log import AuditLog


# ============================================================
# 1. LOGIN SECURITY CONFIGURATION
# ============================================================

MAX_FAILED_LOGIN_ATTEMPTS = 5

LOCKOUT_DURATION = timedelta(minutes=15)


# ============================================================
# 2. LOGIN EXCEPTION
# ============================================================

class LoginAuthenticationError(Exception):
    """Raised when authentication cannot be completed."""


# ============================================================
# 3. LOGIN RESULT
# ============================================================

@dataclass(frozen=True)
class LoginResult:
    """Contains tokens and basic information for a session."""

    user_id: UUID
    role: str

    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)

    token_type: str = "bearer"

    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60


# ============================================================
# 4. UTC DATETIME NORMALIZATION
# ============================================================

def _as_utc(value: datetime) -> datetime:
    """
    Normalize a database datetime to timezone-aware UTC.

    SQLite may return naive datetime values even when the
    SQLAlchemy column uses timezone=True.
    """

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


# ============================================================
# 5. LOGIN AUDIT HELPER
# ============================================================

def _add_login_audit(
    db: Session,
    actor_user_id: UUID | None,
    outcome: str,
    reason: str,
) -> None:
    """
    Add a privacy-safe login audit event to the current transaction.

    No email address, password, or token is stored.
    The caller controls when the transaction is committed.
    """

    audit_event = AuditLog(
        actor_user_id=actor_user_id,
        category="authentication",
        action="login",
        outcome=outcome,
        target_type="user" if actor_user_id is not None else None,
        target_id=(
            str(actor_user_id)
            if actor_user_id is not None
            else None
        ),
        metadata_json={
            "reason": reason,
        },
    )

    db.add(audit_event)


# ============================================================
# 6. LOGIN AUTHENTICATION
# ============================================================

def authenticate_user(
    db: Session,
    email: str,
    password: str,
) -> LoginResult:
    """
    Authenticate a user and create an authenticated session.

    Raises:
        LoginAuthenticationError:
            If the credentials are invalid, the account is
            unavailable, or the account is temporarily locked.

    Successful login, session creation, and its audit event
    are committed in one transaction.

    Failed-login counter updates and their audit events are
    also committed together.
    """

    # --------------------------------------------------------
    # A. Validate and normalize input
    # --------------------------------------------------------

    if not isinstance(email, str) or not isinstance(password, str):
        raise LoginAuthenticationError(
            "Authentication failed."
        )

    normalized_email = email.strip().lower()

    if not normalized_email or not password:
        raise LoginAuthenticationError(
            "Authentication failed."
        )

    now = datetime.now(timezone.utc)

    try:

        # ----------------------------------------------------
        # B. Retrieve credentials and associated user
        # ----------------------------------------------------

        statement = (
            select(UserCredential, User)
            .join(User, User.id == UserCredential.user_id)
            .where(UserCredential.email == normalized_email)
        )

        account_result = db.execute(
            statement
        ).one_or_none()

        if account_result is None:

            # Record an anonymous failed login.
            # Do not store the submitted email address.
            _add_login_audit(
                db=db,
                actor_user_id=None,
                outcome="failure",
                reason="invalid_credentials",
            )

            db.commit()

            raise LoginAuthenticationError(
                "Authentication failed."
            )

        credential, user = account_result

        # ----------------------------------------------------
        # C. Check temporary account lockout
        # ----------------------------------------------------

        if credential.locked_until is not None:

            locked_until = _as_utc(
                credential.locked_until
            )

            if locked_until > now:

                _add_login_audit(
                    db=db,
                    actor_user_id=user.id,
                    outcome="denied",
                    reason="account_locked",
                )

                db.commit()

                raise LoginAuthenticationError(
                    "Authentication failed."
                )

            # Previous lockout has expired.
            credential.locked_until = None
            credential.failed_login_attempts = 0

        # ----------------------------------------------------
        # D. Verify submitted password
        # ----------------------------------------------------

        password_valid, upgraded_hash = (
            verify_and_update_password(
                password,
                credential.password_hash,
            )
        )

        if not password_valid:

            credential.failed_login_attempts += 1

            # Apply temporary lockout after repeated failures.
            if (
                credential.failed_login_attempts
                >= MAX_FAILED_LOGIN_ATTEMPTS
            ):
                credential.locked_until = (
                    now + LOCKOUT_DURATION
                )

            _add_login_audit(
                db=db,
                actor_user_id=user.id,
                outcome="failure",
                reason="invalid_credentials",
            )

            db.commit()

            raise LoginAuthenticationError(
                "Authentication failed."
            )

        # ----------------------------------------------------
        # E. Check account eligibility
        # ----------------------------------------------------

        if (
            not credential.email_verified
            or user.account_status != "active"
        ):

            _add_login_audit(
                db=db,
                actor_user_id=user.id,
                outcome="denied",
                reason="account_unavailable",
            )

            db.commit()

            raise LoginAuthenticationError(
                "Authentication failed."
            )

        # ----------------------------------------------------
        # F. Upgrade password hash when necessary
        # ----------------------------------------------------

        if upgraded_hash is not None:
            credential.password_hash = upgraded_hash

        # ----------------------------------------------------
        # G. Reset failed-login protection
        # ----------------------------------------------------

        credential.failed_login_attempts = 0
        credential.locked_until = None

        # ----------------------------------------------------
        # H. Generate authentication tokens
        # ----------------------------------------------------

        access_token = create_access_token(
            user_id=user.id
        )

        refresh_token = create_refresh_token()

        refresh_token_hash = hash_refresh_token(
            refresh_token
        )

        refresh_expiration = (
            get_refresh_token_expiration()
        )

        # ----------------------------------------------------
        # I. Create persistent user session
        # ----------------------------------------------------

        user_session = UserSession(
            user_id=user.id,
            refresh_token_hash=refresh_token_hash,
            expires_at=refresh_expiration,
        )

        db.add(user_session)

        # ----------------------------------------------------
        # J. Update last successful login timestamp
        # ----------------------------------------------------

        credential.last_login_at = now

        # ----------------------------------------------------
        # K. Add successful login audit event
        # ----------------------------------------------------

        _add_login_audit(
            db=db,
            actor_user_id=user.id,
            outcome="success",
            reason="authenticated",
        )

        # ----------------------------------------------------
        # L. Commit authentication transaction
        # ----------------------------------------------------

        db.commit()

        # ----------------------------------------------------
        # M. Return authentication result
        # ----------------------------------------------------

        return LoginResult(
            user_id=user.id,
            role=user.role,
            access_token=access_token,
            refresh_token=refresh_token,
        )

    except Exception:
        db.rollback()
        raise

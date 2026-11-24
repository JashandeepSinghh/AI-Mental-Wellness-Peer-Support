"""
Resend Email Verification Service
==================================

Responsibilities:
- Locate pending student accounts.
- Enforce a resend cooldown.
- Invalidate previous outstanding verification tokens.
- Issue new verification tokens.
- Commit token changes transactionally.

Security:
- Never store raw verification tokens.
- Never expose whether an email address is registered.
- Never return tokens for already verified accounts.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.auth.verification_service import (
    get_current_utc_time,
    issue_verification_token,
)
from backend.app.database.models.email_verification_token import (
    EmailVerificationToken,
)
from backend.app.database.models.user import User
from backend.app.database.models.user_credential import UserCredential
from backend.app.database.models.audit_log import AuditLog
from backend.app.database.models.audit_log import AuditLog
from backend.app.services.registration_service import normalize_email


# ============================================================
# RESEND VERIFICATION AUDIT HELPER
# ============================================================

def _add_resend_audit(
    db: Session,
    actor_user_id,
    outcome: str,
    reason: str,
) -> None:
    """Add a non-sensitive resend-verification audit event."""

    audit_event = AuditLog(
        actor_user_id=actor_user_id,
        category="account",
        action="email_verification_resend",
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
# RESEND VERIFICATION AUDIT HELPER
# ============================================================

def _add_resend_audit(
    db: Session,
    actor_user_id,
    outcome: str,
    reason: str,
) -> None:
    """Add a non-sensitive resend-verification audit event."""

    audit_event = AuditLog(
        actor_user_id=actor_user_id,
        category="account",
        action="email_verification_resend",
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
# 1. CONFIGURATION
# ============================================================

DEFAULT_RESEND_COOLDOWN_SECONDS = 60


# ============================================================
# 2. RESULT MODEL
# ============================================================

@dataclass(frozen=True)
class ResendVerificationResult:
    """Contains the newly issued verification token."""

    email: str
    raw_token: str = field(repr=False)
    expires_at: datetime


# ============================================================
# 3. TIMESTAMP NORMALIZATION
# ============================================================

def normalize_utc_datetime(value: datetime) -> datetime:
    """
    Ensure database timestamps are timezone-aware UTC values.

    SQLite may return naive datetime objects even when
    timezone-aware values were originally stored.
    """

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


# ============================================================
# 4. RESEND VERIFICATION SERVICE
# ============================================================

def resend_verification_token(
    db: Session,
    email: str,
    cooldown_seconds: int = DEFAULT_RESEND_COOLDOWN_SECONDS,
) -> ResendVerificationResult | None:
    """
    Issue a new verification token for an eligible account.

    Returns:
        ResendVerificationResult:
            When an eligible account can receive a new token.

        None:
            When the account does not exist, is already verified,
            is not pending verification, or is within cooldown.

    The database transaction is committed before returning
    a newly issued token.
    """

    if (
        isinstance(cooldown_seconds, bool)
        or not isinstance(cooldown_seconds, int)
        or cooldown_seconds < 1
    ):
        raise ValueError(
            "Cooldown must be a positive integer."
        )

    normalized_email = normalize_email(email)

    now = get_current_utc_time()

    # --------------------------------------------------------
    # A. Find an eligible pending-verification account
    # --------------------------------------------------------

    statement = (
        select(UserCredential, User)
        .join(
            User,
            User.id == UserCredential.user_id,
        )
        .where(
            UserCredential.email == normalized_email,
            UserCredential.email_verified.is_(False),
            User.account_status == "pending_verification",
        )
    )

    account_result = db.execute(
        statement
    ).one_or_none()

    if account_result is None:
        db.rollback()

        try:
            _add_resend_audit(
                db=db,
                actor_user_id=None,
                outcome="info",
                reason="account_unavailable",
            )
            db.commit()

        except Exception:
            db.rollback()
            raise

        return None

    credential, user = account_result

    # --------------------------------------------------------
    # B. Retrieve the latest verification token
    # --------------------------------------------------------

    latest_token_statement = (
        select(EmailVerificationToken)
        .where(
            EmailVerificationToken.user_id == user.id,
        )
        .order_by(
            EmailVerificationToken.created_at.desc(),
        )
        .limit(1)
    )

    latest_token = db.execute(
        latest_token_statement
    ).scalar_one_or_none()

    # --------------------------------------------------------
    # C. Enforce resend cooldown
    # --------------------------------------------------------

    if latest_token is not None:

        last_created_at = normalize_utc_datetime(
            latest_token.created_at
        )

        elapsed_time = now - last_created_at

        if elapsed_time < timedelta(
            seconds=cooldown_seconds
        ):
            db.rollback()

            try:
                _add_resend_audit(
                    db=db,
                    actor_user_id=user.id,
                    outcome="info",
                    reason="resend_cooldown_active",
                )
                db.commit()

            except Exception:
                db.rollback()
                raise

            return None

    # --------------------------------------------------------
    # D. Issue a new verification token
    # --------------------------------------------------------

    try:

        issued_token = issue_verification_token(
            db=db,
            user_id=user.id,
        )

        # ----------------------------------------------------
        # E. Create success audit event
        # ----------------------------------------------------

        _add_resend_audit(
            db=db,
            actor_user_id=user.id,
            outcome="success",
            reason="token_resent",
        )

        # ----------------------------------------------------
        # F. Commit token changes and audit together
        # ----------------------------------------------------

        db.commit()

        return ResendVerificationResult(
            email=credential.email,
            raw_token=issued_token.raw_token,
            expires_at=issued_token.expires_at,
        )

    except Exception:

        db.rollback()
        raise

"""
Email Verification Service
===========================

Responsibilities:
- Consume valid email verification tokens.
- Locate the associated student account.
- Mark the email address as verified.
- Activate the pending account.
- Record email verification audit events.
- Commit verification changes and audit records transactionally.

Security:
- Tokens are validated using their SHA-256 hashes.
- Tokens can only be consumed once.
- Expired and invalidated tokens are rejected.
- Raw verification tokens are never stored in audit records.
- Account activation, token consumption, and audit creation
  share one transaction.
- Unexpected errors roll back all changes.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.auth.verification_service import (
    consume_verification_token,
    get_current_utc_time,
)

from backend.app.database.models.user import User
from backend.app.database.models.user_credential import UserCredential
from backend.app.database.models.audit_log import AuditLog


# ============================================================
# 1. EMAIL VERIFICATION AUDIT HELPER
# ============================================================

def _add_email_verification_audit(
    db: Session,
    actor_user_id,
    outcome: str,
    reason: str,
) -> None:
    """
    Add a non-sensitive email verification audit event.

    Never include raw tokens, passwords, or email addresses.
    """

    audit_event = AuditLog(
        actor_user_id=actor_user_id,
        category="account",
        action="email_verification",
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
# 2. EMAIL VERIFICATION SERVICE
# ============================================================

def verify_email_address(
    db: Session,
    raw_token: str,
) -> bool:
    """
    Verify an email address and activate its pending account.

    Returns:
        True:
            Token was valid and the pending account was activated.

        False:
            Token was invalid, expired, already used, invalidated,
            or the associated account was not eligible for activation.

    Transaction:
        Token consumption, account activation, and audit creation
        are committed together. Unexpected errors roll back changes.
    """

    try:

        # ----------------------------------------------------
        # A. Atomically consume the verification token
        # ----------------------------------------------------

        user_id = consume_verification_token(
            db=db,
            raw_token=raw_token,
        )

        # ----------------------------------------------------
        # B. Handle invalid or unusable verification token
        # ----------------------------------------------------

        if user_id is None:

            # Clear the token-validation transaction before
            # creating the anonymous audit event.
            db.rollback()

            _add_email_verification_audit(
                db=db,
                actor_user_id=None,
                outcome="failure",
                reason="invalid_or_unusable_token",
            )

            db.commit()

            return False

        # ----------------------------------------------------
        # C. Retrieve the associated user and credentials
        # ----------------------------------------------------

        statement = (
            select(UserCredential, User)
            .join(
                User,
                User.id == UserCredential.user_id,
            )
            .where(
                User.id == user_id,
            )
        )

        account_result = db.execute(
            statement
        ).one_or_none()

        # ----------------------------------------------------
        # D. Handle unavailable account
        # ----------------------------------------------------

        if account_result is None:

            # The token has been consumed, but its associated
            # account is no longer available.
            _add_email_verification_audit(
                db=db,
                actor_user_id=None,
                outcome="denied",
                reason="account_unavailable",
            )

            db.commit()

            return False

        credential, user = account_result

        # ----------------------------------------------------
        # E. Check account eligibility
        # ----------------------------------------------------

        if (
            credential.email_verified
            or user.account_status != "pending_verification"
        ):

            # The token is consumed, but an already verified
            # or otherwise ineligible account is not activated.
            _add_email_verification_audit(
                db=db,
                actor_user_id=user.id,
                outcome="denied",
                reason="account_not_eligible",
            )

            db.commit()

            return False

        # ----------------------------------------------------
        # F. Mark email as verified
        # ----------------------------------------------------

        now = get_current_utc_time()

        credential.email_verified = True
        credential.email_verified_at = now

        # ----------------------------------------------------
        # G. Activate the student account
        # ----------------------------------------------------

        user.account_status = "active"

        # ----------------------------------------------------
        # H. Create successful verification audit event
        # ----------------------------------------------------

        _add_email_verification_audit(
            db=db,
            actor_user_id=user.id,
            outcome="success",
            reason="email_verified",
        )

        # ----------------------------------------------------
        # I. Commit token consumption, account activation,
        #    and audit event in one transaction
        # ----------------------------------------------------

        db.commit()

        return True

    except Exception:

        # Roll back token consumption, account changes,
        # and audit creation if any operation fails.
        db.rollback()

        raise

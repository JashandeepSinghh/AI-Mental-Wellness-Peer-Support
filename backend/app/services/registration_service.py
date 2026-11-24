"""
Student Registration Service
=============================

Responsibilities:
- Normalize registration email addresses.
- Hash passwords using the existing Argon2id utility.
- Create student accounts and credentials.
- Issue email verification tokens.
- Commit registration and audit records transactionally.

Security:
- Plain-text passwords are never stored.
- Raw verification tokens are never stored in the database.
- User roles and account statuses are assigned by the server.
- Sensitive token values are excluded from object representations.
- Audit metadata excludes emails, passwords, and tokens.
- Database failures trigger transaction rollback.
"""

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.auth.password import hash_password
from backend.app.auth.verification_service import (
    issue_verification_token,
)
from backend.app.database.models.user import User
from backend.app.database.models.user_credential import UserCredential
from backend.app.database.models.audit_log import AuditLog


# ============================================================
# 1. REGISTRATION EXCEPTIONS
# ============================================================

class RegistrationConflictError(Exception):
    """Raised when the registration email already exists."""


# ============================================================
# 2. REGISTRATION RESULT
# ============================================================

@dataclass(frozen=True)
class RegistrationResult:
    """Contains information required by the API layer."""

    user_id: UUID
    email: str
    raw_token: str = field(repr=False)
    expires_at: datetime


# ============================================================
# 3. EMAIL NORMALIZATION
# ============================================================

def normalize_email(email: str) -> str:
    """Normalize an already validated email address."""

    if not isinstance(email, str):
        raise TypeError("Email must be a string.")

    normalized_email = email.strip().lower()

    if not normalized_email:
        raise ValueError("Email cannot be empty.")

    if len(normalized_email) > 254:
        raise ValueError("Email address is too long.")

    return normalized_email


# ============================================================
# 4. REGISTRATION AUDIT HELPER
# ============================================================

def _add_registration_audit(
    db: Session,
    actor_user_id: UUID | None,
    outcome: str,
    reason: str,
) -> None:
    """
    Add a registration audit event to the current transaction.

    No email, password, password hash, or raw token is recorded.
    """

    audit_event = AuditLog(
        actor_user_id=actor_user_id,
        category="account",
        action="student_registration",
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
# 5. CREATE STUDENT REGISTRATION
# ============================================================

def register_student(
    db: Session,
    email: str,
    password: str,
) -> RegistrationResult:
    """
    Create a student account and issue a verification token.

    Successful registration and its audit event are committed
    in the same database transaction.

    Email delivery must happen after this function succeeds.
    """

    if not isinstance(password, str):
        raise TypeError("Password must be a string.")

    if not password:
        raise ValueError("Password cannot be empty.")

    normalized_email = normalize_email(email)

    # --------------------------------------------------------
    # A. Check whether email already exists
    # --------------------------------------------------------

    existing_credential = db.execute(
        select(UserCredential.id).where(
            UserCredential.email == normalized_email
        )
    ).scalar_one_or_none()

    if existing_credential is not None:

        try:
            # Record the attempt without identifying the account.
            _add_registration_audit(
                db=db,
                actor_user_id=None,
                outcome="failure",
                reason="duplicate_email",
            )

            db.commit()

        except Exception:
            db.rollback()
            raise

        raise RegistrationConflictError(
            "Registration could not be completed."
        )

    # --------------------------------------------------------
    # B. Generate password hash
    # --------------------------------------------------------

    password_hash = hash_password(password)

    try:

        # ----------------------------------------------------
        # C. Create student account
        # ----------------------------------------------------

        user = User()

        # Role and account status use server-side defaults.

        db.add(user)
        db.flush()

        # ----------------------------------------------------
        # D. Create authentication credentials
        # ----------------------------------------------------

        credential = UserCredential(
            user_id=user.id,
            email=normalized_email,
            password_hash=password_hash,
        )

        db.add(credential)
        db.flush()

        # ----------------------------------------------------
        # E. Issue email verification token
        # ----------------------------------------------------

        issued_token = issue_verification_token(
            db=db,
            user_id=user.id,
        )

        # ----------------------------------------------------
        # F. Add successful registration audit event
        # ----------------------------------------------------

        _add_registration_audit(
            db=db,
            actor_user_id=user.id,
            outcome="success",
            reason="student_registered",
        )

        # ----------------------------------------------------
        # G. Commit registration and audit records together
        # ----------------------------------------------------

        db.commit()

        # ----------------------------------------------------
        # H. Return registration result
        # ----------------------------------------------------

        return RegistrationResult(
            user_id=user.id,
            email=normalized_email,
            raw_token=issued_token.raw_token,
            expires_at=issued_token.expires_at,
        )

    except IntegrityError:

        # ----------------------------------------------------
        # I. Handle database integrity conflicts
        # ----------------------------------------------------

        db.rollback()

        existing_credential = db.execute(
            select(UserCredential.id).where(
                UserCredential.email == normalized_email
            )
        ).scalar_one_or_none()

        if existing_credential is not None:

            try:
                _add_registration_audit(
                    db=db,
                    actor_user_id=None,
                    outcome="failure",
                    reason="duplicate_email",
                )

                db.commit()

            except Exception:
                db.rollback()
                raise

            raise RegistrationConflictError(
                "Registration could not be completed."
            ) from None

        raise

    except Exception:

        # ----------------------------------------------------
        # J. Roll back unsuccessful registration
        # ----------------------------------------------------

        db.rollback()
        raise

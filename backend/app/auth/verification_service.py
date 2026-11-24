# Email Verification Database Service

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import update
from sqlalchemy.orm import Session

from backend.app.auth.verification import (
    generate_verification_token,
    hash_verification_token,
)
from backend.app.core.config import get_settings
from backend.app.database.models.email_verification_token import (
    EmailVerificationToken,
)


@dataclass(frozen=True)
class IssuedVerificationToken:
    raw_token: str
    expires_at: datetime


def get_current_utc_time() -> datetime:
    return datetime.now(timezone.utc)


def issue_verification_token(
    db: Session,
    user_id: UUID,
) -> IssuedVerificationToken:

    settings = get_settings()
    now = get_current_utc_time()

    expires_at = now + timedelta(
        minutes=settings.EMAIL_VERIFICATION_EXPIRE_MINUTES
    )

    # Invalidate previous unused tokens.
    db.execute(
        update(EmailVerificationToken)
        .where(
            EmailVerificationToken.user_id == user_id,
            EmailVerificationToken.used_at.is_(None),
            EmailVerificationToken.invalidated_at.is_(None),
        )
        .values(invalidated_at=now)
    )

    # Generate a secure token.
    raw_token = generate_verification_token()

    # Store only its SHA-256 hash.
    token_hash = hash_verification_token(raw_token)

    verification_record = EmailVerificationToken(
        user_id=user_id,
        token_hash=token_hash,
        expires_at=expires_at,
        created_at=now,
    )

    db.add(verification_record)
    db.flush()

    return IssuedVerificationToken(
        raw_token=raw_token,
        expires_at=expires_at,
    )


def consume_verification_token(
    db: Session,
    raw_token: str,
) -> UUID | None:

    try:
        token_hash = hash_verification_token(raw_token)
    except (TypeError, ValueError):
        return None

    now = get_current_utc_time()

    statement = (
        update(EmailVerificationToken)
        .where(
            EmailVerificationToken.token_hash == token_hash,
            EmailVerificationToken.used_at.is_(None),
            EmailVerificationToken.invalidated_at.is_(None),
            EmailVerificationToken.expires_at > now,
        )
        .values(used_at=now)
        .returning(EmailVerificationToken.user_id)
    )

    result = db.execute(statement)

    return result.scalar_one_or_none()

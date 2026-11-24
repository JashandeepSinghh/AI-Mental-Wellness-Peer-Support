
"""
User Session Model
==================

Manages authenticated sessions for registered users.

Security design:
- Refresh tokens are stored as hashes, never plaintext.
- Sessions have explicit expiration timestamps.
- Revoked sessions cannot be reused.
- Token rotation is tracked through session records.
- IP addresses and browser fingerprints are not collected here.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database.base import Base


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


class UserSession(Base):
    """Represents an authenticated user session."""

    __tablename__ = "user_sessions"

    __table_args__ = (
        CheckConstraint(
            "length(refresh_token_hash) >= 32",
            name="ck_user_sessions_token_hash_length",
        ),
        CheckConstraint(
            "expires_at > created_at",
            name="ck_user_sessions_expiration_after_creation",
        ),
        CheckConstraint(
            "revoked_at IS NULL OR revoked_at >= created_at",
            name="ck_user_sessions_valid_revocation_time",
        ),
        Index(
            "ix_user_sessions_user_id",
            "user_id",
        ),
        Index(
            "ix_user_sessions_expires_at",
            "expires_at",
        ),
    )

    # Unique identifier for the session.
    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # Account associated with this authenticated session.
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Cryptographic hash of the refresh token.
    refresh_token_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
    )

    # Expiration timestamp of the session.
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    # Timestamp when the session was created.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=func.current_timestamp(),
    )

    # Timestamp of the latest successful session activity.
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # NULL means active; a timestamp means revoked.
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Identifies the replacement session after token rotation.
    replaced_by_session_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("user_sessions.id", ondelete="SET NULL"),
        nullable=True,
    )

    def __repr__(self) -> str:
        # Never expose the refresh-token hash in object representations.
        return (
            f"<UserSession("
            f"id={self.id}, "
            f"user_id={self.user_id}, "
            f"expires_at={self.expires_at}, "
            f"revoked={self.revoked_at is not None}"
            f")>"
        )

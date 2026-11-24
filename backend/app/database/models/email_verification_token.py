
"""
Email Verification Token Database Model
=========================================

Stores hashed, time-limited email verification tokens.

Security:
- Raw verification tokens are never stored.
- Token hashes are unique.
- Tokens expire after a defined period.
- Tokens can only be used once.
- Previous outstanding tokens can be invalidated.
- Used and invalidated states cannot overlap.
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
)
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database.base import Base


# ============================================================
# 1. UTC TIMESTAMP HELPER
# ============================================================

def utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


# ============================================================
# 2. EMAIL VERIFICATION TOKEN MODEL
# ============================================================

class EmailVerificationToken(Base):
    """Represents a secure email verification token record."""

    __tablename__ = "email_verification_tokens"

    # --------------------------------------------------------
    # DATABASE CONSTRAINTS AND INDEXES
    # --------------------------------------------------------

    __table_args__ = (

        CheckConstraint(
            "length(token_hash) = 64",
            name="ck_email_verification_token_hash_length",
        ),

        CheckConstraint(
            "expires_at > created_at",
            name="ck_email_verification_expiration",
        ),

        CheckConstraint(
            "used_at IS NULL OR used_at >= created_at",
            name="ck_email_verification_used_timestamp",
        ),

        CheckConstraint(
            "invalidated_at IS NULL OR invalidated_at >= created_at",
            name="ck_email_verification_invalidated_timestamp",
        ),

        CheckConstraint(
            "NOT (used_at IS NOT NULL AND invalidated_at IS NOT NULL)",
            name="ck_email_verification_terminal_state",
        ),

        Index(
            "ix_email_verification_user_id",
            "user_id",
        ),

        Index(
            "ix_email_verification_expires_at",
            "expires_at",
        ),
    )

    # --------------------------------------------------------
    # PRIMARY KEY
    # --------------------------------------------------------

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )

    # --------------------------------------------------------
    # ASSOCIATED USER
    # --------------------------------------------------------

    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    # --------------------------------------------------------
    # TOKEN HASH
    # --------------------------------------------------------

    token_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
    )

    # --------------------------------------------------------
    # TOKEN EXPIRATION
    # --------------------------------------------------------

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    # --------------------------------------------------------
    # TOKEN USAGE
    # --------------------------------------------------------

    used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )

    # --------------------------------------------------------
    # TOKEN INVALIDATION
    # --------------------------------------------------------

    invalidated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )

    # --------------------------------------------------------
    # CREATION TIMESTAMP
    # --------------------------------------------------------

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    # --------------------------------------------------------
    # OBJECT REPRESENTATION
    # --------------------------------------------------------

    def __repr__(self) -> str:
        """Return a representation without exposing token data."""

        return (
            f"<EmailVerificationToken("
            f"id={self.id}, "
            f"user_id={self.user_id}, "
            f"expires_at={self.expires_at}, "
            f"used={self.used_at is not None}, "
            f"invalidated={self.invalidated_at is not None}"
            f")>"
        )

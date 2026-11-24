
"""
User Credential Database Model
================================

Stores authentication credentials separately from user identity.

Security:
- Passwords are stored only as Argon2id hashes.
- Email addresses are normalized before storage.
- Email addresses are unique.
- Failed authentication attempts are tracked.
- Temporary account lockouts are supported.
- Email verification metadata must remain consistent.
"""

import uuid

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database.base import Base


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


class UserCredential(Base):

    __tablename__ = "user_credentials"

    __table_args__ = (
        CheckConstraint(
            "failed_login_attempts >= 0",
            name="ck_credentials_failed_attempts",
        ),
        CheckConstraint(
            "(email_verified = 1 AND email_verified_at IS NOT NULL) "
            "OR "
            "(email_verified = 0 AND email_verified_at IS NULL)",
            name="ck_credentials_email_verification_consistency",
        ),
    )

    # --------------------------------------------------------
    # Primary key
    # --------------------------------------------------------

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )

    # --------------------------------------------------------
    # User relationship
    # --------------------------------------------------------

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        unique=True,
    )

    user: Mapped["User"] = relationship(
        "User",
        back_populates="credential",
    )

    # --------------------------------------------------------
    # Authentication credentials
    # --------------------------------------------------------

    email: Mapped[str] = mapped_column(
        String(254),
        nullable=False,
        unique=True,
    )

    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    # --------------------------------------------------------
    # Email verification
    # --------------------------------------------------------

    email_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="0",
    )

    email_verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )

    # --------------------------------------------------------
    # Login protection
    # --------------------------------------------------------

    failed_login_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    locked_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )

    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )

    # --------------------------------------------------------
    # Password lifecycle
    # --------------------------------------------------------

    password_changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    def __repr__(self) -> str:
        return (
            "UserCredential("
            f"id={self.id!r}, "
            f"user_id={self.user_id!r}, "
            f"email_verified={self.email_verified!r}"
            ")"
        )


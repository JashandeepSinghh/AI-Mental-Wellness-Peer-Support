
"""
User Consent Model
==================

Stores versioned, purpose-specific consent decisions.

Privacy design:
- Consent decisions are recorded as separate events.
- Previous consent records should never be overwritten.
- Conversation storage and analytics consent are independent.
- Counselling referral consent is handled separately for each referral.
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


class UserConsent(Base):
    """Append-only record of an individual user's consent decision."""

    __tablename__ = "user_consents"

    __table_args__ = (
        CheckConstraint(
            "purpose IN ('conversation_storage', 'analytics')",
            name="ck_user_consents_valid_purpose",
        ),
        CheckConstraint(
            "source IN ('web', 'mobile', 'api')",
            name="ck_user_consents_valid_source",
        ),
        CheckConstraint(
            "length(trim(policy_version)) > 0",
            name="ck_user_consents_policy_version_not_empty",
        ),
        Index(
            "ix_user_consents_user_purpose_recorded",
            "user_id",
            "purpose",
            "recorded_at",
        ),
    )

    # Unique identifier for this consent event.
    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # Student account associated with this consent decision.
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Independent consent purposes.
    purpose: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
    )

    # True = consent granted; False = consent denied or withdrawn.
    granted: Mapped[bool] = mapped_column(
        nullable=False,
    )

    # Version of the privacy/consent policy shown to the user.
    policy_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    # Where the consent decision was submitted.
    source: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    # Time when the consent decision was recorded.
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=func.current_timestamp(),
    )

    def __repr__(self) -> str:
        return (
            f"<UserConsent("
            f"id={self.id}, "
            f"user_id={self.user_id}, "
            f"purpose={self.purpose}, "
            f"granted={self.granted}"
            f")>"
        )

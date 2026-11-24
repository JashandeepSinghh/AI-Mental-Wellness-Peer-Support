
"""
Counselling Escalation Request Model
====================================

Stores individual requests for counselling-team referrals.

Privacy and safety design:
- Each referral requires explicit, request-specific student consent.
- General conversation-storage or analytics consent is insufficient.
- Only registered students can create identifiable referrals.
- Risk information is represented using structured categories.
- Raw conversation transcripts are not stored in this model.
- Referral consent records the policy version and decision timestamp.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
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


class EscalationRequest(Base):
    """Represents one counselling referral request."""

    __tablename__ = "escalation_requests"

    __table_args__ = (
        CheckConstraint(
            "risk_level IN ('low', 'moderate', 'high', 'urgent')",
            name="ck_escalation_requests_valid_risk_level",
        ),
        CheckConstraint(
            "status IN ("
            "'awaiting_consent', "
            "'consented', "
            "'submitted', "
            "'accepted', "
            "'declined', "
            "'completed', "
            "'cancelled'"
            ")",
            name="ck_escalation_requests_valid_status",
        ),
        CheckConstraint(
            "consent_granted IS NULL OR consent_granted IN (0, 1)",
            name="ck_escalation_requests_valid_consent",
        ),
        CheckConstraint(
            "("
            "consent_granted = 1 "
            "AND consent_recorded_at IS NOT NULL "
            "AND consent_policy_version IS NOT NULL"
            ") OR ("
            "consent_granted IS NULL "
            "AND consent_recorded_at IS NULL "
            "AND consent_policy_version IS NULL"
            ") OR ("
            "consent_granted = 0 "
            "AND consent_recorded_at IS NOT NULL "
            "AND consent_policy_version IS NOT NULL"
            ")",
            name="ck_escalation_requests_consent_metadata",
        ),
        CheckConstraint(
            "status NOT IN ('consented', 'submitted', 'accepted', 'completed') "
            "OR consent_granted = 1",
            name="ck_escalation_requests_consent_required",
        ),
        Index(
            "ix_escalation_requests_user_created",
            "user_id",
            "created_at",
        ),
        Index(
            "ix_escalation_requests_status_created",
            "status",
            "created_at",
        ),
        Index(
            "ix_escalation_requests_risk_level",
            "risk_level",
        ),
    )

    # Unique identifier for the escalation request.
    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # Identifiable registered student associated with the referral.
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )

    # Related conversation, when the referral originates from one.
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Structured risk classification produced by the safety subsystem.
    risk_level: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    # Structured reason category; do not put raw message text here.
    reason_code: Mapped[str] = mapped_column(
        String(60),
        nullable=False,
    )

    # Current state of the referral workflow.
    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="awaiting_consent",
        server_default="awaiting_consent",
    )

    # NULL = no decision yet; TRUE = granted; FALSE = declined.
    consent_granted: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
    )

    # Timestamp of the specific referral consent decision.
    consent_recorded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Policy version shown before the referral consent decision.
    consent_policy_version: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    # Timestamp when the referral request was created.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=func.current_timestamp(),
    )

    # Timestamp of the latest referral update.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
        server_default=func.current_timestamp(),
    )

    # Timestamp when the referral workflow was closed.
    closed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    def __repr__(self) -> str:
        # Deliberately excludes personal and conversation content.
        return (
            f"<EscalationRequest("
            f"id={self.id}, "
            f"user_id={self.user_id}, "
            f"risk_level={self.risk_level}, "
            f"status={self.status}"
            f")>"
        )


"""
Audit Log Model
===============

Records security, privacy, authentication, and administrative events.

Privacy design:
- Never store passwords, tokens, message content, or raw transcripts.
- Actor references become NULL if the associated account is deleted.
- Store only structured, non-sensitive event metadata.
- Audit records should be append-only at the application level.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    JSON,
    String,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database.base import Base


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


class AuditLog(Base):
    """Represents one security or administrative audit event."""

    __tablename__ = "audit_logs"

    __table_args__ = (
        CheckConstraint(
            "category IN ("
            "'authentication', "
            "'authorization', "
            "'account', "
            "'consent', "
            "'conversation', "
            "'escalation', "
            "'administration', "
            "'security'"
            ")",
            name="ck_audit_logs_valid_category",
        ),
        CheckConstraint(
            "outcome IN ('success', 'failure', 'denied', 'info')",
            name="ck_audit_logs_valid_outcome",
        ),
        CheckConstraint(
            "length(trim(action)) > 0",
            name="ck_audit_logs_action_not_empty",
        ),
        Index(
            "ix_audit_logs_actor_created",
            "actor_user_id",
            "created_at",
        ),
        Index(
            "ix_audit_logs_category_created",
            "category",
            "created_at",
        ),
        Index(
            "ix_audit_logs_target",
            "target_type",
            "target_id",
        ),
    )

    # Unique identifier for the audit event.
    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # Account responsible for the action.
    # NULL can represent anonymous or system-generated events.
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Broad classification of the event.
    category: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )

    # Specific action, for example "login" or "consent_withdrawn".
    action: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    # Result of the attempted action.
    outcome: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    # Type of resource affected, if applicable.
    target_type: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    # Identifier of the affected resource.
    # Stored as text to support UUIDs and other resource identifiers.
    target_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    # Whitelisted, non-sensitive structured event information.
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    # Timestamp when the audit event was recorded.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=func.current_timestamp(),
    )

    def __repr__(self) -> str:
        # Deliberately excludes metadata contents.
        return (
            f"<AuditLog("
            f"id={self.id}, "
            f"category={self.category}, "
            f"action={self.action}, "
            f"outcome={self.outcome}"
            f")>"
        )

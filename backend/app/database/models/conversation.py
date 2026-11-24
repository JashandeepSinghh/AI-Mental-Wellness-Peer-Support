
"""
Conversation Model
==================

Stores conversation metadata for anonymous and registered users.

Privacy design:
- Anonymous conversations do not require a user account.
- Registered conversations can be associated with a student account.
- Message content is stored separately.
- Conversation persistence must respect the applicable storage consent.
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


class Conversation(Base):
    """Represents one anonymous or registered-user conversation."""

    __tablename__ = "conversations"

    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'closed', 'archived')",
            name="ck_conversations_valid_status",
        ),
        Index(
            "ix_conversations_user_created",
            "user_id",
            "created_at",
        ),
        Index(
            "ix_conversations_status",
            "status",
        ),
    )

    # Unique identifier for the conversation.
    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # Null for anonymous conversations.
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Lifecycle state of the conversation.
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="active",
        server_default="active",
    )

    # Timestamp when the conversation was created.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=func.current_timestamp(),
    )

    # Timestamp of the latest conversation activity.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
        server_default=func.current_timestamp(),
    )

    # Timestamp when the conversation was closed or archived.
    closed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    def __repr__(self) -> str:
        return (
            f"<Conversation("
            f"id={self.id}, "
            f"user_id={self.user_id}, "
            f"status={self.status}"
            f")>"
        )
  
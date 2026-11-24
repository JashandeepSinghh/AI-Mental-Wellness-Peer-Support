
"""
Message Model
=============

Stores individual messages belonging to a conversation.

Privacy design:
- Messages are associated with a conversation, not directly with a user.
- Message persistence must respect the applicable conversation-storage consent.
- Raw message content must never be written to application logs.
- Sequence numbers preserve message order within a conversation.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database.base import Base


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


class Message(Base):
    """Represents one message within a conversation."""

    __tablename__ = "messages"

    __table_args__ = (
        CheckConstraint(
            "sender IN ('student', 'assistant', 'system')",
            name="ck_messages_valid_sender",
        ),
        CheckConstraint(
            "message_type IN ('text', 'resource', 'system_event')",
            name="ck_messages_valid_type",
        ),
        CheckConstraint(
            "sequence_number >= 1",
            name="ck_messages_positive_sequence",
        ),
        CheckConstraint(
            "length(trim(content)) > 0",
            name="ck_messages_content_not_empty",
        ),
        UniqueConstraint(
            "conversation_id",
            "sequence_number",
            name="uq_messages_conversation_sequence",
        ),
        Index(
            "ix_messages_conversation_created",
            "conversation_id",
            "created_at",
        ),
    )

    # Unique identifier for this message.
    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # Conversation to which this message belongs.
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Identifies who sent the message.
    sender: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    # Classification of the message.
    message_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="text",
        server_default="text",
    )

    # Position of the message within its conversation.
    sequence_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    # Message content.
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # Timestamp when the message was created.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=func.current_timestamp(),
    )

    def __repr__(self) -> str:
        # Deliberately excludes message content to protect privacy.
        return (
            f"<Message("
            f"id={self.id}, "
            f"conversation_id={self.conversation_id}, "
            f"sender={self.sender}, "
            f"sequence_number={self.sequence_number}"
            f")>"
        )

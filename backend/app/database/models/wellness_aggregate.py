
"""
Wellness Aggregate Model
========================

Stores aggregated, non-individual wellness statistics.

Privacy design:
- No student identifiers.
- No conversation identifiers.
- No raw message content.
- Analytics collection requires the appropriate user consent.
- Minimum group-size thresholds must be enforced before publication.
- This table stores aggregate counts, not individual predictions.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Index,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database.base import Base


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


class WellnessAggregate(Base):
    """Stores aggregate wellness statistics for a reporting period."""

    __tablename__ = "wellness_aggregates"

    __table_args__ = (
        CheckConstraint(
            "period_type IN ('daily', 'weekly', 'monthly')",
            name="ck_wellness_aggregates_valid_period",
        ),
        CheckConstraint(
            "period_end > period_start",
            name="ck_wellness_aggregates_valid_date_range",
        ),
        CheckConstraint(
            "participant_count >= 0 "
            "AND total_conversations >= 0 "
            "AND total_messages >= 0 "
            "AND positive_count >= 0 "
            "AND neutral_count >= 0 "
            "AND negative_count >= 0 "
            "AND low_stress_count >= 0 "
            "AND moderate_stress_count >= 0 "
            "AND high_stress_count >= 0",
            name="ck_wellness_aggregates_non_negative_counts",
        ),
        UniqueConstraint(
            "period_type",
            "period_start",
            "period_end",
            name="uq_wellness_aggregates_reporting_period",
        ),
        Index(
            "ix_wellness_aggregates_period_start",
            "period_start",
        ),
    )

    # Unique identifier for the aggregate record.
    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # Reporting period granularity.
    period_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    # Beginning of the reporting period (UTC).
    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    # End of the reporting period (UTC).
    period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    # Number of distinct consenting participants represented.
    participant_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    # Aggregate usage statistics.
    total_conversations: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    total_messages: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    # Aggregate sentiment classification counts.
    positive_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    neutral_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    negative_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    # Aggregate stress-category counts.
    low_stress_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    moderate_stress_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    high_stress_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    # Timestamp when this aggregate record was created.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=func.current_timestamp(),
    )

    # Timestamp of the latest aggregate update.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
        server_default=func.current_timestamp(),
    )

    def __repr__(self) -> str:
        # Contains only aggregate reporting information.
        return (
            f"<WellnessAggregate("
            f"id={self.id}, "
            f"period_type={self.period_type}, "
            f"participant_count={self.participant_count}"
            f")>"
        )

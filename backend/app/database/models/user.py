
"""
User Database Model
====================

Stores pseudonymous user identity and account lifecycle information.

Authentication credentials are stored separately.
"""

from __future__ import annotations

import uuid

from datetime import datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Index,
    String,
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from backend.app.database.base import Base


# ============================================================
# 1. UTC TIMESTAMP HELPER
# ============================================================

def utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


# ============================================================
# 2. USER DATABASE MODEL
# ============================================================

class User(Base):
    """Represents a pseudonymous application user."""

    __tablename__ = "users"

    # --------------------------------------------------------
    # DATABASE CONSTRAINTS AND INDEXES
    # --------------------------------------------------------

    __table_args__ = (
        CheckConstraint(
            "role IN ('student', 'counsellor', 'administrator')",
            name="ck_users_role",
        ),

        CheckConstraint(
            """
            account_status IN (
                'pending_verification',
                'active',
                'suspended',
                'deleted'
            )
            """,
            name="ck_users_account_status",
        ),

        CheckConstraint(
            """
            (
                account_status = 'deleted'
                AND deleted_at IS NOT NULL
            )
            OR
            (
                account_status != 'deleted'
                AND deleted_at IS NULL
            )
            """,
            name="ck_users_deletion_timestamp",
        ),

        Index(
            "ix_users_role_status",
            "role",
            "account_status",
        ),

        Index(
            "ix_users_created_at",
            "created_at",
        ),
    )

    # --------------------------------------------------------
    # PRIMARY KEY
    # --------------------------------------------------------

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )

    # --------------------------------------------------------
    # USER ROLE
    # --------------------------------------------------------

    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="student",
        server_default="student",
    )

    # --------------------------------------------------------
    # ACCOUNT STATUS
    # --------------------------------------------------------

    account_status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="pending_verification",
        server_default="pending_verification",
    )

    # --------------------------------------------------------
    # ACCOUNT TIMESTAMPS
    # --------------------------------------------------------

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

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )

    # --------------------------------------------------------
    # CREDENTIAL RELATIONSHIP
    # --------------------------------------------------------

    credential: Mapped[UserCredential | None] = relationship(
        "UserCredential",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    # --------------------------------------------------------
    # OBJECT REPRESENTATION
    # --------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"User(id={self.id!r}, "
            f"role={self.role!r}, "
            f"account_status={self.account_status!r})"
        )


"""Improve credential verification constraints.

Revision ID: 50c5f19c1b74
Revises: ef4c6eb3260d
"""

from typing import Sequence, Union

from alembic import op


# Revision identifiers.
revision: str = "50c5f19c1b74"
down_revision: Union[str, Sequence[str], None] = "ef4c6eb3260d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade credential verification schema."""

    with op.batch_alter_table("user_credentials") as batch_op:

        # Remove redundant indexes.
        batch_op.drop_index("ix_credentials_email")
        batch_op.drop_index("ix_credentials_user_id")

        # Enforce email verification consistency.
        batch_op.create_check_constraint(
            "ck_credentials_email_verification_consistency",
            """
            (email_verified = 1 AND email_verified_at IS NOT NULL)
            OR
            (email_verified = 0 AND email_verified_at IS NULL)
            """,
        )


def downgrade() -> None:
    """Restore the previous credential schema."""

    with op.batch_alter_table("user_credentials") as batch_op:

        # Remove the email verification constraint.
        batch_op.drop_constraint(
            "ck_credentials_email_verification_consistency",
            type_="check",
        )

        # Restore the original indexes.
        batch_op.create_index(
            "ix_credentials_email",
            ["email"],
            unique=False,
        )

        batch_op.create_index(
            "ix_credentials_user_id",
            ["user_id"],
            unique=False,
        )

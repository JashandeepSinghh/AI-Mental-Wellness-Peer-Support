"""
Consent Management Service
==========================

Records and retrieves versioned user consent decisions.

Privacy principles:
- Every decision creates a new database record.
- Previous consent decisions are never overwritten.
- Conversation storage and analytics are independent.
- Referral consent is managed separately.
- Every successful consent decision creates an audit event.
- Consent and audit records are committed atomically.
"""

from uuid import UUID

from sqlalchemy.orm import Session

from backend.app.database.models import (
    AuditLog,
    User,
    UserConsent,
)


VALID_PURPOSES = {
    "conversation_storage",
    "analytics",
}

CONSENT_SOURCE = "api"


def record_consent_decision(
    db: Session,
    user_id: UUID,
    purpose: str,
    granted: bool,
    policy_version: str,
) -> UserConsent:

    """Record a consent decision and its corresponding audit event."""

    # A. Validate consent purpose.
    if purpose not in VALID_PURPOSES:
        raise ValueError("Unsupported consent purpose.")

    # B. Validate decision value.
    if not isinstance(granted, bool):
        raise ValueError("Consent decision must be a boolean.")

    # C. Validate policy version.
    if not isinstance(policy_version, str):
        raise ValueError("Policy version must be a string.")

    policy_version = policy_version.strip()

    if not policy_version or len(policy_version) > 50:
        raise ValueError(
            "Policy version must contain between 1 and 50 characters."
        )

    # D. Verify that the associated user exists.
    user_exists = (
        db.query(User.id)
        .filter(User.id == user_id)
        .first()
    )

    if user_exists is None:
        raise ValueError("The specified user does not exist.")

    # E. Create a new append-only consent event.
    consent = UserConsent(
        user_id=user_id,
        purpose=purpose,
        granted=granted,
        policy_version=policy_version,
        source=CONSENT_SOURCE,
    )

    try:

        # F. Flush consent to generate its database identifier.
        # This does not commit the transaction.
        db.add(consent)
        db.flush()

        # G. Create the corresponding audit event.
        audit_event = AuditLog(
            actor_user_id=user_id,
            category="consent",
            action="consent_decision_recorded",
            outcome="success",
            target_type="user_consent",
            target_id=str(consent.id),
            metadata_json={
                "purpose": purpose,
                "granted": granted,
                "policy_version": policy_version,
            },
        )

        db.add(audit_event)

        # H. Commit both records in one transaction.
        db.commit()
        db.refresh(consent)

        return consent

    except Exception:

        # Both consent and audit operations are rolled back.
        db.rollback()
        raise


def get_current_consent_decisions(
    db: Session,
    user_id: UUID,
) -> dict[str, UserConsent]:

    """Retrieve the latest consent decision for each purpose."""

    consent_records = (
        db.query(UserConsent)
        .filter(UserConsent.user_id == user_id)
        .order_by(
            UserConsent.recorded_at.desc(),
            UserConsent.id.desc(),
        )
        .all()
    )

    current_decisions = {}

    for consent in consent_records:

        if consent.purpose not in current_decisions:
            current_decisions[consent.purpose] = consent

    return current_decisions

from uuid import uuid4

import pytest

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.database.base import Base
from backend.app.database.models import AuditLog, User, UserConsent
import backend.app.database.models

from backend.app.services.consent_service import (
    record_consent_decision,
    get_current_consent_decisions,
)


@pytest.fixture
def test_db():

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    TestingSessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    Base.metadata.create_all(bind=engine)

    db = TestingSessionLocal()

    user = User(
        role="student",
        account_status="active",
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    try:
        yield db, user

    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def test_record_granted_consent(test_db):

    db, user = test_db

    consent = record_consent_decision(
        db=db,
        user_id=user.id,
        purpose="conversation_storage",
        granted=True,
        policy_version="1.0",
    )

    assert consent.user_id == user.id
    assert consent.purpose == "conversation_storage"
    assert consent.granted is True
    assert consent.policy_version == "1.0"
    assert consent.source == "api"

    assert db.query(UserConsent).count() == 1


def test_consent_decision_creates_audit_log(test_db):

    db, user = test_db

    consent = record_consent_decision(
        db=db,
        user_id=user.id,
        purpose="analytics",
        granted=False,
        policy_version="2.0",
    )

    audit = (
        db.query(AuditLog)
        .filter_by(
            category="consent",
            action="consent_decision_recorded",
        )
        .one()
    )

    assert audit.actor_user_id == user.id
    assert audit.outcome == "success"
    assert audit.target_type == "user_consent"
    assert audit.target_id == str(consent.id)

    assert audit.metadata_json == {
        "purpose": "analytics",
        "granted": False,
        "policy_version": "2.0",
    }

    assert db.query(AuditLog).count() == 1


def test_audit_failure_rolls_back_consent(test_db):

    db, user = test_db

    def fail_audit_insert(mapper, connection, target):

        raise RuntimeError(
            "Simulated audit-log insertion failure."
        )

    event.listen(
        AuditLog,
        "before_insert",
        fail_audit_insert,
    )

    try:

        with pytest.raises(
            RuntimeError,
            match="Simulated audit-log insertion failure.",
        ):

            record_consent_decision(
                db=db,
                user_id=user.id,
                purpose="analytics",
                granted=True,
                policy_version="1.0",
            )

    finally:

        event.remove(
            AuditLog,
            "before_insert",
            fail_audit_insert,
        )

    assert db.query(UserConsent).count() == 0
    assert db.query(AuditLog).count() == 0


def test_withdrawal_creates_new_record_without_overwriting(
    test_db,
):

    db, user = test_db

    original = record_consent_decision(
        db=db,
        user_id=user.id,
        purpose="conversation_storage",
        granted=True,
        policy_version="1.0",
    )

    withdrawal = record_consent_decision(
        db=db,
        user_id=user.id,
        purpose="conversation_storage",
        granted=False,
        policy_version="1.0",
    )

    assert original.id != withdrawal.id
    assert original.granted is True
    assert withdrawal.granted is False

    assert db.query(UserConsent).count() == 2


def test_consent_purposes_are_independent(test_db):

    db, user = test_db

    record_consent_decision(
        db=db,
        user_id=user.id,
        purpose="conversation_storage",
        granted=True,
        policy_version="1.0",
    )

    record_consent_decision(
        db=db,
        user_id=user.id,
        purpose="analytics",
        granted=False,
        policy_version="1.0",
    )

    decisions = get_current_consent_decisions(
        db,
        user.id,
    )

    assert decisions["conversation_storage"].granted is True
    assert decisions["analytics"].granted is False


def test_latest_decision_is_returned(test_db):

    db, user = test_db

    record_consent_decision(
        db=db,
        user_id=user.id,
        purpose="analytics",
        granted=True,
        policy_version="1.0",
    )

    import time
    time.sleep(0.01)

    latest = record_consent_decision(
        db=db,
        user_id=user.id,
        purpose="analytics",
        granted=False,
        policy_version="2.0",
    )

    decisions = get_current_consent_decisions(
        db,
        user.id,
    )

    assert decisions["analytics"].id == latest.id
    assert decisions["analytics"].granted is False
    assert decisions["analytics"].policy_version == "2.0"

    assert db.query(UserConsent).count() == 2


@pytest.mark.parametrize(
    "purpose",
    [
        "referral",
        "conversation",
        "invalid",
    ],
)
def test_invalid_consent_purpose_is_rejected(
    test_db,
    purpose,
):

    db, user = test_db

    with pytest.raises(ValueError):

        record_consent_decision(
            db=db,
            user_id=user.id,
            purpose=purpose,
            granted=True,
            policy_version="1.0",
        )


@pytest.mark.parametrize(
    "policy_version",
    [
        "",
        "   ",
        "x" * 51,
    ],
)
def test_invalid_policy_version_is_rejected(
    test_db,
    policy_version,
):

    db, user = test_db

    with pytest.raises(ValueError):

        record_consent_decision(
            db=db,
            user_id=user.id,
            purpose="analytics",
            granted=True,
            policy_version=policy_version,
        )


def test_invalid_user_id_does_not_create_consent(test_db):

    db, user = test_db

    with pytest.raises(ValueError):

        record_consent_decision(
            db=db,
            user_id=uuid4(),
            purpose="analytics",
            granted=True,
            policy_version="1.0",
        )

    assert db.query(UserConsent).count() == 0

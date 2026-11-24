from backend.app.database.models.audit_log import AuditLog
from datetime import timedelta, timezone

import pytest

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.database.base import Base
import backend.app.database.models

from backend.app.database.models.user import User
from backend.app.database.models.user_credential import UserCredential
from backend.app.database.models.email_verification_token import (
    EmailVerificationToken,
)

from backend.app.auth.verification_service import (
    get_current_utc_time,
    issue_verification_token,
)
from backend.app.auth.verification import hash_verification_token

from backend.app.services.resend_verification_service import (
    resend_verification_token,
    ResendVerificationResult,
)

from unittest.mock import patch


@pytest.fixture
def database():
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

    yield db

    db.close()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


def create_pending_account(db, email="student@example.com"):
    user = User()
    db.add(user)
    db.flush()

    credential = UserCredential(
        user_id=user.id,
        email=email,
        password_hash="test-password-hash",
    )
    db.add(credential)
    db.flush()

    return user, credential


def create_existing_token(db, user_id):
    issued = issue_verification_token(
        db=db,
        user_id=user_id,
    )
    db.commit()

    token_record = (
        db.query(EmailVerificationToken)
        .filter_by(user_id=user_id)
        .one()
    )

    return issued, token_record


def make_existing_token_older_than_cooldown(db, token_record):
    old_time = get_current_utc_time() - timedelta(seconds=120)

    token_record.created_at = old_time
    db.commit()
    db.refresh(token_record)


def test_successful_token_reissuance(database):
    db = database

    user, credential = create_pending_account(db)

    old_issued, old_record = create_existing_token(
        db,
        user.id,
    )

    make_existing_token_older_than_cooldown(
        db,
        old_record,
    )

    result = resend_verification_token(
        db=db,
        email=" STUDENT@EXAMPLE.COM ",
    )

    assert isinstance(result, ResendVerificationResult)
    assert result.email == "student@example.com"
    assert result.raw_token
    assert result.expires_at > get_current_utc_time()

    db.refresh(old_record)

    assert old_record.invalidated_at is not None
    assert old_record.used_at is None

    new_record = (
        db.query(EmailVerificationToken)
        .filter(
            EmailVerificationToken.token_hash
            == hash_verification_token(result.raw_token)
        )
        .one()
    )

    assert new_record.user_id == user.id
    assert new_record.invalidated_at is None
    assert new_record.used_at is None

    assert old_record.token_hash != new_record.token_hash

    print("PASS: Successful token reissuance")


def test_resend_cooldown_enforced(database):
    db = database

    user, credential = create_pending_account(db)

    issued, existing_record = create_existing_token(
        db,
        user.id,
    )

    result = resend_verification_token(
        db=db,
        email="student@example.com",
    )

    assert result is None

    db.refresh(existing_record)

    assert existing_record.invalidated_at is None

    assert (
        db.query(EmailVerificationToken).count() == 1
    )

    print("PASS: Resend cooldown enforced")


def test_nonexistent_account_returns_none(database):
    result = resend_verification_token(
        db=database,
        email="missing@example.com",
    )

    assert result is None

    assert (
        database.query(EmailVerificationToken).count() == 0
    )

    print("PASS: Nonexistent account handled safely")


def test_already_verified_account_rejected(database):
    db = database

    user, credential = create_pending_account(db)

    now = get_current_utc_time()

    credential.email_verified = True
    credential.email_verified_at = now
    user.account_status = "active"

    db.commit()

    result = resend_verification_token(
        db=db,
        email=credential.email,
    )

    assert result is None

    assert (
        db.query(EmailVerificationToken).count() == 0
    )

    print("PASS: Already-verified account rejected")


def test_database_rollback_on_token_issue_failure(database):
    db = database

    user, credential = create_pending_account(db)

    issued, old_record = create_existing_token(
        db,
        user.id,
    )

    make_existing_token_older_than_cooldown(
        db,
        old_record,
    )

    with patch(
        "backend.app.services.resend_verification_service.issue_verification_token",
        side_effect=RuntimeError("Simulated token issue failure"),
    ):
        with pytest.raises(RuntimeError):
            resend_verification_token(
                db=db,
                email=credential.email,
            )

    db.refresh(old_record)

    assert old_record.invalidated_at is None

    assert (
        db.query(EmailVerificationToken).count() == 1
    )

    print("PASS: Database rollback on token issue failure")


def test_raw_token_not_exposed_in_result_repr(database):
    db = database

    user, credential = create_pending_account(db)

    issued, old_record = create_existing_token(
        db,
        user.id,
    )

    make_existing_token_older_than_cooldown(
        db,
        old_record,
    )

    result = resend_verification_token(
        db=db,
        email=credential.email,
    )

    assert result is not None
    assert result.raw_token not in repr(result)

    print("PASS: Raw token excluded from result representation")


def test_invalid_cooldown_rejected(database):
    with pytest.raises(ValueError):
        resend_verification_token(
            db=database,
            email="student@example.com",
            cooldown_seconds=0,
        )

    with pytest.raises(ValueError):
        resend_verification_token(
            db=database,
            email="student@example.com",
            cooldown_seconds=-10,
        )

    print("PASS: Invalid cooldown values rejected")


if __name__ == "__main__":
    raise SystemExit(
        pytest.main([__file__, "-v", "-s"])
    )

# ============================================================
# RESEND VERIFICATION AUDIT TESTS
# ============================================================

def test_successful_resend_creates_audit_log(database):
    db = database

    user, credential = create_pending_account(db)
    issued, old_record = create_existing_token(db, user.id)

    make_existing_token_older_than_cooldown(db, old_record)

    result = resend_verification_token(
        db=db,
        email=credential.email,
    )

    assert result is not None

    audit = (
        db.query(AuditLog)
        .filter_by(action="email_verification_resend")
        .one()
    )

    assert audit.actor_user_id == user.id
    assert audit.category == "account"
    assert audit.outcome == "success"
    assert audit.target_type == "user"
    assert audit.target_id == str(user.id)
    assert audit.metadata_json == {"reason": "token_resent"}

    assert credential.email not in str(audit.metadata_json)
    assert result.raw_token not in str(audit.metadata_json)


def test_resend_cooldown_creates_informational_audit_log(database):
    db = database

    user, credential = create_pending_account(db)
    create_existing_token(db, user.id)

    result = resend_verification_token(
        db=db,
        email=credential.email,
    )

    assert result is None

    audit = (
        db.query(AuditLog)
        .filter_by(action="email_verification_resend")
        .one()
    )

    assert audit.actor_user_id == user.id
    assert audit.category == "account"
    assert audit.outcome == "info"
    assert audit.metadata_json == {
        "reason": "resend_cooldown_active"
    }


def test_nonexistent_account_creates_anonymous_audit_log(database):
    db = database

    result = resend_verification_token(
        db=db,
        email="missing@example.com",
    )

    assert result is None

    audit = (
        db.query(AuditLog)
        .filter_by(action="email_verification_resend")
        .one()
    )

    assert audit.actor_user_id is None
    assert audit.category == "account"
    assert audit.outcome == "info"
    assert audit.target_type is None
    assert audit.target_id is None
    assert audit.metadata_json == {
        "reason": "account_unavailable"
    }

    assert "missing@example.com" not in str(audit.metadata_json)


def test_resend_rolls_back_if_audit_creation_fails(database):
    db = database

    user, credential = create_pending_account(db)

    issued, old_record = create_existing_token(db, user.id)

    make_existing_token_older_than_cooldown(db, old_record)

    def fail_audit_insert(mapper, connection, target):
        raise RuntimeError("Simulated audit insertion failure")

    event.listen(AuditLog, "before_insert", fail_audit_insert)

    try:
        with pytest.raises(RuntimeError):
            resend_verification_token(
                db=db,
                email=credential.email,
            )
    finally:
        event.remove(AuditLog, "before_insert", fail_audit_insert)

    db.refresh(old_record)

    assert old_record.invalidated_at is None

    assert (
        db.query(EmailVerificationToken)
        .filter_by(user_id=user.id)
        .count()
        == 1
    )

    assert (
        db.query(AuditLog)
        .filter_by(action="email_verification_resend")
        .count()
        == 0
    )
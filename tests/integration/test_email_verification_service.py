from datetime import timedelta

import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.database.base import Base
from backend.app.database.models.audit_log import AuditLog
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

from backend.app.services.email_verification_service import (
    verify_email_address,
)


# ============================================================
# 1. TEMPORARY DATABASE FIXTURE
# ============================================================

@pytest.fixture
def db_session():

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    Base.metadata.create_all(bind=engine)

    TestingSessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )

    db = TestingSessionLocal()

    try:
        yield db

    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


# ============================================================
# 2. TEST ACCOUNT AND TOKEN HELPER
# ============================================================

def create_pending_account(db_session):

    user = User(
        role="student",
        account_status="pending_verification",
    )

    credential = UserCredential(
        email="student@example.com",
        password_hash="test-password-hash",
        email_verified=False,
        email_verified_at=None,
    )

    user.credential = credential

    db_session.add(user)
    db_session.flush()

    token_result = issue_verification_token(
        db=db_session,
        user_id=user.id,
    )

    db_session.commit()

    return user, credential, token_result.raw_token


def get_token_record(db_session, raw_token):

    token_hash = hash_verification_token(raw_token)

    return (
        db_session.query(EmailVerificationToken)
        .filter_by(token_hash=token_hash)
        .one()
    )


# ============================================================
# 3. VALID TOKEN ACTIVATES ACCOUNT
# ============================================================

def test_valid_token_verifies_email_and_activates_account(db_session):

    user, credential, raw_token = create_pending_account(db_session)

    result = verify_email_address(
        db=db_session,
        raw_token=raw_token,
    )

    db_session.refresh(user)
    db_session.refresh(credential)

    token_record = get_token_record(db_session, raw_token)

    assert result is True
    assert credential.email_verified is True
    assert credential.email_verified_at is not None
    assert user.account_status == "active"
    assert token_record.used_at is not None
    assert token_record.invalidated_at is None


# ============================================================
# 4. INVALID TOKEN IS REJECTED
# ============================================================

def test_invalid_token_is_rejected(db_session):

    result = verify_email_address(
        db=db_session,
        raw_token="invalid-verification-token",
    )

    assert result is False


# ============================================================
# 5. EXPIRED TOKEN IS REJECTED
# ============================================================

def test_expired_token_is_rejected(db_session):

    user, credential, raw_token = create_pending_account(db_session)

    token_record = get_token_record(db_session, raw_token)

    now = get_current_utc_time()

    # Preserve the database constraint:
    # expires_at must be later than created_at.
    token_record.created_at = now - timedelta(hours=2)
    token_record.expires_at = now - timedelta(hours=1)

    db_session.commit()

    result = verify_email_address(
        db=db_session,
        raw_token=raw_token,
    )

    db_session.refresh(user)
    db_session.refresh(credential)

    assert result is False
    assert credential.email_verified is False
    assert user.account_status == "pending_verification"


# ============================================================
# 6. INVALIDATED TOKEN IS REJECTED
# ============================================================

def test_invalidated_token_is_rejected(db_session):

    user, credential, raw_token = create_pending_account(db_session)

    token_record = get_token_record(db_session, raw_token)

    token_record.invalidated_at = get_current_utc_time()

    db_session.commit()

    result = verify_email_address(
        db=db_session,
        raw_token=raw_token,
    )

    db_session.refresh(user)
    db_session.refresh(credential)

    assert result is False
    assert credential.email_verified is False
    assert user.account_status == "pending_verification"


# ============================================================
# 7. TOKEN CANNOT BE REUSED
# ============================================================

def test_verification_token_cannot_be_reused(db_session):

    user, credential, raw_token = create_pending_account(db_session)

    first_result = verify_email_address(
        db=db_session,
        raw_token=raw_token,
    )

    second_result = verify_email_address(
        db=db_session,
        raw_token=raw_token,
    )

    db_session.refresh(user)
    db_session.refresh(credential)

    assert first_result is True
    assert second_result is False
    assert credential.email_verified is True
    assert user.account_status == "active"


# ============================================================
# 8. TRANSACTION ROLLBACK ON COMMIT FAILURE
# ============================================================

def test_verification_rolls_back_if_commit_fails(
    db_session,
    monkeypatch,
):

    user, credential, raw_token = create_pending_account(db_session)

    def fail_commit():
        raise RuntimeError("Simulated database commit failure.")

    monkeypatch.setattr(db_session, "commit", fail_commit)

    with pytest.raises(RuntimeError):
        verify_email_address(
            db=db_session,
            raw_token=raw_token,
        )

    # The service must have rolled back token consumption
    # and account activation together.

    db_session.refresh(user)
    db_session.refresh(credential)

    token_record = get_token_record(db_session, raw_token)

    assert user.account_status == "pending_verification"
    assert credential.email_verified is False
    assert credential.email_verified_at is None
    assert token_record.used_at is None



# ============================================================
# 9. EMAIL VERIFICATION AUDIT LOGGING
# ============================================================

def test_successful_verification_creates_audit_log(db_session):

    user, credential, raw_token = create_pending_account(db_session)

    result = verify_email_address(
        db=db_session,
        raw_token=raw_token,
    )

    audit = db_session.query(AuditLog).one()

    assert result is True
    assert audit.actor_user_id == user.id
    assert audit.category == "account"
    assert audit.action == "email_verification"
    assert audit.outcome == "success"
    assert audit.target_type == "user"
    assert audit.target_id == str(user.id)
    assert audit.metadata_json == {
        "reason": "email_verified"
    }

    assert raw_token not in str(audit.metadata_json)
    assert credential.email not in str(audit.metadata_json)


def test_invalid_verification_token_creates_anonymous_audit_log(
    db_session,
):

    result = verify_email_address(
        db=db_session,
        raw_token="invalid-verification-token",
    )

    audit = db_session.query(AuditLog).one()

    assert result is False
    assert audit.actor_user_id is None
    assert audit.category == "account"
    assert audit.action == "email_verification"
    assert audit.outcome == "failure"
    assert audit.target_type is None
    assert audit.target_id is None
    assert audit.metadata_json == {
        "reason": "invalid_or_unusable_token"
    }


def test_ineligible_account_creates_denied_audit_log(db_session):

    user, credential, raw_token = create_pending_account(db_session)

    user.account_status = "suspended"
    db_session.commit()

    result = verify_email_address(
        db=db_session,
        raw_token=raw_token,
    )

    db_session.refresh(user)
    db_session.refresh(credential)

    token_record = get_token_record(db_session, raw_token)
    audit = db_session.query(AuditLog).one()

    assert result is False
    assert user.account_status == "suspended"
    assert credential.email_verified is False
    assert token_record.used_at is not None

    assert audit.actor_user_id == user.id
    assert audit.category == "account"
    assert audit.action == "email_verification"
    assert audit.outcome == "denied"
    assert audit.target_type == "user"
    assert audit.target_id == str(user.id)
    assert audit.metadata_json == {
        "reason": "account_not_eligible"
    }


def test_verification_rolls_back_if_audit_creation_fails(db_session):

    from sqlalchemy import event

    user, credential, raw_token = create_pending_account(db_session)

    def fail_audit_insert(mapper, connection, target):
        raise RuntimeError(
            "Simulated email verification audit failure."
        )

    event.listen(
        AuditLog,
        "before_insert",
        fail_audit_insert,
    )

    try:
        with pytest.raises(RuntimeError):
            verify_email_address(
                db=db_session,
                raw_token=raw_token,
            )
    finally:
        event.remove(
            AuditLog,
            "before_insert",
            fail_audit_insert,
        )

    db_session.refresh(user)
    db_session.refresh(credential)

    token_record = get_token_record(db_session, raw_token)

    assert user.account_status == "pending_verification"
    assert credential.email_verified is False
    assert credential.email_verified_at is None
    assert token_record.used_at is None
    assert db_session.query(AuditLog).count() == 0

from datetime import datetime, timedelta, timezone

import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.database.base import Base
import backend.app.database.models

from backend.app.auth.password import hash_password
from backend.app.auth.tokens import (
    decode_access_token,
    hash_refresh_token,
)

from backend.app.database.models.user import User
from backend.app.database.models.user_credential import UserCredential
from backend.app.database.models.user_session import UserSession

from backend.app.services.login_service import (
    authenticate_user,
    LoginAuthenticationError,
    MAX_FAILED_LOGIN_ATTEMPTS,
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

    TestingSessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    Base.metadata.create_all(bind=engine)

    db = TestingSessionLocal()

    try:
        yield db

    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


# ============================================================
# 2. TEST ACCOUNT HELPER
# ============================================================

def create_account(
    db,
    email="student@example.com",
    password="StrongPassword123!",
    account_status="active",
    email_verified=True,
    failed_login_attempts=0,
):

    now = datetime.now(timezone.utc)

    user = User(
        role="student",
        account_status=account_status,
    )

    if account_status == "deleted":
        user.deleted_at = now

    db.add(user)
    db.flush()

    credential = UserCredential(
        user_id=user.id,
        email=email,
        password_hash=hash_password(password),
        email_verified=email_verified,
        email_verified_at=now if email_verified else None,
        failed_login_attempts=failed_login_attempts,
    )

    db.add(credential)
    db.commit()

    return user, credential


# ============================================================
# 3. SUCCESSFUL LOGIN
# ============================================================

def test_successful_login_creates_authenticated_session(db_session):

    user, credential = create_account(db_session)

    result = authenticate_user(
        db=db_session,
        email=" STUDENT@EXAMPLE.COM ",
        password="StrongPassword123!",
    )

    db_session.refresh(credential)

    assert result.user_id == user.id
    assert result.role == "student"
    assert result.token_type == "bearer"

    assert result.access_token
    assert result.refresh_token

    assert result.refresh_token not in repr(result)

    claims = decode_access_token(result.access_token)

    assert claims["sub"] == str(user.id)
    assert claims["type"] == "access"

    assert credential.last_login_at is not None
    assert credential.failed_login_attempts == 0

    assert db_session.query(UserSession).count() == 1


# ============================================================
# 4. REFRESH TOKEN HASH STORAGE
# ============================================================

def test_refresh_token_is_stored_as_hash_only(db_session):

    user, credential = create_account(db_session)

    result = authenticate_user(
        db=db_session,
        email=credential.email,
        password="StrongPassword123!",
    )

    stored_session = (
        db_session.query(UserSession)
        .filter_by(user_id=user.id)
        .one()
    )

    expected_hash = hash_refresh_token(
        result.refresh_token
    )

    assert stored_session.refresh_token_hash == expected_hash
    assert stored_session.refresh_token_hash != result.refresh_token

    assert stored_session.expires_at is not None
    assert stored_session.revoked_at is None


# ============================================================
# 5. INCORRECT PASSWORD TRACKING
# ============================================================

def test_incorrect_password_increments_failed_attempts(db_session):

    user, credential = create_account(db_session)

    with pytest.raises(LoginAuthenticationError):
        authenticate_user(
            db=db_session,
            email=credential.email,
            password="IncorrectPassword123!",
        )

    db_session.refresh(credential)

    assert credential.failed_login_attempts == 1
    assert credential.locked_until is None

    assert db_session.query(UserSession).count() == 0


# ============================================================
# 6. FIVE FAILED ATTEMPTS TRIGGER LOCKOUT
# ============================================================

def test_five_failed_attempts_trigger_temporary_lockout(db_session):

    user, credential = create_account(db_session)

    for _ in range(MAX_FAILED_LOGIN_ATTEMPTS):

        with pytest.raises(LoginAuthenticationError):
            authenticate_user(
                db=db_session,
                email=credential.email,
                password="IncorrectPassword123!",
            )

    db_session.refresh(credential)

    assert (
        credential.failed_login_attempts
        == MAX_FAILED_LOGIN_ATTEMPTS
    )

    assert credential.locked_until is not None

    locked_until = credential.locked_until

    if locked_until.tzinfo is None:
        locked_until = locked_until.replace(
            tzinfo=timezone.utc
        )

    assert locked_until > datetime.now(timezone.utc)


# ============================================================
# 7. ACTIVE LOCKOUT BLOCKS CORRECT PASSWORD
# ============================================================

def test_active_lockout_blocks_correct_password(db_session):

    user, credential = create_account(db_session)

    credential.failed_login_attempts = MAX_FAILED_LOGIN_ATTEMPTS
    credential.locked_until = (
        datetime.now(timezone.utc) + timedelta(minutes=15)
    )

    db_session.commit()

    with pytest.raises(LoginAuthenticationError):
        authenticate_user(
            db=db_session,
            email=credential.email,
            password="StrongPassword123!",
        )

    assert db_session.query(UserSession).count() == 0


# ============================================================
# 8. UNVERIFIED AND UNAVAILABLE ACCOUNTS ARE REJECTED
# ============================================================

@pytest.mark.parametrize(
    "account_status, email_verified",
    [
        ("pending_verification", False),
        ("suspended", True),
        ("deleted", True),
    ],
)
def test_unavailable_account_cannot_login(
    db_session,
    account_status,
    email_verified,
):

    user, credential = create_account(
        db_session,
        account_status=account_status,
        email_verified=email_verified,
    )

    with pytest.raises(LoginAuthenticationError):
        authenticate_user(
            db=db_session,
            email=credential.email,
            password="StrongPassword123!",
        )

    assert db_session.query(UserSession).count() == 0


# ============================================================
# 9. SUCCESSFUL LOGIN RESETS FAILED ATTEMPTS
# ============================================================

def test_successful_login_resets_failed_attempts(db_session):

    user, credential = create_account(
        db_session,
        failed_login_attempts=3,
    )

    result = authenticate_user(
        db=db_session,
        email=credential.email,
        password="StrongPassword123!",
    )

    db_session.refresh(credential)

    assert result.user_id == user.id
    assert credential.failed_login_attempts == 0
    assert credential.locked_until is None


# ============================================================
# 10. DATABASE FAILURE ROLLS BACK SESSION CREATION
# ============================================================

def test_login_rolls_back_if_commit_fails(
    db_session,
    monkeypatch,
):

    user, credential = create_account(db_session)

    def fail_commit():
        raise RuntimeError(
            "Simulated database commit failure."
        )

    monkeypatch.setattr(
        db_session,
        "commit",
        fail_commit,
    )

    with pytest.raises(RuntimeError):
        authenticate_user(
            db=db_session,
            email=credential.email,
            password="StrongPassword123!",
        )

    db_session.refresh(credential)

    assert credential.last_login_at is None
    assert db_session.query(UserSession).count() == 0

# ============================================================
# 11. LOGIN AUDIT LOGGING
# ============================================================

def test_successful_login_creates_audit_log(db_session):

    from backend.app.database.models.audit_log import AuditLog

    user, credential = create_account(db_session)

    authenticate_user(
        db=db_session,
        email=credential.email,
        password="StrongPassword123!",
    )

    audit = db_session.query(AuditLog).one()

    assert audit.actor_user_id == user.id
    assert audit.category == "authentication"
    assert audit.action == "login"
    assert audit.outcome == "success"
    assert audit.target_type == "user"
    assert audit.target_id == str(user.id)
    assert audit.metadata_json == {"reason": "authenticated"}


def test_incorrect_password_creates_audit_log(db_session):

    from backend.app.database.models.audit_log import AuditLog

    user, credential = create_account(db_session)

    with pytest.raises(LoginAuthenticationError):
        authenticate_user(
            db=db_session,
            email=credential.email,
            password="IncorrectPassword123!",
        )

    db_session.refresh(credential)

    audit = db_session.query(AuditLog).one()

    assert credential.failed_login_attempts == 1
    assert audit.actor_user_id == user.id
    assert audit.category == "authentication"
    assert audit.action == "login"
    assert audit.outcome == "failure"
    assert audit.metadata_json == {"reason": "invalid_credentials"}


def test_unknown_email_creates_anonymous_audit_log(db_session):

    from backend.app.database.models.audit_log import AuditLog

    with pytest.raises(LoginAuthenticationError):
        authenticate_user(
            db=db_session,
            email="unknown@example.com",
            password="SomePassword123!",
        )

    audit = db_session.query(AuditLog).one()

    assert audit.actor_user_id is None
    assert audit.category == "authentication"
    assert audit.action == "login"
    assert audit.outcome == "failure"
    assert audit.metadata_json == {"reason": "invalid_credentials"}

    assert "unknown@example.com" not in str(audit.metadata_json)


# ============================================================
# 12. AUDIT FAILURE ROLLS BACK LOGIN
# ============================================================

def test_login_rolls_back_if_audit_creation_fails(
    db_session,
    monkeypatch,
):

    from backend.app.database.models.audit_log import AuditLog

    user, credential = create_account(db_session)

    def fail_audit_insert(mapper, connection, target):
        raise RuntimeError("Simulated audit failure.")

    from sqlalchemy import event

    event.listen(
        AuditLog,
        "before_insert",
        fail_audit_insert,
    )

    try:
        with pytest.raises(RuntimeError):
            authenticate_user(
                db=db_session,
                email=credential.email,
                password="StrongPassword123!",
            )
    finally:
        event.remove(
            AuditLog,
            "before_insert",
            fail_audit_insert,
        )

    db_session.refresh(credential)

    assert credential.last_login_at is None
    assert credential.failed_login_attempts == 0
    assert db_session.query(UserSession).count() == 0
    assert db_session.query(AuditLog).count() == 0


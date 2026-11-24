from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.main import app
from backend.app.database.base import Base
import backend.app.database.models

from backend.app.database.session import get_db
from backend.app.database.models import (
    User,
    UserCredential,
    EmailVerificationToken,
)

from backend.app.auth.verification_service import (
    get_current_utc_time,
    issue_verification_token,
)


# ============================================================
# 1. TEST DATABASE AND API ENVIRONMENT
# ============================================================

@pytest.fixture
def test_environment():
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

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    try:
        client = TestClient(app)

        yield client, TestingSessionLocal

        client.close()

    finally:
        app.dependency_overrides.pop(get_db, None)
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


# ============================================================
# 2. CREATE A PENDING ACCOUNT WITH A VERIFICATION TOKEN
# ============================================================

def create_pending_account(SessionLocal, email):
    with SessionLocal() as db:
        user = User(
            role="student",
            account_status="pending_verification",
        )

        db.add(user)
        db.flush()

        credential = UserCredential(
            user_id=user.id,
            email=email,
            password_hash="test-password-hash",
            email_verified=False,
        )

        db.add(credential)
        db.flush()

        issued = issue_verification_token(
            db=db,
            user_id=user.id,
        )

        db.commit()

        return user.id, issued.raw_token


# ============================================================
# 3. EXPECTED SUCCESS RESPONSE
# ============================================================

def expected_success_response():
    return {
        "success": True,
        "status": "email_verified",
        "message": (
            "Your email address has been verified. "
            "You can now sign in."
        ),
    }


# ============================================================
# 4. SUCCESSFUL EMAIL VERIFICATION
# ============================================================

def test_successful_email_verification(test_environment):
    client, SessionLocal = test_environment

    user_id, raw_token = create_pending_account(
        SessionLocal,
        "student@example.com",
    )

    response = client.post(
        "/api/v1/auth/verify-email",
        json={"token": raw_token},
    )

    assert response.status_code == 200
    assert response.json() == expected_success_response()

    assert raw_token not in response.text

    with SessionLocal() as db:
        user = db.query(User).filter_by(id=user_id).one()

        credential = (
            db.query(UserCredential)
            .filter_by(user_id=user_id)
            .one()
        )

        token_record = (
            db.query(EmailVerificationToken)
            .filter_by(user_id=user_id)
            .one()
        )

        assert user.account_status == "active"

        assert credential.email_verified is True
        assert credential.email_verified_at is not None

        assert token_record.used_at is not None
        assert token_record.invalidated_at is None

    print("PASS: Successful email verification")


# ============================================================
# 5. INVALID TOKEN REJECTION
# ============================================================

def test_invalid_verification_token_rejected(test_environment):
    client, SessionLocal = test_environment

    user_id, original_token = create_pending_account(
        SessionLocal,
        "invalid-token@example.com",
    )

    invalid_token = "invalid-verification-token-value-123456"

    response = client.post(
        "/api/v1/auth/verify-email",
        json={"token": invalid_token},
    )

    assert response.status_code == 400

    # The global exception handler controls the error response format.
    # Ensure the invalid verification token is never exposed.
    assert invalid_token not in response.text

    with SessionLocal() as db:
        user = db.query(User).filter_by(id=user_id).one()

        credential = (
            db.query(UserCredential)
            .filter_by(user_id=user_id)
            .one()
        )

        token_record = (
            db.query(EmailVerificationToken)
            .filter_by(user_id=user_id)
            .one()
        )

        assert user.account_status == "pending_verification"
        assert credential.email_verified is False
        assert token_record.used_at is None
        assert original_token not in response.text

    print("PASS: Invalid verification token rejected")


# ============================================================
# 6. EXPIRED TOKEN REJECTION
# ============================================================

def test_expired_verification_token_rejected(test_environment):
    client, SessionLocal = test_environment

    user_id, raw_token = create_pending_account(
        SessionLocal,
        "expired@example.com",
    )

    with SessionLocal() as db:
        token_record = (
            db.query(EmailVerificationToken)
            .filter_by(user_id=user_id)
            .one()
        )

        now = get_current_utc_time()

        # Preserve the database constraint:
        # expires_at must be later than created_at.
        token_record.created_at = now - timedelta(hours=2)
        token_record.expires_at = now - timedelta(hours=1)

        db.commit()

    response = client.post(
        "/api/v1/auth/verify-email",
        json={"token": raw_token},
    )

    assert response.status_code == 400

    with SessionLocal() as db:
        user = db.query(User).filter_by(id=user_id).one()

        credential = (
            db.query(UserCredential)
            .filter_by(user_id=user_id)
            .one()
        )

        token_record = (
            db.query(EmailVerificationToken)
            .filter_by(user_id=user_id)
            .one()
        )

        assert user.account_status == "pending_verification"
        assert credential.email_verified is False
        assert token_record.used_at is None

    print("PASS: Expired verification token rejected")


# ============================================================
# 7. INVALIDATED TOKEN REJECTION
# ============================================================

def test_invalidated_verification_token_rejected(test_environment):
    client, SessionLocal = test_environment

    user_id, raw_token = create_pending_account(
        SessionLocal,
        "invalidated@example.com",
    )

    with SessionLocal() as db:
        token_record = (
            db.query(EmailVerificationToken)
            .filter_by(user_id=user_id)
            .one()
        )

        token_record.invalidated_at = get_current_utc_time()

        db.commit()

    response = client.post(
        "/api/v1/auth/verify-email",
        json={"token": raw_token},
    )

    assert response.status_code == 400

    with SessionLocal() as db:
        user = db.query(User).filter_by(id=user_id).one()

        credential = (
            db.query(UserCredential)
            .filter_by(user_id=user_id)
            .one()
        )

        token_record = (
            db.query(EmailVerificationToken)
            .filter_by(user_id=user_id)
            .one()
        )

        assert user.account_status == "pending_verification"
        assert credential.email_verified is False
        assert token_record.invalidated_at is not None
        assert token_record.used_at is None

    print("PASS: Invalidated verification token rejected")


# ============================================================
# 8. TOKEN REUSE REJECTION
# ============================================================

def test_verification_token_cannot_be_reused(test_environment):
    client, SessionLocal = test_environment

    user_id, raw_token = create_pending_account(
        SessionLocal,
        "reuse@example.com",
    )

    first_response = client.post(
        "/api/v1/auth/verify-email",
        json={"token": raw_token},
    )

    second_response = client.post(
        "/api/v1/auth/verify-email",
        json={"token": raw_token},
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 400

    with SessionLocal() as db:
        user = db.query(User).filter_by(id=user_id).one()

        credential = (
            db.query(UserCredential)
            .filter_by(user_id=user_id)
            .one()
        )

        token_record = (
            db.query(EmailVerificationToken)
            .filter_by(user_id=user_id)
            .one()
        )

        assert user.account_status == "active"
        assert credential.email_verified is True
        assert token_record.used_at is not None

    print("PASS: Previously used verification token rejected")


# ============================================================
# 9. SHORT TOKEN VALIDATION
# ============================================================

def test_short_verification_token_rejected(test_environment):
    client, SessionLocal = test_environment

    response = client.post(
        "/api/v1/auth/verify-email",
        json={"token": "short"},
    )

    assert response.status_code == 422

    with SessionLocal() as db:
        assert db.query(User).count() == 0
        assert db.query(EmailVerificationToken).count() == 0

    print("PASS: Short verification token rejected")


# ============================================================
# 10. UNEXPECTED REQUEST FIELDS REJECTED
# ============================================================

def test_unexpected_verification_request_fields_rejected(
    test_environment,
):
    client, SessionLocal = test_environment

    response = client.post(
        "/api/v1/auth/verify-email",
        json={
            "token": "a" * 40,
            "role": "administrator",
        },
    )

    assert response.status_code == 422

    with SessionLocal() as db:
        assert db.query(User).count() == 0
        assert db.query(EmailVerificationToken).count() == 0

    print("PASS: Unexpected verification request fields rejected")


# ============================================================
# 11. DIRECT TEST EXECUTION
# ============================================================

if __name__ == "__main__":
    raise SystemExit(
        pytest.main([__file__, "-v", "-s"])
    )

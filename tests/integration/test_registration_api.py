import re
from urllib.parse import parse_qs, unquote, urlparse
from unittest.mock import AsyncMock, patch

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
from backend.app.auth.verification import hash_verification_token
from backend.app.services.email_service import EmailDeliveryError


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

    with patch(
        "backend.app.api.routes.auth.send_email",
        new_callable=AsyncMock,
    ) as mocked_send_email:
        client = TestClient(app, raise_server_exceptions=False)

        yield client, TestingSessionLocal, mocked_send_email

        client.close()

    app.dependency_overrides.pop(get_db, None)
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


def test_successful_student_registration(test_environment):
    client, SessionLocal, mocked_email = test_environment

    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "  Student.Test@Example.com  ",
            "password": "SecurePassword123!",
        },
    )

    assert response.status_code == 202

    response_data = response.json()

    assert response_data["success"] is True
    assert response_data["status"] == "verification_pending"
    assert "token" not in response_data

    mocked_email.assert_awaited_once()

    email_arguments = mocked_email.await_args.kwargs

    assert email_arguments["to_email"] == "student.test@example.com"
    assert email_arguments["subject"]
    assert email_arguments["text_body"]

    with SessionLocal() as db:
        user = db.query(User).one()
        credential = db.query(UserCredential).one()
        token_record = db.query(EmailVerificationToken).one()

        assert user.role == "student"
        assert user.account_status == "pending_verification"

        assert credential.email == "student.test@example.com"
        assert credential.email_verified is False
        assert credential.password_hash != "SecurePassword123!"

        assert len(token_record.token_hash) == 64
        assert token_record.user_id == user.id

        email_body = email_arguments["text_body"]
        urls = re.findall(r"https?://[^\s<>]+", email_body)

        assert urls, "Verification URL was not found in email body."

        verification_url = urls[0].rstrip(".,)")
        raw_tokens = parse_qs(
            urlparse(verification_url).query
        ).get("token", [])

        assert raw_tokens, "Verification token was not found in URL."

        raw_token = unquote(raw_tokens[0])

        assert hash_verification_token(raw_token) == token_record.token_hash
        assert raw_token not in token_record.token_hash

    print("PASS: Successful student registration")


def test_duplicate_email_registration(test_environment):
    client, SessionLocal, mocked_email = test_environment

    payload = {
        "email": "duplicate@example.com",
        "password": "SecurePassword123!",
    }

    first_response = client.post(
        "/api/v1/auth/register",
        json=payload,
    )

    second_response = client.post(
        "/api/v1/auth/register",
        json=payload,
    )

    assert first_response.status_code == 202
    assert second_response.status_code == 202

    assert first_response.json() == second_response.json()

    mocked_email.assert_awaited_once()

    with SessionLocal() as db:
        assert db.query(User).count() == 1
        assert db.query(UserCredential).count() == 1
        assert db.query(EmailVerificationToken).count() == 1

    print("PASS: Duplicate email handled safely")


@pytest.mark.parametrize(
    "payload",
    [
        {
            "email": "not-an-email",
            "password": "SecurePassword123!",
        },
        {
            "email": "valid@example.com",
            "password": "short",
        },
        {
            "email": "valid@example.com",
            "password": "SecurePassword123!",
            "role": "administrator",
        },
    ],
)
def test_invalid_registration_payloads(test_environment, payload):
    client, SessionLocal, mocked_email = test_environment

    response = client.post(
        "/api/v1/auth/register",
        json=payload,
    )

    assert response.status_code == 422
    mocked_email.assert_not_awaited()

    with SessionLocal() as db:
        assert db.query(User).count() == 0
        assert db.query(UserCredential).count() == 0
        assert db.query(EmailVerificationToken).count() == 0

    print("PASS: Invalid registration payload rejected")


def test_email_delivery_failure(test_environment):
    client, SessionLocal, mocked_email = test_environment

    mocked_email.side_effect = EmailDeliveryError(
        "Simulated SMTP failure"
    )

    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "delivery.failure@example.com",
            "password": "SecurePassword123!",
        },
    )

    assert response.status_code == 503

    with SessionLocal() as db:
        user = db.query(User).one()
        credential = db.query(UserCredential).one()
        token_record = db.query(EmailVerificationToken).one()

        assert credential.user_id == user.id
        assert token_record.user_id == user.id
        assert credential.email_verified is False
        assert user.account_status == "pending_verification"

    mocked_email.assert_awaited_once()

    print("PASS: Email delivery failure handled")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))

# ============================================================
# 5. REGISTRATION AUDIT LOGGING
# ============================================================

def test_successful_registration_creates_audit_log(test_environment):
    from backend.app.database.models.audit_log import AuditLog

    client, SessionLocal, mocked_email = test_environment

    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "audit.success@example.com",
            "password": "SecurePassword123!",
        },
    )

    assert response.status_code == 202

    with SessionLocal() as db:
        user = db.query(User).one()

        audit = (
            db.query(AuditLog)
            .filter_by(
                action="student_registration",
                actor_user_id=user.id,
                    outcome="success",
            )
            .one()
        )

        assert audit.actor_user_id == user.id
        assert audit.category == "account"
        assert audit.outcome == "success"
        assert audit.target_type == "user"
        assert audit.target_id == str(user.id)
        assert audit.metadata_json == {
            "reason": "student_registered"
        }

        assert "audit.success@example.com" not in str(audit.metadata_json)
        assert "SecurePassword123!" not in str(audit.metadata_json)


def test_duplicate_registration_creates_anonymous_audit_log(test_environment):
    from backend.app.database.models.audit_log import AuditLog

    client, SessionLocal, mocked_email = test_environment

    payload = {
        "email": "audit.duplicate@example.com",
        "password": "SecurePassword123!",
    }

    first_response = client.post(
        "/api/v1/auth/register",
        json=payload,
    )

    second_response = client.post(
        "/api/v1/auth/register",
        json=payload,
    )

    assert first_response.status_code == 202
    assert second_response.status_code == 202
    assert first_response.json() == second_response.json()

    mocked_email.assert_awaited_once()

    with SessionLocal() as db:
        audit = (
            db.query(AuditLog)
            .filter_by(
                action="student_registration",
                actor_user_id=None,
                outcome="failure",
            )
            .one()
        )

        assert audit.actor_user_id is None
        assert audit.category == "account"
        assert audit.outcome == "failure"
        assert audit.target_type is None
        assert audit.target_id is None
        assert audit.metadata_json == {
            "reason": "duplicate_email"
        }

        assert "audit.duplicate@example.com" not in str(audit.metadata_json)


def test_registration_audit_metadata_contains_no_sensitive_data(test_environment):
    from backend.app.database.models.audit_log import AuditLog

    client, SessionLocal, mocked_email = test_environment

    email = "privacy.audit@example.com"
    password = "SensitivePassword123!"

    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": password,
        },
    )

    assert response.status_code == 202

    with SessionLocal() as db:
        audit = db.query(AuditLog).one()

        metadata = str(audit.metadata_json)

        assert email not in metadata
        assert password not in metadata
        assert "token" not in metadata.lower()
        assert audit.metadata_json == {
            "reason": "student_registered"
        }




# ============================================================
# 6. REGISTRATION AUDIT FAILURE ROLLBACK
# ============================================================

def test_registration_rolls_back_if_audit_creation_fails(
    test_environment,
):
    from sqlalchemy import event
    from backend.app.database.models.audit_log import AuditLog

    # --------------------------------------------------------
    # A. Get test environment
    # --------------------------------------------------------

    client, SessionLocal, mocked_email = test_environment

    # Allow TestClient to return HTTP 500 instead of
    # re-raising the server-side exception.

    # --------------------------------------------------------
    # B. Simulate audit insertion failure
    # --------------------------------------------------------

    def fail_audit_insert(mapper, connection, target):
        raise RuntimeError(
            "Simulated registration audit failure."
        )

    event.listen(
        AuditLog,
        "before_insert",
        fail_audit_insert,
    )

    # --------------------------------------------------------
    # C. Attempt student registration
    # --------------------------------------------------------

    try:
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "audit.rollback@example.com",
                "password": "SecurePassword123!",
            },
        )

    finally:
        # Always remove the SQLAlchemy event listener.
        event.remove(
            AuditLog,
            "before_insert",
            fail_audit_insert,
        )

    # --------------------------------------------------------
    # D. Verify server error
    # --------------------------------------------------------

    assert response.status_code == 500

    # Email must not be sent because registration failed.
    mocked_email.assert_not_awaited()

    # --------------------------------------------------------
    # E. Verify complete transaction rollback
    # --------------------------------------------------------

    with SessionLocal() as db:

        # No student account should remain.
        assert db.query(User).count() == 0

        # No authentication credentials should remain.
        assert db.query(UserCredential).count() == 0

        # No verification token should remain.
        assert db.query(EmailVerificationToken).count() == 0

        # No audit record should remain.
        assert db.query(AuditLog).count() == 0
import re
from datetime import timedelta
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
from backend.app.auth.verification_service import (
    get_current_utc_time,
    issue_verification_token,
)
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

    try:
        with patch(
            "backend.app.api.routes.auth.send_email",
            new_callable=AsyncMock,
        ) as mocked_email:

            client = TestClient(app)

            yield client, TestingSessionLocal, mocked_email

            client.close()

    finally:
        app.dependency_overrides.pop(get_db, None)
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def create_pending_account(SessionLocal, email):
    with SessionLocal() as db:
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

        issued = issue_verification_token(
            db=db,
            user_id=user.id,
        )

        db.commit()

        user_id = user.id

        return user_id, issued.raw_token


def age_existing_token(SessionLocal, user_id):
    with SessionLocal() as db:
        token = (
            db.query(EmailVerificationToken)
            .filter_by(user_id=user_id)
            .one()
        )

        token.created_at = (
            get_current_utc_time() - timedelta(seconds=120)
        )

        db.commit()


def expected_generic_response():
    return {
        "success": True,
        "status": "verification_request_received",
        "message": (
            "If this address is eligible for verification, "
            "verification instructions will be sent."
        ),
    }


def test_successful_resend_verification(test_environment):
    client, SessionLocal, mocked_email = test_environment

    user_id, old_raw_token = create_pending_account(
        SessionLocal,
        "student@example.com",
    )

    age_existing_token(SessionLocal, user_id)

    response = client.post(
        "/api/v1/auth/resend-verification",
        json={"email": "student@example.com"},
    )

    assert response.status_code == 202
    assert response.json() == expected_generic_response()

    mocked_email.assert_awaited_once()

    email_arguments = mocked_email.await_args.kwargs

    assert email_arguments["to_email"] == "student@example.com"
    assert email_arguments["subject"]
    assert email_arguments["text_body"]

    token_match = re.search(
        r"[?&]token=([A-Za-z0-9_-]+)",
        email_arguments["text_body"],
    )

    assert token_match is not None

    new_raw_token = token_match.group(1)

    assert new_raw_token != old_raw_token

    with SessionLocal() as db:
        tokens = (
            db.query(EmailVerificationToken)
            .filter_by(user_id=user_id)
            .order_by(EmailVerificationToken.created_at)
            .all()
        )

        assert len(tokens) == 2

        old_token = tokens[0]
        new_token = tokens[1]

        assert old_token.invalidated_at is not None
        assert old_token.used_at is None

        assert new_token.invalidated_at is None
        assert new_token.used_at is None

        assert (
            new_token.token_hash
            == hash_verification_token(new_raw_token)
        )

    assert new_raw_token not in response.text

    print("PASS: Successful resend API request")


def test_resend_cooldown_returns_generic_response(test_environment):
    client, SessionLocal, mocked_email = test_environment

    create_pending_account(
        SessionLocal,
        "cooldown@example.com",
    )

    response = client.post(
        "/api/v1/auth/resend-verification",
        json={"email": "cooldown@example.com"},
    )

    assert response.status_code == 202
    assert response.json() == expected_generic_response()

    mocked_email.assert_not_awaited()

    with SessionLocal() as db:
        assert db.query(EmailVerificationToken).count() == 1

    print("PASS: Cooldown handled with generic response")


def test_nonexistent_account_returns_generic_response(test_environment):
    client, SessionLocal, mocked_email = test_environment

    response = client.post(
        "/api/v1/auth/resend-verification",
        json={"email": "missing@example.com"},
    )

    assert response.status_code == 202
    assert response.json() == expected_generic_response()

    mocked_email.assert_not_awaited()

    with SessionLocal() as db:
        assert db.query(User).count() == 0
        assert db.query(EmailVerificationToken).count() == 0

    print("PASS: Nonexistent account handled safely")


def test_verified_account_returns_generic_response(test_environment):
    client, SessionLocal, mocked_email = test_environment

    with SessionLocal() as db:
        user = User()
        db.add(user)
        db.flush()

        credential = UserCredential(
            user_id=user.id,
            email="verified@example.com",
            password_hash="test-password-hash",
            email_verified=True,
            email_verified_at=get_current_utc_time(),
        )

        user.account_status = "active"

        db.add(credential)
        db.commit()

    response = client.post(
        "/api/v1/auth/resend-verification",
        json={"email": "verified@example.com"},
    )

    assert response.status_code == 202
    assert response.json() == expected_generic_response()

    mocked_email.assert_not_awaited()

    print("PASS: Verified account handled safely")


def test_email_failure_returns_generic_response(test_environment):
    client, SessionLocal, mocked_email = test_environment

    user_id, old_raw_token = create_pending_account(
        SessionLocal,
        "delivery-failure@example.com",
    )

    age_existing_token(SessionLocal, user_id)

    mocked_email.side_effect = EmailDeliveryError(
        "Simulated SMTP failure"
    )

    response = client.post(
        "/api/v1/auth/resend-verification",
        json={"email": "delivery-failure@example.com"},
    )

    assert response.status_code == 202
    assert response.json() == expected_generic_response()

    mocked_email.assert_awaited_once()

    with SessionLocal() as db:
        tokens = (
            db.query(EmailVerificationToken)
            .filter_by(user_id=user_id)
            .order_by(EmailVerificationToken.created_at)
            .all()
        )

        assert len(tokens) == 2
        assert tokens[0].invalidated_at is not None
        assert tokens[1].invalidated_at is None

    print("PASS: SMTP failure handled without account disclosure")


def test_invalid_request_rejected(test_environment):
    client, SessionLocal, mocked_email = test_environment

    response = client.post(
        "/api/v1/auth/resend-verification",
        json={
            "email": "invalid-email",
        },
    )

    assert response.status_code == 422

    mocked_email.assert_not_awaited()

    print("PASS: Invalid email rejected")


def test_unexpected_fields_rejected(test_environment):
    client, SessionLocal, mocked_email = test_environment

    response = client.post(
        "/api/v1/auth/resend-verification",
        json={
            "email": "student@example.com",
            "role": "administrator",
        },
    )

    assert response.status_code == 422

    mocked_email.assert_not_awaited()

    print("PASS: Unexpected request fields rejected")


if __name__ == "__main__":
    raise SystemExit(
        pytest.main([__file__, "-v", "-s"])
    )

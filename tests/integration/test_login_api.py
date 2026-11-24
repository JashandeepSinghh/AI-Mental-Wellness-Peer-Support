from datetime import datetime, timedelta, timezone

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
    UserSession,
)

from backend.app.auth.password import hash_password
from backend.app.auth.tokens import (
    decode_access_token,
    hash_refresh_token,
)

from backend.app.services.login_service import (
    MAX_FAILED_LOGIN_ATTEMPTS,
)


# ============================================================
# 1. TEST ENVIRONMENT
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

    client = TestClient(app)

    try:
        yield client, TestingSessionLocal

    finally:
        client.close()

        app.dependency_overrides.pop(get_db, None)

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
    locked_until=None,
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
        locked_until=locked_until,
    )

    db.add(credential)
    db.commit()

    return user, credential


# ============================================================
# 3. SUCCESSFUL LOGIN
# ============================================================

def test_successful_login_returns_access_token_and_cookie(
    test_environment,
):

    client, SessionLocal = test_environment

    with SessionLocal() as db:
        user, credential = create_account(db)

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "STUDENT@EXAMPLE.COM",
            "password": "StrongPassword123!",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["token_type"] == "bearer"
    assert data["access_token"]
    assert data["expires_in"] > 0

    assert data["user"]["id"] == str(user.id)
    assert data["user"]["role"] == "student"

    assert "refresh_token" not in data

    claims = decode_access_token(data["access_token"])

    assert claims["sub"] == str(user.id)
    assert claims["type"] == "access"

    refresh_token = response.cookies.get("refresh_token")

    assert refresh_token

    cookie_header = response.headers.get("set-cookie", "").lower()

    assert "httponly" in cookie_header
    assert "samesite=lax" in cookie_header
    assert "path=/api/v1/auth" in cookie_header

    with SessionLocal() as db:

        credential = db.query(UserCredential).one()
        stored_session = db.query(UserSession).one()

        assert credential.last_login_at is not None
        assert credential.failed_login_attempts == 0

        assert (
            stored_session.refresh_token_hash
            == hash_refresh_token(refresh_token)
        )

        assert stored_session.refresh_token_hash != refresh_token

    print("PASS: Successful login and secure refresh cookie")


# ============================================================
# 4. INCORRECT PASSWORD
# ============================================================

def test_login_rejects_incorrect_password(test_environment):

    client, SessionLocal = test_environment

    with SessionLocal() as db:
        create_account(db)

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "student@example.com",
            "password": "IncorrectPassword123!",
        },
    )

    assert response.status_code == 401

    error_data = response.json()

    assert error_data["success"] is False
    assert error_data["error"]["code"] == "UNAUTHORIZED"
    assert error_data["error"]["message"] == (
        "Authentication is required."
    )
    assert error_data["error"]["details"] == []

    assert "refresh_token" not in response.cookies

    with SessionLocal() as db:

        credential = db.query(UserCredential).one()

        assert credential.failed_login_attempts == 1
        assert db.query(UserSession).count() == 0

    print("PASS: Incorrect password rejected")


# ============================================================
# 5. UNVERIFIED ACCOUNT
# ============================================================

def test_login_rejects_unverified_account(test_environment):

    client, SessionLocal = test_environment

    with SessionLocal() as db:

        create_account(
            db,
            account_status="pending_verification",
            email_verified=False,
        )

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "student@example.com",
            "password": "StrongPassword123!",
        },
    )

    assert response.status_code == 401

    assert "refresh_token" not in response.cookies

    with SessionLocal() as db:
        assert db.query(UserSession).count() == 0

    print("PASS: Unverified account rejected")


# ============================================================
# 6. SUSPENDED ACCOUNT
# ============================================================

def test_login_rejects_suspended_account(test_environment):

    client, SessionLocal = test_environment

    with SessionLocal() as db:

        create_account(
            db,
            account_status="suspended",
            email_verified=True,
        )

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "student@example.com",
            "password": "StrongPassword123!",
        },
    )

    assert response.status_code == 401

    assert "refresh_token" not in response.cookies

    with SessionLocal() as db:
        assert db.query(UserSession).count() == 0

    print("PASS: Suspended account rejected")


# ============================================================
# 7. ACCOUNT LOCKOUT
# ============================================================

def test_login_rejects_locked_account(test_environment):

    client, SessionLocal = test_environment

    with SessionLocal() as db:

        create_account(
            db,
            failed_login_attempts=MAX_FAILED_LOGIN_ATTEMPTS,
            locked_until=(
                datetime.now(timezone.utc)
                + timedelta(minutes=15)
            ),
        )

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "student@example.com",
            "password": "StrongPassword123!",
        },
    )

    assert response.status_code == 401

    assert "refresh_token" not in response.cookies

    with SessionLocal() as db:
        assert db.query(UserSession).count() == 0

    print("PASS: Locked account rejected")


# ============================================================
# 8. INVALID LOGIN PAYLOADS
# ============================================================

@pytest.mark.parametrize(
    "payload",
    [
        {
            "email": "invalid-email",
            "password": "StrongPassword123!",
        },
        {
            "email": "student@example.com",
            "password": "",
        },
        {
            "email": "student@example.com",
            "password": "StrongPassword123!",
            "role": "administrator",
        },
        {
            "email": "student@example.com",
        },
    ],
)
def test_invalid_login_payloads(test_environment, payload):

    client, SessionLocal = test_environment

    response = client.post(
        "/api/v1/auth/login",
        json=payload,
    )

    assert response.status_code == 422

    with SessionLocal() as db:
        assert db.query(User).count() == 0
        assert db.query(UserCredential).count() == 0
        assert db.query(UserSession).count() == 0

    print("PASS: Invalid login payload rejected")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))


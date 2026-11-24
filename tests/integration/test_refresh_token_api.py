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
from backend.app.database.models import AuditLog, User, UserSession

from backend.app.auth.tokens import (
    create_refresh_token,
    decode_access_token,
    hash_refresh_token,
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
# 2. USER HELPER
# ============================================================

def create_user(db, account_status="active"):

    user = User(
        role="student",
        account_status=account_status,
    )

    if account_status == "deleted":
        user.deleted_at = datetime.now(timezone.utc)

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


# ============================================================
# 3. SESSION HELPER
# ============================================================

def create_session(
    db,
    user_id,
    raw_token=None,
    expires_at=None,
):

    now = datetime.now(timezone.utc)

    if raw_token is None:
        raw_token = create_refresh_token()

    if expires_at is None:
        expires_at = now + timedelta(days=7)

    created_at = (
        expires_at - timedelta(days=1)
        if expires_at <= now
        else now
    )

    user_session = UserSession(
        user_id=user_id,
        refresh_token_hash=hash_refresh_token(raw_token),
        created_at=created_at,
        expires_at=expires_at,
    )

    db.add(user_session)
    db.commit()
    db.refresh(user_session)

    return user_session, raw_token


# ============================================================
# 4. SUCCESSFUL REFRESH AND ROTATION
# ============================================================

def test_successful_refresh_rotates_cookie_and_returns_access_token(
    test_environment,
):

    client, SessionLocal = test_environment

    with SessionLocal() as db:

        user = create_user(db)

        old_session, old_token = create_session(
            db,
            user.id,
        )
        old_session_id = old_session.id

    response = client.post(
        "/api/v1/auth/refresh",
        cookies={"refresh_token": old_token},
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

    new_token = response.cookies.get("refresh_token")

    assert new_token
    assert new_token != old_token

    cookie_header = response.headers.get(
        "set-cookie",
        "",
    ).lower()

    assert "httponly" in cookie_header
    assert "samesite=lax" in cookie_header
    assert "path=/api/v1/auth" in cookie_header

    with SessionLocal() as db:

        old_session = db.get(
            UserSession,
            old_session_id,
        )

        assert old_session is not None
        assert old_session.revoked_at is not None
        assert old_session.replaced_by_session_id is not None

        replacement = db.get(
            UserSession,
            old_session.replaced_by_session_id,
        )

        assert replacement is not None

        assert (
            replacement.refresh_token_hash
            == hash_refresh_token(new_token)
        )

        assert replacement.refresh_token_hash != new_token

        assert replacement.revoked_at is None

        audit_events = db.query(AuditLog).all()

        assert len(audit_events) == 1

        audit_event = audit_events[0]

        assert audit_event.actor_user_id == user.id
        assert audit_event.category == "authentication"
        assert audit_event.action == "refresh"
        assert audit_event.outcome == "success"
        assert audit_event.target_type == "session"
        assert audit_event.target_id == str(old_session_id)
        assert audit_event.metadata_json == {
            "reason": "token_rotated"
        }

    print("PASS: Refresh API rotates cookie and records audit event")


# ============================================================
# 5. MISSING REFRESH COOKIE
# ============================================================

def test_refresh_rejects_missing_cookie(test_environment):

    client, SessionLocal = test_environment

    response = client.post(
        "/api/v1/auth/refresh",
    )

    assert response.status_code == 401

    assert response.json()["success"] is False

    with SessionLocal() as db:
        assert db.query(UserSession).count() == 0
        assert db.query(AuditLog).count() == 0

    print("PASS: Missing refresh cookie rejected without audit event")


# ============================================================
# 6. UNKNOWN REFRESH TOKEN
# ============================================================

def test_refresh_rejects_unknown_cookie(test_environment):

    client, SessionLocal = test_environment

    response = client.post(
        "/api/v1/auth/refresh",
        cookies={"refresh_token": "unknown-token"},
    )

    assert response.status_code == 401

    assert response.json()["success"] is False

    cookie_header = response.headers.get(
        "set-cookie",
        "",
    ).lower()

    assert "max-age=0" in cookie_header

    with SessionLocal() as db:
        assert db.query(UserSession).count() == 0
        assert db.query(AuditLog).count() == 0

    print("PASS: Unknown refresh cookie rejected and cleared without audit event")


# ============================================================
# 7. EXPIRED REFRESH TOKEN
# ============================================================

def test_refresh_rejects_expired_session(test_environment):

    client, SessionLocal = test_environment

    now = datetime.now(timezone.utc)

    with SessionLocal() as db:

        user = create_user(db)

        expired_session, expired_token = create_session(
            db,
            user.id,
            expires_at=now - timedelta(days=1),
        )

        expired_session_id = expired_session.id

    response = client.post(
        "/api/v1/auth/refresh",
        cookies={"refresh_token": expired_token},
    )

    assert response.status_code == 401

    assert "max-age=0" in response.headers.get(
        "set-cookie",
        "",
    ).lower()

    with SessionLocal() as db:

        expired_session = db.get(
            UserSession,
            expired_session_id,
        )

        assert expired_session is not None
        assert expired_session.revoked_at is not None

        audit_events = db.query(AuditLog).all()

        assert len(audit_events) == 1
        assert audit_events[0].actor_user_id == expired_session.user_id
        assert audit_events[0].category == "authentication"
        assert audit_events[0].action == "refresh"
        assert audit_events[0].outcome == "denied"
        assert audit_events[0].target_id == str(expired_session_id)
        assert audit_events[0].metadata_json == {
            "reason": "expired"
        }

    print("PASS: Expired refresh session rejected, revoked and audited")


# ============================================================
# 8. ROTATED TOKEN REPLAY
# ============================================================

def test_refresh_replay_revokes_all_active_sessions(
    test_environment,
):

    client, SessionLocal = test_environment

    with SessionLocal() as db:

        user = create_user(db)

        old_session, old_token = create_session(
            db,
            user.id,
        )
        old_session_id = old_session.id

    # First request: successful rotation.
    first_response = client.post(
        "/api/v1/auth/refresh",
        cookies={"refresh_token": old_token},
    )

    assert first_response.status_code == 200

    new_token = first_response.cookies.get("refresh_token")

    assert new_token
    assert new_token != old_token

    # Second request: reuse the old token.
    replay_response = client.post(
        "/api/v1/auth/refresh",
        cookies={"refresh_token": old_token},
    )

    assert replay_response.status_code == 401

    with SessionLocal() as db:

        sessions = (
            db.query(UserSession)
            .filter_by(user_id=user.id)
            .all()
        )

        assert len(sessions) == 2

        assert all(
            session.revoked_at is not None
            for session in sessions
        )

        audit_events = db.query(AuditLog).all()

        assert len(audit_events) == 2

        success_event = next(
            event for event in audit_events
            if event.action == "refresh"
        )

        reuse_event = next(
            event for event in audit_events
            if event.action == "refresh_token_reuse"
        )

        assert success_event.outcome == "success"
        assert success_event.target_id == str(old_session_id)
        assert success_event.metadata_json == {
            "reason": "token_rotated"
        }

        assert reuse_event.actor_user_id == user.id
        assert reuse_event.category == "security"
        assert reuse_event.outcome == "denied"
        assert reuse_event.target_id == str(old_session_id)
        assert reuse_event.metadata_json == {
            "reason": "rotated_token_reuse"
        }

    print("PASS: Refresh-token replay detected, revoked and audited")


# ============================================================
# 9. SUSPENDED ACCOUNT
# ============================================================

def test_refresh_rejects_suspended_account(
    test_environment,
):

    client, SessionLocal = test_environment

    with SessionLocal() as db:

        user = create_user(db)

        first_session, first_token = create_session(
            db,
            user.id,
        )

        second_session, second_token = create_session(
            db,
            user.id,
        )

        first_session_id = first_session.id
        second_session_id = second_session.id

        user.account_status = "suspended"
        db.commit()

    response = client.post(
        "/api/v1/auth/refresh",
        cookies={"refresh_token": first_token},
    )

    assert response.status_code == 401

    with SessionLocal() as db:

        first_session = db.get(
            UserSession,
            first_session_id,
        )

        second_session = db.get(
            UserSession,
            second_session_id,
        )

        assert first_session is not None
        assert second_session is not None

        assert first_session.revoked_at is not None
        assert second_session.revoked_at is not None

        audit_events = db.query(AuditLog).all()

        assert len(audit_events) == 1
        assert audit_events[0].actor_user_id == user.id
        assert audit_events[0].category == "authentication"
        assert audit_events[0].action == "refresh"
        assert audit_events[0].outcome == "denied"
        assert audit_events[0].target_id == str(first_session_id)
        assert audit_events[0].metadata_json == {
            "reason": "account_unavailable"
        }

    print("PASS: Suspended account refresh rejected and audited")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))



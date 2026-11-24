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
    hash_refresh_token,
)


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


def create_user(db):

    user = User(
        role="student",
        account_status="active",
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


def create_session(db, user_id, raw_token=None):

    now = datetime.now(timezone.utc)

    if raw_token is None:
        raw_token = create_refresh_token()

    user_session = UserSession(
        user_id=user_id,
        refresh_token_hash=hash_refresh_token(raw_token),
        created_at=now,
        expires_at=now + timedelta(days=7),
    )

    db.add(user_session)
    db.commit()
    db.refresh(user_session)

    return user_session, raw_token


def test_successful_logout_revokes_session_and_clears_cookie(
    test_environment,
):

    client, SessionLocal = test_environment

    with SessionLocal() as db:

        user = create_user(db)

        user_session, raw_token = create_session(
            db,
            user.id,
        )

        session_id = user_session.id

    response = client.post(
        "/api/v1/auth/logout",
        cookies={"refresh_token": raw_token},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["message"] == "Logout successful."

    cookie_header = response.headers.get(
        "set-cookie",
        "",
    ).lower()

    assert "refresh_token=" in cookie_header
    assert "max-age=0" in cookie_header
    assert "httponly" in cookie_header
    assert "path=/api/v1/auth" in cookie_header

    with SessionLocal() as db:

        stored_session = db.get(
            UserSession,
            session_id,
        )

        assert stored_session is not None
        assert stored_session.revoked_at is not None
        assert stored_session.replaced_by_session_id is None

        audit_events = db.query(AuditLog).all()

        assert len(audit_events) == 1

        audit_event = audit_events[0]

        assert audit_event.actor_user_id == user.id
        assert audit_event.category == "authentication"
        assert audit_event.action == "logout"
        assert audit_event.outcome == "success"
        assert audit_event.target_type == "session"
        assert audit_event.target_id == str(session_id)
        assert audit_event.metadata_json == {
            "reason": "session_revoked"
        }

    print("PASS: Logout API revokes session, clears cookie and audits")


def test_logout_does_not_revoke_other_device_sessions(
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

    response = client.post(
        "/api/v1/auth/logout",
        cookies={"refresh_token": first_token},
    )

    assert response.status_code == 200

    with SessionLocal() as db:

        first_session = db.get(
            UserSession,
            first_session_id,
        )

        second_session = db.get(
            UserSession,
            second_session_id,
        )

        assert first_session.revoked_at is not None
        assert second_session.revoked_at is None

        audit_events = db.query(AuditLog).all()

        assert len(audit_events) == 1
        assert audit_events[0].actor_user_id == user.id
        assert audit_events[0].action == "logout"
        assert audit_events[0].outcome == "success"
        assert audit_events[0].target_id == str(first_session_id)

    print("PASS: Logout API preserves other sessions and audits target session")


def test_logout_with_missing_cookie_succeeds_and_clears_cookie(
    test_environment,
):

    client, SessionLocal = test_environment

    response = client.post(
        "/api/v1/auth/logout",
    )

    assert response.status_code == 200
    assert response.json()["success"] is True

    cookie_header = response.headers.get(
        "set-cookie",
        "",
    ).lower()

    assert "max-age=0" in cookie_header

    with SessionLocal() as db:
        assert db.query(UserSession).count() == 0
        assert db.query(AuditLog).count() == 0

    print("PASS: Missing-cookie logout handled safely without audit event")


def test_logout_with_unknown_cookie_succeeds(
    test_environment,
):

    client, SessionLocal = test_environment

    response = client.post(
        "/api/v1/auth/logout",
        cookies={"refresh_token": "unknown-refresh-token"},
    )

    assert response.status_code == 200
    assert response.json()["success"] is True

    cookie_header = response.headers.get(
        "set-cookie",
        "",
    ).lower()

    assert "max-age=0" in cookie_header

    with SessionLocal() as db:
        assert db.query(UserSession).count() == 0
        assert db.query(AuditLog).count() == 0

    print("PASS: Unknown-cookie logout handled safely without audit event")


def test_repeated_logout_succeeds_without_reactivating_session(
    test_environment,
):

    client, SessionLocal = test_environment

    with SessionLocal() as db:

        user = create_user(db)

        user_session, raw_token = create_session(
            db,
            user.id,
        )

        session_id = user_session.id

    first_response = client.post(
        "/api/v1/auth/logout",
        cookies={"refresh_token": raw_token},
    )

    second_response = client.post(
        "/api/v1/auth/logout",
        cookies={"refresh_token": raw_token},
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    with SessionLocal() as db:

        stored_session = db.get(
            UserSession,
            session_id,
        )

        assert stored_session.revoked_at is not None

        audit_events = db.query(AuditLog).all()

        assert len(audit_events) == 1
        assert audit_events[0].actor_user_id == user.id
        assert audit_events[0].action == "logout"
        assert audit_events[0].outcome == "success"
        assert audit_events[0].target_id == str(session_id)
        assert audit_events[0].metadata_json == {
            "reason": "session_revoked"
        }

    print("PASS: Repeated logout remains safe without duplicate audit events")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))

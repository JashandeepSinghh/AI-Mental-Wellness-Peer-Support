from datetime import datetime, timedelta, timezone

import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.database.base import Base
import backend.app.database.models

from backend.app.database.models import AuditLog, User, UserSession

from backend.app.auth.tokens import (
    create_refresh_token,
    hash_refresh_token,
)

from backend.app.services.refresh_token_service import (
    revoke_user_session,
    refresh_user_session,
    RefreshTokenError,
)


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


def test_successful_logout_revokes_current_session(db_session):

    user = create_user(db_session)

    user_session, raw_token = create_session(
        db_session,
        user.id,
    )

    session_id = user_session.id

    result = revoke_user_session(
        db_session,
        raw_token,
    )

    assert result is True

    revoked_session = db_session.get(
        UserSession,
        session_id,
    )

    assert revoked_session is not None
    assert revoked_session.revoked_at is not None
    assert revoked_session.replaced_by_session_id is None

    audit_events = db_session.query(AuditLog).all()

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

    print("PASS: Logout revokes session and records audit event")


def test_logout_does_not_revoke_other_device_sessions(db_session):

    user = create_user(db_session)

    first_session, first_token = create_session(
        db_session,
        user.id,
    )

    second_session, second_token = create_session(
        db_session,
        user.id,
    )

    first_session_id = first_session.id
    second_session_id = second_session.id

    result = revoke_user_session(
        db_session,
        first_token,
    )

    assert result is True

    first_session = db_session.get(
        UserSession,
        first_session_id,
    )

    second_session = db_session.get(
        UserSession,
        second_session_id,
    )

    assert first_session.revoked_at is not None
    assert second_session.revoked_at is None

    audit_events = db_session.query(AuditLog).all()

    assert len(audit_events) == 1
    assert audit_events[0].target_id == str(first_session_id)
    assert audit_events[0].actor_user_id == user.id

    print("PASS: Other device session remains active")


@pytest.mark.parametrize(
    "invalid_token",
    [
        "",
        None,
        12345,
        "unknown-refresh-token",
    ],
)
def test_invalid_logout_token_returns_false(
    db_session,
    invalid_token,
):

    result = revoke_user_session(
        db_session,
        invalid_token,
    )

    assert result is False
    assert db_session.query(UserSession).count() == 0
    assert db_session.query(AuditLog).count() == 0

    print("PASS: Invalid logout token handled safely")


def test_repeated_logout_does_not_change_revoked_session(db_session):

    user = create_user(db_session)

    user_session, raw_token = create_session(
        db_session,
        user.id,
    )

    first_result = revoke_user_session(
        db_session,
        raw_token,
    )

    assert first_result is True

    second_result = revoke_user_session(
        db_session,
        raw_token,
    )

    assert second_result is False

    assert user_session.revoked_at is not None
    assert user_session.replaced_by_session_id is None

    audit_events = db_session.query(AuditLog).all()

    assert len(audit_events) == 1
    assert audit_events[0].action == "logout"
    assert audit_events[0].outcome == "success"

    print("PASS: Repeated logout creates no duplicate audit event")


def test_logout_token_cannot_be_used_for_refresh(db_session):

    user = create_user(db_session)

    user_session, raw_token = create_session(
        db_session,
        user.id,
    )

    result = revoke_user_session(
        db_session,
        raw_token,
    )

    assert result is True

    with pytest.raises(RefreshTokenError):
        refresh_user_session(
            db_session,
            raw_token,
        )

    active_sessions = (
        db_session.query(UserSession)
        .filter(
            UserSession.user_id == user.id,
            UserSession.revoked_at.is_(None),
        )
        .count()
    )

    assert active_sessions == 0

    print("PASS: Logged-out token cannot refresh")


def test_logout_rolls_back_if_commit_fails(
    db_session,
    monkeypatch,
):

    user = create_user(db_session)

    user_session, raw_token = create_session(
        db_session,
        user.id,
    )

    session_id = user_session.id

    def fail_commit():
        raise RuntimeError("Simulated database commit failure.")

    monkeypatch.setattr(
        db_session,
        "commit",
        fail_commit,
    )

    with pytest.raises(RuntimeError):
        revoke_user_session(
            db_session,
            raw_token,
        )

    monkeypatch.undo()

    stored_session = db_session.get(
        UserSession,
        session_id,
    )

    assert stored_session is not None
    assert stored_session.revoked_at is None
    assert db_session.query(AuditLog).count() == 0

    print("PASS: Logout and audit transaction rollback verified")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))

from datetime import datetime, timedelta, timezone

import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.database.base import Base
import backend.app.database.models

from backend.app.database.models import (
    AuditLog,
    User,
    UserSession,
)

from backend.app.auth.tokens import (
    create_refresh_token,
    hash_refresh_token,
)

from backend.app.services.refresh_token_service import (
    refresh_user_session,
    RefreshTokenError,
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
# 2. TEST USER HELPER
# ============================================================

def create_user(
    db,
    account_status="active",
):

    user = User(
        role="student",
        account_status=account_status,
    )

    if account_status == "deleted":
        user.deleted_at = datetime.now(timezone.utc)

    db.add(user)
    db.flush()

    return user


# ============================================================
# 3. TEST SESSION HELPER
# ============================================================

def create_session(
    db,
    user_id,
    raw_token=None,
    expires_at=None,
    created_at=None,
    revoked_at=None,
):

    now = datetime.now(timezone.utc)

    if raw_token is None:
        raw_token = create_refresh_token()

    if created_at is None:
        created_at = now

    if expires_at is None:
        expires_at = now + timedelta(days=7)

    user_session = UserSession(
        user_id=user_id,
        refresh_token_hash=hash_refresh_token(raw_token),
        created_at=created_at,
        expires_at=expires_at,
        revoked_at=revoked_at,
    )

    db.add(user_session)
    db.commit()
    db.refresh(user_session)

    return user_session, raw_token


# ============================================================
# 4. SUCCESSFUL TOKEN ROTATION
# ============================================================

def test_successful_refresh_rotates_session(db_session):

    user = create_user(db_session)

    old_session, old_token = create_session(
        db_session,
        user.id,
    )

    result = refresh_user_session(
        db_session,
        old_token,
    )

    assert result.user_id == user.id
    assert result.role == "student"

    assert result.access_token
    assert result.refresh_token

    assert result.refresh_token != old_token

    assert result.refresh_token not in repr(result)
    assert result.access_token not in repr(result)

    db_session.refresh(old_session)

    assert old_session.revoked_at is not None
    assert old_session.last_used_at is not None
    assert old_session.replaced_by_session_id is not None

    replacement = db_session.get(
        UserSession,
        old_session.replaced_by_session_id,
    )

    assert replacement is not None
    assert replacement.user_id == user.id

    assert (
        replacement.refresh_token_hash
        == hash_refresh_token(result.refresh_token)
    )

    assert replacement.refresh_token_hash != result.refresh_token

    assert replacement.revoked_at is None

    active_sessions = (
        db_session.query(UserSession)
        .filter(
            UserSession.user_id == user.id,
            UserSession.revoked_at.is_(None),
        )
        .count()
    )

    assert active_sessions == 1

    audit_events = db_session.query(AuditLog).all()

    assert len(audit_events) == 1

    audit_event = audit_events[0]

    assert audit_event.actor_user_id == user.id
    assert audit_event.category == "authentication"
    assert audit_event.action == "refresh"
    assert audit_event.outcome == "success"
    assert audit_event.target_type == "session"
    assert audit_event.target_id == str(old_session.id)
    assert audit_event.metadata_json == {
        "reason": "token_rotated"
    }

    print("PASS: Refresh token successfully rotated and audited")


# ============================================================
# 5. REPLAY DETECTION
# ============================================================

def test_reused_rotated_token_revokes_active_sessions(
    db_session,
):

    user = create_user(db_session)

    old_session, old_token = create_session(
        db_session,
        user.id,
    )

    # First use: successful rotation.
    result = refresh_user_session(
        db_session,
        old_token,
    )

    assert result.refresh_token

    # Second use: attempt to reuse the previous token.
    with pytest.raises(RefreshTokenError):

        refresh_user_session(
            db_session,
            old_token,
        )

    db_session.refresh(old_session)

    assert old_session.revoked_at is not None

    # Reuse detection must revoke the replacement session too.
    sessions = (
        db_session.query(UserSession)
        .filter_by(user_id=user.id)
        .all()
    )

    assert len(sessions) == 2

    assert all(
        session.revoked_at is not None
        for session in sessions
    )

    audit_events = db_session.query(AuditLog).all()

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
    assert success_event.metadata_json == {
        "reason": "token_rotated"
    }

    assert reuse_event.category == "security"
    assert reuse_event.outcome == "denied"
    assert reuse_event.actor_user_id == user.id
    assert reuse_event.target_id == str(old_session.id)
    assert reuse_event.metadata_json == {
        "reason": "rotated_token_reuse"
    }

    print("PASS: Rotated-token reuse detected and audited")


# ============================================================
# 6. UNKNOWN REFRESH TOKEN
# ============================================================

def test_unknown_refresh_token_is_rejected(db_session):

    user = create_user(db_session)

    existing_session, existing_token = create_session(
        db_session,
        user.id,
    )

    with pytest.raises(RefreshTokenError):

        refresh_user_session(
            db_session,
            "unknown-refresh-token",
        )

    db_session.refresh(existing_session)

    assert existing_session.revoked_at is None

    assert db_session.query(UserSession).count() == 1
    assert db_session.query(AuditLog).count() == 0

    print("PASS: Unknown refresh token rejected without audit noise")


# ============================================================
# 7. EXPIRED REFRESH TOKEN
# ============================================================

def test_expired_refresh_token_is_rejected(db_session):

    user = create_user(db_session)

    now = datetime.now(timezone.utc)

    expired_session, expired_token = create_session(
        db_session,
        user.id,
        created_at=now - timedelta(days=2),
        expires_at=now - timedelta(days=1),
    )

    with pytest.raises(RefreshTokenError):

        refresh_user_session(
            db_session,
            expired_token,
        )

    db_session.refresh(expired_session)

    assert expired_session.revoked_at is not None

    assert db_session.query(UserSession).count() == 1

    audit_events = db_session.query(AuditLog).all()

    assert len(audit_events) == 1
    assert audit_events[0].action == "refresh"
    assert audit_events[0].outcome == "denied"
    assert audit_events[0].actor_user_id == user.id
    assert audit_events[0].target_id == str(expired_session.id)
    assert audit_events[0].metadata_json == {
        "reason": "expired"
    }

    print("PASS: Expired refresh token rejected and audited")


# ============================================================
# 8. PREVIOUSLY REVOKED SESSION
# ============================================================

def test_revoked_session_without_replacement_is_rejected(
    db_session,
):

    user = create_user(db_session)

    now = datetime.now(timezone.utc)

    revoked_session, revoked_token = create_session(
        db_session,
        user.id,
        revoked_at=now,
    )

    active_session, active_token = create_session(
        db_session,
        user.id,
    )

    with pytest.raises(RefreshTokenError):

        refresh_user_session(
            db_session,
            revoked_token,
        )

    db_session.refresh(active_session)

    assert active_session.revoked_at is None

    assert (
        db_session.query(UserSession)
        .filter(
            UserSession.user_id == user.id,
            UserSession.revoked_at.is_(None),
        )
        .count()
        == 1
    )

    print("PASS: Previously revoked session rejected")


# ============================================================
# 9. UNAVAILABLE ACCOUNT
# ============================================================

def test_unavailable_account_revokes_active_sessions(
    db_session,
):

    user = create_user(db_session)

    first_session, first_token = create_session(
        db_session,
        user.id,
    )

    second_session, second_token = create_session(
        db_session,
        user.id,
    )

    user.account_status = "suspended"
    db_session.commit()

    with pytest.raises(RefreshTokenError):

        refresh_user_session(
            db_session,
            first_token,
        )

    db_session.refresh(first_session)
    db_session.refresh(second_session)

    assert first_session.revoked_at is not None
    assert second_session.revoked_at is not None

    audit_events = db_session.query(AuditLog).all()

    assert len(audit_events) == 1
    assert audit_events[0].action == "refresh"
    assert audit_events[0].outcome == "denied"
    assert audit_events[0].actor_user_id == user.id
    assert audit_events[0].target_id == str(first_session.id)
    assert audit_events[0].metadata_json == {
        "reason": "account_unavailable"
    }

    print("PASS: Unavailable account sessions revoked and audited")


# ============================================================
# 10. INVALID REFRESH TOKEN INPUT
# ============================================================

@pytest.mark.parametrize(
    "invalid_token",
    [
        "",
        None,
        12345,
    ],
)
def test_invalid_refresh_token_input_is_rejected(
    db_session,
    invalid_token,
):

    with pytest.raises(RefreshTokenError):

        refresh_user_session(
            db_session,
            invalid_token,
        )

    assert db_session.query(UserSession).count() == 0
    assert db_session.query(AuditLog).count() == 0

    print("PASS: Invalid refresh token input rejected")


# ============================================================
# 11. DATABASE FAILURE ROLLBACK
# ============================================================

def test_refresh_rolls_back_if_commit_fails(
    db_session,
    monkeypatch,
):

    user = create_user(db_session)

    old_session, old_token = create_session(
        db_session,
        user.id,
    )

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

        refresh_user_session(
            db_session,
            old_token,
        )

    db_session.refresh(old_session)

    assert old_session.revoked_at is None
    assert old_session.replaced_by_session_id is None

    assert db_session.query(UserSession).count() == 1
    assert db_session.query(AuditLog).count() == 0

    print("PASS: Failed refresh and audit transaction rolled back")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))

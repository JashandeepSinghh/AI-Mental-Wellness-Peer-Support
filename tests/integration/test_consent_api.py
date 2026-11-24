from datetime import datetime, timezone

import pytest

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.main import app
from backend.app.database.base import Base
import backend.app.database.models

from backend.app.database.session import get_db
from backend.app.database.models import User, UserConsent

from backend.app.auth.tokens import create_access_token


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


def create_test_user(db):

    user = User(
        role="student",
        account_status="active",
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


def get_auth_headers(user_id):

    token = create_access_token(user_id)

    return {
        "Authorization": f"Bearer {token}",
    }


def test_record_consent_with_valid_token(test_environment):

    client, SessionLocal = test_environment

    with SessionLocal() as db:
        user = create_test_user(db)
        user_id = user.id

    response = client.post(
        "/api/v1/consent",
        headers=get_auth_headers(user_id),
        json={
            "purpose": "conversation_storage",
            "granted": True,
            "policy_version": "1.0",
        },
    )

    assert response.status_code == 201

    data = response.json()

    assert data["success"] is True
    assert data["consent"]["purpose"] == "conversation_storage"
    assert data["consent"]["granted"] is True
    assert data["consent"]["policy_version"] == "1.0"
    assert data["consent"]["source"] == "api"

    with SessionLocal() as db:
        assert db.query(UserConsent).count() == 1


def test_retrieve_current_consent_decisions(test_environment):

    client, SessionLocal = test_environment

    with SessionLocal() as db:

        user = create_test_user(db)

        db.add_all([
            UserConsent(
                user_id=user.id,
                purpose="conversation_storage",
                granted=True,
                policy_version="1.0",
                source="api",
            ),
            UserConsent(
                user_id=user.id,
                purpose="analytics",
                granted=False,
                policy_version="1.0",
                source="api",
            ),
        ])

        db.commit()

        user_id = user.id

    response = client.get(
        "/api/v1/consent",
        headers=get_auth_headers(user_id),
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True

    assert (
        data["consents"]["conversation_storage"]["granted"]
        is True
    )

    assert (
        data["consents"]["analytics"]["granted"]
        is False
    )


def test_missing_token_is_rejected(test_environment):

    client, SessionLocal = test_environment

    response = client.get("/api/v1/consent")

    assert response.status_code == 401


def test_invalid_token_is_rejected(test_environment):

    client, SessionLocal = test_environment

    response = client.get(
        "/api/v1/consent",
        headers={
            "Authorization": "Bearer invalid-token",
        },
    )

    assert response.status_code == 401


def test_unsupported_consent_purpose_is_rejected(test_environment):

    client, SessionLocal = test_environment

    with SessionLocal() as db:
        user = create_test_user(db)
        user_id = user.id

    response = client.post(
        "/api/v1/consent",
        headers=get_auth_headers(user_id),
        json={
            "purpose": "referral",
            "granted": True,
            "policy_version": "1.0",
        },
    )

    assert response.status_code == 422


def test_client_cannot_supply_user_id(test_environment):

    client, SessionLocal = test_environment

    with SessionLocal() as db:
        user = create_test_user(db)
        user_id = user.id

    response = client.post(
        "/api/v1/consent",
        headers=get_auth_headers(user_id),
        json={
            "user_id": "00000000-0000-0000-0000-000000000001",
            "purpose": "analytics",
            "granted": True,
            "policy_version": "1.0",
        },
    )

    assert response.status_code == 422

    with SessionLocal() as db:
        assert db.query(UserConsent).count() == 0


def test_withdrawal_preserves_previous_consent(test_environment):

    client, SessionLocal = test_environment

    with SessionLocal() as db:
        user = create_test_user(db)
        user_id = user.id

    headers = get_auth_headers(user_id)

    first_response = client.post(
        "/api/v1/consent",
        headers=headers,
        json={
            "purpose": "analytics",
            "granted": True,
            "policy_version": "1.0",
        },
    )

    second_response = client.post(
        "/api/v1/consent",
        headers=headers,
        json={
            "purpose": "analytics",
            "granted": False,
            "policy_version": "1.0",
        },
    )

    assert first_response.status_code == 201
    assert second_response.status_code == 201

    assert (
        first_response.json()["consent"]["id"]
        != second_response.json()["consent"]["id"]
    )

    with SessionLocal() as db:

        records = (
            db.query(UserConsent)
            .filter(UserConsent.user_id == user_id)
            .all()
        )

        assert len(records) == 2
        assert sorted(
            record.granted for record in records
        ) == [False, True]


def test_user_cannot_retrieve_another_users_consent(test_environment):

    client, SessionLocal = test_environment

    with SessionLocal() as db:

        first_user = create_test_user(db)
        second_user = create_test_user(db)

        db.add(
            UserConsent(
                user_id=first_user.id,
                purpose="analytics",
                granted=True,
                policy_version="1.0",
                source="api",
            )
        )

        db.commit()

        second_user_id = second_user.id

    response = client.get(
        "/api/v1/consent",
        headers=get_auth_headers(second_user_id),
    )

    assert response.status_code == 200

    data = response.json()

    assert data["consents"]["analytics"] is None
    assert data["consents"]["conversation_storage"] is None


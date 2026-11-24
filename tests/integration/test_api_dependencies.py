import pytest

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.api.dependencies import get_current_user
from backend.app.auth.tokens import create_access_token
from backend.app.database.base import Base
from backend.app.database.models import User
import backend.app.database.models
from backend.app.database.session import get_db


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

    app = FastAPI()

    app.dependency_overrides[get_db] = override_get_db

    @app.get("/protected")
    def protected_route(
        current_user: User = Depends(get_current_user),
    ):

        return {
            "success": True,
            "user_id": str(current_user.id),
            "role": current_user.role,
        }

    client = TestClient(app)

    try:
        yield client, TestingSessionLocal

    finally:
        client.close()
        app.dependency_overrides.clear()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def create_test_user(
    db,
    account_status="active",
):

    user = User(
        role="student",
        account_status=account_status,
    )

    if account_status == "deleted":
        from datetime import datetime, timezone
        user.deleted_at = datetime.now(timezone.utc)

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


def test_valid_access_token_authenticates_active_user(
    test_environment,
):

    client, SessionLocal = test_environment

    with SessionLocal() as db:
        user = create_test_user(db)
        user_id = str(user.id)

    token = create_access_token(user_id)

    response = client.get(
        "/protected",
        headers={
            "Authorization": f"Bearer {token}",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["user_id"] == user_id
    assert data["role"] == "student"


def test_missing_access_token_is_rejected(
    test_environment,
):

    client, SessionLocal = test_environment

    response = client.get("/protected")

    assert response.status_code == 401


def test_invalid_access_token_is_rejected(
    test_environment,
):

    client, SessionLocal = test_environment

    response = client.get(
        "/protected",
        headers={
            "Authorization": "Bearer invalid-token",
        },
    )

    assert response.status_code == 401


def test_suspended_account_is_rejected(
    test_environment,
):

    client, SessionLocal = test_environment

    with SessionLocal() as db:
        user = create_test_user(
            db,
            account_status="suspended",
        )

    token = create_access_token(user.id)

    response = client.get(
        "/protected",
        headers={
            "Authorization": f"Bearer {token}",
        },
    )

    assert response.status_code == 401


def test_nonexistent_user_is_rejected(
    test_environment,
):

    client, SessionLocal = test_environment

    from uuid import uuid4

    token = create_access_token(uuid4())

    response = client.get(
        "/protected",
        headers={
            "Authorization": f"Bearer {token}",
        },
    )

    assert response.status_code == 401

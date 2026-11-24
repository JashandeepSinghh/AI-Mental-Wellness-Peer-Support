"""
Consent Management API Routes
==============================

Provides authenticated endpoints for recording and retrieving
general privacy consent decisions.

Security:
- Requires a valid JWT access token.
- Uses the authenticated user's database identity.
- Never accepts a client-supplied user ID.
- Preserves consent history through append-only records.
- Handles referral consent separately.
"""

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from backend.app.api.dependencies import get_current_user
from backend.app.database.models import User
from backend.app.database.session import get_db

from backend.app.schemas.consent import ConsentDecisionRequest

from backend.app.services.consent_service import (
    record_consent_decision,
    get_current_consent_decisions,
)


# ============================================================
# 1. ROUTER CONFIGURATION
# ============================================================

router = APIRouter(
    prefix="/consent",
    tags=["Consent Management"],
)


# ============================================================
# 2. RECORD CONSENT DECISION
# ============================================================

@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Record a consent decision",
    description=(
        "Records a new conversation-storage or analytics "
        "consent decision for the authenticated user."
    ),
)
async def create_consent_decision(
    consent_request: ConsentDecisionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):

    """
    Record a new consent decision.

    Every request creates a separate consent event.
    Previous decisions are never overwritten.
    """

    try:

        consent = await run_in_threadpool(
            record_consent_decision,
            db,
            current_user.id,
            consent_request.purpose,
            consent_request.granted,
            consent_request.policy_version,
        )

    except ValueError as error:

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error

    return {
        "success": True,
        "message": "Consent decision recorded successfully.",
        "consent": {
            "id": str(consent.id),
            "purpose": consent.purpose,
            "granted": consent.granted,
            "policy_version": consent.policy_version,
            "source": consent.source,
            "recorded_at": consent.recorded_at,
        },
    }


# ============================================================
# 3. RETRIEVE CURRENT CONSENT DECISIONS
# ============================================================

@router.get(
    "",
    status_code=status.HTTP_200_OK,
    summary="Retrieve current consent decisions",
    description=(
        "Returns the latest conversation-storage and "
        "analytics consent decisions for the authenticated user."
    ),
)
async def retrieve_current_consent_decisions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):

    """
    Retrieve the latest consent decision for each purpose.

    If no decision exists for a purpose, its value is null.
    Previous consent events remain stored in the database.
    """

    decisions = await run_in_threadpool(
        get_current_consent_decisions,
        db,
        current_user.id,
    )

    # --------------------------------------------------------
    # A. Build a consistent response for both purposes.
    # --------------------------------------------------------

    response_decisions = {}

    for purpose in (
        "conversation_storage",
        "analytics",
    ):

        consent = decisions.get(purpose)

        if consent is None:

            response_decisions[purpose] = None

        else:

            response_decisions[purpose] = {
                "id": str(consent.id),
                "granted": consent.granted,
                "policy_version": consent.policy_version,
                "source": consent.source,
                "recorded_at": consent.recorded_at,
            }

    # --------------------------------------------------------
    # B. Return the authenticated user's consent decisions.
    # --------------------------------------------------------

    return {
        "success": True,
        "consents": response_decisions,
    }

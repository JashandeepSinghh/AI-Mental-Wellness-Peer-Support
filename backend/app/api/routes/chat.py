"""
Chat API Routes
===============

Receives student messages and connects them
to the unified AI inference pipeline.

Privacy principles:
- Messages are not stored.
- Raw messages are not returned.
- Only necessary request information is processed.
- Inference outputs are linguistic signals,
  not clinical diagnoses.
"""

import logging

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    status,
)

from starlette.concurrency import run_in_threadpool

from backend.app.schemas.chat import ChatRequest

from backend.app.inference.inference_pipeline import (
    InferencePipelineError,
)

from backend.app.services.supportive_response_service import (
    generate_supportive_guidance,
)


logger = logging.getLogger(__name__)


# ============================================================
# 1. ROUTER CONFIGURATION
# ============================================================

router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)


# ============================================================
# 2. CHAT MESSAGE ENDPOINT
# ============================================================

@router.post(
    "/messages",
    status_code=status.HTTP_200_OK,
    summary="Analyze a student chat message",
    description=(
        "Accepts a validated student message and "
        "runs the unified AI inference pipeline."
    )
)
async def receive_chat_message(
    request: Request,
    chat_request: ChatRequest,
):

    """
    Process a student message through the inference pipeline.

    The original message is neither stored nor returned.
    """

    # --------------------------------------------------------
    # A. Retrieve shared inference pipeline
    # --------------------------------------------------------

    inference_pipeline = getattr(
        request.app.state,
        "inference_pipeline",
        None,
    )

    if inference_pipeline is None:

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service is not available.",
        )

    # --------------------------------------------------------
    # B. Execute inference outside the event loop
    # --------------------------------------------------------

    try:

        analysis = await run_in_threadpool(
            inference_pipeline.analyze_text,
            chat_request.message,
        )

    except ValueError as error:

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error

    except InferencePipelineError as error:

        logger.exception(
            "Inference pipeline execution failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Message analysis is temporarily unavailable.",
        ) from error

    except Exception as error:

        logger.exception(
            "Unexpected chat inference error."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during message analysis.",
        ) from error

    # --------------------------------------------------------
    # C. Generate supportive wellness response and safety guidance
    # --------------------------------------------------------

    guidance = generate_supportive_guidance(analysis)

    # --------------------------------------------------------
    # D. Return structured response and guidance
    # --------------------------------------------------------

    return {

        "success": True,

        "status": "analysis_completed",

        "message": "Message analysis completed successfully.",

        "conversation_id": (
            str(chat_request.conversation_id)
            if chat_request.conversation_id
            else None
        ),

        "message_length": len(chat_request.message),

        "analysis": analysis,

        "supportive_response": guidance["supportive_response"],

        "detected_emotions": guidance["detected_emotions"],

        "sentiment": guidance["sentiment"],

        "stress_detected": guidance["stress_detected"],

        "loneliness_detected": guidance["loneliness_detected"],

        "wellness_resources": guidance["wellness_resources"],

        "safety_guidance": guidance["safety_guidance"],

    }
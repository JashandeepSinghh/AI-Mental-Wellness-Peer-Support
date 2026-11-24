
"""
Centralized API Exception Handlers
===================================

Provides consistent error responses for:
1. HTTP exceptions
2. Request validation errors
3. Unexpected application exceptions

Privacy:
- Never return submitted request values.
- Never expose raw exception messages.
- Never expose stack traces to API clients.
- Never log student conversation content.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

from backend.app.schemas.error import APIErrorResponse
from backend.app.core.logging_config import (
    get_logger,
    log_exception,
)


# ============================================================
# 1. LOGGER
# ============================================================

logger = get_logger("exceptions")


# ============================================================
# 2. HTTP STATUS MESSAGE MAPPING
# ============================================================

HTTP_ERROR_MESSAGES = {

    400: (
        "BAD_REQUEST",
        "The request could not be processed."
    ),

    401: (
        "UNAUTHORIZED",
        "Authentication is required."
    ),

    403: (
        "FORBIDDEN",
        "You do not have permission to access this resource."
    ),

    404: (
        "NOT_FOUND",
        "The requested resource was not found."
    ),

    405: (
        "METHOD_NOT_ALLOWED",
        "The requested HTTP method is not allowed."
    ),

    409: (
        "CONFLICT",
        "The request conflicts with the current resource state."
    ),

    413: (
        "PAYLOAD_TOO_LARGE",
        "The submitted request is too large."
    ),

    415: (
        "UNSUPPORTED_MEDIA_TYPE",
        "The submitted media type is not supported."
    ),

    422: (
        "UNPROCESSABLE_ENTITY",
        "The request contains invalid data."
    ),

    429: (
        "RATE_LIMIT_EXCEEDED",
        "Too many requests. Please try again later."
    ),

}


# ============================================================
# 3. HTTP EXCEPTION HANDLER
# ============================================================

async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
):

    """
    Handles HTTP exceptions raised by application routes.

    Raw exception details are deliberately not returned.
    """

    status_code = exc.status_code

    error_code, message = HTTP_ERROR_MESSAGES.get(

        status_code,

        (
            "HTTP_ERROR",
            "The request could not be completed."
        )

    )

    logger.warning(

        "HTTP exception | method=%s | status=%s",

        request.method,

        status_code,

    )

    response = APIErrorResponse.create(

        code=error_code,

        message=message,

    )

    return JSONResponse(

        status_code=status_code,

        content=response.model_dump(mode="json"),

        headers=exc.headers,

    )


# ============================================================
# 4. REQUEST VALIDATION HANDLER
# ============================================================

async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):

    """
    Handles invalid request data.

    Only structural validation information is returned.

    Submitted values and raw request bodies are excluded.
    """

    safe_details = []

    for error in exc.errors():

        safe_details.append({

            "location": list(error.get("loc", [])),

            "message": "Invalid or missing request value.",

            "error_type": str(
                error.get("type", "validation_error")
            ),

        })

    logger.warning(

        "Request validation failed | method=%s | error_count=%s",

        request.method,

        len(safe_details),

    )

    response = APIErrorResponse.create(

        code="VALIDATION_ERROR",

        message="Request validation failed.",

        details=safe_details,

    )

    return JSONResponse(

        status_code=422,

        content=response.model_dump(mode="json"),

    )


# ============================================================
# 5. UNEXPECTED EXCEPTION HANDLER
# ============================================================

async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
):

    """
    Handles unexpected application errors.

    Internal exception information is not exposed to clients.
    """

    log_exception(

        logger,

        "Unexpected application exception",

        exc,

    )

    response = APIErrorResponse.create(

        code="INTERNAL_SERVER_ERROR",

        message=(
            "An unexpected server error occurred. "
            "Please try again later."
        ),

    )

    return JSONResponse(

        status_code=500,

        content=response.model_dump(mode="json"),

    )


# ============================================================
# 6. REGISTER GLOBAL EXCEPTION HANDLERS
# ============================================================

def register_exception_handlers(app: FastAPI):

    """
    Registers all centralized exception handlers.

    Handler registration order is not significant;
    FastAPI selects handlers based on exception type.
    """

    app.add_exception_handler(

        StarletteHTTPException,

        http_exception_handler,

    )

    app.add_exception_handler(

        RequestValidationError,

        validation_exception_handler,

    )

    app.add_exception_handler(

        Exception,

        unexpected_exception_handler,

    )

    logger.info(
        "Global API exception handlers registered."
    )

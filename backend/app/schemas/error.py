
"""
Standardized API Error Schemas
================================

Defines consistent error-response structures
for all backend API endpoints.
"""

from datetime import datetime, timezone

from typing import Any

from pydantic import BaseModel, Field


# ============================================================
# 1. ERROR DETAIL SCHEMA
# ============================================================

class ErrorDetail(BaseModel):

    """
    Represents an individual validation or application error.

    Sensitive submitted input values are intentionally excluded.
    """

    location: list[str | int] = Field(
        default_factory=list,
        description="Location of the error within the request."
    )

    message: str = Field(
        description="Human-readable error description."
    )

    error_type: str = Field(
        description="Machine-readable error category."
    )


# ============================================================
# 2. ERROR INFORMATION SCHEMA
# ============================================================

class ErrorInformation(BaseModel):

    """
    Contains the primary error information.
    """

    code: str = Field(
        description="Application error code."
    )

    message: str = Field(
        description="Human-readable error message."
    )

    details: list[ErrorDetail] = Field(
        default_factory=list,
        description="Additional error details."
    )


# ============================================================
# 3. STANDARD API ERROR RESPONSE
# ============================================================

class APIErrorResponse(BaseModel):

    """
    Standard response returned whenever an API request fails.
    """

    success: bool = Field(
        default=False,
        description="Indicates whether the request succeeded."
    )

    error: ErrorInformation

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the error response."
    )

    @classmethod
    def create(
        cls,
        code: str,
        message: str,
        details: list[dict[str, Any]] | None = None,
    ):

        """
        Creates a standardized API error response.
        """

        return cls(

            success=False,

            error=ErrorInformation(

                code=code,

                message=message,

                details=details or [],

            ),

        )

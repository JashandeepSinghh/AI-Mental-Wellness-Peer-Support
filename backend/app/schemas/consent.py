"""
Consent API Schemas
====================

Validates general privacy consent decisions.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConsentDecisionRequest(BaseModel):
    """Request body for recording a consent decision."""

    model_config = ConfigDict(extra="forbid")

    purpose: Literal[
        "conversation_storage",
        "analytics",
    ]

    granted: bool

    policy_version: str = Field(
        min_length=1,
        max_length=50,
    )

    @field_validator("policy_version")
    @classmethod
    def validate_policy_version(cls, value: str) -> str:

        value = value.strip()

        if not value:
            raise ValueError(
                "Policy version cannot be empty."
            )

        return value

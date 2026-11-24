
"""
Chat Request Schemas
====================

Defines and validates incoming chatbot requests.

Privacy principles:
- Accept only necessary information.
- Do not require student names or email addresses.
- Use UUIDs for conversation identification.
- Restrict optional client metadata.
- Reject empty or whitespace-only messages.
"""

from enum import Enum
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


# ============================================================
# 1. REQUEST CONFIGURATION
# ============================================================

MAX_MESSAGE_LENGTH = 20000


# ============================================================
# 2. CLIENT PLATFORM ENUM
# ============================================================

class ClientPlatform(str, Enum):

    """
    Supported client application platforms.
    """

    WEB = "web"

    MOBILE = "mobile"

    DESKTOP = "desktop"

    UNKNOWN = "unknown"


# ============================================================
# 3. CLIENT METADATA SCHEMA
# ============================================================

class ClientMetadata(BaseModel):

    """
    Optional, limited client metadata.

    Avoid collecting unnecessary device or personal details.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    platform: ClientPlatform = Field(
        default=ClientPlatform.UNKNOWN,
        description="Platform used to access the chatbot."
    )

    app_version: str | None = Field(
        default=None,
        min_length=1,
        max_length=30,
        description="Version of the client application."
    )

    @field_validator("app_version")
    @classmethod
    def validate_app_version(cls, value):

        if value is not None and not value.strip():

            raise ValueError(
                "Application version cannot be empty."
            )

        return value


# ============================================================
# 4. CHAT REQUEST SCHEMA
# ============================================================

class ChatRequest(BaseModel):

    """
    Validates an incoming student chatbot message.

    Required:
        message

    Optional:
        conversation_id
        metadata
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    # --------------------------------------------------------
    # Student message
    # --------------------------------------------------------

    message: str = Field(

        min_length=1,

        max_length=MAX_MESSAGE_LENGTH,

        description="Text message submitted by the student."

    )

    # --------------------------------------------------------
    # Conversation identifier
    # --------------------------------------------------------

    conversation_id: UUID | None = Field(

        default=None,

        description=(
            "Optional UUID identifying an existing conversation."
        )

    )

    # --------------------------------------------------------
    # Client metadata
    # --------------------------------------------------------

    metadata: ClientMetadata | None = Field(

        default=None,

        description="Optional limited client application metadata."

    )

    # --------------------------------------------------------
    # Message validation
    # --------------------------------------------------------

    @field_validator("message")
    @classmethod
    def validate_message(cls, value):

        """
        Reject empty or whitespace-only messages.

        Preserve the original message instead of silently
        modifying student-provided text.
        """

        if not value.strip():

            raise ValueError(
                "Message cannot contain only whitespace."
            )

        return value

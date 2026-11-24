"""
Authentication Request Schemas
================================

Defines and validates incoming authentication requests.

Privacy and security principles:
- Accept only the information required for each operation.
- Validate email addresses using Pydantic EmailStr.
- Enforce a reasonable password-length policy.
- Never accept user roles or account statuses from clients.
- Reject unexpected request fields.
- Avoid exposing passwords or verification tokens in representations.
"""

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    field_validator,
)


# ============================================================
# 1. REGISTRATION REQUEST SCHEMA
# ============================================================

class RegistrationRequest(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
    )

    email: EmailStr = Field(
        description="Email address used for account registration.",
    )

    password: str = Field(
        min_length=12,
        max_length=128,
        repr=False,
        description=(
            "Account password. Must contain between "
            "12 and 128 characters."
        ),
    )

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:

        if not value.strip():
            raise ValueError(
                "Password cannot contain only whitespace."
            )

        return value


# ============================================================
# 2. RESEND EMAIL VERIFICATION REQUEST
# ============================================================

class ResendVerificationRequest(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
    )

    email: EmailStr = Field(
        description=(
            "Email address associated with the account "
            "requiring email verification."
        ),
    )


# ============================================================
# 3. EMAIL VERIFICATION REQUEST
# ============================================================

class EmailVerificationRequest(BaseModel):
    """
    Validates an email verification token submitted by the client.

    Required:
        token

    Security:
        - Reject unexpected fields.
        - Never expose the token in the model representation.
        - Preserve the original token without trimming it.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    token: str = Field(
        min_length=32,
        max_length=256,
        repr=False,
        description="Single-use email verification token.",
    )

    @field_validator("token")
    @classmethod
    def validate_token(cls, value: str) -> str:

        if not value.strip():
            raise ValueError(
                "Verification token cannot be empty."
            )

        return value

# ============================================================
# 4. LOGIN REQUEST SCHEMA
# ============================================================

class LoginRequest(BaseModel):
    """
    Validates user login credentials.

    Security:
        - Accepts only email and password.
        - Rejects unexpected fields.
        - Never exposes the password in representations.
        - Does not apply registration password rules to login.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    email: EmailStr = Field(
        description="Email address associated with the account.",
    )

    password: str = Field(
        min_length=1,
        max_length=128,
        repr=False,
        description="Account password.",
    )

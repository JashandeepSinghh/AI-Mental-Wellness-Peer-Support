"""
Authentication API Routes
===========================

Handles student registration, email verification delivery,
email address verification, and student login.

Security:
- Never return passwords or verification tokens.
- Avoid exposing whether an email is already registered.
- Never log sensitive authentication information.
- Reject invalid, expired, and previously used tokens.
- Store refresh tokens in HttpOnly cookies.

"""

import logging
from datetime import datetime, timezone

from fastapi.responses import JSONResponse

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    status,
)

from sqlalchemy.orm import Session

from starlette.concurrency import run_in_threadpool

from backend.app.core.config import get_settings
from backend.app.database.session import get_db

from backend.app.schemas.auth import (
    RegistrationRequest,
    ResendVerificationRequest,
    EmailVerificationRequest,
    LoginRequest,
)

from backend.app.services.registration_service import (
    register_student,
    RegistrationConflictError,
)

from backend.app.services.login_service import (
    authenticate_user,
    LoginAuthenticationError,
)

from backend.app.services.refresh_token_service import (
    refresh_user_session,
    revoke_user_session,
    RefreshTokenError,
)

from backend.app.services.resend_verification_service import (
    resend_verification_token,
)

from backend.app.services.email_verification_service import (
    verify_email_address,
)

from backend.app.auth.verification import (
    build_verification_url,
)

from backend.app.services.email_templates import (
    build_verification_email,
)

from backend.app.services.email_service import (
    send_email,
    EmailDeliveryError,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


# ============================================================
# 1. REGISTRATION RESPONSE
# ============================================================

def registration_accepted_response() -> dict:
    return {
        "success": True,
        "status": "verification_pending",
        "message": (
            "If this address can be registered, "
            "verification instructions will be sent."
        ),
    }


# ============================================================
# 2. RESEND VERIFICATION RESPONSE
# ============================================================

def resend_verification_accepted_response() -> dict:
    return {
        "success": True,
        "status": "verification_request_received",
        "message": (
            "If this address is eligible for verification, "
            "verification instructions will be sent."
        ),
    }


# ============================================================
# 3. EMAIL VERIFICATION RESPONSE
# ============================================================

def email_verification_success_response() -> dict:
    return {
        "success": True,
        "status": "email_verified",
        "message": (
            "Your email address has been verified. "
            "You can now sign in."
        ),
    }


# ============================================================
# 4. STUDENT REGISTRATION ENDPOINT
# ============================================================

@router.post(
    "/register",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Register a new student account",
    description=(
        "Creates a student account and sends an "
        "email verification link."
    ),
)
async def register(
    registration_request: RegistrationRequest,
    db: Session = Depends(get_db),
):
    """
    Register a student and send an email verification link.
    """

    # A. Create student account
    try:
        registration_result = await run_in_threadpool(
            register_student,
            db,
            str(registration_request.email),
            registration_request.password,
        )

    except RegistrationConflictError:
        return registration_accepted_response()

    # B. Generate verification URL
    settings = get_settings()

    verification_url = build_verification_url(
        registration_result.raw_token
    )

    # C. Construct verification email
    email_content = build_verification_email(
        verification_url=verification_url,
        expiration_minutes=(
            settings.EMAIL_VERIFICATION_EXPIRE_MINUTES
        ),
    )

    # D. Send verification email
    try:
        await send_email(
            to_email=registration_result.email,
            subject=email_content["subject"],
            text_body=email_content["text_body"],
            html_body=email_content["html_body"],
        )

    except EmailDeliveryError:
        logger.error(
            "Registration verification email delivery failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Registration was received, but verification "
                "email delivery is temporarily unavailable."
            ),
        ) from None

    return registration_accepted_response()


# ============================================================
# 5. RESEND EMAIL VERIFICATION ENDPOINT
# ============================================================

@router.post(
    "/resend-verification",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Resend an email verification link",
    description=(
        "Requests a new email verification link for an "
        "eligible pending student account."
    ),
)
async def resend_verification(
    request: ResendVerificationRequest,
    db: Session = Depends(get_db),
):
    """
    Request a new email verification link.

    A generic response is returned for all valid requests
    to prevent account enumeration.
    """

    # A. Request a new verification token
    result = await run_in_threadpool(
        resend_verification_token,
        db,
        str(request.email),
    )

    # B. Handle ineligible or cooldown-restricted accounts
    if result is None:
        return resend_verification_accepted_response()

    # C. Generate verification URL
    settings = get_settings()

    verification_url = build_verification_url(
        result.raw_token
    )

    # D. Construct verification email
    email_content = build_verification_email(
        verification_url=verification_url,
        expiration_minutes=(
            settings.EMAIL_VERIFICATION_EXPIRE_MINUTES
        ),
    )

    # E. Deliver verification email
    try:
        await send_email(
            to_email=result.email,
            subject=email_content["subject"],
            text_body=email_content["text_body"],
            html_body=email_content["html_body"],
        )

    except EmailDeliveryError:
        logger.error(
            "Resend verification email delivery failed."
        )

    # F. Return generic response
    return resend_verification_accepted_response()


# ============================================================
# 6. EMAIL VERIFICATION ENDPOINT
# ============================================================

@router.post(
    "/verify-email",
    status_code=status.HTTP_200_OK,
    summary="Verify a student's email address",
    description=(
        "Validates an email verification token, marks the "
        "email address as verified, and activates the account."
    ),
)
async def verify_email(
    request: EmailVerificationRequest,
    db: Session = Depends(get_db),
):
    """
    Verify an email address using its verification token.

    Invalid, expired, and previously used tokens are rejected.
    """

    # A. Validate token and activate account
    verification_success = await run_in_threadpool(
        verify_email_address,
        db,
        request.token,
    )

    # B. Reject unsuccessful verification
    if not verification_success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "The verification link is invalid, expired, "
                "or has already been used."
            ),
        )

    # C. Return successful verification response
    return email_verification_success_response()


# ============================================================
# 7. STUDENT LOGIN ENDPOINT
# ============================================================

@router.post(
    "/login",
    status_code=status.HTTP_200_OK,
    summary="Authenticate a student",
    description=(
        "Authenticates a verified student and returns an "
        "access token. The refresh token is stored in an "
        "HttpOnly cookie."
    ),
)
async def login(
    login_request: LoginRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    """
    Authenticate a student using email and password.

    Security:
        - Returns a generic authentication error.
        - Does not expose refresh tokens in JSON.
        - Stores refresh tokens in an HttpOnly cookie.
        - Enables Secure cookies in production.
    """

    # A. Authenticate user
    try:
        login_result = await run_in_threadpool(
            authenticate_user,
            db,
            str(login_request.email),
            login_request.password,
        )

    except LoginAuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Invalid email or password, "
                "or account unavailable."
            ),
            headers={"WWW-Authenticate": "Bearer"},
        ) from None

    # B. Load authentication settings
    settings = get_settings()

    # C. Calculate refresh cookie expiration
    refresh_max_age = (
        settings.REFRESH_TOKEN_EXPIRE_DAYS
        * 24
        * 60
        * 60
    )

    # D. Store refresh token in an HttpOnly cookie
    response.set_cookie(
        key="refresh_token",
        value=login_result.refresh_token,
        max_age=refresh_max_age,
        httponly=True,
        secure=settings.ENVIRONMENT == "production",
        samesite="lax",
        path=f"{settings.API_PREFIX}/auth",
    )

    # E. Return access token and safe user information
    return {
        "success": True,
        "token_type": login_result.token_type,
        "access_token": login_result.access_token,
        "expires_in": login_result.expires_in,
        "user": {
            "id": str(login_result.user_id),
            "role": login_result.role,
        },
    }

# ============================================================
# 8. REFRESH ACCESS TOKEN ENDPOINT
# ============================================================

@router.post(
    "/refresh",
    status_code=status.HTTP_200_OK,
    summary="Refresh authentication tokens",
    description=(
        "Rotates the refresh token stored in the HttpOnly cookie "
        "and returns a new access token."
    ),
)
async def refresh(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    """
    Refresh authentication tokens.

    Security:
        - Reads refresh token from an HttpOnly cookie.
        - Rotates the refresh token after successful validation.
        - Never returns refresh tokens in JSON.
        - Clears invalid refresh-token cookies.
    """

    settings = get_settings()

    cookie_path = f"{settings.API_PREFIX}/auth"
    cookie_secure = settings.ENVIRONMENT == "production"

    def authentication_error_response():
        error_response = JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "success": False,
                "error": {
                    "code": "UNAUTHORIZED",
                    "message": "Authentication is required.",
                    "details": [],
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

        error_response.delete_cookie(
            key="refresh_token",
            path=cookie_path,
            httponly=True,
            secure=cookie_secure,
            samesite="lax",
        )

        return error_response

    # A. Read refresh token from the cookie.
    refresh_token = request.cookies.get("refresh_token")

    if not refresh_token:
        return authentication_error_response()

    # B. Validate and rotate refresh token.
    try:
        refresh_result = await run_in_threadpool(
            refresh_user_session,
            db,
            refresh_token,
        )

    except RefreshTokenError:
        return authentication_error_response()

    # C. Calculate replacement cookie expiration.
    refresh_max_age = (
        settings.REFRESH_TOKEN_EXPIRE_DAYS
        * 24
        * 60
        * 60
    )

    # D. Store replacement refresh token in HttpOnly cookie.
    response.set_cookie(
        key="refresh_token",
        value=refresh_result.refresh_token,
        max_age=refresh_max_age,
        httponly=True,
        secure=cookie_secure,
        samesite="lax",
        path=cookie_path,
    )

    # E. Return new access token and safe user information.
    return {
        "success": True,
        "token_type": refresh_result.token_type,
        "access_token": refresh_result.access_token,
        "expires_in": refresh_result.expires_in,
        "user": {
            "id": str(refresh_result.user_id),
            "role": refresh_result.role,
        },
    }
# ============================================================
# 9. STUDENT LOGOUT ENDPOINT
# ============================================================

@router.post(
    "/logout",
    status_code=status.HTTP_200_OK,
    summary="Log out the current session",
    description=(
        "Revokes the current refresh-token session and clears "
        "the refresh-token cookie."
    ),
)
async def logout(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    """
    Log out the current authenticated session.

    Security:
        - Reads the refresh token from the HttpOnly cookie.
        - Revokes only the associated session.
        - Does not revoke other device sessions.
        - Clears the refresh-token cookie.
        - Returns a generic response for missing or invalid tokens.
    """

    settings = get_settings()

    cookie_path = f"{settings.API_PREFIX}/auth"
    cookie_secure = settings.ENVIRONMENT == "production"

    # A. Read the refresh token.
    refresh_token = request.cookies.get("refresh_token")

    # B. Revoke the associated session when a token is present.
    if refresh_token:
        await run_in_threadpool(
            revoke_user_session,
            db,
            refresh_token,
        )

    # C. Clear the refresh-token cookie.
    response.delete_cookie(
        key="refresh_token",
        path=cookie_path,
        httponly=True,
        secure=cookie_secure,
        samesite="lax",
    )

    # D. Return a generic logout response.
    return {
        "success": True,
        "message": "Logout successful.",
    }
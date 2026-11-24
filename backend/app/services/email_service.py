
"""
Email Delivery Service
=======================

Responsibilities:
- Send application emails through asynchronous SMTP.
- Support plain-text and HTML email content.
- Load SMTP configuration from application settings.
- Use encrypted SMTP connections.
- Handle delivery failures without exposing credentials.

Security:
- SMTP credentials are never logged.
- Email content is not logged.
- Verification tokens are never logged.
- SMTP passwords are retrieved from SecretStr configuration.
"""

import asyncio

from email.message import EmailMessage
from email.utils import formataddr

import aiosmtplib

from aiosmtplib.errors import SMTPException

from backend.app.core.config import get_settings


# ============================================================
# 1. EMAIL DELIVERY EXCEPTION
# ============================================================

class EmailDeliveryError(Exception):
    """Raised when an email cannot be delivered."""


# ============================================================
# 2. EMAIL SENDING FUNCTION
# ============================================================

async def send_email(
    to_email: str,
    subject: str,
    text_body: str,
    html_body: str | None = None,
) -> None:
    """
    Send an email using the configured SMTP provider.

    Args:
        to_email:
            Recipient email address.

        subject:
            Email subject.

        text_body:
            Plain-text email content.

        html_body:
            Optional HTML email content.

    Returns:
        None

    Raises:
        EmailDeliveryError:
            If SMTP is not configured or delivery fails.
    """

    settings = get_settings()

    # --------------------------------------------------------
    # 1. Verify SMTP configuration
    # --------------------------------------------------------

    if not settings.SMTP_HOST.strip():
        raise EmailDeliveryError(
            "Email delivery is not configured."
        )

    # --------------------------------------------------------
    # 2. Construct email message
    # --------------------------------------------------------

    message = EmailMessage()

    message["From"] = formataddr(
        (
            settings.SMTP_FROM_NAME,
            settings.SMTP_FROM_EMAIL,
        )
    )

    message["To"] = to_email

    message["Subject"] = subject

    message.set_content(text_body)

    # --------------------------------------------------------
    # 3. Add optional HTML alternative
    # --------------------------------------------------------

    if html_body is not None:

        message.add_alternative(
            html_body,
            subtype="html",
        )

    # --------------------------------------------------------
    # 4. Retrieve SMTP password securely
    # --------------------------------------------------------

    smtp_password = (
        settings.SMTP_PASSWORD.get_secret_value()
    )

    # --------------------------------------------------------
    # 5. Deliver email asynchronously
    # --------------------------------------------------------

    try:

        await aiosmtplib.send(
            message,

            hostname=settings.SMTP_HOST,

            port=settings.SMTP_PORT,

            username=settings.SMTP_USERNAME,

            password=smtp_password,

            start_tls=settings.SMTP_STARTTLS,

            use_tls=settings.SMTP_USE_TLS,

            timeout=settings.SMTP_TIMEOUT_SECONDS,
        )

    except (
        SMTPException,
        OSError,
        asyncio.TimeoutError,
    ) as exc:

        # Do not include credentials or message content
        # in the exception exposed to application code.

        raise EmailDeliveryError(
            "Email delivery failed."
        ) from exc

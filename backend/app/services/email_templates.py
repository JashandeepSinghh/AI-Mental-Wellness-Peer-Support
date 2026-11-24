
"""
Email Verification Templates
=============================

Responsibilities:
- Generate email verification subjects.
- Generate plain-text verification emails.
- Generate HTML verification emails.
- Include verification links and expiration information.

Security:
- Never log verification URLs or tokens.
- Escape dynamic values before inserting them into HTML.
- Avoid including sensitive personal information.
"""


from html import escape


# ============================================================
# 1. VERIFICATION EMAIL TEMPLATE
# ============================================================

def build_verification_email(
    verification_url: str,
    expiration_minutes: int = 30,
) -> dict[str, str]:
    """
    Build a student email verification message.

    Args:
        verification_url:
            Complete verification URL containing the
            single-use verification token.

        expiration_minutes:
            Number of minutes before the token expires.

    Returns:
        dict containing:
            subject
            text_body
            html_body
    """

    if not isinstance(verification_url, str):
        raise TypeError(
            "Verification URL must be a string."
        )

    if not verification_url.startswith(("https://", "http://")):
        raise ValueError(
            "Verification URL must use HTTP or HTTPS."
        )

    if expiration_minutes < 1:
        raise ValueError(
            "Expiration must be at least one minute."
        )

    # --------------------------------------------------------
    # 1. Email subject
    # --------------------------------------------------------

    subject = (
        "Verify your AI Mental Wellness account"
    )

    # --------------------------------------------------------
    # 2. Plain-text email
    # --------------------------------------------------------

    text_body = f"""
Hello,

Thank you for registering with AI Mental Wellness
and Peer Support.

Please verify your email address using the link below:

{verification_url}

This verification link expires in
{expiration_minutes} minutes.

If you did not request this account, you can ignore
this email.

For your security, do not share this verification link
with anyone.

AI Mental Wellness Support
""".strip()

    # --------------------------------------------------------
    # 3. Escape dynamic HTML values
    # --------------------------------------------------------

    safe_url = escape(
        verification_url,
        quote=True,
    )

    safe_expiration = escape(
        str(expiration_minutes),
        quote=True,
    )

    # --------------------------------------------------------
    # 4. HTML email
    # --------------------------------------------------------

    html_body = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Verify Your Account</title>
</head>

<body style="
    margin: 0;
    padding: 24px;
    background-color: #f5f7fb;
    font-family: Arial, sans-serif;
    color: #253047;
">

    <div style="
        max-width: 560px;
        margin: 0 auto;
        padding: 32px;
        background-color: #ffffff;
        border-radius: 12px;
    ">

        <h2 style="color: #253047;">
            Verify your email address
        </h2>

        <p>Hello,</p>

        <p>
            Thank you for registering with
            <strong>AI Mental Wellness and Peer Support</strong>.
        </p>

        <p>
            Please use the button below to verify your
            email address.
        </p>

        <p style="margin: 32px 0;">
            <a
                href="{safe_url}"
                style="
                    display: inline-block;
                    padding: 13px 24px;
                    background-color: #2563eb;
                    color: #ffffff;
                    text-decoration: none;
                    border-radius: 7px;
                    font-weight: bold;
                "
            >
                Verify Email Address
            </a>
        </p>

        <p>
            This link expires in
            <strong>{safe_expiration} minutes</strong>.
        </p>

        <p>
            If the button does not work, copy and paste
            the following link into your browser:
        </p>

        <p style="word-break: break-all;">
            <a href="{safe_url}">{safe_url}</a>
        </p>

        <hr style="
            border: none;
            border-top: 1px solid #e5e7eb;
            margin: 28px 0;
        ">

        <p style="font-size: 13px; color: #64748b;">
            If you did not request this account, you can
            safely ignore this email.
        </p>

        <p style="font-size: 13px; color: #64748b;">
            Do not share your verification link with anyone.
        </p>

        <p style="font-size: 13px; color: #64748b;">
            AI Mental Wellness Support
        </p>

    </div>

</body>
</html>
"""

    return {
        "subject": subject,
        "text_body": text_body,
        "html_body": html_body,
    }

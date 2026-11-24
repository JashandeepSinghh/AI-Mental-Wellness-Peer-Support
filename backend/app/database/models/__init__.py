
"""
Database Models Package
========================

Central registry for SQLAlchemy application models.

All database models must be imported here so that
SQLAlchemy metadata and Alembic can discover them.
"""

from backend.app.database.models.user import User
from backend.app.database.models.user_credential import UserCredential
from backend.app.database.models.user_consent import UserConsent
from backend.app.database.models.conversation import Conversation
from backend.app.database.models.message import Message
from backend.app.database.models.escalation_request import EscalationRequest
from backend.app.database.models.user_session import UserSession
from backend.app.database.models.audit_log import AuditLog
from backend.app.database.models.wellness_aggregate import WellnessAggregate
from backend.app.database.models.email_verification_token import EmailVerificationToken


__all__ = [
    "User",
    "UserCredential",
    "UserConsent",
    "Conversation",
    "Message",
    "EscalationRequest",
    "UserSession",
    "AuditLog",
    "WellnessAggregate",
    "EmailVerificationToken",
]

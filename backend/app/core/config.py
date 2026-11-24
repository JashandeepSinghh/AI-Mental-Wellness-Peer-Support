
"""
Backend Application Configuration
==================================

Centralized configuration using Pydantic Settings v2.

Responsibilities:
- Environment variable management
- Application settings
- API configuration
- CORS configuration
- SQLite database configuration
- Security settings
- SMTP and email verification configuration
- Logging configuration
- Project and model paths
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import (
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"
ENV_FILE = BACKEND_ROOT / ".env"


# ============================================================
# 2. APPLICATION SETTINGS
# ============================================================

class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    # --------------------------------------------------------
    # Application identity
    # --------------------------------------------------------

    APP_NAME: str = "AI Mental Wellness and Peer Support API"

    APP_VERSION: str = "1.0.0"

    APP_DESCRIPTION: str = (
        "Backend API for an AI-powered mental wellness "
        "and peer support application."
    )

    ENVIRONMENT: Literal[
        "development",
        "testing",
        "production",
    ] = "development"

    DEBUG: bool = True

    # --------------------------------------------------------
    # Server configuration
    # --------------------------------------------------------

    HOST: str = "127.0.0.1"

    PORT: int = Field(default=8000, ge=1, le=65535)

    API_PREFIX: str = "/api/v1"

    # --------------------------------------------------------
    # CORS configuration
    # --------------------------------------------------------

    CORS_ORIGINS: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )

    CORS_ALLOW_CREDENTIALS: bool = True

    CORS_ALLOW_METHODS: list[str] = Field(
        default_factory=lambda: [
            "GET",
            "POST",
            "PUT",
            "PATCH",
            "DELETE",
            "OPTIONS",
        ]
    )

    CORS_ALLOW_HEADERS: list[str] = Field(
        default_factory=lambda: [
            "Authorization",
            "Content-Type",
            "Accept",
        ]
    )

    # --------------------------------------------------------
    # SQLite database configuration
    # --------------------------------------------------------

    SQLITE_DATABASE_PATH: Path = (
        BACKEND_ROOT / "data" / "mental_wellness.db"
    )

    SQLITE_ECHO: bool = False

    @field_validator("SQLITE_DATABASE_PATH", mode="before")
    @classmethod
    def resolve_sqlite_database_path(cls, value):

        path = Path(value)

        if not path.is_absolute():
            path = BACKEND_ROOT / path

        return path.resolve()

    # --------------------------------------------------------
    # Security configuration
    # --------------------------------------------------------

    SECRET_KEY: str = ""

    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        ge=1,
        le=1440,
    )

    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=7,
        ge=1,
        le=90,
    )

    # --------------------------------------------------------
    # Email / SMTP configuration
    # --------------------------------------------------------

    SMTP_HOST: str = ""

    SMTP_PORT: int = Field(
        default=587,
        ge=1,
        le=65535,
    )

    SMTP_USERNAME: str = ""

    SMTP_PASSWORD: SecretStr = SecretStr("")

    SMTP_FROM_EMAIL: str = ""

    SMTP_FROM_NAME: str = "AI Mental Wellness Support"

    SMTP_STARTTLS: bool = True

    SMTP_USE_TLS: bool = False

    SMTP_TIMEOUT_SECONDS: int = Field(
        default=20,
        ge=1,
        le=120,
    )

    # --------------------------------------------------------
    # Email verification configuration
    # --------------------------------------------------------

    FRONTEND_BASE_URL: str = "http://localhost:5173"

    EMAIL_VERIFICATION_EXPIRE_MINUTES: int = Field(
        default=30,
        ge=5,
        le=1440,
    )

    @field_validator("FRONTEND_BASE_URL", mode="before")
    @classmethod
    def validate_frontend_base_url(cls, value):

        value = str(value or "").strip().rstrip("/")

        if not value:
            import os
            render_url = os.getenv("RENDER_EXTERNAL_URL", "")
            if render_url:
                value = render_url.strip().rstrip("/")
            else:
                raise ValueError(
                    "FRONTEND_BASE_URL cannot be empty."
                )

        if not (value.startswith("http://") or value.startswith("https://")):
            value = f"https://{value}"

        parsed_url = urlsplit(value)

        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(
                "FRONTEND_BASE_URL must use HTTP or HTTPS."
            )

        if not parsed_url.hostname:
            raise ValueError(
                "FRONTEND_BASE_URL must contain a valid hostname."
            )

        if parsed_url.username or parsed_url.password:
            raise ValueError(
                "Frontend URL cannot contain credentials."
            )

        if parsed_url.query or parsed_url.fragment:
            raise ValueError(
                "Frontend URL cannot contain a query or fragment."
            )

        try:
            if parsed_url.port is not None:
                if not 1 <= parsed_url.port <= 65535:
                    raise ValueError(
                        "Frontend URL contains an invalid port."
                    )
        except ValueError as exc:
            raise ValueError(
                "Frontend URL contains an invalid port."
            ) from exc

        return value

    # --------------------------------------------------------
    # Logging configuration
    # --------------------------------------------------------

    LOG_LEVEL: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "INFO"

    LOG_FILE: str = "logs/backend.log"

    # --------------------------------------------------------
    # Request limits
    # --------------------------------------------------------

    MAX_REQUEST_TEXT_LENGTH: int = Field(
        default=20000,
        ge=1,
        le=100000,
    )

    REQUEST_TIMEOUT_SECONDS: int = Field(
        default=60,
        ge=1,
        le=600,
    )

    # --------------------------------------------------------
    # Model configuration
    # --------------------------------------------------------

    MODEL_ROOT: Path = PROJECT_ROOT / "models"

    EMOTION_MODEL_PATH: Path = (
        PROJECT_ROOT
        / "models"
        / "emotion"
        / "goemotions_distilbert"
        / "best_checkpoint"
    )

    SENTIMENT_MODEL_PATH: Path = (
        PROJECT_ROOT
        / "models"
        / "sentiment"
        / "tweeteval_distilbert"
        / "best_checkpoint"
    )

    SENTIMENT_BILSTM_MODEL_PATH: Path = (
        PROJECT_ROOT
        / "models"
        / "sentiment"
        / "tweeteval_bilstm"
        / "best_checkpoint.pt"
    )

    SENTIMENT_BILSTM_VOCAB_PATH: Path = (
        PROJECT_ROOT
        / "models"
        / "sentiment"
        / "tweeteval_bilstm"
        / "vocabulary.json"
    )

    STRESS_MODEL_PATH: Path = (
        PROJECT_ROOT
        / "models"
        / "stress"
        / "dreaddit_distilbert"
        / "best_checkpoint"
    )

    LONELINESS_MODEL_PATH: Path = (
        PROJECT_ROOT
        / "models"
        / "loneliness"
        / "fig_loneliness_distilbert"
        / "best_checkpoint"
    )

    VAD_LEXICON_PATH: Path = (
        PROJECT_ROOT
        / "datasets"
        / "processed"
        / "nrc_vad"
        / "standardized"
        / "nrc_vad.csv"
    )

    # --------------------------------------------------------
    # Inference configuration
    # --------------------------------------------------------

    INFERENCE_DEVICE: Literal[
        "auto",
        "cpu",
        "cuda",
    ] = "auto"

    MODEL_LOAD_ON_STARTUP: bool = False

    INFERENCE_BATCH_SIZE: int = Field(
        default=8,
        ge=1,
        le=128,
    )

    # --------------------------------------------------------
    # Privacy and data handling
    # --------------------------------------------------------

    STORE_CONVERSATIONS: bool = False

    STORE_ANALYTICS: bool = False

    ENABLE_REQUEST_LOGGING: bool = True

    # ========================================================
    # 3. FIELD VALIDATORS
    # ========================================================

    @field_validator("APP_NAME", "APP_VERSION")
    @classmethod
    def validate_nonempty_strings(cls, value):

        value = value.strip()

        if not value:
            raise ValueError(
                "Application name and version cannot be empty."
            )

        return value

    @field_validator("API_PREFIX")
    @classmethod
    def validate_api_prefix(cls, value):

        value = value.strip()

        if not value.startswith("/"):
            raise ValueError(
                "API_PREFIX must begin with '/'."
            )

        return value.rstrip("/")

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def validate_cors_origins(cls, origins):
        import json

        if isinstance(origins, str):
            origins = origins.strip()
            if origins.startswith("[") and origins.endswith("]"):
                try:
                    origins = json.loads(origins)
                except Exception:
                    origins = [x.strip() for x in origins[1:-1].split(",") if x.strip()]
            else:
                origins = [x.strip() for x in origins.split(",") if x.strip()]

        cleaned = []

        for origin in origins:

            origin = str(origin).strip().rstrip("/")

            if not origin:
                continue

            if origin not in cleaned:
                cleaned.append(origin)

        return cleaned

    # ========================================================
    # 4. MODEL-LEVEL VALIDATION
    # ========================================================

    @model_validator(mode="after")
    def validate_environment_settings(self):

        import os

        # Auto-include FRONTEND_BASE_URL and Render hostname in CORS_ORIGINS
        if self.FRONTEND_BASE_URL and self.FRONTEND_BASE_URL not in self.CORS_ORIGINS:
            self.CORS_ORIGINS.append(self.FRONTEND_BASE_URL)

        render_ext_url = os.getenv("RENDER_EXTERNAL_URL", "")
        if render_ext_url:
            clean_render = render_ext_url.strip().rstrip("/")
            if clean_render and clean_render not in self.CORS_ORIGINS:
                self.CORS_ORIGINS.append(clean_render)

        # ----------------------------------------------------
        # Production security
        # ----------------------------------------------------

        if self.ENVIRONMENT == "production":

            if self.DEBUG:
                raise ValueError(
                    "DEBUG must be False in production."
                )

            if len(self.SECRET_KEY) < 32:
                raise ValueError(
                    "Production requires a SECRET_KEY "
                    "containing at least 32 characters."
                )

            if "*" in self.CORS_ORIGINS:
                raise ValueError(
                    "Wildcard CORS origins are prohibited "
                    "in production."
                )

            if not self.FRONTEND_BASE_URL.startswith("https://"):
                raise ValueError(
                    "Production verification links must use HTTPS."
                )

        # ----------------------------------------------------
        # CORS credential validation
        # ----------------------------------------------------

        if (
            self.CORS_ALLOW_CREDENTIALS
            and "*" in self.CORS_ORIGINS
        ):
            raise ValueError(
                "Wildcard CORS origins cannot be combined "
                "with credential support."
            )

        # ----------------------------------------------------
        # SMTP security validation
        # ----------------------------------------------------

        if self.SMTP_STARTTLS and self.SMTP_USE_TLS:
            raise ValueError(
                "SMTP_STARTTLS and SMTP_USE_TLS cannot "
                "both be enabled."
            )

        smtp_host_configured = bool(self.SMTP_HOST.strip())

        smtp_credentials_present = any([
            self.SMTP_USERNAME.strip(),
            self.SMTP_PASSWORD.get_secret_value(),
            self.SMTP_FROM_EMAIL.strip(),
        ])

        if smtp_credentials_present and not smtp_host_configured:
            raise ValueError(
                "SMTP_HOST is required when SMTP credentials "
                "or sender configuration is provided."
            )

        if smtp_host_configured:

            if not self.SMTP_USERNAME.strip():
                raise ValueError(
                    "SMTP_USERNAME is required."
                )

            if not self.SMTP_PASSWORD.get_secret_value():
                raise ValueError(
                    "SMTP_PASSWORD is required."
                )

            if not self.SMTP_FROM_EMAIL.strip():
                raise ValueError(
                    "SMTP_FROM_EMAIL is required."
                )

            if not (self.SMTP_STARTTLS or self.SMTP_USE_TLS):
                raise ValueError(
                    "SMTP must use STARTTLS or direct TLS."
                )

        # ----------------------------------------------------
        # Testing environment
        # ----------------------------------------------------

        if self.ENVIRONMENT == "testing":

            if self.STORE_CONVERSATIONS:
                raise ValueError(
                    "Conversation storage must be disabled "
                    "in testing by default."
                )

        return self


# ============================================================
# 5. SETTINGS FACTORY
# ============================================================

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached application configuration instance.

    Caching prevents repeated environment-file parsing
    throughout the application.
    """

    return Settings()


# ============================================================
# 6. CONFIGURATION SUMMARY
# ============================================================

def get_config_summary():

    settings = get_settings()

    return {

        "app_name": settings.APP_NAME,

        "app_version": settings.APP_VERSION,

        "environment": settings.ENVIRONMENT,

        "debug": settings.DEBUG,

        "host": settings.HOST,

        "port": settings.PORT,

        "api_prefix": settings.API_PREFIX,

        "cors_origins": settings.CORS_ORIGINS,

        "database_engine": "SQLite",

        "database_configured": bool(
            settings.SQLITE_DATABASE_PATH
        ),

        "sqlite_database_path": str(
            settings.SQLITE_DATABASE_PATH
        ),

        "sqlite_echo": settings.SQLITE_ECHO,

        "log_level": settings.LOG_LEVEL,

        "model_root": str(settings.MODEL_ROOT),

        "inference_device": settings.INFERENCE_DEVICE,

        "model_load_on_startup": (
            settings.MODEL_LOAD_ON_STARTUP
        ),

        "frontend_base_url": settings.FRONTEND_BASE_URL,

        "email_verification_expire_minutes": (
            settings.EMAIL_VERIFICATION_EXPIRE_MINUTES
        ),

        "smtp_host_configured": bool(settings.SMTP_HOST),

        "store_conversations": (
            settings.STORE_CONVERSATIONS
        ),

        "store_analytics": (
            settings.STORE_ANALYTICS
        ),

        "secret_key_configured": bool(
            settings.SECRET_KEY
        ),
    }


# ============================================================
# 7. CONFIGURATION VALIDATION
# ============================================================

def validate_configuration():

    settings = get_settings()

    if settings.APP_NAME != (
        "AI Mental Wellness and Peer Support API"
    ):
        raise ValueError(
            "Unexpected application name."
        )

    if settings.PORT < 1 or settings.PORT > 65535:
        raise ValueError(
            "Invalid server port."
        )

    if not isinstance(settings.MODEL_ROOT, Path):
        raise ValueError(
            "Model root must be a pathlib Path."
        )

    if not isinstance(settings.VAD_LEXICON_PATH, Path):
        raise ValueError(
            "VAD lexicon path must be a pathlib Path."
        )

    if not isinstance(settings.SQLITE_DATABASE_PATH, Path):
        raise ValueError(
            "SQLite database path must be a pathlib Path."
        )

    return True


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":

    validate_configuration()

    print("=" * 70)

    print("BACKEND CONFIGURATION VALIDATION: PASSED")

    print("=" * 70)

    for key, value in get_config_summary().items():

        print(f"{key}: {value}")

    print("\nConfiguration loaded successfully.")

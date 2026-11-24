
"""
AI Mental Wellness and Peer Support API
========================================

Main FastAPI application entry point.

Responsibilities:
- Initialize FastAPI
- Configure CORS
- Manage application lifecycle
- Initialize SQLite database
- Initialize inference pipeline
- Manage model resources
- Register global exception handlers
- Register API routes
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from backend.app.core.config import get_settings
from backend.app.core.logging_config import (
    configure_logging,
    get_logger,
)
from backend.app.core.exception_handlers import (
    register_exception_handlers,
)
from backend.app.database.session import (
    initialize_sqlite_settings,
)
from backend.app.api.routes.chat import router as chat_router
from backend.app.api.routes.auth import router as auth_router
from backend.app.api.routes.consent import router as consent_router
from backend.app.inference.inference_pipeline import (
    InferencePipeline,
)


# ============================================================
# 1. LOAD APPLICATION SETTINGS
# ============================================================

settings = get_settings()


# ============================================================
# 2. INITIALIZE LOGGING
# ============================================================

configure_logging()

logger = get_logger("application")


# ============================================================
# 3. APPLICATION LIFECYCLE
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    """
    Initialize and release application resources.

    Startup sequence:
    1. Initialize SQLite database settings.
    2. Initialize the shared inference pipeline.
    3. Optionally load transformer models.

    Transformer models remain lazily loaded unless
    MODEL_LOAD_ON_STARTUP is enabled.
    """

    logger.info(
        "Starting %s | version=%s | environment=%s",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.ENVIRONMENT,
    )

    inference_pipeline = None

    try:

        # ----------------------------------------------------
        # Initialize SQLite database.
        # ----------------------------------------------------

        journal_mode = await run_in_threadpool(
            initialize_sqlite_settings
        )

        logger.info(
            "SQLite database initialized successfully | "
            "journal_mode=%s",
            journal_mode,
        )

        # ----------------------------------------------------
        # Initialize inference pipeline.
        # ----------------------------------------------------

        inference_pipeline = InferencePipeline()

        app.state.inference_pipeline = inference_pipeline

        logger.info(
            "Inference pipeline initialized successfully."
        )

        # ----------------------------------------------------
        # Optional eager transformer loading.
        # ----------------------------------------------------

        if settings.MODEL_LOAD_ON_STARTUP:

            logger.info(
                "Eager transformer model loading enabled."
            )

            await run_in_threadpool(
                inference_pipeline.model_manager.load_all_transformers
            )

            logger.info(
                "All transformer models loaded successfully."
            )

        yield

    finally:

        # ----------------------------------------------------
        # Release inference resources.
        # ----------------------------------------------------

        if inference_pipeline is not None:

            await run_in_threadpool(
                inference_pipeline.model_manager.shutdown
            )

        app.state.inference_pipeline = None

        logger.info(
            "Shutting down %s",
            settings.APP_NAME,
        )


# ============================================================
# 4. CREATE FASTAPI APPLICATION
# ============================================================

is_development = (
    settings.ENVIRONMENT != "production"
)

app = FastAPI(

    title=settings.APP_NAME,

    description=settings.APP_DESCRIPTION,

    version=settings.APP_VERSION,

    debug=settings.DEBUG,

    lifespan=lifespan,

    docs_url="/docs" if is_development else None,

    redoc_url="/redoc" if is_development else None,

    openapi_url=(
        f"{settings.API_PREFIX}/openapi.json"
        if is_development
        else None
    ),
)


# ============================================================
# 5. REGISTER GLOBAL EXCEPTION HANDLERS
# ============================================================

register_exception_handlers(app)


# ============================================================
# 6. CONFIGURE CORS MIDDLEWARE
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=settings.CORS_ORIGINS,

    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,

    allow_methods=settings.CORS_ALLOW_METHODS,

    allow_headers=settings.CORS_ALLOW_HEADERS,
)


# ============================================================
# 7. CREATE MAIN API ROUTER
# ============================================================

api_router = APIRouter()


# ============================================================
# 8. HEALTH-CHECK ENDPOINT
# ============================================================

@api_router.get(
    "/health",
    tags=["System"],
    summary="Check API health",
)
async def health_check():

    """
    Return the current API service status.

    This is a liveness check, not a complete readiness check.
    """

    return {

        "status": "ok",

        "service": settings.APP_NAME,

        "version": settings.APP_VERSION,

        "environment": settings.ENVIRONMENT,

    }


# ============================================================
# 9. REGISTER CHAT ROUTER
# ============================================================

api_router.include_router(
    chat_router
)

api_router.include_router(
    auth_router
)

api_router.include_router(
    consent_router
)


# ============================================================
# 10. REGISTER API ROUTES
# ============================================================

app.include_router(

    api_router,

    prefix=settings.API_PREFIX,

)


# ============================================================
# 11. ROOT ENDPOINT
# ============================================================

@app.get(
    "/",
    tags=["System"],
    summary="API information",
)
async def root():

    return {

        "message": "AI Mental Wellness and Peer Support API",

        "version": settings.APP_VERSION,

        "documentation": (
            "/docs" if is_development else None
        ),

        "health_check": f"{settings.API_PREFIX}/health",

    }


# ============================================================
# 12. APPLICATION CONFIGURATION MESSAGE
# ============================================================

logger.info(
    "FastAPI application configured successfully."
)



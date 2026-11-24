
"""
SQLite Database Engine and Session Management
===============================================

Responsibilities:
- Initialize the SQLite database engine.
- Configure SQLite connection behavior.
- Enable database concurrency optimizations.
- Provide SQLAlchemy database sessions.
- Manage session lifecycle.
"""

from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.config import get_settings


# ============================================================
# 1. DATABASE CONFIGURATION
# ============================================================

settings = get_settings()

DATABASE_PATH = settings.SQLITE_DATABASE_PATH

# Create the database directory if it does not exist.
DATABASE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"


# ============================================================
# 2. DATABASE ENGINE
# ============================================================

engine = create_engine(
    DATABASE_URL,
    echo=settings.SQLITE_ECHO,
    connect_args={
        "check_same_thread": False,
        "timeout": 30,
    },
)


# ============================================================
# 3. SQLITE CONNECTION CONFIGURATION
# ============================================================

@event.listens_for(engine, "connect")
def configure_sqlite_connection(
    dbapi_connection,
    connection_record,
):
    """
    Configure settings for every new SQLite connection.

    Foreign-key enforcement, busy timeout and synchronous
    mode are applied whenever SQLAlchemy creates a connection.
    """

    cursor = dbapi_connection.cursor()

    try:
        # Enforce foreign-key constraints.
        cursor.execute("PRAGMA foreign_keys=ON")

        # Wait up to 30 seconds when another connection is writing.
        cursor.execute("PRAGMA busy_timeout=30000")

        # Use NORMAL synchronization in WAL mode.
        cursor.execute("PRAGMA synchronous=NORMAL")

    finally:
        cursor.close()


# ============================================================
# 4. SQLITE DATABASE INITIALIZATION
# ============================================================

def initialize_sqlite_settings() -> str:
    """
    Initialize and verify SQLite database settings.

    WAL mode improves read/write concurrency.

    SQLite still permits only one active writer at a time.

    Returns:
        str: The active SQLite journal mode.

    Raises:
        RuntimeError: If WAL mode cannot be enabled.
    """

    with engine.connect() as connection:

        journal_mode = connection.exec_driver_sql(
            "PRAGMA journal_mode=WAL"
        ).scalar()

        if journal_mode is None:
            raise RuntimeError(
                "Unable to determine SQLite journal mode."
            )

        journal_mode = journal_mode.lower()

        if journal_mode != "wal":
            raise RuntimeError(
                "SQLite WAL initialization failed. "
                f"Active journal mode: {journal_mode}"
            )

        # Verify connection-level settings.
        foreign_keys = connection.exec_driver_sql(
            "PRAGMA foreign_keys"
        ).scalar()

        synchronous = connection.exec_driver_sql(
            "PRAGMA synchronous"
        ).scalar()

        if foreign_keys != 1:
            raise RuntimeError(
                "SQLite foreign-key enforcement is disabled."
            )

        if synchronous != 1:
            raise RuntimeError(
                "SQLite synchronous mode is not NORMAL."
            )

        return journal_mode


# ============================================================
# 5. SESSION FACTORY
# ============================================================

SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


# ============================================================
# 6. DATABASE SESSION DEPENDENCY
# ============================================================

def get_db() -> Generator[Session, None, None]:
    """
    Provide a database session and always close it afterward.

    Intended for FastAPI dependency injection.
    """

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()

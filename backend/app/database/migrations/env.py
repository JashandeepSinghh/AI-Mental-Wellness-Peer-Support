
"""
Alembic Migration Environment
==============================

Connects SQLAlchemy models and the SQLite engine
to Alembic's migration system.
"""

from logging.config import fileConfig

from alembic import context

from backend.app.database.base import Base
from backend.app.database.session import engine

# Import every registered model so Base.metadata is populated.
import backend.app.database.models  # noqa: F401


# Alembic configuration.
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


# SQLAlchemy metadata containing all registered models.
target_metadata = Base.metadata

# Use the SQLite URL from the application configuration.
database_url = engine.url.render_as_string(hide_password=False)

config.set_main_option(
    "sqlalchemy.url",
    database_url.replace("%", "%%"),
)


def run_migrations_offline() -> None:
    """Run migrations without creating a database connection."""

    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations using the configured SQLite connection."""

    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_as_batch=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

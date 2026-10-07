"""Database engine construction and schema initialisation."""

from pathlib import Path
from typing import Any

from sqlalchemy import Engine, event
from sqlalchemy.engine import make_url
from sqlmodel import SQLModel, create_engine

from app.db import models  # noqa: F401  (registers the tables on SQLModel.metadata)


def create_db_engine(database_url: str) -> Engine:
    """Create an engine for ``database_url``.

    SQLite connections are opened with ``check_same_thread=False`` because background processing
    runs in a worker thread, with foreign keys enforced, and in WAL mode so that status polls can
    read while a file's features are being written.
    """
    if not database_url.startswith("sqlite"):
        return create_engine(database_url)

    engine = create_engine(database_url, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _configure_sqlite(dbapi_connection: Any, _connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    return engine


def init_db(engine: Engine) -> None:
    """Create the SQLite directory if needed, then create any missing tables."""
    if engine.dialect.name == "sqlite":
        database = make_url(engine.url).database
        if database and database != ":memory:":
            Path(database).parent.mkdir(parents=True, exist_ok=True)
    SQLModel.metadata.create_all(engine)

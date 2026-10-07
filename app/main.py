"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlmodel import Session

from app import __version__
from app.api.errors import register_exception_handlers
from app.api.routes import files, health
from app.core.config import Settings, get_settings
from app.core.logging import get_logger, setup_logging
from app.db.session import create_db_engine, init_db
from app.services import repository

logger = get_logger(__name__)

DESCRIPTION = (
    "Upload a zipped Shapefile or a KML file, then fetch per-feature area and length "
    "measurements computed in a local equal-area or equidistant projection."
)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application; ``settings`` defaults to the values read from the environment."""
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    engine = create_db_engine(settings.database_url)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        settings.storage_dir.mkdir(parents=True, exist_ok=True)
        init_db(engine)
        with Session(engine) as session:
            interrupted = repository.fail_interrupted(session)
        if interrupted:
            logger.warning("failed files interrupted by a restart", extra={"count": interrupted})
        logger.info("application started", extra={"storage_dir": str(settings.storage_dir)})
        yield
        engine.dispose()

    app = FastAPI(
        title="Geospatial File Measurement API",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.engine = engine
    register_exception_handlers(app)
    app.include_router(files.router)
    app.include_router(health.router)
    return app


app = create_app()

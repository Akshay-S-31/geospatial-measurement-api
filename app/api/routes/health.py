"""Liveness and dependency check."""

import pyogrio
from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.api.deps import SessionDep
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


class HealthOut(BaseModel):
    """Service health report."""

    status: str
    db: str
    gdal_version: str


@router.get("/health", response_model=HealthOut, summary="Health check")
def health(session: SessionDep, response: Response) -> HealthOut:
    """Report whether the service can reach its database, and which GDAL build it runs on."""
    try:
        session.execute(text("SELECT 1"))
        db_status = "ok"
    except SQLAlchemyError:
        logger.exception("health check could not reach the database")
        db_status = "error"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthOut(
        status="ok" if db_status == "ok" else "degraded",
        db=db_status,
        gdal_version=pyogrio.__gdal_version_string__,
    )

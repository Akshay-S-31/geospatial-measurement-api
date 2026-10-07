"""FastAPI dependencies shared by the routes."""

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy import Engine
from sqlmodel import Session

from app.core.config import Settings


def get_settings(request: Request) -> Settings:
    """Return the settings the application was created with."""
    return request.app.state.settings


def get_engine(request: Request) -> Engine:
    """Return the application's database engine."""
    return request.app.state.engine


SettingsDep = Annotated[Settings, Depends(get_settings)]
EngineDep = Annotated[Engine, Depends(get_engine)]


def get_session(engine: EngineDep) -> Iterator[Session]:
    """Yield a database session scoped to one request."""
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]

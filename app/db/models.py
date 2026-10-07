"""Database tables: one row per uploaded file and one row per feature."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, Column, DateTime, Index, Text
from sqlmodel import Field, SQLModel


class FileStatus(StrEnum):
    """Processing lifecycle of an uploaded file."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


def utc_now() -> datetime:
    """Current time as a naive UTC datetime.

    Timestamp columns are plain ``DateTime`` holding UTC, so values read back identically from
    every database backend (SQLite has no timezone-aware type). The API adds the ``Z`` suffix
    when serialising them.
    """
    return datetime.now(UTC).replace(tzinfo=None)


class GeoFile(SQLModel, table=True):
    """An uploaded file and the outcome of processing it."""

    __tablename__ = "geo_file"

    id: str = Field(primary_key=True)
    original_filename: str
    file_type: str
    status: str = Field(default=FileStatus.PENDING, index=True)
    crs: str | None = None
    crs_assumed: bool = False
    had_z: bool = False
    layer_count: int = 0
    feature_count: int = 0
    layers: list[dict[str, Any]] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    warnings: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    error_code: str | None = None
    error_message: str | None = None
    size_bytes: int
    created_at: datetime = Field(default_factory=utc_now, sa_type=DateTime, index=True)
    processing_started_at: datetime | None = Field(default=None, sa_type=DateTime)
    processed_at: datetime | None = Field(default=None, sa_type=DateTime)
    processing_ms: int | None = None


class Feature(SQLModel, table=True):
    """One feature of a file with its measurement, or the reason it has none.

    Measurements are stored at full precision; rounding happens only in API output.
    """

    __tablename__ = "feature"
    __table_args__ = (
        Index("ix_feature_file_id_index", "file_id", "index", unique=True),
        Index("ix_feature_file_id_geometry_type", "file_id", "geometry_type"),
    )

    id: int | None = Field(default=None, primary_key=True)
    file_id: str = Field(foreign_key="geo_file.id", ondelete="CASCADE")
    index: int
    layer: str
    geometry_type: str
    geometry_geojson: str | None = Field(default=None, sa_column=Column(Text))
    source_crs: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    geometry_repaired: bool = False
    measurement_kind: str | None = None
    area_m2: float | None = None
    perimeter_m: float | None = None
    length_m: float | None = None
    projection_method: str | None = None
    projected_crs: str | None = None
    reason: str | None = None

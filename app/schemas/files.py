"""Response models for file resources."""

from datetime import UTC, datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, PlainSerializer, WithJsonSchema

from app.db.models import FileStatus, GeoFile
from app.schemas.pagination import Pagination
from app.services.ingest import FileType

_EXAMPLE_ID = "3f9c2a7e1b0d4e6f8a1c2b3d4e5f6a7b"
_EXAMPLE_LINKS = {
    "self": f"/api/files/{_EXAMPLE_ID}/",
    "measurements": f"/api/files/{_EXAMPLE_ID}/measurements/",
}


def _format_utc(value: datetime) -> str:
    """Render a stored timestamp as ISO 8601 UTC with a ``Z`` suffix, to the second."""
    if value.tzinfo is not None:
        value = value.astimezone(UTC)
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


UtcDatetime = Annotated[
    datetime,
    PlainSerializer(_format_utc, return_type=str),
    WithJsonSchema({"type": "string", "format": "date-time"}, mode="serialization"),
]


class Links(BaseModel):
    """Relative URLs of the file and of its measurements."""

    self: str
    measurements: str

    @classmethod
    def for_file(cls, file_id: str) -> "Links":
        """Build the links of one file."""
        return cls(self=file_url(file_id), measurements=f"{file_url(file_id)}measurements/")


class FileError(BaseModel):
    """Why a file or one of its layers failed."""

    code: str
    message: str


class LayerOut(BaseModel):
    """One layer of the file: a shapefile in the archive, or a KML folder."""

    name: str
    feature_count: int
    crs: str | None
    error: FileError | None


class FileAccepted(BaseModel):
    """Body of the ``202`` returned by an upload."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": _EXAMPLE_ID,
                    "filename": "kochi_plots.zip",
                    "file_type": "SHAPEFILE",
                    "status": "PENDING",
                    "size_bytes": 18234,
                    "created_at": "2026-10-07T08:15:02Z",
                    "links": _EXAMPLE_LINKS,
                }
            ]
        }
    )

    id: str
    filename: str
    file_type: FileType
    status: FileStatus
    size_bytes: int
    created_at: UtcDatetime
    links: Links

    @classmethod
    def from_record(cls, geo_file: GeoFile) -> "FileAccepted":
        """Build the response from a stored file."""
        return cls(
            id=geo_file.id,
            filename=geo_file.original_filename,
            file_type=FileType(geo_file.file_type),
            status=FileStatus(geo_file.status),
            size_bytes=geo_file.size_bytes,
            created_at=geo_file.created_at,
            links=Links.for_file(geo_file.id),
        )


class FileOut(BaseModel):
    """Everything known about an uploaded file."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": _EXAMPLE_ID,
                    "filename": "survey.kml",
                    "file_type": "KML",
                    "status": "COMPLETED",
                    "feature_count": 120,
                    "crs": "EPSG:4326",
                    "crs_assumed": False,
                    "had_z": True,
                    "layers": [
                        {"name": "Fields", "feature_count": 80, "crs": "EPSG:4326", "error": None},
                        {"name": "Roads", "feature_count": 40, "crs": "EPSG:4326", "error": None},
                    ],
                    "warnings": ["Z coordinates were dropped; measurements are 2D (planimetric)."],
                    "error": None,
                    "size_bytes": 48211,
                    "created_at": "2026-10-07T08:15:02Z",
                    "processed_at": "2026-10-07T08:15:03Z",
                    "processing_ms": 412,
                    "links": _EXAMPLE_LINKS,
                }
            ]
        }
    )

    id: str
    filename: str
    file_type: FileType
    status: FileStatus
    feature_count: int
    crs: str | None
    crs_assumed: bool
    had_z: bool
    layers: list[LayerOut]
    warnings: list[str]
    error: FileError | None
    size_bytes: int
    created_at: UtcDatetime
    processed_at: UtcDatetime | None
    processing_ms: int | None
    links: Links

    @classmethod
    def from_record(cls, geo_file: GeoFile) -> "FileOut":
        """Build the response from a stored file."""
        return cls(
            id=geo_file.id,
            filename=geo_file.original_filename,
            file_type=FileType(geo_file.file_type),
            status=FileStatus(geo_file.status),
            feature_count=geo_file.feature_count,
            crs=geo_file.crs,
            crs_assumed=geo_file.crs_assumed,
            had_z=geo_file.had_z,
            layers=[LayerOut.model_validate(layer) for layer in geo_file.layers],
            warnings=geo_file.warnings,
            error=file_error(geo_file),
            size_bytes=geo_file.size_bytes,
            created_at=geo_file.created_at,
            processed_at=geo_file.processed_at,
            processing_ms=geo_file.processing_ms,
            links=Links.for_file(geo_file.id),
        )


class FileListOut(BaseModel):
    """One page of files, newest first."""

    items: list[FileOut]
    pagination: Pagination


def file_url(file_id: str) -> str:
    """Relative URL of a file resource."""
    return f"/api/files/{file_id}/"


def file_error(geo_file: GeoFile) -> FileError | None:
    """The stored failure of a file, if it has one."""
    if geo_file.error_code is None:
        return None
    return FileError(code=geo_file.error_code, message=geo_file.error_message or "")

"""File upload, file information and measurement endpoints."""

from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Query, Response, UploadFile, status
from sqlmodel import Session

from app.api.deps import EngineDep, SessionDep, SettingsDep
from app.api.errors import file_failed, file_not_found, file_not_ready
from app.db.models import FileStatus, GeoFile
from app.schemas.errors import ErrorBody
from app.schemas.files import FileAccepted, FileListOut, FileOut, file_url
from app.schemas.measurements import FeatureOut, MeasurementsPage, Summary
from app.schemas.pagination import Pagination
from app.services import ingest, pipeline, repository

router = APIRouter(prefix="/api/files", tags=["files"])

MAX_PAGE_SIZE = 1000
DEFAULT_PAGE_SIZE = 100

Limit = Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE, description="Page size.")]
Offset = Annotated[int, Query(ge=0, description="Number of items to skip.")]


def _errors(*responses: tuple[int, str]) -> dict[int | str, dict[str, Any]]:
    """Document error responses in OpenAPI with the uniform error body."""
    return {code: {"model": ErrorBody, "description": text} for code, text in responses}


NOT_FOUND = (404, "`FILE_NOT_FOUND`: no file has this id.")
INVALID_QUERY = (422, "`VALIDATION_ERROR`: a query parameter is out of range.")


@router.post(
    "/",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=FileAccepted,
    summary="Upload a file",
    responses=_errors(
        (413, "`FILE_TOO_LARGE`: the file is larger than `MAX_UPLOAD_MB`."),
        (415, "`UNSUPPORTED_FILE_TYPE`: the extension is not `.zip` or `.kml`."),
        (
            422,
            "`EMPTY_FILE`, `INVALID_ZIP` or `INVALID_KML`: the content does not match the "
            "extension. `VALIDATION_ERROR`: the `file` field is missing.",
        ),
    ),
)
def upload_file(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    response: Response,
    session: SessionDep,
    engine: EngineDep,
    settings: SettingsDep,
) -> FileAccepted:
    """Upload a `.zip` containing a Shapefile, or a `.kml` file.

    The file is validated and stored, then processed in the background. Poll the `self` link
    until `status` is `COMPLETED` or `FAILED`.
    """
    upload = ingest.store_upload(
        file.file,
        file.filename,
        storage_dir=settings.storage_dir,
        max_bytes=settings.max_upload_bytes,
    )
    geo_file = repository.create_file(session, upload)
    background_tasks.add_task(pipeline.process, geo_file.id, engine, settings)
    response.headers["Location"] = file_url(geo_file.id)
    return FileAccepted.from_record(geo_file)


@router.get("/", response_model=FileListOut, summary="List files", responses=_errors(INVALID_QUERY))
def list_files(
    session: SessionDep, limit: Limit = DEFAULT_PAGE_SIZE, offset: Offset = 0
) -> FileListOut:
    """List uploaded files, newest first."""
    files, total = repository.list_files(session, limit=limit, offset=offset)
    return FileListOut(
        items=[FileOut.from_record(geo_file) for geo_file in files],
        pagination=Pagination.of(limit=limit, offset=offset, total=total),
    )


@router.get(
    "/{file_id}/",
    response_model=FileOut,
    summary="Get file information",
    responses=_errors(NOT_FOUND),
)
def get_file(file_id: str, session: SessionDep) -> FileOut:
    """Return the file's status, CRS, layers, warnings and, if it failed, the reason."""
    return FileOut.from_record(_get_file_or_404(session, file_id))


@router.delete(
    "/{file_id}/",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a file",
    responses=_errors(NOT_FOUND, (409, "`FILE_NOT_READY`: the file is still being processed.")),
)
def delete_file(file_id: str, session: SessionDep, settings: SettingsDep) -> None:
    """Delete the file, its features and its stored upload."""
    geo_file = _get_file_or_404(session, file_id)
    _require_finished(geo_file)
    repository.delete_file(session, geo_file)
    ingest.delete_upload(settings.storage_dir, file_id)


@router.get(
    "/{file_id}/measurements/",
    response_model=MeasurementsPage,
    summary="Get measurements",
    responses=_errors(
        NOT_FOUND,
        (
            409,
            "`FILE_NOT_READY`: the file is still being processed. "
            "`FILE_FAILED`: processing failed; `details.error` holds the reason.",
        ),
        INVALID_QUERY,
    ),
)
def get_measurements(
    file_id: str,
    session: SessionDep,
    limit: Limit = DEFAULT_PAGE_SIZE,
    offset: Offset = 0,
    geometry_type: Annotated[
        str | None,
        Query(description="Return only features of this geometry type, e.g. `Polygon`."),
    ] = None,
    include_geometry: Annotated[
        bool, Query(description="Set to `false` to leave out each feature's GeoJSON geometry.")
    ] = True,
) -> MeasurementsPage:
    """Return per-feature measurements with a whole-file summary.

    Polygons have `area_m2` and `perimeter_m`, lines have `length_m`. Features that cannot be
    measured have `measurement: null` and a coded `reason`. The summary always covers the whole
    file; `geometry_type` filters only `features` and `pagination.total`.
    """
    geo_file = _get_file_or_404(session, file_id)
    _require_finished(geo_file)
    if geo_file.status == FileStatus.FAILED:
        raise file_failed(geo_file)

    features, total = repository.page_features(
        session,
        file_id,
        limit=limit,
        offset=offset,
        geometry_type=geometry_type,
        include_geometry=include_geometry,
    )
    return MeasurementsPage(
        file_id=geo_file.id,
        crs=geo_file.crs,
        summary=Summary.from_totals(repository.summarise(session, file_id)),
        pagination=Pagination.of(limit=limit, offset=offset, total=total),
        features=[
            FeatureOut.from_record(feature, include_geometry=include_geometry)
            for feature in features
        ],
    )


def _get_file_or_404(session: Session, file_id: str) -> GeoFile:
    geo_file = repository.get_file(session, file_id)
    if geo_file is None:
        raise file_not_found(file_id)
    return geo_file


def _require_finished(geo_file: GeoFile) -> None:
    if geo_file.status in (FileStatus.PENDING, FileStatus.PROCESSING):
        raise file_not_ready(geo_file)

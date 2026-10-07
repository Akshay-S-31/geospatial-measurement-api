"""Database reads and writes for files and their features."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, func, insert, update
from sqlalchemy.orm import defer
from sqlmodel import Session, col, select

from app.core.errors import ErrorCode
from app.db.models import Feature, FileStatus, GeoFile, utc_now
from app.services.ingest import StoredUpload

INSERT_BATCH_SIZE = 1000
INTERRUPTED_MESSAGE = "Processing was interrupted by a server restart. Upload the file again."


@dataclass(frozen=True, slots=True)
class FeatureSummary:
    """Whole-file totals over the stored features."""

    feature_count: int
    measured_count: int
    total_area_m2: float
    total_length_m: float
    count_by_geometry_type: dict[str, int]

    @property
    def unmeasured_count(self) -> int:
        """Features kept without a measurement."""
        return self.feature_count - self.measured_count


def create_file(session: Session, upload: StoredUpload) -> GeoFile:
    """Insert the record of a newly stored upload, in status ``PENDING``."""
    geo_file = GeoFile(
        id=upload.file_id,
        original_filename=upload.original_filename,
        file_type=upload.file_type,
        size_bytes=upload.size_bytes,
    )
    session.add(geo_file)
    session.commit()
    session.refresh(geo_file)
    return geo_file


def get_file(session: Session, file_id: str) -> GeoFile | None:
    """Fetch a file by id."""
    return session.get(GeoFile, file_id)


def list_files(session: Session, *, limit: int, offset: int) -> tuple[list[GeoFile], int]:
    """Return one page of files, newest first, and the total number of files."""
    total = session.exec(select(func.count()).select_from(GeoFile)).one()
    statement = (
        select(GeoFile)
        .order_by(col(GeoFile.created_at).desc(), col(GeoFile.id))
        .offset(offset)
        .limit(limit)
    )
    return list(session.exec(statement).all()), total


def delete_file(session: Session, geo_file: GeoFile) -> None:
    """Delete a file and all of its features."""
    session.exec(delete(Feature).where(col(Feature.file_id) == geo_file.id))
    session.delete(geo_file)
    session.commit()


def start_processing(session: Session, file_id: str) -> GeoFile | None:
    """Move a file to ``PROCESSING``. Returns ``None`` if the file no longer exists."""
    geo_file = session.get(GeoFile, file_id)
    if geo_file is None:
        return None
    geo_file.status = FileStatus.PROCESSING
    geo_file.processing_started_at = utc_now()
    geo_file.error_code = None
    geo_file.error_message = None
    session.add(geo_file)
    session.commit()
    session.refresh(geo_file)
    return geo_file


def complete_processing(
    session: Session,
    file_id: str,
    *,
    features: Sequence[Feature],
    layers: list[dict[str, Any]],
    crs: str | None,
    crs_assumed: bool,
    had_z: bool,
    warnings: list[str],
) -> None:
    """Store a file's features and mark it ``COMPLETED``, in a single transaction.

    Features are written with multi-row inserts in batches rather than one statement per row,
    which is what keeps a 10,000-feature file fast. Rows left by an earlier run are replaced, so
    processing the same file twice gives the same result.
    """
    geo_file = session.get(GeoFile, file_id)
    if geo_file is None:
        return
    session.exec(delete(Feature).where(col(Feature.file_id) == file_id))
    for start in range(0, len(features), INSERT_BATCH_SIZE):
        batch = features[start : start + INSERT_BATCH_SIZE]
        session.exec(insert(Feature), params=[row.model_dump(exclude={"id"}) for row in batch])

    geo_file.status = FileStatus.COMPLETED
    geo_file.feature_count = len(features)
    geo_file.layer_count = len(layers)
    geo_file.layers = layers
    geo_file.crs = crs
    geo_file.crs_assumed = crs_assumed
    geo_file.had_z = had_z
    geo_file.warnings = warnings
    _stamp_finished(geo_file)
    session.add(geo_file)
    session.commit()


def fail_processing(
    session: Session,
    file_id: str,
    *,
    code: ErrorCode,
    message: str,
    layers: list[dict[str, Any]],
) -> None:
    """Mark a file ``FAILED`` with a code and a message that is safe to show to clients."""
    geo_file = session.get(GeoFile, file_id)
    if geo_file is None:
        return
    geo_file.status = FileStatus.FAILED
    geo_file.error_code = code
    geo_file.error_message = message
    geo_file.layers = layers
    geo_file.layer_count = len(layers)
    _stamp_finished(geo_file)
    session.add(geo_file)
    session.commit()


def fail_interrupted(session: Session) -> int:
    """Fail every file that is still ``PENDING`` or ``PROCESSING``; return how many there were.

    Called once at startup. Background tasks live in the server process, so a file that is
    unfinished when the process starts lost its task in a restart and would otherwise be reported
    as in progress forever.
    """
    statement = (
        update(GeoFile)
        .where(col(GeoFile.status).in_([FileStatus.PENDING, FileStatus.PROCESSING]))
        .values(
            status=FileStatus.FAILED,
            error_code=ErrorCode.INTERRUPTED,
            error_message=INTERRUPTED_MESSAGE,
            processed_at=utc_now(),
        )
    )
    interrupted = session.exec(statement).rowcount
    session.commit()
    return interrupted


def page_features(
    session: Session,
    file_id: str,
    *,
    limit: int,
    offset: int,
    geometry_type: str | None = None,
    include_geometry: bool = True,
) -> tuple[list[Feature], int]:
    """Return one page of a file's features in index order, and the number matching the filter.

    With ``include_geometry=False`` the GeoJSON column is not loaded at all, which is the bulk of
    a row for detailed geometries.
    """
    conditions = [col(Feature.file_id) == file_id]
    if geometry_type is not None:
        conditions.append(col(Feature.geometry_type) == geometry_type)

    total = session.exec(select(func.count()).select_from(Feature).where(*conditions)).one()
    statement = (
        select(Feature).where(*conditions).order_by(col(Feature.index)).offset(offset).limit(limit)
    )
    if not include_geometry:
        statement = statement.options(defer(col(Feature.geometry_geojson)))
    return list(session.exec(statement).all()), total


def summarise(session: Session, file_id: str) -> FeatureSummary:
    """Compute whole-file totals with SQL aggregates, without loading the features."""
    totals = select(
        func.count(),
        func.count(col(Feature.measurement_kind)),
        func.coalesce(func.sum(col(Feature.area_m2)), 0.0),
        func.coalesce(func.sum(col(Feature.length_m)), 0.0),
    ).where(col(Feature.file_id) == file_id)
    feature_count, measured_count, total_area, total_length = session.exec(totals).one()

    by_type = (
        select(col(Feature.geometry_type), func.count())
        .where(col(Feature.file_id) == file_id)
        .group_by(col(Feature.geometry_type))
        .order_by(col(Feature.geometry_type))
    )
    return FeatureSummary(
        feature_count=feature_count,
        measured_count=measured_count,
        total_area_m2=float(total_area),
        total_length_m=float(total_length),
        count_by_geometry_type=dict(session.exec(by_type).all()),
    )


def _stamp_finished(geo_file: GeoFile) -> None:
    geo_file.processed_at = utc_now()
    if geo_file.processing_started_at is not None:
        elapsed = geo_file.processed_at - geo_file.processing_started_at
        geo_file.processing_ms = round(elapsed.total_seconds() * 1000)

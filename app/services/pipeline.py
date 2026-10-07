"""Background processing of one uploaded file: read, normalise, measure, persist.

``process`` takes only a file id, an engine and settings, so it can be handed to any task runner.
It opens its own database sessions and never raises: every failure ends up on the file record.
"""

import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from pyproj import CRS
from pyproj.exceptions import ProjError
from shapely.geometry.base import BaseGeometry
from sqlalchemy import Engine
from sqlmodel import Session

from app.core.config import Settings
from app.core.errors import ErrorCode, ProcessingError
from app.core.logging import bind_file_id, get_logger
from app.db.models import Feature
from app.services import crs, geometry, ingest, measure, reader, repository
from app.services.ingest import FileType
from app.services.reader import LayerData, RawFeature
from app.services.reasons import ReasonCode

logger = get_logger(__name__)

EXTRACT_DIR_NAME = "extracted"
INTERNAL_ERROR_MESSAGE = "An unexpected error occurred while processing the file."
Z_DROPPED_WARNING = "Z coordinates were dropped; measurements are 2D (planimetric)."
_MAX_LISTED_INDEXES = 10


@dataclass(slots=True)
class _Notes:
    """File-level observations gathered while features are processed."""

    had_z: bool = False
    wide_features: list[int] = field(default_factory=list)
    failed_features: int = 0


def process(file_id: str, engine: Engine, settings: Settings) -> None:
    """Process an uploaded file and store the result, or the reason it failed.

    File-level problems (unsafe or oversized zip, no shapefile, no readable layer, no features)
    set the file to ``FAILED`` with a coded reason. Anything unexpected is logged with its
    traceback and reported as ``INTERNAL_ERROR``, so internals never reach the API. The temporary
    extraction directory is removed either way; the raw upload is kept so the file can be
    processed again.
    """
    with bind_file_id(file_id):
        work_dir = ingest.upload_dir(settings.storage_dir, file_id) / EXTRACT_DIR_NAME
        layers: list[LayerData] = []
        try:
            file_type = _start(engine, file_id)
            if file_type is None:
                logger.warning("file no longer exists; nothing to process")
                return
            with _timed("read"):
                layers = _read(file_id, file_type, work_dir, settings)
            with _timed("measure"):
                features, notes = _measure_layers(file_id, layers, settings.densify_max_segment_deg)
            with _timed("persist"):
                _save(engine, file_id, layers, features, notes)
            logger.info("file processed", extra={"feature_count": len(features)})
        except ProcessingError as exc:
            logger.warning("file rejected", extra={"error_code": exc.code})
            _fail(engine, file_id, exc.code, exc.message, layers)
        except Exception:
            logger.exception("file processing crashed")
            _fail(engine, file_id, ErrorCode.INTERNAL_ERROR, INTERNAL_ERROR_MESSAGE, layers)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


def _start(engine: Engine, file_id: str) -> FileType | None:
    with Session(engine) as session:
        geo_file = repository.start_processing(session, file_id)
        return None if geo_file is None else FileType(geo_file.file_type)


def _read(file_id: str, file_type: FileType, work_dir: Path, settings: Settings) -> list[LayerData]:
    source = ingest.upload_path(settings.storage_dir, file_id, file_type)
    if file_type is FileType.SHAPEFILE:
        paths = ingest.safe_extract(
            source, work_dir, max_unzipped_bytes=settings.max_unzipped_bytes
        )
    else:
        paths = [source]
    return reader.read_layers(paths, file_type)


def _measure_layers(
    file_id: str, layers: list[LayerData], densify_deg: float
) -> tuple[list[Feature], _Notes]:
    """Turn every feature of the readable layers into a row. A file with none has failed."""
    errors = [layer.error for layer in layers if layer.error is not None]
    if errors and len(errors) == len(layers):
        codes = {error.code for error in errors}
        code = codes.pop() if len(codes) == 1 else ErrorCode.READ_ERROR
        raise ProcessingError(code, errors[0].message)

    notes = _Notes()
    features = [
        _safe_feature(file_id, raw, densify_deg, notes) for raw in reader.iter_features(layers)
    ]
    if not features:
        raise ProcessingError(ErrorCode.NO_FEATURES, "The file does not contain any features.")
    return features, notes


def _safe_feature(file_id: str, raw: RawFeature, densify_deg: float, notes: _Notes) -> Feature:
    """Build a feature row, isolating any failure to this one feature."""
    try:
        return _feature(file_id, raw, densify_deg, notes)
    except Exception:
        # Log the first failure in full; a file with thousands of bad features would flood the log.
        if notes.failed_features == 0:
            logger.exception("feature could not be processed", extra={"feature_index": raw.index})
        notes.failed_features += 1
        return Feature(
            file_id=file_id,
            index=raw.index,
            layer=raw.layer.name,
            geometry_type=(
                raw.geometry.geom_type
                if raw.geometry is not None
                else geometry.UNKNOWN_GEOMETRY_TYPE
            ),
            source_crs=raw.layer.crs_label,
            reason=ReasonCode.MEASUREMENT_FAILED,
        )


def _feature(file_id: str, raw: RawFeature, densify_deg: float, notes: _Notes) -> Feature:
    layer = raw.layer
    normalised = geometry.normalise(raw.geometry)
    notes.had_z = notes.had_z or normalised.had_z

    if normalised.reason is not None:
        outcome = measure.unmeasured(normalised.reason)
    else:
        outcome = measure.measure(normalised.geometry, layer.crs, densify_deg=densify_deg)

    display = _display_geometry(normalised.geometry, layer.crs)
    if display is not None and layer.crs is not None:
        min_lon, _, max_lon, _ = display.bounds
        if max_lon - min_lon > 180:
            notes.wide_features.append(raw.index)

    return Feature(
        file_id=file_id,
        index=raw.index,
        layer=layer.name,
        geometry_type=normalised.geometry_type,
        geometry_geojson=None if display is None else geometry.to_geojson(display),
        source_crs=layer.crs_label,
        properties=geometry.json_safe(raw.properties),
        geometry_repaired=normalised.repaired,
        measurement_kind=outcome.kind,
        area_m2=outcome.area_m2,
        perimeter_m=outcome.perimeter_m,
        length_m=outcome.length_m,
        projection_method=outcome.method,
        projected_crs=outcome.projected_crs,
        reason=outcome.reason_code,
    )


def _display_geometry(geom: BaseGeometry | None, src_crs: CRS | None) -> BaseGeometry | None:
    """Return the geometry to show in the API: the original vertices in WGS84, not densified.

    GeoJSON is defined in WGS84 (RFC 7946). When the source CRS is unknown the coordinates cannot
    be converted, so they are returned as stored; ``source_crs`` is ``null`` in that case. A
    geometry whose coordinates cannot be expressed in WGS84 is omitted.
    """
    if geom is None or src_crs is None:
        return geom
    try:
        geom_ll = crs.to_wgs84(geom, src_crs)
    except ProjError:
        return None
    return geom_ll if np.isfinite(shapely.get_coordinates(geom_ll)).all() else None


def _save(
    engine: Engine, file_id: str, layers: list[LayerData], features: list[Feature], notes: _Notes
) -> None:
    with Session(engine) as session:
        repository.complete_processing(
            session,
            file_id,
            features=features,
            layers=_layer_summaries(layers),
            crs=reader.file_crs_label(layers),
            crs_assumed=any(layer.crs_assumed for layer in layers),
            had_z=notes.had_z,
            warnings=_warnings(layers, notes),
        )


def _fail(
    engine: Engine, file_id: str, code: ErrorCode, message: str, layers: list[LayerData]
) -> None:
    try:
        with Session(engine) as session:
            repository.fail_processing(
                session, file_id, code=code, message=message, layers=_layer_summaries(layers)
            )
    except Exception:
        logger.exception("could not record the failure", extra={"error_code": code})


def _layer_summaries(layers: list[LayerData]) -> list[dict[str, Any]]:
    return [
        {
            "name": layer.name,
            "feature_count": layer.feature_count,
            "crs": layer.crs_label,
            "error": (
                None
                if layer.error is None
                else {"code": layer.error.code, "message": layer.error.message}
            ),
        }
        for layer in layers
    ]


def _warnings(layers: list[LayerData], notes: _Notes) -> list[str]:
    warnings = [warning for layer in layers for warning in layer.warnings]
    warnings.extend(
        f"{layer.name}: layer skipped. {layer.error.message}"
        for layer in layers
        if layer.error is not None
    )
    if notes.had_z:
        warnings.append(Z_DROPPED_WARNING)
    if notes.wide_features:
        warnings.append(
            f"{len(notes.wide_features)} feature(s) span more than 180 degrees of longitude and "
            "may cross the antimeridian; their measurements may be wrong. Feature indexes: "
            f"{_index_list(notes.wide_features)}."
        )
    if notes.failed_features:
        warnings.append(f"{notes.failed_features} feature(s) could not be processed.")
    return warnings


def _index_list(indexes: list[int]) -> str:
    listed = ", ".join(str(index) for index in indexes[:_MAX_LISTED_INDEXES])
    return f"{listed}, ..." if len(indexes) > _MAX_LISTED_INDEXES else listed


@contextmanager
def _timed(stage: str) -> Iterator[None]:
    """Log how long a processing stage took."""
    started = time.perf_counter()
    yield
    duration_ms = round((time.perf_counter() - started) * 1000)
    logger.info("stage finished", extra={"stage": stage, "duration_ms": duration_ms})

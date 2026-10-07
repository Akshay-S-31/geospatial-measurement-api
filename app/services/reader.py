"""Reading layers and features from Shapefile and KML sources, and resolving their CRS."""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pyogrio
from pyproj import CRS
from shapely.geometry.base import BaseGeometry

from app.core.errors import ErrorCode, ProcessingError
from app.core.logging import get_logger
from app.services.ingest import FileType

logger = get_logger(__name__)

MIXED_CRS = "MIXED"
_REQUIRED_SIDECARS = (".shx", ".dbf")
_KML_DRIVERS = ("LIBKML", "KML")


@dataclass(frozen=True, slots=True)
class LayerError:
    """Why a layer could not be read."""

    code: ErrorCode
    message: str


@dataclass(slots=True)
class LayerData:
    """One layer of a source file: a shapefile, or a KML folder.

    ``frame`` is ``None`` when the layer could not be read, in which case ``error`` says why.
    ``crs`` is ``None`` when the layer was read but its CRS is unknown.
    """

    name: str
    frame: gpd.GeoDataFrame | None = None
    crs: CRS | None = None
    crs_assumed: bool = False
    warnings: list[str] = field(default_factory=list)
    error: LayerError | None = None

    @property
    def feature_count(self) -> int:
        """Number of features read from the layer."""
        return 0 if self.frame is None else len(self.frame)

    @property
    def crs_label(self) -> str | None:
        """The layer CRS as reported by the API."""
        return crs_label(self.crs)


@dataclass(frozen=True, slots=True)
class RawFeature:
    """A feature as read from the source, before normalisation."""

    index: int
    layer: LayerData
    geometry: BaseGeometry | None
    properties: dict[str, Any]


def kml_driver() -> str | None:
    """Name of the GDAL driver that reads KML in this build, or ``None`` if there is none.

    LIBKML is preferred: it exposes every ``<Folder>`` as a layer and reads extended attributes.
    GDAL registers it ahead of the basic KML driver, so it is used whenever it is compiled in.
    """
    available = pyogrio.list_drivers()
    return next((driver for driver in _KML_DRIVERS if driver in available), None)


def read_layers(paths: Sequence[Path], file_type: FileType) -> list[LayerData]:
    """Read every layer of every source path.

    A shapefile is one layer per ``.shp``; a KML has one layer per folder. A layer that cannot be
    read is returned with its ``error`` set instead of raising, so one broken layer does not cost
    the caller the others.
    """
    layers: list[LayerData] = []
    for path in paths:
        missing = _missing_sidecars(path) if file_type is FileType.SHAPEFILE else []
        if missing:
            message = f"Shapefile '{path.stem}' is missing its {' and '.join(missing)} file."
            layers.append(_failed_layer(path.stem, ErrorCode.SHAPEFILE_INCOMPLETE, message))
        else:
            layers.extend(_read_source(path, file_type))
    return layers


def iter_features(layers: Iterable[LayerData]) -> Iterator[RawFeature]:
    """Yield every feature of the readable layers with a file-wide, 0-based index.

    The order is layer order, then feature order within the layer, so an index always refers to
    the same feature when a file is processed again.
    """
    index = 0
    for layer in layers:
        if layer.frame is None:
            continue
        attributes = layer.frame.drop(columns=layer.frame.geometry.name)
        records = attributes.to_dict(orient="records")
        for geom, properties in zip(layer.frame.geometry.array, records, strict=True):
            yield RawFeature(index, layer, geom, properties)
            index += 1


def crs_label(crs: CRS | None) -> str | None:
    """Describe a CRS as ``EPSG:<code>`` when it maps to one, otherwise by its name."""
    if crs is None:
        return None
    epsg = crs.to_epsg()
    return f"EPSG:{epsg}" if epsg is not None else crs.name


def file_crs_label(layers: Iterable[LayerData]) -> str | None:
    """Summarise the CRS of a whole file from its readable layers.

    Returns the shared label when all layers agree, ``"MIXED"`` when they differ, and ``None``
    when the CRS is unknown.
    """
    labels = {layer.crs_label for layer in layers if layer.frame is not None}
    if len(labels) > 1:
        return MIXED_CRS
    return labels.pop() if labels else None


def _missing_sidecars(shp_path: Path) -> list[str]:
    """List the mandatory companion files (.shx, .dbf) absent next to a ``.shp``."""
    siblings = {
        path.suffix.lower()
        for path in shp_path.parent.iterdir()
        if path.stem.lower() == shp_path.stem.lower()
    }
    return [suffix for suffix in _REQUIRED_SIDECARS if suffix not in siblings]


def _read_source(path: Path, file_type: FileType) -> list[LayerData]:
    try:
        listed = pyogrio.list_layers(path)
    except Exception as exc:
        logger.warning("source could not be opened", extra={"source": path.name}, exc_info=True)
        if file_type is FileType.KML:
            raise ProcessingError(ErrorCode.READ_ERROR, "The KML file could not be read.") from exc
        message = f"Shapefile '{path.stem}' could not be read."
        return [_failed_layer(path.stem, ErrorCode.READ_ERROR, message)]

    return [
        _read_layer(path, position, str(name), file_type)
        for position, (name, _geometry_type) in enumerate(listed)
    ]


def _read_layer(path: Path, position: int, name: str, file_type: FileType) -> LayerData:
    # Layers are read by position because two KML folders may share a name.
    try:
        frame = pyogrio.read_dataframe(path, layer=position)
    except Exception:
        logger.warning("layer could not be read", extra={"layer": name}, exc_info=True)
        return _failed_layer(name, ErrorCode.READ_ERROR, f"Layer '{name}' could not be read.")

    if file_type is FileType.KML:
        # OGC KML 2.2 fixes the CRS to WGS84 lon/lat; the file itself cannot declare another.
        return LayerData(name, _drop_all_null_columns(frame), crs=CRS.from_epsg(4326))
    return _with_resolved_crs(LayerData(name, frame, crs=frame.crs))


def _with_resolved_crs(layer: LayerData) -> LayerData:
    """Handle a shapefile that has no ``.prj``.

    The CRS is assumed to be EPSG:4326 only when every coordinate fits within lon/lat bounds, and
    the assumption is flagged. Coordinates outside those bounds are clearly projected, but in an
    unknowable CRS, so the layer is left without one rather than guessed.
    """
    if layer.crs is not None:
        return layer
    if _within_lon_lat_bounds(layer.frame):
        layer.crs = CRS.from_epsg(4326)
        layer.crs_assumed = True
        layer.warnings.append(
            f"{layer.name}: No .prj found; coordinates look geographic, assumed EPSG:4326."
        )
    else:
        layer.warnings.append(
            f"{layer.name}: No .prj found and coordinates are not lon/lat; the CRS is unknown, "
            "so features are not measured and geometry is returned in file coordinates."
        )
    return layer


def _within_lon_lat_bounds(frame: gpd.GeoDataFrame) -> bool:
    min_x, min_y, max_x, max_y = frame.total_bounds
    if not np.isfinite([min_x, min_y, max_x, max_y]).all():
        return False
    return bool(min_x >= -180 and max_x <= 180 and min_y >= -90 and max_y <= 90)


def _drop_all_null_columns(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop attribute columns with no value in any row.

    The LIBKML driver adds a fixed set of columns (``timestamp``, ``begin``, ``end``, ``icon``
    and others) to every layer whether or not the file uses them.
    """
    attributes = frame.columns.drop(frame.geometry.name)
    empty = [column for column in attributes if frame[column].isna().all()]
    return frame.drop(columns=empty)


def _failed_layer(name: str, code: ErrorCode, message: str) -> LayerData:
    return LayerData(name, error=LayerError(code, message))

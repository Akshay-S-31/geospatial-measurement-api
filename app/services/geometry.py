"""Geometry normalisation and JSON-safe conversion of geometries and attributes."""

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import MultiLineString, MultiPolygon
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry

from app.services.reasons import ReasonCode

UNKNOWN_GEOMETRY_TYPE = "Unknown"
GEOJSON_COORD_DECIMALS = 8

_POLYGONAL_TYPES = frozenset({"Polygon", "MultiPolygon"})
_LINEAL_TYPES = frozenset({"LineString", "MultiLineString", "LinearRing"})


@dataclass(frozen=True, slots=True)
class NormalisedGeometry:
    """A geometry made ready for measurement.

    ``geometry`` is 2D and valid, or ``None`` when the feature has no usable geometry.
    ``reason`` is set when normalisation already determined that the feature cannot be measured.
    """

    geometry: BaseGeometry | None
    geometry_type: str
    repaired: bool = False
    had_z: bool = False
    reason: ReasonCode | None = None


def normalise(raw: BaseGeometry | None) -> NormalisedGeometry:
    """Return ``raw`` as a valid 2D geometry, flagging what had to be changed.

    * Z and M values are dropped: measurements are planimetric, and a stray Z must not leak into
      the output geometry.
    * Missing and empty geometries are kept as features without geometry, so feature indexes still
      line up with the source file.
    * Invalid geometries (for example a self-intersecting "bow-tie" polygon) are repaired with
      ``make_valid`` rather than rejected, because their area is otherwise ill-defined. The repair
      can return a mixed collection; only the parts of the original dimension are kept. If nothing
      of that dimension is left, the original geometry is kept for display and marked invalid.
    """
    if raw is None:
        return NormalisedGeometry(None, UNKNOWN_GEOMETRY_TYPE, reason=ReasonCode.EMPTY_GEOMETRY)

    had_z = bool(shapely.has_z(raw))
    geom = shapely.force_2d(raw)
    if geom.is_empty:
        return NormalisedGeometry(
            None, geom.geom_type, had_z=had_z, reason=ReasonCode.EMPTY_GEOMETRY
        )
    if geom.is_valid:
        return NormalisedGeometry(geom, geom.geom_type, had_z=had_z)

    repaired = _same_dimension_parts(shapely.make_valid(geom), geom.geom_type)
    if repaired is None:
        return NormalisedGeometry(
            geom, geom.geom_type, had_z=had_z, reason=ReasonCode.INVALID_GEOMETRY
        )
    return NormalisedGeometry(repaired, repaired.geom_type, repaired=True, had_z=had_z)


def to_geojson(geom: BaseGeometry) -> str:
    """Serialise ``geom`` as a GeoJSON geometry string.

    Coordinates are rounded to 8 decimal places (about 1 mm in degrees). That trims the
    floating-point noise reprojection adds without moving any vertex by a meaningful amount.
    Polygon rings are wound as RFC 7946 asks, exterior counter-clockwise and holes clockwise,
    which is the opposite of the Shapefile convention.
    """
    rounded = shapely.transform(geom, lambda coords: np.round(coords, GEOJSON_COORD_DECIMALS))
    return shapely.to_geojson(shapely.orient_polygons(rounded))


def json_safe(properties: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a feature's attributes into a dict that serialises to strict JSON.

    Attribute tables carry types JSON has no form for: dates become ISO-8601 strings, NaN and
    other missing-value markers become ``null``, numpy scalars become Python numbers, and binary
    values are dropped because they have no faithful text representation.
    """
    return {
        str(key): _json_value(value)
        for key, value in properties.items()
        if not isinstance(value, bytes | bytearray | memoryview)
    }


def _same_dimension_parts(repaired: BaseGeometry, original_type: str) -> BaseGeometry | None:
    """Keep the parts of ``repaired`` that have the dimension of the original geometry type."""
    if original_type in _POLYGONAL_TYPES:
        wanted, collection = "Polygon", MultiPolygon
    elif original_type in _LINEAL_TYPES:
        wanted, collection = "LineString", MultiLineString
    else:
        return repaired

    parts = [part for part in _flatten(repaired) if part.geom_type == wanted and not part.is_empty]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else collection(parts)


def _flatten(geom: BaseGeometry) -> Iterator[BaseGeometry]:
    if isinstance(geom, BaseMultipartGeometry):
        for part in geom.geoms:
            yield from _flatten(part)
    else:
        yield geom


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, np.datetime64):
        return _json_value(pd.Timestamp(value))
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if isinstance(value, Decimal):
        return _json_value(float(value))
    if isinstance(value, Mapping):
        return json_safe(value)
    if isinstance(value, list | tuple | set | np.ndarray):
        return [_json_value(item) for item in value if not isinstance(item, bytes)]
    return str(value)

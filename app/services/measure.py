"""Per-geometry measurement: area and perimeter for polygons, length for lines.

Nothing here is computed on degrees. Every geometry is densified, brought to WGS84 lon/lat and
projected to a feature-centred projection by :mod:`app.services.crs` before shapely measures it.
"""

import math
from dataclasses import dataclass
from enum import StrEnum

from pyproj import CRS
from pyproj.exceptions import ProjError
from shapely.geometry.base import BaseGeometry

from app.services import crs
from app.services.crs import LocalProjection
from app.services.reasons import REASON_MESSAGES, ReasonCode


class MeasurementKind(StrEnum):
    """What was measured for a feature."""

    AREA = "area"
    LENGTH = "length"


_KIND_BY_GEOMETRY_TYPE: dict[str, MeasurementKind] = {
    "Polygon": MeasurementKind.AREA,
    "MultiPolygon": MeasurementKind.AREA,
    "LineString": MeasurementKind.LENGTH,
    "MultiLineString": MeasurementKind.LENGTH,
    "LinearRing": MeasurementKind.LENGTH,
}
_POINT_TYPES = frozenset({"Point", "MultiPoint"})


@dataclass(frozen=True, slots=True)
class MeasureOutcome:
    """Result of measuring one geometry.

    Either ``kind`` and its values are set, or ``reason_code`` and ``reason_message`` explain why
    there is no measurement. ``method`` and ``projected_crs`` describe the projection behind the
    headline value: LAEA for area, AEQD for length. A polygon's perimeter is measured in the AEQD
    projection with the same centre.
    """

    kind: MeasurementKind | None = None
    area_m2: float | None = None
    perimeter_m: float | None = None
    length_m: float | None = None
    method: str | None = None
    projected_crs: str | None = None
    reason_code: ReasonCode | None = None
    reason_message: str | None = None

    @property
    def measured(self) -> bool:
        """Whether a measurement was produced."""
        return self.kind is not None


def unmeasured(code: ReasonCode) -> MeasureOutcome:
    """Build the outcome for a feature that is kept but not measured, with its coded reason."""
    return MeasureOutcome(reason_code=code, reason_message=REASON_MESSAGES[code])


def measure(
    geom: BaseGeometry | None,
    src_crs: CRS | None,
    *,
    densify_deg: float = crs.DEFAULT_DENSIFY_DEG,
) -> MeasureOutcome:
    """Measure ``geom``, whose coordinates are in ``src_crs``, in SI units.

    Polygons get ``area_m2`` from a local equal-area projection and ``perimeter_m`` from a local
    equidistant one; lines get ``length_m`` from the equidistant one. The two projections are used
    because no single projection preserves both area and distance.

    Geometries that cannot be measured (missing, points, mixed collections, unknown CRS, or
    coordinates the projection rejects) return an outcome with a reason instead of raising, so one
    bad feature never fails a file.
    """
    if geom is None or geom.is_empty:
        return unmeasured(ReasonCode.EMPTY_GEOMETRY)

    kind = _KIND_BY_GEOMETRY_TYPE.get(geom.geom_type)
    if kind is None:
        if geom.geom_type in _POINT_TYPES:
            return unmeasured(ReasonCode.NOT_MEASURABLE)
        return unmeasured(ReasonCode.UNSUPPORTED_GEOMETRY_TYPE)
    if src_crs is None:
        return unmeasured(ReasonCode.UNKNOWN_CRS)

    try:
        return _project_and_measure(geom, kind, src_crs, densify_deg)
    except ProjError:
        return unmeasured(ReasonCode.MEASUREMENT_FAILED)


def _project_and_measure(
    geom: BaseGeometry, kind: MeasurementKind, src_crs: CRS, densify_deg: float
) -> MeasureOutcome:
    # Densify in the source CRS, before any reprojection, so edges keep their shape.
    geom_ll = crs.to_wgs84(crs.densify(geom, src_crs, densify_deg), src_crs)
    lat0, lon0 = crs.projection_centre(geom_ll)
    if not (math.isfinite(lat0) and math.isfinite(lon0)):
        return unmeasured(ReasonCode.MEASUREMENT_FAILED)

    equidistant, aeqd_string = crs.project_local(geom_ll, LocalProjection.AEQD, lat0, lon0)
    if kind is MeasurementKind.AREA:
        equal_area, laea_string = crs.project_local(geom_ll, LocalProjection.LAEA, lat0, lon0)
        outcome = MeasureOutcome(
            kind=kind,
            area_m2=equal_area.area,
            perimeter_m=equidistant.length,
            method=LocalProjection.LAEA.value,
            projected_crs=laea_string,
        )
    else:
        outcome = MeasureOutcome(
            kind=kind,
            length_m=equidistant.length,
            method=LocalProjection.AEQD.value,
            projected_crs=aeqd_string,
        )

    values = (outcome.area_m2, outcome.perimeter_m, outcome.length_m)
    if not all(math.isfinite(value) for value in values if value is not None):
        return unmeasured(ReasonCode.MEASUREMENT_FAILED)
    return outcome

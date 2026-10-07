"""Response models for per-feature measurements."""

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, SerializerFunctionWrapHandler, model_serializer

from app.db.models import Feature
from app.schemas.pagination import Pagination
from app.services.measure import MeasurementKind
from app.services.reasons import reason_message
from app.services.repository import FeatureSummary

OUTPUT_DECIMALS = 3

_EXAMPLE = {
    "file_id": "3f9c2a7e1b0d4e6f8a1c2b3d4e5f6a7b",
    "crs": "EPSG:4326",
    "summary": {
        "feature_count": 2,
        "measured_count": 1,
        "unmeasured_count": 1,
        "total_area_m2": 1212678.319,
        "total_length_m": 0.0,
        "count_by_geometry_type": {"Point": 1, "Polygon": 1},
    },
    "pagination": {"limit": 100, "offset": 0, "total": 2, "next_offset": None},
    "features": [
        {
            "index": 0,
            "layer": "Fields",
            "geometry_type": "Polygon",
            "source_crs": "EPSG:4326",
            "properties": {"Name": "F1"},
            "geometry_repaired": False,
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[76.0, 10.0], [76.01, 10.0], [76.01, 10.01], [76.0, 10.01], [76.0, 10.0]]
                ],
            },
            "measurement": {
                "kind": "area",
                "area_m2": 1212678.319,
                "perimeter_m": 4404.91,
                "length_m": None,
                "method": "LAEA",
                "projected_crs": (
                    "+proj=laea +lat_0=10.0 +lon_0=76.0 +datum=WGS84 +units=m +no_defs"
                ),
            },
            "reason": None,
        },
        {
            "index": 1,
            "layer": "Fields",
            "geometry_type": "Point",
            "source_crs": "EPSG:4326",
            "properties": {"Name": "P1"},
            "geometry_repaired": False,
            "geometry": {"type": "Point", "coordinates": [76.0, 10.0]},
            "measurement": None,
            "reason": {"code": "NOT_MEASURABLE", "message": "Points have no area or length."},
        },
    ],
}


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, OUTPUT_DECIMALS)


class MeasurementOut(BaseModel):
    """A measurement in SI units and the projection it was computed in.

    ``method`` and ``projected_crs`` describe the projection behind the headline value: LAEA for
    ``area_m2``, AEQD for ``length_m``. ``perimeter_m`` uses AEQD with the same centre.
    """

    kind: MeasurementKind
    area_m2: float | None
    perimeter_m: float | None
    length_m: float | None
    method: str
    projected_crs: str


class ReasonOut(BaseModel):
    """Why a feature has no measurement."""

    code: str
    message: str


class FeatureOut(BaseModel):
    """One feature with either a measurement or the reason it has none."""

    index: int
    layer: str
    geometry_type: str
    source_crs: str | None
    properties: dict[str, Any]
    geometry_repaired: bool
    geometry: dict[str, Any] | None = None
    measurement: MeasurementOut | None
    reason: ReasonOut | None

    @model_serializer(mode="wrap")
    def _omit_geometry_unless_set(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Leave the ``geometry`` key out entirely when it was not requested."""
        data = handler(self)
        if "geometry" not in self.model_fields_set:
            data.pop("geometry", None)
        return data

    @classmethod
    def from_record(cls, feature: Feature, *, include_geometry: bool) -> "FeatureOut":
        """Build the response from a stored feature, rounding values for output only."""
        measurement = None
        if feature.measurement_kind is not None:
            measurement = MeasurementOut(
                kind=MeasurementKind(feature.measurement_kind),
                area_m2=_rounded(feature.area_m2),
                perimeter_m=_rounded(feature.perimeter_m),
                length_m=_rounded(feature.length_m),
                method=feature.projection_method or "",
                projected_crs=feature.projected_crs or "",
            )
        reason = None
        if feature.reason is not None:
            reason = ReasonOut(code=feature.reason, message=reason_message(feature.reason))

        fields: dict[str, Any] = {
            "index": feature.index,
            "layer": feature.layer,
            "geometry_type": feature.geometry_type,
            "source_crs": feature.source_crs,
            "properties": feature.properties,
            "geometry_repaired": feature.geometry_repaired,
            "measurement": measurement,
            "reason": reason,
        }
        if include_geometry:
            geojson = feature.geometry_geojson
            fields["geometry"] = None if geojson is None else json.loads(geojson)
        return cls(**fields)


class Summary(BaseModel):
    """Totals for the whole file, independent of paging and filtering."""

    feature_count: int
    measured_count: int
    unmeasured_count: int
    total_area_m2: float
    total_length_m: float
    count_by_geometry_type: dict[str, int]

    @classmethod
    def from_totals(cls, totals: FeatureSummary) -> "Summary":
        """Build the response from the repository's aggregates."""
        return cls(
            feature_count=totals.feature_count,
            measured_count=totals.measured_count,
            unmeasured_count=totals.unmeasured_count,
            total_area_m2=round(totals.total_area_m2, OUTPUT_DECIMALS),
            total_length_m=round(totals.total_length_m, OUTPUT_DECIMALS),
            count_by_geometry_type=totals.count_by_geometry_type,
        )


class MeasurementsPage(BaseModel):
    """A page of features with the whole-file summary."""

    model_config = ConfigDict(json_schema_extra={"examples": [_EXAMPLE]})

    file_id: str
    crs: str | None
    summary: Summary
    pagination: Pagination
    features: list[FeatureOut]

"""Geometry normalisation and JSON-safe attributes."""

import json
from datetime import UTC, date, datetime
from decimal import Decimal

import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from app.services import geometry
from app.services.measure import measure
from app.services.reasons import ReasonCode
from tests.fixtures import factory


def test_z_coordinates_are_dropped_and_flagged() -> None:
    """A 3D polygon comes out 2D with ``had_z`` set."""
    polygon = Polygon([(76.0, 10.0, 5.0), (76.1, 10.0, 6.0), (76.1, 10.1, 7.0), (76.0, 10.0, 5.0)])

    result = geometry.normalise(polygon)

    assert result.had_z
    assert not result.geometry.has_z
    assert result.geometry_type == "Polygon"
    assert not result.repaired
    assert result.reason is None


def test_valid_2d_geometry_is_returned_unchanged() -> None:
    polygon = factory.square(76.0, 10.0, 0.1)

    result = geometry.normalise(polygon)

    assert result.geometry.equals(polygon)
    assert not result.had_z
    assert not result.repaired


def test_bow_tie_polygon_is_repaired() -> None:
    """A self-intersecting polygon becomes a MultiPolygon with a positive area."""
    bow_tie = Polygon([(76.0, 10.0), (76.1, 10.1), (76.1, 10.0), (76.0, 10.1), (76.0, 10.0)])
    assert not bow_tie.is_valid

    result = geometry.normalise(bow_tie)

    assert result.repaired
    assert isinstance(result.geometry, MultiPolygon)
    assert result.geometry.is_valid
    assert result.geometry_type == "MultiPolygon"
    assert measure(result.geometry, factory.crs("EPSG:4326")).area_m2 > 0


def test_polygon_that_collapses_on_repair_is_flagged_invalid() -> None:
    """A zero-area sliver repairs to a line, so nothing polygonal is left to measure."""
    sliver = Polygon([(76.0, 10.0), (76.1, 10.0), (76.2, 10.0), (76.0, 10.0)])

    result = geometry.normalise(sliver)

    assert result.reason == ReasonCode.INVALID_GEOMETRY
    assert result.geometry_type == "Polygon"
    assert result.geometry is not None
    assert not result.repaired


def test_missing_geometry() -> None:
    """A feature without geometry is kept, typed as unknown."""
    result = geometry.normalise(None)

    assert result.geometry is None
    assert result.geometry_type == geometry.UNKNOWN_GEOMETRY_TYPE
    assert result.reason == ReasonCode.EMPTY_GEOMETRY


def test_empty_geometry_keeps_its_type() -> None:
    """An empty geometry has no coordinates but still names its type."""
    result = geometry.normalise(LineString())

    assert result.geometry is None
    assert result.geometry_type == "LineString"
    assert result.reason == ReasonCode.EMPTY_GEOMETRY


def test_properties_become_json_serialisable() -> None:
    """Dates, NaN, numpy scalars and bytes are converted or dropped."""
    properties = {
        "surveyed_at": datetime(2026, 10, 7, 8, 15, 2, tzinfo=UTC),
        "surveyed_on": date(2026, 10, 7),
        "timestamp": pd.Timestamp("2026-10-07T08:15:02"),
        "missing_date": pd.NaT,
        "missing_number": float("nan"),
        "infinite": float("inf"),
        "count": np.int64(42),
        "ratio": np.float64(0.5),
        "flag": np.bool_(True),
        "price": Decimal("12.50"),
        "blob": b"\x00\x01",
        "tags": ["a", np.int64(1)],
        "name": "Plot 1",
        "nothing": None,
    }

    safe = geometry.json_safe(properties)

    assert json.loads(json.dumps(safe, allow_nan=False)) == {
        "surveyed_at": "2026-10-07T08:15:02+00:00",
        "surveyed_on": "2026-10-07",
        "timestamp": "2026-10-07T08:15:02",
        "missing_date": None,
        "missing_number": None,
        "infinite": None,
        "count": 42,
        "ratio": 0.5,
        "flag": True,
        "price": 12.5,
        "tags": ["a", 1],
        "name": "Plot 1",
        "nothing": None,
    }
    assert "blob" not in safe


def test_geojson_output() -> None:
    assert json.loads(geometry.to_geojson(Point(76.123456789012, 10.0))) == {
        "type": "Point",
        "coordinates": [76.12345679, 10.0],
    }

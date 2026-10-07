"""Measurement correctness against geodesic ground truth."""

import pytest
import shapely
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

from app.services.measure import MeasurementKind, measure
from app.services.reasons import ReasonCode
from tests.fixtures import factory

WGS84 = factory.crs("EPSG:4326")
TOLERANCE = 1e-4  # 0.01 %


@pytest.mark.parametrize("lat", [-85, -60, -30, 0, 10, 45, 60, 85])
def test_area_matches_geodesic_at_every_latitude(lat: float) -> None:
    """A 0.1 degree square is within 0.01 % of its geodesic area from pole to pole."""
    polygon = factory.square(76.0, lat, 0.1)

    outcome = measure(polygon, WGS84)

    assert outcome.kind is MeasurementKind.AREA
    assert outcome.area_m2 == pytest.approx(factory.geodesic_area(polygon), rel=TOLERANCE)


def test_area_is_independent_of_source_crs() -> None:
    """The same ground polygon in EPSG:4326, UTM 43N and Web Mercator gives one area."""
    polygon = factory.square(76.3, 10.0, 0.1)

    geographic = measure(polygon, WGS84).area_m2
    utm = measure(factory.reproject(polygon, "EPSG:32643"), factory.crs("EPSG:32643")).area_m2
    mercator = measure(factory.reproject(polygon, "EPSG:3857"), factory.crs("EPSG:3857")).area_m2

    assert utm == pytest.approx(geographic, rel=TOLERANCE)
    assert mercator == pytest.approx(geographic, rel=TOLERANCE)
    assert geographic == pytest.approx(factory.geodesic_area(polygon), rel=TOLERANCE)


def test_web_mercator_source_is_not_measured_in_place() -> None:
    """At 60 degrees north, planar Web Mercator area is about four times too large."""
    polygon = factory.square(10.0, 60.0, 0.1)
    in_mercator = factory.reproject(polygon, "EPSG:3857")

    outcome = measure(in_mercator, factory.crs("EPSG:3857"))

    assert in_mercator.area > 3.5 * factory.geodesic_area(polygon)
    assert outcome.area_m2 == pytest.approx(factory.geodesic_area(polygon), rel=TOLERANCE)


def test_polygon_hole_is_subtracted() -> None:
    """Area is the outer ring minus the hole."""
    polygon = factory.polygon_with_hole(76.0, 10.0, 0.1, 0.04)
    outer = factory.square(76.0, 10.0, 0.1)
    hole = factory.square(76.0, 10.0, 0.04)

    outcome = measure(polygon, WGS84)

    assert outcome.area_m2 == pytest.approx(factory.geodesic_area(polygon), rel=TOLERANCE)
    assert outcome.area_m2 == pytest.approx(
        factory.geodesic_area(outer) - factory.geodesic_area(hole), rel=TOLERANCE
    )


def test_multipolygon_area_is_sum_of_parts() -> None:
    """A MultiPolygon measures as the sum of its polygons."""
    first = factory.square(76.0, 10.0, 0.1)
    second = factory.square(76.3, 10.2, 0.05)

    outcome = measure(MultiPolygon([first, second]), WGS84)

    expected = factory.geodesic_area(first) + factory.geodesic_area(second)
    assert outcome.kind is MeasurementKind.AREA
    assert outcome.area_m2 == pytest.approx(expected, rel=TOLERANCE)


@pytest.mark.parametrize("lat", [10.0, 60.0])
def test_line_length_matches_geodesic(lat: float) -> None:
    """A two-vertex line and a bent five-vertex line are within 0.01 % of geodesic length."""
    straight = LineString([(76.0, lat), (76.05, lat + 0.02)])
    bent = factory.bent_line(76.0, lat)

    for line in (straight, bent):
        outcome = measure(line, WGS84)

        assert outcome.kind is MeasurementKind.LENGTH
        assert outcome.length_m == pytest.approx(factory.geodesic_length(line), rel=TOLERANCE)
        assert outcome.area_m2 is None
        assert outcome.perimeter_m is None


def test_multilinestring_length_is_sum_of_parts() -> None:
    """A MultiLineString measures as the sum of its lines."""
    first = LineString([(76.0, 10.0), (76.05, 10.02)])
    second = factory.bent_line(76.1, 10.1)

    outcome = measure(MultiLineString([first, second]), WGS84)

    expected = factory.geodesic_length(first) + factory.geodesic_length(second)
    assert outcome.length_m == pytest.approx(expected, rel=TOLERANCE)


def test_linear_ring_is_measured_as_length() -> None:
    """A LinearRing is a closed line, so it gets a length, not an area."""
    ring = factory.square(76.0, 10.0, 0.1).exterior

    outcome = measure(ring, WGS84)

    assert outcome.kind is MeasurementKind.LENGTH
    assert outcome.length_m == pytest.approx(factory.geodesic_length(ring), rel=TOLERANCE)


def test_polygon_perimeter_matches_geodesic() -> None:
    """Perimeter, holes included, is within 0.01 % of the geodesic perimeter."""
    for polygon in (
        factory.square(76.0, 45.0, 0.1),
        factory.polygon_with_hole(76.0, 10.0, 0.1, 0.04),
    ):
        outcome = measure(polygon, WGS84)

        assert outcome.perimeter_m == pytest.approx(
            factory.geodesic_perimeter(polygon), rel=TOLERANCE
        )


def test_axis_order_is_lon_lat() -> None:
    """(lon 76, lat 10) must not be read as (lat 76, lon 10)."""
    polygon = factory.square(76.0, 10.0, 0.1)
    swapped = factory.square(10.0, 76.0, 0.1)

    outcome = measure(polygon, WGS84)

    assert outcome.area_m2 == pytest.approx(factory.geodesic_area(polygon), rel=TOLERANCE)
    assert factory.geodesic_area(swapped) < 0.3 * outcome.area_m2


def test_area_is_not_square_degrees() -> None:
    """A 0.01 degree square at the equator is about 1.2 km2, not 1e-4 "square degrees"."""
    polygon = factory.square(30.0, 0.0, 0.01)

    outcome = measure(polygon, WGS84)

    assert polygon.area == pytest.approx(1e-4)
    assert 1.2e6 < outcome.area_m2 < 1.25e6


@pytest.mark.parametrize("geom", [Point(76.0, 10.0), MultiPoint([(76.0, 10.0), (76.1, 10.1)])])
def test_points_are_not_measurable(geom: shapely.Geometry) -> None:
    """Points are kept but carry no measurement."""
    outcome = measure(geom, WGS84)

    assert not outcome.measured
    assert outcome.reason_code == ReasonCode.NOT_MEASURABLE
    assert outcome.reason_message


def test_geometry_collection_is_unsupported() -> None:
    """A mixed collection has no single measurement."""
    collection = GeometryCollection([Point(76.0, 10.0), factory.square(76.0, 10.0, 0.1)])

    outcome = measure(collection, WGS84)

    assert not outcome.measured
    assert outcome.reason_code == ReasonCode.UNSUPPORTED_GEOMETRY_TYPE


@pytest.mark.parametrize("geom", [None, Polygon(), LineString()])
def test_missing_or_empty_geometry(geom: shapely.Geometry | None) -> None:
    """Missing and empty geometries are reported, not measured as zero."""
    outcome = measure(geom, WGS84)

    assert not outcome.measured
    assert outcome.reason_code == ReasonCode.EMPTY_GEOMETRY


def test_unknown_crs_is_not_measured() -> None:
    """Without a CRS the coordinates have no unit, so nothing is guessed."""
    outcome = measure(factory.square(76.0, 10.0, 0.1), None)

    assert not outcome.measured
    assert outcome.reason_code == ReasonCode.UNKNOWN_CRS


def test_points_stay_not_measurable_without_crs() -> None:
    """A point's reason is its geometry type, whether or not the CRS is known."""
    assert measure(Point(1.0, 2.0), None).reason_code == ReasonCode.NOT_MEASURABLE


def test_source_in_us_survey_feet() -> None:
    """A State Plane source in US survey feet (EPSG:2227) still yields square metres."""
    polygon = factory.square(-122.0, 37.8, 0.05)
    line = factory.bent_line(-122.0, 37.8, step_deg=0.01)
    state_plane = factory.crs("EPSG:2227")

    area = measure(factory.reproject(polygon, "EPSG:2227"), state_plane).area_m2
    length = measure(factory.reproject(line, "EPSG:2227"), state_plane).length_m

    assert area == pytest.approx(factory.geodesic_area(polygon), rel=TOLERANCE)
    assert length == pytest.approx(factory.geodesic_length(line), rel=TOLERANCE)


def test_long_edges_are_densified_before_projection() -> None:
    """A 5 degree square is only accurate because its edges are densified."""
    polygon = factory.square(76.0, 45.0, 5.0)
    truth = factory.geodesic_area(polygon)

    densified = measure(polygon, WGS84).area_m2
    undensified = measure(polygon, WGS84, densify_deg=90.0).area_m2

    assert densified == pytest.approx(truth, rel=TOLERANCE)
    assert abs(undensified - truth) / truth > 5 * TOLERANCE


def test_outcome_reports_the_projection_used() -> None:
    """Area reports its LAEA projection, length its AEQD projection."""
    area = measure(factory.square(76.02, 10.02, 0.01), WGS84)
    length = measure(LineString([(76.0, 10.0), (76.04, 10.04)]), WGS84)

    assert area.method == "LAEA"
    assert area.projected_crs == "+proj=laea +lat_0=10.0 +lon_0=76.0 +datum=WGS84 +units=m +no_defs"
    assert length.method == "AEQD"
    assert length.projected_crs == (
        "+proj=aeqd +lat_0=10.0 +lon_0=76.0 +datum=WGS84 +units=m +no_defs"
    )


def test_coordinates_outside_the_crs_domain_do_not_raise() -> None:
    """A latitude of 95 degrees cannot be projected; the feature is flagged instead."""
    outcome = measure(shapely.box(76.0, 95.0, 76.1, 95.1), WGS84)

    assert not outcome.measured
    assert outcome.reason_code == ReasonCode.MEASUREMENT_FAILED

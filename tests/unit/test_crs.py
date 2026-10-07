"""Projection selection, transformer caching and reprojection helpers."""

import pytest
import shapely
from shapely.geometry import LineString, Point

from app.services import crs
from app.services.crs import LocalProjection
from app.services.measure import measure
from tests.fixtures import factory

WGS84 = factory.crs("EPSG:4326")


def test_transformer_cache_is_reused_for_neighbouring_features() -> None:
    """Two features whose centres round to the same 0.1 degree share transformers."""
    measure(factory.square(21.01, 52.01, 0.01), WGS84)
    hits_before, _ = crs.transformer_cache_stats()

    measure(factory.square(21.03, 52.02, 0.01), WGS84)
    hits_after, _ = crs.transformer_cache_stats()

    assert hits_after >= hits_before + 2


def test_get_transformer_returns_the_same_object_for_the_same_pair() -> None:
    first = crs.get_transformer(factory.crs("EPSG:32643"), crs.WGS84)
    second = crs.get_transformer(factory.crs("EPSG:32643"), crs.WGS84)

    assert first is second


def test_transformers_use_lon_lat_order() -> None:
    """Always_xy, so x is longitude even though EPSG:4326 is lat-first."""
    transformer = crs.get_transformer(crs.WGS84, "EPSG:32643")

    easting, northing = transformer.transform(75.0, 10.0)

    # 75 E is the central meridian of UTM zone 43, where the easting is exactly 500 km.
    assert easting == pytest.approx(500000.0, abs=1e-3)
    assert northing == pytest.approx(1105412.5, abs=1.0)


def test_local_proj_string_format() -> None:
    assert crs.local_proj_string(LocalProjection.LAEA, 10.1, 76.3) == (
        "+proj=laea +lat_0=10.1 +lon_0=76.3 +datum=WGS84 +units=m +no_defs"
    )
    assert crs.local_proj_string(LocalProjection.AEQD, -33.9, 18.4) == (
        "+proj=aeqd +lat_0=-33.9 +lon_0=18.4 +datum=WGS84 +units=m +no_defs"
    )


def test_projection_centre_is_rounded_to_a_tenth_of_a_degree() -> None:
    assert crs.projection_centre(Point(76.2641, 9.9312)) == (9.9, 76.3)


def test_projection_centre_has_no_negative_zero() -> None:
    lat0, lon0 = crs.projection_centre(Point(-0.01, -0.01))

    assert str(lat0) == "0.0"
    assert str(lon0) == "0.0"


def test_local_projection_puts_the_centre_at_the_origin() -> None:
    projected, _ = crs.project_local(Point(76.3, 9.9), LocalProjection.AEQD, 9.9, 76.3)

    assert projected.x == pytest.approx(0.0, abs=1e-6)
    assert projected.y == pytest.approx(0.0, abs=1e-6)


def test_to_wgs84_is_a_no_op_for_wgs84_sources() -> None:
    line = LineString([(76.0, 10.0), (76.1, 10.1)])

    assert crs.to_wgs84(line, WGS84) is line
    assert crs.to_wgs84(line, factory.crs("OGC:CRS84")) is line


def test_to_wgs84_reprojects_projected_sources() -> None:
    point = crs.to_wgs84(Point(500000.0, 1105412.4913), factory.crs("EPSG:32643"))

    assert point.x == pytest.approx(75.0, abs=1e-6)
    assert point.y == pytest.approx(10.0, abs=1e-6)


def test_densify_geographic_source_uses_degrees() -> None:
    line = LineString([(0.0, 0.0), (1.0, 0.0)])

    densified = crs.densify(line, WGS84, 0.01)

    assert shapely.get_num_coordinates(densified) == 101


def test_densify_projected_source_uses_one_kilometre_in_source_units() -> None:
    """A 10 km edge is split into 1 km pieces whether the CRS is in metres or in feet."""
    in_metres = crs.densify(LineString([(0.0, 0.0), (10000.0, 0.0)]), factory.crs("EPSG:32643"))
    ten_km_in_feet = 10000.0 / factory.crs("EPSG:2227").axis_info[0].unit_conversion_factor
    in_feet = crs.densify(LineString([(0.0, 0.0), (ten_km_in_feet, 0.0)]), factory.crs("EPSG:2227"))

    assert shapely.get_num_coordinates(in_metres) == 11
    assert shapely.get_num_coordinates(in_feet) == 11

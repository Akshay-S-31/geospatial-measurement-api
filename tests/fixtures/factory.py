"""Builders for test geometries and ground-truth values.

Ground truth comes from ``pyproj.Geod``, which integrates directly on the WGS84 ellipsoid and uses
no map projection at all. It is independent of the projection-based code under test.
"""

import shapely
from pyproj import CRS, Geod, Transformer
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

GEOD = Geod(ellps="WGS84")
DENSIFY_DEG = 0.01


def square(lon: float, lat: float, size_deg: float) -> Polygon:
    """A lon/lat square centred on ``(lon, lat)`` with sides of ``size_deg`` degrees."""
    half = size_deg / 2
    return shapely.box(lon - half, lat - half, lon + half, lat + half)


def polygon_with_hole(lon: float, lat: float, size_deg: float, hole_deg: float) -> Polygon:
    """A square with a concentric square hole."""
    return Polygon(
        square(lon, lat, size_deg).exterior.coords,
        holes=[square(lon, lat, hole_deg).exterior.coords],
    )


def bent_line(lon: float, lat: float, step_deg: float = 0.02) -> LineString:
    """A five-vertex zig-zag line starting at ``(lon, lat)``."""
    return LineString(
        [
            (lon, lat),
            (lon + step_deg, lat + step_deg),
            (lon + 2 * step_deg, lat),
            (lon + 3 * step_deg, lat + 2 * step_deg),
            (lon + 4 * step_deg, lat - step_deg),
        ]
    )


def reproject(geom: BaseGeometry, dst_crs: str, src_crs: str = "EPSG:4326") -> BaseGeometry:
    """Reproject ``geom`` after densifying it, so both versions trace the same ground shape."""
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    densified = shapely.segmentize(geom, DENSIFY_DEG / 10)
    return shapely.transform(densified, transformer.transform, interleaved=False)


def geodesic_area(geom: BaseGeometry) -> float:
    """Geodesic area in square metres of a lon/lat polygon or multipolygon.

    Rings are measured one by one because ``Geod`` signs each ring's area by its winding order,
    which would turn a hole wound the same way as its shell into extra area.
    """
    total = 0.0
    for polygon in _polygons(geom):
        total += _ring_area(polygon.exterior)
        total -= sum(_ring_area(hole) for hole in polygon.interiors)
    return total


def geodesic_perimeter(geom: BaseGeometry) -> float:
    """Geodesic perimeter in metres of a lon/lat polygon or multipolygon, holes included."""
    return sum(
        geodesic_length(ring)
        for polygon in _polygons(geom)
        for ring in (polygon.exterior, *polygon.interiors)
    )


def geodesic_length(geom: BaseGeometry) -> float:
    """Geodesic length in metres of a lon/lat line."""
    return GEOD.geometry_length(shapely.segmentize(geom, DENSIFY_DEG))


def _polygons(geom: BaseGeometry) -> list[Polygon]:
    return list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]


def _ring_area(ring: BaseGeometry) -> float:
    area, _ = GEOD.geometry_area_perimeter(shapely.segmentize(Polygon(ring), DENSIFY_DEG))
    return abs(area)


def crs(code: str) -> CRS:
    """Shorthand for building a pyproj CRS in tests."""
    return CRS.from_user_input(code)

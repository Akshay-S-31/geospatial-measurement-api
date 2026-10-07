"""Reproduce the accuracy figures quoted in the README.

Usage: ``python scripts/projection_accuracy.py``

Compares this service's measurements, and two common alternatives, against geodesic values
computed directly on the WGS84 ellipsoid by ``pyproj.Geod``. Run it from the repository root
with the project installed (``pip install -e .``).
"""

import logging

import shapely
from pyproj import CRS, Geod, Transformer
from shapely.geometry import LineString, Polygon

from app.services.measure import measure

logger = logging.getLogger("projection_accuracy")

GEOD = Geod(ellps="WGS84")
WGS84 = CRS.from_epsg(4326)
UTM_43N = "EPSG:32643"
WEB_MERCATOR = "EPSG:3857"
LONGITUDE = 76.5  # 1.5 degrees east of the central meridian of UTM zone 43
LATITUDES = (0, 10, 45, 60, 85)
SIZES_DEG = (0.1, 1.0, 5.0)
DENSIFY_DEG = 0.01


def square(lon: float, lat: float, size_deg: float) -> Polygon:
    """A lon/lat square centred on ``(lon, lat)``."""
    half = size_deg / 2
    return shapely.box(lon - half, lat - half, lon + half, lat + half)


def geodesic_area(polygon: Polygon) -> float:
    """Geodesic area of a lon/lat polygon, in square metres."""
    area, _ = GEOD.geometry_area_perimeter(shapely.segmentize(polygon, DENSIFY_DEG))
    return abs(area)


def geodesic_perimeter(polygon: Polygon) -> float:
    """Geodesic length of a polygon's exterior ring, in metres."""
    return GEOD.geometry_length(shapely.segmentize(polygon.exterior, DENSIFY_DEG))


def geodesic_length(line: LineString) -> float:
    """Geodesic length of a lon/lat line, in metres."""
    return GEOD.geometry_length(shapely.segmentize(line, DENSIFY_DEG))


def planar_area(polygon: Polygon, crs: str) -> float:
    """Area of ``polygon`` measured directly in ``crs``, the naive approach."""
    transformer = Transformer.from_crs(WGS84, crs, always_xy=True)
    densified = shapely.segmentize(polygon, DENSIFY_DEG)
    return shapely.transform(densified, transformer.transform, interleaved=False).area


def error_percent(value: float, truth: float) -> str:
    """Signed relative error of ``value`` against ``truth``, as a percentage."""
    return f"{100 * (value - truth) / truth:+.4f} %"


def area_table() -> list[str]:
    """Markdown table of area error by latitude for three projection strategies."""
    rows = [
        "| Latitude | Local LAEA (this service) | UTM zone 43N | Web Mercator |",
        "|---|---|---|---|",
    ]
    for lat in LATITUDES:
        polygon = square(LONGITUDE, lat, 0.1)
        truth = geodesic_area(polygon)
        rows.append(
            f"| {lat} | {error_percent(measure(polygon, WGS84).area_m2, truth)} "
            f"| {error_percent(planar_area(polygon, UTM_43N), truth)} "
            f"| {error_percent(planar_area(polygon, WEB_MERCATOR), truth)} |"
        )
    return rows


def length_table() -> list[str]:
    """Markdown table of perimeter, length and area error by feature size.

    The line follows two sides of the square, so it does not pass through the projection centre,
    where an azimuthal equidistant projection would be exact by construction.
    """
    rows = [
        "| Feature size | Latitude | Perimeter (AEQD) | L-shaped line (AEQD) | Area (LAEA) |",
        "|---|---|---|---|---|",
    ]
    for size in SIZES_DEG:
        for lat in (10, 60):
            polygon = square(LONGITUDE, lat, size)
            min_lon, min_lat, max_lon, max_lat = polygon.bounds
            line = LineString([(min_lon, min_lat), (max_lon, min_lat), (max_lon, max_lat)])
            outcome = measure(polygon, WGS84)
            rows.append(
                f"| {size} deg | {lat} "
                f"| {error_percent(outcome.perimeter_m, geodesic_perimeter(polygon))} "
                f"| {error_percent(measure(line, WGS84).length_m, geodesic_length(line))} "
                f"| {error_percent(outcome.area_m2, geodesic_area(polygon))} |"
            )
    return rows


def densification_effect() -> str:
    """Area error of a large square with and without densifying its edges."""
    polygon = square(LONGITUDE, 45, 5.0)
    truth = geodesic_area(polygon)
    with_densify = error_percent(measure(polygon, WGS84).area_m2, truth)
    without = error_percent(measure(polygon, WGS84, densify_deg=90.0).area_m2, truth)
    return f"5 deg square at 45 N: {with_densify} densified, {without} not densified"


def main() -> None:
    """Log the tables."""
    logging.basicConfig(format="%(message)s")
    logger.setLevel(logging.INFO)
    logger.info("Area error of a 0.1 deg square at longitude %s, vs geodesic\n", LONGITUDE)
    logger.info("\n".join(area_table()))
    logger.info("\nError by feature size, vs geodesic\n")
    logger.info("\n".join(length_table()))
    logger.info("\nEffect of densifying edges before projection\n")
    logger.info(densification_effect())


if __name__ == "__main__":
    main()

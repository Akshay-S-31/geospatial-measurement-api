"""Projection selection and coordinate transformation for measurement.

Coordinates in a geographic CRS such as EPSG:4326 are angles. One degree of longitude spans about
111 km at the equator and about 56 km at 60 degrees north, so an area computed on raw lon/lat is in
"square degrees" and means nothing on the ground. Reprojecting to just any metric CRS is not enough
either, because every projection distorts something: Web Mercator (EPSG:3857) inflates area by
1/cos^2(latitude), which is +100 % at 45 degrees, and a UTM zone is only accurate inside its own
6-degree strip and is undefined near the poles.

This module therefore builds a projection for each feature, centred on that feature, and picks the
projection by what is being measured:

* area: Lambert Azimuthal Equal-Area (LAEA), which preserves area everywhere;
* length and perimeter: Azimuthal Equidistant (AEQD), whose length distortion grows with the
  square of the distance from its centre and so is negligible when the centre sits on the feature.

Both are defined on the WGS84 ellipsoid, not a sphere. Every source CRS, projected or geographic,
takes the same route: source -> WGS84 lon/lat -> local projection. A source that is already
projected is not measured in place, because it may be a CRS such as Web Mercator that is unfit for
measurement.

The projection centre is rounded to 0.1 degree so that neighbouring features share one cached
transformer. That rounding is safe: LAEA is equal-area whatever its centre, so area is unaffected,
and moving an AEQD centre by at most 0.05 degree (about 5.5 km) changes a length by far less than
the 0.01 % accuracy target.
"""

import threading
from enum import StrEnum
from functools import lru_cache

import shapely
from pyproj import CRS, Transformer
from shapely.geometry.base import BaseGeometry

WGS84 = "EPSG:4326"
DEFAULT_DENSIFY_DEG = 0.01
PROJECTED_SEGMENT_METRES = 1000.0
CENTRE_DECIMALS = 1
TRANSFORMER_CACHE_SIZE = 2048

_WGS84_CRS = CRS.from_user_input(WGS84)


class LocalProjection(StrEnum):
    """Feature-centred projections used for measurement."""

    LAEA = "LAEA"
    AEQD = "AEQD"


def local_proj_string(projection: LocalProjection, lat0: float, lon0: float) -> str:
    """Return the PROJ string of a WGS84 ``projection`` centred on ``(lat0, lon0)``.

    The string is reported with every measurement so a reader can reproduce the number.
    """
    return (
        f"+proj={projection.value.lower()} +lat_0={lat0} +lon_0={lon0} "
        "+datum=WGS84 +units=m +no_defs"
    )


@lru_cache(maxsize=TRANSFORMER_CACHE_SIZE)
def _cached_transformer(src: str, dst: str, _thread_id: int) -> Transformer:
    return Transformer.from_crs(src, dst, always_xy=True)


def get_transformer(src: CRS | str, dst: CRS | str) -> Transformer:
    """Return a cached transformer from ``src`` to ``dst`` in x/y (lon/lat) axis order.

    ``always_xy=True`` matters: EPSG:4326 is officially latitude-first, and without the flag
    pyproj would read shapely's (x, y) coordinates as (lat, lon) and every result would be wrong.

    Building a transformer costs about a millisecond, so they are cached by CRS pair. The calling
    thread is part of the cache key because a PROJ context must not be used from two threads at
    once, and files are processed in a thread pool.
    """
    return _cached_transformer(_definition(src), _definition(dst), threading.get_ident())


def transformer_cache_stats() -> tuple[int, int]:
    """Return ``(hits, misses)`` of the transformer cache, to verify that reuse happens."""
    info = _cached_transformer.cache_info()
    return info.hits, info.misses


def transform(geom: BaseGeometry, transformer: Transformer) -> BaseGeometry:
    """Apply ``transformer`` to every coordinate of ``geom`` in one vectorised call."""
    return shapely.transform(geom, transformer.transform, interleaved=False)


def is_wgs84(crs: CRS) -> bool:
    """Tell whether ``crs`` is geographic WGS84, in either axis order."""
    return crs.equals(_WGS84_CRS, ignore_axis_order=True)


def to_wgs84(geom: BaseGeometry, src_crs: CRS) -> BaseGeometry:
    """Reproject ``geom`` from ``src_crs`` to WGS84 lon/lat; a no-op if it is already there."""
    if is_wgs84(src_crs):
        return geom
    return transform(geom, get_transformer(src_crs, WGS84))


def densify(
    geom: BaseGeometry, src_crs: CRS, max_segment_deg: float = DEFAULT_DENSIFY_DEG
) -> BaseGeometry:
    """Insert vertices so no edge is longer than ~1 km, in the source CRS.

    An edge is a straight line in the CRS it was drawn in. Projecting only its two end points
    would replace it with a straight line in the target projection, which is a different curve on
    the ground: for a polygon 5 degrees across that alone shifts the area by about 0.1 %.
    Densifying before reprojection makes the projected edge follow the original one.

    Geographic sources are split by ``max_segment_deg``; projected sources by 1 km expressed in
    the source's own linear unit, so files in feet are handled as well.
    """
    if src_crs.is_geographic:
        return shapely.segmentize(geom, max_segment_deg)
    metres_per_unit = src_crs.axis_info[0].unit_conversion_factor
    return shapely.segmentize(geom, PROJECTED_SEGMENT_METRES / metres_per_unit)


def projection_centre(geom_ll: BaseGeometry) -> tuple[float, float]:
    """Return ``(lat0, lon0)`` for the local projection: the centroid, rounded to 0.1 degree.

    See the module docstring for why the rounding does not cost accuracy.
    """
    centroid = geom_ll.centroid
    # Adding 0.0 turns a rounded -0.0 into 0.0 so the PROJ string stays canonical.
    return round(centroid.y, CENTRE_DECIMALS) + 0.0, round(centroid.x, CENTRE_DECIMALS) + 0.0


def project_local(
    geom_ll: BaseGeometry, projection: LocalProjection, lat0: float, lon0: float
) -> tuple[BaseGeometry, str]:
    """Project WGS84 lon/lat ``geom_ll`` to a local ``projection`` centred on ``(lat0, lon0)``.

    Returns the projected geometry, in metres, and the PROJ string that produced it.
    """
    proj_string = local_proj_string(projection, lat0, lon0)
    return transform(geom_ll, get_transformer(WGS84, proj_string)), proj_string


def _definition(crs: CRS | str) -> str:
    return crs if isinstance(crs, str) else crs.to_wkt()

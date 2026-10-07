"""Builders for test geometries, test files and ground-truth values.

No binary fixture is committed: every Shapefile, KML and zip used by the tests is generated here.

Ground truth comes from ``pyproj.Geod``, which integrates directly on the WGS84 ellipsoid and uses
no map projection at all. It is independent of the projection-based code under test.
"""

import tempfile
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from xml.sax.saxutils import escape

import geopandas as gpd
import shapely
from pyproj import CRS, Geod, Transformer
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry

GEOD = Geod(ellps="WGS84")
DENSIFY_DEG = 0.01
KML_ALTITUDE = 12.5

Placemark = tuple[str, BaseGeometry]


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


def make_shapefile_zip(
    tmp: Path,
    gdf: gpd.GeoDataFrame,
    *,
    name: str = "layer",
    include_prj: bool = True,
    drop: Sequence[str] = (),
    nested_dir: str | None = None,
    extra_members: Iterable[tuple[str, bytes]] = (),
) -> Path:
    """Write ``gdf`` as a shapefile and zip it.

    ``drop`` removes sidecar files by extension (``".dbf"``), ``nested_dir`` places the files in
    a sub-folder of the archive, and ``extra_members`` adds arbitrary entries such as
    ``__MACOSX`` junk.
    """
    removed = {*drop, *(() if include_prj else (".prj",))}
    archive_path = _scratch_dir(tmp) / f"{name}.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in _write_shapefile(tmp, gdf, name):
            if path.suffix not in removed:
                archive.write(path, f"{nested_dir}/{path.name}" if nested_dir else path.name)
        for member_name, content in extra_members:
            archive.writestr(member_name, content)
    return archive_path


def make_multi_shapefile_zip(tmp: Path, layers: Mapping[str, gpd.GeoDataFrame]) -> Path:
    """Zip several shapefiles, one per ``{name: gdf}`` entry, into one archive."""
    archive_path = _scratch_dir(tmp) / "layers.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, gdf in layers.items():
            for path in _write_shapefile(tmp, gdf, name):
                archive.write(path, path.name)
    return archive_path


def make_kml(
    tmp: Path,
    folders: Mapping[str, Sequence[Placemark]],
    *,
    with_z: bool = True,
    name: str = "test.kml",
) -> Path:
    """Write a KML document with one ``<Folder>`` per entry of ``folders``."""
    body = "".join(
        f"<Folder><name>{escape(folder)}</name>"
        + "".join(
            f"<Placemark><name>{escape(label)}</name>{_kml_geometry(geom, with_z)}</Placemark>"
            for label, geom in placemarks
        )
        + "</Folder>"
        for folder, placemarks in folders.items()
    )
    path = _scratch_dir(tmp) / name
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        f"{body}</Document></kml>",
        encoding="utf-8",
    )
    return path


def make_zip_slip(tmp: Path, member_name: str = "../../evil.txt") -> Path:
    """A zip whose single member tries to escape the extraction directory."""
    archive_path = _scratch_dir(tmp) / "slip.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(zipfile.ZipInfo(member_name), b"escaped")
    return archive_path


def make_zip_bomb_like(tmp: Path, uncompressed_bytes: int) -> Path:
    """A small zip that expands to ``uncompressed_bytes`` of zeros."""
    archive_path = _scratch_dir(tmp) / "bomb.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("zeros.shp", b"\0" * uncompressed_bytes)
    return archive_path


def _scratch_dir(tmp: Path) -> Path:
    return Path(tempfile.mkdtemp(dir=tmp))


def _write_shapefile(tmp: Path, gdf: gpd.GeoDataFrame, name: str) -> list[Path]:
    directory = _scratch_dir(tmp)
    gdf.to_file(directory / f"{name}.shp")
    return sorted(directory.iterdir())


def _kml_geometry(geom: BaseGeometry, with_z: bool) -> str:
    if isinstance(geom, BaseMultipartGeometry):
        parts = "".join(_kml_geometry(part, with_z) for part in geom.geoms)
        return f"<MultiGeometry>{parts}</MultiGeometry>"
    if isinstance(geom, Polygon):
        rings = f"<outerBoundaryIs>{_kml_ring(geom.exterior, with_z)}</outerBoundaryIs>"
        rings += "".join(
            f"<innerBoundaryIs>{_kml_ring(hole, with_z)}</innerBoundaryIs>"
            for hole in geom.interiors
        )
        return f"<Polygon>{rings}</Polygon>"
    return f"<{geom.geom_type}>{_kml_coordinates(geom, with_z)}</{geom.geom_type}>"


def _kml_ring(ring: BaseGeometry, with_z: bool) -> str:
    return f"<LinearRing>{_kml_coordinates(ring, with_z)}</LinearRing>"


def _kml_coordinates(geom: BaseGeometry, with_z: bool) -> str:
    suffix = f",{KML_ALTITUDE}" if with_z else ""
    tuples = " ".join(f"{x},{y}{suffix}" for x, y in shapely.get_coordinates(geom).tolist())
    return f"<coordinates>{tuples}</coordinates>"

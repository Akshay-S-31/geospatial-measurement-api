"""Regenerate the sample files in ``samples/``.

Usage: ``python scripts/make_samples.py``

* ``kochi_plots.zip``: four land parcels near Kochi, India, as a Shapefile in UTM zone 43N
  (EPSG:32643) with attributes.
* ``survey.kml``: two folders, one with field polygons and one with roads and a control point,
  with altitudes so the Z handling is visible.
* ``no_prj_parcels.zip``: a Shapefile in lon/lat without a ``.prj``, to show the CRS assumption.
"""

import logging
import tempfile
import zipfile
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger("make_samples")

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
SHAPEFILE_SUFFIXES = (".shp", ".shx", ".dbf", ".prj", ".cpg")
SURVEY_ALTITUDE_M = 12.0

KOCHI_PLOTS = gpd.GeoDataFrame(
    {
        "plot_id": ["KCH-001", "KCH-002", "KCH-003", "KCH-004"],
        "owner": ["A. Menon", "R. Nair", "S. Thomas", "Kochi Corporation"],
        "land_use": ["residential", "residential", "paddy", "park"],
        "surveyed": [date(2026, 9, 14), date(2026, 9, 14), date(2026, 9, 15), date(2026, 9, 16)],
    },
    geometry=[
        Polygon([(76.2610, 9.9310), (76.2618, 9.9311), (76.2617, 9.9318), (76.2609, 9.9317)]),
        Polygon([(76.2620, 9.9311), (76.2629, 9.9312), (76.2628, 9.9319), (76.2619, 9.9318)]),
        Polygon(
            [
                (76.2640, 9.9300),
                (76.2672, 9.9303),
                (76.2670, 9.9331),
                (76.2655, 9.9336),
                (76.2638, 9.9328),
            ]
        ),
        Polygon(
            [(76.2590, 9.9340), (76.2606, 9.9340), (76.2606, 9.9352), (76.2590, 9.9352)],
            holes=[[(76.2596, 9.9344), (76.2600, 9.9344), (76.2600, 9.9348), (76.2596, 9.9348)]],
        ),
    ],
    crs="EPSG:4326",
).to_crs("EPSG:32643")

NO_PRJ_PARCELS = gpd.GeoDataFrame(
    {"parcel": ["P-1", "P-2", "P-3"], "zone": ["north", "north", "south"]},
    geometry=[
        Polygon([(77.5900, 12.9700), (77.5912, 12.9700), (77.5912, 12.9710), (77.5900, 12.9710)]),
        Polygon([(77.5915, 12.9700), (77.5930, 12.9701), (77.5929, 12.9711), (77.5915, 12.9710)]),
        Polygon([(77.5902, 12.9680), (77.5925, 12.9682), (77.5921, 12.9695), (77.5903, 12.9694)]),
    ],
    crs="EPSG:4326",
)

SURVEY_FOLDERS: dict[str, list[tuple[str, str, BaseGeometry]]] = {
    "Fields": [
        (
            "North field",
            "Paddy, surveyed by drone",
            Polygon([(76.300, 10.020), (76.306, 10.020), (76.307, 10.025), (76.301, 10.026)]),
        ),
        (
            "South field",
            "Coconut grove",
            Polygon([(76.301, 10.012), (76.308, 10.013), (76.307, 10.018), (76.300, 10.017)]),
        ),
    ],
    "Roads": [
        (
            "Access road",
            "Unpaved",
            LineString([(76.298, 10.011), (76.302, 10.015), (76.303, 10.019), (76.308, 10.022)]),
        ),
        (
            "Canal path",
            "Footpath along the canal",
            LineString([(76.299, 10.027), (76.309, 10.026)]),
        ),
        ("Control point GCP-1", "Ground control point", Point(76.3035, 10.0190)),
    ],
}


def write_shapefile_zip(frame: gpd.GeoDataFrame, name: str, *, include_prj: bool) -> Path:
    """Write ``frame`` as ``samples/<name>.zip``, optionally leaving out the ``.prj``."""
    target = SAMPLES_DIR / f"{name}.zip"
    with tempfile.TemporaryDirectory() as scratch:
        frame.to_file(Path(scratch) / f"{name}.shp")
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for suffix in SHAPEFILE_SUFFIXES:
                part = Path(scratch) / f"{name}{suffix}"
                if part.exists() and (include_prj or suffix != ".prj"):
                    archive.write(part, part.name)
    return target


def write_kml(folders: dict[str, list[tuple[str, str, BaseGeometry]]], name: str) -> Path:
    """Write ``samples/<name>.kml`` with one ``<Folder>`` per entry of ``folders``."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        "<Document>",
        f"  <name>{escape(name)}</name>",
    ]
    for folder, placemarks in folders.items():
        lines.append(f"  <Folder>\n    <name>{escape(folder)}</name>")
        for label, description, geom in placemarks:
            lines.append(
                "    <Placemark>\n"
                f"      <name>{escape(label)}</name>\n"
                f"      <description>{escape(description)}</description>\n"
                f"      {kml_geometry(geom)}\n"
                "    </Placemark>"
            )
        lines.append("  </Folder>")
    lines += ["</Document>", "</kml>", ""]

    target = SAMPLES_DIR / f"{name}.kml"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def kml_geometry(geom: BaseGeometry) -> str:
    """Render a point, line or simple polygon as a KML geometry element."""
    if isinstance(geom, Polygon):
        ring = f"<LinearRing>{kml_coordinates(geom.exterior.coords)}</LinearRing>"
        return f"<Polygon><outerBoundaryIs>{ring}</outerBoundaryIs></Polygon>"
    return f"<{geom.geom_type}>{kml_coordinates(geom.coords)}</{geom.geom_type}>"


def kml_coordinates(coords: object) -> str:
    """Render coordinates as KML ``lon,lat,alt`` tuples."""
    tuples = " ".join(f"{x},{y},{SURVEY_ALTITUDE_M}" for x, y, *_ in coords)
    return f"<coordinates>{tuples}</coordinates>"


def main() -> None:
    logging.basicConfig(format="%(message)s")
    logger.setLevel(logging.INFO)
    SAMPLES_DIR.mkdir(exist_ok=True)
    written = [
        write_shapefile_zip(KOCHI_PLOTS, "kochi_plots", include_prj=True),
        write_kml(SURVEY_FOLDERS, "survey"),
        write_shapefile_zip(NO_PRJ_PARCELS, "no_prj_parcels", include_prj=False),
    ]
    for path in written:
        logger.info(
            "wrote %s (%d bytes)", path.relative_to(SAMPLES_DIR.parent), path.stat().st_size
        )


if __name__ == "__main__":
    main()

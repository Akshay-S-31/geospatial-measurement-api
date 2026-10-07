"""Layer discovery, feature extraction and CRS resolution."""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from app.core.errors import ErrorCode, ProcessingError
from app.services import ingest, reader
from app.services.ingest import FileType
from app.services.reader import LayerData
from tests.fixtures import factory

MB = 1024 * 1024


def read_zip(archive: Path, tmp_path: Path) -> list[LayerData]:
    shapefiles = ingest.safe_extract(archive, tmp_path / "extract", max_unzipped_bytes=10 * MB)
    return reader.read_layers(shapefiles, FileType.SHAPEFILE)


def plots(crs: str | None, lon: float = 76.0, lat: float = 10.0) -> gpd.GeoDataFrame:
    frame = gpd.GeoDataFrame(
        {"name": ["north", "south"]},
        geometry=[factory.square(lon, lat, 0.01), factory.square(lon, lat - 0.05, 0.01)],
        crs="EPSG:4326",
    )
    if crs is None:
        return frame.set_crs(None, allow_override=True)
    return frame.to_crs(crs)


def test_kml_folders_become_layers(tmp_path: Path) -> None:
    """Two folders with a polygon, a line and a point; indexes run across layers."""
    kml = factory.make_kml(
        tmp_path,
        {
            "Fields": [("F1", factory.square(76.0, 10.0, 0.01))],
            "Roads": [
                ("R1", LineString([(76.0, 10.0), (76.05, 10.02)])),
                ("P1", Point(76.0, 10.0)),
            ],
        },
        with_z=True,
    )

    layers = reader.read_layers([kml], FileType.KML)
    features = list(reader.iter_features(layers))

    assert [(layer.name, layer.feature_count) for layer in layers] == [("Fields", 1), ("Roads", 2)]
    assert all(layer.crs_label == "EPSG:4326" and not layer.crs_assumed for layer in layers)
    assert reader.file_crs_label(layers) == "EPSG:4326"
    assert [feature.index for feature in features] == [0, 1, 2]
    assert [feature.layer.name for feature in features] == ["Fields", "Roads", "Roads"]
    assert [feature.geometry.geom_type for feature in features] == [
        "Polygon",
        "LineString",
        "Point",
    ]
    assert all(feature.geometry.has_z for feature in features)
    assert [feature.properties["Name"] for feature in features] == ["F1", "R1", "P1"]


def test_kml_all_null_columns_are_dropped(tmp_path: Path) -> None:
    """Columns the KML driver adds but the file never fills are removed."""
    kml = factory.make_kml(tmp_path, {"Fields": [("F1", factory.square(76.0, 10.0, 0.01))]})

    (layer,) = reader.read_layers([kml], FileType.KML)
    (feature,) = reader.iter_features([layer])

    assert feature.properties["Name"] == "F1"
    assert not {"timestamp", "begin", "end", "icon", "description"} & set(feature.properties)
    assert all(value is not None for value in feature.properties.values())


def test_kml_placemark_without_geometry_is_kept(tmp_path: Path) -> None:
    """A placemark with no geometry is still a feature."""
    kml = tmp_path / "bare.kml"
    kml.write_text(
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Folder><name>Notes</name>'
        "<Placemark><name>no geometry</name></Placemark>"
        "<Placemark><name>P</name><Point><coordinates>76,10</coordinates></Point></Placemark>"
        "</Folder></Document></kml>"
    )

    features = list(reader.iter_features(reader.read_layers([kml], FileType.KML)))

    assert [feature.geometry is None for feature in features] == [True, False]


def test_unreadable_kml_is_a_file_level_error(tmp_path: Path) -> None:
    """A KML that GDAL cannot open fails the file with ``READ_ERROR``."""
    kml = tmp_path / "broken.kml"
    kml.write_bytes(b"\x00\x01\x02 not kml at all")

    with pytest.raises(ProcessingError) as raised:
        reader.read_layers([kml], FileType.KML)

    assert raised.value.code == ErrorCode.READ_ERROR


def test_shapefile_with_prj_reports_its_crs(tmp_path: Path) -> None:
    """The CRS declared in the ``.prj`` is used and reported as an EPSG code."""
    archive = factory.make_shapefile_zip(tmp_path, plots("EPSG:32643"), name="plots")

    (layer,) = read_zip(archive, tmp_path)
    features = list(reader.iter_features([layer]))

    assert layer.name == "plots"
    assert layer.crs_label == "EPSG:32643"
    assert not layer.crs_assumed
    assert not layer.warnings
    assert [feature.properties for feature in features] == [{"name": "north"}, {"name": "south"}]


def test_shapefile_without_prj_in_lon_lat_range_is_assumed_wgs84(tmp_path: Path) -> None:
    """Lon/lat-looking coordinates are assumed EPSG:4326, and flagged."""
    archive = factory.make_shapefile_zip(tmp_path, plots("EPSG:4326"), include_prj=False)

    (layer,) = read_zip(archive, tmp_path)

    assert layer.crs_label == "EPSG:4326"
    assert layer.crs_assumed
    assert layer.warnings == [
        "layer: No .prj found; coordinates look geographic, assumed EPSG:4326."
    ]


def test_shapefile_without_prj_in_projected_range_has_unknown_crs(tmp_path: Path) -> None:
    """UTM-sized coordinates without a ``.prj`` are never guessed."""
    archive = factory.make_shapefile_zip(tmp_path, plots("EPSG:32643"), include_prj=False)

    (layer,) = read_zip(archive, tmp_path)

    assert layer.crs is None
    assert layer.crs_label is None
    assert not layer.crs_assumed
    assert layer.feature_count == 2
    assert len(layer.warnings) == 1
    assert reader.file_crs_label([layer]) is None


def test_zip_with_two_shapefiles_in_different_crs(tmp_path: Path) -> None:
    """Each ``.shp`` is a layer; differing CRSs make the file ``MIXED``."""
    archive = factory.make_multi_shapefile_zip(
        tmp_path, {"geographic": plots("EPSG:4326"), "utm": plots("EPSG:32643")}
    )

    layers = read_zip(archive, tmp_path)

    assert [(layer.name, layer.crs_label) for layer in layers] == [
        ("geographic", "EPSG:4326"),
        ("utm", "EPSG:32643"),
    ]
    assert reader.file_crs_label(layers) == "MIXED"
    assert [feature.index for feature in reader.iter_features(layers)] == [0, 1, 2, 3]


def test_shapefile_in_nested_folder_is_read(tmp_path: Path) -> None:
    """A shapefile inside a sub-folder of the archive is found."""
    archive = factory.make_shapefile_zip(tmp_path, plots("EPSG:4326"), nested_dir="survey/2026")

    (layer,) = read_zip(archive, tmp_path)

    assert layer.error is None
    assert layer.feature_count == 2


@pytest.mark.parametrize("missing", [".dbf", ".shx"])
def test_shapefile_missing_a_mandatory_sidecar(tmp_path: Path, missing: str) -> None:
    """Without its ``.dbf`` or ``.shx`` the layer fails as incomplete."""
    archive = factory.make_shapefile_zip(tmp_path, plots("EPSG:4326"), drop=[missing])

    (layer,) = read_zip(archive, tmp_path)

    assert layer.frame is None
    assert layer.feature_count == 0
    assert layer.error.code == ErrorCode.SHAPEFILE_INCOMPLETE
    assert missing in layer.error.message
    assert list(reader.iter_features([layer])) == []


def test_incomplete_layer_does_not_hide_a_good_one(tmp_path: Path) -> None:
    """One broken layer is recorded while the other layers are still read."""
    archive = factory.make_multi_shapefile_zip(
        tmp_path, {"good": plots("EPSG:4326"), "broken": plots("EPSG:4326")}
    )
    shapefiles = ingest.safe_extract(archive, tmp_path / "extract", max_unzipped_bytes=10 * MB)
    (tmp_path / "extract" / "broken.dbf").unlink()

    layers = reader.read_layers(shapefiles, FileType.SHAPEFILE)

    assert [(layer.name, layer.error is None) for layer in layers] == [
        ("broken", False),
        ("good", True),
    ]
    assert reader.file_crs_label(layers) == "EPSG:4326"
    assert len(list(reader.iter_features(layers))) == 2


def test_corrupt_shapefile_is_a_layer_error(tmp_path: Path) -> None:
    """Garbage in a ``.shp`` is reported on the layer, not raised."""
    directory = tmp_path / "extract"
    directory.mkdir()
    for suffix in (".shp", ".shx", ".dbf"):
        (directory / f"junk{suffix}").write_bytes(b"not a shapefile")

    (layer,) = reader.read_layers([directory / "junk.shp"], FileType.SHAPEFILE)

    assert layer.frame is None
    assert layer.error.code == ErrorCode.READ_ERROR


def test_kml_driver_is_available() -> None:
    assert reader.kml_driver() in {"LIBKML", "KML"}

"""End-to-end processing against a temporary database, without HTTP."""

import json
import zipfile
from datetime import date
from pathlib import Path

import geopandas as gpd
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import LineString, Point, Polygon
from sqlalchemy import Engine
from sqlmodel import Session, col, select

from app.core.config import Settings
from app.core.errors import ErrorCode
from app.db.models import Feature, FileStatus, GeoFile
from app.db.session import create_db_engine, init_db
from app.main import create_app
from app.services import ingest, measure, pipeline, reader, repository
from app.services.reasons import ReasonCode
from tests.fixtures import factory

TOLERANCE = 1e-4


@pytest.fixture
def engine(settings: Settings) -> Engine:
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    return engine


class Runner:
    """Stores a file the way the upload endpoint does, then processes it synchronously."""

    def __init__(self, engine: Engine, settings: Settings) -> None:
        self.engine = engine
        self.settings = settings

    def submit(self, path: Path, filename: str | None = None) -> str:
        with path.open("rb") as stream:
            upload = ingest.store_upload(
                stream,
                filename or path.name,
                storage_dir=self.settings.storage_dir,
                max_bytes=self.settings.max_upload_bytes,
            )
        with Session(self.engine) as session:
            repository.create_file(session, upload)
        return upload.file_id

    def run(self, path: Path, filename: str | None = None) -> tuple[GeoFile, list[Feature]]:
        file_id = self.submit(path, filename)
        pipeline.process(file_id, self.engine, self.settings)
        return self.result(file_id)

    def result(self, file_id: str) -> tuple[GeoFile, list[Feature]]:
        with Session(self.engine) as session:
            geo_file = session.get(GeoFile, file_id)
            features = session.exec(
                select(Feature).where(col(Feature.file_id) == file_id).order_by(col(Feature.index))
            ).all()
            return geo_file, list(features)


@pytest.fixture
def runner(engine: Engine, settings: Settings) -> Runner:
    return Runner(engine, settings)


def plots(crs: str | None = "EPSG:4326") -> gpd.GeoDataFrame:
    frame = gpd.GeoDataFrame(
        {"name": ["north", "south"], "surveyed": [date(2026, 10, 7), date(2026, 10, 8)]},
        geometry=[factory.square(76.3, 10.0, 0.01), factory.square(76.3, 9.9, 0.02)],
        crs="EPSG:4326",
    )
    if crs is None:
        return frame.set_crs(None, allow_override=True)
    return frame.to_crs(crs)


def test_kml_is_processed_end_to_end(tmp_path: Path, runner: Runner) -> None:
    """Layers, global indexes, measurements, flags and timings are stored."""
    field_shape = factory.square(76.0, 10.0, 0.01)
    road = LineString([(76.0, 10.0), (76.05, 10.02)])
    kml = factory.make_kml(
        tmp_path,
        {"Fields": [("F1", field_shape)], "Roads": [("R1", road), ("P1", Point(76.0, 10.0))]},
        with_z=True,
    )

    geo_file, features = runner.run(kml, "survey.kml")

    assert geo_file.status == FileStatus.COMPLETED
    assert geo_file.original_filename == "survey.kml"
    assert geo_file.file_type == "KML"
    assert geo_file.feature_count == 3
    assert geo_file.layer_count == 2
    assert geo_file.crs == "EPSG:4326"
    assert not geo_file.crs_assumed
    assert geo_file.had_z
    assert geo_file.warnings == [pipeline.Z_DROPPED_WARNING]
    assert geo_file.layers == [
        {"name": "Fields", "feature_count": 1, "crs": "EPSG:4326", "error": None},
        {"name": "Roads", "feature_count": 2, "crs": "EPSG:4326", "error": None},
    ]
    assert geo_file.error_code is None
    assert geo_file.processing_started_at >= geo_file.created_at
    assert geo_file.processed_at >= geo_file.processing_started_at
    assert geo_file.processing_ms >= 0

    polygon, line, point = features
    assert [feature.index for feature in features] == [0, 1, 2]
    assert [feature.layer for feature in features] == ["Fields", "Roads", "Roads"]
    assert [feature.geometry_type for feature in features] == ["Polygon", "LineString", "Point"]
    assert all(feature.source_crs == "EPSG:4326" for feature in features)

    assert polygon.measurement_kind == "area"
    assert polygon.area_m2 == pytest.approx(factory.geodesic_area(field_shape), rel=TOLERANCE)
    assert polygon.perimeter_m == pytest.approx(
        factory.geodesic_perimeter(field_shape), rel=TOLERANCE
    )
    assert polygon.length_m is None
    assert polygon.projection_method == "LAEA"
    assert polygon.projected_crs.startswith("+proj=laea")
    assert polygon.reason is None
    assert polygon.properties["Name"] == "F1"

    assert line.measurement_kind == "length"
    assert line.length_m == pytest.approx(factory.geodesic_length(road), rel=TOLERANCE)
    assert line.projection_method == "AEQD"

    assert point.measurement_kind is None
    assert point.reason == ReasonCode.NOT_MEASURABLE
    assert '"type":"Point"' in point.geometry_geojson.replace(" ", "")
    assert "12.5" not in point.geometry_geojson


def test_projected_shapefile_is_measured_and_returned_in_wgs84(
    tmp_path: Path, runner: Runner
) -> None:
    """A UTM source is measured via WGS84 and its GeoJSON is lon/lat."""
    archive = factory.make_shapefile_zip(tmp_path, plots("EPSG:32643"), name="plots")

    geo_file, features = runner.run(archive)

    assert geo_file.status == FileStatus.COMPLETED
    assert geo_file.file_type == "SHAPEFILE"
    assert geo_file.crs == "EPSG:32643"
    assert not geo_file.had_z
    assert geo_file.warnings == []
    north, south = features
    assert north.source_crs == "EPSG:32643"
    assert north.area_m2 == pytest.approx(
        factory.geodesic_area(factory.square(76.3, 10.0, 0.01)), rel=TOLERANCE
    )
    assert south.area_m2 == pytest.approx(
        factory.geodesic_area(factory.square(76.3, 9.9, 0.02)), rel=TOLERANCE
    )
    assert north.properties == {"name": "north", "surveyed": "2026-10-07"}
    lon, lat = Polygon(_coordinates(north)[0]).centroid.coords[0]
    assert lon == pytest.approx(76.3, abs=1e-6)
    assert lat == pytest.approx(10.0, abs=1e-6)


def test_output_geometry_is_not_densified(tmp_path: Path, runner: Runner) -> None:
    """Densification is for measuring only; the output keeps source vertices."""
    kml = factory.make_kml(tmp_path, {"Big": [("B", factory.square(76.0, 10.0, 2.0))]})

    _, (feature,) = runner.run(kml)

    assert len(_coordinates(feature)[0]) == 5


def test_missing_prj_with_lon_lat_coordinates_is_flagged(tmp_path: Path, runner: Runner) -> None:
    """The EPSG:4326 assumption is visible on the file and features are measured."""
    archive = factory.make_shapefile_zip(tmp_path, plots(), include_prj=False, name="parcels")

    geo_file, features = runner.run(archive)

    assert geo_file.status == FileStatus.COMPLETED
    assert geo_file.crs == "EPSG:4326"
    assert geo_file.crs_assumed
    assert geo_file.warnings == [
        "parcels: No .prj found; coordinates look geographic, assumed EPSG:4326."
    ]
    assert all(feature.measurement_kind == "area" for feature in features)


def test_missing_prj_with_projected_coordinates_is_not_measured(
    tmp_path: Path, runner: Runner
) -> None:
    """Unknown CRS keeps the features, completes the file, and measures nothing."""
    archive = factory.make_shapefile_zip(tmp_path, plots("EPSG:32643"), include_prj=False)

    geo_file, features = runner.run(archive)

    assert geo_file.status == FileStatus.COMPLETED
    assert geo_file.crs is None
    assert not geo_file.crs_assumed
    assert len(geo_file.warnings) == 1
    assert len(features) == 2
    for feature in features:
        assert feature.reason == ReasonCode.UNKNOWN_CRS
        assert feature.measurement_kind is None
        assert feature.source_crs is None
        assert _coordinates(feature)[0][0][0] > 180


def test_layers_with_different_crs_are_mixed(tmp_path: Path, runner: Runner) -> None:
    """The file CRS is ``MIXED`` while each feature reports its own source CRS."""
    archive = factory.make_multi_shapefile_zip(
        tmp_path, {"geographic": plots("EPSG:4326"), "utm": plots("EPSG:32643")}
    )

    geo_file, features = runner.run(archive)

    assert geo_file.crs == "MIXED"
    assert geo_file.layer_count == 2
    assert [feature.source_crs for feature in features] == [
        "EPSG:4326",
        "EPSG:4326",
        "EPSG:32643",
        "EPSG:32643",
    ]
    assert features[0].area_m2 == pytest.approx(features[2].area_m2, rel=TOLERANCE)


def test_feature_level_problems_do_not_fail_the_file(tmp_path: Path, runner: Runner) -> None:
    """Repaired, collapsed and missing geometries are all kept."""
    bow_tie = Polygon([(76.0, 10.0), (76.1, 10.1), (76.1, 10.0), (76.0, 10.1), (76.0, 10.0)])
    sliver = Polygon([(76.0, 10.0), (76.1, 10.0), (76.2, 10.0), (76.0, 10.0)])
    frame = gpd.GeoDataFrame(
        {"name": ["good", "bow tie", "sliver", "missing"]},
        geometry=[factory.square(76.0, 10.0, 0.01), bow_tie, sliver, None],
        crs="EPSG:4326",
    )

    geo_file, features = runner.run(factory.make_shapefile_zip(tmp_path, frame))

    good, repaired, collapsed, missing = features
    assert geo_file.status == FileStatus.COMPLETED
    assert geo_file.feature_count == 4

    assert good.measurement_kind == "area"
    assert not good.geometry_repaired

    assert repaired.geometry_repaired
    assert repaired.geometry_type == "MultiPolygon"
    assert repaired.area_m2 > 0

    assert collapsed.reason == ReasonCode.INVALID_GEOMETRY
    assert collapsed.measurement_kind is None
    assert collapsed.geometry_geojson is not None

    assert missing.reason == ReasonCode.EMPTY_GEOMETRY
    assert missing.geometry_geojson is None
    assert missing.properties == {"name": "missing"}


def test_unexpected_feature_error_is_isolated(
    tmp_path: Path, runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One feature that blows up is flagged; the others are still measured."""
    real_measure = measure.measure
    calls = iter(range(10))

    def flaky_measure(*args: object, **kwargs: object) -> measure.MeasureOutcome:
        if next(calls) == 0:
            raise RuntimeError("boom")
        return real_measure(*args, **kwargs)

    monkeypatch.setattr(measure, "measure", flaky_measure)

    geo_file, (first, second) = runner.run(factory.make_shapefile_zip(tmp_path, plots()))

    assert geo_file.status == FileStatus.COMPLETED
    assert first.reason == ReasonCode.MEASUREMENT_FAILED
    assert first.geometry_type == "Polygon"
    assert second.measurement_kind == "area"
    assert geo_file.warnings == ["1 feature(s) could not be processed."]


def test_feature_spanning_the_antimeridian_is_flagged(tmp_path: Path, runner: Runner) -> None:
    """A feature wider than 180 degrees of longitude gets a warning."""
    crossing = LineString([(179.9, 10.0), (-179.9, 10.0)])
    kml = factory.make_kml(
        tmp_path, {"Routes": [("crossing", crossing), ("local", LineString([(1, 1), (2, 2)]))]}
    )

    geo_file, _ = runner.run(kml)

    assert geo_file.status == FileStatus.COMPLETED
    assert any(
        "antimeridian" in warning and "Feature indexes: 0." in warning
        for warning in geo_file.warnings
    )


def test_zip_without_shapefile_fails(tmp_path: Path, runner: Runner) -> None:
    """The file ends ``FAILED`` with ``NO_SHAPEFILE`` and no features."""
    archive = tmp_path / "docs.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("readme.txt", "nothing spatial")

    geo_file, features = runner.run(archive)

    assert geo_file.status == FileStatus.FAILED
    assert geo_file.error_code == ErrorCode.NO_SHAPEFILE
    assert geo_file.error_message == "The zip archive does not contain a .shp file."
    assert geo_file.processed_at is not None
    assert features == []


def test_zip_slip_archive_fails_safely(tmp_path: Path, runner: Runner) -> None:
    """An archive with an escaping member fails and leaves nothing behind."""
    geo_file, _ = runner.run(factory.make_zip_slip(tmp_path))

    assert geo_file.status == FileStatus.FAILED
    assert geo_file.error_code == ErrorCode.ZIP_UNSAFE_PATH
    assert not list(tmp_path.rglob("evil.txt"))


def test_zip_over_the_unzipped_cap_fails(
    tmp_path: Path, engine: Engine, settings: Settings
) -> None:
    """Exceeding ``MAX_UNZIPPED_MB`` fails the file with ``ZIP_TOO_LARGE``."""
    small_cap = settings.model_copy(update={"max_unzipped_mb": 1})
    bomb = factory.make_zip_bomb_like(tmp_path, 2 * 1024 * 1024)

    geo_file, _ = Runner(engine, small_cap).run(bomb)

    assert geo_file.status == FileStatus.FAILED
    assert geo_file.error_code == ErrorCode.ZIP_TOO_LARGE


def test_file_fails_when_no_layer_is_readable(tmp_path: Path, runner: Runner) -> None:
    """With its only layer incomplete, the file fails and the layer error is kept."""
    archive = factory.make_shapefile_zip(tmp_path, plots(), drop=[".dbf"], name="plots")

    geo_file, features = runner.run(archive)

    assert geo_file.status == FileStatus.FAILED
    assert geo_file.error_code == ErrorCode.SHAPEFILE_INCOMPLETE
    assert ".dbf" in geo_file.error_message
    assert geo_file.layers[0]["name"] == "plots"
    assert geo_file.layers[0]["error"]["code"] == ErrorCode.SHAPEFILE_INCOMPLETE
    assert features == []


def test_broken_layer_is_reported_next_to_a_good_one(tmp_path: Path, runner: Runner) -> None:
    """One unreadable layer is recorded and the rest of the file completes."""
    archive = tmp_path / "two.zip"
    good = factory.make_shapefile_zip(tmp_path, plots(), name="good")
    with zipfile.ZipFile(good) as source, zipfile.ZipFile(archive, "w") as target:
        for member in source.namelist():
            target.writestr(member, source.read(member))
        target.writestr("broken.shp", b"missing its sidecars")

    geo_file, features = runner.run(archive)

    assert geo_file.status == FileStatus.COMPLETED
    assert geo_file.feature_count == 2
    assert [(layer["name"], layer["error"] is None) for layer in geo_file.layers] == [
        ("broken", False),
        ("good", True),
    ]
    assert geo_file.warnings[0].startswith("broken: layer skipped.")
    assert len(features) == 2


def test_file_without_features_fails(tmp_path: Path, runner: Runner) -> None:
    """A readable file with zero features is ``FAILED`` with ``NO_FEATURES``."""
    geo_file, _ = runner.run(factory.make_kml(tmp_path, {"Empty": []}))

    assert geo_file.status == FileStatus.FAILED
    assert geo_file.error_code == ErrorCode.NO_FEATURES


def test_unexpected_error_is_reported_without_internals(
    tmp_path: Path, runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash becomes ``INTERNAL_ERROR`` with a generic message, and never raises."""

    def explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("secret path /srv/internal/db")

    monkeypatch.setattr(reader, "read_layers", explode)

    geo_file, _ = runner.run(factory.make_kml(tmp_path, {"A": [("p", Point(1, 1))]}))

    assert geo_file.status == FileStatus.FAILED
    assert geo_file.error_code == ErrorCode.INTERNAL_ERROR
    assert geo_file.error_message == pipeline.INTERNAL_ERROR_MESSAGE
    assert "secret" not in geo_file.error_message


def test_extraction_directory_is_removed_and_upload_is_kept(
    tmp_path: Path, runner: Runner, settings: Settings
) -> None:
    """Extraction is temporary; the raw upload stays for reprocessing."""
    geo_file, _ = runner.run(factory.make_shapefile_zip(tmp_path, plots()))

    stored = list(ingest.upload_dir(settings.storage_dir, geo_file.id).iterdir())

    assert [path.name for path in stored] == [f"{geo_file.id}.zip"]


def test_processing_twice_gives_the_same_result(tmp_path: Path, runner: Runner) -> None:
    """The pipeline can be re-run for a file without duplicating its features."""
    geo_file, first = runner.run(factory.make_shapefile_zip(tmp_path, plots()))

    pipeline.process(geo_file.id, runner.engine, runner.settings)
    _, second = runner.result(geo_file.id)

    assert [(f.index, f.area_m2) for f in second] == [(f.index, f.area_m2) for f in first]


def test_processing_a_deleted_file_is_a_no_op(runner: Runner) -> None:
    """A task whose file row is gone must not raise."""
    pipeline.process("0" * 32, runner.engine, runner.settings)


def test_ten_thousand_features_are_processed_quickly(tmp_path: Path, runner: Runner) -> None:
    """10,000 features are well inside the 30 s budget."""
    count = 10_000
    frame = gpd.GeoDataFrame(
        {"plot_id": range(count)},
        geometry=[
            factory.square(76.0 + (i % 100) * 0.002, 10.0 + (i // 100) * 0.002, 0.001)
            for i in range(count)
        ],
        crs="EPSG:4326",
    )

    geo_file, features = runner.run(factory.make_shapefile_zip(tmp_path, frame))

    assert geo_file.status == FileStatus.COMPLETED
    assert len(features) == count
    assert geo_file.processing_ms < 30_000


def test_summary_and_paging_queries(tmp_path: Path, runner: Runner) -> None:
    """Totals come from SQL aggregates; pages are ordered and filterable."""
    field_shape = factory.square(76.0, 10.0, 0.01)
    road = LineString([(76.0, 10.0), (76.05, 10.02)])
    kml = factory.make_kml(
        tmp_path,
        {"Mixed": [("F", field_shape), ("R", road), ("P1", Point(76, 10)), ("P2", Point(77, 11))]},
    )
    geo_file, _ = runner.run(kml)

    with Session(runner.engine) as session:
        summary = repository.summarise(session, geo_file.id)
        page, total = repository.page_features(session, geo_file.id, limit=2, offset=1)
        points, point_total = repository.page_features(
            session, geo_file.id, limit=10, offset=0, geometry_type="Point"
        )

    assert summary.feature_count == 4
    assert summary.measured_count == 2
    assert summary.unmeasured_count == 2
    assert summary.total_area_m2 == pytest.approx(factory.geodesic_area(field_shape), rel=TOLERANCE)
    assert summary.total_length_m == pytest.approx(factory.geodesic_length(road), rel=TOLERANCE)
    assert summary.count_by_geometry_type == {"LineString": 1, "Point": 2, "Polygon": 1}
    assert total == 4
    assert [feature.index for feature in page] == [1, 2]
    assert point_total == 2
    assert [feature.index for feature in points] == [2, 3]


def test_interrupted_files_are_failed_on_startup(settings: Settings, engine: Engine) -> None:
    """Files left unfinished by a restart are failed instead of hanging forever."""
    with Session(engine) as session:
        for file_id, status in (("a" * 32, "PROCESSING"), ("b" * 32, "PENDING")):
            session.add(
                GeoFile(
                    id=file_id,
                    original_filename="x.kml",
                    file_type="KML",
                    size_bytes=1,
                    status=status,
                )
            )
        session.add(
            GeoFile(
                id="c" * 32,
                original_filename="done.kml",
                file_type="KML",
                size_bytes=1,
                status="COMPLETED",
            )
        )
        session.commit()

    with TestClient(create_app(settings)):
        pass

    with Session(engine) as session:
        statuses = {geo_file.id[0]: geo_file for geo_file in session.exec(select(GeoFile)).all()}
    assert statuses["a"].status == FileStatus.FAILED
    assert statuses["a"].error_code == ErrorCode.INTERRUPTED
    assert statuses["b"].status == FileStatus.FAILED
    assert statuses["c"].status == FileStatus.COMPLETED
    assert statuses["c"].error_code is None


def _coordinates(feature: Feature) -> list:
    return json.loads(feature.geometry_geojson)["coordinates"]

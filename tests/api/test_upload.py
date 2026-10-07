"""POST /api/files/."""

import zipfile
from pathlib import Path

import geopandas as gpd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from tests.api.helpers import assert_error, upload, upload_bytes
from tests.fixtures import factory


def test_upload_kml_then_read_info_and_measurements(client: TestClient, survey_kml: Path) -> None:
    """202 with the resource, then COMPLETED info, then measurements with a summary."""
    accepted = upload(client, survey_kml)

    assert accepted.status_code == 202
    body = accepted.json()
    file_id = body["id"]
    assert set(body) == {
        "id",
        "filename",
        "file_type",
        "status",
        "size_bytes",
        "created_at",
        "links",
    }
    assert body["filename"] == "survey.kml"
    assert body["file_type"] == "KML"
    assert body["status"] == "PENDING"
    assert body["size_bytes"] == survey_kml.stat().st_size
    assert body["created_at"].endswith("Z")
    assert body["links"] == {
        "self": f"/api/files/{file_id}/",
        "measurements": f"/api/files/{file_id}/measurements/",
    }
    assert accepted.headers["location"] == f"/api/files/{file_id}/"

    info = client.get(body["links"]["self"])
    assert info.status_code == 200
    assert info.json()["status"] == "COMPLETED"
    assert info.json()["feature_count"] == 3

    measurements = client.get(body["links"]["measurements"])
    assert measurements.status_code == 200
    page = measurements.json()
    assert page["file_id"] == file_id
    assert page["summary"]["feature_count"] == 3
    assert page["summary"]["measured_count"] == 2
    assert page["summary"]["unmeasured_count"] == 1
    assert page["summary"]["count_by_geometry_type"] == {
        "LineString": 1,
        "Point": 1,
        "Polygon": 1,
    }
    assert len(page["features"]) == 3


def test_upload_shapefile_zip(client: TestClient, tmp_path: Path) -> None:
    """A zipped shapefile completes with its EPSG code and SHAPEFILE type."""
    frame = gpd.GeoDataFrame(
        {"plot": ["A", "B"]},
        geometry=[factory.square(76.3, 10.0, 0.01), factory.square(76.3, 9.9, 0.01)],
        crs="EPSG:4326",
    ).to_crs("EPSG:32643")
    archive = factory.make_shapefile_zip(tmp_path, frame, name="kochi_plots")

    accepted = upload(client, archive)
    info = client.get(accepted.json()["links"]["self"]).json()
    page = client.get(accepted.json()["links"]["measurements"]).json()

    assert accepted.status_code == 202
    assert accepted.json()["file_type"] == "SHAPEFILE"
    assert info["status"] == "COMPLETED"
    assert info["file_type"] == "SHAPEFILE"
    assert info["crs"] == "EPSG:32643"
    assert info["feature_count"] == 2
    assert page["crs"] == "EPSG:32643"
    assert [feature["properties"] for feature in page["features"]] == [{"plot": "A"}, {"plot": "B"}]
    assert all(feature["measurement"]["kind"] == "area" for feature in page["features"])


def test_unsupported_extension_is_415(client: TestClient) -> None:
    """A ``.txt`` upload is refused."""
    error = assert_error(
        upload_bytes(client, b"plain text", "notes.txt"), 415, "UNSUPPORTED_FILE_TYPE"
    )

    assert error["details"]["allowed"] == [".kml", ".zip"]


def test_kmz_is_415(client: TestClient) -> None:
    """KMZ is out of scope and rejected by extension."""
    assert_error(upload_bytes(client, b"PK\x03\x04", "area.kmz"), 415, "UNSUPPORTED_FILE_TYPE")


def test_zip_with_random_bytes_is_422(client: TestClient) -> None:
    """A ``.zip`` that is not an archive."""
    assert_error(upload_bytes(client, b"\x13\x37 random bytes", "shapes.zip"), 422, "INVALID_ZIP")


def test_kml_that_is_not_kml_is_422(client: TestClient) -> None:
    """A ``.kml`` that is not XML, and one that is XML but not KML."""
    assert_error(upload_bytes(client, b"not xml at all", "a.kml"), 422, "INVALID_KML")
    assert_error(upload_bytes(client, b"<html><body/></html>", "b.kml"), 422, "INVALID_KML")


def test_empty_file_is_422(client: TestClient) -> None:
    """Zero bytes."""
    assert_error(upload_bytes(client, b"", "empty.kml"), 422, "EMPTY_FILE")


def test_file_over_the_size_limit_is_413(settings: Settings) -> None:
    """With ``MAX_UPLOAD_MB=1`` a file just over 1 MB is refused and nothing is kept."""
    limited: FastAPI = create_app(settings.model_copy(update={"max_upload_mb": 1}))

    with TestClient(limited) as client:
        response = upload_bytes(client, b"x" * (1024 * 1024 + 1), "big.kml")

    error = assert_error(response, 413, "FILE_TOO_LARGE")
    assert error["details"] == {"max_upload_mb": 1}
    assert not any(settings.storage_dir.iterdir())


def test_missing_file_field_is_422_in_the_uniform_shape(client: TestClient) -> None:
    """FastAPI's own validation error is mapped to the same error body."""
    response = client.post("/api/files/", files={"document": ("survey.kml", b"<kml/>")})

    error = assert_error(response, 422, "VALIDATION_ERROR")
    assert error["details"]["errors"][0]["loc"] == ["body", "file"]


def test_zip_without_shapefile_is_accepted_then_fails(client: TestClient, tmp_path: Path) -> None:
    """Content problems found while processing fail the file, not the upload."""
    archive = tmp_path / "docs.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("readme.txt", "nothing spatial")

    accepted = upload(client, archive)
    info = client.get(accepted.json()["links"]["self"]).json()
    measurements = client.get(accepted.json()["links"]["measurements"])

    assert accepted.status_code == 202
    assert info["status"] == "FAILED"
    assert info["error"] == {
        "code": "NO_SHAPEFILE",
        "message": "The zip archive does not contain a .shp file.",
    }
    assert info["feature_count"] == 0
    error = assert_error(measurements, 409, "FILE_FAILED")
    assert error["details"]["error"]["code"] == "NO_SHAPEFILE"


def test_stored_file_never_uses_the_client_filename(
    client: TestClient, settings: Settings, survey_kml: Path
) -> None:
    """A hostile filename is reduced to its last component and never used as a path."""
    accepted = upload(client, survey_kml, "../../../tmp/evil.kml")

    file_id = accepted.json()["id"]
    assert accepted.status_code == 202
    assert accepted.json()["filename"] == "evil.kml"
    assert [path.name for path in (settings.storage_dir / file_id).iterdir()] == [f"{file_id}.kml"]

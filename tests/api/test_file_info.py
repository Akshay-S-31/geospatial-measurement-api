"""GET /api/files/{id}/, the list and delete endpoints, and framework errors."""

from pathlib import Path

import geopandas as gpd
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import Settings
from app.db.models import GeoFile
from tests.api.helpers import assert_error, upload
from tests.fixtures import factory


def insert_file(app: FastAPI, file_id: str, status: str) -> None:
    """Insert a file row directly, without scheduling any processing."""
    with Session(app.state.engine) as session:
        session.add(
            GeoFile(
                id=file_id,
                original_filename="queued.kml",
                file_type="KML",
                size_bytes=10,
                status=status,
            )
        )
        session.commit()


def test_file_info_has_the_fields_of_the_brief(client: TestClient, survey_id: str) -> None:
    """The brief's ``id``, ``filename``, ``feature_count``, ``crs`` and ``status`` exist."""
    body = client.get(f"/api/files/{survey_id}/").json()

    assert {"id", "filename", "feature_count", "crs", "status"} <= set(body)
    assert body["id"] == survey_id
    assert body["filename"] == "survey.kml"
    assert body["feature_count"] == 3
    assert body["crs"] == "EPSG:4326"
    assert body["status"] == "COMPLETED"


def test_file_info_full_shape(client: TestClient, survey_id: str) -> None:
    """The complete file resource."""
    body = client.get(f"/api/files/{survey_id}/").json()

    assert set(body) == {
        "id",
        "filename",
        "file_type",
        "status",
        "feature_count",
        "crs",
        "crs_assumed",
        "had_z",
        "layers",
        "warnings",
        "error",
        "size_bytes",
        "created_at",
        "processed_at",
        "processing_ms",
        "links",
    }
    assert body["file_type"] == "KML"
    assert body["crs_assumed"] is False
    assert body["had_z"] is True
    assert body["layers"] == [
        {"name": "Fields", "feature_count": 1, "crs": "EPSG:4326", "error": None},
        {"name": "Roads", "feature_count": 2, "crs": "EPSG:4326", "error": None},
    ]
    assert body["warnings"] == ["Z coordinates were dropped; measurements are 2D (planimetric)."]
    assert body["error"] is None
    assert body["processed_at"].endswith("Z")
    assert body["processing_ms"] >= 0
    assert body["links"]["measurements"] == f"/api/files/{survey_id}/measurements/"


def test_unknown_id_is_404(client: TestClient) -> None:
    """Info and measurements both answer ``FILE_NOT_FOUND``."""
    assert_error(client.get("/api/files/doesnotexist/"), 404, "FILE_NOT_FOUND")
    assert_error(client.get("/api/files/doesnotexist/measurements/"), 404, "FILE_NOT_FOUND")
    assert_error(client.delete("/api/files/doesnotexist/"), 404, "FILE_NOT_FOUND")


def test_pending_file_info_and_measurements(app: FastAPI, client: TestClient) -> None:
    """An unfinished file is readable, but its measurements answer ``FILE_NOT_READY``."""
    for file_id, status in (("a" * 32, "PENDING"), ("b" * 32, "PROCESSING")):
        insert_file(app, file_id, status)

        info = client.get(f"/api/files/{file_id}/")
        measurements = client.get(f"/api/files/{file_id}/measurements/")

        assert info.status_code == 200
        assert info.json()["status"] == status
        assert info.json()["processed_at"] is None
        error = assert_error(measurements, 409, "FILE_NOT_READY")
        assert error["details"] == {"status": status}


def test_paths_without_trailing_slash_redirect(client: TestClient, survey_id: str) -> None:
    """The canonical URLs end with a slash, as in the brief."""
    response = client.get(f"/api/files/{survey_id}", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"].endswith(f"/api/files/{survey_id}/")


def test_list_files_newest_first(client: TestClient, tmp_path: Path, survey_kml: Path) -> None:
    """The list is paginated and ordered newest first."""
    first = upload(client, survey_kml, "first.kml").json()["id"]
    second = upload(client, survey_kml, "second.kml").json()["id"]

    everything = client.get("/api/files/").json()
    page = client.get("/api/files/", params={"limit": 1, "offset": 0}).json()

    assert [item["id"] for item in everything["items"]] == [second, first]
    assert everything["pagination"] == {"limit": 100, "offset": 0, "total": 2, "next_offset": None}
    assert [item["filename"] for item in page["items"]] == ["second.kml"]
    assert page["pagination"] == {"limit": 1, "offset": 0, "total": 2, "next_offset": 1}


def test_delete_removes_rows_and_stored_upload(
    client: TestClient, settings: Settings, survey_id: str
) -> None:
    """After a delete the file, its features and its upload directory are gone."""
    assert (settings.storage_dir / survey_id).is_dir()

    response = client.delete(f"/api/files/{survey_id}/")

    assert response.status_code == 204
    assert response.content == b""
    assert not (settings.storage_dir / survey_id).exists()
    assert_error(client.get(f"/api/files/{survey_id}/"), 404, "FILE_NOT_FOUND")
    assert client.get("/api/files/").json()["pagination"]["total"] == 0


def test_delete_is_refused_while_processing(app: FastAPI, client: TestClient) -> None:
    """A file that is being processed cannot be deleted from under its background task."""
    insert_file(app, "a" * 32, "PROCESSING")

    assert_error(client.delete(f"/api/files/{'a' * 32}/"), 409, "FILE_NOT_READY")


def test_unknown_route_and_wrong_method_use_the_uniform_error_body(client: TestClient) -> None:
    """Framework-level errors share the error shape."""
    assert_error(client.get("/api/nothing-here"), 404, "NOT_FOUND")
    assert_error(client.put("/api/files/"), 405, "METHOD_NOT_ALLOWED")


def test_health_reports_the_kml_driver(client: TestClient) -> None:
    body = client.get("/health").json()

    assert body == {
        "status": "ok",
        "db": "ok",
        "gdal_version": body["gdal_version"],
        "kml_driver": body["kml_driver"],
    }
    assert body["kml_driver"] in {"LIBKML", "KML"}


def test_openapi_documents_every_endpoint_with_models_and_examples(client: TestClient) -> None:
    """``/docs`` shows all endpoints, their response models and example values."""
    schema = client.get("/openapi.json").json()

    assert set(schema["paths"]) == {
        "/api/files/",
        "/api/files/{file_id}/",
        "/api/files/{file_id}/measurements/",
        "/health",
    }
    measurements = schema["paths"]["/api/files/{file_id}/measurements/"]["get"]
    assert set(measurements["responses"]) == {"200", "404", "409", "422"}
    for name in ("FileAccepted", "FileOut", "MeasurementsPage", "ErrorBody"):
        assert schema["components"]["schemas"][name]["examples"]
    assert client.get("/docs").status_code == 200


def test_shapefile_layer_error_is_visible_in_file_info(client: TestClient, tmp_path: Path) -> None:
    """A file whose only layer is incomplete reports the layer and the file error."""
    frame = gpd.GeoDataFrame(geometry=[factory.square(76.0, 10.0, 0.01)], crs="EPSG:4326")
    archive = factory.make_shapefile_zip(tmp_path, frame, drop=[".shx"], name="plots")

    info = client.get(upload(client, archive).json()["links"]["self"]).json()

    assert info["status"] == "FAILED"
    assert info["error"]["code"] == "SHAPEFILE_INCOMPLETE"
    assert info["layers"] == [
        {
            "name": "plots",
            "feature_count": 0,
            "crs": None,
            "error": {
                "code": "SHAPEFILE_INCOMPLETE",
                "message": "Shapefile 'plots' is missing its .shx file.",
            },
        }
    ]

"""Request helpers and shared geometries for the API tests."""

from pathlib import Path

from fastapi.testclient import TestClient
from httpx import Response
from shapely.geometry import LineString

from tests.fixtures import factory

FIELD = factory.square(76.0, 10.0, 0.01)
ROAD = LineString([(76.0, 10.0), (76.05, 10.02)])


def upload(client: TestClient, path: Path, filename: str | None = None) -> Response:
    """POST a file from disk to the upload endpoint."""
    return upload_bytes(client, path.read_bytes(), filename or path.name)


def upload_bytes(client: TestClient, content: bytes, filename: str) -> Response:
    """POST raw bytes to the upload endpoint under ``filename``."""
    return client.post("/api/files/", files={"file": (filename, content)})


def assert_error(response: Response, status: int, code: str) -> dict:
    """Check the status and the uniform error body; return the ``error`` object."""
    assert response.status_code == status
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "details"}
    assert body["error"]["code"] == code
    assert body["error"]["message"]
    return body["error"]

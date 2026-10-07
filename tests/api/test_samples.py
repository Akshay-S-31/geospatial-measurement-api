"""The sample files shipped for reviewers stay valid inputs."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.api.helpers import upload

SAMPLES = Path(__file__).resolve().parents[2] / "samples"


@pytest.mark.parametrize(
    ("filename", "crs", "crs_assumed", "feature_count", "measured_count"),
    [
        ("kochi_plots.zip", "EPSG:32643", False, 4, 4),
        ("survey.kml", "EPSG:4326", False, 5, 4),
        ("no_prj_parcels.zip", "EPSG:4326", True, 3, 3),
    ],
)
def test_sample_file_is_processed(
    client: TestClient,
    filename: str,
    crs: str,
    crs_assumed: bool,
    feature_count: int,
    measured_count: int,
) -> None:
    links = upload(client, SAMPLES / filename).json()["links"]

    info = client.get(links["self"]).json()
    summary = client.get(links["measurements"]).json()["summary"]

    assert info["status"] == "COMPLETED"
    assert info["crs"] == crs
    assert info["crs_assumed"] is crs_assumed
    assert summary["feature_count"] == feature_count
    assert summary["measured_count"] == measured_count

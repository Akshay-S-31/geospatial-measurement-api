"""Fixtures shared by the API tests."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from shapely.geometry import Point

from tests.api.helpers import FIELD, ROAD, upload
from tests.fixtures import factory


@pytest.fixture
def survey_kml(tmp_path: Path) -> Path:
    """A KML with a polygon in one folder, and a line and a point in another."""
    return factory.make_kml(
        tmp_path,
        {"Fields": [("F1", FIELD)], "Roads": [("R1", ROAD), ("P1", Point(76.0, 10.0))]},
        name="survey.kml",
    )


@pytest.fixture
def survey_id(client: TestClient, survey_kml: Path) -> str:
    """Id of the processed ``survey_kml``."""
    return upload(client, survey_kml).json()["id"]

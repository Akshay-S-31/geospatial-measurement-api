"""GET /api/files/{id}/measurements/."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from shapely.geometry import Point

from tests.api.helpers import FIELD, ROAD, assert_error, upload
from tests.fixtures import factory

TOLERANCE = 1e-4


def measurements(client: TestClient, file_id: str, **params: object) -> dict:
    response = client.get(f"/api/files/{file_id}/measurements/", params=params)
    assert response.status_code == 200
    return response.json()


def test_measurements_response_shape_and_values(client: TestClient, survey_id: str) -> None:
    """The documented shape, with values that match geodesic ground truth."""
    page = measurements(client, survey_id)
    polygon, line, point = page["features"]

    assert set(page) == {"file_id", "crs", "summary", "pagination", "features"}
    assert page["crs"] == "EPSG:4326"
    assert page["pagination"] == {"limit": 100, "offset": 0, "total": 3, "next_offset": None}
    assert page["summary"]["total_area_m2"] == pytest.approx(
        factory.geodesic_area(FIELD), rel=TOLERANCE
    )
    assert page["summary"]["total_length_m"] == pytest.approx(
        factory.geodesic_length(ROAD), rel=TOLERANCE
    )

    assert set(polygon) == {
        "index",
        "layer",
        "geometry_type",
        "source_crs",
        "properties",
        "geometry_repaired",
        "geometry",
        "measurement",
        "reason",
    }
    assert (polygon["index"], polygon["layer"], polygon["geometry_type"]) == (
        0,
        "Fields",
        "Polygon",
    )
    assert polygon["source_crs"] == "EPSG:4326"
    assert polygon["properties"]["Name"] == "F1"
    assert polygon["geometry_repaired"] is False
    assert polygon["geometry"]["type"] == "Polygon"
    assert len(polygon["geometry"]["coordinates"][0][0]) == 2
    assert polygon["reason"] is None
    assert polygon["measurement"] == {
        "kind": "area",
        "area_m2": pytest.approx(factory.geodesic_area(FIELD), rel=TOLERANCE),
        "perimeter_m": pytest.approx(factory.geodesic_perimeter(FIELD), rel=TOLERANCE),
        "length_m": None,
        "method": "LAEA",
        "projected_crs": "+proj=laea +lat_0=10.0 +lon_0=76.0 +datum=WGS84 +units=m +no_defs",
    }

    assert (line["index"], line["layer"], line["geometry_type"]) == (1, "Roads", "LineString")
    assert line["measurement"] == {
        "kind": "length",
        "area_m2": None,
        "perimeter_m": None,
        "length_m": pytest.approx(factory.geodesic_length(ROAD), rel=TOLERANCE),
        "method": "AEQD",
        "projected_crs": "+proj=aeqd +lat_0=10.0 +lon_0=76.0 +datum=WGS84 +units=m +no_defs",
    }
    assert (point["index"], point["geometry_type"]) == (2, "Point")


def test_values_are_rounded_to_three_decimals(client: TestClient, survey_id: str) -> None:
    """API output carries millimetre precision, no more."""
    page = measurements(client, survey_id)
    values = [
        page["features"][0]["measurement"]["area_m2"],
        page["features"][0]["measurement"]["perimeter_m"],
        page["features"][1]["measurement"]["length_m"],
        page["summary"]["total_area_m2"],
        page["summary"]["total_length_m"],
    ]

    assert all(value == round(value, 3) for value in values)


def test_point_has_null_measurement_and_a_reason(client: TestClient, survey_id: str) -> None:
    """A point is returned with a reason; the file is still COMPLETED."""
    point = measurements(client, survey_id)["features"][2]

    assert point["measurement"] is None
    assert point["reason"] == {
        "code": "NOT_MEASURABLE",
        "message": "Points have no area or length.",
    }
    assert point["geometry"] == {"type": "Point", "coordinates": [76.0, 10.0]}
    assert client.get(f"/api/files/{survey_id}/").json()["status"] == "COMPLETED"


def test_pagination_over_250_features(client: TestClient, tmp_path: Path) -> None:
    """``limit=100&offset=200`` of 250 returns the last 50 and no ``next_offset``."""
    placemarks = [(f"P{i}", Point(76.0 + i * 0.001, 10.0)) for i in range(250)]
    file_id = upload(client, factory.make_kml(tmp_path, {"Points": placemarks})).json()["id"]

    first = measurements(client, file_id)
    last = measurements(client, file_id, limit=100, offset=200)

    assert len(first["features"]) == 100
    assert first["pagination"] == {"limit": 100, "offset": 0, "total": 250, "next_offset": 100}
    assert len(last["features"]) == 50
    assert last["pagination"] == {"limit": 100, "offset": 200, "total": 250, "next_offset": None}
    assert [feature["index"] for feature in last["features"]] == list(range(200, 250))
    assert last["summary"]["feature_count"] == 250


def test_geometry_type_filter_leaves_the_summary_unchanged(
    client: TestClient, survey_id: str
) -> None:
    """The filter narrows ``features`` and ``pagination.total`` only."""
    everything = measurements(client, survey_id)
    points = measurements(client, survey_id, geometry_type="Point")
    nothing = measurements(client, survey_id, geometry_type="MultiPolygon")

    assert [feature["geometry_type"] for feature in points["features"]] == ["Point"]
    assert points["pagination"]["total"] == 1
    assert points["summary"] == everything["summary"]
    assert nothing["features"] == []
    assert nothing["pagination"]["total"] == 0


def test_include_geometry_false_omits_the_key(client: TestClient, survey_id: str) -> None:
    """``include_geometry=false`` drops the ``geometry`` key from every feature."""
    page = measurements(client, survey_id, include_geometry="false")

    assert len(page["features"]) == 3
    assert all("geometry" not in feature for feature in page["features"])
    assert page["features"][0]["measurement"]["kind"] == "area"


@pytest.mark.parametrize(
    "params", [{"limit": 0}, {"limit": 5000}, {"offset": -1}, {"limit": "many"}]
)
def test_out_of_range_paging_is_422(client: TestClient, survey_id: str, params: dict) -> None:
    """Paging parameters outside their range are validation errors."""
    response = client.get(f"/api/files/{survey_id}/measurements/", params=params)

    assert_error(response, 422, "VALIDATION_ERROR")


def test_feature_without_geometry_has_null_geometry(client: TestClient, tmp_path: Path) -> None:
    """A placemark without geometry is listed with ``geometry: null`` and a reason."""
    kml = tmp_path / "notes.kml"
    kml.write_text(
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Folder><name>Notes</name>'
        "<Placemark><name>no geometry</name></Placemark>"
        "<Placemark><name>P</name><Point><coordinates>76,10</coordinates></Point></Placemark>"
        "</Folder></Document></kml>"
    )

    page = measurements(client, upload(client, kml).json()["id"])
    bare = page["features"][0]

    assert bare["geometry"] is None
    assert bare["geometry_type"] == "Unknown"
    assert bare["measurement"] is None
    assert bare["reason"]["code"] == "EMPTY_GEOMETRY"
    assert page["summary"]["count_by_geometry_type"] == {"Point": 1, "Unknown": 1}

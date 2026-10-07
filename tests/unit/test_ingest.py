"""Upload validation and safe zip extraction."""

import io
import stat
import zipfile
from pathlib import Path

import geopandas as gpd
import pytest

from app.core.errors import AppError, ErrorCode, ProcessingError
from app.services import ingest
from app.services.ingest import FileType
from tests.fixtures import factory

MB = 1024 * 1024

BILLION_LAUGHS_KML = b"""<?xml version="1.0"?>
<!DOCTYPE kml [
  <!ENTITY a "aaaaaaaaaa">
  <!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">
  <!ENTITY c "&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;">
]>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>&c;</name></Document></kml>
"""


def store(
    content: bytes, filename: str, storage_dir: Path, max_bytes: int = MB
) -> ingest.StoredUpload:
    return ingest.store_upload(
        io.BytesIO(content), filename, storage_dir=storage_dir, max_bytes=max_bytes
    )


def one_plot() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"name": ["plot"]}, geometry=[factory.square(76.0, 10.0, 0.01)], crs="EPSG:4326"
    )


def extract(archive: Path, dest: Path, max_unzipped_bytes: int = 10 * MB) -> list[Path]:
    return ingest.safe_extract(archive, dest, max_unzipped_bytes=max_unzipped_bytes)


def test_zip_slip_member_is_rejected(tmp_path: Path) -> None:
    """A ``../`` member fails the archive and writes nothing outside the target."""
    dest = tmp_path / "work" / "extract"

    with pytest.raises(ProcessingError) as raised:
        extract(factory.make_zip_slip(tmp_path), dest)

    assert raised.value.code == ErrorCode.ZIP_UNSAFE_PATH
    assert not list(tmp_path.rglob("evil.txt"))
    assert not any(dest.iterdir())


@pytest.mark.parametrize("member", ["/etc/evil.txt", "C:\\Windows\\evil.txt"])
def test_absolute_path_member_is_rejected(tmp_path: Path, member: str) -> None:
    """Absolute POSIX and Windows paths are rejected."""
    with pytest.raises(ProcessingError) as raised:
        extract(factory.make_zip_slip(tmp_path, member), tmp_path / "extract")

    assert raised.value.code == ErrorCode.ZIP_UNSAFE_PATH


def test_symlink_member_is_rejected(tmp_path: Path) -> None:
    """A symlink entry could redirect a later write, so it is refused."""
    archive_path = tmp_path / "link.zip"
    link = zipfile.ZipInfo("link.shp")
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(link, "/etc/passwd")

    with pytest.raises(ProcessingError) as raised:
        extract(archive_path, tmp_path / "extract")

    assert raised.value.code == ErrorCode.ZIP_UNSAFE_PATH


def test_uncompressed_size_is_capped(tmp_path: Path) -> None:
    """A small archive that expands past the cap is refused before extraction."""
    bomb = factory.make_zip_bomb_like(tmp_path, 2 * MB)
    dest = tmp_path / "extract"
    assert bomb.stat().st_size < MB // 10

    with pytest.raises(ProcessingError) as raised:
        extract(bomb, dest, max_unzipped_bytes=MB)

    assert raised.value.code == ErrorCode.ZIP_TOO_LARGE
    assert not any(dest.iterdir())


def test_macos_junk_and_hidden_files_are_ignored(tmp_path: Path) -> None:
    """``__MACOSX`` resource forks and dotfiles are neither extracted nor read."""
    archive = factory.make_shapefile_zip(
        tmp_path,
        one_plot(),
        extra_members=[
            ("__MACOSX/._layer.shp", b"resource fork"),
            (".DS_Store", b"finder metadata"),
            ("data/.hidden.shp", b"hidden"),
        ],
    )
    dest = tmp_path / "extract"

    shapefiles = extract(archive, dest)

    assert [path.name for path in shapefiles] == ["layer.shp"]
    assert not (dest / "__MACOSX").exists()
    assert not (dest / ".DS_Store").exists()
    assert not (dest / "data").exists()


def test_archive_without_shapefile(tmp_path: Path) -> None:
    """A valid zip with no ``.shp`` in it fails with ``NO_SHAPEFILE``."""
    archive_path = tmp_path / "docs.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("readme.txt", "no shapes here")

    with pytest.raises(ProcessingError) as raised:
        extract(archive_path, tmp_path / "extract")

    assert raised.value.code == ErrorCode.NO_SHAPEFILE


def test_shapefiles_are_found_in_nested_folders(tmp_path: Path) -> None:
    """The ``.shp`` may sit in a sub-folder, with an upper-case extension."""
    archive_path = tmp_path / "nested.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("survey/2026/PLOTS.SHP", b"shp")
        archive.writestr("survey/2026/PLOTS.DBF", b"dbf")

    shapefiles = extract(archive_path, tmp_path / "extract")

    assert [path.name for path in shapefiles] == ["PLOTS.SHP"]
    assert shapefiles[0].parent.name == "2026"


def test_entity_expansion_kml_is_rejected(tmp_path: Path) -> None:
    """An XML entity-expansion payload is refused and removed from disk."""
    with pytest.raises(AppError) as raised:
        store(BILLION_LAUGHS_KML, "laughs.kml", tmp_path)

    assert raised.value.code == ErrorCode.INVALID_KML
    assert raised.value.http_status == 422
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("content", "filename"),
    [
        (b"this is not xml", "notes.kml"),
        (b"<gpx><trk/></gpx>", "track.kml"),
        (b"<kml><Document>", "truncated.kml"),
    ],
)
def test_kml_content_is_checked(tmp_path: Path, content: bytes, filename: str) -> None:
    """A ``.kml`` must be well-formed XML with a ``kml`` root."""
    with pytest.raises(AppError) as raised:
        store(content, filename, tmp_path)

    assert raised.value.code == ErrorCode.INVALID_KML


def test_kml_root_is_accepted_in_any_namespace(tmp_path: Path) -> None:
    """KML 2.1, 2.2 and un-namespaced documents are all accepted."""
    for namespace in ("http://www.opengis.net/kml/2.2", "http://earth.google.com/kml/2.1", ""):
        attribute = f' xmlns="{namespace}"' if namespace else ""
        stored = store(f"<kml{attribute}><Document/></kml>".encode(), "doc.kml", tmp_path)

        assert stored.file_type is FileType.KML


def test_zip_content_is_checked(tmp_path: Path) -> None:
    """A ``.zip`` must really be a zip archive."""
    with pytest.raises(AppError) as raised:
        store(b"definitely not a zip archive", "shapes.zip", tmp_path)

    assert raised.value.code == ErrorCode.INVALID_ZIP
    assert raised.value.http_status == 422


@pytest.mark.parametrize("filename", ["notes.txt", "area.kmz", "data.geojson", "noextension", ""])
def test_unsupported_extension(tmp_path: Path, filename: str) -> None:
    """Anything other than ``.zip`` and ``.kml`` is a 415, KMZ included."""
    with pytest.raises(AppError) as raised:
        store(b"content", filename, tmp_path)

    assert raised.value.code == ErrorCode.UNSUPPORTED_FILE_TYPE
    assert raised.value.http_status == 415
    assert not any(tmp_path.iterdir())


def test_extension_check_is_case_insensitive(tmp_path: Path) -> None:
    """``SURVEY.KML`` is a KML file."""
    stored = store(b"<kml><Document/></kml>", "SURVEY.KML", tmp_path)

    assert stored.file_type is FileType.KML


def test_empty_upload(tmp_path: Path) -> None:
    """A zero-byte file is rejected with its own code."""
    with pytest.raises(AppError) as raised:
        store(b"", "empty.kml", tmp_path)

    assert raised.value.code == ErrorCode.EMPTY_FILE
    assert not any(tmp_path.iterdir())


def test_upload_over_the_size_limit(tmp_path: Path) -> None:
    """Copying stops once the cap is passed and the partial file is removed."""
    with pytest.raises(AppError) as raised:
        store(b"x" * (MB + 1), "big.kml", tmp_path, max_bytes=MB)

    assert raised.value.code == ErrorCode.FILE_TOO_LARGE
    assert raised.value.http_status == 413
    assert not any(tmp_path.iterdir())


def test_stored_file_is_named_by_id_not_by_client(tmp_path: Path) -> None:
    """The client's filename is kept for display only and never becomes a path."""
    content = b"<kml><Document/></kml>"

    stored = store(content, "../../etc/cron.d/survey.kml", tmp_path)

    assert stored.original_filename == "survey.kml"
    assert stored.path == tmp_path / stored.file_id / f"{stored.file_id}.kml"
    assert stored.path == ingest.upload_path(tmp_path, stored.file_id, FileType.KML)
    assert stored.path.read_bytes() == content
    assert stored.size_bytes == len(content)
    assert len(stored.file_id) == 32

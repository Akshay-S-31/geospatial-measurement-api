"""Upload validation, storage, and safe extraction of zip archives.

Uploaded files are untrusted. Nothing the client controls is ever used as a path: files are stored
under a generated id, and every zip member is checked before anything is written.
"""

import shutil
import stat
import uuid
import zipfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO
from xml.etree.ElementTree import ParseError

from defusedxml import DefusedXmlException
from defusedxml.ElementTree import iterparse

from app.core.errors import AppError, ErrorCode, ProcessingError
from app.core.logging import get_logger

logger = get_logger(__name__)

CHUNK_BYTES = 1024 * 1024
_BYTES_PER_MB = 1024 * 1024
_MAX_FILENAME_LENGTH = 255
_IGNORED_DIRECTORY = "__MACOSX"


class FileType(StrEnum):
    """Supported upload formats."""

    SHAPEFILE = "SHAPEFILE"
    KML = "KML"


FILE_TYPE_BY_EXTENSION: dict[str, FileType] = {
    ".zip": FileType.SHAPEFILE,
    ".kml": FileType.KML,
}
_EXTENSION_BY_FILE_TYPE = {file_type: ext for ext, file_type in FILE_TYPE_BY_EXTENSION.items()}


@dataclass(frozen=True, slots=True)
class StoredUpload:
    """An upload that passed validation and was written to disk."""

    file_id: str
    original_filename: str
    file_type: FileType
    path: Path
    size_bytes: int


def detect_file_type(filename: str) -> FileType:
    """Map a filename to its file type by extension, case-insensitively.

    The extension only selects which content check to run; it is never trusted on its own.
    """
    extension = PurePosixPath(filename).suffix.lower()
    file_type = FILE_TYPE_BY_EXTENSION.get(extension)
    if file_type is None:
        raise AppError(
            ErrorCode.UNSUPPORTED_FILE_TYPE,
            "Unsupported file type. Upload a .zip containing a Shapefile, or a .kml file.",
            http_status=415,
            details={"extension": extension, "allowed": sorted(FILE_TYPE_BY_EXTENSION)},
        )
    return file_type


def upload_dir(storage_dir: Path, file_id: str) -> Path:
    """Directory that holds everything stored for one file."""
    return storage_dir / file_id


def upload_path(storage_dir: Path, file_id: str, file_type: FileType) -> Path:
    """Path of the raw upload: the generated id plus the extension, never the client's name."""
    return upload_dir(storage_dir, file_id) / f"{file_id}{_EXTENSION_BY_FILE_TYPE[file_type]}"


def store_upload(
    stream: BinaryIO, filename: str | None, *, storage_dir: Path, max_bytes: int
) -> StoredUpload:
    """Validate an upload and write it to ``storage_dir/<id>/<id>.<ext>``.

    The stream is copied in 1 MB chunks so memory use does not depend on file size, and copying
    stops as soon as the size cap is passed. The content is then checked against its extension,
    because a renamed file must be rejected here rather than fail later in the background.

    Raises ``AppError`` with 415, 413 or 422. Nothing is left on disk when validation fails.
    """
    original_filename = _display_name(filename)
    file_type = detect_file_type(original_filename)
    file_id = uuid.uuid4().hex
    directory = upload_dir(storage_dir, file_id)
    path = upload_path(storage_dir, file_id, file_type)
    directory.mkdir(parents=True)

    try:
        size_bytes = _copy_with_limit(stream, path, max_bytes)
        _check_content(path, file_type)
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise

    return StoredUpload(file_id, original_filename, file_type, path, size_bytes)


def safe_extract(zip_path: Path, dest_dir: Path, *, max_unzipped_bytes: int) -> list[Path]:
    """Extract a zip archive into ``dest_dir`` and return its ``.shp`` files in a stable order.

    Every member is validated before any byte is written:

    * a member whose resolved path leaves ``dest_dir``, an absolute path, or a symlink is a
      zip-slip attempt and rejects the whole archive;
    * the total uncompressed size is capped (zip bomb). The sizes declared in the archive are
      enough for this, because ``zipfile`` never yields more bytes than a member declares.

    ``__MACOSX`` folders and hidden files are skipped: they are packaging noise, and macOS
    resource forks named ``._layer.shp`` would otherwise be mistaken for shapefiles.

    Raises ``ProcessingError`` with ``ZIP_UNSAFE_PATH``, ``ZIP_TOO_LARGE``, ``READ_ERROR`` or
    ``NO_SHAPEFILE``.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    root = dest_dir.resolve()

    try:
        with zipfile.ZipFile(zip_path) as archive:
            members = [
                (info, _member_target(root, info))
                for info in archive.infolist()
                if not info.is_dir()
            ]
            wanted = [(info, target) for info, target in members if not _is_ignored(info)]
            if sum(info.file_size for info, _ in wanted) > max_unzipped_bytes:
                raise ProcessingError(
                    ErrorCode.ZIP_TOO_LARGE,
                    "The zip archive expands to more than "
                    f"{max_unzipped_bytes // _BYTES_PER_MB} MB.",
                )

            for info, target in wanted:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink, CHUNK_BYTES)
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
        # RuntimeError: encrypted member. NotImplementedError: unsupported compression method.
        raise ProcessingError(
            ErrorCode.READ_ERROR, "The zip archive is corrupt or uses an unsupported feature."
        ) from exc

    shapefiles = sorted(target for _, target in wanted if target.suffix.lower() == ".shp")
    if not shapefiles:
        raise ProcessingError(
            ErrorCode.NO_SHAPEFILE, "The zip archive does not contain a .shp file."
        )
    return shapefiles


def _display_name(filename: str | None) -> str:
    """Reduce a client-supplied filename to its final component, for display only."""
    name = PurePosixPath((filename or "").replace("\\", "/")).name
    return name[:_MAX_FILENAME_LENGTH]


def _copy_with_limit(stream: BinaryIO, path: Path, max_bytes: int) -> int:
    size = 0
    with path.open("wb") as sink:
        while chunk := stream.read(CHUNK_BYTES):
            size += len(chunk)
            if size > max_bytes:
                raise AppError(
                    ErrorCode.FILE_TOO_LARGE,
                    f"The file exceeds the {max_bytes // _BYTES_PER_MB} MB upload limit.",
                    http_status=413,
                    details={"max_upload_mb": max_bytes // _BYTES_PER_MB},
                )
            sink.write(chunk)
    if size == 0:
        raise AppError(ErrorCode.EMPTY_FILE, "The uploaded file is empty.", http_status=422)
    return size


def _check_content(path: Path, file_type: FileType) -> None:
    if file_type is FileType.SHAPEFILE and not zipfile.is_zipfile(path):
        raise AppError(
            ErrorCode.INVALID_ZIP,
            "The file has a .zip extension but is not a valid zip archive.",
            http_status=422,
        )
    if file_type is FileType.KML and not _is_kml(path):
        raise AppError(
            ErrorCode.INVALID_KML,
            "The file has a .kml extension but is not a valid KML document.",
            http_status=422,
        )


def _is_kml(path: Path) -> bool:
    """Check that the file is well-formed XML with a ``kml`` root element, in any namespace.

    Parsing goes through defusedxml, which refuses entity declarations and so stops
    entity-expansion ("billion laughs") and external-entity payloads. Elements are cleared as
    they close to keep memory flat on large files.
    """
    root_seen = False
    try:
        for event, element in iterparse(path, events=("start", "end")):
            if not root_seen:
                if _local_name(element.tag) != "kml":
                    return False
                root_seen = True
            elif event == "end":
                element.clear()
    except (ParseError, DefusedXmlException):
        return False
    return root_seen


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _member_target(root: Path, info: zipfile.ZipInfo) -> Path:
    """Resolve where a zip member would be written, rejecting anything that escapes ``root``."""
    name = info.filename.replace("\\", "/")
    is_symlink = stat.S_ISLNK(info.external_attr >> 16)
    is_absolute = name.startswith("/") or bool(PureWindowsPath(name).drive)
    target = (root / name).resolve()
    if is_symlink or is_absolute or not target.is_relative_to(root):
        logger.warning("rejected unsafe zip member", extra={"member": info.filename})
        raise ProcessingError(
            ErrorCode.ZIP_UNSAFE_PATH,
            "The zip archive contains an entry that points outside the archive.",
        )
    return target


def _is_ignored(info: zipfile.ZipInfo) -> bool:
    parts = PurePosixPath(info.filename.replace("\\", "/")).parts
    return any(part == _IGNORED_DIRECTORY or part.startswith(".") for part in parts)

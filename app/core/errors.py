"""Error types shared by the service and API layers.

This module imports no web framework, so services can raise these errors and stay testable
without HTTP. The API layer turns them into the uniform error body.
"""

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    """Stable machine-readable error codes."""

    # Rejected at upload time, returned as HTTP errors.
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    EMPTY_FILE = "EMPTY_FILE"
    INVALID_ZIP = "INVALID_ZIP"
    INVALID_KML = "INVALID_KML"
    VALIDATION_ERROR = "VALIDATION_ERROR"

    # Returned by the read endpoints.
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FILE_NOT_READY = "FILE_NOT_READY"
    FILE_FAILED = "FILE_FAILED"

    # Raised during background processing and stored on the file or one of its layers.
    NO_SHAPEFILE = "NO_SHAPEFILE"
    SHAPEFILE_INCOMPLETE = "SHAPEFILE_INCOMPLETE"
    ZIP_UNSAFE_PATH = "ZIP_UNSAFE_PATH"
    ZIP_TOO_LARGE = "ZIP_TOO_LARGE"
    NO_FEATURES = "NO_FEATURES"
    READ_ERROR = "READ_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    INTERRUPTED = "INTERRUPTED"


class AppError(Exception):
    """A request-level error that maps to exactly one HTTP response."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        http_status: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status
        self.details = details


class ProcessingError(Exception):
    """A file-level failure during background processing.

    It is never sent as an HTTP error: the pipeline records the code and message on the file and
    sets its status to ``FAILED``.
    """

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

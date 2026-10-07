"""Exception handlers that give every error the same response body."""

from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.errors import AppError, ErrorCode
from app.core.logging import get_logger
from app.db.models import FileStatus, GeoFile
from app.schemas.errors import ErrorBody, ErrorDetail

logger = get_logger(__name__)


def file_not_found(file_id: str) -> AppError:
    """404 for an id that matches no file."""
    return AppError(
        ErrorCode.FILE_NOT_FOUND,
        "No file exists with this id.",
        http_status=404,
        details={"id": file_id},
    )


def file_not_ready(geo_file: GeoFile) -> AppError:
    """409 for a file whose processing has not finished."""
    return AppError(
        ErrorCode.FILE_NOT_READY,
        "File is still being processed. Try again shortly.",
        http_status=409,
        details={"status": geo_file.status},
    )


def file_failed(geo_file: GeoFile) -> AppError:
    """409 for a file whose processing failed; the stored failure is in the details."""
    return AppError(
        ErrorCode.FILE_FAILED,
        "File processing failed, so it has no measurements.",
        http_status=409,
        details={
            "status": FileStatus.FAILED,
            "error": {"code": geo_file.error_code, "message": geo_file.error_message},
        },
    )


def error_response(
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Build the uniform error response."""
    body = ErrorBody(error=ErrorDetail(code=code, message=message, details=details))
    return JSONResponse(body.model_dump(mode="json"), status_code=status_code, headers=headers)


def register_exception_handlers(app: FastAPI) -> None:
    """Route every kind of failure through :func:`error_response`."""

    @app.exception_handler(AppError)
    async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
        return error_response(exc.http_status, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Only loc, msg and type are echoed: the raw error can hold the rejected input itself.
        errors = [
            {"loc": list(error["loc"]), "message": error["msg"], "type": error["type"]}
            for error in exc.errors()
        ]
        return error_response(
            422, ErrorCode.VALIDATION_ERROR, "The request is invalid.", {"errors": errors}
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        # Raised by the framework itself, e.g. 404 for an unknown route or 405 for a wrong method.
        status = HTTPStatus(exc.status_code)
        return error_response(exc.status_code, status.name, str(exc.detail), headers=exc.headers)

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, _exc: Exception) -> JSONResponse:
        logger.exception("unhandled error", extra={"path": request.url.path})
        return error_response(500, ErrorCode.INTERNAL_ERROR, "An unexpected error occurred.")

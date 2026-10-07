"""The uniform error body returned by every endpoint."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class ErrorDetail(BaseModel):
    """A machine-readable code, a human-readable message and optional structured details."""

    code: str
    message: str
    details: dict[str, Any] | None = None


class ErrorBody(BaseModel):
    """Envelope of every error response."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "error": {
                        "code": "FILE_NOT_READY",
                        "message": "File is still being processed. Try again shortly.",
                        "details": {"status": "PROCESSING"},
                    }
                }
            ]
        }
    )

    error: ErrorDetail

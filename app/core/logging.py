"""Structured ``key=value`` logging for the service.

Every record is rendered as one logfmt line. Fields passed through ``extra=`` become additional
``key=value`` pairs, and the id of the file being processed is attached automatically to every
record emitted inside a :func:`bind_file_id` block, so processing logs can be filtered per file.
"""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

APP_LOGGER_NAME = "app"

_file_id: ContextVar[str | None] = ContextVar("file_id", default=None)

_STANDARD_RECORD_ATTRS = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


def _quote(value: Any) -> str:
    """Render a value for logfmt, quoting it when it contains spaces, quotes or ``=``."""
    text = str(value)
    if text == "" or any(char in text for char in ' "=\n'):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    return text


class KeyValueFormatter(logging.Formatter):
    """Format records as ``ts=... level=... logger=... msg=... key=value ...``, with UTC times."""

    converter = time.gmtime

    def format(self, record: logging.LogRecord) -> str:
        fields: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        fields.update(
            (key, value)
            for key, value in vars(record).items()
            if key not in _STANDARD_RECORD_ATTRS and value is not None
        )
        line = " ".join(f"{key}={_quote(value)}" for key, value in fields.items())
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


class _FileIdFilter(logging.Filter):
    """Copy the file id bound to the current context onto each record."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "file_id"):
            record.file_id = _file_id.get()
        return True


def setup_logging(level: str = "INFO") -> None:
    """Attach the structured handler to the application logger. Safe to call more than once."""
    logger = logging.getLogger(APP_LOGGER_NAME)
    logger.setLevel(level.upper())
    if any(isinstance(handler.formatter, KeyValueFormatter) for handler in logger.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(KeyValueFormatter())
    handler.addFilter(_FileIdFilter())
    logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the application namespace, e.g. ``get_logger(__name__)``."""
    if name == APP_LOGGER_NAME or name.startswith(f"{APP_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{APP_LOGGER_NAME}.{name}")


@contextmanager
def bind_file_id(file_id: str) -> Iterator[None]:
    """Attach ``file_id`` to every log record emitted inside the ``with`` block."""
    token = _file_id.set(file_id)
    try:
        yield
    finally:
        _file_id.reset(token)

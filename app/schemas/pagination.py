"""Offset pagination metadata shared by the list endpoints."""

from pydantic import BaseModel


class Pagination(BaseModel):
    """Where a page sits in the full result set."""

    limit: int
    offset: int
    total: int
    next_offset: int | None

    @classmethod
    def of(cls, *, limit: int, offset: int, total: int) -> "Pagination":
        """Build the metadata; ``next_offset`` is ``None`` on the last page."""
        following = offset + limit
        return cls(
            limit=limit,
            offset=offset,
            total=total,
            next_offset=following if following < total else None,
        )

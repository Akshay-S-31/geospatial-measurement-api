"""Coded reasons explaining why a feature has no measurement."""

from enum import StrEnum


class ReasonCode(StrEnum):
    """Why a feature was kept but not measured."""

    NOT_MEASURABLE = "NOT_MEASURABLE"
    UNSUPPORTED_GEOMETRY_TYPE = "UNSUPPORTED_GEOMETRY_TYPE"
    EMPTY_GEOMETRY = "EMPTY_GEOMETRY"
    INVALID_GEOMETRY = "INVALID_GEOMETRY"
    UNKNOWN_CRS = "UNKNOWN_CRS"
    MEASUREMENT_FAILED = "MEASUREMENT_FAILED"


REASON_MESSAGES: dict[ReasonCode, str] = {
    ReasonCode.NOT_MEASURABLE: "Points have no area or length.",
    ReasonCode.UNSUPPORTED_GEOMETRY_TYPE: (
        "Geometry collections that mix geometry types are not measured."
    ),
    ReasonCode.EMPTY_GEOMETRY: "The feature has no geometry.",
    ReasonCode.INVALID_GEOMETRY: (
        "The geometry is invalid and nothing measurable remained after repairing it."
    ),
    ReasonCode.UNKNOWN_CRS: (
        "The coordinate reference system is unknown, so the geometry cannot be measured."
    ),
    ReasonCode.MEASUREMENT_FAILED: (
        "The geometry could not be projected; its coordinates may be outside the valid "
        "range of the source CRS."
    ),
}


def reason_message(code: str) -> str:
    """Return the human-readable message for a stored reason code."""
    return REASON_MESSAGES[ReasonCode(code)]

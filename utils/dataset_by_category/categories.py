"""Event-name classification shared by the category dataset builder."""

from __future__ import annotations

from typing import Literal, TypeAlias


Category: TypeAlias = Literal["blitz", "rapid", "classic"]
VALID_CATEGORIES: tuple[Category, ...] = ("blitz", "rapid", "classic")


def get_event_category(event_name: str | None) -> Category:
    """Infer a time-control category from a TWIC ``Event`` header.

    TWIC does not expose a consistently machine-readable time control for all
    historical games, so classification follows the convention used by the
    project's exploratory splitter: ``blitz`` takes precedence over ``rapid``
    when both words occur, and all other event names are treated as classical.
    Matching is case-insensitive and tolerates repeated whitespace.
    """

    normalized = " ".join((event_name or "").split()).casefold()
    if "blitz" in normalized:
        return "blitz"
    if "rapid" in normalized:
        return "rapid"
    return "classic"

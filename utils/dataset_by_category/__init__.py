"""Build training-ready TWIC datasets grouped by event time control."""

from .categories import Category, get_event_category

__all__ = ["Category", "get_event_category"]

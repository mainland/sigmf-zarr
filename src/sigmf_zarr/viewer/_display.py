"""Bounded text for metadata displays."""

from __future__ import annotations

import json

MAX_METADATA_TEXT = 262144
"""Maximum number of characters shown for one metadata document."""

MAX_TOOLTIP_TEXT = 4096
"""Maximum number of characters shown in one region tooltip."""


def bounded_json(value: object, limit: int) -> str:
    """Format JSON for display, stopping after a character limit.

    The encoder produces text incrementally, so the cost is proportional to
    the limit rather than to the size of the value.

    Args:
        value: JSON-compatible value.
        limit: Maximum number of characters of JSON text.

    Returns:
        Indented ASCII JSON text. Truncated text ends with a line that says
        so.
    """
    encoder = json.JSONEncoder(indent=2, ensure_ascii=True)
    parts: list[str] = []
    size = 0
    for chunk in encoder.iterencode(value):
        parts.append(chunk)
        size += len(chunk)
        if size > limit:
            text = "".join(parts)[:limit]
            return f"{text}\n... (truncated after {limit:,} characters)"
    return "".join(parts)


__all__ = ["MAX_METADATA_TEXT", "MAX_TOOLTIP_TEXT", "bounded_json"]

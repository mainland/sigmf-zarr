"""Group handles and traversal that ignore Zarr consolidated metadata.

SigMF-Zarr never reads consolidated metadata. A consolidated copy can differ
from the node metadata documents, so reading it would hide tampering from
integrity checks, and a handle that reads it does not observe writes made
through the node documents.

Opening a group with `use_consolidated=False` ignores the consolidated
metadata of that group only. In Zarr format 3, a child group read through its
parent keeps any `consolidated_metadata` field of its own `zarr.json`, and
zarr then serves that child's descendants from the copy. Zarr's recursive
member listing also follows such a copy. Code that traverses a store
therefore removes the field from each group handle that it uses, and lists
descendants one level at a time.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from zarr.core.array import Array
from zarr.core.common import ZarrFormat
from zarr.core.group import AsyncGroup, Group
from zarr.core.sync import sync
from zarr.storage import StorePath


def without_consolidated_metadata(group: Group) -> Group:
    """Return a handle for a group that ignores its consolidated metadata.

    The handle is built from metadata that is already loaded, so no store
    read occurs.

    Args:
        group: Group handle, which may carry consolidated metadata.

    Returns:
        `group` itself when it carries no consolidated metadata, or an
        equivalent handle that reads each child from its own metadata
        document.
    """
    metadata = group.metadata
    if metadata.consolidated_metadata is None:
        return group
    return Group(
        AsyncGroup(
            metadata=replace(metadata, consolidated_metadata=None),
            store_path=group.store_path,
        )
    )


def open_group(store_path: StorePath, *, zarr_format: ZarrFormat) -> Group:
    """Open an existing group without reading consolidated metadata.

    `Group.open()` reads consolidated metadata whenever it is present and
    offers no way to decline it.

    Args:
        store_path: Location of the group.
        zarr_format: Physical Zarr format of the group.

    Returns:
        Group handle read from the group's own metadata document.

    Raises:
        FileNotFoundError: If no group exists at `store_path`.
    """
    return Group(
        sync(
            AsyncGroup.open(
                store_path, zarr_format=zarr_format, use_consolidated=False
            )
        )
    )


def members(
    group: Group, max_depth: int | None = 0
) -> tuple[tuple[str, Array[Any] | Group], ...]:
    """List the descendants of a group from their own metadata documents.

    This replaces `Group.members()`, which reads the consolidated metadata of
    child groups.

    Args:
        group: Group to traverse.
        max_depth: Greatest depth below the immediate children to list, or
            ``None`` for every depth.

    Returns:
        Relative paths paired with arrays and groups. Every returned group
        ignores its consolidated metadata.

    Raises:
        ValueError: If `max_depth` is negative.
    """
    if max_depth is not None and max_depth < 0:
        raise ValueError(f"max_depth must be None or >= 0, got {max_depth}")
    found: list[tuple[str, Array[Any] | Group]] = []
    # An explicit stack keeps deeply nested hierarchies within the
    # interpreter recursion limit.
    pending: list[tuple[str, Group, int | None]] = [
        ("", without_consolidated_metadata(group), max_depth)
    ]
    while pending:
        prefix, current, depth = pending.pop()
        for name, member in current.members():
            path = f"{prefix}/{name}" if prefix else name
            if isinstance(member, Group):
                member = without_consolidated_metadata(member)
                if depth is None or depth > 0:
                    pending.append(
                        (path, member, None if depth is None else depth - 1)
                    )
            found.append((path, member))
    return tuple(found)


__all__ = ["members", "open_group", "without_consolidated_metadata"]

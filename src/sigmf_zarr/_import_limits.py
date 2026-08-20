"""Byte limits shared by the streaming dataset importers.

Importers read item-aligned sources in batches. The `batch_size` arguments
cap the item count of each batch, and the byte budget below also caps its
decoded size, so the memory of one batch does not follow a large declared
item length. Items larger than the per-item limit are rejected before any of
their samples are read.
"""

from __future__ import annotations

IMPORT_BATCH_BYTES = 64 * 1024 * 1024
"""Target decoded size of one importer batch, in bytes."""

MAX_IMPORT_ITEM_BYTES = 64 * 1024 * 1024
"""Largest decoded size of one imported item, in bytes."""


def check_item_bytes(item_bytes: int, *, source: str) -> None:
    """Reject an item whose decoded size exceeds the per-item limit.

    Args:
        item_bytes: Decoded size of one item, in bytes.
        source: Source description for the error message.

    Raises:
        ValueError: If the item exceeds `MAX_IMPORT_ITEM_BYTES`.
    """
    if item_bytes > MAX_IMPORT_ITEM_BYTES:
        raise ValueError(
            f"{source} declares {item_bytes} bytes per item, more than the "
            f"importer limit of {MAX_IMPORT_ITEM_BYTES} bytes"
        )


def batch_items(item_bytes: int, batch_size: int) -> int:
    """Return the number of items to read per batch.

    Args:
        item_bytes: Decoded size of one item, in bytes.
        batch_size: Caller's maximum number of items per batch.

    Returns:
        At most `batch_size` items and at most `IMPORT_BATCH_BYTES` of
        decoded data, but always at least one item.
    """
    return max(1, min(batch_size, IMPORT_BATCH_BYTES // max(item_bytes, 1)))

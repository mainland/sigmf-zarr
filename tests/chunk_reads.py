"""Count chunk fetches from local Zarr stores."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest
from zarr.storage import LocalStore


def count_chunk_reads(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    """Record every fetch of a stored chunk from a local store.

    Zarr fetches and decodes a whole unsharded chunk to return any part of
    it, so the fetch count of a chunk key is its decode count.

    Args:
        monkeypatch: Patch fixture that restores the store afterward.

    Returns:
        Fetch counts by store key, filled in as reads occur. Metadata
        documents are not counted.
    """
    reads: Counter[str] = Counter()
    original = LocalStore.get

    async def get(
        self: LocalStore, key: str, *args: Any, **kwargs: Any
    ) -> Any:
        """Count a chunk fetch and delegate to the store.

        Args:
            self: Local store.
            key: Store key.
            *args: Positional read arguments.
            **kwargs: Keyword read arguments.

        Returns:
            Stored buffer, or None.
        """
        if not key.rsplit("/", 1)[-1].startswith((".z", "zarr.json")):
            reads[key] += 1
        return await original(self, key, *args, **kwargs)

    monkeypatch.setattr(LocalStore, "get", get)
    return reads

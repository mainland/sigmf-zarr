"""Shared state for commands operating on one SigMF-Zarr store."""

from __future__ import annotations

from os import PathLike
from pathlib import Path

from sigmf_zarr.store import SigMFZarrStore
from sigmf_zarr.store._common import store_location


class StoreContext:
    """Lazily open and share one read-only store between child commands."""

    _location: str | Path | None
    """Normalized location of the cached store."""

    _store: SigMFZarrStore | None
    """Cached store instance."""

    def __init__(self) -> None:
        """Initialize an empty store context."""
        self._location = None
        self._store = None

    def open(self, location: str | PathLike[str]) -> SigMFZarrStore:
        """Open or return the cached store at `location`.

        Args:
            location: Store path or URL. A string is a URL only when it
                starts with an explicit scheme, such as `s3://` or
                `file://`. Any other string or path-like object names a
                local directory.

        Returns:
            Opened read-only store.
        """
        normalized = store_location(location)
        if self._store is None or self._location != normalized:
            self._store = SigMFZarrStore.open(normalized)
            self._location = normalized
        return self._store


__all__ = ["StoreContext"]

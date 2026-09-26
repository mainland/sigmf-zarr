"""Shared state for commands operating on one SigMF-Zarr store."""

from __future__ import annotations

from os import PathLike, fspath

from sigmf_zarr.store import SigMFZarrStore


class StoreContext:
    """Lazily open and share one read-only store between child commands."""

    _location: str | None
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
            location: Store path or URL.

        Returns:
            Opened read-only store.
        """
        normalized = fspath(location)
        if self._store is None or self._location != normalized:
            self._store = SigMFZarrStore.open(normalized)
            self._location = normalized
        return self._store


__all__ = ["StoreContext"]

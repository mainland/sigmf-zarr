"""Zarr-backed SigMF-Zarr collection implementation."""

from __future__ import annotations

import json
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar, cast

from zarr.core.group import Group

from sigmf_zarr.integrity import (
    INTEGRITY_ALGORITHM,
    INTEGRITY_ATTR,
    INTEGRITY_VERSION,
    calculate_group_metadata_sha512,
    integrity_is_supported,
)
from sigmf_zarr.json import JSONObject, json_object
from sigmf_zarr.readonly import ReadOnlyGroup

if TYPE_CHECKING:
    from sigmf_zarr.store._container import SigMFZarrStore


class SigMFCollection:
    """Thin wrapper around one collection of recording identifiers."""

    _store: SigMFZarrStore
    """Parent SigMF-Zarr store containing this collection."""

    _collection_name: str
    """Collection identifier relative to the store's `collections/` group."""

    _INTEGRITY_METADATA_FIELD: ClassVar[str] = "metadata_sha512"
    """Internal integrity field containing the collection metadata hash."""

    def __init__(
        self,
        store: SigMFZarrStore,
        collection_name: str,
        *,
        create: bool | None = None,
        overwrite: bool = False,
        metadata: JSONObject | None = None,
        recording_ids: list[str] | tuple[str, ...] = (),
    ) -> None:
        """Initialize a collection wrapper relative to a store.

        Args:
            store: Parent SigMF-Zarr store.
            collection_name: Collection identifier under `collections/`.
            create: Whether a missing collection may be created.
            overwrite: Whether an existing collection should be replaced.
            metadata: Collection metadata.
            recording_ids: Recording identifiers referenced by the collection.

        Raises:
            KeyError: If the collection does not exist and creation is not
                allowed.
        """
        self._store = store
        self._collection_name = collection_name
        self._ensure_exists(
            create=create,
            overwrite=overwrite,
            metadata=metadata,
            recording_ids=recording_ids,
        )
        self._validate()

    def _validate(self) -> None:
        """Validate collection metadata and recording references.

        Raises:
            ValueError: If metadata or recording identifiers are malformed.
        """
        attrs = self._raw_group.attrs
        for field in ("metadata", "recording_ids"):
            if field not in attrs:
                raise ValueError(f"Collection is missing required {field!r}")
        _ = json_object(
            attrs["metadata"],
            name="collection metadata",
        )
        recording_ids = attrs["recording_ids"]
        if not isinstance(recording_ids, list) or not all(
            isinstance(recording_id, str) for recording_id in recording_ids
        ):
            raise ValueError(
                "Collection recording_ids must be a list of strings"
            )

    def _ensure_exists(
        self,
        *,
        create: bool | None,
        overwrite: bool,
        metadata: JSONObject | None,
        recording_ids: list[str] | tuple[str, ...],
    ) -> None:
        """Create or validate the backing collection group.

        Args:
            create: Whether a missing collection may be created.
            overwrite: Whether an existing collection should be replaced.
            metadata: Collection metadata.
            recording_ids: Recording identifiers referenced by the collection.

        Raises:
            KeyError: If the collection does not exist and creation is not
                allowed.
        """
        should_create = self._store._mode != "r" if create is None else create
        exists = self.name in self._store._collections_group

        if exists and not overwrite:
            return
        if exists and overwrite:
            self._store._invalidate_metadata_integrity()
            del self._store._collections_group[self.name]
            exists = False

        if not exists and not should_create:
            raise KeyError(self.name)

        if not exists:
            self._store._invalidate_metadata_integrity()
            group = self._store._collections_group.create_group(self.name)
            group.attrs.update(
                {
                    "metadata": metadata or {},
                    "recording_ids": list(recording_ids),
                }
            )

    def __repr__(self) -> str:
        """Return a concise debug representation.

        Returns:
            Debug representation string.
        """
        return (
            f"{type(self).__name__}("
            f"name={self.name!r}, "
            f"num_recordings={len(self.recording_ids)})"
        )

    @property
    def name(self) -> str:
        """Collection identifier relative to the `collections/` group.

        Returns:
            Collection name.
        """
        return self._collection_name

    @property
    def _raw_group(self) -> Group:
        """Writable collection group used by managed implementation code.

        Returns:
            Backing collection group.
        """
        return cast(
            Group,
            self._store._collections_group[self._collection_name],
        )

    @property
    def group(self) -> ReadOnlyGroup:
        """Collection group under the parent store.

        Returns:
            Read-only Zarr group for the collection.
        """
        return ReadOnlyGroup(self._raw_group)

    @property
    def metadata(self) -> JSONObject:
        """Collection metadata object.

        Returns:
            Collection metadata.
        """
        return cast(JSONObject, self._raw_group.attrs.get("metadata", {}))

    @property
    def recording_ids(self) -> tuple[str, ...]:
        """Recording identifiers referenced by the collection.

        Returns:
            Recording identifiers referenced by the collection.
        """
        values = cast(
            list[str], self._raw_group.attrs.get("recording_ids", [])
        )
        return tuple(values)

    @property
    def integrity(self) -> JSONObject:
        """SigMF-Zarr-native integrity metadata for this collection.

        Returns:
            Integrity object containing any current internal hash.
        """
        value = self._raw_group.attrs.get(INTEGRITY_ATTR, {})
        return cast(JSONObject, value) if isinstance(value, dict) else {}

    @property
    def metadata_sha512(self) -> str | None:
        """Internal hash of the collection's logical metadata.

        Returns:
            Declared metadata SHA-512 digest, or ``None`` when absent.
        """
        value = self.integrity.get(self._INTEGRITY_METADATA_FIELD)
        return value if isinstance(value, str) else None

    def calculate_metadata_sha512(self) -> str:
        """Calculate the internal collection metadata hash.

        Returns:
            Lowercase SHA-512 hexadecimal digest.
        """
        return calculate_group_metadata_sha512(self._raw_group)

    def update_integrity(self) -> JSONObject:
        """Calculate and store the current collection metadata hash.

        Returns:
            Updated integrity metadata object.
        """
        integrity: JSONObject = {
            "algorithm": INTEGRITY_ALGORITHM,
            "version": INTEGRITY_VERSION,
            self._INTEGRITY_METADATA_FIELD: (
                self.calculate_metadata_sha512()
            ),
        }
        self._raw_group.attrs[INTEGRITY_ATTR] = integrity
        return self.integrity

    def _invalidate_integrity(self) -> None:
        """Invalidate collection and store metadata hashes."""
        self._store._invalidate_metadata_integrity()
        if INTEGRITY_ATTR in self._raw_group.attrs:
            del self._raw_group.attrs[INTEGRITY_ATTR]

    def set_metadata(self, metadata: JSONObject) -> None:
        """Replace collection metadata through the managed mutation path.

        Args:
            metadata: JSON-compatible collection metadata object.
        """
        copied = json_object(dict(metadata), name="collection metadata")
        try:
            json.dumps(copied, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Collection metadata must be finite JSON"
            ) from exc
        self._invalidate_integrity()
        self._raw_group.attrs["metadata"] = copied

    def set_recording_ids(self, recording_ids: Sequence[str]) -> None:
        """Replace the recording identifiers referenced by the collection.

        Args:
            recording_ids: Recording names in collection order.

        Raises:
            ValueError: If an identifier is invalid or missing from the store.
        """
        values = list(recording_ids)
        if not all(isinstance(value, str) and value for value in values):
            raise ValueError(
                "Collection recording IDs must be non-empty strings"
            )
        missing = sorted(set(values) - set(self._store.list_recordings()))
        if missing:
            raise ValueError(
                f"Collection references missing recordings: {missing!r}"
            )
        self._invalidate_integrity()
        self._raw_group.attrs["recording_ids"] = values

    @contextmanager
    def mutate(self) -> Generator[Group, None, None]:
        """Provide writable collection access with hash invalidation.

        Yields:
            Writable backing collection group.

        Raises:
            ValueError: If the resulting collection attributes are malformed.
        """
        self._invalidate_integrity()
        yield self._raw_group
        # Validation occurs only after a normal exit. An exceptional mutation
        # leaves integrity absent without replacing the caller's exception.
        self._validate()

    def verify_integrity(self) -> bool:
        """Verify the internal collection metadata hash.

        Returns:
            True only when the hash is present and matches current metadata.
        """
        return (
            integrity_is_supported(self.integrity)
            and self.metadata_sha512 is not None
            and self.metadata_sha512 == self.calculate_metadata_sha512()
        )

    def info(self) -> str:
        """Return a human-readable summary of one collection.

        Returns:
            Multiline collection summary.
        """
        lines = [
            f"{type(self).__name__}(",
            f"  name={self.name!r},",
            f"  num_recordings={len(self.recording_ids)},",
            f"  recording_ids={list(self.recording_ids)!r},",
            f"  metadata_sha512={self.metadata_sha512!r},",
            f"  metadata_keys={sorted(self.metadata.keys())},",
            ")",
        ]
        return "\n".join(lines)

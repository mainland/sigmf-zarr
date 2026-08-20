"""Zarr-backed SigMF-Zarr collection implementation."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar, cast

from zarr.core.group import Group

from sigmf_zarr._consolidated import without_consolidated_metadata
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
        create: bool = False,
        overwrite: bool = False,
        metadata: JSONObject | None = None,
        recording_ids: list[str] | tuple[str, ...] = (),
    ) -> None:
        """Open an existing collection or create a new one.

        Use :meth:`SigMFCollections.open` and
        :meth:`SigMFCollections.create` rather than calling the constructor
        directly. Without `create`, the collection must exist and the
        creation arguments are ignored.

        Args:
            store: Parent SigMF-Zarr store.
            collection_name: Collection identifier under `collections/`.
            create: Whether to create the collection instead of opening it.
            overwrite: Whether creation may replace an existing collection.
            metadata: Collection metadata.
            recording_ids: Recording identifiers referenced by the collection.

        Raises:
            KeyError: If `create` is false and the collection does not exist.
            ValueError: If `overwrite` is requested without `create`, the
                collection exists and `overwrite` is false, new metadata is
                not finite JSON, a recording identifier is invalid, repeated,
                or absent from the store, or stored attributes are malformed.
        """
        if overwrite and not create:
            raise ValueError("overwrite requires create=True")
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
        create: bool,
        overwrite: bool,
        metadata: JSONObject | None,
        recording_ids: list[str] | tuple[str, ...],
    ) -> None:
        """Create the backing collection group or require that it exists.

        Args:
            create: Whether to create the collection instead of opening it.
            overwrite: Whether creation may replace an existing collection.
            metadata: Collection metadata.
            recording_ids: Recording identifiers referenced by the collection.

        Raises:
            KeyError: If `create` is false and the collection does not exist.
            ValueError: If the collection exists and `overwrite` is false,
                new metadata is not finite JSON, or a recording identifier is
                invalid, repeated, or absent from the store.
        """
        exists = self.name in self._store._collections_group
        if not create:
            if not exists:
                raise KeyError(self.name)
            return
        if exists and not overwrite:
            raise ValueError(
                f"Collection {self.name!r} already exists. Pass "
                "overwrite=True to replace it"
            )
        # Validate before creating the group, using the same rules as the
        # managed setters, so creation cannot produce an invalid store.
        copied = json_object(dict(metadata or {}), name="collection metadata")
        ids = self._checked_recording_ids(recording_ids)
        self._store._invalidate_metadata_integrity()
        if exists:
            del self._store._collections_group[self.name]
        group = self._store._collections_group.create_group(self.name)
        group.attrs.update({"metadata": copied, "recording_ids": ids})

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
        return without_consolidated_metadata(
            cast(Group, self._store._collections_group[self._collection_name])
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
            self._INTEGRITY_METADATA_FIELD: (self.calculate_metadata_sha512()),
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
            ValueError: If an identifier is invalid, repeated, or missing from
                the store.
        """
        values = self._checked_recording_ids(recording_ids)
        self._invalidate_integrity()
        self._raw_group.attrs["recording_ids"] = values

    def _checked_recording_ids(
        self, recording_ids: Sequence[str]
    ) -> list[str]:
        """Validate recording identifiers against the parent store.

        Args:
            recording_ids: Recording names in collection order.

        Returns:
            Copied identifiers.

        Raises:
            ValueError: If an identifier is not a nonempty string, appears
                more than once, or names a recording that is absent from the
                store.
        """
        if isinstance(recording_ids, str | bytes):
            raise ValueError(
                "Collection recording IDs must be a sequence of names"
            )
        values = list(recording_ids)
        # Untyped callers can pass any value.
        if not all(
            isinstance(value, str) and value  # type: ignore[redundant-expr]
            for value in values
        ):
            raise ValueError(
                "Collection recording IDs must be non-empty strings"
            )
        duplicates = sorted(
            name for name, count in Counter(values).items() if count > 1
        )
        if duplicates:
            raise ValueError(
                f"Collection recording IDs must be unique, got {duplicates!r} "
                "more than once"
            )
        missing = sorted(set(values) - set(self._store.list_recordings()))
        if missing:
            raise ValueError(
                f"Collection references missing recordings: {missing!r}"
            )
        return values

    @contextmanager
    def mutate(self) -> Generator[Group, None, None]:
        """Provide writable collection access with hash invalidation.

        If the body raises or the resulting attributes are malformed, the
        original attributes are restored and the metadata hashes remain
        absent.

        Yields:
            Writable backing collection group.

        Raises:
            ValueError: If the resulting collection attributes are malformed.
        """
        self._invalidate_integrity()
        group = self._raw_group
        original = group.attrs.asdict()
        try:
            yield group
            self._validate()
        except BaseException:
            # Malformed attributes would make the collection unopenable.
            group.attrs.put(original)
            raise

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

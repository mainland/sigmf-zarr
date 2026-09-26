"""Root SigMF-Zarr store and child-container views."""

from __future__ import annotations

import json
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from os import PathLike, fspath
from pathlib import PurePosixPath
from typing import Any, ClassVar, Literal, cast

import numpy as np
import numpy.typing as npt
import zarr
from numcodecs.abc import Codec as NumcodecsCodec
from zarr.codecs import VLenUTF8Codec
from zarr.core.array import (
    Array,
    CompressorLike,
    CompressorsLike,
    SerializerLike,
    ShardsLike,
)
from zarr.core.common import AccessModeLiteral
from zarr.core.dtype import VariableLengthUTF8
from zarr.core.group import Group
from zarr.storage import StoreLike

from sigmf_zarr.integrity import (
    INTEGRITY_ALGORITHM,
    INTEGRITY_ATTR,
    INTEGRITY_VERSION,
    STORE_DOMAIN,
    calculate_payload_sha512,
    group_metadata_payload,
    integrity_is_supported,
)
from sigmf_zarr.json import JSONObject, json_object
from sigmf_zarr.readonly import ReadOnlyArray, ReadOnlyGroup
from sigmf_zarr.store._collection import SigMFCollection
from sigmf_zarr.store._common import (
    ChecksumName,
    ZarrFormat,
    _validate_resource_name,
)
from sigmf_zarr.store._recording import SigMFRecording
from sigmf_zarr.store._transaction import replacement_transaction


class SigMFRecordings:
    """Proxy view over the recordings in a SigMF-Zarr store."""

    _store: SigMFZarrStore
    """Parent SigMF-Zarr store."""

    def __init__(self, store: SigMFZarrStore) -> None:
        """Initialize a recordings view for one store.

        Args:
            store: Parent SigMF-Zarr store.
        """
        self._store = store

    def __iter__(self) -> Iterator[SigMFRecording]:
        """Iterate over recordings in sorted name order.

        Returns:
            Iterator over recording wrappers.
        """
        return self.values()

    def __getitem__(self, recording_name: str) -> SigMFRecording:
        """Return one recording by name.

        Args:
            recording_name: Recording name to open.

        Returns:
            Recording wrapper.

        Raises:
            KeyError: If no recording has the requested name.
        """
        return self.open(recording_name)

    def __contains__(self, recording_name: object) -> bool:
        """Return whether a recording with this name exists.

        Args:
            recording_name: Candidate recording name.

        Returns:
            True when a recording with this name exists.
        """
        if not isinstance(recording_name, str):
            return False
        try:
            _validate_resource_name(recording_name, kind="Recording")
        except ValueError:
            return False
        return recording_name in self._store._recordings_group

    def __len__(self) -> int:
        """Return the number of recordings in the store.

        Returns:
            Number of recordings.
        """
        return len(self._store.list_recordings())

    def names(self) -> tuple[str, ...]:
        """Return recording identifiers in sorted order.

        Returns:
            Recording names.
        """
        return self._store.list_recordings()

    def values(
        self,
        *,
        recording_cls: type[SigMFRecording] = SigMFRecording,
    ) -> Iterator[SigMFRecording]:
        """Iterate over recordings in sorted name order.

        Args:
            recording_cls: Recording wrapper class to instantiate.

        Returns:
            Iterator over recording wrappers.
        """
        for recording_name in self.names():
            yield self.open(recording_name, recording_cls=recording_cls)

    def open(
        self,
        recording_name: str,
        *,
        recording_cls: type[SigMFRecording] = SigMFRecording,
    ) -> SigMFRecording:
        """Open one existing recording.

        Args:
            recording_name: Recording name.
            recording_cls: Recording wrapper class to instantiate.

        Returns:
            Recording wrapper.

        Raises:
            KeyError: If the recording does not exist.
            ValueError: If stored metadata is invalid.
        """
        return recording_cls(self._store, recording_name)

    def create(
        self,
        recording_name: str,
        *,
        sample_shape: tuple[int, ...],
        overwrite: bool = False,
        batched: bool = False,
        recording_cls: type[SigMFRecording] = SigMFRecording,
        sample_dtype: npt.DTypeLike = np.float32,
        sample_axes: tuple[str, ...] | None = None,
        global_metadata: JSONObject | None = None,
        captures: list[JSONObject] | None = None,
        annotations: list[JSONObject] | None = None,
        channel_metadata: Sequence[JSONObject] | None = None,
        sample_chunks: tuple[int, ...] | None = None,
        sample_shards: ShardsLike | None = None,
        sample_compressor: CompressorLike = "auto",
        sample_checksum: ChecksumName | None = "crc32c",
    ) -> SigMFRecording:
        """Create one recording.

        Args:
            recording_name: Recording name.
            sample_shape: Per-item shape for a batched recording, or the
                complete sample array shape for an unbatched recording.
            overwrite: Whether to replace an existing recording.
            batched: Whether the sample array has a leading item axis.
            recording_cls: Recording wrapper class to instantiate.
            sample_dtype: Element dtype of the sample array.
            sample_axes: Optional axis names for `sample_shape`.
            global_metadata: SigMF-like global metadata.
            captures: Capture metadata.
            annotations: Annotation metadata.
            channel_metadata: Optional metadata for each channel.
            sample_chunks: Optional sample chunk shape.
            sample_shards: Optional sample shard shape.
            sample_compressor: Optional sample compressor.
            sample_checksum: Optional chunk-level sample checksum.

        Returns:
            Recording wrapper.

        Raises:
            ValueError: If the recording exists and `overwrite` is false, or
                the requested layout or metadata is invalid.
        """
        return recording_cls(
            self._store,
            recording_name,
            create=True,
            overwrite=overwrite,
            batched=batched,
            sample_dtype=sample_dtype,
            sample_shape=sample_shape,
            sample_axes=sample_axes,
            global_metadata=global_metadata,
            captures=captures,
            annotations=annotations,
            channel_metadata=channel_metadata,
            sample_chunks=sample_chunks,
            sample_shards=sample_shards,
            sample_compressor=sample_compressor,
            sample_checksum=sample_checksum,
        )

    def remove(self, recording_name: str) -> None:
        """Remove a recording through the managed mutation boundary.

        Args:
            recording_name: Recording name to remove.

        Raises:
            KeyError: If the recording does not exist.
            ValueError: If a collection references the recording. Update
                those collections first so removal cannot leave dangling
                references.
        """
        if recording_name not in self:
            raise KeyError(recording_name)
        referencing = [
            collection.name
            for collection in self._store.collections
            if recording_name in collection.recording_ids
        ]
        if referencing:
            raise ValueError(
                f"Recording {recording_name!r} is referenced by collections "
                f"{referencing!r}"
            )
        self._discard(recording_name)

    def _discard(self, recording_name: str) -> None:
        """Delete a recording without checking collection references.

        Import rollback uses this to remove a recording that it created.

        Args:
            recording_name: Existing recording name.
        """
        self._store._invalidate_metadata_integrity()
        del self._store._recordings_group[recording_name]


class SigMFCollections:
    """Proxy view over the collections in a SigMF-Zarr store."""

    _store: SigMFZarrStore
    """Parent SigMF-Zarr store."""

    def __init__(self, store: SigMFZarrStore) -> None:
        """Initialize a collections view for one store.

        Args:
            store: Parent SigMF-Zarr store.
        """
        self._store = store

    def __iter__(self) -> Iterator[SigMFCollection]:
        """Iterate over collections in sorted name order.

        Returns:
            Iterator over collection wrappers.
        """
        return self.values()

    def __getitem__(self, collection_name: str) -> SigMFCollection:
        """Return one collection by name.

        Args:
            collection_name: Collection name to open.

        Returns:
            Collection wrapper.

        Raises:
            KeyError: If no collection has the requested name.
        """
        return self.open(collection_name)

    def __contains__(self, collection_name: object) -> bool:
        """Return whether a collection with this name exists.

        Args:
            collection_name: Candidate collection name.

        Returns:
            True when a collection with this name exists.
        """
        if not isinstance(collection_name, str):
            return False
        try:
            _validate_resource_name(collection_name, kind="Collection")
        except ValueError:
            return False
        return collection_name in self._store._collections_group

    def __len__(self) -> int:
        """Return the number of collections in the store.

        Returns:
            Number of collections.
        """
        return len(self._store.list_collections())

    def names(self) -> tuple[str, ...]:
        """Return collection identifiers in sorted order.

        Returns:
            Collection names.
        """
        return self._store.list_collections()

    def values(
        self,
        *,
        collection_cls: type[SigMFCollection] = SigMFCollection,
    ) -> Iterator[SigMFCollection]:
        """Iterate over collections in sorted name order.

        Args:
            collection_cls: Collection wrapper class to instantiate.

        Returns:
            Iterator over collection wrappers.
        """
        for collection_name in self.names():
            yield self.open(collection_name, collection_cls=collection_cls)

    def open(
        self,
        collection_name: str,
        *,
        collection_cls: type[SigMFCollection] = SigMFCollection,
    ) -> SigMFCollection:
        """Open one existing collection.

        Args:
            collection_name: Collection name.
            collection_cls: Collection wrapper class to instantiate.

        Returns:
            Collection wrapper.

        Raises:
            KeyError: If the collection does not exist.
            ValueError: If stored attributes are malformed.
        """
        return collection_cls(self._store, collection_name)

    def create(
        self,
        collection_name: str,
        *,
        overwrite: bool = False,
        collection_cls: type[SigMFCollection] = SigMFCollection,
        metadata: JSONObject | None = None,
        recording_ids: list[str] | tuple[str, ...] = (),
    ) -> SigMFCollection:
        """Create one collection.

        Args:
            collection_name: Collection name.
            overwrite: Whether to replace an existing collection.
            collection_cls: Collection wrapper class to instantiate.
            metadata: Collection metadata.
            recording_ids: Recording identifiers referenced by the collection.

        Returns:
            Collection wrapper.

        Raises:
            ValueError: If the collection exists and `overwrite` is false,
                new metadata is not finite JSON, or a recording identifier is
                invalid or absent from the store.
        """
        return collection_cls(
            self._store,
            collection_name,
            create=True,
            overwrite=overwrite,
            metadata=metadata,
            recording_ids=recording_ids,
        )


class SigMFZarrStore:
    """Container for multiple SigMF-Zarr recordings and collections.

    The container keeps the core SigMF concepts visible while adapting
    them to Zarr storage:

    - The `recordings/<recording_name>/samples` array stores one homogeneous
      sample tensor per recording.
    - The `global`, `captures`, and `annotations` attributes on
      `recordings/<recording_name>` mirror the standard SigMF metadata objects.
    - The `recordings/<recording_name>.attrs["global"]` attribute also carries
      SigMF-Zarr extension declarations and namespaced metadata.
    - The `recordings/<recording_name>/extensions/` group stores structured
      extension-owned arrays.
    - The `recordings/<recording_name>/indexes/` group stores dense metadata
      indexes aligned with sample axes.
    - The `recordings/<recording_name>/channels/` group stores channel
      metadata.
    - The optional `recordings/<recording_name>/item_metadata` array stores
      per-item JSON metadata bundles.
    - The `collections/<collection_name>` group stores relationships among
      recordings.
    - The `indexes/` group stores optional cross-recording indexes.

    This is intentionally a minimal SigMF-to-Zarr mapping rather than
    a byte-for-byte archive translation. Recordings remain the primary
    organizational unit. SigMF extension metadata lives in the
    recording `global` object, while `extensions/` and top-level
    recording-level and top-level `indexes/` provide the main escape
    hatches for ML-oriented and application-specific query metadata.
    """

    SCHEMA_NAME: ClassVar[str] = "sigmf-zarr"
    """Schema identifier stored in root metadata."""

    SCHEMA_VERSION: ClassVar[int] = 1
    """Schema version stored in root metadata."""

    REQUIRED_GROUPS: ClassVar[tuple[str, ...]] = (
        "recordings",
        "collections",
        "indexes",
    )
    """Top-level groups required for a valid SigMF-Zarr store."""

    _INTEGRITY_METADATA_FIELD: ClassVar[str] = "metadata_sha512"
    """Internal integrity field containing the store metadata hash."""

    _INDEX_VALID_FIELD: ClassVar[str] = "sigmf-zarr:valid"
    """Index attribute indicating whether mutation completed successfully."""

    _INDEX_INVALID_REASON_FIELD: ClassVar[str] = "sigmf-zarr:invalid-reason"
    """Index attribute explaining why an index is invalid."""

    _store: StoreLike | None
    """Zarr store backend to use."""

    _group: Group
    """Opened root Zarr group backing the SigMF-Zarr store."""

    _mode: AccessModeLiteral
    """Resolved Zarr access mode used to open this store."""

    def __init__(
        self,
        store: StoreLike | str | PathLike[str] | None = None,
        overwrite: bool = False,
        *,
        mode: AccessModeLiteral | None = None,
        storage_options: dict[str, Any] | None = None,
        zarr_format: ZarrFormat | None = None,
    ) -> None:
        """Open a SigMF-Zarr store with an explicit Zarr access mode.

        Prefer :meth:`open` for existing stores and :meth:`create` for new
        stores. The constructor follows Zarr access modes. Modes `"r"` and
        `"r+"` require an existing store. Mode `"a"` opens an existing store
        or initializes an empty target. Modes `"w"` and `"w-"` create a
        store, and `"w"` replaces existing content.

        Args:
            store: Optional Zarr store backend.
            overwrite: When true, recreate the target store.
            mode: Optional Zarr access mode. If omitted, defaults to
                `"w"` when `overwrite` is true and `"a"` otherwise.
            storage_options: Optional backend-specific storage options.
            zarr_format: Physical Zarr format to create or require. When
                omitted, existing stores are auto-detected and new stores use
                Zarr format 3.

        Raises:
            FileNotFoundError: If mode `"r"` or `"r+"` names a missing target.
            FileExistsError: If mode `"w-"` names an existing target.
            ValueError: If an existing target is not a supported SigMF-Zarr
                store or uses a different physical Zarr format.
        """
        resolved_mode: AccessModeLiteral = (
            ("w" if overwrite else "a") if mode is None else mode
        )
        self._store = (
            fspath(store) if isinstance(store, str | PathLike) else store
        )
        self._mode = resolved_mode
        resolved_format = zarr_format
        if resolved_mode not in {"w", "w-"}:
            # Requesting a format that an existing store does not use would
            # otherwise create a second hierarchy beside the existing one.
            existing_format = type(self)._existing_zarr_format(
                self._store, storage_options
            )
            if (
                existing_format is not None
                and zarr_format is not None
                and existing_format != zarr_format
            ):
                raise ValueError(
                    f"Target uses Zarr format {existing_format}, but "
                    f"zarr_format={zarr_format} was requested"
                )
            resolved_format = existing_format or zarr_format
        if resolved_mode not in {"r", "r+"} and resolved_format is None:
            resolved_format = 3
        self._group = zarr.open_group(
            self._store,
            mode=resolved_mode,
            storage_options=storage_options,
            zarr_format=resolved_format,
        )
        self._initialize_or_validate(mode=resolved_mode)

    @staticmethod
    def _existing_zarr_format(
        store: StoreLike | None,
        storage_options: dict[str, Any] | None,
    ) -> ZarrFormat | None:
        """Detect the physical format of an existing root group.

        Args:
            store: Zarr store backend.
            storage_options: Optional backend-specific storage options.

        Returns:
            Existing root group format, or ``None`` when no group exists.
        """
        if store is None:
            return None
        try:
            existing = zarr.open_group(
                store, mode="r", storage_options=storage_options
            )
        except FileNotFoundError:
            # Zarr's GroupNotFoundError is also a FileNotFoundError.
            return None
        return existing.metadata.zarr_format

    @property
    def zarr_format(self) -> ZarrFormat:
        """Physical Zarr format used by this store.

        Returns:
            Zarr format number, either 2 or 3.
        """
        return self._group.metadata.zarr_format

    def _refresh_root(self) -> Group:
        """Reopen the root group to observe changes made by other handles.

        Zarr caches group attributes in each handle. Integrity metadata must
        reflect the backend, because another handle or process may have
        written or removed the root hash.

        Returns:
            Fresh root group handle, which also replaces the cached handle.
        """
        self._group = Group.open(
            self._group.store_path, zarr_format=self.zarr_format
        )
        return self._group

    @property
    def integrity(self) -> JSONObject:
        """SigMF-Zarr-native store integrity metadata.

        Returns:
            Integrity object containing any current store metadata hash.
        """
        value = self._refresh_root().attrs.get(INTEGRITY_ATTR, {})
        return cast(JSONObject, value) if isinstance(value, dict) else {}

    @property
    def metadata_sha512(self) -> str | None:
        """Hash of store-wide metadata and child resource digests.

        Returns:
            Declared metadata SHA-512 digest, or ``None`` when absent.
        """
        value = self.integrity.get(self._INTEGRITY_METADATA_FIELD)
        return value if isinstance(value, str) else None

    def _invalidate_metadata_integrity(self) -> None:
        """Remove the store-wide metadata hash after a managed mutation."""
        if self._mode == "r":
            return
        attrs = self._refresh_root().attrs
        if INTEGRITY_ATTR in attrs:
            del attrs[INTEGRITY_ATTR]

    @staticmethod
    def _child_digest_payload(
        group: Group, *, kind: str, fields: tuple[str, ...]
    ) -> dict[str, object]:
        """Describe a resource container by its children's declared digests.

        Args:
            group: Recordings or collections group.
            kind: Resource kind used in error messages.
            fields: Integrity fields each child must declare.

        Returns:
            Payload with the container's own metadata and child digests.

        Raises:
            ValueError: If a child lacks supported integrity metadata.
        """
        child_names = tuple(sorted(group.group_keys()))
        payload = group_metadata_payload(
            group, exclude_groups=frozenset(child_names)
        )
        digests: dict[str, object] = {}
        for name in child_names:
            child = group[name]
            integrity = child.attrs.get(INTEGRITY_ATTR)
            if not integrity_is_supported(integrity) or not all(
                isinstance(cast(JSONObject, integrity).get(field), str)
                for field in fields
            ):
                raise ValueError(
                    f"{kind} {name!r} has no current integrity hashes. Call "
                    "update_integrity() to calculate them"
                )
            digests[name] = {
                field: cast(JSONObject, integrity)[field] for field in fields
            }
        payload["groups"] = digests
        return payload

    def calculate_metadata_sha512(self) -> str:
        """Calculate the store-wide metadata hash.

        The hash covers root attributes, store-wide indexes, any other root
        content, and the declared digests of every recording and collection.
        Recording and collection content is bound through those digests, so
        this hash does not reread their arrays.

        Returns:
            Lowercase SHA-512 hexadecimal digest.

        Raises:
            ValueError: If a recording or collection has no current supported
                integrity hashes.
        """
        root = self._refresh_root()
        payload = group_metadata_payload(
            root, exclude_groups=frozenset({"recordings", "collections"})
        )
        groups = cast(dict[str, object], payload["groups"])
        groups["recordings"] = type(self)._child_digest_payload(
            self._recordings_group,
            kind="Recording",
            fields=("metadata_sha512", "sample_sha512"),
        )
        groups["collections"] = type(self)._child_digest_payload(
            self._collections_group,
            kind="Collection",
            fields=("metadata_sha512",),
        )
        return calculate_payload_sha512(payload, domain=STORE_DOMAIN)

    def update_integrity(self) -> JSONObject:
        """Calculate and store all recording, collection, and store hashes.

        Returns:
            Updated store integrity metadata object.
        """
        for recording in self.recordings.values():
            recording.update_integrity()
        for collection in self.collections.values():
            collection.update_integrity()
        return self.update_metadata_integrity()

    def update_metadata_integrity(self) -> JSONObject:
        """Calculate and store the store-wide metadata hash.

        Every recording and collection must already declare current hashes.
        Use :meth:`update_integrity` to calculate all hashes together.

        Returns:
            Updated store integrity metadata object.

        Raises:
            ValueError: If a store-wide index is invalid, or a recording or
                collection has no current integrity hashes.
        """
        invalid_index = self._invalid_store_index_reason()
        if invalid_index is not None:
            raise ValueError(f"Cannot update store integrity: {invalid_index}")
        integrity: JSONObject = {
            "algorithm": INTEGRITY_ALGORITHM,
            "version": INTEGRITY_VERSION,
            self._INTEGRITY_METADATA_FIELD: self.calculate_metadata_sha512(),
        }
        self._refresh_root().attrs[INTEGRITY_ATTR] = integrity
        return self.integrity

    def _update_metadata_integrity_when_complete(self) -> None:
        """Update the store-wide hash when every input is hashable.

        Importers call this after hashing the resource they wrote.
        Another resource without hashes, or an invalid store-wide index,
        leaves the store-wide hash absent instead of failing the import.
        """
        try:
            self.update_metadata_integrity()
        except ValueError:
            self._invalidate_metadata_integrity()

    def verify_metadata_integrity(self) -> bool:
        """Verify the store-wide hash without rereading child content.

        Returns:
            True when the store-wide hash is present and matches root
            metadata, store-wide indexes, and the declared child digests.
        """
        integrity = self.integrity
        declared = integrity.get(self._INTEGRITY_METADATA_FIELD)
        if (
            not integrity_is_supported(integrity)
            or not isinstance(declared, str)
            or self._invalid_store_index_reason() is not None
        ):
            return False
        try:
            current = self.calculate_metadata_sha512()
        except ValueError:
            return False
        return declared == current

    def verify_integrity(self) -> bool:
        """Verify every stored internal integrity hash.

        Returns:
            True only when all resource hashes and the store-wide hash are
            present and match current logical content.
        """
        return (
            self.verify_metadata_integrity()
            and all(
                recording.verify_integrity()
                for recording in self.recordings.values()
            )
            and all(
                collection.verify_integrity()
                for collection in self.collections.values()
            )
        )

    @classmethod
    def _required_groups_present(cls, group: Group) -> bool:
        """Return whether all required top-level groups are present.

        Args:
            group: Root Zarr group to inspect.

        Returns:
            True when all required groups are present.
        """
        existing_groups = set(group.group_keys())
        return set(cls.REQUIRED_GROUPS).issubset(existing_groups)

    @classmethod
    def _has_content(cls, group: Group) -> bool:
        """Return whether the root group already contains any content.

        Args:
            group: Root Zarr group to inspect.

        Returns:
            True when the group contains groups, arrays, or attributes.
        """
        return bool(
            tuple(group.group_keys())
            or tuple(group.array_keys())
            or dict(group.attrs)
        )

    @classmethod
    def _is_initialized(cls, group: Group) -> bool:
        """Return whether a root group already looks like SigMF-Zarr.

        Args:
            group: Root Zarr group to inspect.

        Returns:
            True when the group has the SigMF-Zarr schema marker and required
            groups.
        """
        schema_name = group.attrs.get("schema_name")
        return schema_name == cls.SCHEMA_NAME and cls._required_groups_present(
            group
        )

    @classmethod
    def _initialize_root(cls, group: Group) -> None:
        """Create the minimal top-level SigMF-Zarr structure.

        Args:
            group: Root Zarr group to initialize.
        """
        for group_name in cls.REQUIRED_GROUPS:
            group.require_group(group_name)
        group.attrs.update(
            {
                "schema_name": cls.SCHEMA_NAME,
                "schema_version": cls.SCHEMA_VERSION,
            }
        )

    @classmethod
    def _validate_root(cls, group: Group) -> None:
        """Validate that a root group contains a supported SigMF-Zarr store.

        Args:
            group: Root Zarr group to validate.

        Raises:
            ValueError: If the group is not a supported SigMF-Zarr store.
        """
        schema_name = group.attrs.get("schema_name")
        schema_version = group.attrs.get("schema_version")

        if schema_name != cls.SCHEMA_NAME:
            raise ValueError(
                "Target does not contain a SigMF-Zarr store: expected "
                f"schema_name={cls.SCHEMA_NAME!r}, got {schema_name!r}"
            )
        if schema_version != cls.SCHEMA_VERSION:
            raise ValueError(
                "Unsupported SigMF-Zarr schema version: expected "
                f"{cls.SCHEMA_VERSION}, got {schema_version!r}"
            )
        if not cls._required_groups_present(group):
            raise ValueError(
                "Target SigMF-Zarr store is missing one or more required "
                f"top-level groups: {cls.REQUIRED_GROUPS!r}"
            )

    def _initialize_or_validate(self, *, mode: AccessModeLiteral) -> None:
        """Initialize writable stores or validate existing ones.

        Args:
            mode: Resolved Zarr access mode.

        Raises:
            ValueError: If an existing target is not a valid SigMF-Zarr store.
        """
        # Append-style modes initialize only an empty target. Existing content
        # without the schema marker is never adopted implicitly.
        if mode in {"w", "w-"}:
            type(self)._initialize_root(self._group)
            return

        if mode in {"r", "r+"}:
            type(self)._validate_root(self._group)
            return
        if type(self)._is_initialized(self._group):
            type(self)._validate_root(self._group)
            return

        if type(self)._has_content(self._group):
            raise ValueError(
                "Target exists but is not a valid SigMF-Zarr store. "
                "Use mode='w' or overwrite=True to recreate it."
            )

        type(self)._initialize_root(self._group)

    @staticmethod
    def default_sample_chunks(
        sample_dtype: npt.DTypeLike,
        sample_shape: tuple[int, ...],
        num_samples: int,
        *,
        batched: bool = True,
        target_chunk_bytes: int | None = None,
    ) -> tuple[int, ...]:
        """Choose a sample-major chunk shape for recording samples.

        Args:
            sample_dtype: Sample array dtype.
            sample_shape: Per-sample shape.
            num_samples: Number of logical samples.
            batched: Whether the sample array includes a batch axis.
            target_chunk_bytes: Optional target chunk size in bytes.

        Returns:
            Chunk shape for the sample array.

        Raises:
            ValueError: If `target_chunk_bytes` is not positive.
        """
        if not batched:
            return sample_shape

        if target_chunk_bytes is None:
            # A bounded item batch avoids a single chunk for large datasets
            # while keeping individual fixed-size samples contiguous.
            max_batch = 1024
        else:
            if target_chunk_bytes <= 0:
                raise ValueError(
                    "target_chunk_bytes must be positive, got "
                    f"{target_chunk_bytes}"
                )

            dtype = np.dtype(sample_dtype)
            elements_per_sample = int(np.prod(sample_shape, dtype=np.int64))
            bytes_per_sample = dtype.itemsize * elements_per_sample
            max_batch = max(1, target_chunk_bytes // bytes_per_sample)

        batch = min(max_batch, max(1, num_samples))
        return (batch, *sample_shape)

    @staticmethod
    def default_index_chunks(num_values: int) -> tuple[int]:
        """Choose a default chunk shape for one-dimensional index arrays.

        Args:
            num_values: Number of values in the index.

        Returns:
            One-dimensional chunk shape.
        """
        return (min(4096, max(1, num_values)),)

    @property
    def _indexes_group(self) -> Group:
        """Writable store-wide index group used by managed mutation paths.

        Returns:
            Backing Zarr indexes group.
        """
        return cast(Group, self._group["indexes"])

    @property
    def indexes(self) -> ReadOnlyGroup:
        """Top-level group containing store-wide index arrays.

        Returns:
            Read-only store-wide indexes group.
        """
        return ReadOnlyGroup(self._indexes_group)

    @staticmethod
    def _create_array_v2(
        group: Group,
        name: str,
        values: npt.NDArray[Any],
        *,
        chunks: tuple[int, ...],
        compressor: CompressorLike,
        string_data: bool,
    ) -> Array[Any]:
        """Create an initialized Zarr format 2 array.

        Args:
            group: Parent group.
            name: Array name within `group`.
            values: Values used to initialize the array.
            chunks: Logical chunk shape.
            compressor: Format-2 compressor configuration.
            string_data: Whether values require variable-length UTF-8 storage.

        Returns:
            Created array.
        """
        if string_data:
            array = group.create_array(
                name,
                shape=values.shape,
                dtype=cast(Any, VariableLengthUTF8()),
                chunks=chunks,
                compressor=compressor,
            )
            array[:] = values
            return array
        return group.create_array(
            name,
            data=values,
            chunks=chunks,
            compressor=compressor,
        )

    @staticmethod
    def _create_array_v3(
        group: Group,
        name: str,
        values: npt.NDArray[Any],
        *,
        chunks: tuple[int, ...],
        shards: ShardsLike | None,
        compressors: CompressorsLike,
        compressor: CompressorLike,
        serializer: SerializerLike,
        string_data: bool,
    ) -> Array[Any]:
        """Create an initialized Zarr format 3 array.

        Args:
            group: Parent group.
            name: Array name within `group`.
            values: Values used to initialize the array.
            chunks: Logical chunk shape.
            shards: Optional physical shard shape.
            compressors: Compressor pipeline configuration.
            compressor: Optional single-compressor compatibility setting.
            serializer: Array serializer configuration.
            string_data: Whether values require variable-length UTF-8 storage.

        Returns:
            Created array.
        """
        if string_data:
            array = group.create_array(
                name,
                shape=values.shape,
                dtype=cast(Any, VariableLengthUTF8()),
                chunks=chunks,
                shards=shards,
                compressors=compressors,
                compressor=compressor,
                serializer=serializer,
            )
            array[:] = values
            return array
        return group.create_array(
            name,
            data=values,
            chunks=chunks,
            shards=shards,
            compressors=compressors,
            compressor=compressor,
            serializer=serializer,
        )

    def create_array(
        self,
        group: Group,
        name: str,
        data: npt.ArrayLike,
        *,
        overwrite: bool,
        chunks: tuple[int, ...] | None = None,
        shards: ShardsLike | None = None,
        compressors: CompressorsLike = "auto",
        compressor: CompressorLike = "auto",
        serializer: SerializerLike = "auto",
    ) -> Array[Any]:
        """Create one array under a group, optionally at a nested path.

        Args:
            group: Parent group in which to create the array.
            name: Array name or nested relative path under `group`.
            data: Array-like values used to initialize the new array.
            overwrite: Whether an existing array at the same path may be
                replaced.
            chunks: Optional chunk shape for the created array. When
                omitted, one-dimensional index-style chunks are used.
            shards: Optional shard shape passed through to Zarr.
            compressors: Optional compressors configuration passed through
                to Zarr.
            compressor: Optional single-compressor configuration passed
                through to Zarr.
            serializer: Optional serializer configuration passed through
                to Zarr.

        Returns:
            The created Zarr array.

        Raises:
            ValueError: If the target array already exists and
                `overwrite` is false, or an encoding available only in Zarr
                format 3 is requested for a Zarr format 2 store.
        """
        with replacement_transaction(
            group,
            name,
            overwrite=overwrite,
            parents=(self._refresh_root(),),
        ):
            self._invalidate_metadata_integrity()
            # Zarr paths always use "/" separators, independently of the OS.
            name_path = PurePosixPath(name)
            array_name = name_path.name
            array_group = (
                group
                if name_path.parent == PurePosixPath(".")
                else group.require_group(str(name_path.parent))
            )

            if array_name in array_group:
                if overwrite:
                    del array_group[array_name]
                else:
                    raise ValueError(
                        f"Array {name!r} already exists. Pass overwrite=True "
                        "to replace it"
                    )

            value_array = np.asarray(data)

            string_data = value_array.dtype.kind in {"U", "O"}
            if string_data and serializer == "auto":
                # UTF-8 codecs own the wire representation instead of fixing
                # a NumPy Unicode width.
                value_array = value_array.astype(str).astype(object)
                if self.zarr_format == 3:
                    serializer = VLenUTF8Codec()

            resolved_chunks = chunks
            if resolved_chunks is None:
                resolved_chunks = self.default_index_chunks(len(value_array))

            if self.zarr_format == 2:
                if shards is not None:
                    raise ValueError(
                        "Array sharding requires Zarr format 3. Omit shards "
                        "or create a Zarr format 3 store"
                    )
                if serializer != "auto":
                    raise ValueError(
                        "Explicit serializers require Zarr format 3"
                    )
                resolved_compressor = type(self)._compressor_v2(
                    compressors=compressors,
                    compressor=compressor,
                )
                return type(self)._create_array_v2(
                    array_group,
                    array_name,
                    value_array,
                    chunks=resolved_chunks,
                    compressor=resolved_compressor,
                    string_data=string_data,
                )

            return type(self)._create_array_v3(
                array_group,
                array_name,
                value_array,
                chunks=resolved_chunks,
                shards=shards,
                compressors=compressors,
                compressor=compressor,
                serializer=serializer,
                string_data=string_data,
            )

    @classmethod
    def _compressor_v2(
        cls,
        *,
        compressors: CompressorsLike,
        compressor: CompressorLike,
    ) -> NumcodecsCodec | None | Literal["auto"]:
        """Resolve generic array compression for Zarr format 2.

        Args:
            compressors: Zarr format 3 compressor pipeline.
            compressor: Single compressor configuration.

        Returns:
            A Zarr format 2 compressor configuration.

        Raises:
            ValueError: If multiple compressors or conflicting settings are
                supplied, or a codec cannot be represented in Zarr format 2.
        """
        if compressors == "auto":
            return SigMFRecording._sample_compressor_v2(compressor)
        if compressor != "auto":
            raise ValueError(
                "Specify either compressors or compressor, not both"
            )
        if compressors is None:
            return None
        if not isinstance(compressors, Sequence) or len(compressors) != 1:
            raise ValueError(
                "Zarr format 2 supports one compressor, not a compressor "
                "pipeline"
            )
        return SigMFRecording._sample_compressor_v2(
            cast(CompressorLike, compressors[0])
        )

    def add_index(
        self,
        index_name: str,
        values: npt.ArrayLike,
        *,
        overwrite: bool = False,
        chunks: tuple[int] | None = None,
    ) -> ReadOnlyArray:
        """Create or replace a one-dimensional store-wide index array.

        Args:
            index_name: Index array name.
            values: One-dimensional index values.
            overwrite: Whether an existing index may be replaced.
            chunks: Optional chunk shape.

        Returns:
            Created index array.

        Raises:
            ValueError: If `values` is not one-dimensional or the target index
                exists and `overwrite` is false.
        """
        value_array = np.asarray(values)
        if value_array.ndim != 1:
            raise ValueError(
                f"Index {index_name!r} must be one-dimensional, got "
                f"shape {value_array.shape}"
            )
        index = self.create_array(
            self._indexes_group,
            index_name,
            value_array,
            overwrite=overwrite,
            chunks=chunks or self.default_index_chunks(len(value_array)),
        )
        return ReadOnlyArray(index)

    def index(self, index_name: str) -> ReadOnlyArray:
        """Return one store-wide index array.

        Args:
            index_name: Index array name.

        Returns:
            Read-only store-wide index array.

        Raises:
            ValueError: If an earlier mutation did not complete.
        """
        index = cast(Array[Any], self._indexes_group[index_name])
        if index.attrs.get(self._INDEX_VALID_FIELD, True) is not True:
            reason = index.attrs.get(
                self._INDEX_INVALID_REASON_FIELD,
                "unknown reason",
            )
            raise ValueError(
                f"Index {index_name!r} is invalid: {reason}. Repair it "
                "with mutate_index() or replace it with add_index()"
            )
        if index.ndim != 1:
            raise ValueError(
                f"Index {index_name!r} must be one-dimensional, got "
                f"shape {index.shape}"
            )
        return ReadOnlyArray(index)

    def _invalid_store_index_reason(self) -> str | None:
        """Return the reason that any store-wide index is invalid.

        Returns:
            Error message for the first invalid index, or ``None``.
        """
        for index_name, index in sorted(
            self._indexes_group.members(max_depth=None)
        ):
            if not isinstance(index, Array):
                continue
            if index.attrs.get(self._INDEX_VALID_FIELD, True) is not True:
                reason = index.attrs.get(
                    self._INDEX_INVALID_REASON_FIELD,
                    "unknown reason",
                )
                return f"Index {index_name!r} is invalid: {reason}"
            if index.ndim != 1:
                return (
                    f"Index {index_name!r} must be one-dimensional, got "
                    f"shape {index.shape}"
                )
        return None

    @contextmanager
    def mutate_index(
        self,
        index_name: str,
    ) -> Generator[Array[Any], None, None]:
        """Provide writable access to one store-wide index.

        The index is marked invalid before writable access is granted. A
        successful context exit validates that it remains one-dimensional and
        marks it current again. If the body raises, it remains invalid.

        Args:
            index_name: Store-wide index name.

        Yields:
            Writable backing Zarr index array.

        Raises:
            KeyError: If the index does not exist.
            ValueError: If the mutated index is not one-dimensional.
        """
        if index_name not in self._indexes_group:
            raise KeyError(index_name)
        index = cast(Array[Any], self._indexes_group[index_name])
        self._invalidate_metadata_integrity()
        # Mark invalid before exposing the array. An exception or abandoned
        # mutation therefore cannot make incomplete data appear trustworthy.
        index.attrs[self._INDEX_VALID_FIELD] = False
        index.attrs[self._INDEX_INVALID_REASON_FIELD] = (
            "mutation did not complete"
        )
        yield index
        # Code after the yield runs only for a normal context-manager exit.
        if index.ndim != 1:
            raise ValueError(
                f"Index {index_name!r} must be one-dimensional, got "
                f"shape {index.shape}"
            )
        if self._INDEX_VALID_FIELD in index.attrs:
            del index.attrs[self._INDEX_VALID_FIELD]
        if self._INDEX_INVALID_REASON_FIELD in index.attrs:
            del index.attrs[self._INDEX_INVALID_REASON_FIELD]

    def decode_json_index(self, index_name: str) -> list[JSONObject]:
        """Decode a JSON-string store-wide index array into objects.

        Args:
            index_name: Index array name.

        Returns:
            Decoded JSON objects from the index.

        Raises:
            ValueError: If the index is invalid, or an entry is not a finite
                JSON object.
        """
        array = self.index(index_name)
        raw_values = np.asarray(array[:], dtype=object)
        result = []
        for position, value in enumerate(raw_values):
            name = f"index {index_name!r} entry {position}"
            try:
                decoded = json.loads(str(value))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{name} is not valid JSON") from exc
            result.append(json_object(decoded, name=name))
        return result

    @classmethod
    def open(
        cls,
        store: StoreLike | str | PathLike[str],
        *,
        mode: Literal["r", "r+"] = "r",
        storage_options: dict[str, Any] | None = None,
        zarr_format: ZarrFormat | None = None,
    ) -> SigMFZarrStore:
        """Open an existing SigMF-Zarr store without creating anything.

        Args:
            store: Zarr store backend, path, or URL.
            mode: `"r"` for read-only access or `"r+"` for writes.
            storage_options: Optional backend-specific storage options.
            zarr_format: Optional physical Zarr format requirement. When
                omitted, the existing format is auto-detected.

        Returns:
            Opened SigMF-Zarr store.

        Raises:
            FileNotFoundError: If the target does not exist.
            ValueError: If `mode` is unsupported, or the target is not a
                supported SigMF-Zarr store in the requested format.
        """
        if mode not in {"r", "r+"}:
            raise ValueError(
                f"SigMFZarrStore.open() mode must be 'r' or 'r+', got {mode!r}"
            )
        return cls(
            store=store,
            mode=mode,
            storage_options=storage_options,
            zarr_format=zarr_format,
        )

    @classmethod
    def create(
        cls,
        store: StoreLike | str | PathLike[str] | None = None,
        *,
        overwrite: bool = False,
        storage_options: dict[str, Any] | None = None,
        zarr_format: ZarrFormat | None = None,
    ) -> SigMFZarrStore:
        """Create an empty SigMF-Zarr store.

        Args:
            store: Optional Zarr store backend, path, or URL. When omitted,
                the store is held in memory.
            overwrite: Whether to replace existing content at the target.
            storage_options: Optional backend-specific storage options.
            zarr_format: Physical Zarr format. Defaults to Zarr format 3.

        Returns:
            Created SigMF-Zarr store.

        Raises:
            FileExistsError: If the target already contains a Zarr hierarchy
                and `overwrite` is false.
        """
        try:
            return cls(
                store=store,
                mode="w" if overwrite else "w-",
                storage_options=storage_options,
                zarr_format=zarr_format,
            )
        except FileExistsError as exc:
            raise FileExistsError(
                "Target already contains a Zarr hierarchy. Pass "
                "overwrite=True to replace it, or use SigMFZarrStore.open()"
            ) from exc

    def close(self) -> None:
        """Release the backend store.

        Zarr reopens local and in-memory backends on later access, so
        close a store when it is no longer needed rather than to prevent
        access.
        """
        self._group.store.close()

    def __enter__(self) -> SigMFZarrStore:
        """Enter a store context manager.

        Returns:
            This store.
        """
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Close the backend store when leaving a context manager.

        Args:
            exc_type: Exception type, if an exception was raised.
            exc: Exception instance, if an exception was raised.
            tb: Traceback object, if an exception was raised.
        """
        self.close()

    def __repr__(self) -> str:
        """Return a concise debug representation.

        Returns:
            Debug representation string.
        """
        return (
            f"{type(self).__name__}("
            f"num_recordings={len(self.list_recordings())}, "
            f"num_collections={len(self.list_collections())})"
        )

    def __iter__(self) -> Iterator[SigMFRecording]:
        """Iterate over recordings in sorted name order.

        Returns:
            Iterator over recording wrappers.
        """
        return iter(self.recordings)

    def __getitem__(self, name: str) -> SigMFRecording | SigMFCollection:
        """Return a recording or collection by name.

        Args:
            name: Recording or collection name.

        Returns:
            Recording or collection wrapper.

        Raises:
            KeyError: If `name` matches neither object type, or if it
                matches both a recording and a collection.
        """
        has_recording = name in self.recordings
        has_collection = name in self.collections

        if has_recording and has_collection:
            raise KeyError(
                f"Name {name!r} is ambiguous: it matches both a "
                "recording and a collection"
            )
        if has_recording:
            return self.recordings.open(name)
        if has_collection:
            return self.collections.open(name)
        raise KeyError(name)

    @property
    def _recordings_group(self) -> Group:
        """Top-level Zarr group containing recording subgroups.

        Returns:
            Top-level recordings group.
        """
        return cast(Group, self._group["recordings"])

    @property
    def _collections_group(self) -> Group:
        """Top-level Zarr group containing collection subgroups.

        Returns:
            Top-level collections group.
        """
        return cast(Group, self._group["collections"])

    @property
    def recordings(self) -> SigMFRecordings:
        """Proxy view over recordings in this store.

        Returns:
            Recordings proxy view.
        """
        return SigMFRecordings(self)

    @property
    def collections(self) -> SigMFCollections:
        """Proxy view over collections in this store.

        Returns:
            Collections proxy view.
        """
        return SigMFCollections(self)

    def list_recordings(self) -> tuple[str, ...]:
        """Return recording identifiers in sorted order.

        Returns:
            Recording names.
        """
        return tuple(sorted(self._recordings_group.group_keys()))

    def list_collections(self) -> tuple[str, ...]:
        """Return collection identifiers in sorted order.

        Returns:
            Collection names.
        """
        return tuple(sorted(self._collections_group.group_keys()))

    def metadata(self) -> dict[str, Any]:
        """Return all root attributes as a plain dictionary.

        Returns:
            Root attributes copied into a dictionary.
        """
        return dict(self._refresh_root().attrs)

    def info(self) -> str:
        """Return a human-readable summary of the container.

        Returns:
            Multiline store summary.
        """
        index_names = sorted(
            name
            for name, member in self.indexes.members(max_depth=None)
            if isinstance(member, ReadOnlyArray)
        )
        lines = [
            f"{type(self).__name__}(",
            f"  schema_name={self.SCHEMA_NAME!r},",
            f"  schema_version={self.SCHEMA_VERSION!r},",
            f"  zarr_format={self.zarr_format!r},",
            f"  metadata_sha512={self.metadata_sha512!r},",
            f"  recordings={list(self.list_recordings())!r},",
            f"  collections={list(self.list_collections())!r},",
            f"  indexes={index_names},",
            ")",
        ]
        return "\n".join(lines)


__all__ = ["SigMFZarrStore"]

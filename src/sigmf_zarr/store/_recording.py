"""Zarr-backed SigMF-Zarr recording implementation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import numpy as np
import numpy.typing as npt
from numcodecs import CRC32C as NumcodecsCRC32C
from numcodecs import Blosc as NumcodecsBlosc
from numcodecs import Zstd as NumcodecsZstd
from numcodecs.abc import Codec as NumcodecsCodec
from sigmf import SHA512_KEY
from sigmf.sigmffile import SigMFAccessError, SigMFFile
from zarr.codecs import (
    BloscCodec,
    Crc32cCodec,
    ZstdCodec,
)
from zarr.core.array import (
    Array,
    CompressorLike,
    CompressorsLike,
    ShardsLike,
)
from zarr.core.group import Group

from sigmf_zarr.integrity import (
    INTEGRITY_ATTR,
)
from sigmf_zarr.json import (
    JSONObject,
    JSONValue,
    json_object,
    json_object_list,
    json_value,
)
from sigmf_zarr.readonly import ReadOnlyArray, ReadOnlyGroup
from sigmf_zarr.store._common import ChecksumName, ZarrFormat

if TYPE_CHECKING:
    from sigmf_zarr.store._container import SigMFZarrStore


class SigMFRecording:
    """Thin wrapper around one SigMF-Zarr recording stored in Zarr.

    This class models the per-recording part of the container. The
    core SigMF mapping is kept on the recording group itself:

    - The `samples` array stores the recording's sample tensor.
    - The `channels/` group stores metadata for each explicit sample channel.
    - The optional `item_metadata` array stores JSON bundles aligned with batch
      items.
    - The recording attributes `global`, `captures`, and `annotations` mirror
      the standard SigMF metadata objects.

    SigMF extension declarations and namespaced extension metadata live
    in the recording's `global` metadata object. Dense metadata arrays
    aligned with runtime sample axes live under `indexes/`, while
    extension-owned Zarr arrays may still live under `extensions/` when
    an extension needs structured data outside the generic index model.
    Sample chunks use CRC32C integrity checks by default.

    In this schema, a recording may store either one unbatched sample
    array or a homogeneous batch of equal-shaped samples with shared
    recording metadata. That is close to SigMF's recording concept,
    but adapted for Zarr-backed tensor storage and contiguous loading.
    """

    _store: SigMFZarrStore
    """Parent SigMF-Zarr store containing this recording."""

    _recording_name: str
    """Recording identifier relative to the store's `recordings/` group."""

    ZARR_EXTENSION_NAMESPACE: ClassVar[str] = "sigmf-zarr"
    """Namespace used for SigMF-Zarr metadata extension fields."""

    ZARR_EXTENSION_VERSION: ClassVar[str] = "0.1.0"
    """Version of the SigMF-Zarr metadata extension schema."""

    _SHA512_ENCODING_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "core:datatype",
            "sigmf-zarr:sample-axes",
        }
    )
    """Metadata fields whose changes alter standard SigMF dataset bytes."""

    _INTEGRITY_SAMPLE_FIELD: ClassVar[str] = "sample_sha512"
    """Internal integrity field containing the logical sample-array hash."""

    _INTEGRITY_METADATA_FIELD: ClassVar[str] = "metadata_sha512"
    """Internal integrity field containing the logical metadata hash."""

    _INDEX_VALID_FIELD: ClassVar[str] = "sigmf-zarr:valid"
    """Index attribute indicating whether alignment is still valid."""

    _INDEX_INVALID_REASON_FIELD: ClassVar[str] = (
        "sigmf-zarr:invalid-reason"
    )
    """Index attribute explaining why an index is invalid."""

    def __init__(
        self,
        store: SigMFZarrStore,
        recording_name: str,
        *,
        create: bool | None = None,
        overwrite: bool = False,
        batched: bool = False,
        sample_dtype: npt.DTypeLike = np.float32,
        sample_shape: tuple[int, ...] | None = None,
        sample_axes: tuple[str, ...] | None = None,
        global_metadata: JSONObject | None = None,
        captures: list[JSONObject] | None = None,
        annotations: list[JSONObject] | None = None,
        channel_metadata: Sequence[JSONObject] | None = None,
        sample_chunks: tuple[int, ...] | None = None,
        sample_shards: ShardsLike | None = None,
        sample_compressor: CompressorLike = "auto",
        sample_checksum: ChecksumName | None = "crc32c",
    ) -> None:
        """Initialize a recording wrapper and optionally create it.

        The constructor binds the wrapper to a parent store and
        recording name. Depending on `create`, `overwrite`, and the
        store's access mode, it either opens an existing recording or
        creates a new one.

        Args:
            store: Parent SigMF-Zarr store containing the recording.
            recording_name: Recording identifier under `recordings/`.
            create: Whether a missing recording may be created. When
                `None`, the behavior follows the store mode: creation
                is allowed in writable modes and disallowed in read
                mode.
            overwrite: Whether an existing recording should be replaced
                before initialization.
            batched: Whether the `samples` array includes a leading
                batch axis of homogeneous samples.
            sample_dtype: Element dtype for the sample array when
                creating a new recording.
            sample_shape: Per-sample tensor shape when creating a new
                recording. When `batched` is true, this excludes the
                leading batch axis. Otherwise, it is the full sample array
                shape.
            sample_axes: Optional axis names for `sample_shape`. When omitted,
                axes are inferred from the shape.
            global_metadata: SigMF-like global metadata to store on a
                newly created recording.
            captures: SigMF-like captures metadata to store on a newly
                created recording.
            annotations: SigMF-like annotations metadata to store on a
                newly created recording.
            channel_metadata: Optional metadata object for each declared
                channel in a newly created recording.
            sample_chunks: Optional chunk shape for the sample array
                when creating a new recording.
            sample_shards: Optional shard shape for the sample array
                when creating a new recording.
            sample_compressor: Optional compressor for the sample array
                when creating a new recording.
            sample_checksum: Optional chunk-level checksum for the sample
                array. Defaults to `crc32c`.

        Raises:
            KeyError: If the recording does not exist and creation is
                not allowed.
            ValueError: If recording creation is requested without a
                valid `sample_shape`.
        """
        self._store = store
        self._recording_name = recording_name
        self._ensure_exists(
            create=create,
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
        self._validate_metadata_attrs()
        self._validate_sample_array_shape()
        self._validate_item_metadata_array()
        self._validate_channel_metadata_groups()
        self._validate_sample_checksum()

    def _validate_metadata_attrs(self) -> None:
        """Validate recording JSON attributes and stored dtype metadata.

        Raises:
            ValueError: If metadata is malformed or the declared dtype does
                not describe the samples array.
        """
        global_metadata = json_object(
            self._raw_group.attrs.get("global", {}),
            name=f"recording {self.name!r} global metadata",
        )
        json_object_list(
            self._raw_group.attrs.get("captures", []),
            name=f"recording {self.name!r} captures",
        )
        json_object_list(
            self._raw_group.attrs.get("annotations", []),
            name=f"recording {self.name!r} annotations",
        )
        field = f"{self.ZARR_EXTENSION_NAMESPACE}:dtype"
        declared_dtype = global_metadata.get(field)
        if not isinstance(declared_dtype, str):
            raise ValueError(
                f"Recording {self.name!r} is missing required {field!r} "
                "metadata"
            )
        try:
            normalized_dtype = np.dtype(declared_dtype).str
        except TypeError as exc:
            raise ValueError(
                f"Recording {self.name!r} has invalid {field!r}: "
                f"{declared_dtype!r}"
            ) from exc
        actual_dtype = np.dtype(cast(Any, self._samples_array.dtype)).str
        if normalized_dtype != actual_dtype:
            raise ValueError(
                f"Recording {self.name!r} {field!r} value {declared_dtype!r} "
                f"does not match samples dtype {actual_dtype!r}"
            )

    def _ensure_exists(
        self,
        *,
        create: bool | None,
        overwrite: bool,
        batched: bool,
        sample_dtype: npt.DTypeLike,
        sample_shape: tuple[int, ...] | None,
        sample_axes: tuple[str, ...] | None,
        global_metadata: JSONObject | None,
        captures: list[JSONObject] | None,
        annotations: list[JSONObject] | None,
        channel_metadata: Sequence[JSONObject] | None,
        sample_chunks: tuple[int, ...] | None,
        sample_shards: ShardsLike | None,
        sample_compressor: CompressorLike,
        sample_checksum: ChecksumName | None,
    ) -> None:
        """Create or validate the backing recording group.

        Args:
            create: Whether a missing recording may be created.
            overwrite: Whether an existing recording should be replaced.
            batched: Whether the sample array includes a batch axis.
            sample_dtype: Element dtype for a new sample array.
            sample_shape: Per-sample shape for a new sample array.
            sample_axes: Optional axis names for a new sample array.
            global_metadata: SigMF-like global metadata for a new recording.
            captures: Capture metadata for a new recording.
            annotations: Annotation metadata for a new recording.
            channel_metadata: Optional metadata for each channel.
            sample_chunks: Optional sample chunk shape.
            sample_shards: Optional sample shard shape.
            sample_compressor: Optional sample compressor.
            sample_checksum: Optional chunk-level sample checksum.

        Raises:
            KeyError: If the recording does not exist and creation is not
                allowed.
            ValueError: If recording creation is requested without a valid
                `sample_shape`.
        """
        should_create = self._store._mode != "r" if create is None else create
        exists = self.name in self._store._recordings_group

        if exists and not overwrite:
            return
        if exists and overwrite:
            self._store._invalidate_metadata_integrity()
            del self._store._recordings_group[self.name]
            exists = False

        if not exists and not should_create:
            raise KeyError(self.name)

        if not exists:
            self._store._invalidate_metadata_integrity()
            self._create_group(
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

    def _create_group(
        self,
        *,
        batched: bool,
        sample_dtype: npt.DTypeLike,
        sample_shape: tuple[int, ...] | None,
        sample_axes: tuple[str, ...] | None,
        global_metadata: JSONObject | None,
        captures: list[JSONObject] | None,
        annotations: list[JSONObject] | None,
        channel_metadata: Sequence[JSONObject] | None,
        sample_chunks: tuple[int, ...] | None,
        sample_shards: ShardsLike | None,
        sample_compressor: CompressorLike,
        sample_checksum: ChecksumName | None,
    ) -> None:
        """Create the Zarr group and arrays for a new recording.

        Args:
            batched: Whether the sample array includes a batch axis.
            sample_dtype: Element dtype for the sample array.
            sample_shape: Per-sample shape for the sample array.
            sample_axes: Optional axis names for the sample array.
            global_metadata: SigMF-like global metadata.
            captures: Capture metadata.
            annotations: Annotation metadata.
            channel_metadata: Optional metadata for each channel.
            sample_chunks: Optional sample chunk shape.
            sample_shards: Optional sample shard shape.
            sample_compressor: Optional sample compressor.
            sample_checksum: Optional chunk-level sample checksum.

        Raises:
            ValueError: If `sample_shape` is empty or contains non-positive
                dimensions.
        """
        if not sample_shape:
            raise ValueError("sample_shape must not be empty when creating")
        if any(dim <= 0 for dim in sample_shape):
            raise ValueError(
                f"sample_shape dimensions must be positive, got {sample_shape}"
            )
        if sample_checksum not in {None, "crc32c"}:
            raise ValueError(
                f"Unsupported sample checksum {sample_checksum!r}"
            )
        if self._store.zarr_format == 2 and sample_shards is not None:
            raise ValueError(
                "Sample sharding requires Zarr format 3. Omit sample_shards "
                "or create a Zarr format 3 store"
            )
        sample_compressor_v2 = (
            type(self)._sample_compressor_v2(sample_compressor)
            if self._store.zarr_format == 2
            else None
        )
        sample_compressors_v3 = (
            type(self)._sample_compressors(
                sample_compressor,
                sample_checksum=sample_checksum,
            )
            if self._store.zarr_format == 3
            else None
        )

        normalized_axes = type(self)._normalize_sample_axes(
            sample_shape,
            sample_axes=sample_axes,
        )
        normalized_channels = type(self)._normalize_channel_metadata(
            sample_shape,
            sample_axes=normalized_axes,
            channel_metadata=channel_metadata,
        )
        group = self._store._recordings_group.create_group(self.name)
        chunk_shape = sample_chunks or type(self._store).default_sample_chunks(
            sample_dtype,
            sample_shape,
            1,
            batched=batched,
        )
        array_shape = (0, *sample_shape) if batched else sample_shape
        if self._store.zarr_format == 2:
            group.create_array(
                "samples",
                shape=array_shape,
                dtype=sample_dtype,
                chunks=chunk_shape,
                compressor=sample_compressor_v2,
                filters=(
                    [NumcodecsCRC32C()]
                    if sample_checksum == "crc32c"
                    else None
                ),
            )
        else:
            group.create_array(
                "samples",
                shape=array_shape,
                dtype=sample_dtype,
                chunks=chunk_shape,
                shards=sample_shards,
                compressors=sample_compressors_v3,
            )
        group.create_group("extensions")
        group.create_group("indexes")
        channels = group.create_group("channels")
        for channel_index, metadata in enumerate(normalized_channels):
            channel = channels.create_group(str(channel_index))
            channel.attrs.update(metadata)
        recording_global = self._normalize_global_metadata(
            global_metadata,
            sample_dtype=cast(Array, group["samples"]).dtype,
            sample_shape=sample_shape,
            sample_axes=normalized_axes,
            sample_checksum=sample_checksum,
        )
        group.attrs.update(
            {
                "global": recording_global,
                "captures": json_object_list(
                    captures or [], name="captures"
                ),
                "annotations": json_object_list(
                    annotations or [], name="annotations"
                ),
            }
        )

    @classmethod
    def _sample_compressors(
        cls,
        sample_compressor: CompressorLike,
        *,
        sample_checksum: ChecksumName | None,
    ) -> CompressorsLike:
        """Build the sample array byte-to-byte codec pipeline.

        Args:
            sample_compressor: Requested sample compressor.
            sample_checksum: Requested chunk-level checksum.

        Returns:
            Zarr compressors configuration with the checksum last.

        Raises:
            ValueError: If the checksum name is unsupported.
        """
        if sample_checksum not in {None, "crc32c"}:
            raise ValueError(
                f"Unsupported sample checksum {sample_checksum!r}"
            )

        if sample_compressor == "auto":
            if sample_checksum is None:
                return "auto"
            codecs: list[object] = [ZstdCodec()]
        elif sample_compressor is None:
            codecs = []
        else:
            codecs = [sample_compressor]

        if sample_checksum == "crc32c":
            # Checksums protect compressed bytes and therefore terminate the
            # format-3 byte-to-byte codec pipeline.
            codecs.append(Crc32cCodec())
        return cast(CompressorsLike, tuple(codecs))

    @staticmethod
    def _sample_compressor_v2(
        sample_compressor: CompressorLike,
    ) -> NumcodecsCodec | None | Literal["auto"]:
        """Translate a sample compressor to a Zarr format 2 codec.

        Args:
            sample_compressor: Requested sample compressor.

        Returns:
            Equivalent numcodecs compressor for Zarr format 2.

        Raises:
            ValueError: If the compressor cannot be represented in Zarr format
                2.
        """
        if sample_compressor == "auto":
            return NumcodecsZstd()
        if sample_compressor is None:
            return None
        if isinstance(sample_compressor, NumcodecsCodec):
            return sample_compressor
        if isinstance(sample_compressor, ZstdCodec):
            return NumcodecsZstd(
                level=sample_compressor.level,
                checksum=sample_compressor.checksum,
            )
        if isinstance(sample_compressor, BloscCodec):
            shuffle_name = getattr(
                sample_compressor.shuffle,
                "value",
                sample_compressor.shuffle,
            )
            shuffle = {
                "noshuffle": NumcodecsBlosc.NOSHUFFLE,
                "shuffle": NumcodecsBlosc.SHUFFLE,
                "bitshuffle": NumcodecsBlosc.BITSHUFFLE,
            }.get(str(shuffle_name))
            if shuffle is None:
                raise ValueError(
                    "Unsupported Blosc shuffle setting for Zarr format 2: "
                    f"{shuffle_name!r}"
                )
            cname = getattr(
                sample_compressor.cname,
                "value",
                sample_compressor.cname,
            )
            blosc_options: dict[str, Any] = {
                "cname": str(cname),
                "clevel": sample_compressor.clevel,
                "shuffle": shuffle,
                "blocksize": sample_compressor.blocksize,
            }
            if sample_compressor.typesize is not None:
                blosc_options["typesize"] = sample_compressor.typesize
            try:
                return NumcodecsBlosc(**blosc_options)
            except TypeError as exc:
                # Older numcodecs releases do not accept the Zarr 3 typesize
                # setting. Retry only for that compatibility failure.
                if "typesize" not in blosc_options or "typesize" not in str(
                    exc
                ):
                    raise
                del blosc_options["typesize"]
                return NumcodecsBlosc(**blosc_options)
        raise ValueError(
            "Sample compressor cannot be represented in Zarr format 2: "
            f"{sample_compressor!r}"
        )

    @classmethod
    def _normalize_channel_metadata(
        cls,
        sample_shape: tuple[int, ...],
        *,
        sample_axes: tuple[str, ...],
        channel_metadata: Sequence[JSONObject] | None,
    ) -> tuple[JSONObject, ...]:
        """Validate metadata supplied for an explicit channel axis.

        Args:
            sample_shape: Logical sample tensor shape.
            sample_axes: Validated logical sample axes.
            channel_metadata: Optional metadata object for each channel.

        Returns:
            One copied metadata object per explicit channel.

        Raises:
            ValueError: If metadata is supplied without a channel axis, has
                the wrong length, or is not JSON serializable.
        """
        if "channel" not in sample_axes:
            if channel_metadata is not None:
                raise ValueError(
                    "channel_metadata requires a declared 'channel' axis"
                )
            return ()

        num_channels = sample_shape[sample_axes.index("channel")]
        if channel_metadata is None:
            return tuple({} for _ in range(num_channels))
        if len(channel_metadata) != num_channels:
            raise ValueError(
                f"Expected metadata for {num_channels} channels, got "
                f"{len(channel_metadata)}"
            )

        normalized: list[JSONObject] = []
        for channel_index, metadata in enumerate(channel_metadata):
            normalized.append(
                json_object(
                    dict(metadata),
                    name=f"channel {channel_index} metadata",
                )
            )
        return tuple(normalized)

    @classmethod
    def _zarr_extension_declaration(cls) -> JSONObject:
        """Return the SigMF extension declaration for SigMF-Zarr.

        Returns:
            SigMF extension declaration object.
        """
        return {
            "name": cls.ZARR_EXTENSION_NAMESPACE,
            "version": cls.ZARR_EXTENSION_VERSION,
            "optional": True,
        }

    @classmethod
    def _normalize_global_metadata(
        cls,
        global_metadata: JSONObject | None,
        *,
        sample_dtype: npt.DTypeLike,
        sample_shape: tuple[int, ...],
        sample_axes: tuple[str, ...] | None,
        sample_checksum: ChecksumName | None,
    ) -> JSONObject:
        """Merge SigMF-Zarr extension metadata into `global`.

        Args:
            global_metadata: Optional source global metadata.
            sample_dtype: Element dtype for the sample array.
            sample_shape: Per-sample shape for the sample array.
            sample_axes: Optional axis names for the sample array.
            sample_checksum: Optional chunk-level sample checksum.

        Returns:
            Global metadata with SigMF-Zarr extension fields present.
        """
        normalized_axes = cls._normalize_sample_axes(
            sample_shape,
            sample_axes=sample_axes,
        )
        extension_field = "core:extensions"
        zarr_prefix = f"{cls.ZARR_EXTENSION_NAMESPACE}:"
        source_metadata = json_object(
            global_metadata or {}, name="global metadata"
        )
        # Replace any prior SigMF-Zarr declaration and storage fields as one
        # unit so caller-supplied values cannot contradict the array layout.
        normalized = {
            key: value
            for key, value in source_metadata.items()
            if key != extension_field and not key.startswith(zarr_prefix)
        }
        declared_extensions = json_object_list(
            source_metadata.get(extension_field, []),
            name=extension_field,
        )
        declared_extensions = [
            extension
            for extension in declared_extensions
            if extension.get("name") != cls.ZARR_EXTENSION_NAMESPACE
        ]
        declared_extensions.append(cls._zarr_extension_declaration())
        normalized[extension_field] = cast(JSONValue, declared_extensions)
        normalized[f"{cls.ZARR_EXTENSION_NAMESPACE}:dtype"] = np.dtype(
            sample_dtype
        ).str
        normalized[f"{cls.ZARR_EXTENSION_NAMESPACE}:sample-shape"] = [
            int(dim) for dim in sample_shape
        ]
        normalized[f"{cls.ZARR_EXTENSION_NAMESPACE}:sample-axes"] = list(
            normalized_axes
        )
        num_channels = (
            sample_shape[normalized_axes.index("channel")]
            if "channel" in normalized_axes
            else 1
        )
        normalized[f"{cls.ZARR_EXTENSION_NAMESPACE}:num-channels"] = int(
            num_channels
        )
        normalized[f"{cls.ZARR_EXTENSION_NAMESPACE}:checksum"] = (
            sample_checksum
        )
        return normalized

    @classmethod
    def _default_sample_axes(
        cls,
        sample_shape: tuple[int, ...],
    ) -> tuple[str, ...]:
        """Infer a default axis order for a logical sample shape.

        Args:
            sample_shape: Logical sample shape.

        Returns:
            Inferred sample-axis names.

        Raises:
            ValueError: If the shape cannot be mapped to default axes.
        """
        if len(sample_shape) == 1:
            return ("time",)
        if len(sample_shape) == 2:
            if sample_shape[0] == 2:
                return ("iq", "time")
            if sample_shape[1] == 2:
                return ("time", "iq")
            return ("channel", "time")
        if len(sample_shape) == 3:
            if sample_shape[1] == 2:
                return ("channel", "iq", "time")
            if sample_shape[0] == 2:
                return ("iq", "channel", "time")
            if sample_shape[2] == 2:
                return ("time", "channel", "iq")
        raise ValueError(
            "sample_axes must be provided for sample_shape "
            f"{sample_shape!r}"
        )

    @classmethod
    def _normalize_sample_axes(
        cls,
        sample_shape: tuple[int, ...],
        *,
        sample_axes: tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        """Validate and normalize sample-axis metadata.

        Args:
            sample_shape: Logical sample shape.
            sample_axes: Optional axis names for the sample shape.

        Returns:
            Normalized axis-name tuple.

        Raises:
            ValueError: If axis metadata is missing, ambiguous, or invalid.
        """
        axes = (
            cls._default_sample_axes(sample_shape)
            if sample_axes is None
            else tuple(sample_axes)
        )
        if len(axes) != len(sample_shape):
            raise ValueError(
                "sample_axes must have the same length as sample_shape: "
                f"got {axes!r} for {sample_shape!r}"
            )
        if len(set(axes)) != len(axes):
            raise ValueError(
                f"sample_axes must not contain duplicates: {axes!r}"
            )
        if "item" in axes:
            raise ValueError(
                "sample_axes must not contain reserved axis 'item'"
            )
        if axes.count("time") != 1:
            raise ValueError(
                "sample_axes must contain exactly one 'time' axis"
            )
        if axes.count("iq") > 1:
            raise ValueError("sample_axes may contain at most one 'iq' axis")
        if axes.count("channel") > 1:
            raise ValueError(
                "sample_axes may contain at most one 'channel' axis"
            )
        if "iq" in axes:
            iq_axis = axes.index("iq")
            if sample_shape[iq_axis] != 2:
                raise ValueError(
                    "The 'iq' axis must have length 2, got "
                    f"{sample_shape[iq_axis]}"
                )
        return axes

    def _stored_sample_shape(self) -> tuple[int, ...]:
        """Return the stored logical sample shape metadata.

        Returns:
            Logical sample shape stored in global metadata.

        Raises:
            ValueError: If the metadata is missing or malformed.
        """
        field = f"{self.ZARR_EXTENSION_NAMESPACE}:sample-shape"
        value = self.global_metadata.get(field)
        if not isinstance(value, list):
            raise ValueError(
                f"Recording {self.name!r} is missing required {field!r} "
                "metadata"
            )

        sample_shape: list[int] = []
        for dim in value:
            if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
                raise ValueError(
                    f"Recording {self.name!r} has invalid {field!r} "
                    f"metadata: {value!r}"
                )
            sample_shape.append(dim)
        return tuple(sample_shape)

    def _stored_sample_axes(self) -> tuple[str, ...]:
        """Return the stored logical sample-axis metadata.

        Returns:
            Logical sample-axis names stored in global metadata.

        Raises:
            ValueError: If the metadata is missing or malformed.
        """
        field = f"{self.ZARR_EXTENSION_NAMESPACE}:sample-axes"
        value = self.global_metadata.get(field)
        if not isinstance(value, list):
            raise ValueError(
                f"Recording {self.name!r} is missing required {field!r} "
                "metadata"
            )
        if not all(isinstance(axis, str) for axis in value):
            raise ValueError(
                f"Recording {self.name!r} has invalid {field!r} metadata: "
                f"{value!r}"
            )
        return tuple(cast(list[str], value))

    def _validate_sample_array_shape(self) -> None:
        """Validate that sample rank agrees with stored sample shape.

        Raises:
            ValueError: If the sample array rank is inconsistent with
                `sigmf-zarr:sample-shape` or `sigmf-zarr:sample-axes`.
        """
        sample_shape = self._stored_sample_shape()
        _ = self.sample_axes
        actual_shape = (
            tuple(int(dim) for dim in self.samples.shape[1:])
            if self.batched
            else tuple(int(dim) for dim in self.samples.shape)
        )
        if actual_shape != sample_shape:
            raise ValueError(
                f"Recording {self.name!r} sample array shape {actual_shape!r} "
                "is inconsistent with stored "
                f"{self.ZARR_EXTENSION_NAMESPACE}:sample-shape "
                f"{sample_shape!r}"
            )

    def _validate_item_metadata_array(self) -> None:
        """Validate optional per-item metadata storage.

        Raises:
            ValueError: If the recording layout, array shape, checksum, or
                an encoded metadata entry is invalid.
        """
        present = "item_metadata" in self._raw_group.array_keys()
        if not present:
            return
        if not self.batched:
            raise ValueError(
                f"Recording {self.name!r} has item_metadata but is not batched"
            )

        item_metadata = self._item_metadata_array
        expected_shape = (len(self),)
        if item_metadata.shape != expected_shape:
            raise ValueError(
                f"Recording {self.name!r} item_metadata shape "
                f"{item_metadata.shape!r} does not match item axis shape "
                f"{expected_shape!r}"
            )
        array_metadata = cast(Any, item_metadata.metadata)
        if self._store.zarr_format == 2:
            filters = array_metadata.filters or ()
            has_crc32c = any(
                isinstance(codec, NumcodecsCRC32C) for codec in filters
            )
        else:
            has_crc32c = type(self)._codecs_have_crc32c(
                array_metadata.codecs
            )
        if not has_crc32c:
            raise ValueError(
                f"Recording {self.name!r} item_metadata chunks must use "
                "CRC32C"
            )

        # Validate by physical chunk to bound memory and avoid one read per
        # item on remote stores.
        chunk_length = max(int(item_metadata.chunks[0]), 1)
        for start in range(0, len(self), chunk_length):
            stop = min(start + chunk_length, len(self))
            encoded_entries = np.asarray(
                item_metadata[start:stop], dtype=object
            )
            for offset, raw in enumerate(encoded_entries):
                item_index = start + offset
                type(self)._decode_item_metadata_entry(
                    raw,
                    item_index=item_index,
                )

    def _validate_channel_metadata_groups(self) -> None:
        """Validate channel count metadata and channel subgroups.

        Raises:
            ValueError: If the stored channel count or subgroup names do not
                match the declared sample axes.
        """
        if "channels" not in self._raw_group.group_keys():
            raise ValueError(
                f"Recording {self.name!r} is missing required 'channels' group"
            )

        field = f"{self.ZARR_EXTENSION_NAMESPACE}:num-channels"
        declared = self.global_metadata.get(field)
        if (
            isinstance(declared, bool)
            or not isinstance(declared, int)
            or declared <= 0
        ):
            raise ValueError(
                f"Recording {self.name!r} has invalid {field!r} metadata: "
                f"{declared!r}"
            )
        if declared != self.num_channels:
            raise ValueError(
                f"Recording {self.name!r} {field!r} value {declared} does "
                f"not match channel axis length {self.num_channels}"
            )

        expected_groups = (
            {str(index) for index in range(self.num_channels)}
            if "channel" in self.sample_axes
            else set()
        )
        actual_groups = set(self._channels_group.group_keys())
        if actual_groups != expected_groups:
            raise ValueError(
                f"Recording {self.name!r} channel groups "
                f"{sorted(actual_groups)!r} do not match expected channels "
                f"{sorted(expected_groups)!r}"
            )

    @classmethod
    def _codecs_have_crc32c(cls, codecs: Sequence[object]) -> bool:
        """Return whether a codec pipeline protects sample chunks with CRC32C.

        Args:
            codecs: Top-level Zarr codec sequence.

        Returns:
            True if CRC32C appears directly or inside a sharding codec's
            inner chunk codec sequence.
        """
        for codec in codecs:
            if isinstance(codec, Crc32cCodec):
                return True
            nested = getattr(codec, "codecs", None)
            if isinstance(nested, tuple | list) and cls._codecs_have_crc32c(
                nested
            ):
                return True
        return False

    def _validate_sample_checksum(self) -> None:
        """Validate checksum metadata against the sample codec pipeline.

        Raises:
            ValueError: If checksum metadata is invalid or disagrees with
                the sample array codecs.
        """
        field = f"{self.ZARR_EXTENSION_NAMESPACE}:checksum"
        declared = self.global_metadata.get(field)
        if declared not in {None, "crc32c"}:
            raise ValueError(
                f"Recording {self.name!r} has unsupported {field!r} "
                f"metadata: {declared!r}"
            )
        if self._store.zarr_format == 2:
            filters = cast(
                Sequence[object],
                getattr(self.samples.metadata, "filters", ()) or (),
            )
            has_crc32c = any(
                isinstance(codec, NumcodecsCRC32C) for codec in filters
            )
        else:
            codecs = cast(
                Sequence[object],
                getattr(self.samples.metadata, "codecs", ()),
            )
            has_crc32c = type(self)._codecs_have_crc32c(codecs)
        actual: ChecksumName | None = "crc32c" if has_crc32c else None
        if declared != actual:
            raise ValueError(
                f"Recording {self.name!r} {field!r} value {declared!r} "
                f"does not match sample codec checksum {actual!r}"
            )

    def __enter__(self) -> SigMFRecording:
        """Enter a recording context manager.

        Returns:
            This recording wrapper.
        """
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Exit a recording context manager.

        Args:
            exc_type: Exception type, if an exception was raised.
            exc: Exception instance, if an exception was raised.
            tb: Traceback object, if an exception was raised.
        """
        return None

    def __len__(self) -> int:
        """Return the number of batch items or one unbatched recording.

        Returns:
            Number of logical items in the recording.
        """
        return int(self.samples.shape[0]) if self.batched else 1

    def __repr__(self) -> str:
        """Return a concise debug representation.

        Returns:
            Debug representation string.
        """
        return (
            f"{type(self).__name__}("
            f"name={self.name!r}, "
            f"num_samples={len(self)}, "
            f"sample_shape={self.sample_shape}, "
            f"sample_dtype={self.sample_dtype!r})"
        )

    def __getattr__(self, name: str) -> object:
        """Enable dynamic access to core global metadata fields.

        For example, `recording.sample_rate` maps to the
        `core:sample_rate` field in the recording's global metadata.

        Args:
            name: Attribute name to resolve.

        Returns:
            Metadata field value.

        Raises:
            SigMFAccessError: If the dynamic SigMF field is known but absent.
            AttributeError: If the name is not a supported dynamic attribute.
        """
        for key in SigMFFile.VALID_GLOBAL_KEYS:
            if key.startswith("core:") and key[5:] == name:
                field_value = self.get_global_field(key)
                if field_value is None:
                    raise SigMFAccessError(
                        f"Core field '{key}' does not exist in global metadata"
                    )
                return field_value

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: object) -> None:
        """Enable dynamic setting of core global metadata fields.

        Args:
            name: Attribute name to set.
            value: Attribute value.
        """
        if name.startswith("_") or hasattr(type(self), name):
            super().__setattr__(name, value)
            return

        for key in SigMFFile.VALID_GLOBAL_KEYS:
            if key.startswith("core:") and key[5:] == name:
                self.set_global_field(key, value)
                return

        super().__setattr__(name, value)

    @property
    def name(self) -> str:
        """Recording identifier relative to the `recordings/` group.

        Returns:
            Recording name.
        """
        return self._recording_name

    @property
    def _raw_group(self) -> Group:
        """Writable recording group used by managed implementation code.

        Returns:
            Backing Zarr group.
        """
        return cast(
            Group,
            self._store._recordings_group[self._recording_name],
        )

    @property
    def group(self) -> ReadOnlyGroup:
        """Read-only recording group under the parent store.

        Returns:
            Read-only Zarr group view for the recording.
        """
        return ReadOnlyGroup(self._raw_group)

    @property
    def _samples_array(self) -> Array:
        """Writable sample array used by managed mutation paths.

        Returns:
            Backing Zarr sample array.
        """
        return cast(Array, self._raw_group["samples"])

    @property
    def samples(self) -> ReadOnlyArray:
        """Chunked sample array for the recording.

        Returns:
            Read-only Zarr sample array view.
        """
        return ReadOnlyArray(self._samples_array)

    @property
    def extensions(self) -> ReadOnlyGroup:
        """Optional extension-specific groups for this recording.

        Returns:
            Read-only recording extensions group.
        """
        return ReadOnlyGroup(self._extensions_group)

    @property
    def _extensions_group(self) -> Group:
        """Writable extension group used by managed mutation paths.

        Returns:
            Backing extensions group.
        """
        return cast(Group, self._raw_group["extensions"])

    @property
    def _indexes_group(self) -> Group:
        """Writable index group used by managed mutation paths.

        Returns:
            Backing Zarr indexes group.
        """
        return cast(Group, self._raw_group["indexes"])

    @property
    def indexes(self) -> ReadOnlyGroup:
        """Axis-aligned metadata indexes for this recording.

        Returns:
            Read-only recording indexes group.
        """
        return ReadOnlyGroup(self._indexes_group)

    @property
    def _channels_group(self) -> Group:
        """Writable channel group used by managed implementation code.

        Returns:
            Backing channels group.
        """
        return cast(Group, self._raw_group["channels"])

    @property
    def channels(self) -> ReadOnlyGroup:
        """Channel-specific metadata groups for this recording.

        Returns:
            Recording channels group. It is empty when no explicit channel
            axis is declared.
        """
        return ReadOnlyGroup(self._channels_group)

    @property
    def sample_shape(self) -> tuple[int, ...]:
        """Per-sample tensor shape.

        This is the stored logical sample shape. When the recording is
        batched, it excludes the leading batch axis.

        Returns:
            Per-sample tensor shape.
        """
        return self._stored_sample_shape()

    @property
    def sample_axes(self) -> tuple[str, ...]:
        """Axis names for one logical sample item.

        Returns:
            Sample-axis names corresponding to `sample_shape`.

        Raises:
            ValueError: If the stored axis metadata is invalid.
        """
        return type(self)._normalize_sample_axes(
            self.sample_shape,
            sample_axes=self._stored_sample_axes(),
        )

    @property
    def runtime_axes(self) -> tuple[str, ...]:
        """Axis names for the actual sample array.

        Returns:
            Runtime axis names, including leading `item` for batched
            recordings.
        """
        if self.batched:
            return ("item", *self.sample_axes)
        return self.sample_axes

    def axis_index(self, axis_name: str) -> int:
        """Return the runtime index for a named sample axis.

        Args:
            axis_name: Axis name to locate.

        Returns:
            Axis index in the actual `samples` array.

        Raises:
            ValueError: If the axis is not present.
        """
        try:
            return self.runtime_axes.index(axis_name)
        except ValueError as exc:
            raise ValueError(
                f"Recording {self.name!r} has no axis {axis_name!r}"
            ) from exc

    @property
    def sample_count(self) -> int:
        """Number of samples along the SigMF time axis.

        Returns:
            Size of the `time` axis in the actual sample array.
        """
        return int(self.samples.shape[self.axis_index("time")])

    @property
    def num_channels(self) -> int:
        """Number of simultaneous sample channels.

        Returns:
            Explicit channel-axis length, or one when no channel axis is
            declared.
        """
        if "channel" not in self.sample_axes:
            return 1
        return int(self.samples.shape[self.axis_index("channel")])

    @property
    def batched(self) -> bool:
        """Whether the stored sample array includes a batch axis.

        Returns:
            True when the sample array has a leading batch axis.

        Raises:
            ValueError: If the sample array rank is inconsistent with
                `sigmf-zarr:sample-shape`.
        """
        stored_sample_shape = self._stored_sample_shape()
        sample_ndim = len(self.samples.shape)
        sample_shape_ndim = len(stored_sample_shape)

        if sample_ndim == sample_shape_ndim:
            return False
        if sample_ndim == sample_shape_ndim + 1:
            return True
        raise ValueError(
            f"Recording {self.name!r} sample array rank {sample_ndim} "
            "is inconsistent with stored "
            f"{self.ZARR_EXTENSION_NAMESPACE}:sample-shape "
            f"{stored_sample_shape!r}"
        )

    @property
    def sample_dtype(self) -> str:
        """Element dtype of the sample array.

        Returns:
            Sample array dtype string.
        """
        return str(self.samples.dtype)

    @property
    def zarr_format(self) -> ZarrFormat:
        """Physical Zarr format used by the parent store.

        Returns:
            Zarr format number, either 2 or 3.
        """
        return self._store.zarr_format

    @property
    def sample_checksum(self) -> ChecksumName | None:
        """Chunk-level checksum protecting the sample array.

        Returns:
            Checksum name, or `None` when checksums are disabled.
        """
        return cast(
            ChecksumName | None,
            self.global_metadata.get(
                f"{self.ZARR_EXTENSION_NAMESPACE}:checksum"
            ),
        )

    @property
    def integrity(self) -> JSONObject:
        """SigMF-Zarr-native integrity metadata for this recording.

        Returns:
            Integrity object containing any current internal hashes.
        """
        value = self._raw_group.attrs.get(INTEGRITY_ATTR, {})
        return cast(JSONObject, value) if isinstance(value, dict) else {}

    @property
    def sample_sha512(self) -> str | None:
        """Internal hash of the logical sample array.

        Returns:
            Declared sample SHA-512 digest, or ``None`` when absent.
        """
        value = self.integrity.get(self._INTEGRITY_SAMPLE_FIELD)
        return value if isinstance(value, str) else None

    @property
    def metadata_sha512(self) -> str | None:
        """Internal hash of the recording's logical metadata.

        Returns:
            Declared metadata SHA-512 digest, or ``None`` when absent.
        """
        value = self.integrity.get(self._INTEGRITY_METADATA_FIELD)
        return value if isinstance(value, str) else None


    def _invalidate_integrity(
        self,
        *,
        metadata: bool = False,
        samples: bool = False,
    ) -> None:
        """Remove internal hashes affected by a managed mutation.

        Args:
            metadata: Whether logical metadata changed.
            samples: Whether logical sample values changed.
        """
        if metadata:
            invalidate_store = getattr(
                self._store,
                "_invalidate_metadata_integrity",
                None,
            )
            if callable(invalidate_store):
                invalidate_store()
        group = getattr(self, "group", None)
        if group is None:
            return
        integrity = dict(self.integrity)
        if metadata:
            integrity.pop(self._INTEGRITY_METADATA_FIELD, None)
        if samples:
            integrity.pop(self._INTEGRITY_SAMPLE_FIELD, None)
        hash_fields = {
            self._INTEGRITY_METADATA_FIELD,
            self._INTEGRITY_SAMPLE_FIELD,
        }
        if not hash_fields.intersection(integrity):
            if INTEGRITY_ATTR in self._raw_group.attrs:
                del self._raw_group.attrs[INTEGRITY_ATTR]
            return
        self._raw_group.attrs[INTEGRITY_ATTR] = integrity


    @property
    def sha512(self) -> str | None:
        """Whole-recording standard SigMF dataset digest.

        Returns:
            Declared ``core:sha512`` value, or ``None`` when absent or not a
            string.
        """
        value = self.global_metadata.get(SHA512_KEY)
        return value if isinstance(value, str) else None

    @property
    def has_item_metadata(self) -> bool:
        """Whether this batched recording has per-item metadata.

        Returns:
            True when the optional `item_metadata` array is present.
        """
        return "item_metadata" in self._raw_group.array_keys()

    @property
    def _item_metadata_array(self) -> Array:
        """Writable item metadata array used by managed mutation paths.

        Returns:
            Backing item metadata array.

        Raises:
            KeyError: If no per-item metadata array is present.
        """
        if not self.has_item_metadata:
            raise KeyError("item_metadata")
        return cast(Array, self._raw_group["item_metadata"])


    @property
    def global_metadata(self) -> JSONObject:
        """SigMF-like global metadata object for the recording.

        Returns:
            Recording global metadata.
        """
        return cast(JSONObject, self._raw_group.attrs.get("global", {}))

    def get_global_field(
        self, key: str, default: JSONValue | None = None
    ) -> JSONValue | None:
        """Return one field from the recording's global metadata.

        Args:
            key: Metadata field key.
            default: Value returned when the key is absent.

        Returns:
            Metadata field value or `default`.
        """
        return self.global_metadata.get(key, default)

    def set_global_field(self, key: str, value: object) -> None:
        """Set one field in the recording's global metadata.

        Args:
            key: Metadata field key.
            value: JSON-compatible value to store.
        """
        if not isinstance(key, str) or not key:
            raise ValueError("Global metadata keys must be non-empty strings")
        normalized_value = json_value(value, name=f"global metadata {key!r}")
        self._invalidate_integrity(metadata=True)
        metadata = dict(self.global_metadata)
        if key in self._SHA512_ENCODING_FIELDS:
            metadata.pop(SHA512_KEY, None)
        metadata[key] = normalized_value
        self._raw_group.attrs["global"] = metadata

    def clear_sha512(self) -> None:
        """Remove ``core:sha512`` after unmanaged sample data changes."""
        metadata = dict(self.global_metadata)
        if metadata.pop(SHA512_KEY, None) is not None:
            self._invalidate_integrity(metadata=True)
            self._raw_group.attrs["global"] = metadata


    @property
    def captures(self) -> list[JSONObject]:
        """SigMF-like captures array for the recording.

        Returns:
            Capture metadata list.
        """
        return cast(
            list[JSONObject], self._raw_group.attrs.get("captures", [])
        )

    @property
    def annotations(self) -> list[JSONObject]:
        """SigMF-like annotations array for the recording.

        Returns:
            Annotation metadata list.
        """
        return cast(
            list[JSONObject], self._raw_group.attrs.get("annotations", [])
        )

    def metadata(self) -> dict[str, Any]:
        """Return all recording attributes as a plain dictionary.

        Returns:
            Recording attributes copied into a dictionary.
        """
        return dict(self._raw_group.attrs)

    @classmethod
    def _normalize_item_metadata_entry(
        cls,
        value: JSONObject | None,
        *,
        name: str,
    ) -> JSONObject:
        """Validate and copy one per-item metadata bundle.

        Args:
            value: Metadata bundle or `None` for no item-specific metadata.
            name: Human-readable value name for errors.

        Returns:
            Normalized metadata bundle.

        Raises:
            ValueError: If the bundle has unknown keys, malformed SigMF
                sections, storage-level overrides, or non-JSON values.
        """
        if value is None:
            return {}
        entry = json_object(value, name=name)
        allowed_keys = {"global", "captures", "annotations"}
        unknown_keys = set(entry) - allowed_keys
        if unknown_keys:
            raise ValueError(
                f"{name} has unsupported keys {sorted(unknown_keys)!r}"
            )

        normalized: JSONObject = {}
        if "global" in entry:
            item_global = json_object(
                entry["global"],
                name=f"{name} global metadata",
            )
            forbidden = [
                key
                for key in item_global
                if key == "core:extensions"
                or key.startswith(f"{cls.ZARR_EXTENSION_NAMESPACE}:")
            ]
            if forbidden:
                raise ValueError(
                    f"{name} global metadata cannot override storage "
                    f"fields {sorted(forbidden)!r}"
                )
            normalized["global"] = dict(item_global)
        if "captures" in entry:
            normalized["captures"] = cast(
                JSONValue,
                json_object_list(
                    entry["captures"],
                    name=f"{name} captures",
                ),
            )
        if "annotations" in entry:
            normalized["annotations"] = cast(
                JSONValue,
                json_object_list(
                    entry["annotations"],
                    name=f"{name} annotations",
                ),
            )
        return normalized


    @classmethod
    def _decode_item_metadata_entry(
        cls,
        raw: object,
        *,
        item_index: int,
    ) -> JSONObject:
        """Decode and validate one stored per-item metadata bundle.

        Args:
            raw: Encoded JSON value read from the Zarr array.
            item_index: Item-axis index used in validation errors.

        Returns:
            Validated item metadata bundle.

        Raises:
            ValueError: If the stored value is not a valid metadata bundle.
        """
        name = f"stored item_metadata[{item_index}]"
        try:
            decoded = json.loads(str(raw))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} is not valid JSON") from exc
        return cls._normalize_item_metadata_entry(
            json_object(decoded, name=name),
            name=name,
        )


    def get_item_metadata(self, item_index: int) -> JSONObject:
        """Return one item-specific metadata bundle.

        Args:
            item_index: Item-axis index. Negative indexes are supported.

        Returns:
            Decoded item-specific metadata. Returns an empty object when no
            `item_metadata` array exists.

        Raises:
            IndexError: If the item index is out of range.
            ValueError: If stored item metadata is malformed.
        """
        normalized_index = (
            item_index if item_index >= 0 else len(self) + item_index
        )
        if normalized_index < 0 or normalized_index >= len(self):
            raise IndexError(item_index)
        if not self.has_item_metadata:
            return {}
        return type(self)._decode_item_metadata_entry(
            self._item_metadata_array[normalized_index],
            item_index=normalized_index,
        )


    def channel_metadata(self, channel_index: int) -> JSONObject:
        """Return metadata for one explicit sample channel.

        Args:
            channel_index: Zero-based channel index.

        Returns:
            Channel metadata object.

        Raises:
            ValueError: If the recording has no explicit channel axis.
            IndexError: If the channel index is out of range.
        """
        if "channel" not in self.sample_axes:
            raise ValueError(
                f"Recording {self.name!r} has no explicit channel axis"
            )
        if channel_index < 0 or channel_index >= self.num_channels:
            raise IndexError(channel_index)
        channel = cast(Group, self._channels_group[str(channel_index)])
        return cast(JSONObject, dict(channel.attrs))


    def info(self) -> str:
        """Return a human-readable summary of one recording.

        Returns:
            Multiline recording summary.
        """
        sample_shape = tuple(int(dim) for dim in self.samples.shape)
        sample_chunks = tuple(int(dim) for dim in self.samples.chunks)
        index_names = sorted(
            name
            for name, member in self.indexes.members(max_depth=None)
            if isinstance(member, ReadOnlyArray)
        )
        lines = [
            f"{type(self).__name__}(",
            f"  name={self.name!r},",
            f"  zarr_format={self.zarr_format!r},",
            f"  num_samples={len(self)},",
            f"  batched={self.batched!r},",
            f"  samples_shape={sample_shape},",
            f"  samples_chunks={sample_chunks},",
            f"  sample_axes={self.sample_axes!r},",
            f"  sample_dtype={self.sample_dtype!r},",
            f"  sample_checksum={self.sample_checksum!r},",
            f"  sha512={self.sha512!r},",
            f"  sample_sha512={self.sample_sha512!r},",
            f"  metadata_sha512={self.metadata_sha512!r},",
            f"  num_channels={self.num_channels},",
            f"  has_item_metadata={self.has_item_metadata!r},",
            f"  global_keys={sorted(self.global_metadata.keys())},",
            f"  num_captures={len(self.captures)},",
            f"  num_annotations={len(self.annotations)},",
            f"  extensions={sorted(self.extensions.group_keys())},",
            f"  indexes={index_names},",
            ")",
        ]
        return "\n".join(lines)

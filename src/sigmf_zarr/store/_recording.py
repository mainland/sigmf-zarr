"""Zarr-backed SigMF-Zarr recording implementation."""

from __future__ import annotations

import json
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import numpy as np
import numpy.typing as npt
from numcodecs import CRC32C as NumcodecsCRC32C
from numcodecs import Blosc as NumcodecsBlosc
from numcodecs import VLenUTF8 as NumcodecsVLenUTF8
from numcodecs import Zstd as NumcodecsZstd
from numcodecs.abc import Codec as NumcodecsCodec
from sigmf import SHA512_KEY
from sigmf.sigmffile import SigMFAccessError, SigMFFile
from zarr.codecs import (
    BloscCodec,
    Crc32cCodec,
    VLenUTF8Codec,
    ZstdCodec,
)
from zarr.core.array import (
    Array,
    CompressorLike,
    CompressorsLike,
    SerializerLike,
    ShardsLike,
)
from zarr.core.dtype import VariableLengthUTF8
from zarr.core.group import Group

from sigmf_zarr.indexes import (
    _read_index_selection,
    validate_categorical_values,
)
from sigmf_zarr.integrity import (
    INTEGRITY_ALGORITHM,
    INTEGRITY_ATTR,
    INTEGRITY_VERSION,
    calculate_array_sha512,
    calculate_group_metadata_sha512,
    integrity_is_supported,
)
from sigmf_zarr.json import (
    JSONObject,
    JSONValue,
    json_object,
    json_object_list,
    json_value,
)
from sigmf_zarr.readonly import ReadOnlyArray, ReadOnlyGroup
from sigmf_zarr.splits import (
    SplitIndex,
    SplitMethod,
    SplitType,
    _encode_assignments,
    _split_labels,
    _validate_assignments,
)
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
        validation: Literal["full", "structural"] = "full",
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
            validation: Whether to validate all item JSON entries (`full`) or
                only their storage descriptors (`structural`) on open.
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
                valid `sample_shape`, validation mode is unsupported, or
                stored metadata is malformed.
        """
        if validation not in {"full", "structural"}:
            raise ValueError("validation must be full or structural")
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
        self._validate_item_metadata_array(entries=validation == "full")
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

    def _validate_item_metadata_array(self, *, entries: bool = True) -> None:
        """Validate optional per-item metadata storage.

        Args:
            entries: Whether to validate every stored JSON entry as well.

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
        stored_dtype = (
            array_metadata.dtype
            if self._store.zarr_format == 2
            else array_metadata.data_type
        )
        if not isinstance(stored_dtype, VariableLengthUTF8):
            raise ValueError("item_metadata must use variable-length UTF-8")
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

        if not entries:
            return

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

    def _set_integrity_hash(self, field: str, value: str) -> None:
        """Store one internal integrity hash.

        Args:
            field: Integrity object field to set.
            value: Lowercase SHA-512 digest.
        """
        integrity = dict(self.integrity)
        integrity.update(
            {
                "algorithm": INTEGRITY_ALGORITHM,
                "version": INTEGRITY_VERSION,
                field: value,
            }
        )
        self._raw_group.attrs[INTEGRITY_ATTR] = integrity

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

    @contextmanager
    def _mutation(
        self,
        *,
        metadata: bool = False,
        samples: bool = False,
    ) -> Generator[None, None, None]:
        """Invalidate affected hashes before a managed mutation.

        Args:
            metadata: Whether logical metadata may change.
            samples: Whether logical sample values may change.

        Yields:
            Control to the mutation body after invalidation.
        """
        # Invalidate before yielding so an exception can never leave a stale
        # hash that still claims to describe partially mutated content.
        self._invalidate_integrity(metadata=metadata, samples=samples)
        if samples:
            self.clear_sha512()
        yield

    def _invalidate_axis_indexes(self, axis: str) -> None:
        """Mark every index aligned with an axis as invalid.

        Args:
            axis: Runtime axis whose length or positions changed.
        """
        self._invalidate_integrity(metadata=True)
        for _, index in self._indexes_group.members(max_depth=None):
            if not isinstance(index, Array):
                continue
            if index.attrs.get("axis") != axis:
                continue
            index.attrs[self._INDEX_VALID_FIELD] = False
            index.attrs[self._INDEX_INVALID_REASON_FIELD] = (
                f"axis {axis!r} changed"
            )

    def _validate_mutated_index(self, index_name: str, index: Array) -> None:
        """Validate and mark one successfully mutated index as current.

        Args:
            index_name: Recording-level index name.
            index: Mutated backing Zarr array.

        Raises:
            ValueError: If the array is not one-dimensional, its axis is
                invalid, or its length does not match the indexed axis.
        """
        error = self._index_structure_error(index_name, index)
        if error is not None:
            raise ValueError(error)
        if self._INDEX_VALID_FIELD in index.attrs:
            del index.attrs[self._INDEX_VALID_FIELD]
        if self._INDEX_INVALID_REASON_FIELD in index.attrs:
            del index.attrs[self._INDEX_INVALID_REASON_FIELD]

    def _index_structure_error(
        self,
        index_name: str,
        index: Array,
    ) -> str | None:
        """Return a structural error for one recording index.

        Args:
            index_name: Recording-level index name.
            index: Backing Zarr index array.

        Returns:
            Error message, or ``None`` when the index is structurally valid.
        """
        if index.ndim != 1:
            return (
                f"Index {index_name!r} must be one-dimensional, got "
                f"shape {index.shape}"
            )
        axis = index.attrs.get("axis")
        if not isinstance(axis, str) or axis not in self.runtime_axes:
            return f"Index {index_name!r} has invalid axis {axis!r}"
        axis_length = int(self._samples_array.shape[self.axis_index(axis)])
        index_length = int(index.shape[0])
        if index_length != axis_length:
            return (
                f"Index {index_name!r} length {index_length} does not match "
                f"axis {axis!r} length {axis_length}"
            )
        return None

    def _invalid_index_reason(self) -> str | None:
        """Return the reason that any recording index is invalid.

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
            error = self._index_structure_error(index_name, index)
            if error is not None:
                return error
        return None

    @contextmanager
    def mutate_samples(self) -> Generator[Array, None, None]:
        """Provide writable sample access with conservative invalidation.

        The context is intended for value updates that do not resize or
        otherwise reconfigure the sample array. Use :meth:`set_samples` or
        :meth:`append_samples` for shape changes so aligned metadata can be
        maintained.

        Yields:
            Writable backing Zarr sample array.

        Raises:
            ValueError: If the mutation changes the sample array shape.
        """
        samples = self._samples_array
        original_shape = tuple(int(size) for size in samples.shape)
        try:
            with self._mutation(metadata=True, samples=True):
                yield samples
        finally:
            # Shape changes must invalidate aligned indexes even when the
            # caller exits the context by raising an exception.
            current_shape = tuple(int(size) for size in samples.shape)
            if current_shape != original_shape:
                for axis, old_size, new_size in zip(
                    self.runtime_axes,
                    original_shape,
                    current_shape,
                    strict=False,
                ):
                    if old_size != new_size:
                        self._invalidate_axis_indexes(axis)
        if current_shape != original_shape:
            raise ValueError(
                "mutate_samples() cannot resize the sample array. Use "
                "set_samples() or append_samples()"
            )

    @contextmanager
    def mutate_extensions(self) -> Generator[Group, None, None]:
        """Provide writable access to extension-owned structured data.

        Internal recording and store metadata hashes are invalidated before
        access is granted. Extension code remains responsible for preserving
        the schema of its own arrays and groups.

        Yields:
            Writable backing extensions group.
        """
        with self._mutation(metadata=True):
            yield self._extensions_group

    @contextmanager
    def mutate_index(
        self,
        index_name: str,
    ) -> Generator[Array, None, None]:
        """Provide writable access to one recording-level index.

        The index is marked invalid before writable access is granted. A
        successful context exit validates its shape and alignment and marks it
        current again. If the body raises, the index remains invalid.

        Args:
            index_name: Recording-level index name.

        Yields:
            Writable backing Zarr index array.

        Raises:
            KeyError: If the index does not exist.
            ValueError: If the mutated index is structurally invalid.
        """
        if index_name not in self._indexes_group:
            raise KeyError(index_name)
        index = cast(Array, self._indexes_group[index_name])
        self._invalidate_integrity(metadata=True)
        index.attrs[self._INDEX_VALID_FIELD] = False
        index.attrs[self._INDEX_INVALID_REASON_FIELD] = (
            "mutation did not complete"
        )
        yield index
        # Reaching this line means the caller exited normally. Validation
        # removes the fail-safe invalid marker only after alignment is proven.
        self._validate_mutated_index(index_name, index)

    def calculate_sample_sha512(self) -> str:
        """Calculate the internal hash of the logical sample array.

        Returns:
            Lowercase SHA-512 hexadecimal digest.
        """
        return calculate_array_sha512(self._samples_array)

    def calculate_metadata_sha512(self) -> str:
        """Calculate the internal hash of logical recording metadata.

        The sample array descriptor is included, but its values are covered by
        :meth:`calculate_sample_sha512` and omitted here.

        Returns:
            Lowercase SHA-512 hexadecimal digest.
        """
        # Sample values have a separate digest. Including them here would make
        # every sample mutation require two full-array hashes.
        return calculate_group_metadata_sha512(
            self._raw_group,
            exclude_array_values=frozenset({"samples"}),
        )

    def update_integrity(self) -> JSONObject:
        """Calculate and store current internal sample and metadata hashes.

        Returns:
            Updated integrity metadata object.
        """
        invalid_index = self._invalid_index_reason()
        if invalid_index is not None:
            raise ValueError(
                f"Cannot update recording integrity: {invalid_index}"
            )
        sample_hash = self.calculate_sample_sha512()
        self._set_integrity_hash(self._INTEGRITY_SAMPLE_FIELD, sample_hash)
        metadata_hash = self.calculate_metadata_sha512()
        self._set_integrity_hash(
            self._INTEGRITY_METADATA_FIELD,
            metadata_hash,
        )
        return self.integrity

    def verify_integrity(self) -> bool:
        """Verify both internal recording hashes.

        Returns:
            True only when both hashes are present and match current content.
        """
        return (
            integrity_is_supported(self.integrity)
            and self._invalid_index_reason() is None
            and self.sample_sha512 is not None
            and self.metadata_sha512 is not None
            and self.sample_sha512 == self.calculate_sample_sha512()
            and self.metadata_sha512 == self.calculate_metadata_sha512()
        )

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
    def item_metadata_array(self) -> ReadOnlyArray:
        """Encoded per-item metadata array.

        Returns:
            One-dimensional variable-length UTF-8 JSON string array.

        Raises:
            KeyError: If no per-item metadata array is present.
        """
        return ReadOnlyArray(self._item_metadata_array)

    @contextmanager
    def mutate_item_metadata(self) -> Generator[Array, None, None]:
        """Provide validated writable access to per-item metadata.

        Yields:
            Writable backing item metadata array.

        Raises:
            KeyError: If the recording has no item metadata array.
            ValueError: If the resulting array is malformed or misaligned.
        """
        with self._mutation(metadata=True):
            yield self._item_metadata_array
        # Validation runs only after a normal context exit. If the caller
        # raises, the original exception propagates and integrity stays absent.
        self._validate_item_metadata_array()

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

    def calculate_sha512(self) -> str:
        """Calculate SHA-512 over the standard SigMF dataset byte stream.

        Returns:
            Lowercase SHA-512 hexadecimal digest.

        Raises:
            ValueError: If the recording cannot be encoded as standard SigMF.
        """
        from sigmf_zarr.sigmf import calculate_sha512

        return calculate_sha512(self)

    def update_sha512(self) -> str:
        """Calculate and store ``core:sha512`` for the current samples.

        Returns:
            Stored lowercase SHA-512 hexadecimal digest.

        Raises:
            ValueError: If the recording cannot be encoded as standard SigMF.
        """
        value = self.calculate_sha512()
        self.set_global_field(SHA512_KEY, value)
        return value

    def verify_sha512(self) -> bool:
        """Verify the current samples against declared ``core:sha512``.

        Returns:
            True only when a valid declared digest is present and matches.

        Raises:
            ValueError: If the recording cannot be encoded as standard SigMF.
        """
        from sigmf_zarr.sigmf import verify_sha512

        return verify_sha512(self)

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
    def _encode_item_metadata(
        cls,
        values: Sequence[JSONObject | None],
    ) -> list[str]:
        """Encode per-item metadata bundles as canonical JSON strings.

        Args:
            values: Item metadata bundles.

        Returns:
            Compact JSON object strings in item order.
        """
        return [
            cls._encode_item_metadata_entry(
                value,
                item_index=index,
            )
            for index, value in enumerate(values)
        ]

    @classmethod
    def _encode_item_metadata_entry(
        cls,
        value: JSONObject | None,
        *,
        item_index: int,
    ) -> str:
        """Validate and encode one per-item metadata bundle.

        Args:
            value: Metadata bundle or `None` for an empty bundle.
            item_index: Item-axis index used in validation errors.

        Returns:
            Canonical compact JSON object string.
        """
        # Canonical JSON prevents insignificant formatting differences from
        # changing logical metadata hashes.
        return json.dumps(
            cls._normalize_item_metadata_entry(
                value,
                name=f"item_metadata[{item_index}]",
            ),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

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

    def _create_item_metadata_array(
        self,
        encoded: npt.NDArray[np.object_],
        *,
        overwrite: bool,
    ) -> Array:
        """Create the checksummed per-item metadata array.

        Args:
            encoded: Canonical JSON strings in item order.
            overwrite: Whether an existing array may be replaced.

        Returns:
            Writable backing item metadata array.

        Raises:
            ValueError: If the array already exists and overwrite is false.
        """
        self._invalidate_integrity(metadata=True)
        group = self._raw_group
        if "item_metadata" in group:
            if not overwrite:
                raise ValueError(
                    "Array 'item_metadata' already exists. Pass "
                    "overwrite=True to replace it"
                )
            del group["item_metadata"]

        chunks = self._store.default_index_chunks(len(encoded))
        # Both physical formats store the same logical UTF-8 JSON strings, but
        # their serializer and codec APIs are intentionally different.
        if self._store.zarr_format == 2:
            array = group.create_array(
                "item_metadata",
                shape=encoded.shape,
                dtype=cast(Any, VariableLengthUTF8()),
                chunks=chunks,
                compressor=NumcodecsZstd(),
                filters=[NumcodecsVLenUTF8(), NumcodecsCRC32C()],
            )
        else:
            array = group.create_array(
                "item_metadata",
                shape=encoded.shape,
                dtype=cast(Any, VariableLengthUTF8()),
                chunks=chunks,
                serializer=VLenUTF8Codec(),
                compressors=(ZstdCodec(), Crc32cCodec()),
            )
        array[:] = encoded
        return array

    def set_item_metadata(
        self,
        values: Sequence[JSONObject | None],
        *,
        overwrite: bool = False,
    ) -> ReadOnlyArray:
        """Create or replace metadata bundles aligned with the item axis.

        Args:
            values: One metadata bundle per batch item. `None` represents no
                item-specific overrides.
            overwrite: Whether to replace an existing metadata array.

        Returns:
            Encoded `item_metadata` Zarr array.

        Raises:
            ValueError: If the recording is unbatched, the value count does
                not match the item count, or a bundle is invalid.
        """
        if not self.batched:
            raise ValueError(
                "item_metadata is only supported for batched recordings"
            )
        if len(values) != len(self):
            raise ValueError(
                f"Expected metadata for {len(self)} items, got {len(values)}"
            )
        encoded = np.asarray(
            type(self)._encode_item_metadata(values),
            dtype=object,
        )
        array = self._create_item_metadata_array(
            encoded,
            overwrite=overwrite,
        )
        return ReadOnlyArray(array)

    def set_item_metadata_entry(
        self,
        item_index: int,
        value: JSONObject | None,
    ) -> None:
        """Set one validated per-item metadata bundle.

        An absent metadata array is created with empty bundles for every
        other item.

        Args:
            item_index: Item-axis index. Negative indexes are supported.
            value: Metadata bundle or `None` for an empty bundle.

        Raises:
            IndexError: If the item index is out of range.
            ValueError: If the recording is unbatched or the bundle is
                invalid.
        """
        if not self.batched:
            raise ValueError(
                "item_metadata is only supported for batched recordings"
            )
        normalized_index = (
            item_index if item_index >= 0 else len(self) + item_index
        )
        if normalized_index < 0 or normalized_index >= len(self):
            raise IndexError(item_index)
        encoded = type(self)._encode_item_metadata_entry(
            value,
            item_index=normalized_index,
        )
        if not self.has_item_metadata:
            entries = np.full(len(self), "{}", dtype=object)
            entries[normalized_index] = encoded
            self._create_item_metadata_array(entries, overwrite=False)
            return
        self._invalidate_integrity(metadata=True)
        self._item_metadata_array[normalized_index] = encoded

    def set_item_metadata_slice(
        self,
        selection: slice,
        values: Sequence[JSONObject | None],
    ) -> None:
        """Set a contiguous slice of validated item metadata bundles.

        An absent metadata array is created with empty bundles outside the
        selected range.

        Args:
            selection: Contiguous item-axis slice with step 1.
            values: One metadata bundle for each selected item.

        Raises:
            ValueError: If the recording is unbatched, the slice is not
                contiguous, the value count is wrong, or a bundle is invalid.
        """
        if not self.batched:
            raise ValueError(
                "item_metadata is only supported for batched recordings"
            )
        start, stop, step = selection.indices(len(self))
        if step != 1:
            raise ValueError("item metadata slices must use a step of 1")
        item_indexes = range(start, stop)
        if len(values) != len(item_indexes):
            raise ValueError(
                f"Expected metadata for {len(item_indexes)} selected items, "
                f"got {len(values)}"
            )
        encoded = np.asarray(
            [
                type(self)._encode_item_metadata_entry(
                    value,
                    item_index=item_index,
                )
                for item_index, value in zip(
                    item_indexes,
                    values,
                    strict=True,
                )
            ],
            dtype=object,
        )
        if not len(encoded):
            return
        normalized_selection = slice(start, stop)
        if not self.has_item_metadata:
            entries = np.full(len(self), "{}", dtype=object)
            entries[normalized_selection] = encoded
            self._create_item_metadata_array(entries, overwrite=False)
            return
        self._invalidate_integrity(metadata=True)
        self._item_metadata_array[normalized_selection] = encoded

    def clear_item_metadata(self) -> None:
        """Remove optional per-item metadata from this recording."""
        self._invalidate_integrity(metadata=True)
        if self.has_item_metadata:
            del self._raw_group["item_metadata"]

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

    def resolved_item_metadata(self, item_index: int) -> JSONObject:
        """Resolve shared recording metadata for one batch item.

        Item `global` fields shallowly override shared global fields. Item
        captures and annotations are appended to their shared counterparts.

        Args:
            item_index: Item-axis index.

        Returns:
            Object containing resolved `global`, `captures`, and `annotations`.
        """
        return self.resolved_item_metadata_batch([item_index])[0]

    def resolved_item_metadata_batch(
        self, item_indices: Sequence[int]
    ) -> list[JSONObject]:
        """Resolve selected item JSON with one batched array selection.

        Args:
            item_indices: Original item positions. Order and duplicates are
                preserved. Negative positions count from the end.

        Returns:
            Independent resolved metadata objects. Shared global fields are
            shallowly overridden by item fields. Captures and annotations are
            appended. Indexes are not consulted. Memory scales with the
            requested entries, so callers should use bounded batches.

        Raises:
            IndexError: If a position is out of range.
            TypeError: If a position is not an integer or is Boolean.
            ValueError: If selected item JSON is malformed.
        """
        length = len(self)
        positions = []
        for item in item_indices:
            if isinstance(item, bool | np.bool_) or not isinstance(
                item, int | np.integer
            ):
                raise TypeError("Item positions must be non-Boolean integers")
            if not -length <= item < length:
                raise IndexError(item)
            positions.append(int(item) % length)
        if not positions:
            return []
        # One orthogonal read preserves requested order and repeats without a
        # backend round trip per item. Missing local metadata acts as an empty
        # override, so shared metadata still resolves for every selected item.
        encoded = (
            _read_index_selection(self.item_metadata_array, positions)
            if self.has_item_metadata else ["{}"] * len(positions)
        )
        shared = self.metadata()
        result: list[JSONObject] = []
        for item, raw in zip(positions, encoded, strict=True):
            entry = type(self)._decode_item_metadata_entry(
                raw, item_index=item
            )
            metadata: JSONObject = {
                "global": {
                    **shared.get("global", {}),
                    **cast(JSONObject, entry.get("global", {})),
                },
                "captures": [
                    *shared.get("captures", []),
                    *cast(list[JSONObject], entry.get("captures", [])),
                ],
                "annotations": [
                    *shared.get("annotations", []),
                    *cast(list[JSONObject], entry.get("annotations", [])),
                ],
            }
            # The merges above still share nested objects. Detach each result
            # so editing one item cannot alter another, including duplicates.
            result.append(deepcopy(metadata))
        return result

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

    def set_channel_metadata(
        self,
        channel_index: int,
        metadata: JSONObject,
    ) -> None:
        """Replace metadata for one explicit sample channel.

        Args:
            channel_index: Zero-based channel index.
            metadata: JSON-compatible channel metadata object.

        Raises:
            ValueError: If the recording has no channel axis or metadata is
                not JSON serializable.
            IndexError: If the channel index is out of range.
        """
        _ = self.channel_metadata(channel_index)
        copied = json_object(
            dict(metadata),
            name=f"channel {channel_index} metadata",
        )
        self._invalidate_integrity(metadata=True)
        channel = cast(Group, self._channels_group[str(channel_index)])
        for key in tuple(channel.attrs):
            del channel.attrs[key]
        channel.attrs.update(copied)

    def _append_capture(
        self,
        capture: JSONObject | None,
        *,
        sample_start: int,
    ) -> None:
        """Append one optional capture metadata entry.

        Args:
            capture: Optional capture metadata to append.
            sample_start: Default `core:sample_start` value.
        """
        if capture is None:
            return
        self._invalidate_integrity(metadata=True)
        entry = dict(capture)
        entry.setdefault("core:sample_start", sample_start)
        self._raw_group.attrs["captures"] = list(self.captures) + [entry]

    def _append_batched_samples(
        self,
        sample_array: npt.NDArray[Any],
        *,
        capture: JSONObject | None,
        item_metadata: Sequence[JSONObject | None] | None,
    ) -> None:
        """Append one batch of samples to a batched recording.

        Args:
            sample_array: Batch of samples with shape `(N, *sample_shape)`.
            capture: Optional capture metadata for the appended batch.
            item_metadata: Optional metadata bundle for each appended item.

        Raises:
            ValueError: If rank differs from the sample array rank or the
                per-item sample shape does not match.
        """
        if sample_array.ndim != self.samples.ndim:
            raise ValueError(
                f"Expected samples with ndim {self.samples.ndim}, got "
                f"{sample_array.ndim}"
            )
        tail_shape = tuple(int(dim) for dim in sample_array.shape[1:])
        if tail_shape != self.sample_shape:
            raise ValueError(
                f"Expected sample shape {self.sample_shape}, got {tail_shape}"
            )

        append_count = int(sample_array.shape[0])
        if item_metadata is not None and len(item_metadata) != append_count:
            raise ValueError(
                f"Expected metadata for {append_count} appended items, got "
                f"{len(item_metadata)}"
            )
        encoded_item_metadata = (
            None
            if item_metadata is None
            else type(self)._encode_item_metadata(item_metadata)
        )

        start = len(self)
        stop = start + append_count
        if append_count:
            # Existing item indexes no longer align after the axis grows.
            self._invalidate_axis_indexes("item")
        with self._mutation(metadata=True, samples=True):
            samples = self._samples_array
            samples.resize((stop, *self.sample_shape))
            samples[start:stop] = sample_array

        if self.has_item_metadata:
            values = encoded_item_metadata or ["{}"] * append_count
            self._item_metadata_array.resize((stop,))
            self._item_metadata_array[start:stop] = np.asarray(
                values,
                dtype=object,
            )
        elif encoded_item_metadata is not None:
            self._create_item_metadata_array(
                np.asarray(
                    ["{}"] * start + encoded_item_metadata, dtype=object
                ),
                overwrite=False,
            )
        self._append_capture(capture, sample_start=start)

    def _append_unbatched_samples(
        self,
        sample_array: npt.NDArray[Any],
        *,
        capture: JSONObject | None,
    ) -> None:
        """Append samples along the explicit time axis.

        Args:
            sample_array: Samples to append along the `time` axis.
            capture: Optional capture metadata for the appended time segment.

        Raises:
            ValueError: If rank differs from the sample array rank or any
                non-time axis changes size.
        """
        if sample_array.ndim != self.samples.ndim:
            raise ValueError(
                f"Expected samples with ndim {self.samples.ndim}, got "
                f"{sample_array.ndim}"
            )

        for axis_name in self.sample_axes:
            if axis_name == "time":
                continue
            axis = self.axis_index(axis_name)
            if sample_array.shape[axis] != self.samples.shape[axis]:
                raise ValueError(
                    f"Expected {axis_name!r} axis length "
                    f"{self.samples.shape[axis]}, got "
                    f"{sample_array.shape[axis]}"
                )

        time_axis = self.axis_index("time")
        start = self.sample_count
        stop = start + int(sample_array.shape[time_axis])
        shape = list(int(dim) for dim in self._samples_array.shape)
        shape[time_axis] = stop
        if stop != start:
            # A longer time axis invalidates every dense time-aligned index.
            self._invalidate_axis_indexes("time")
        with self._mutation(metadata=True, samples=True):
            samples = self._samples_array
            samples.resize(tuple(shape))

            key: list[slice] = [slice(None)] * samples.ndim
            key[time_axis] = slice(start, stop)
            samples[tuple(key)] = sample_array

        metadata = dict(self.global_metadata)
        metadata[f"{self.ZARR_EXTENSION_NAMESPACE}:sample-shape"] = [
            int(dim) for dim in shape
        ]
        self._raw_group.attrs["global"] = metadata
        self._append_capture(capture, sample_start=start)

    def append_samples(
        self,
        samples: npt.ArrayLike,
        *,
        capture: JSONObject | None = None,
        item_metadata: Sequence[JSONObject | None] | None = None,
    ) -> None:
        """Append samples to a recording.

        Batched recordings append along the leading `item` axis. Unbatched
        recordings append along the explicit `time` axis.

        Args:
            samples: Samples to append.
            capture: Optional capture metadata for the appended samples. When
                provided, `core:sample_start` defaults to the old item index
                for batched recordings or the old time sample count for
                unbatched recordings.
            item_metadata: Optional metadata bundle for each appended item.
                Only valid for batched recordings.

        Raises:
            ValueError: If rank differs from the sample array rank or any
                fixed axis changes size, the samples cannot be converted to
                the stored dtype, or metadata is invalid.
        """
        # Conversion and metadata validation must finish before any shape,
        # index validity, or integrity state changes.
        sample_array = np.asarray(samples, dtype=self._samples_array.dtype)
        if capture is not None:
            capture = json_object(capture, name="capture metadata")
        if self.batched:
            self._append_batched_samples(
                sample_array,
                capture=capture,
                item_metadata=item_metadata,
            )
            return
        if item_metadata is not None:
            raise ValueError(
                "item_metadata is only supported for batched recordings"
            )
        self._append_unbatched_samples(sample_array, capture=capture)

    def set_samples(self, samples: npt.ArrayLike) -> None:
        """Set the full sample array for an unbatched recording.

        Args:
            samples: Full sample array to write.

        Raises:
            ValueError: If the recording is batched, rank differs from the
                sample array rank, any non-time axis changes size, the time
                axis is empty, or conversion to the stored dtype fails.
        """
        if self.batched:
            raise ValueError(
                "set_samples is only supported for unbatched recordings"
            )
        sample_array = np.asarray(samples)
        if sample_array.ndim != self.samples.ndim:
            raise ValueError(
                f"Expected samples with ndim {self.samples.ndim}, got "
                f"{sample_array.ndim}"
            )

        for axis_name in self.sample_axes:
            if axis_name == "time":
                continue
            axis = self.axis_index(axis_name)
            if sample_array.shape[axis] != self.samples.shape[axis]:
                raise ValueError(
                    f"Expected {axis_name!r} axis length "
                    f"{self.samples.shape[axis]}, got "
                    f"{sample_array.shape[axis]}"
                )

        sample_shape = tuple(int(dim) for dim in sample_array.shape)
        old_time_length = int(
            self._samples_array.shape[self.axis_index("time")]
        )
        new_time_length = int(sample_shape[self.axis_index("time")])
        if new_time_length == 0:
            raise ValueError("The time axis must contain at least one sample")
        sample_array = np.asarray(
            sample_array, dtype=self._samples_array.dtype
        )
        if new_time_length != old_time_length:
            self._invalidate_axis_indexes("time")
        with self._mutation(metadata=True, samples=True):
            samples_array = self._samples_array
            samples_array.resize(sample_shape)
            samples_array[:] = sample_array

        metadata = dict(self.global_metadata)
        metadata[f"{self.ZARR_EXTENSION_NAMESPACE}:sample-shape"] = [
            int(dim) for dim in sample_shape
        ]
        self._raw_group.attrs["global"] = metadata

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

    def _create_array(
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
    ) -> Array:
        """Create one managed array, optionally at a nested path.

        Args:
            group: Parent group in which to create the array.
            name: Array name or nested relative path under `group`.
            data: Array-like values used to initialize the new array.
            overwrite: Whether an existing array may be replaced.
            chunks: Optional chunk shape.
            shards: Optional shard shape passed through to Zarr.
            compressors: Optional compressors configuration.
            compressor: Optional single-compressor configuration.
            serializer: Optional serializer configuration.

        Returns:
            Created Zarr array.

        Raises:
            ValueError: If the target exists and `overwrite` is false.
        """
        self._invalidate_integrity(metadata=True)
        return self._store.create_array(
            group,
            name,
            data,
            overwrite=overwrite,
            chunks=chunks,
            shards=shards,
            compressors=compressors,
            compressor=compressor,
            serializer=serializer,
        )

    def add_extension_array(
        self,
        name: str,
        data: npt.ArrayLike,
        *,
        overwrite: bool = False,
        chunks: tuple[int, ...] | None = None,
        shards: ShardsLike | None = None,
        compressors: CompressorsLike = "auto",
        compressor: CompressorLike = "auto",
        serializer: SerializerLike = "auto",
    ) -> ReadOnlyArray:
        """Create an extension-owned array with integrity invalidation.

        Args:
            name: Array name or nested relative path under `extensions/`.
            data: Array-like values used to initialize the new array.
            overwrite: Whether an existing array may be replaced.
            chunks: Optional chunk shape.
            shards: Optional shard shape.
            compressors: Optional Zarr format 3 compressor pipeline.
            compressor: Optional single compressor.
            serializer: Optional Zarr format 3 serializer.

        Returns:
            Read-only view of the created array.
        """
        array = self._create_array(
            self._extensions_group,
            name,
            data,
            overwrite=overwrite,
            chunks=chunks,
            shards=shards,
            compressors=compressors,
            compressor=compressor,
            serializer=serializer,
        )
        return ReadOnlyArray(array)

    def add_index(
        self,
        index_name: str,
        values: npt.ArrayLike,
        *,
        axis: str,
        field: str,
        kind: str = "metadata",
        unit: str | None = None,
        labels: list[JSONValue] | tuple[JSONValue, ...] | None = None,
        attributes: Mapping[str, JSONValue] | None = None,
        overwrite: bool = False,
        chunks: tuple[int] | None = None,
    ) -> ReadOnlyArray:
        """Create or replace one recording-level metadata index.

        Index arrays are dense one-dimensional metadata arrays aligned
        with one runtime sample axis such as `item`, `time`, or `channel`.

        Args:
            index_name: Index array name.
            values: One-dimensional index values.
            axis: Runtime sample axis indexed by the values.
            field: Metadata field represented by the index.
            kind: Index kind. Defaults to `metadata`.
            unit: Optional value unit.
            labels: Optional labels or lookup values for integer indexes.
            attributes: Additional descriptive JSON attributes. Must not
                override base index attributes or managed mutation markers.
            overwrite: Whether an existing index may be replaced.
            chunks: Optional chunk shape.

        Returns:
            Created index array.

        Raises:
            ValueError: If values are not one-dimensional, the axis is not
                present, the index length does not match the axis length,
                or additional attributes are invalid or reserved.
        """
        extra = json_object(dict(attributes or {}), name="index attributes")
        reserved = {
            "axis",
            "field",
            "kind",
            "unit",
            "labels",
            self._INDEX_VALID_FIELD,
            self._INDEX_INVALID_REASON_FIELD,
        }
        if conflicts := reserved.intersection(extra):
            raise ValueError(
                f"Cannot override index attributes: {sorted(conflicts)}"
            )
        value_array = np.asarray(values)
        if value_array.ndim != 1:
            raise ValueError(
                f"Index {index_name!r} must be one-dimensional, got "
                f"shape {value_array.shape}"
            )
        axis_length = int(self.samples.shape[self.axis_index(axis)])
        if len(value_array) != axis_length:
            raise ValueError(
                f"Index {index_name!r} length {len(value_array)} does not "
                f"match axis {axis!r} length {axis_length}"
            )

        index = self._create_array(
            self._indexes_group,
            index_name,
            value_array,
            overwrite=overwrite,
            chunks=(
                chunks or self._store.default_index_chunks(len(value_array))
            ),
        )
        attrs: JSONObject = {
            **extra,
            "axis": axis,
            "field": field,
            "kind": kind,
        }
        if unit is not None:
            attrs["unit"] = unit
        if labels is not None:
            attrs["labels"] = list(labels)
        index.attrs.update(attrs)
        if isinstance(index, Array):
            return ReadOnlyArray(index)
        return cast(ReadOnlyArray, index)

    def index(self, index_name: str) -> ReadOnlyArray:
        """Return one recording-level metadata index.

        Args:
            index_name: Index array name.

        Returns:
            Read-only recording-level index array.

        Raises:
            ValueError: If the index was invalidated by an axis mutation.
        """
        index = cast(Array, self._indexes_group[index_name])
        if index.attrs.get(self._INDEX_VALID_FIELD, True) is not True:
            reason = index.attrs.get(
                self._INDEX_INVALID_REASON_FIELD,
                "unknown reason",
            )
            raise ValueError(
                f"Index {index_name!r} is invalid: {reason}. Repair it "
                "with mutate_index() or replace it with add_index()"
            )
        error = self._index_structure_error(index_name, index)
        if error is not None:
            raise ValueError(error)
        return ReadOnlyArray(index)

    def find_indexes(
        self, field: str, *, axis: str | None = None
    ) -> tuple[str, ...]:
        """Find every index describing a field without reading its values.

        Args:
            field: Exact metadata field key to match.
            axis: Optional runtime axis to match.

        Returns:
            Sorted names relative to `indexes/`, including nested names.
            Results may include invalid indexes. No default is selected.
        """
        return tuple(
            sorted(
                name
                for name, member in self._indexes_group.members(max_depth=None)
                if isinstance(member, Array)
                and member.attrs.get("field") == field
                and (axis is None or member.attrs.get("axis") == axis)
            )
        )

    def decode_index(
        self,
        index_name: str,
        *,
        selection: int | slice | Sequence[int],
    ) -> npt.NDArray[np.object_]:
        """Decode selected category IDs through an index's labels table.

        Ordinary index access continues to return raw values. JSON metadata is
        neither read nor modified by this explicit decoding operation.

        Args:
            index_name: Index name relative to `indexes/`.
            selection: Integer position, slice, or integer positions to read.

        Returns:
            One-dimensional object array of JSON lookup values. Order and
            duplicates are preserved, and mutable values are independent.
            An integer selection returns an array of length one.

        Raises:
            KeyError: If the index does not exist.
            ValueError: If index data or labels are invalid, or a slice step
                is zero.
            TypeError: If selection positions are not integers or are Boolean.
            IndexError: If a selected position is outside the index.
        """
        array = self.index(index_name)
        values = _read_index_selection(array, selection)
        labels = validate_categorical_values(
            values, labels=array.attrs.get("labels")
        )
        result = np.empty(len(values), dtype=object)
        for position, category in enumerate(values):
            result[position] = deepcopy(labels[int(category)])
        return result

    def add_split(
        self,
        index_name: str,
        assignments: npt.ArrayLike,
        *,
        labels: Sequence[str],
        split_type: SplitType = "custom",
        method: SplitMethod = "custom",
        group_index: str | None = None,
        seed: int | None = None,
        description: str | None = None,
        generator: str | None = None,
        overwrite: bool = False,
        chunks: tuple[int] | None = None,
    ) -> SplitIndex:
        """Store explicit split assignments after validating group isolation.

        This operation does not generate assignments. Method and seed describe
        their provenance. Later source edits do not refresh or invalidate them.

        Args:
            index_name: Unique recording-level split index name.
            assignments: One partition ID or label string per item.
            labels: Ordered, unique, nonempty partition names. Every partition
                must contain at least one item, with at least two partitions.
            split_type: Interpretation of the partitions.
            method: Method used to produce the supplied assignments.
            group_index: Optional item-aligned integer or string group IDs.
                Each identity must occur in only one partition.
            seed: Optional nonnegative generator seed.
            description: Optional human-readable split purpose.
            generator: Optional generator name and version.
            overwrite: Whether to replace an existing named index.
            chunks: Optional assignment chunk shape.

        Returns:
            Read-only view of the newly created split.

        Raises:
            KeyError: If the grouping index is absent.
            ValueError: If descriptors, assignments, or group isolation are
                invalid, or the index exists without overwrite permission.
        """
        if isinstance(labels, str | bytes):
            raise ValueError("Split labels must be a sequence of names")
        if group_index == index_name:
            raise ValueError("A split cannot use itself as its grouping index")
        attributes: JSONObject = {"split_type": split_type, "method": method}
        for key, value in (
            ("group_index", group_index),
            ("seed", seed),
            ("description", description),
            ("generator", generator),
        ):
            if value is not None:
                attributes[key] = value
        descriptors = json_object(
            {
                **attributes,
                "axis": "item",
                "field": "sigmf-zarr:split",
                "kind": "split",
                "labels": list(labels),
            },
            name="split descriptors",
        )
        vocabulary = _split_labels(descriptors)
        values = _encode_assignments(assignments, vocabulary)
        _validate_assignments(self, values, descriptors)
        self.add_index(
            index_name,
            values,
            axis="item",
            field="sigmf-zarr:split",
            kind="split",
            labels=list(vocabulary),
            attributes=attributes,
            overwrite=overwrite,
            chunks=chunks,
        )
        return self.split(index_name)

    def split(self, index_name: str) -> SplitIndex:
        """Open one explicitly named split without checking source groups.

        Args:
            index_name: Recording-relative index name.

        Returns:
            Read-only split view with assignments, labels, and provenance.

        Raises:
            KeyError: If the index is absent.
            ValueError: If index structure or split descriptors are invalid.
        """
        return SplitIndex(self, index_name)

    def validate_split(self, index_name: str) -> None:
        """Check a split against its declared source groups without mutation.

        Args:
            index_name: Recording-relative split index name.

        Raises:
            KeyError: If the split or grouping index is absent.
            ValueError: If assignments, descriptors, or group isolation fail.
        """
        self.split(index_name).validate()

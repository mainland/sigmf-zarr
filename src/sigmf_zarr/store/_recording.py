"""Zarr-backed SigMF-Zarr recording implementation."""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import numpy as np
import numpy.typing as npt
from numcodecs import CRC32C as NumcodecsCRC32C
from numcodecs import Blosc as NumcodecsBlosc
from numcodecs import GZip as NumcodecsGZip
from numcodecs import VLenUTF8 as NumcodecsVLenUTF8
from numcodecs import Zstd as NumcodecsZstd
from numcodecs.abc import Codec as NumcodecsCodec
from sigmf import SHA512_KEY
from sigmf.sigmffile import SigMFAccessError, SigMFFile
from zarr.abc.codec import BytesBytesCodec
from zarr.codecs import (
    BloscCodec,
    Crc32cCodec,
    GzipCodec,
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

from sigmf_zarr._consolidated import members, without_consolidated_metadata
from sigmf_zarr._dtype import (
    is_logical_sample_dtype,
    is_numeric_sample_dtype,
    logical_sample_dtype,
)
from sigmf_zarr.integrity import (
    INTEGRITY_ALGORITHM,
    INTEGRITY_ATTR,
    INTEGRITY_VERSION,
    METADATA_DOMAIN,
    calculate_array_sha512,
    calculate_group_metadata_sha512,
    calculate_payload_sha512,
    group_metadata_payload,
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
from sigmf_zarr.store._common import (
    DEFAULT_SAMPLE_CHUNK_BYTES,
    ChecksumName,
    ZarrFormat,
    _check_array_dtype,
    _check_array_target,
    _python_dimensions,
    _require_array,
    _validate_resource_name,
    index_marker_error,
)
from sigmf_zarr.store._transaction import (
    _settled_on_failure,
    replacement_transaction,
)

if TYPE_CHECKING:
    from sigmf_zarr.store._container import SigMFZarrStore


def _exact_samples(
    samples: npt.ArrayLike, dtype: npt.DTypeLike
) -> npt.NDArray[Any]:
    """Convert samples to a stored dtype only when every value is preserved.

    Args:
        samples: Numeric or Boolean sample values.
        dtype: Stored sample dtype.

    Returns:
        Samples with the stored dtype.

    Raises:
        ValueError: If the values are not numeric, or the conversion would
            wrap, round, saturate, or discard an imaginary component.
    """
    source = np.asarray(samples)
    target = np.dtype(dtype)
    if source.dtype.kind not in "biufc":
        raise ValueError(f"Samples must be numeric, got dtype {source.dtype}")
    if source.dtype.kind in "iu" and target.kind in "iu":
        source_limits = np.iinfo(source.dtype)
        target_limits = np.iinfo(target)
        # Signedness changes can wrap and still survive a round trip. Each
        # compared bound fits the source dtype, avoiding mixed-sign promotion.
        if (
            target_limits.min > source_limits.min
            and np.any(source < target_limits.min)
        ) or (
            target_limits.max < source_limits.max
            and np.any(source > target_limits.max)
        ):
            raise ValueError(
                f"Samples of dtype {source.dtype} cannot be stored as "
                f"{target} "
                "without changing their values"
            )
    # Widening within one kind is exact. Other casts, including integer to
    # floating point, are checked by a round trip through the stored dtype.
    if source.dtype.kind == target.kind and np.can_cast(
        source.dtype, target, casting="safe"
    ):
        return np.asarray(source, dtype=target)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        converted = source.astype(target)
        restored = converted.astype(source.dtype)
    # Compare components separately. Complex equal_nan comparison otherwise
    # treats a differing finite component as equal when the other is NaN.
    components = (
        ((source.real, restored.real), (source.imag, restored.imag))
        if np.iscomplexobj(source)
        else ((source, restored),)
    )
    # A round trip does not detect infinity. Integer conversion maps it to
    # an integer bound, and float16 maps a large integer bound back to it.
    non_finite = (
        target.kind in "biu" and not np.all(np.isfinite(source))
    ) or (
        target.kind in "fc"
        and np.any(np.isfinite(source) & ~np.isfinite(converted))
    )
    if non_finite or not all(
        np.array_equal(original, recovered, equal_nan=True)
        for original, recovered in components
    ):
        raise ValueError(
            f"Samples of dtype {source.dtype} cannot be stored as {target} "
            "without changing their values"
        )
    return converted


_INT64_MAX = 2**63 - 1
"""Greatest sample index that the SigMF schema allows."""

_NAME_PREVIEW_LENGTH = 8
"""Greatest number of names that an error message lists."""


def _check_sample_index(value: object, *, name: str) -> None:
    """Require a sample index within the range of the SigMF schema.

    The schema bounds `core:offset` and `core:sample_start` to the int64
    range, on which coordinate arithmetic relies.

    Args:
        value: Candidate sample index.
        name: Field name for the error message.

    Raises:
        ValueError: If `value` is not an integer from 0 to 2**63 - 1.
    """
    if type(value) is not int or not 0 <= value <= _INT64_MAX:
        raise ValueError(
            f"{name} must be a nonnegative integer at most {_INT64_MAX}"
        )


def _name_preview(names: Sequence[str]) -> str:
    """Format a list of names for an error message, truncating long lists.

    Args:
        names: Names in display order.

    Returns:
        Bracketed name list, followed by the number of omitted names.
    """
    shown = ", ".join(repr(name) for name in names[:_NAME_PREVIEW_LENGTH])
    omitted = len(names) - _NAME_PREVIEW_LENGTH
    return f"[{shown}]" + (f" and {omitted} more" if omitted > 0 else "")


def _is_channel_group_name(name: str, count: int) -> bool:
    """Return whether a name is the canonical decimal of a channel index.

    Args:
        name: Child group name below `channels/`.
        count: Number of declared channels.

    Returns:
        True for a name such as `3` that names a channel below `count`.
    """
    return (
        name.isascii()
        and name.isdecimal()
        and len(name) <= len(str(count))
        and (name == "0" or not name.startswith("0"))
        and int(name) < count
    )


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
        {"core:datatype"}
    )
    """Settable metadata fields whose changes alter SigMF dataset bytes."""

    _INTEGRITY_SAMPLE_FIELD: ClassVar[str] = "sample_sha512"
    """Internal integrity field containing the logical sample-array hash."""

    _INTEGRITY_METADATA_FIELD: ClassVar[str] = "metadata_sha512"
    """Internal integrity field containing the logical metadata hash."""

    _INDEX_VALID_FIELD: ClassVar[str] = "sigmf-zarr:valid"
    """Index attribute indicating whether alignment is still valid."""

    _INDEX_INVALID_REASON_FIELD: ClassVar[str] = "sigmf-zarr:invalid-reason"
    """Index attribute explaining why an index is invalid."""

    def __init__(
        self,
        store: SigMFZarrStore,
        recording_name: str,
        *,
        create: bool = False,
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
        """Open an existing recording or create a new one.

        Use :meth:`SigMFRecordings.open` and :meth:`SigMFRecordings.create`
        rather than calling the constructor directly. Without `create`, the
        recording must exist and the creation arguments are ignored.

        Args:
            store: Parent SigMF-Zarr store containing the recording.
            recording_name: Recording identifier under `recordings/`.
            create: Whether to create the recording instead of opening it.
            overwrite: Whether creation may replace an existing recording.
            batched: Whether the `samples` array includes a leading
                batch axis of homogeneous samples.
            sample_dtype: Element dtype for the sample array when
                creating a new recording: a Boolean, integer,
                floating-point, or complex dtype. A big-endian dtype is
                stored and recorded in little-endian byte order.
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
                when creating a new recording. Accepted forms follow
                :meth:`SigMFZarrStore.create_array`.
            sample_checksum: Optional chunk-level checksum for the sample
                array. Defaults to `crc32c`.

        Raises:
            KeyError: If `create` is false and the recording does not exist.
            ValueError: If `overwrite` is requested without `create`, the
                recording exists and `overwrite` is false, or creation lacks a
                valid `sample_shape`.
        """
        if overwrite and not create:
            raise ValueError("overwrite requires create=True")
        _validate_resource_name(recording_name, kind="Recording")
        self._store = store
        self._recording_name = recording_name
        # Only creation needs rollback. An open must not write or delete
        # anything, even when the recording is damaged.
        with (
            replacement_transaction(
                store._recordings_group,
                recording_name,
                overwrite=overwrite,
                parents=(store._refresh_root(),),
            )
            if create
            else nullcontext()
        ):
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
            ValueError: If metadata is malformed, the declared dtype is not
                numeric, is not the canonical NumPy dtype string, is
                big-endian, or does not describe the samples array, or a
                complex dtype is declared with an `iq` axis.
        """
        global_metadata = json_object(
            self._raw_group.attrs.get("global", {}),
            name=f"recording {self.name!r} global metadata",
        )
        declarations = json_object_list(
            global_metadata.get("core:extensions", []), name="core:extensions"
        )
        managed = [
            entry
            for entry in declarations
            if entry.get("name") == self.ZARR_EXTENSION_NAMESPACE
        ]
        if (
            len(managed) != 1
            or managed[0].get("version") != self.ZARR_EXTENSION_VERSION
            or managed[0].get("optional") is not True
        ):
            raise ValueError(
                f"Recording {self.name!r} core:extensions must declare "
                f"one optional {self.ZARR_EXTENSION_NAMESPACE!r} extension "
                f"with version {self.ZARR_EXTENSION_VERSION!r}"
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
        if not is_numeric_sample_dtype(declared_dtype):
            raise ValueError(
                f"Recording {self.name!r} {field!r} value {declared_dtype!r} "
                "is not a Boolean, integer, floating-point, or complex dtype"
            )
        if not is_logical_sample_dtype(declared_dtype):
            raise ValueError(
                f"Recording {self.name!r} {field!r} value {declared_dtype!r} "
                "must use little-endian byte order"
            )
        # Other spellings, such as `float32` or `=f4`, would change the
        # metadata hash for identical content and need NumPy to interpret.
        if normalized_dtype != declared_dtype:
            raise ValueError(
                f"Recording {self.name!r} {field!r} value {declared_dtype!r} "
                f"is not the canonical dtype string {normalized_dtype!r}"
            )
        actual_dtype = np.dtype(cast(Any, self._samples_array.dtype)).str
        if normalized_dtype != actual_dtype:
            raise ValueError(
                f"Recording {self.name!r} {field!r} value {declared_dtype!r} "
                f"does not match samples dtype {actual_dtype!r}"
            )
        type(self)._check_complex_axes(
            np.dtype(declared_dtype), self._stored_sample_axes()
        )

    @staticmethod
    def _check_complex_axes(
        sample_dtype: np.dtype[Any], sample_axes: Sequence[str]
    ) -> None:
        """Reject a complex sample dtype combined with an `iq` axis.

        Each element of a complex dtype holds one I/Q pair, so an `iq` axis
        would give a second, contradictory place for the components.

        Args:
            sample_dtype: Sample element dtype.
            sample_axes: Sample axis names.

        Raises:
            ValueError: If `sample_dtype` is complex and `sample_axes`
                contains `iq`.
        """
        if sample_dtype.kind == "c" and "iq" in sample_axes:
            raise ValueError(
                f"A complex sample dtype {sample_dtype.str!r} holds I and Q "
                "in each element, so the sample axes must not contain an "
                f"'iq' axis, got {tuple(sample_axes)!r}"
            )

    def _ensure_exists(
        self,
        *,
        create: bool,
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
        """Create the backing recording group or require that it exists.

        Args:
            create: Whether to create the recording instead of opening it.
            overwrite: Whether creation may replace an existing recording.
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
            KeyError: If `create` is false and the recording does not exist.
            ValueError: If the recording exists and `overwrite` is false, or
                creation lacks a valid `sample_shape`.
        """
        exists = self.name in self._store._recordings_group
        if not create:
            if not exists:
                raise KeyError(self.name)
            return
        if exists and not overwrite:
            raise ValueError(
                f"Recording {self.name!r} already exists. Pass "
                "overwrite=True to replace it"
            )
        self._store._invalidate_metadata_integrity()
        if exists:
            del self._store._recordings_group[self.name]
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
                dimensions, `sample_dtype` is not a Boolean, integer,
                floating-point, or complex dtype, or a complex
                `sample_dtype` is combined with an `iq` axis.
        """
        if not sample_shape:
            raise ValueError("sample_shape must not be empty when creating")
        sample_shape = _python_dimensions(sample_shape)
        sample_chunks = _python_dimensions(sample_chunks)
        sample_shards = _python_dimensions(sample_shards)
        # Both Zarr formats store and record the same little-endian dtype, so
        # recording metadata does not depend on the source byte order.
        sample_dtype = logical_sample_dtype(sample_dtype)
        if not is_numeric_sample_dtype(sample_dtype):
            raise ValueError(
                "sample_dtype must be a Boolean, integer, floating-point, or "
                f"complex dtype, got {sample_dtype.str!r}"
            )
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
        type(self)._check_complex_axes(sample_dtype, normalized_axes)
        normalized_channels = type(self)._normalize_channel_metadata(
            sample_shape,
            sample_axes=normalized_axes,
            channel_metadata=channel_metadata,
        )
        # Reject malformed caller metadata before writing any arrays.
        json_object(global_metadata or {}, name="global metadata")
        recording_captures = json_object_list(captures or [], name="captures")
        recording_annotations = json_object_list(
            annotations or [], name="annotations"
        )
        group = self._store._recordings_group.create_group(self.name)
        chunk_shape = sample_chunks or type(self)._default_creation_chunks(
            sample_dtype,
            sample_shape,
            sample_axes=normalized_axes,
            batched=batched,
            sample_shards=sample_shards,
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
            sample_dtype=cast(Array[Any], group["samples"]).dtype,
            sample_shape=sample_shape,
            sample_axes=normalized_axes,
            sample_checksum=sample_checksum,
        )
        group.attrs.update(
            {
                "global": recording_global,
                "captures": recording_captures,
                "annotations": recording_annotations,
            }
        )

    @staticmethod
    def _largest_divisor_at_most(value: int, limit: int) -> int:
        """Return the largest divisor of a positive integer within a limit.

        Args:
            value: Positive integer to divide.
            limit: Positive upper bound.

        Returns:
            Largest `d <= limit` such that `value % d == 0`.
        """
        best = 1
        for small in range(1, math.isqrt(value) + 1):
            if value % small == 0:
                for divisor in (small, value // small):
                    if best < divisor <= limit:
                        best = divisor
        return best

    @classmethod
    def _default_creation_chunks(
        cls,
        sample_dtype: npt.DTypeLike,
        sample_shape: tuple[int, ...],
        *,
        sample_axes: tuple[str, ...],
        batched: bool,
        sample_shards: ShardsLike | None,
    ) -> tuple[int, ...]:
        """Choose chunks for a new recording whose final length is unknown.

        Chunks keep complete items, or complete I/Q and channel planes, and
        span about `DEFAULT_SAMPLE_CHUNK_BYTES` along the growth axis. Unlike
        import chunks, they are not limited by the initial length, because
        appends extend that axis. With an explicit shard shape, the chunk
        length is the largest divisor of the shard length within that size,
        so every shard holds whole chunks.

        Args:
            sample_dtype: Sample array dtype.
            sample_shape: Per-item shape, or the complete unbatched shape.
            sample_axes: Validated axis names for `sample_shape`.
            batched: Whether the sample array has a leading item axis.
            sample_shards: Optional sample shard shape.

        Returns:
            Logical chunk shape for the complete sample array.
        """
        itemsize = np.dtype(sample_dtype).itemsize
        if batched:
            growth_axis = 0
            chunks = [1, *sample_shape]
            position_elements = math.prod(sample_shape)
        else:
            growth_axis = sample_axes.index("time")
            chunks = list(sample_shape)
            position_elements = math.prod(
                size
                for axis, size in enumerate(sample_shape)
                if axis != growth_axis
            )
        length = max(
            1, DEFAULT_SAMPLE_CHUNK_BYTES // (itemsize * position_elements)
        )
        # Zarr accepts a shard shape as a sequence or as the "shape" entry of
        # a mapping. Each regular form gets the same divisor rule.
        shard_shape: object = (
            sample_shards.get("shape")
            if isinstance(sample_shards, Mapping)
            else sample_shards
        )
        if (
            isinstance(shard_shape, Sequence)
            and not isinstance(shard_shape, str)
            and len(shard_shape) == len(chunks)
            and all(
                isinstance(size, int | np.integer)
                and not isinstance(size, bool)
                and size > 0
                for size in shard_shape
            )
        ):
            length = cls._largest_divisor_at_most(
                int(shard_shape[growth_axis]), length
            )
        chunks[growth_axis] = length
        return tuple(chunks)

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
            ValueError: If the checksum name is unsupported or the compressor
                cannot be represented in Zarr format 3.
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
            codecs = [cls._compressor_v3(sample_compressor)]

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
        if sample_compressor is None or isinstance(
            sample_compressor, NumcodecsCodec
        ):
            return sample_compressor
        if isinstance(sample_compressor, ZstdCodec):
            return NumcodecsZstd(
                level=sample_compressor.level,
                checksum=sample_compressor.checksum,
            )
        if isinstance(sample_compressor, GzipCodec):
            return NumcodecsGZip(level=sample_compressor.level)
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
            "Compressor cannot be represented in Zarr format 2: "
            f"{sample_compressor!r}. Use 'auto', None, a numcodecs codec, "
            "or a zarr.codecs BloscCodec, ZstdCodec, or GzipCodec"
        )

    @staticmethod
    def _compressor_v3(compressor: object) -> object:
        """Translate one compressor to a Zarr format 3 codec.

        Args:
            compressor: Requested compressor other than `"auto"` or `None`.

        Returns:
            Zarr format 3 bytes-to-bytes codec or codec mapping.

        Raises:
            ValueError: If the compressor cannot be represented in Zarr
                format 3.
        """
        if isinstance(compressor, BytesBytesCodec | Mapping):
            # Zarr validates codec mappings when it creates the array.
            return compressor
        if isinstance(compressor, NumcodecsBlosc):
            config = compressor.get_config()
            shuffles: dict[int, str | None] = {
                NumcodecsBlosc.NOSHUFFLE: "noshuffle",
                NumcodecsBlosc.SHUFFLE: "shuffle",
                NumcodecsBlosc.BITSHUFFLE: "bitshuffle",
                # Zarr format 3 chooses the shuffle from the array dtype when
                # it is unset, as numcodecs does for automatic shuffle.
                NumcodecsBlosc.AUTOSHUFFLE: None,
            }
            if config["shuffle"] not in shuffles:
                raise ValueError(
                    "Unsupported Blosc shuffle setting for Zarr format 3: "
                    f"{config['shuffle']!r}"
                )
            return BloscCodec(
                cname=config["cname"],
                clevel=config["clevel"],
                shuffle=cast(Any, shuffles[config["shuffle"]]),
                blocksize=config["blocksize"],
            )
        if isinstance(compressor, NumcodecsZstd):
            config = compressor.get_config()
            return ZstdCodec(
                level=config["level"],
                checksum=bool(config.get("checksum", False)),
            )
        if isinstance(compressor, NumcodecsGZip):
            return GzipCodec(level=compressor.get_config()["level"])
        raise ValueError(
            "Compressor cannot be represented in Zarr format 3: "
            f"{compressor!r}. Use 'auto', None, a Zarr format 3 "
            "bytes-to-bytes codec or codec mapping, or a numcodecs Blosc, "
            "Zstd, or GZip codec"
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
    def _normalize_extension_declarations(
        cls, declarations: object
    ) -> list[JSONObject]:
        """Preserve other declarations and maintain the storage extension.

        Args:
            declarations: Caller-supplied extension declarations.

        Returns:
            Declarations with one canonical SigMF-Zarr entry.

        Raises:
            ValueError: If declarations are not a list of finite JSON objects.
        """
        extensions = json_object_list(declarations, name="core:extensions")
        result = [
            entry
            for entry in extensions
            if entry.get("name") != cls.ZARR_EXTENSION_NAMESPACE
        ]
        result.append(cls._zarr_extension_declaration())
        return result

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
        declared_extensions = cls._normalize_extension_declarations(
            source_metadata.get(extension_field, [])
        )
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
            f"sample_axes must be provided for sample_shape {sample_shape!r}"
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
            has_crc32c = type(self)._codecs_have_crc32c(array_metadata.codecs)
        if not has_crc32c:
            raise ValueError(
                f"Recording {self.name!r} item_metadata chunks must use CRC32C"
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

        # The declared count can be far larger than the store, so the names
        # are compared without building the expected set.
        expected = self.num_channels if "channel" in self.sample_axes else 0
        actual_groups = sorted(self._channels_group.group_keys())
        if len(actual_groups) != expected or not all(
            _is_channel_group_name(name, expected) for name in actual_groups
        ):
            expected_text = (
                f"channels '0' to {str(expected - 1)!r}"
                if expected
                else "no channel groups"
            )
            raise ValueError(
                f"Recording {self.name!r} channel groups "
                f"{_name_preview(actual_groups)} do not match {expected_text}"
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
        if declared is not None and declared != "crc32c":
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
            f"item_count={len(self)}, "
            f"sample_count={self.sample_count}, "
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
        return without_consolidated_metadata(
            cast(Group, self._store._recordings_group[self._recording_name])
        )

    @property
    def group(self) -> ReadOnlyGroup:
        """Read-only recording group under the parent store.

        Returns:
            Read-only Zarr group view for the recording.
        """
        return ReadOnlyGroup(self._raw_group)

    @property
    def _samples_array(self) -> Array[Any]:
        """Writable sample array used by managed mutation paths.

        Returns:
            Backing Zarr sample array.
        """
        return cast(Array[Any], self._raw_group["samples"])

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

        Raises:
            ValueError: If `extensions` is not a group.
        """
        return self._required_child_group("extensions")

    @property
    def _indexes_group(self) -> Group:
        """Writable index group used by managed mutation paths.

        Returns:
            Backing Zarr indexes group.

        Raises:
            ValueError: If `indexes` is not a group.
        """
        return self._required_child_group("indexes")

    def _required_child_group(self, name: str) -> Group:
        """Return a required child group of this recording.

        Args:
            name: Child group name.

        Returns:
            Backing child group.

        Raises:
            KeyError: If the child is absent.
            ValueError: If the child is an array rather than a group.
        """
        member = self._raw_group[name]
        if not isinstance(member, Group):
            raise ValueError(
                f"Recording {self.name!r} {name!r} must be a group"
            )
        return without_consolidated_metadata(member)

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
        return without_consolidated_metadata(
            cast(Group, self._raw_group["channels"])
        )

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
        for _, index in members(self._indexes_group, max_depth=None):
            if not isinstance(index, Array):
                continue
            if index.attrs.get("axis") != axis:
                continue
            index.attrs[self._INDEX_VALID_FIELD] = False
            index.attrs[self._INDEX_INVALID_REASON_FIELD] = (
                f"axis {axis!r} changed"
            )

    def _validate_mutated_index(
        self, index_name: str, index: Array[Any]
    ) -> None:
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
        index: Array[Any],
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
            members(self._indexes_group, max_depth=None)
        ):
            if not isinstance(index, Array):
                continue
            marker_error = index_marker_error(index_name, index.attrs)
            if marker_error is not None:
                return marker_error
            error = self._index_structure_error(index_name, index)
            if error is not None:
                return error
        return None

    @contextmanager
    def mutate_samples(self) -> Generator[Array[Any], None, None]:
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
            # Restore the shape only after any failed or interrupted write
            # finishes, so a late write cannot undo the restoration.
            with (
                _settled_on_failure(),
                self._mutation(metadata=True, samples=True),
            ):
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
    ) -> Generator[Array[Any], None, None]:
        """Provide writable access to one recording-level index.

        The index is marked invalid before writable access is granted. A
        successful context exit validates its shape and alignment and marks it
        current again. If the body raises, the index remains invalid.

        Args:
            index_name: Recording-level index name.

        Yields:
            Writable backing Zarr index array.

        Raises:
            KeyError: If no index array has this name.
            ValueError: If `index_name` is malformed or the mutated index is
                structurally invalid.
        """
        index = _require_array(self._indexes_group, index_name, kind="Index")
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

        Slabs follow the growth axis, which is `item` for a batched recording
        and `time` otherwise, so each sample chunk is read once.

        Returns:
            Lowercase SHA-512 hexadecimal digest.
        """
        return calculate_array_sha512(
            self._samples_array, axis=self.axis_index(self._growth_axis)
        )

    @property
    def _growth_axis(self) -> str:
        """Runtime axis extended by appends.

        Returns:
            `item` for a batched recording, otherwise `time`.
        """
        return "item" if self.batched else "time"

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

    def calculate_source_metadata_sha512(self) -> str:
        """Hash recording metadata while excluding recording indexes.

        Sample values have a separate hash. Selected input indexes must be
        hashed separately. Excluding indexes lets an output index retain its
        input binding without including itself in that binding. The global
        `core:sha512` field is also excluded. It is a digest of the sample
        values, so `update_sha512()` does not change this hash.

        Returns:
            Lowercase SHA-512 hexadecimal digest.
        """
        payload = group_metadata_payload(
            self._raw_group,
            exclude_array_values=frozenset({"samples"}),
            exclude_groups=frozenset({"indexes"}),
        )
        attrs = cast(dict[str, object], payload["attrs"])
        shared = attrs.get("global")
        if isinstance(shared, dict):
            attrs["global"] = {
                key: value
                for key, value in shared.items()
                if key != SHA512_KEY
            }
        return calculate_payload_sha512(payload, domain=METADATA_DOMAIN)

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
    def _item_metadata_array(self) -> Array[Any]:
        """Writable item metadata array used by managed mutation paths.

        Returns:
            Backing item metadata array.

        Raises:
            KeyError: If no per-item metadata array is present.
        """
        if not self.has_item_metadata:
            raise KeyError("item_metadata")
        return cast(Array[Any], self._raw_group["item_metadata"])

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
    def mutate_item_metadata(self) -> Generator[Array[Any], None, None]:
        """Provide validated writable access to per-item metadata.

        The context is intended for value updates that do not resize the
        array. Use :meth:`set_item_metadata` or :meth:`append_samples` for
        length changes so the item axis stays aligned.

        Yields:
            Writable backing item metadata array.

        Raises:
            KeyError: If the recording has no item metadata array.
            ValueError: If the mutation changes the array shape, or the
                resulting entries are malformed. After a shape change, the
                original shape is restored. Entries outside a shrunken extent
                are lost and become empty metadata objects, and the metadata
                hashes remain absent.
        """
        array = self._item_metadata_array
        original_shape = tuple(int(size) for size in array.shape)
        try:
            # Restore the shape only after any failed or interrupted write
            # finishes, so a late write cannot undo the restoration.
            with _settled_on_failure(), self._mutation(metadata=True):
                yield array
        finally:
            # Restore the shape even when the caller raises. The item axis
            # length and aligned indexes describe the original length.
            current_shape = tuple(int(size) for size in array.shape)
            if current_shape != original_shape:
                array.resize(original_shape)
                kept = min(current_shape[0], original_shape[0])
                if kept < original_shape[0]:
                    # Zarr may delete chunks outside the shrunken extent, and
                    # the fill value is not JSON. Store empty bundles instead.
                    array[kept:] = np.full(
                        original_shape[0] - kept, "{}", dtype=object
                    )
        if current_shape != original_shape:
            raise ValueError(
                "mutate_item_metadata() cannot resize the item metadata "
                "array. Use set_item_metadata() or append_samples()"
            )
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

        Storage fields in the `sigmf-zarr` namespace describe the sample
        array layout. The recording maintains them, so callers cannot set
        them here. Changes to `core:extensions` retain the canonical
        SigMF-Zarr declaration.

        Args:
            key: Metadata field key.
            value: JSON-compatible value to store.

        Raises:
            ValueError: If the key is empty, contains a lone surrogate, or
                names a managed `sigmf-zarr` storage field, or the value is
                not finite JSON.
        """
        # Untyped callers can pass any value.
        if not isinstance(key, str) or not key:  # type: ignore[redundant-expr]
            raise ValueError("Global metadata keys must be non-empty strings")
        # Opening the recording rejects a key with no UTF-8 encoding, so the
        # key needs the same check as the value.
        json_value(key, name="global metadata key")
        if key.startswith(f"{self.ZARR_EXTENSION_NAMESPACE}:"):
            raise ValueError(
                f"Global metadata field {key!r} is managed by the recording "
                "and cannot be set directly"
            )
        if key == "core:extensions":
            normalized_value = cast(
                JSONValue, type(self)._normalize_extension_declarations(value)
            )
        else:
            normalized_value = json_value(
                value, name=f"global metadata {key!r}"
            )
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
            _check_sample_index(
                item_global.get("core:offset", 0),
                name=f"{name} global metadata core:offset",
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
        except RecursionError as exc:
            raise ValueError(f"{name} is nested too deeply") from exc
        return cls._normalize_item_metadata_entry(
            json_object(decoded, name=name),
            name=name,
        )

    def _create_item_metadata_array(
        self,
        encoded: npt.NDArray[np.object_],
        *,
        overwrite: bool,
    ) -> Array[Any]:
        """Create the checksummed per-item metadata array.

        Args:
            encoded: Canonical JSON strings in item order.
            overwrite: Whether an existing array may be replaced.

        Returns:
            Writable backing item metadata array.

        Raises:
            ValueError: If the array already exists and overwrite is false.
        """
        _check_array_target(
            self._raw_group, "item_metadata", kind="Array", overwrite=overwrite
        )
        with replacement_transaction(
            self._raw_group,
            "item_metadata",
            overwrite=overwrite,
            parents=(self._store._refresh_root(), self._raw_group),
        ):
            self._invalidate_integrity(metadata=True)
            group = self._raw_group
            if "item_metadata" in group:
                del group["item_metadata"]

            # Appends grow this array, so its chunk length must not follow
            # the first write. Matching the sample objects along the item
            # axis keeps one-item appends from writing one object per item.
            samples = self._samples_array
            sample_items = int((samples.shards or samples.chunks)[0])
            chunks = self._store.default_index_chunks(
                max(len(encoded), sample_items)
            )
            # Both formats store UTF-8 JSON strings through different
            # serializer and codec APIs.
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
        item_index: int | np.integer[Any],
        value: JSONObject | None,
    ) -> None:
        """Set one validated per-item metadata bundle.

        An absent metadata array is created with empty bundles for every
        other item.

        Args:
            item_index: Item-axis index. Negative indexes and NumPy integers
                are supported.
            value: Metadata bundle or `None` for an empty bundle.

        Raises:
            IndexError: If the item index is out of range.
            TypeError: If the item index is not an integer or is Boolean.
            ValueError: If the recording is unbatched or the bundle is
                invalid.
        """
        if not self.batched:
            raise ValueError(
                "item_metadata is only supported for batched recordings"
            )
        normalized_index = self._item_position(item_index)
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
            TypeError: If the selection is not a slice or has non-integer
                bounds.
            ValueError: If the recording is unbatched, the slice is not
                contiguous, the value count is wrong, or a bundle is invalid.
        """
        if not self.batched:
            raise ValueError(
                "item_metadata is only supported for batched recordings"
            )
        # Untyped callers can pass any value.
        if not isinstance(selection, slice):
            raise TypeError("Item metadata selection must be a slice")
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

    def _item_position(self, item_index: object) -> int:
        """Check and normalize one item-axis position before any access.

        Args:
            item_index: Requested position. Negative positions count from the
                end.

        Returns:
            Nonnegative item position.

        Raises:
            TypeError: If the position is not an integer or is Boolean.
            IndexError: If the position is out of range.
        """
        # A Boolean is an integer in Python, but it would silently select
        # item 0 or 1.
        if isinstance(item_index, bool | np.bool_) or not isinstance(
            item_index, int | np.integer
        ):
            raise TypeError("Item positions must be non-Boolean integers")
        length = len(self)
        if not -length <= item_index < length:
            raise IndexError(item_index)
        return int(item_index) % length

    def get_item_metadata(
        self, item_index: int | np.integer[Any]
    ) -> JSONObject:
        """Return one item-specific metadata bundle.

        Args:
            item_index: Item-axis index. Negative indexes and NumPy integers
                are supported.

        Returns:
            Decoded item-specific metadata. Returns an empty object when no
            `item_metadata` array exists.

        Raises:
            IndexError: If the item index is out of range.
            TypeError: If the item index is not an integer or is Boolean.
            ValueError: If stored item metadata is malformed.
        """
        normalized_index = self._item_position(item_index)
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
        entry = self.get_item_metadata(item_index)
        item_global = cast(JSONObject, entry.get("global", {}))
        item_captures = cast(list[JSONObject], entry.get("captures", []))
        item_annotations = cast(list[JSONObject], entry.get("annotations", []))
        return {
            "global": {**self.global_metadata, **item_global},
            "captures": [*self.captures, *item_captures],
            "annotations": [*self.annotations, *item_annotations],
        }

    def channel_metadata(
        self, channel_index: int | np.integer[Any]
    ) -> JSONObject:
        """Return metadata for one explicit sample channel.

        Args:
            channel_index: Zero-based channel index. NumPy integers are
                accepted.

        Returns:
            Channel metadata object.

        Raises:
            TypeError: If the channel index is Boolean or not an integer.
            ValueError: If the recording has no explicit channel axis.
            IndexError: If the channel index is out of range.
        """
        channel = self._channel_group(channel_index)
        return cast(JSONObject, dict(channel.attrs))

    def _channel_group(self, channel_index: int | np.integer[Any]) -> Group:
        """Return the group of one explicit sample channel.

        Args:
            channel_index: Zero-based channel index. NumPy integers are
                accepted.

        Returns:
            Backing channel group.

        Raises:
            TypeError: If the channel index is Boolean or not an integer.
            ValueError: If the recording has no explicit channel axis.
            IndexError: If the channel index is out of range.
        """
        # Untyped callers can pass any value. A Boolean or float would
        # otherwise name a group such as "True" or "1.0".
        if isinstance(channel_index, bool | np.bool_) or not isinstance(
            channel_index, int | np.integer
        ):
            raise TypeError(
                "Channel index must be an integer, got "
                f"{type(channel_index).__name__}"
            )
        position = int(channel_index)
        if "channel" not in self.sample_axes:
            raise ValueError(
                f"Recording {self.name!r} has no explicit channel axis"
            )
        if position < 0 or position >= self.num_channels:
            raise IndexError(channel_index)
        return cast(Group, self._channels_group[str(position)])

    def set_channel_metadata(
        self,
        channel_index: int | np.integer[Any],
        metadata: JSONObject,
    ) -> None:
        """Replace metadata for one explicit sample channel.

        Args:
            channel_index: Zero-based channel index.
            metadata: JSON-compatible channel metadata object.

        Raises:
            TypeError: If the channel index is Boolean or not an integer.
            ValueError: If the recording has no channel axis or metadata is
                not JSON serializable.
            IndexError: If the channel index is out of range.
        """
        channel = self._channel_group(channel_index)
        copied = json_object(
            dict(metadata),
            name=f"channel {channel_index} metadata",
        )
        self._invalidate_integrity(metadata=True)
        # One write replaces every attribute, so a failure cannot leave the
        # channel with only some of its old or new attributes.
        channel.attrs.put(cast(dict[str, Any], copied))

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
        self._raw_group.attrs["captures"] = [*self.captures, entry]

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
            ValueError: If rank differs from the sample array rank, the
                per-item sample shape does not match, or `capture` is
                supplied with zero items.
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
        if append_count == 0:
            if capture is not None:
                raise ValueError(
                    "A capture must describe at least one item, but the "
                    "append has no items"
                )
            # Nothing changes, so every hash remains current.
            return

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
            ValueError: If rank differs from the sample array rank, any
                non-time axis changes size, or `capture` is supplied with
                zero samples.
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
        capture_start = start
        if capture is not None and "core:sample_start" not in capture:
            offset = self.global_metadata.get("core:offset", 0)
            if type(offset) is not int or offset < 0:
                raise ValueError("core:offset must be a nonnegative integer")
            capture_start += offset
        stop = start + int(sample_array.shape[time_axis])
        if stop == start:
            if capture is not None:
                raise ValueError(
                    "A capture must describe at least one sample, but the "
                    "append has no samples"
                )
            # Nothing changes, so every hash remains current.
            return
        shape = [int(dim) for dim in self._samples_array.shape]
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
        self._append_capture(capture, sample_start=capture_start)

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
                fixed axis changes size, the samples are not numeric or
                cannot be stored in the sample dtype without changing a
                value, metadata is invalid, or `capture` is supplied with
                zero items or samples.
        """
        # Conversion and metadata validation must finish before any shape,
        # index validity, or integrity state changes.
        sample_array = _exact_samples(samples, self._samples_array.dtype)
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
                axis is empty, or the samples are not numeric or cannot be
                stored in the sample dtype without changing a value.
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
        sample_array = _exact_samples(sample_array, self._samples_array.dtype)
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
            f"  item_count={len(self)},",
            f"  sample_count={self.sample_count},",
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
        attributes: JSONObject | None = None,
    ) -> Array[Any]:
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
            attributes: Validated attributes written before committing.

        Returns:
            Created Zarr array.

        Raises:
            ValueError: If `name` is malformed, `data` has a structured or
                void dtype, the target exists and `overwrite` is false,
                `overwrite` targets a group, or an encoding option is not
                available in the store's format.
        """
        _check_array_target(group, name, kind="Array", overwrite=overwrite)
        values = np.asarray(data)
        # Reject the dtype and encoding options before invalidating any hash.
        _check_array_dtype(values.dtype)
        self._store._resolve_encoding(
            shards=shards,
            compressors=compressors,
            compressor=compressor,
            serializer=serializer,
        )
        with replacement_transaction(
            group,
            name,
            overwrite=overwrite,
            parents=(self._store._refresh_root(), self._raw_group),
        ):
            self._invalidate_integrity(metadata=True)
            array = self._store.create_array(
                group,
                name,
                values,
                overwrite=overwrite,
                chunks=chunks,
                shards=shards,
                compressors=compressors,
                compressor=compressor,
                serializer=serializer,
            )
            if attributes is not None:
                array.attrs.update(attributes)
            return array

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

        The array may have any rank, including zero. Compressors follow
        the rules of :meth:`SigMFZarrStore.create_array`.

        Args:
            name: Array name or nested relative path under `extensions/`.
            data: Array-like values used to initialize the new array.
            overwrite: Whether an existing array may be replaced.
            chunks: Optional chunk shape. When omitted, a one-dimensional
                array uses chunks of at most 4096 values, and an array of
                any other rank uses Zarr's automatic chunk shape.
            shards: Optional Zarr format 3 shard shape.
            compressors: Optional compressor or compressor pipeline. Zarr
                format 2 accepts at most one compressor.
            compressor: Optional single compressor. Specify at most one of
                `compressors` and `compressor`.
            serializer: Optional Zarr format 3 serializer.

        Returns:
            Read-only view of the created array.

        Raises:
            ValueError: If `name` is empty, starts or ends with `/`, or has
                an empty, `.`, or `..` segment, `data` has a structured or
                void dtype, the target exists and `overwrite` is false,
                `overwrite` targets a group, or an encoding option is not
                available in the store's format.
        """
        _check_array_target(
            self._extensions_group,
            name,
            kind="Extension array",
            overwrite=overwrite,
        )
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
            overwrite: Whether an existing index may be replaced.
            chunks: Optional chunk shape.

        Returns:
            Created index array.

        Raises:
            ValueError: If values are not one-dimensional, the axis is not
                present, or the index length does not match the axis length.
        """
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

        attrs: JSONObject = {
            "axis": axis,
            "field": field,
            "kind": kind,
        }
        if unit is not None:
            attrs["unit"] = unit
        if labels is not None:
            attrs["labels"] = list(labels)
        index = self._create_array(
            self._indexes_group,
            index_name,
            value_array,
            overwrite=overwrite,
            attributes=json_object(attrs, name="index attributes"),
            chunks=(
                chunks or self._store.default_index_chunks(len(value_array))
            ),
        )
        return ReadOnlyArray(index)

    def index(self, index_name: str) -> ReadOnlyArray:
        """Return one recording-level metadata index.

        Args:
            index_name: Index array name.

        Returns:
            Read-only recording-level index array.

        Raises:
            KeyError: If no index array has this name.
            ValueError: If `index_name` is malformed, or the index is
                structurally invalid or was invalidated by an axis mutation.
        """
        index = _require_array(self._indexes_group, index_name, kind="Index")
        marker_error = index_marker_error(index_name, index.attrs)
        if marker_error is not None:
            raise ValueError(
                f"{marker_error}. Repair it with mutate_index() or replace "
                "it with add_index()"
            )
        error = self._index_structure_error(index_name, index)
        if error is not None:
            raise ValueError(error)
        return ReadOnlyArray(index)

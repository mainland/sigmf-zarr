"""Import and export helpers for standard SigMF files and archives."""

from __future__ import annotations

import errno
import hashlib
import hmac
import json
import os
import stat
import tarfile
import tempfile
import warnings
from collections.abc import Iterator, Sequence
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, cast

import numpy as np
import numpy.typing as npt
from sigmf import SHA512_KEY
from sigmf import __specification__ as SIGMF_SPECIFICATION
from sigmf.error import SigMFError
from sigmf.sigmffile import SigMFCollection as StandardSigMFCollection
from sigmf.sigmffile import (
    SigMFFile,
    dtype_info,
    fromfile,
    get_sigmf_filenames,
)
from zarr.core.array import CompressorLike, ShardsLike

from sigmf_zarr._dtype import logical_sample_dtype
from sigmf_zarr._import_target import (
    open_import_store,
    remove_new_store_on_failure,
)
from sigmf_zarr.export_plan import (
    export_omissions,
    project_item_indexes,
    resolve_projected_indexes,
    validate_sigmf_collection_schema,
    validate_sigmf_schema,
)
from sigmf_zarr.integrity import integrity_is_supported
from sigmf_zarr.json import (
    JSONObject,
    JSONValue,
    json_object,
    json_object_list,
)
from sigmf_zarr.sample_storage import resolve_import_sample_storage
from sigmf_zarr.store import (
    SigMFCollection,
    SigMFRecording,
    SigMFZarrStore,
    ZarrFormat,
)
from sigmf_zarr.store._common import _validate_resource_name
from sigmf_zarr.store._transaction import (
    collection_import_transaction,
    recording_import_transaction,
    retained_backup_directory,
)


def coerce_sigmf_samples(
    data: npt.ArrayLike,
    *,
    num_channels: int = 1,
) -> npt.NDArray[Any]:
    """Normalize SigMF sample buffers into a Zarr-friendly array.

    Args:
        data: SigMF sample buffer.
        num_channels: Number of interleaved standard SigMF channels.

    Returns:
        Sample array with complex values represented by a leading I/Q axis.

    Raises:
        ValueError: If `num_channels` is not positive or the flat sample
            count is not divisible by `num_channels`.
    """
    if num_channels <= 0:
        raise ValueError(f"num_channels must be positive, got {num_channels}")
    array = np.asarray(data)
    if array.size % num_channels != 0:
        raise ValueError(
            "SigMF sample count "
            f"{array.size} is not divisible by core:num_channels "
            f"{num_channels}"
        )

    flat = array.reshape(-1)
    if np.iscomplexobj(array):
        component_dtype = np.asarray(array.real).dtype
        if num_channels > 1:
            # Standard SigMF interleaves channels within time. SigMF-Zarr
            # gives channel and I/Q components explicit axes.
            time_channel = flat.reshape((-1, num_channels))
            stacked = np.stack(
                (
                    time_channel.real.T,
                    time_channel.imag.T,
                ),
                axis=1,
            )
            return np.asarray(stacked, dtype=component_dtype)
        stacked = np.stack((flat.real, flat.imag), axis=0)
        return np.asarray(stacked, dtype=component_dtype)
    if num_channels > 1:
        return flat.reshape((-1, num_channels)).T
    return array


def sigmf_sample_axes(
    data: npt.ArrayLike,
    *,
    num_channels: int = 1,
) -> tuple[str, ...]:
    """Return SigMF-Zarr axes for a standard SigMF sample buffer.

    Args:
        data: SigMF sample buffer.
        num_channels: Number of interleaved standard SigMF channels.

    Returns:
        Sample-axis names for the coerced SigMF-Zarr array.
    """
    array = np.asarray(data)
    if np.iscomplexobj(array):
        if num_channels > 1:
            return ("channel", "iq", "time")
        return ("iq", "time")
    if num_channels > 1:
        return ("channel", "time")
    return ("time",)


def _iter_import_sample_chunks(
    sigmf_file: SigMFFile,
    *,
    chunk_size: int,
) -> Iterator[npt.NDArray[Any]]:
    """Read bounded sample chunks without scaling or narrowing components.

    Args:
        sigmf_file: Open standard SigMF recording.
        chunk_size: Maximum number of time samples per chunk.

    Yields:
        Logical sample tensors with time as the final axis.

    Raises:
        ValueError: If the source has no dataset or ends before its declared
            sample count.
    """
    datatype = sigmf_file.get_global_field("core:datatype")
    info = dtype_info(datatype)
    if sigmf_file.data_file is None:
        raise ValueError("SigMF import requires a dataset file")
    num_channels = int(sigmf_file.num_channels)
    sample_count = int(sigmf_file.sample_count)
    integer_iq = info["is_complex"] and info["is_fixedpoint"]
    storage_dtype = np.dtype(cast(np.dtype[Any], info["memmap_map_type"]))
    with Path(sigmf_file.data_file).open("rb") as handle:
        handle.seek(int(getattr(sigmf_file, "data_offset", 0)))
        for start in range(0, sample_count, chunk_size):
            length = min(chunk_size, sample_count - start)
            count = length * num_channels * (2 if integer_iq else 1)
            raw = np.fromfile(handle, dtype=storage_dtype, count=count)
            if raw.size != count:
                raise ValueError("SigMF dataset ended before its sample count")
            if integer_iq:
                if num_channels > 1:
                    yield raw.reshape(length, num_channels, 2).transpose(
                        1, 2, 0
                    )
                else:
                    yield raw.reshape(length, 2).T
            else:
                yield coerce_sigmf_samples(raw, num_channels=num_channels)


def sigmf_dtype_to_zarr(
    datatype: str,
) -> tuple[np.dtype[Any], tuple[int, ...]]:
    """Map a SigMF datatype string to SigMF-Zarr sample storage info.

    Args:
        datatype: SigMF datatype string.

    Returns:
        Little-endian element dtype for the Zarr array and the per-sample
        shape suffix implied by the SigMF datatype. Complex SigMF datatypes
        are stored as real-valued arrays with an I/Q axis of length 2. The
        byte order of `datatype` applies only to the standard SigMF dataset.
    """
    info = dtype_info(datatype)
    if info["is_complex"]:
        component_dtype = logical_sample_dtype(
            cast(np.dtype[Any], info["component_dtype"])
        )
        return component_dtype, (2,)
    sample_dtype = logical_sample_dtype(
        cast(np.dtype[Any], info["sample_dtype"])
    )
    return sample_dtype, ()


def _validate_sigmf_sample_axes(sample_axes: tuple[str, ...]) -> None:
    """Reject sample coordinates with no standard SigMF representation.

    Args:
        sample_axes: Semantic sample axes.

    Raises:
        ValueError: If an axis is not time, I/Q, or channel.
    """
    unsupported = sorted(set(sample_axes) - {"time", "iq", "channel"})
    if unsupported:
        raise ValueError(
            f"Standard SigMF cannot represent sample axes {unsupported!r}"
        )


def coerce_samples_for_sigmf(
    samples: npt.ArrayLike,
    *,
    sample_axes: tuple[str, ...] | None = None,
) -> npt.NDArray[Any]:
    """Normalize recording samples into an array writable by SigMF.

    Args:
        samples: SigMF-Zarr sample array.
        sample_axes: Optional semantic axes for `samples`.

    Returns:
        Array suitable for writing as standard SigMF sample data.

    Raises:
        ValueError: If the shape or axes have no standard SigMF
            representation, or an `iq` axis holds complex values.
    """
    array = np.asarray(samples)
    if sample_axes is None:
        # A trailing length-2 axis of real values holds I/Q pairs. Complex
        # values are already complex samples.
        if (
            array.ndim >= 1
            and array.shape[-1] == 2
            and not np.iscomplexobj(array)
        ):
            return _complex_from_iq(array[..., 0], array[..., 1])
        return array

    axes = tuple(sample_axes)
    _validate_sigmf_sample_axes(axes)
    _require_real_iq(array.dtype, axes)
    if len(axes) != array.ndim:
        raise ValueError(
            f"sample_axes {axes!r} do not match sample shape {array.shape!r}"
        )
    if "time" not in axes:
        raise ValueError("sample_axes must include a 'time' axis")

    if "iq" in axes:
        iq_axis = axes.index("iq")
        if array.shape[iq_axis] != 2:
            raise ValueError(
                f"Expected 'iq' axis length 2, got {array.shape[iq_axis]}"
            )
        real = np.take(array, 0, axis=iq_axis)
        imag = np.take(array, 1, axis=iq_axis)
        array = _complex_from_iq(real, imag)
        axes = tuple(axis for axis in axes if axis != "iq")

    time_axis = axes.index("time")
    if time_axis != 0:
        array = np.moveaxis(array, time_axis, 0)
    return array


def _require_real_iq(dtype: npt.DTypeLike, sample_axes: Sequence[str]) -> None:
    """Reject complex values in a sample array with an `iq` axis.

    The `iq` axis holds the in-phase and quadrature components as separate
    real planes. A complex plane has no standard SigMF encoding, and
    combining two of them would discard their imaginary parts.

    Args:
        dtype: Sample array dtype.
        sample_axes: Semantic sample axes.

    Raises:
        ValueError: If `sample_axes` has an `iq` axis and `dtype` is complex.
    """
    if "iq" in sample_axes and np.dtype(dtype).kind == "c":
        raise ValueError(
            "A sample array with an 'iq' axis must have a real dtype, not "
            f"{np.dtype(dtype)}"
        )


def _complex_from_iq(
    real: npt.ArrayLike,
    imag: npt.ArrayLike,
) -> npt.NDArray[Any]:
    """Combine separate I/Q components without reducing their precision.

    Args:
        real: In-phase components.
        imag: Quadrature components.

    Returns:
        Complex array whose component precision can represent both inputs.

    Raises:
        ValueError: If the component arrays have different shapes or either
            is complex.
    """
    real_array = np.asarray(real)
    imag_array = np.asarray(imag)
    if np.iscomplexobj(real_array) or np.iscomplexobj(imag_array):
        raise ValueError("I/Q components must be real, not complex")
    if real_array.shape != imag_array.shape:
        raise ValueError(
            "I and Q components must have the same shape, got "
            f"{real_array.shape} and {imag_array.shape}"
        )
    complex_dtype = np.result_type(
        real_array.dtype,
        imag_array.dtype,
        np.complex64,
    )
    # Constructing with real + 1j * imag may promote or narrow intermediates.
    # Assign the components directly into the selected complex dtype.
    result = np.empty(real_array.shape, dtype=complex_dtype)
    result.real = real_array
    result.imag = imag_array
    return result


def _encode_sigmf_samples(
    samples: npt.ArrayLike,
    *,
    sample_axes: tuple[str, ...],
    global_metadata: JSONObject,
) -> npt.NDArray[Any]:
    """Encode samples in the exact component order and dtype SigMF declares.

    Args:
        samples: Logical sample tensor.
        sample_axes: Semantic sample axes.
        global_metadata: Standard SigMF global metadata.

    Returns:
        Contiguous array in standard dataset byte order.

    Raises:
        ValueError: If integer I/Q values cannot be represented by the
            declared datatype, or an `iq` axis holds complex values.
    """
    _validate_sigmf_sample_axes(sample_axes)
    _require_real_iq(np.asarray(samples).dtype, sample_axes)
    datatype = global_metadata.get("core:datatype")
    if not isinstance(datatype, str):
        return np.ascontiguousarray(
            coerce_samples_for_sigmf(samples, sample_axes=sample_axes)
        )
    info = dtype_info(datatype)
    storage_dtype = np.dtype(cast(np.dtype[Any], info["memmap_map_type"]))
    array = np.asarray(samples)
    is_complex = "iq" in sample_axes or np.iscomplexobj(array)
    if bool(info["is_complex"]) != is_complex:
        raise ValueError(
            f"Sample complexity does not match core:datatype {datatype!r}"
        )
    if info["is_complex"] and info["is_fixedpoint"]:
        if "iq" in sample_axes:
            iq_axis = sample_axes.index("iq")
            if array.shape[iq_axis] != 2:
                raise ValueError(
                    "Complex SigMF encoding requires two I/Q components"
                )
            encoded = np.moveaxis(
                array, (sample_axes.index("time"), iq_axis), (0, -1)
            )
        else:
            values = coerce_samples_for_sigmf(array, sample_axes=sample_axes)
            if not np.iscomplexobj(values):
                raise ValueError(
                    "Complex SigMF encoding requires I/Q components"
                )
            encoded = np.stack((values.real, values.imag), axis=-1)
    else:
        encoded = coerce_samples_for_sigmf(samples, sample_axes=sample_axes)
    if storage_dtype.kind in "iu":
        _validate_integer_components(encoded, storage_dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.ascontiguousarray(encoded, dtype=storage_dtype)
        restored = result.astype(encoded.dtype)
    # Compare components separately: complex equal_nan comparison otherwise
    # treats differing finite components as equal when the other is NaN.
    components = (
        ((encoded.real, restored.real), (encoded.imag, restored.imag))
        if np.iscomplexobj(encoded)
        else ((encoded, restored),)
    )
    if any(
        not np.array_equal(original, recovered, equal_nan=True)
        for original, recovered in components
    ):
        raise ValueError(
            f"Samples cannot be represented losslessly by {datatype!r}"
        )
    return result


def _validate_integer_components(
    components: npt.NDArray[Any],
    dtype: np.dtype[Any],
) -> None:
    """Check integer component representability before encoding.

    Args:
        components: Real-valued I/Q components.
        dtype: Declared integer storage dtype.

    Raises:
        ValueError: If values are fractional, non-finite, or out of range.
    """
    if components.size == 0:
        return
    limits = np.iinfo(dtype)
    if (
        components.dtype.kind not in "iuf"
        or not np.all(np.isfinite(components))
        or (
            components.dtype.kind == "f"
            and np.any(components != np.trunc(components))
        )
        or int(components.min()) < limits.min
        or int(components.max()) > limits.max
    ):
        raise ValueError(f"Sample components cannot be represented by {dtype}")


def strip_zarr_metadata(global_metadata: JSONObject) -> JSONObject:
    """Return SigMF global metadata without SigMF-Zarr storage fields.

    Args:
        global_metadata: SigMF global metadata copied from a SigMF-Zarr
            recording.

    Returns:
        A copy of ``global_metadata`` with all ``sigmf-zarr:*`` keys removed
        and the ``sigmf-zarr`` declaration removed from
        ``core:extensions``. If no extension declarations remain,
        ``core:extensions`` is omitted.
    """
    zarr_namespace = SigMFRecording.ZARR_EXTENSION_NAMESPACE
    zarr_prefix = f"{zarr_namespace}:"
    extension_field = "core:extensions"
    stripped: JSONObject = {}

    for key, value in global_metadata.items():
        if key.startswith(zarr_prefix):
            continue

        if key != extension_field:
            stripped[key] = value
            continue

        if not isinstance(value, list):
            stripped[key] = value
            continue

        extensions: list[object] = []
        for extension in value:
            if (
                isinstance(extension, dict)
                and extension.get("name") == zarr_namespace
            ):
                continue
            extensions.append(
                dict(extension) if isinstance(extension, dict) else extension
            )

        if extensions:
            stripped[key] = cast(JSONValue, extensions)

    return stripped


def _export_target(path: str | PathLike[str]) -> Path:
    """Require an export target to name a file.

    Args:
        path: User-provided target path or stem.

    Returns:
        Target as a path.

    Raises:
        ValueError: If `path` is empty, ends with a path separator, or its
            final component is `.` or `..`.
    """
    # Path() drops a trailing separator and a final "." component, so
    # inspect the path as written.
    final = os.fspath(path)
    for separator in (os.sep, os.altsep):
        if separator:
            final = final.rpartition(separator)[2]
    if final in {"", ".", ".."}:
        raise ValueError(
            f"Export target {os.fspath(path)!r} must name a file, not a "
            "directory"
        )
    return Path(path)


def archive_path(path: str | PathLike[str]) -> Path:
    """Normalize a target path to the standard `.sigmf` suffix.

    Args:
        path: User-provided archive target path or stem. A stem may contain
            dots, which are retained.

    Returns:
        Path ending with `.sigmf`.

    Raises:
        ValueError: If `path` is empty, ends with a path separator, or its
            final component is `.` or `..`.
    """
    target = _export_target(path)
    return (
        target
        if target.suffix == ".sigmf"
        else target.with_name(f"{target.name}.sigmf")
    )


def _metadata_path(path: str | PathLike[str]) -> Path:
    """Normalize a recording target to its `.sigmf-meta` path.

    Args:
        path: `.sigmf-meta` path, `.sigmf-data` path, or path stem. A stem
            may contain dots, which are retained.

    Returns:
        Path ending with `.sigmf-meta`.

    Raises:
        ValueError: If `path` is empty, ends with a path separator, or its
            final component is `.` or `..`.
    """
    target = _export_target(path)
    if target.suffix == ".sigmf-meta":
        return target
    if target.suffix == ".sigmf-data":
        return target.with_suffix(".sigmf-meta")
    return target.with_name(f"{target.name}.sigmf-meta")


def _metadata_integer(value: JSONValue, *, field: str, label: str) -> int:
    """Validate one metadata field as an integer.

    Args:
        value: Metadata value to validate.
        field: Metadata field name.
        label: Human-readable metadata entry label.

    Returns:
        Metadata value as an integer.

    Raises:
        ValueError: If `value` is not an integer.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} field {field!r} must be an integer")
    return value


def _sigmf_num_channels(global_metadata: JSONObject) -> int:
    """Return the standard SigMF channel count from global metadata.

    Args:
        global_metadata: SigMF global metadata object.

    Returns:
        Positive channel count. Missing `core:num_channels` defaults to `1`.

    Raises:
        ValueError: If `core:num_channels` is present but is not a positive
            integer.
    """
    field = "core:num_channels"
    if field not in global_metadata:
        return 1
    num_channels = _metadata_integer(
        global_metadata[field],
        field=field,
        label="global metadata",
    )
    if num_channels <= 0:
        raise ValueError(f"global metadata field {field!r} must be positive")
    return num_channels


def _prepare_export_global_metadata(
    recording: SigMFRecording, *, metadata: JSONObject | None = None
) -> JSONObject:
    """Return SigMF global metadata prepared for standard export.

    Args:
        recording: Recording being exported.
        metadata: Optional resolved global metadata for a selected item.

    Returns:
        Stripped global metadata with `core:num_channels` validated against
        the declared channel axis and filled for multi-channel recordings.

    Raises:
        ValueError: If `core:num_channels` is not a positive integer or does
            not match the derived channel count.
    """
    metadata = strip_zarr_metadata(
        recording.global_metadata if metadata is None else metadata
    )
    # Every export writes a conforming pair with matching filename stems.
    metadata.pop("core:dataset", None)
    channel_count = recording.num_channels
    field = "core:num_channels"
    declared_value = metadata.get(field)

    if declared_value is None:
        if channel_count > 1:
            metadata[field] = channel_count
        return metadata

    declared_count = _metadata_integer(
        declared_value,
        field=field,
        label="global metadata",
    )
    if declared_count <= 0:
        raise ValueError(f"global metadata field {field!r} must be positive")
    if declared_count != channel_count:
        raise ValueError(
            f"global metadata field {field!r} is {declared_count}, but "
            f"the recording channel axis has length {channel_count}"
        )
    return metadata


def _validate_sample_span(
    entry: JSONObject,
    *,
    sample_count: int,
    label: str,
    sample_offset: int = 0,
) -> int | None:
    """Validate one metadata entry against a sample count.

    Args:
        entry: Capture or annotation metadata entry.
        sample_count: Number of samples.
        label: Human-readable metadata entry label.
        sample_offset: Absolute index of the first sample.

    Returns:
        End sample for entries with `core:sample_count`, otherwise `None`.

    Raises:
        ValueError: If the entry has an invalid or out-of-bounds sample span.
    """
    sample_start = entry.get("core:sample_start")
    if sample_start is None:
        return None

    start = _metadata_integer(
        sample_start,
        field="core:sample_start",
        label=label,
    )
    sample_end = sample_offset + sample_count
    if start < sample_offset or start > sample_end:
        raise ValueError(
            f"{label} starts at sample {start}, outside sample range "
            f"{sample_offset}..{sample_end}"
        )

    sample_length = entry.get("core:sample_count")
    if sample_length is None:
        return None

    length = _metadata_integer(
        sample_length,
        field="core:sample_count",
        label=label,
    )
    if length < 0:
        raise ValueError(f"{label} has negative core:sample_count {length}")

    end = start + length
    if end > sample_end:
        raise ValueError(
            f"{label} ends at sample {end}, beyond sample end {sample_end}"
        )
    return end


def _validate_export_metadata(
    recording: SigMFRecording, *, metadata: JSONObject | None = None
) -> None:
    """Validate sample-indexed metadata before standard SigMF export.

    Args:
        recording: Recording to validate.
        metadata: Optional projected metadata for a selected item.

    Raises:
        ValueError: If captures or annotations are inconsistent with the
            exported sample count.
    """
    _validate_sample_metadata(
        (
            recording.global_metadata
            if metadata is None
            else json_object(metadata["global"], name="global")
        ),
        (
            recording.captures
            if metadata is None
            else json_object_list(metadata["captures"], name="captures")
        ),
        (
            recording.annotations
            if metadata is None
            else json_object_list(metadata["annotations"], name="annotations")
        ),
        sample_count=recording.sample_count,
        remedy=(
            " Update captures before export or pass force=True to export "
            "anyway."
        ),
    )


def _validate_sample_metadata(
    global_info: JSONObject,
    captures: Sequence[JSONObject],
    annotations: Sequence[JSONObject],
    *,
    sample_count: int,
    remedy: str = "",
) -> None:
    """Validate sample-indexed metadata against a sample count.

    Import and default export both apply this check, so that every imported
    recording can be exported.

    Args:
        global_info: Global metadata with an optional `core:offset`.
        captures: Capture entries.
        annotations: Annotation entries.
        sample_count: Number of samples that the metadata describes.
        remedy: Text appended to the message for a stale capture count.

    Raises:
        ValueError: If a sample index or count is not an integer, a span lies
            outside the samples, or the last capture's `core:sample_count`
            does not end at the last sample.
    """
    sample_offset = _metadata_integer(
        global_info.get("core:offset", 0),
        field="core:offset",
        label="global metadata",
    )
    if sample_offset < 0:
        raise ValueError("core:offset must be nonnegative")
    sample_end = sample_offset + sample_count
    last_capture_end: int | None = None

    for index, capture in enumerate(captures):
        end = _validate_sample_span(
            capture,
            sample_count=sample_count,
            label=f"capture {index}",
            sample_offset=sample_offset,
        )
        if index == len(captures) - 1:
            last_capture_end = end

    for index, annotation in enumerate(annotations):
        _validate_sample_span(
            annotation,
            sample_count=sample_count,
            label=f"annotation {index}",
            sample_offset=sample_offset,
        )

    if last_capture_end is not None and last_capture_end != sample_end:
        raise ValueError(
            "Capture metadata describes "
            f"{last_capture_end - sample_offset} samples, "
            f"but the recording contains {sample_count} samples.{remedy}"
        )


def _sample_chunk_size(
    samples: object,
    *,
    sample_count: int,
    time_axis: int,
) -> int:
    """Return the chunk size to use for streaming sample export.

    Args:
        samples: Zarr-like sample array.
        sample_count: Total number of samples.
        time_axis: Runtime index of the time axis.

    Returns:
        Positive number of samples to write per export chunk.
    """
    chunks = getattr(samples, "chunks", None)
    if isinstance(chunks, tuple) and 0 <= time_axis < len(chunks):
        chunk_size = int(cast(int, chunks[time_axis]))
        if chunk_size > 0:
            return chunk_size
    return max(sample_count, 1)


def _iter_sigmf_data_chunks(
    recording: SigMFRecording,
    *,
    item_index: int | None = None,
    global_metadata: JSONObject | None = None,
) -> Iterator[npt.NDArray[Any]]:
    """Yield contiguous chunks in standard SigMF dataset byte order.

    Args:
        recording: Recording whose samples should be encoded.
        item_index: Optional selected batch item.
        global_metadata: Optional resolved sample encoding metadata.

    Yields:
        Contiguous arrays ready to be written to a ``.sigmf-data`` file.

    Raises:
        ValueError: If a batch has no selected item or has no sample axis.
    """
    if recording.batched and item_index is None:
        raise ValueError(
            "Standard SigMF data encoding only supports unbatched recordings"
        )
    samples = recording.samples
    sample_count = recording.sample_count
    time_axis = recording.axis_index("time")
    sample_axes = recording.sample_axes
    chunk_size = _sample_chunk_size(
        samples,
        sample_count=sample_count,
        time_axis=time_axis,
    )

    for start in range(0, sample_count, chunk_size):
        stop = min(start + chunk_size, sample_count)
        key: list[slice | int] = [slice(None)] * len(samples.shape)
        key[time_axis] = slice(start, stop)
        if item_index is not None:
            key[0] = item_index
        chunk = _encode_sigmf_samples(
            samples[tuple(key)],
            sample_axes=sample_axes,
            global_metadata=(
                recording.global_metadata
                if global_metadata is None
                else global_metadata
            ),
        )
        yield chunk


def calculate_sha512(recording: SigMFRecording) -> str:
    """Calculate ``core:sha512`` without materializing the recording.

    The digest covers the exact byte stream that standard SigMF export writes
    to the ``.sigmf-data`` file. It does not cover Zarr metadata, compressed
    chunks, or a ``.sigmf`` archive container.

    Args:
        recording: Unbatched recording to hash.

    Returns:
        Lowercase SHA-512 hexadecimal digest.

    Raises:
        ValueError: If the recording cannot be encoded as standard SigMF.
    """
    digest = hashlib.sha512()
    for chunk in _iter_sigmf_data_chunks(recording):
        digest.update(chunk.tobytes())
    return digest.hexdigest()


def verify_sha512(recording: SigMFRecording) -> bool:
    """Verify a recording against its declared ``core:sha512`` value.

    Args:
        recording: Unbatched recording to verify.

    Returns:
        True only when a valid-looking declared digest is present and matches
        the encoded standard SigMF dataset bytes.

    Raises:
        ValueError: If the recording cannot be encoded as standard SigMF.
    """
    declared = recording.global_metadata.get(SHA512_KEY)
    if not isinstance(declared, str) or len(declared) != 128:
        return False
    try:
        bytes.fromhex(declared)
    except ValueError:
        return False
    return hmac.compare_digest(declared.lower(), calculate_sha512(recording))


def _write_sigmf_data_file(
    recording: SigMFRecording,
    data_path: Path,
    *,
    item_index: int | None = None,
    global_metadata: JSONObject | None = None,
) -> str:
    """Write recording samples and calculate their SHA-512 by chunk.

    Args:
        recording: Recording whose samples should be exported.
        data_path: Target ``.sigmf-data`` path.
        item_index: Optional selected batch item.
        global_metadata: Optional resolved sample encoding metadata.

    Returns:
        Lowercase SHA-512 hexadecimal digest of the written file.

    Raises:
        ValueError: If the recording cannot be encoded as standard SigMF.
    """
    digest = hashlib.sha512()
    with data_path.open("wb") as handle:
        for chunk in _iter_sigmf_data_chunks(
            recording, item_index=item_index, global_metadata=global_metadata
        ):
            encoded = chunk.tobytes()
            handle.write(encoded)
            digest.update(encoded)
    return digest.hexdigest()


def _check_replaceable_target(path: Path) -> None:
    """Require an existing export target to be a file that may be replaced.

    Args:
        path: Target path that publication may replace.

    Raises:
        ValueError: If `path` exists and is neither a regular file nor a
            symbolic link.
    """
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    # Publication moves the existing target into a backup directory that is
    # deleted after success, so a directory would be removed with it.
    if not (stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
        raise ValueError(
            f"Export target {str(path)!r} exists and is not a regular file "
            "or symbolic link"
        )


def _check_moved_target(backup: Path, target: Path) -> None:
    """Require an entry moved into the backup directory to be a file.

    Another process can create a directory at a target after the check that
    `_check_replaceable_target` makes. Checking the moved entry itself keeps
    the backup cleanup from deleting that directory.

    Args:
        backup: Backup path that now holds the former target entry.
        target: Target path from which the entry was moved.

    Raises:
        ValueError: If the moved entry is neither a regular file nor a
            symbolic link.
    """
    mode = backup.lstat().st_mode
    if not (stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
        raise ValueError(
            f"Export target {str(target)!r} changed during export and is not "
            "a regular file or symbolic link"
        )


def _replace_sigmf_pair(
    *,
    tmp_meta: Path,
    tmp_data: Path,
    meta_path: Path,
    data_path: Path,
) -> None:
    """Replace a SigMF metadata/data pair with rollback on failure.

    All paths must reside on the same filesystem. Existing targets are moved
    into the temporary directory before either new file becomes visible.

    Args:
        tmp_meta: Complete temporary metadata file.
        tmp_data: Complete temporary data file.
        meta_path: Final metadata path.
        data_path: Final data path.

    Raises:
        ValueError: If an existing target became a directory or another
            non-file entry after the initial check. It is restored.
        OSError: If replacement fails. Existing files are restored when
            possible before the error is propagated.
    """
    with retained_backup_directory(meta_path.parent) as backup_dir:
        _publish_sigmf_pair(
            tmp_meta=tmp_meta,
            tmp_data=tmp_data,
            meta_path=meta_path,
            data_path=data_path,
            backup_dir=backup_dir,
        )


def _publish_sigmf_pair(
    *,
    tmp_meta: Path,
    tmp_data: Path,
    meta_path: Path,
    data_path: Path,
    backup_dir: Path,
) -> None:
    """Publish a pair while tracking ownership of each destination file.

    Args:
        tmp_meta: Complete temporary metadata file.
        tmp_data: Complete temporary data file.
        meta_path: Final metadata path.
        data_path: Final data path.
        backup_dir: Recovery directory retained if rollback fails.

    Raises:
        ValueError: If an existing target became a directory or another
            non-file entry after the initial check. It is restored.
        OSError: If backup, publication, or restoration fails.
    """
    backup_meta = backup_dir / ".previous.sigmf-meta"
    backup_data = backup_dir / ".previous.sigmf-data"
    had_meta = os.path.lexists(meta_path)
    had_data = os.path.lexists(data_path)
    moved_meta = False
    moved_data = False
    installed_meta = False
    installed_data = False
    try:
        # Publish data before metadata so a visible new metadata file never
        # points at an old data file during a successful replacement.
        if had_meta:
            meta_path.replace(backup_meta)
            moved_meta = True
            _check_moved_target(backup_meta, meta_path)
        if had_data:
            data_path.replace(backup_data)
            moved_data = True
            _check_moved_target(backup_data, data_path)
        tmp_data.replace(data_path)
        installed_data = True
        tmp_meta.replace(meta_path)
        installed_meta = True
    except BaseException:
        # Roll back after any exception, including KeyboardInterrupt.
        if installed_meta:
            meta_path.unlink(missing_ok=True)
        if installed_data:
            data_path.unlink(missing_ok=True)
        if moved_data:
            backup_data.replace(data_path)
        if moved_meta:
            backup_meta.replace(meta_path)
        raise


# Errors by which os.link reports a file system without hard links.
_NO_HARD_LINK_ERRNOS = frozenset(
    {errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS}
)


def _publish_new_file(source: Path, target: Path) -> None:
    """Publish a complete file at a path that no entry occupies.

    Unlike a rename, publication never replaces an entry created after an
    earlier existence check, including a dangling symbolic link.

    Args:
        source: Complete file on the same file system as `target`.
        target: Final path.

    Raises:
        FileExistsError: If an entry exists at `target`.
        OSError: If publication fails for another reason.
    """
    try:
        os.link(source, target)
    except OSError as exc:
        if exc.errno not in _NO_HARD_LINK_ERRNOS:
            raise
        # Without hard links, claim the name exclusively and then replace
        # only that empty placeholder.
        os.close(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
        source.replace(target)


def _publish_new_sigmf_pair(
    *,
    tmp_meta: Path,
    tmp_data: Path,
    meta_path: Path,
    data_path: Path,
) -> None:
    """Publish a SigMF metadata/data pair at paths that must not exist.

    Args:
        tmp_meta: Complete temporary metadata file.
        tmp_data: Complete temporary data file.
        meta_path: Final metadata path.
        data_path: Final data path.

    Raises:
        FileExistsError: If an entry exists at either final path. A data file
            published by this call is removed again.
        OSError: If publication fails for another reason.
    """
    data_identity = tmp_data.stat()
    # Publish data before metadata, as replacement does.
    _publish_new_file(tmp_data, data_path)
    try:
        _publish_new_file(tmp_meta, meta_path)
    except BaseException:
        # Remove the data file only while it is still the one published here.
        try:
            if os.path.samestat(data_path.lstat(), data_identity):
                data_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _validate_import_metadata(metadata: JSONObject) -> None:
    """Reject input structures that the sample-only importer cannot retain.

    Args:
        metadata: Original standard SigMF metadata document.

    Raises:
        ValueError: If conversion would omit metadata or non-sample bytes,
            or the source uses the reserved `sigmf-zarr` namespace.
    """
    extra = sorted(set(metadata) - {"global", "captures", "annotations"})
    if extra:
        raise ValueError(f"Unsupported SigMF top-level metadata: {extra}")
    global_info = json_object(metadata.get("global"), name="global")
    _validate_import_zarr_namespace(global_info)
    captures = json_object_list(metadata.get("captures"), name="captures")
    dataset = global_info.get("core:dataset")
    if dataset is not None and (
        not isinstance(dataset, str)
        or not dataset
        or Path(dataset).name != dataset
        or "\\" in dataset
        or dataset in {".", ".."}
    ):
        raise ValueError("core:dataset must name a file in the same directory")
    if global_info.get("core:metadata_only"):
        raise ValueError("Metadata-only SigMF import is not supported")
    if global_info.get("core:trailing_bytes") or any(
        capture.get("core:header_bytes") for capture in captures
    ):
        raise ValueError(
            "SigMF datasets with header or trailing bytes are not supported"
        )


def _validate_import_zarr_namespace(global_info: JSONObject) -> None:
    """Reject source fields that the store would replace on import.

    Args:
        global_info: Source global metadata.

    Raises:
        ValueError: If the source has `sigmf-zarr:*` global fields or a
            `sigmf-zarr` extension declaration other than the canonical one.
    """
    namespace = SigMFRecording.ZARR_EXTENSION_NAMESPACE
    reserved = sorted(
        key for key in global_info if key.startswith(f"{namespace}:")
    )
    if reserved:
        raise ValueError(
            f"SigMF import does not accept reserved {namespace!r} global "
            f"fields: {reserved}"
        )
    extensions = global_info.get("core:extensions")
    if not isinstance(extensions, list):
        return
    declarations = [
        entry
        for entry in extensions
        if isinstance(entry, dict) and entry.get("name") == namespace
    ]
    canonical = SigMFRecording._zarr_extension_declaration()
    # Compare serialized JSON so that `1` does not match `true`.
    expected = json.dumps(canonical, sort_keys=True)
    if len(declarations) > 1 or any(
        json.dumps(entry, sort_keys=True) != expected for entry in declarations
    ):
        raise ValueError(
            f"SigMF import accepts only the canonical {namespace!r} "
            f"extension declaration {canonical!r}"
        )


def _validate_import_source(metadata: JSONObject) -> None:
    """Reject a source document that import cannot store and export again.

    Args:
        metadata: Original standard SigMF metadata document.

    Raises:
        ValueError: If `_validate_import_metadata` rejects the document, it
            violates the SigMF schema that export applies, or a capture or
            annotation has a reserved `sigmf-zarr:*` field.
    """
    _validate_import_metadata(metadata)
    prefix = f"{SigMFRecording.ZARR_EXTENSION_NAMESPACE}:"
    for section in ("captures", "annotations"):
        # The schema check below reports a missing section.
        entries = json_object_list(metadata.get(section, []), name=section)
        for index, entry in enumerate(entries):
            reserved = sorted(key for key in entry if key.startswith(prefix))
            if reserved:
                raise ValueError(
                    f"SigMF import does not accept reserved {prefix!r} "
                    f"fields in {section} entry {index}: {reserved}"
                )
    validate_sigmf_schema(metadata)


def _load_json_object(path: Path, *, name: str) -> JSONObject:
    """Read one JSON object from a metadata file.

    Args:
        path: Metadata file path.
        name: Document name for error messages.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If the file is not UTF-8 JSON text, nests too deeply to
            parse, or does not hold a valid JSON object.
    """
    try:
        with path.open(encoding="utf-8") as handle:
            return json_object(json.load(handle), name=name)
    except RecursionError as exc:
        raise ValueError(f"{name} in {str(path)!r} nests too deeply") from exc


def _import_data_path(meta_path: Path, global_metadata: JSONObject) -> Path:
    """Locate and check the dataset file of a standard SigMF recording.

    A conforming dataset file shares the stem of `meta_path`, which the
    caller named, so a symbolic link to it is followed. A non-conforming
    `core:dataset` value names a file that the caller did not name, so it
    must be a regular file and not a symbolic link.

    Args:
        meta_path: Source `.sigmf-meta` path.
        global_metadata: Source global metadata with a validated
            `core:dataset` value, if any.

    Returns:
        Dataset path. It may be missing, which sigmf-python reports.

    Raises:
        ValueError: If the dataset exists and is not a regular file, or a
            `core:dataset` file is a symbolic link.
    """
    dataset = global_metadata.get("core:dataset")
    if isinstance(dataset, str):
        data_path = meta_path.parent / dataset
        description = "core:dataset file"
        status = os.lstat
    else:
        data_path = meta_path.parent / (
            meta_path.name.removesuffix(".sigmf-meta") + ".sigmf-data"
        )
        description = "SigMF dataset"
        status = os.stat
    try:
        mode = status(data_path).st_mode
    except FileNotFoundError:
        return data_path
    if not stat.S_ISREG(mode):
        raise ValueError(
            f"{description} {str(data_path)!r} is not a regular file"
        )
    return data_path


def _import_sample_count(data_path: Path, sample_bytes: int) -> int | None:
    """Count the samples of a source dataset, which must hold at least one.

    Args:
        data_path: Source dataset path.
        sample_bytes: Dataset bytes per sample across all channels.

    Returns:
        Number of complete samples, or `None` if the dataset is missing.

    Raises:
        ValueError: If the dataset exists and is shorter than one sample.
    """
    try:
        size = data_path.stat().st_size
    except FileNotFoundError:
        # sigmf-python reports the missing dataset.
        return None
    if size < sample_bytes:
        raise ValueError(
            f"SigMF dataset {str(data_path)!r} holds no complete sample. "
            "SigMF-Zarr recordings need at least one sample"
        )
    return size // sample_bytes


@dataclass(frozen=True)
class _SigMFSource:
    """Standard SigMF recording that import has checked but not stored."""

    name: str
    """Target recording name."""

    meta_path: Path
    """Source `.sigmf-meta` path."""

    data_path: Path
    """Source dataset path."""

    sigmf_file: SigMFFile
    """Open reader for the source samples."""

    global_metadata: JSONObject
    """Source global metadata as written."""

    captures: list[JSONObject]
    """Source captures as written."""

    annotations: list[JSONObject]
    """Source annotations as written."""

    sample_dtype: np.dtype[Any]
    """Stored sample dtype."""

    sample_shape: tuple[int, ...]
    """Stored sample-array shape."""

    sample_axes: tuple[str, ...]
    """Stored sample axes."""

    chunk_size: int
    """Time samples per conversion buffer."""

    sha512: str
    """SHA-512 of the standard dataset bytes that conversion reproduces."""

    def sample_storage(
        self,
        zarr_format: ZarrFormat,
        *,
        sample_chunks: tuple[int, ...] | None,
        sample_shards: ShardsLike | None,
        automatic_sharding: bool,
    ) -> tuple[tuple[int, ...], ShardsLike | None]:
        """Resolve the sample-array storage layout of this recording.

        Args:
            zarr_format: Physical Zarr format of the target store.
            sample_chunks: Optional sample-array chunk shape.
            sample_shards: Optional sample-array shard shape.
            automatic_sharding: Whether to derive format-3 shards when
                explicit shards are not supplied.

        Returns:
            Chunk shape and optional shard shape.

        Raises:
            ValueError: If an explicit shape is invalid or sharding is
                requested for Zarr format 2.
        """
        return resolve_import_sample_storage(
            self.sample_dtype,
            self.sample_shape,
            1,
            batched=False,
            zarr_format=zarr_format,
            sample_chunks=sample_chunks,
            sample_shards=sample_shards,
            automatic_sharding=automatic_sharding,
        )


def _read_sigmf_source(
    source_path: str | PathLike[str], recording_name: str | None
) -> _SigMFSource:
    """Check a standard SigMF recording and verify its conversion.

    The check reads every sample once, without changing any store.

    Args:
        source_path: Source `.sigmf-meta` path, `.sigmf-data` path, or path
            stem.
        recording_name: Optional recording name override.

    Returns:
        Checked source.

    Raises:
        ValueError: If the recording name or the source is invalid or
            unsupported, default export would reject the metadata, or the
            samples cannot reproduce the source bytes or `core:sha512`.
    """
    filenames = get_sigmf_filenames(str(source_path))
    target_name = (
        Path(filenames["base_fn"]).name
        if recording_name is None
        else recording_name
    )
    _validate_resource_name(target_name, kind="Recording")
    # sigmf-python inserts defaults and rewrites core:version while reading.
    # Keep source metadata separate from its normalized sample reader.
    meta_path = Path(filenames["meta_fn"])
    metadata = _load_json_object(meta_path, name="SigMF metadata")
    _validate_import_source(metadata)
    global_metadata = json_object(metadata.get("global"), name="global")
    captures = json_object_list(metadata.get("captures"), name="captures")
    annotations = json_object_list(
        metadata.get("annotations"), name="annotations"
    )
    num_channels = _sigmf_num_channels(global_metadata)
    datatype = global_metadata.get("core:datatype")
    if not isinstance(datatype, str):
        raise ValueError("SigMF global core:datatype must be a string")
    data_path = _import_data_path(meta_path, global_metadata)
    try:
        sample_dtype, component_shape = sigmf_dtype_to_zarr(datatype)
        bytes_per_time_sample = (
            sample_dtype.itemsize
            * num_channels
            * (2 if component_shape else 1)
        )
        sample_count = _import_sample_count(data_path, bytes_per_time_sample)
        if sample_count is not None:
            # Apply the checks of default export, so that the recording can
            # be exported again.
            _validate_sample_metadata(
                global_metadata,
                captures,
                annotations,
                sample_count=sample_count,
            )
        # The core:sha512 check below verifies the converted byte stream, so
        # sigmf-python's own full read of the dataset is redundant.
        sigmf_file = fromfile(str(meta_path), skip_checksum=True)
    except SigMFError as exc:
        raise ValueError(f"Invalid SigMF recording: {exc}") from exc
    sample_count = int(sigmf_file.sample_count)
    sample_shape = (
        *((num_channels,) if num_channels > 1 else ()),
        *component_shape,
        sample_count,
    )
    sample_axes = (
        *(("channel",) if num_channels > 1 else ()),
        *(("iq",) if component_shape else ()),
        "time",
    )
    # Limit conversion buffers to approximately eight MiB of source samples.
    chunk_size = max(1, (8 * 1024 * 1024) // bytes_per_time_sample)

    # Verify representability before changing the target store. Conversion of
    # a source datatype must reproduce the exact standard SigMF byte stream.
    source_sha512 = sigmf_file.get_global_field(SHA512_KEY)
    digest = hashlib.sha512()
    for samples in _iter_import_sample_chunks(
        sigmf_file, chunk_size=chunk_size
    ):
        digest.update(
            _encode_sigmf_samples(
                samples,
                sample_axes=sample_axes,
                global_metadata=global_metadata,
            ).tobytes()
        )
    imported_sha512 = digest.hexdigest()
    if isinstance(source_sha512, str) and not hmac.compare_digest(
        source_sha512.lower(), imported_sha512
    ):
        raise ValueError(
            "Imported samples do not reproduce the source core:sha512. "
            "The declared hash is stale, or the source datatype or byte "
            "layout cannot be represented losslessly"
        )
    return _SigMFSource(
        name=target_name,
        meta_path=meta_path,
        data_path=data_path,
        sigmf_file=sigmf_file,
        global_metadata=global_metadata,
        captures=captures,
        annotations=annotations,
        sample_dtype=sample_dtype,
        sample_shape=sample_shape,
        sample_axes=sample_axes,
        chunk_size=chunk_size,
        sha512=imported_sha512,
    )


def _write_sigmf_recording(
    store: SigMFZarrStore,
    source: _SigMFSource,
    *,
    sample_storage: tuple[tuple[int, ...], ShardsLike | None],
    sample_compressor: CompressorLike,
    overwrite_recording: bool,
) -> SigMFRecording:
    """Store a checked SigMF recording inside an import transaction.

    Args:
        store: Target store.
        source: Checked source recording.
        sample_storage: Resolved chunk shape and optional shard shape.
        sample_compressor: Optional sample-array compressor.
        overwrite_recording: Whether to replace an existing recording.

    Returns:
        Imported recording with current hashes.

    Raises:
        ValueError: If the source samples changed after the check.
    """
    sample_chunks, sample_shards = sample_storage
    recording = store.recordings.create(
        source.name,
        batched=False,
        sample_dtype=source.sample_dtype,
        sample_shape=source.sample_shape,
        sample_axes=source.sample_axes,
        global_metadata=source.global_metadata,
        captures=source.captures,
        annotations=source.annotations,
        sample_chunks=sample_chunks,
        sample_shards=sample_shards,
        sample_compressor=sample_compressor,
        overwrite=overwrite_recording,
    )
    written_digest = hashlib.sha512()
    with recording.mutate_samples() as destination:
        start = 0
        for samples in _iter_import_sample_chunks(
            source.sigmf_file, chunk_size=source.chunk_size
        ):
            stop = start + samples.shape[-1]
            destination[..., start:stop] = samples
            written_digest.update(
                _encode_sigmf_samples(
                    samples,
                    sample_axes=source.sample_axes,
                    global_metadata=source.global_metadata,
                ).tobytes()
            )
            start = stop
    if not hmac.compare_digest(source.sha512, written_digest.hexdigest()):
        raise ValueError("SigMF source data changed during import")
    recording.set_global_field(SHA512_KEY, source.sha512)
    recording.update_integrity()
    return recording


def _check_export_loss(
    recording: SigMFRecording,
    *,
    allow_lossy: bool,
    project_indexes: Sequence[str] = (),
) -> None:
    """Require explicit permission to omit native metadata structures.

    Args:
        recording: Source recording.
        allow_lossy: Whether omissions may be reported as warnings.
        project_indexes: Explicit item indexes preserved in extension JSON.

    Raises:
        ValueError: If export would discard metadata without permission.
    """
    omitted = export_omissions(recording, project_indexes)
    if not omitted:
        return
    message = "Standard SigMF export omits " + ", ".join(omitted)
    if not allow_lossy:
        raise ValueError(message + ". Pass allow_lossy=True to omit them")
    warnings.warn(message, UserWarning, stacklevel=3)


@remove_new_store_on_failure
def import_sigmf(
    store_path: str | PathLike[str],
    source_path: str | PathLike[str],
    *,
    recording_name: str | None = None,
    overwrite_store: bool = False,
    overwrite_recording: bool = False,
    sample_chunks: tuple[int, ...] | None = None,
    sample_shards: ShardsLike | None = None,
    automatic_sharding: bool = True,
    sample_compressor: CompressorLike = "auto",
    zarr_format: ZarrFormat | None = None,
) -> SigMFRecording:
    """Import one standard SigMF recording into a SigMF-Zarr store.

    Args:
        store_path: Target SigMF-Zarr store path.
        source_path: Source `.sigmf-meta` path, `.sigmf-data` path, or path
            stem.
        recording_name: Optional recording name override. The default is
            the source file name without its SigMF suffix.
        overwrite_store: Whether to replace an existing SigMF-Zarr store at
            `store_path`. The target must be absent, an empty directory, or
            a SigMF-Zarr store, and must not contain the source files.
        overwrite_recording: Whether to replace an existing recording.
        sample_chunks: Optional sample-array chunk shape.
        sample_shards: Optional sample-array shard shape.
        automatic_sharding: Whether to derive format-3 shards when explicit
            shards are not supplied.
        sample_compressor: Optional sample-array compressor.
        zarr_format: Optional physical Zarr format requirement. Existing
            stores are auto-detected when omitted. New stores default to Zarr
            format 3.

    Returns:
        Imported recording wrapper.

    Raises:
        ValueError: If the recording name is invalid, the source metadata or
            datatype is invalid or unsupported, the source metadata violates
            the SigMF schema or the checks of default export, a
            `core:dataset` file is a symbolic link or the dataset is not a
            regular file, the dataset holds no complete sample, the source
            bytes do not match `core:sha512` or cannot be represented
            losslessly, a sample chunk or shard shape is invalid,
            `overwrite_store` names a target that it may not replace, the
            recording exists without overwrite permission, or replacement
            uses a backend other than a local directory store.
    """
    source = _read_sigmf_source(source_path, recording_name)
    store, sample_storage = open_import_store(
        store_path,
        overwrite_store=overwrite_store,
        zarr_format=zarr_format,
        sources=(source.meta_path, source.data_path),
        resolve_storage=lambda target_format: source.sample_storage(
            target_format,
            sample_chunks=sample_chunks,
            sample_shards=sample_shards,
            automatic_sharding=automatic_sharding,
        ),
    )
    with recording_import_transaction(
        store, source.name, overwrite=overwrite_recording
    ):
        recording = _write_sigmf_recording(
            store,
            source,
            sample_storage=sample_storage,
            sample_compressor=sample_compressor,
            overwrite_recording=overwrite_recording,
        )
        store._update_metadata_integrity_when_complete()
    return recording


def _project_item_metadata(
    recording: SigMFRecording, item_index: int
) -> JSONObject:
    """Resolve one independent item's captures into absolute sample indices.

    Args:
        recording: Batched source recording.
        item_index: Nonnegative item position.

    Returns:
        Detached metadata with sorted captures and annotations. Local captures
        win at duplicate starts. Shared acquisition fields supply defaults,
        but timestamps never carry into local captures or subsequent items.

    Raises:
        ValueError: If capture or annotation coordinates are malformed.
    """
    return recording.signal(item_index).metadata


def _validate_export_integrity(
    recording: SigMFRecording, *, samples: bool = True
) -> None:
    """Check declared internal hashes before planning or publishing samples.

    Args:
        recording: Source recording.
        samples: Whether to verify the whole-recording sample hash. Exports
            of one batch item skip it, because the hash covers every item.
            Chunk CRC32C checks the selected item's reads, and export
            computes `core:sha512` from the emitted bytes.

    Raises:
        ValueError: If integrity metadata is unsupported or does not match.
    """
    if recording.integrity and not integrity_is_supported(recording.integrity):
        raise ValueError(
            "Recording uses an unsupported internal integrity algorithm or "
            "version"
        )
    if (
        samples
        and recording.sample_sha512 is not None
        and recording.sample_sha512 != recording.calculate_sample_sha512()
    ):
        raise ValueError(
            "Recording internal sample SHA-512 does not match current data"
        )
    if (
        recording.metadata_sha512 is not None
        and recording.metadata_sha512 != recording.calculate_metadata_sha512()
    ):
        raise ValueError(
            "Recording internal metadata SHA-512 does not match current "
            "metadata"
        )


def _export_metadata(
    recording: SigMFRecording,
    *,
    item_index: int | None,
    force: bool,
    project_indexes: Sequence[str],
) -> JSONObject:
    """Prepare the same metadata for preflight and actual export.

    Args:
        recording: Source recording.
        item_index: Optional selected batch item.
        force: Whether stale sample spans are permitted.
        project_indexes: Explicit indexes projected to extension JSON.

    Returns:
        Detached metadata ready for serialization.

    Raises:
        ValueError: If selection, layout, or metadata is incompatible.
    """
    _validate_sigmf_sample_axes(recording.sample_axes)
    _require_real_iq(recording.sample_dtype, recording.sample_axes)
    if recording.batched and item_index is None:
        raise ValueError("Batched export requires item_index")
    if item_index is not None and (
        not recording.batched
        or type(item_index) is not int
        or not 0 <= item_index < len(recording)
    ):
        raise ValueError("item_index must select an existing batch item")
    projected = (
        None
        if item_index is None
        else _project_item_metadata(recording, item_index)
    )
    if not force:
        _validate_export_metadata(recording, metadata=projected)
    global_metadata = _prepare_export_global_metadata(
        recording,
        metadata=(
            None
            if projected is None
            else json_object(projected["global"], name="global")
        ),
    )
    metadata: dict[str, object] = {
        "global": global_metadata,
        "captures": recording.captures
        if projected is None
        else projected["captures"],
        "annotations": recording.annotations
        if projected is None
        else projected["annotations"],
    }

    _validate_import_metadata(json_object(metadata, name="export metadata"))
    if not isinstance(global_metadata.get("core:datatype"), str):
        raise ValueError("Export requires a declared core:datatype")
    return project_item_indexes(
        recording,
        json_object(metadata, name="export metadata"),
        project_indexes,
        item_index,
    )


def export_sigmf(
    store: SigMFZarrStore,
    recording_name: str,
    output_path: str | PathLike[str],
    *,
    overwrite: bool = False,
    pretty: bool = True,
    force: bool = False,
    allow_lossy: bool = False,
    item_index: int | None = None,
    project_indexes: Sequence[str] = (),
) -> Path:
    """Export one recording from a SigMF-Zarr store to standard SigMF files.

    Args:
        store: Source SigMF-Zarr store.
        recording_name: Recording name to export.
        output_path: Target `.sigmf-meta` path or path stem. Dots in a stem
            are retained.
        overwrite: Whether to replace existing target files.
        pretty: Whether to pretty-print metadata JSON.
        force: Whether to export even when sample-indexed metadata appears
            stale. This does not permit metadata loss.
        allow_lossy: Whether to warn and omit native indexes, extension
            arrays, and per-channel metadata. Defaults to rejection.
        item_index: Nonnegative item position to export from a batch. Each
            selected item becomes a separate standard SigMF recording.
        project_indexes: Explicit item indexes retained with their descriptors
            in the sigmf-zarr-indexes extension.

    Returns:
        Written `.sigmf-meta` path.

    Raises:
        KeyError: If the recording or a projected index does not exist.
        ValueError: If a batch has no valid selected item, the target is
            empty or a directory name, target files or symbolic links exist
            without overwrite permission, a target is neither a regular file
            nor a symbolic link, metadata cannot be preserved or violates the
            SigMF schema, sample coordinates are invalid, or a declared hash
            is stale.
            Item exports check the recording metadata hash but not the
            whole-recording sample hash.
    """
    recording = store.recordings.open(recording_name)
    resolve_projected_indexes(recording, project_indexes)
    if recording.batched and item_index is None:
        raise ValueError(
            "Standard SigMF export only supports unbatched recordings "
            "unless item_index is selected"
        )
    if item_index is not None and (
        not recording.batched
        or type(item_index) is not int
        or not 0 <= item_index < len(recording)
    ):
        raise ValueError("item_index must select an existing batch item")
    _check_export_loss(
        recording,
        allow_lossy=allow_lossy,
        project_indexes=project_indexes,
    )
    _validate_export_integrity(recording, samples=item_index is None)
    metadata = _export_metadata(
        recording,
        item_index=item_index,
        force=force,
        project_indexes=project_indexes,
    )
    global_metadata = json_object(metadata["global"], name="global")
    sigmf_file = SigMFFile(metadata=deepcopy(metadata))
    if "core:version" in global_metadata:
        sigmf_file.set_global_field(
            "core:version", global_metadata["core:version"]
        )
    validate_sigmf_schema(metadata)
    meta_path = _metadata_path(output_path)
    data_path = meta_path.with_suffix(".sigmf-data")

    exists_message = (
        "Target SigMF files already exist. Pass overwrite=True to replace them"
    )
    # A dangling symbolic link is an existing target.
    if not overwrite and (
        os.path.lexists(meta_path) or os.path.lexists(data_path)
    ):
        raise ValueError(exists_message)
    for target in (meta_path, data_path):
        _check_replaceable_target(target)

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=meta_path.parent,
        prefix=".sigmf-zarr-export-",
    ) as tmpdir:
        tmp_path = Path(tmpdir)
        tmp_meta = tmp_path / meta_path.name
        tmp_data = tmp_path / data_path.name
        exported_sha512 = (
            _write_sigmf_data_file(recording, tmp_data)
            if item_index is None
            else _write_sigmf_data_file(
                recording,
                tmp_data,
                item_index=item_index,
                global_metadata=global_metadata,
            )
        )
        declared_sha512 = global_metadata.get(SHA512_KEY)
        if isinstance(declared_sha512, str) and not hmac.compare_digest(
            declared_sha512.lower(), exported_sha512
        ):
            raise ValueError(
                "Recording core:sha512 does not match the exported sample "
                "data. Recalculate it or remove the stale field before export"
            )
        sigmf_file.set_data_file(data_file=tmp_data, skip_checksum=True)
        sigmf_file.set_global_field(SHA512_KEY, exported_sha512)
        sigmf_file.tofile(str(tmp_meta), pretty=pretty, overwrite=True)
        if overwrite:
            _replace_sigmf_pair(
                tmp_meta=tmp_meta,
                tmp_data=tmp_data,
                meta_path=meta_path,
                data_path=data_path,
            )
        else:
            try:
                _publish_new_sigmf_pair(
                    tmp_meta=tmp_meta,
                    tmp_data=tmp_data,
                    meta_path=meta_path,
                    data_path=data_path,
                )
            except FileExistsError as exc:
                raise ValueError(exists_message) from exc
    return meta_path


def _archive_collection_recording_ids(
    collection: JSONObject,
    meta_files: list[Path],
) -> tuple[str, ...]:
    """Resolve collection identifiers and verify their exact metadata files.

    Args:
        collection: Standard collection object.
        meta_files: Metadata files extracted from the source archive.

    Returns:
        Distinct recording identifiers in collection order.

    Raises:
        ValueError: If `core:streams` is not a list of objects, a stream has
            unsupported fields, a name with path components, a missing
            metadata file, or a differing metadata hash, or two streams name
            the same recording.
    """
    by_name = {
        path.name.removesuffix(".sigmf-meta"): path for path in meta_files
    }
    recording_ids: list[str] = []
    streams = json_object_list(
        collection.get("core:streams", []),
        name="collection core:streams",
    )
    for stream in streams:
        if set(stream) - {"name", "hash"}:
            raise ValueError("Unsupported metadata in collection core:streams")
        name = stream.get("name")
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or "\\" in name
            or name in {".", ".."}
        ):
            raise ValueError(
                "Collection stream names must be single path components, "
                f"not {name!r}"
            )
        if name not in by_name:
            name = name.removesuffix(".sigmf-meta")
        if name not in by_name:
            raise ValueError(f"Collection stream {name!r} is missing")
        if name in recording_ids:
            raise ValueError(
                f"Collection streams name recording {name!r} more than once"
            )
        with by_name[name].open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha512").hexdigest()
        declared = stream.get("hash")
        if not isinstance(declared, str) or not hmac.compare_digest(
            declared.lower(), digest
        ):
            raise ValueError(
                f"Collection stream {name!r} metadata hash does not match"
            )
        recording_ids.append(name)
    return tuple(recording_ids)


def _read_archive_collection(
    collection_file: Path, meta_files: list[Path]
) -> tuple[str, JSONObject, tuple[str, ...]]:
    """Read and check the collection of a SigMF archive.

    Args:
        collection_file: Extracted `.sigmf-collection` file.
        meta_files: Metadata files extracted from the source archive.

    Returns:
        Collection name, collection metadata without `core:streams`, and
        recording identifiers in collection order.

    Raises:
        ValueError: If the document is not a JSON object with only a
            `collection` object, the collection violates the SigMF
            collection schema, or `_archive_collection_recording_ids`
            rejects its streams.
    """
    metadata = _load_json_object(
        collection_file, name="SigMF collection metadata"
    )
    extra = sorted(set(metadata) - {"collection"})
    if extra:
        raise ValueError(
            f"Unsupported SigMF collection top-level metadata: {extra}"
        )
    collection_info = json_object(
        metadata.get("collection"), name="SigMF collection"
    )
    validate_sigmf_collection_schema(collection_info)
    recording_ids = _archive_collection_recording_ids(
        collection_info, meta_files
    )
    collection_metadata = dict(collection_info)
    collection_metadata.pop("core:streams", None)
    name = collection_file.name.removesuffix(".sigmf-collection")
    return name, collection_metadata, recording_ids


def _check_archive_files(
    root: Path,
    sources: Sequence[_SigMFSource],
    collection_files: list[Path],
) -> None:
    """Reject archive files that import would otherwise discard.

    Args:
        root: Extracted archive root.
        sources: Checked recordings.
        collection_files: Collection metadata files.

    Raises:
        ValueError: If the archive has a file other than recording metadata,
            recording datasets, and collection metadata.
    """
    retained = set(collection_files)
    for source in sources:
        retained.update((source.meta_path, source.data_path))
    extra = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path not in retained
    )
    if extra:
        raise ValueError(f"Unsupported auxiliary SigMF archive files: {extra}")


# Leading bytes of compressed streams that tarfile would otherwise expand.
_COMPRESSION_SIGNATURES = (
    ("gzip", b"\x1f\x8b"),
    ("bzip2", b"BZh"),
    ("xz", b"\xfd7zXZ\x00"),
    ("Zstandard", b"\x28\xb5\x2f\xfd"),
)

# File suffixes that name SigMF files, which a directory must not use.
_SIGMF_FILE_SUFFIXES = (".sigmf-meta", ".sigmf-data", ".sigmf-collection")

# Errors that a malformed header, member name, or member layout causes,
# rather than the file system that holds or receives the files.
_ARCHIVE_LAYOUT_ERRNOS = frozenset(
    {
        errno.EEXIST,
        errno.EILSEQ,
        errno.EINVAL,
        errno.EISDIR,
        errno.ENAMETOOLONG,
        errno.ENOTDIR,
    }
)


def _archive_compression(handle: BinaryIO) -> str | None:
    """Name the compression format of a stream that is not a tar file.

    Args:
        handle: Seekable binary archive stream.

    Returns:
        Compression format name, or None when no known signature matches.
    """
    handle.seek(0)
    header = handle.read(6)
    for name, signature in _COMPRESSION_SIGNATURES:
        if header.startswith(signature):
            return name
    return None


def _read_archive_members(
    archive: tarfile.TarFile, archive_size: int
) -> list[tarfile.TarInfo]:
    """Read every member header with a bound that does not trust `tarfile`.

    Some CPython releases loop without bound on a header whose negative size
    points back at itself (CVE-2025-8194). Each header occupies at least one
    512-byte block of the archive, so header offsets must increase and the
    archive holds at most `archive_size // 512` members.

    Args:
        archive: Archive opened for reading.
        archive_size: Size of the archive file in bytes.

    Returns:
        Every member in archive order.

    Raises:
        ValueError: If a header offset does not increase or the archive
            declares more members than its size allows.
    """
    limit = archive_size // tarfile.BLOCKSIZE
    members: list[tarfile.TarInfo] = []
    previous_offset = -1
    while (member := archive.next()) is not None:
        if member.offset <= previous_offset:
            raise ValueError(
                f"SigMF archive member {member.name!r} does not follow the "
                "previous member"
            )
        previous_offset = member.offset
        members.append(member)
        if len(members) > limit:
            raise ValueError(
                f"SigMF archive declares more members than its {archive_size} "
                "bytes can hold"
            )
    return members


def _archive_member_path(member: tarfile.TarInfo) -> PurePosixPath:
    """Check the type, size, and name of one archive member.

    Args:
        member: Member header.

    Returns:
        Member name as a path without `.` components.

    Raises:
        ValueError: If the member is sparse, is neither a regular file nor a
            directory, has a negative size, has an absolute name, a `..`
            component, or a NUL character, or names the archive root as a
            regular file.
    """
    name = member.name
    if member.issparse():
        raise ValueError(f"SigMF archive member {name!r} is a sparse file")
    if not (member.isreg() or member.isdir()):
        raise ValueError(
            f"SigMF archive member {name!r} is not a regular file or directory"
        )
    if member.size < 0:
        raise ValueError(f"SigMF archive member {name!r} has a negative size")
    if "\x00" in name:
        raise ValueError(f"SigMF archive member {name!r} has a NUL character")
    # PurePosixPath drops "." components, so "./a" and "a" name one entry.
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"SigMF archive member {name!r} is outside the archive root"
        )
    if member.isreg() and not path.parts:
        raise ValueError(
            f"SigMF archive member {name!r} names the archive root as a file"
        )
    return path


def _check_archive_members(
    members: Sequence[tarfile.TarInfo], archive_size: int
) -> None:
    """Require archive members that extract within bounds and the root.

    These checks do not depend on the `tarfile` data filter, which has
    known bypasses in CPython releases before 3.12.11 and 3.13.4.

    Args:
        members: Every member of an uncompressed tar archive.
        archive_size: Size of the archive file in bytes.

    Raises:
        ValueError: If `_archive_member_path` rejects a member, two regular
            files have the same name, a name is both a regular file and a
            directory, a directory has a SigMF file suffix, the regular
            files declare more bytes than the archive holds, or extraction
            would create more files and directories than the archive has
            512-byte blocks.
    """
    total = 0
    files: set[PurePosixPath] = set()
    directories: set[PurePosixPath] = set()
    for member in members:
        path = _archive_member_path(member)
        # Extraction creates the parents of every member as directories.
        directories.update(path.parents)
        if member.isdir():
            directories.add(path)
        elif path in files:
            raise ValueError(
                f"SigMF archive has more than one member {member.name!r}"
            )
        else:
            files.add(path)
            total += member.size
    directories.discard(PurePosixPath("."))
    conflicts = sorted(str(path) for path in files & directories)
    if conflicts:
        raise ValueError(
            "SigMF archive members are both files and directories: "
            f"{conflicts}"
        )
    misnamed = sorted(
        str(path)
        for path in directories
        if path.name.endswith(_SIGMF_FILE_SUFFIXES)
    )
    if misnamed:
        raise ValueError(
            f"SigMF archive directories have SigMF file suffixes: {misnamed}"
        )
    # Without compression or sparse members, the data of each regular file
    # occupies the archive itself.
    if total > archive_size:
        raise ValueError(
            f"SigMF archive members declare {total} bytes, more than the "
            f"{archive_size}-byte archive holds"
        )
    # Each explicit member has its own header block. Implied parent
    # directories have none, so count them against the same bound.
    if len(files) + len(directories) > archive_size // tarfile.BLOCKSIZE:
        raise ValueError(
            "SigMF archive would extract more files and directories than "
            f"its {archive_size} bytes hold headers for"
        )


def _extract_sigmf_archive(
    archive_source: str | PathLike[str], destination: Path
) -> None:
    """Extract an uncompressed SigMF archive after checking every member.

    Args:
        archive_source: Source `.sigmf` archive path.
        destination: Empty extraction directory.

    Raises:
        ValueError: If the archive is compressed, is not a readable tar file,
            has a member that `_check_archive_members` or the `tarfile` data
            filter rejects, or has member names that the file system cannot
            hold.
    """
    with Path(archive_source).open("rb") as handle:
        archive_size = os.fstat(handle.fileno()).st_size
        try:
            with tarfile.open(fileobj=handle, mode="r:") as archive:
                members = _read_archive_members(archive, archive_size)
                _check_archive_members(members, archive_size)
                # The data filter is a second check on member names, types,
                # and permissions.
                archive.extractall(destination, members, filter="data")
        except tarfile.TarError as exc:
            compression = _archive_compression(handle)
            if compression is not None:
                raise ValueError(
                    f"SigMF archive is {compression}-compressed. A .sigmf "
                    "archive must be an uncompressed tar file"
                ) from exc
            # Some tarfile messages span lines. Keep the error on one line.
            detail = " ".join(str(exc).split())
            raise ValueError(f"Invalid SigMF archive: {detail}") from exc
        except OSError as exc:
            if exc.errno not in _ARCHIVE_LAYOUT_ERRNOS:
                raise
            detail = os.strerror(exc.errno)
            if exc.filename is not None:
                detail += f": {exc.filename!r}"
            raise ValueError(f"Invalid SigMF archive: {detail}") from exc


@remove_new_store_on_failure
def import_sigmf_archive(
    store_path: str | PathLike[str],
    source_path: str | PathLike[str],
    *,
    overwrite_store: bool = False,
    overwrite_recording: bool = False,
    overwrite_collection: bool = False,
    sample_chunks: tuple[int, ...] | None = None,
    sample_shards: ShardsLike | None = None,
    automatic_sharding: bool = True,
    sample_compressor: CompressorLike = "auto",
    zarr_format: ZarrFormat | None = None,
) -> SigMFZarrStore:
    """Import a standard SigMF archive into a SigMF-Zarr store.

    Args:
        store_path: Target SigMF-Zarr store path.
        source_path: Source `.sigmf` archive path.
        overwrite_store: Whether to replace an existing SigMF-Zarr store at
            `store_path`. The target must be absent, an empty directory, or
            a SigMF-Zarr store, and must not contain the archive.
        overwrite_recording: Whether to replace each existing recording that
            has the same name as an archived recording.
        overwrite_collection: Whether to replace an existing collection that
            has the same name as the archived collection.
        sample_chunks: Optional sample-array chunk shape for every imported
            recording. Its rank must match each recording's sample array.
        sample_shards: Optional sample-array shard shape for every imported
            recording.
        automatic_sharding: Whether to derive format-3 shards for imported
            recordings when explicit shards are not supplied.
        sample_compressor: Optional sample-array compressor.
        zarr_format: Optional physical Zarr format requirement. Existing
            stores are auto-detected when omitted. New stores default to Zarr
            format 3.

    Returns:
        Updated SigMF-Zarr store.

    Raises:
        ValueError: If the archive is compressed or is not a readable tar
            file, has a sparse member, a member other than a regular file or
            directory, a member outside the archive root, conflicting or
            duplicate member names, or members that declare more bytes or
            entries than the archive holds, or has invalid, unsupported, or
            inconsistent SigMF content. Also if a sample chunk or shard shape
            is invalid, sample sharding is requested for Zarr format 2,
            `overwrite_store` names a target that it may not replace, or a
            recording or the collection exists without overwrite permission.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        _extract_sigmf_archive(source_path, tmp_path)

        meta_files = sorted(tmp_path.rglob("*.sigmf-meta"))
        target_names = tuple(
            path.name.removesuffix(".sigmf-meta") for path in meta_files
        )
        _validate_archive_names(target_names)
        collection_files = sorted(tmp_path.rglob("*.sigmf-collection"))
        if not meta_files:
            raise ValueError("No SigMF recordings in archive")
        if len(collection_files) > 1:
            raise ValueError("Multiple SigMF collections in one archive")
        sources = [_read_sigmf_source(path, None) for path in meta_files]
        _check_archive_files(tmp_path, sources, collection_files)
        collection = (
            _read_archive_collection(collection_files[0], meta_files)
            if collection_files
            else None
        )

        store, sample_storage = open_import_store(
            store_path,
            overwrite_store=overwrite_store,
            zarr_format=zarr_format,
            sources=(source_path,),
            resolve_storage=lambda target_format: [
                source.sample_storage(
                    target_format,
                    sample_chunks=sample_chunks,
                    sample_shards=sample_shards,
                    automatic_sharding=automatic_sharding,
                )
                for source in sources
            ],
        )
        if (
            collection is not None
            and not overwrite_collection
            and collection[0] in store.collections
        ):
            raise ValueError(
                f"Collection {collection[0]!r} already exists. Pass "
                "overwrite_collection=True to replace it"
            )
        with ExitStack() as rollback:
            for name in target_names:
                rollback.enter_context(
                    recording_import_transaction(
                        store, name, overwrite=overwrite_recording
                    )
                )
            if collection is not None:
                rollback.enter_context(
                    collection_import_transaction(store, collection[0])
                )
            for source, storage in zip(sources, sample_storage, strict=True):
                _write_sigmf_recording(
                    store,
                    source,
                    sample_storage=storage,
                    sample_compressor=sample_compressor,
                    overwrite_recording=overwrite_recording,
                )

            if collection is not None:
                collection_name, collection_metadata, recording_ids = (
                    collection
                )
                imported = store.collections.create(
                    collection_name,
                    metadata=collection_metadata,
                    recording_ids=recording_ids,
                    overwrite=True,
                )
                imported.update_integrity()
            store._update_metadata_integrity_when_complete()

    return store


def _validate_archive_names(names: tuple[str, ...]) -> None:
    """Require unique, flat archive filenames without path normalization.

    Args:
        names: Recording identifiers or a collection identifier.

    Raises:
        ValueError: If names collide or contain path components.
    """
    if len(set(names)) != len(names):
        raise ValueError("Archive names must be unique")
    for name in names:
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Archive names must be single path components")


def _archive_selection(
    store: SigMFZarrStore,
    *,
    recording_names: Sequence[str] | None,
    collection_name: str | None,
) -> tuple[tuple[str, ...], SigMFCollection | None]:
    """Resolve and check the recordings and collection for archive export.

    Args:
        store: Source SigMF-Zarr store.
        recording_names: Explicit recording names, if any.
        collection_name: Optional collection to include.

    Returns:
        Selected recording names and the opened collection, if any.

    Raises:
        KeyError: If the selected collection does not exist.
        ValueError: If no recordings are selected, names are not valid
            archive members, the collection hash is stale or unsupported,
            the collection metadata violates the SigMF collection schema,
            the collection lists a recording more than once, or the
            collection references recordings outside the selection.
    """
    collection: SigMFCollection | None = None
    if collection_name is not None:
        _validate_archive_names((collection_name,))
        collection = store.collections.open(collection_name)
        if len(set(collection.recording_ids)) != len(collection.recording_ids):
            raise ValueError(
                f"Collection {collection_name!r} lists a recording more "
                "than once"
            )
        # Export writes the sigmf-python specification version when the
        # collection declares none.
        validate_sigmf_collection_schema(
            {"core:version": SIGMF_SPECIFICATION, **collection.metadata}
        )

    if recording_names is not None:
        selected_recordings = tuple(recording_names)
    elif collection is not None:
        selected_recordings = collection.recording_ids
    else:
        selected_recordings = store.list_recordings()
    if not selected_recordings:
        raise ValueError("No recordings selected for export")
    _validate_archive_names(selected_recordings)

    if collection is not None:
        if collection.integrity and not integrity_is_supported(
            collection.integrity
        ):
            raise ValueError(
                "Collection uses an unsupported internal integrity "
                "algorithm or version"
            )
        if (
            collection.metadata_sha512 is not None
            and collection.metadata_sha512
            != collection.calculate_metadata_sha512()
        ):
            raise ValueError(
                "Collection internal metadata SHA-512 does not match current "
                "metadata"
            )
        missing = sorted(
            set(collection.recording_ids) - set(selected_recordings)
        )
        if missing:
            raise ValueError(
                "Selected collection references recordings not "
                f"included in the archive export: {missing}"
            )

    return selected_recordings, collection


def _reproducible_member(member: tarfile.TarInfo) -> tarfile.TarInfo:
    """Remove account, permission, and time details from an archive member.

    Args:
        member: Member header that `tarfile` derived from the file system.

    Returns:
        The member with UID and GID 0, empty owner names, mode `0o755` for
        directories and `0o644` for files, and modification time 0.
    """
    member.uid = 0
    member.gid = 0
    member.uname = ""
    member.gname = ""
    member.mode = 0o755 if member.isdir() else 0o644
    member.mtime = 0
    return member


def export_sigmf_archive(
    store: SigMFZarrStore,
    output_path: str | PathLike[str],
    *,
    recording_names: list[str] | tuple[str, ...] | None = None,
    collection_name: str | None = None,
    overwrite: bool = False,
    pretty: bool = True,
    force: bool = False,
    allow_lossy: bool = False,
) -> Path:
    """Export recordings from a SigMF-Zarr store to a standard archive.

    Members have UID and GID 0, empty owner names, mode `0o755` for
    directories and `0o644` for files, and modification time 0, so identical
    exports produce identical archive bytes.

    Args:
        store: Source SigMF-Zarr store.
        output_path: Target `.sigmf` archive path.
        recording_names: Optional recording names to export. When omitted,
            the selected collection's recordings are exported, or every
            recording when no collection is selected.
        collection_name: Optional collection to include in the archive.
        overwrite: Whether to replace an existing archive.
        pretty: Whether to pretty-print metadata JSON.
        force: Whether to export recordings even when sample-indexed metadata
            appears stale.
        allow_lossy: Whether to warn and omit unsupported recording metadata.

    Returns:
        Written archive path.

    Raises:
        KeyError: If a selected recording or collection does not exist.
        ValueError: If the target is empty or a directory name, a file or
            symbolic link exists at the target without overwrite permission,
            the target is neither a regular file nor a symbolic link, no
            recordings are selected, or the selected collection violates the
            SigMF collection schema, lists a recording more than once, or
            references recordings outside the export set.
    """
    target = archive_path(output_path)
    exists_message = (
        "Target SigMF archive already exists. Pass overwrite=True to "
        "replace it"
    )
    # A dangling symbolic link is an existing target.
    if os.path.lexists(target) and not overwrite:
        raise ValueError(exists_message)
    _check_replaceable_target(target)

    selected_recordings, collection = _archive_selection(
        store, recording_names=recording_names, collection_name=collection_name
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    root_name = target.stem
    # Place the temporary archive beside the target so the final rename stays
    # on one filesystem and is atomic where the platform supports it.
    with tempfile.TemporaryDirectory(
        dir=target.parent,
        prefix=".sigmf-zarr-archive-",
    ) as tmpdir:
        tmp_path = Path(tmpdir)
        archive_root = tmp_path / root_name
        archive_root.mkdir()
        temporary_target = tmp_path / target.name

        metafiles: dict[str, str] = {}
        for recording_name in selected_recordings:
            exported_meta = export_sigmf(
                store,
                recording_name,
                archive_root / f"{recording_name}.sigmf-meta",
                overwrite=False,
                pretty=pretty,
                force=force,
                allow_lossy=allow_lossy,
            )
            metafiles[recording_name] = exported_meta.name

        if collection is not None:
            metadata = {
                StandardSigMFCollection.COLLECTION_KEY: dict(
                    collection.metadata
                )
            }
            collection_obj = StandardSigMFCollection(
                metafiles=[
                    metafiles[name] for name in collection.recording_ids
                ],
                metadata=deepcopy(metadata),
                base_path=archive_root,
                # set_streams() computes hashes from the explicit filenames.
                # Its redundant verifier misresolves IDs ending in a SigMF
                # suffix, such as a recording named "rec.sigmf-meta".
                skip_checksums=True,
            )
            if "core:version" in collection.metadata:
                collection_obj.set_collection_field(
                    "core:version", collection.metadata["core:version"]
                )
            collection_obj.tofile(
                archive_root / f"{collection.name}.sigmf-collection",
                pretty=pretty,
                overwrite=True,
            )

        with tarfile.open(
            temporary_target, mode="w", format=tarfile.PAX_FORMAT
        ) as archive:
            # tarfile adds directory entries in sorted order.
            archive.add(
                archive_root, arcname=root_name, filter=_reproducible_member
            )
        if overwrite:
            temporary_target.replace(target)
        else:
            try:
                _publish_new_file(temporary_target, target)
            except FileExistsError as exc:
                raise ValueError(exists_message) from exc

    return target


__all__ = [
    "archive_path",
    "calculate_sha512",
    "coerce_samples_for_sigmf",
    "coerce_sigmf_samples",
    "export_sigmf",
    "export_sigmf_archive",
    "import_sigmf",
    "import_sigmf_archive",
    "sigmf_dtype_to_zarr",
    "sigmf_sample_axes",
    "strip_zarr_metadata",
    "verify_sha512",
]

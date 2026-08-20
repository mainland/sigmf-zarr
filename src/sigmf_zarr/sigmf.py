"""Import and export helpers for standard SigMF files and archives."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from sigmf import SHA512_KEY
from sigmf.sigmffile import dtype_info

from sigmf_zarr.json import (
    JSONObject,
    JSONValue,
)
from sigmf_zarr.store import (
    SigMFRecording,
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


def sigmf_dtype_to_zarr(
    datatype: str,
) -> tuple[np.dtype[Any], tuple[int, ...]]:
    """Map a SigMF datatype string to SigMF-Zarr sample storage info.

    Args:
        datatype: SigMF datatype string.

    Returns:
        Element dtype for the Zarr array and the per-sample shape suffix
        implied by the SigMF datatype. Complex SigMF datatypes are stored as
        real-valued arrays with an I/Q axis of length 2.
    """
    info = dtype_info(datatype)
    if info["is_complex"]:
        component_dtype = np.dtype(
            cast(np.dtype[Any], info["component_dtype"])
        )
        return component_dtype, (2,)
    sample_dtype = np.dtype(cast(np.dtype[Any], info["sample_dtype"]))
    return sample_dtype, ()


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
    """
    array = np.asarray(samples)
    if sample_axes is None:
        if array.ndim >= 1 and array.shape[-1] == 2:
            return _complex_from_iq(array[..., 0], array[..., 1])
        return array

    axes = tuple(sample_axes)
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
        ValueError: If the component arrays have different shapes.
    """
    real_array = np.asarray(real)
    imag_array = np.asarray(imag)
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


def _sigmf_storage_dtype(
    recording: SigMFRecording,
) -> np.dtype[Any] | None:
    """Return the raw on-disk dtype declared for standard SigMF export.

    Args:
        recording: Recording being exported.

    Returns:
        Declared raw dataset dtype, or `None` when no safe direct mapping is
        available.
    """
    return _sigmf_storage_dtype_from_metadata(recording.global_metadata)


def _sigmf_storage_dtype_from_metadata(
    global_metadata: JSONObject,
) -> np.dtype[Any] | None:
    """Return the standard on-disk dtype declared by global metadata.

    Args:
        global_metadata: Standard SigMF global metadata.

    Returns:
        Declared raw dataset dtype, or None when no direct mapping is safe.
    """
    datatype = global_metadata.get("core:datatype")
    if not isinstance(datatype, str):
        return None
    info = dtype_info(datatype)
    if info["is_complex"] and info["is_fixedpoint"]:
        return None
    return np.dtype(cast(np.dtype[Any], info["memmap_map_type"]))


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
) -> Iterator[npt.NDArray[Any]]:
    """Yield contiguous chunks in standard SigMF dataset byte order.

    Args:
        recording: Recording whose samples should be encoded.

    Yields:
        Contiguous arrays ready to be written to a ``.sigmf-data`` file.

    Raises:
        ValueError: If the recording is batched or has no sample axis.
    """
    if recording.batched:
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
    storage_dtype = _sigmf_storage_dtype(recording)

    for start in range(0, sample_count, chunk_size):
        stop = min(start + chunk_size, sample_count)
        key: list[slice] = [slice(None)] * len(samples.shape)
        key[time_axis] = slice(start, stop)
        chunk = coerce_samples_for_sigmf(
            samples[tuple(key)],
            sample_axes=sample_axes,
        )
        if storage_dtype is not None:
            chunk = np.asarray(chunk, dtype=storage_dtype)
        yield np.ascontiguousarray(chunk)


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


__all__ = [
    "calculate_sha512",
    "coerce_samples_for_sigmf",
    "coerce_sigmf_samples",
    "sigmf_sample_axes",
    "sigmf_dtype_to_zarr",
    "strip_zarr_metadata",
    "verify_sha512",
]

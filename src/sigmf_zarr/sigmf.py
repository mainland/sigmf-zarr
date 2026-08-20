"""Import and export helpers for standard SigMF files and archives."""

from __future__ import annotations

import hashlib
import hmac
import json
import tarfile
import tempfile
from collections.abc import Iterator
from os import PathLike
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from sigmf import SHA512_KEY
from sigmf.sigmffile import SigMFCollection as StandardSigMFCollection
from sigmf.sigmffile import SigMFFile, dtype_info, fromfile

from sigmf_zarr.integrity import integrity_is_supported
from sigmf_zarr.json import (
    JSONObject,
    JSONValue,
    json_object,
    json_object_list,
)
from sigmf_zarr.store import (
    SigMFCollection,
    SigMFRecording,
    SigMFZarrStore,
    ZarrFormat,
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


def _read_sigmf_samples(sigmf_file: SigMFFile) -> npt.NDArray[Any]:
    """Read SigMF samples without narrowing floating complex precision.

    sigmf-python intentionally returns all floating complex data as
    `complex64`. Read floating complex files directly so `cf64_*` values retain
    their 64-bit components. Other datatypes continue to use sigmf-python's
    conversion and autoscaling behavior.

    Args:
        sigmf_file: Open standard SigMF recording.

    Returns:
        Source samples in their supported logical dtype.
    """
    datatype = sigmf_file.get_global_field("core:datatype")
    if not isinstance(datatype, str):
        return np.asarray(sigmf_file.read_samples())
    info = dtype_info(datatype)
    if not info["is_complex"] or info["is_fixedpoint"]:
        return np.asarray(sigmf_file.read_samples())
    if sigmf_file.data_file is None:
        return np.asarray(sigmf_file.read_samples())

    num_channels = int(sigmf_file.num_channels)
    count = int(sigmf_file.sample_count) * num_channels
    storage_dtype = np.dtype(
        cast(np.dtype[Any], info["memmap_map_type"])
    )
    samples = np.fromfile(
        sigmf_file.data_file,
        dtype=storage_dtype,
        count=count,
        offset=int(getattr(sigmf_file, "data_offset", 0)),
    )
    if num_channels > 1:
        return samples.reshape((-1, num_channels))
    return samples


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


def _array_sigmf_sha512(
    samples: npt.ArrayLike,
    *,
    sample_axes: tuple[str, ...],
    global_metadata: JSONObject,
) -> str:
    """Hash an in-memory array in its declared standard SigMF encoding.

    Args:
        samples: SigMF-Zarr logical sample tensor.
        sample_axes: Semantic axes for the sample tensor.
        global_metadata: Standard SigMF global metadata.

    Returns:
        Lowercase SHA-512 digest of the encoded dataset bytes.
    """
    encoded = coerce_samples_for_sigmf(samples, sample_axes=sample_axes)
    storage_dtype = _sigmf_storage_dtype_from_metadata(global_metadata)
    if storage_dtype is not None:
        encoded = np.asarray(encoded, dtype=storage_dtype)
    return hashlib.sha512(np.ascontiguousarray(encoded).tobytes()).hexdigest()


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


def archive_path(path: str | PathLike[str]) -> Path:
    """Normalize a target path to the standard `.sigmf` suffix.

    Args:
        path: User-provided archive target path.

    Returns:
        Path ending with `.sigmf`.
    """
    target = Path(path)
    return (
        target if target.suffix == ".sigmf" else target.with_suffix(".sigmf")
    )


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
        raise ValueError(
            f"global metadata field {field!r} must be positive"
        )
    return num_channels


def _prepare_export_global_metadata(recording: SigMFRecording) -> JSONObject:
    """Return SigMF global metadata prepared for standard export.

    Args:
        recording: Recording being exported.

    Returns:
        Stripped global metadata with `core:num_channels` validated against
        the declared channel axis and filled for multi-channel recordings.

    Raises:
        ValueError: If `core:num_channels` is not a positive integer or does
            not match the derived channel count.
    """
    metadata = strip_zarr_metadata(recording.global_metadata)
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
        raise ValueError(
            f"global metadata field {field!r} must be positive"
        )
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
    """Validate one metadata entry against the exported sample count.

    Args:
        entry: Capture or annotation metadata entry.
        sample_count: Number of samples being exported.
        label: Human-readable metadata entry label.
        sample_offset: Absolute index of the first exported sample.

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
            f"{label} starts at sample {start}, outside exported sample "
            f"range {sample_offset}..{sample_end}"
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
            f"{label} ends at sample {end}, beyond exported sample end "
            f"{sample_end}"
        )
    return end


def _validate_export_metadata(recording: SigMFRecording) -> None:
    """Validate sample-indexed metadata before standard SigMF export.

    Args:
        recording: Recording to validate.

    Raises:
        ValueError: If captures or annotations are inconsistent with the
            exported sample count.
    """
    sample_count = recording.sample_count
    sample_offset = _metadata_integer(
        recording.global_metadata.get("core:offset", 0),
        field="core:offset",
        label="global metadata",
    )
    if sample_offset < 0:
        raise ValueError("core:offset must be nonnegative")
    sample_end = sample_offset + sample_count
    captures = recording.captures
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

    for index, annotation in enumerate(recording.annotations):
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
            "but the recording contains "
            f"{sample_count} samples. Update captures before export or pass "
            "force=True to export anyway."
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


def _write_sigmf_data_file(
    recording: SigMFRecording,
    data_path: Path,
) -> str:
    """Write recording samples and calculate their SHA-512 by chunk.

    Args:
        recording: Recording whose samples should be exported.
        data_path: Target ``.sigmf-data`` path.

    Returns:
        Lowercase SHA-512 hexadecimal digest of the written file.

    Raises:
        ValueError: If the recording cannot be encoded as standard SigMF.
    """
    digest = hashlib.sha512()
    with data_path.open("wb") as handle:
        for chunk in _iter_sigmf_data_chunks(recording):
            encoded = chunk.tobytes()
            handle.write(encoded)
            digest.update(encoded)
    return digest.hexdigest()


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
        OSError: If replacement fails. Existing files are restored when
            possible before the error is propagated.
    """
    backup_meta = tmp_meta.parent / ".previous.sigmf-meta"
    backup_data = tmp_data.parent / ".previous.sigmf-data"
    had_meta = meta_path.exists()
    had_data = data_path.exists()
    moved_meta = False
    moved_data = False
    try:
        # Publish data before metadata so a visible new metadata file never
        # points at an old data file during a successful replacement.
        if had_meta:
            meta_path.replace(backup_meta)
            moved_meta = True
        if had_data:
            data_path.replace(backup_data)
            moved_data = True
        tmp_data.replace(data_path)
        tmp_meta.replace(meta_path)
    except OSError:
        meta_path.unlink(missing_ok=True)
        data_path.unlink(missing_ok=True)
        if moved_data:
            backup_data.replace(data_path)
        if moved_meta:
            backup_meta.replace(meta_path)
        raise


def import_sigmf(
    store_path: str | PathLike[str],
    sigmf_path: str | PathLike[str],
    *,
    recording_name: str | None = None,
    overwrite_store: bool = False,
    overwrite_recording: bool = False,
    zarr_format: ZarrFormat | None = None,
) -> SigMFRecording:
    """Import one standard SigMF recording into a SigMF-Zarr store.

    Args:
        store_path: Target SigMF-Zarr store path.
        sigmf_path: Source `.sigmf-meta` path.
        recording_name: Optional recording name override.
        overwrite_store: Whether to recreate the target store root.
        overwrite_recording: Whether to replace an existing recording.
        zarr_format: Optional physical Zarr format requirement. Existing
            stores are auto-detected when omitted. New stores default to Zarr
            format 3.

    Returns:
        Imported recording wrapper.

    Raises:
        ValueError: If the SigMF data has no sample axis.
    """
    sigmf_file = fromfile(str(sigmf_path))
    metadata = cast(dict[str, object], sigmf_file.get_global_info())
    global_metadata = json_object(metadata, name="global")
    captures = json_object_list(sigmf_file.get_captures(), name="captures")
    annotations = json_object_list(
        sigmf_file.get_annotations(), name="annotations"
    )
    num_channels = _sigmf_num_channels(global_metadata)
    sigmf_samples = _read_sigmf_samples(sigmf_file)
    samples = coerce_sigmf_samples(
        sigmf_samples,
        num_channels=num_channels,
    )
    sample_axes = sigmf_sample_axes(
        sigmf_samples,
        num_channels=num_channels,
    )
    if samples.ndim == 0:
        raise ValueError("SigMF sample data must have at least one axis")

    # Verify representability before changing the target store. Conversion of
    # a source datatype must reproduce the exact standard SigMF byte stream.
    source_sha512 = global_metadata.get(SHA512_KEY)
    imported_sha512 = _array_sigmf_sha512(
        samples,
        sample_axes=sample_axes,
        global_metadata=global_metadata,
    )
    if isinstance(source_sha512, str) and not hmac.compare_digest(
        source_sha512.lower(), imported_sha512
    ):
        raise ValueError(
            "Imported samples do not reproduce the source core:sha512. "
            "The source datatype or byte layout cannot be represented "
            "losslessly"
        )

    target_name = recording_name or Path(sigmf_path).stem.replace(".sigmf", "")
    store = SigMFZarrStore.create(
        store_path,
        overwrite=overwrite_store,
        zarr_format=zarr_format,
    )
    try:
        recording = store.recordings.open(
            target_name,
            create=True,
            batched=False,
            sample_dtype=samples.dtype,
            sample_shape=tuple(int(dim) for dim in samples.shape),
            sample_axes=sample_axes,
            global_metadata=global_metadata,
            captures=captures,
            annotations=annotations,
            overwrite=overwrite_recording,
        )
        recording.set_samples(samples)
        recording.set_global_field(SHA512_KEY, imported_sha512)
        recording.update_integrity()
        store.update_metadata_integrity()
    except Exception:
        # Import is resource-atomic even when the containing store predates it.
        if target_name in store.recordings:
            store.recordings.remove(target_name)
        raise
    return recording


def export_sigmf(
    store: SigMFZarrStore,
    recording_name: str,
    output_path: str | PathLike[str],
    *,
    overwrite: bool = False,
    pretty: bool = True,
    force: bool = False,
) -> Path:
    """Export one recording from a SigMF-Zarr store to standard SigMF files.

    Args:
        store: Source SigMF-Zarr store.
        recording_name: Recording name to export.
        output_path: Target `.sigmf-meta` path or path stem.
        overwrite: Whether to replace existing target files.
        pretty: Whether to pretty-print metadata JSON.
        force: Whether to export even when sample-indexed metadata appears
            stale.

    Returns:
        Written `.sigmf-meta` path.

    Raises:
        ValueError: If the recording is batched, target files already exist,
            or metadata is inconsistent with the exported sample count.
    """
    recording = store.recordings.open(recording_name)
    if recording.batched:
        raise ValueError(
            "Standard SigMF export only supports unbatched recordings"
        )
    if recording.integrity and not integrity_is_supported(
        recording.integrity
    ):
        raise ValueError(
            "Recording uses an unsupported internal integrity algorithm or "
            "version"
        )
    if (
        recording.sample_sha512 is not None
        and recording.sample_sha512 != recording.calculate_sample_sha512()
    ):
        raise ValueError(
            "Recording internal sample SHA-512 does not match current data"
        )
    if (
        recording.metadata_sha512 is not None
        and recording.metadata_sha512
        != recording.calculate_metadata_sha512()
    ):
        raise ValueError(
            "Recording internal metadata SHA-512 does not match current "
            "metadata"
        )
    if not force:
        _validate_export_metadata(recording)
    global_metadata = _prepare_export_global_metadata(recording)
    metadata: dict[str, object] = {
        SigMFFile.GLOBAL_KEY: global_metadata,
        SigMFFile.CAPTURE_KEY: list(recording.captures),
        SigMFFile.ANNOTATION_KEY: list(recording.annotations),
    }

    sigmf_file = SigMFFile(metadata=metadata)
    output = Path(output_path)
    meta_path = (
        output
        if output.suffix == ".sigmf-meta"
        else output.with_suffix(".sigmf-meta")
    )
    data_path = meta_path.with_suffix(".sigmf-data")

    if not overwrite and (meta_path.exists() or data_path.exists()):
        raise ValueError(
            "Target SigMF files already exist. Pass overwrite=True to "
            "replace them"
        )

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=meta_path.parent,
        prefix=".sigmf-zarr-export-",
    ) as tmpdir:
        tmp_path = Path(tmpdir)
        tmp_meta = tmp_path / meta_path.name
        tmp_data = tmp_path / data_path.name
        exported_sha512 = _write_sigmf_data_file(recording, tmp_data)
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
        _replace_sigmf_pair(
            tmp_meta=tmp_meta,
            tmp_data=tmp_data,
            meta_path=meta_path,
            data_path=data_path,
        )
    return meta_path


def import_sigmf_archive(
    store_path: str | PathLike[str],
    archive_source: str | PathLike[str],
    *,
    overwrite_store: bool = False,
    overwrite_recordings: bool = False,
    zarr_format: ZarrFormat | None = None,
) -> SigMFZarrStore:
    """Import a standard SigMF archive into a SigMF-Zarr store.

    Args:
        store_path: Target SigMF-Zarr store path.
        archive_source: Source `.sigmf` archive path.
        overwrite_store: Whether to recreate the target store root.
        overwrite_recordings: Whether to replace existing recordings.
        zarr_format: Optional physical Zarr format requirement. Existing
            stores are auto-detected when omitted. New stores default to Zarr
            format 3.

    Returns:
        Updated SigMF-Zarr store.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with tarfile.open(archive_source, mode="r") as archive:
            # The data filter rejects archive members that could escape the
            # temporary extraction directory.
            archive.extractall(tmp_path, filter="data")

        meta_files = sorted(tmp_path.rglob("*.sigmf-meta"))
        collection_files = sorted(tmp_path.rglob("*.sigmf-collection"))
        collection_name: str | None = None
        collection_metadata: JSONObject | None = None
        recording_ids: tuple[str, ...] = ()
        if collection_files:
            collection_file = collection_files[0]
            with collection_file.open("r", encoding="utf-8") as handle:
                metadata = cast(dict[str, object], json.load(handle))
            collection_obj = StandardSigMFCollection(
                metadata=metadata,
                base_path=collection_file.parent,
            )
            collection_info = cast(
                JSONObject, collection_obj.get_collection_info()
            )
            recording_ids = tuple(
                Path(stream_name).stem.replace(".sigmf", "")
                for stream_name in collection_obj.get_stream_names()
            )
            collection_metadata = dict(collection_info)
            collection_metadata.pop("core:streams", None)
            collection_name = collection_file.stem.replace(".sigmf", "")

        store = SigMFZarrStore.create(
            store_path,
            overwrite=overwrite_store,
            zarr_format=zarr_format,
        )
        imported_names: list[str] = []
        try:
            for meta_file in meta_files:
                recording = import_sigmf(
                    store_path,
                    meta_file,
                    overwrite_store=False,
                    overwrite_recording=overwrite_recordings,
                    zarr_format=zarr_format,
                )
                imported_names.append(recording.name)

            if collection_name is not None:
                collection = store.collections.open(
                    collection_name,
                    create=True,
                    metadata=collection_metadata,
                    recording_ids=recording_ids,
                    overwrite=True,
                )
                collection.update_integrity()
            store.update_metadata_integrity()
        except Exception:
            # Remove only recordings created by this archive operation.
            for imported_name in reversed(imported_names):
                if imported_name in store.recordings:
                    store.recordings.remove(imported_name)
            raise

    return store


def export_sigmf_archive(
    store: SigMFZarrStore,
    archive_target: str | PathLike[str],
    *,
    recording_names: list[str] | tuple[str, ...] | None = None,
    collection_name: str | None = None,
    overwrite: bool = False,
    pretty: bool = True,
    force: bool = False,
) -> Path:
    """Export recordings from a SigMF-Zarr store to a standard archive.

    Args:
        store: Source SigMF-Zarr store.
        archive_target: Target `.sigmf` archive path.
        recording_names: Optional recording names to export. When omitted,
            all recordings are exported.
        collection_name: Optional collection to include in the archive.
        overwrite: Whether to replace an existing archive.
        pretty: Whether to pretty-print metadata JSON.
        force: Whether to export recordings even when sample-indexed metadata
            appears stale.

    Returns:
        Written archive path.

    Raises:
        ValueError: If the target exists, no recordings are selected, or the
            selected collection references recordings outside the export set.
    """
    target = archive_path(archive_target)
    if target.exists() and not overwrite:
        raise ValueError(
            "Target SigMF archive already exists. Pass overwrite=True to "
            "replace it"
        )

    selected_recordings = tuple(
        recording_names
        if recording_names is not None
        else store.list_recordings()
    )
    if not selected_recordings:
        raise ValueError("No recordings selected for export")

    collection: SigMFCollection | None = None
    if collection_name is not None:
        collection = store.collections.open(collection_name)
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

        metafiles: list[str] = []
        for recording_name in selected_recordings:
            exported_meta = export_sigmf(
                store,
                recording_name,
                archive_root / recording_name,
                overwrite=True,
                pretty=pretty,
                force=force,
            )
            metafiles.append(exported_meta.name)

        if collection is not None:
            metadata = {
                StandardSigMFCollection.COLLECTION_KEY: dict(
                    collection.metadata
                )
            }
            collection_obj = StandardSigMFCollection(
                metafiles=metafiles,
                metadata=metadata,
                base_path=archive_root,
                skip_checksums=False,
            )
            collection_obj.tofile(
                archive_root / collection.name,
                pretty=pretty,
                overwrite=True,
            )

        with tarfile.open(temporary_target, mode="w") as archive:
            archive.add(archive_root, arcname=root_name)
        temporary_target.replace(target)

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
    "sigmf_sample_axes",
    "sigmf_dtype_to_zarr",
    "strip_zarr_metadata",
    "verify_sha512",
]

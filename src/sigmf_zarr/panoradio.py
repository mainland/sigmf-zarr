"""Import Panoradio HF NumPy samples and CSV tags into SigMF-Zarr."""

from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from zarr.core.array import CompressorLike, ShardsLike

from sigmf_zarr._rfml import _with_rfml_profile
from sigmf_zarr.json import JSONObject
from sigmf_zarr.sample_storage import resolve_import_sample_storage
from sigmf_zarr.store import SigMFZarrStore, ZarrFormat
from sigmf_zarr.store._transaction import recording_import_transaction


def _load_panoradio_samples(path: Path) -> np.memmap:
    """Map a Panoradio sample array without loading its full contents.

    Args:
        path: Source NumPy file.

    Returns:
        Read-only complex sample array with shape `(N, T)`.

    Raises:
        ValueError: If the file is not a non-empty, two-dimensional NumPy
            complex64 or complex128 array.
        OSError: If the source file cannot be read.
    """
    samples = np.load(path, mmap_mode="r", allow_pickle=False)
    if isinstance(samples, np.lib.npyio.NpzFile):
        samples.close()
        raise ValueError("Panoradio samples must be a .npy array, not .npz")
    if not isinstance(samples, np.memmap):
        raise ValueError(
            "Panoradio samples must be a memory-mapped .npy array"
        )
    if (
        samples.ndim != 2
        or any(size <= 0 for size in samples.shape)
        or samples.dtype.kind != "c"
        or samples.dtype.itemsize not in (8, 16)
    ):
        raise ValueError(
            "Panoradio samples must be a non-empty complex64 or complex128 "
            f"array with shape (N, T), got {samples.shape} and {samples.dtype}"
        )
    return samples


def _tag_values(row: dict[str, str]) -> tuple[int, str, int]:
    """Validate and decode one Panoradio tag row.

    Args:
        row: CSV fields keyed by their normalized column names.

    Returns:
        Sample index, transmission-mode name, and SNR in decibels.

    Raises:
        ValueError: If the index, mode, or SNR is invalid.
    """
    index = int(row["idx"])
    mode = row["mode"].strip()
    if not mode:
        raise ValueError("mode must be non-empty")
    # Check integrality in decimal arithmetic so binary floating-point
    # rounding cannot turn a fractional CSV value into an accepted integer.
    try:
        snr = Decimal(row["snr"])
    except InvalidOperation as exc:
        raise ValueError("snr must be numeric") from exc
    limits = np.iinfo(np.int16)
    if (
        not snr.is_finite()
        or snr != snr.to_integral_value()
        or not limits.min <= snr <= limits.max
    ):
        raise ValueError("snr must be an integral int16 value in dB")
    return index, mode, int(snr)


def _load_panoradio_tags(
    path: Path,
    item_count: int,
) -> tuple[list[str], npt.NDArray[np.int32], npt.NDArray[np.int16]]:
    """Read CSV tags and align them with the source sample indexes.

    Args:
        path: Source tags file with `idx`, `mode`, and `snr` columns.
        item_count: Number of sample vectors in the NumPy file.

    Returns:
        Sorted mode vocabulary, per-item mode IDs, and per-item SNR values.

    Raises:
        ValueError: If the header or tags are invalid, or sample indexes are
            duplicated, missing, or outside the sample array.
        OSError: If the tags file cannot be read.
    """
    # CSV rows need not follow sample order. Place tags by idx, using the
    # rejected empty mode as a sentinel for missing and duplicate entries.
    modes = [""] * item_count
    snr_db = np.empty(item_count, dtype=np.int16)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, skipinitialspace=True, strict=True)
        try:
            columns = [value.strip() for value in next(reader, [])]
            if len(columns) != 3 or set(columns) != {"idx", "mode", "snr"}:
                raise ValueError(
                    "Panoradio tags must have idx, mode, snr columns"
                )
            for values in reader:
                if not values:
                    continue
                if len(values) != 3:
                    raise ValueError("each tag row must contain three fields")
                index, mode, snr = _tag_values(
                    dict(zip(columns, values, strict=True))
                )
                if not 0 <= index < item_count:
                    raise ValueError(f"sample index {index} is out of range")
                if modes[index]:
                    raise ValueError(f"duplicate sample index {index}")
                modes[index] = mode
                snr_db[index] = snr
        except (ValueError, csv.Error) as exc:
            raise ValueError(
                f"Invalid Panoradio tag at line {reader.line_num}: {exc}"
            ) from exc
    if "" in modes:
        raise ValueError("Panoradio tags are missing sample indexes")
    classes = sorted(set(modes))
    class_ids = {mode: index for index, mode in enumerate(classes)}
    mode_id = np.fromiter(
        (class_ids[mode] for mode in modes), dtype=np.int32, count=item_count
    )
    return classes, mode_id, snr_db


def import_panoradio_dataset(
    store_path: str | PathLike[str],
    source_path: str | PathLike[str],
    tags_path: str | PathLike[str],
    *,
    source_dataset: str | None = None,
    recording_name: str = "panoradio",
    overwrite_store: bool = False,
    overwrite_recording: bool = False,
    batch_size: int = 4096,
    global_metadata: JSONObject | None = None,
    sample_chunks: tuple[int, int, int] | None = None,
    sample_shards: ShardsLike | None = None,
    automatic_sharding: bool = True,
    sample_compressor: CompressorLike = "auto",
    zarr_format: ZarrFormat | None = None,
) -> SigMFZarrStore:
    """Stream a Panoradio HF dataset into a batched recording.

    The importer preserves source sample order and component precision. CSV
    `idx` values align the mode and SNR tags with the sample vectors. Samples
    use the canonical `(N, 2, T)` I/Q layout and the published 6000 Hz sample
    rate. The complete tag arrays remain in memory. Samples are memory mapped
    and converted in bounded batches.

    Args:
        store_path: Target SigMF-Zarr store.
        source_path: NumPy file containing complex samples with shape `(N, T)`.
        tags_path: CSV file with `idx`, `mode`, and `snr` columns.
        source_dataset: Source dataset name. Defaults to the NumPy file stem.
        recording_name: Recording name to create.
        overwrite_store: Whether to recreate the target store.
        overwrite_recording: Whether to replace an existing recording.
        batch_size: Maximum number of items per sample conversion and append.
        global_metadata: Optional global metadata. Source dataset, datatype,
            and sample rate values take precedence over supplied values.
        sample_chunks: Optional sample-array chunk shape.
        sample_shards: Optional sample-array shard shape.
        automatic_sharding: Whether to derive format-3 shards when explicit
            shards are not supplied.
        sample_compressor: Optional sample compressor.
        zarr_format: Optional physical Zarr format requirement. Existing
            stores are auto-detected. New stores default to Zarr format 3.

    Returns:
        Updated SigMF-Zarr store.

    Raises:
        ValueError: If samples, tags, or import options are invalid, or the
            recording exists and replacement is disabled.
        OSError: If a source file cannot be read.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    samples = _load_panoradio_samples(Path(source_path))
    item_count = int(samples.shape[0])
    sample_shape = (2, int(samples.shape[1]))
    classes, mode_id, snr_db = _load_panoradio_tags(
        Path(tags_path), item_count
    )
    sample_dtype = samples.real.dtype.newbyteorder("=")
    metadata = _with_rfml_profile(global_metadata)
    metadata["panoradio:source_dataset"] = (
        source_dataset or Path(source_path).stem
    )
    metadata["core:sample_rate"] = 6000
    metadata["core:datatype"] = f"cf{sample_dtype.itemsize * 8}_le"
    store = SigMFZarrStore.create(
        store_path, overwrite=overwrite_store, zarr_format=zarr_format
    )
    resolved_chunks, resolved_shards = resolve_import_sample_storage(
        sample_dtype,
        sample_shape,
        item_count,
        batched=True,
        zarr_format=store.zarr_format,
        sample_chunks=sample_chunks,
        sample_shards=sample_shards,
        automatic_sharding=automatic_sharding,
    )
    with recording_import_transaction(
        store, recording_name, overwrite=overwrite_recording
    ):
        recording = store.recordings.open(
            recording_name,
            create=True,
            batched=True,
            sample_dtype=sample_dtype,
            sample_shape=sample_shape,
            sample_axes=("iq", "time"),
            global_metadata=metadata,
            sample_chunks=resolved_chunks,
            sample_shards=resolved_shards,
            sample_compressor=sample_compressor,
            overwrite=overwrite_recording,
        )
        # Split complex values into I and Q planes only for the current batch.
        # The memory map keeps the full source tensor out of the conversion
        # buffer, while sample_dtype preserves each component's precision.
        for start in range(0, item_count, batch_size):
            batch: npt.NDArray[Any] = samples[start:start + batch_size]
            recording.append_samples(
                np.stack((batch.real, batch.imag), axis=1).astype(
                    sample_dtype, copy=False
                )
            )
        recording.add_index(
            "mode_id",
            mode_id,
            axis="item",
            field="rfml-dataset:class",
            labels=tuple(classes),
        )
        recording.add_index(
            "snr_db",
            snr_db,
            axis="item",
            field="panoradio:snr",
            unit="dB",
        )
        recording.update_integrity()
        store.update_metadata_integrity()
    return store


__all__ = ["import_panoradio_dataset"]

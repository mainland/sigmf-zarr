"""Import RadioML 2018 HDF5 datasets into SigMF-Zarr."""

from __future__ import annotations

import ast
import json
import math
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
from zarr.core.array import CompressorLike, ShardsLike

from sigmf_zarr._hdf5 import (
    HDF5RowReader,
    dataset_filters,
    hdf5_plugins_disabled,
)
from sigmf_zarr._import_limits import (
    MAX_IMPORT_ITEM_BYTES,
    batch_items,
    check_item_bytes,
)
from sigmf_zarr._import_target import (
    remove_new_store_on_failure,
)
from sigmf_zarr.json import JSONObject
from sigmf_zarr.provenance import file_identity, import_metadata
from sigmf_zarr.store import SigMFRecording, SigMFZarrStore, ZarrFormat
from sigmf_zarr.store._transaction import recording_import_transaction

RADIOML2018_MODULATION_CLASSES: tuple[str, ...] = (
    "OOK",
    "4ASK",
    "8ASK",
    "BPSK",
    "QPSK",
    "8PSK",
    "16PSK",
    "32PSK",
    "16APSK",
    "32APSK",
    "64APSK",
    "128APSK",
    "16QAM",
    "32QAM",
    "64QAM",
    "128QAM",
    "256QAM",
    "AM-SSB-WC",
    "AM-SSB-SC",
    "AM-DSB-WC",
    "AM-DSB-SC",
    "FM",
    "GMSK",
    "OQPSK",
)
"""Corrected one-hot order from the RadioML 2018.01A ``classes-fixed.json``.

The file is included in pinxau1000's Kaggle mirror of DeepSig's dataset:
https://www.kaggle.com/datasets/pinxau1000/radioml2018
"""

_MAX_MODULATION_CLASSES = int(np.iinfo(np.int16).max) + 1
"""Largest class count whose identifiers fit the `int16` class index."""


def _sample_shape(samples: h5py.Dataset) -> tuple[tuple[int, int], bool]:
    """Validate the sample tensor and resolve its I/Q layout.

    Args:
        samples: HDF5 sample dataset.

    Returns:
        SigMF-Zarr sample shape and whether I/Q is the last HDF5 axis.

    Raises:
        ValueError: If the sample tensor has no I/Q axis of length two.
    """
    if samples.ndim != 3:
        raise ValueError(
            "RadioML 2018 samples must have shape (N, T, 2) or (N, 2, T), "
            f"got {samples.shape}"
        )
    # Prefer the published (N, T, 2) layout when both trailing axes have length
    # two. Shape alone cannot distinguish the conventions in that case.
    if samples.shape[2] == 2:
        return (2, int(samples.shape[1])), True
    if samples.shape[1] == 2:
        return (2, int(samples.shape[2])), False
    raise ValueError(
        "RadioML 2018 samples must have an I/Q axis of length 2, "
        f"got {samples.shape}"
    )


def _modulation_ids(
    labels: npt.NDArray[Any],
    *,
    num_classes: int,
) -> npt.NDArray[np.int16]:
    """Decode one RadioML 2018 one-hot label batch.

    Args:
        labels: One-hot label matrix.
        num_classes: Expected class count.

    Returns:
        Integer modulation identifiers.

    Raises:
        ValueError: If labels are not one-hot encoded.
    """
    values = np.asarray(labels)
    if values.ndim != 2 or values.shape[1] != num_classes:
        raise ValueError(
            "RadioML 2018 labels must have shape (N, C) with "
            f"C={num_classes}, got {values.shape}"
        )
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("RadioML 2018 labels must be numeric")
    if not np.all(np.isfinite(values)) or not np.all(values >= 0):
        raise ValueError("RadioML 2018 labels must be finite and non-negative")
    if not np.allclose(values.sum(axis=1), 1.0):
        raise ValueError("RadioML 2018 labels must be one-hot encoded")
    # The sum check alone would accept fractional distributions such as
    # [0.5, 0.5, 0, ...]. Require exactly one nonzero column as well.
    if not np.all(np.count_nonzero(values, axis=1) == 1):
        raise ValueError("RadioML 2018 labels must contain one active class")
    return np.asarray(np.argmax(values, axis=1), dtype=np.int16)


def _snr_values(snr: npt.NDArray[Any]) -> npt.NDArray[np.int16]:
    """Normalize one RadioML 2018 SNR batch.

    Args:
        snr: SNR values with shape `(N,)` or `(N, 1)`.

    Returns:
        One-dimensional integer SNR values.

    Raises:
        ValueError: If values are not integral `int16` values.
    """
    values = np.asarray(snr)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError(
            "RadioML 2018 SNR labels must have shape (N,) or (N, 1), "
            f"got {values.shape}"
        )
    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("RadioML 2018 SNR labels must be numeric") from exc
    # Rounding is only an integrality check. Reject fractional or out-of-range
    # SNR values before casting, which would otherwise truncate or wrap them.
    rounded = np.rint(numeric)
    limits = np.iinfo(np.int16)
    if (
        not np.all(np.isfinite(numeric))
        or not np.array_equal(numeric, rounded)
        or np.any(rounded < limits.min)
        or np.any(rounded > limits.max)
    ):
        raise ValueError(
            "RadioML 2018 SNR labels must be integral int16 values"
        )
    return np.asarray(rounded, dtype=np.int16)


def _validated_modulation_classes(
    modulation_classes: Sequence[str],
) -> tuple[str, ...]:
    """Validate and copy a RadioML modulation-class sequence.

    Args:
        modulation_classes: Class names in one-hot column order.

    Returns:
        Validated modulation classes.

    Raises:
        ValueError: If the sequence is empty, contains invalid or duplicate
            names, or has more names than the `int16` class index holds.
    """
    classes = tuple(modulation_classes)
    if not classes or any(not value for value in classes):
        raise ValueError("modulation_classes must contain non-empty names")
    if len(set(classes)) != len(classes):
        raise ValueError("modulation_classes must not contain duplicates")
    if len(classes) > _MAX_MODULATION_CLASSES:
        raise ValueError(
            f"modulation_classes has {len(classes)} names. The int16 class "
            f"index holds at most {_MAX_MODULATION_CLASSES}"
        )
    return classes


_MAX_HDF5_SOFT_LINKS = 16
"""Soft links followed while resolving one HDF5 path, as in HDF5's default."""


def _hdf5_local_object(source: h5py.File, path: str) -> object | None:
    """Resolve an HDF5 path without leaving the source file.

    Each path component is inspected before it is traversed. Hard links are
    followed. Soft links are resolved within the file by their target path.
    External links are rejected without opening the file that they name.

    Args:
        source: Open source HDF5 file.
        path: Object path within the file.

    Returns:
        Resolved HDF5 object, or `None` if a component is missing.

    Raises:
        ValueError: If the path uses an external or user-defined link, or
            follows too many soft links.
    """
    root = source["/"]
    current: object = root
    pending = [part for part in path.split("/") if part not in {"", "."}]
    soft_links = 0
    while pending:
        name = pending.pop(0)
        if not isinstance(current, h5py.Group):
            return None
        encoded = name.encode("utf-8", "surrogateescape")
        links = current.id.links
        if not links.exists(encoded):
            return None
        link_type = links.get_info(encoded).type
        if link_type == h5py.h5l.TYPE_HARD:
            current = current[encoded]
        elif link_type == h5py.h5l.TYPE_SOFT:
            soft_links += 1
            if soft_links > _MAX_HDF5_SOFT_LINKS:
                raise ValueError(
                    f"HDF5 path {path!r} follows more than "
                    f"{_MAX_HDF5_SOFT_LINKS} soft links"
                )
            target = links.get_val(encoded).decode("utf-8", "surrogateescape")
            if target.startswith("/"):
                current = root
            pending[:0] = [
                part for part in target.split("/") if part not in {"", "."}
            ]
        else:
            # External and user-defined links can name objects in other
            # files. Reject them before HDF5 opens any other file.
            raise ValueError(
                f"HDF5 path {path!r} uses an external link. RadioML 2018 "
                "import reads only objects stored in the input file"
            )
    return current


def _hdf5_dataset(
    source: h5py.File,
    path: str,
    *,
    label: str,
) -> h5py.Dataset:
    """Return one required HDF5 dataset stored in the source file.

    Args:
        source: Open source HDF5 file.
        path: Dataset path within the file.
        label: Human-readable dataset role for errors.

    Returns:
        Requested HDF5 dataset.

    Raises:
        ValueError: If the path does not identify a dataset, reaches it
            through an external link, or the dataset reads raw data from
            other files, uses an HDF5 filter other than gzip, shuffle, and
            fletcher32, or has chunks larger than the per-item limit.
    """
    dataset = _hdf5_local_object(source, path)
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"Missing HDF5 {label} dataset {path!r}")
    plist = dataset.id.get_create_plist()
    if plist.get_layout() == h5py.h5d.VIRTUAL:
        raise ValueError(
            f"HDF5 {label} dataset {path!r} is a virtual dataset. RadioML "
            "2018 import reads only data stored in the input file"
        )
    if plist.get_external_count() > 0:
        raise ValueError(
            f"HDF5 {label} dataset {path!r} uses external raw storage. "
            "RadioML 2018 import reads only data stored in the input file"
        )
    dataset_filters(dataset, description=f"HDF5 {label} dataset {path!r}")
    if dataset.chunks is not None:
        # A read decodes whole chunks, so the chunk size bounds the memory of
        # even a one-item read. HDF5RowReader inflates gzip chunks no further.
        chunk_bytes = math.prod(dataset.chunks) * dataset.dtype.itemsize
        if chunk_bytes > MAX_IMPORT_ITEM_BYTES:
            raise ValueError(
                f"HDF5 {label} dataset {path!r} has {chunk_bytes}-byte "
                f"chunks, more than the importer limit of "
                f"{MAX_IMPORT_ITEM_BYTES} bytes"
            )
    return dataset


def _source_layout(
    samples: h5py.Dataset,
    labels: h5py.Dataset,
    snr: h5py.Dataset,
    *,
    num_classes: int,
) -> tuple[tuple[int, int], bool, int]:
    """Validate aligned RadioML 2018 source array shapes.

    Args:
        samples: Source sample tensor.
        labels: Source one-hot label matrix.
        snr: Source SNR array.
        num_classes: Expected number of one-hot columns.

    Returns:
        Per-item sample shape, whether I/Q is last, and the item count.

    Raises:
        ValueError: If the source arrays are empty or misaligned, label or
            SNR values are not scalar numbers, or one row of any array
            exceeds the per-item byte limit.
    """
    sample_shape, iq_last = _sample_shape(samples)
    check_item_bytes(
        math.prod(sample_shape) * samples.dtype.itemsize,
        source="RadioML 2018 sample dataset",
    )
    for dataset, label in ((labels, "label"), (snr, "SNR")):
        # An HDF5 array or compound type can hide any number of values in one
        # element. Check element types before reading any label data.
        if dataset.dtype.subdtype is not None or dataset.dtype.kind not in {
            "i",
            "u",
            "f",
        }:
            raise ValueError(
                f"RadioML 2018 {label} values must be scalar integers or "
                f"floating-point numbers, got HDF5 type {dataset.dtype}"
            )
        check_item_bytes(
            math.prod(dataset.shape[1:]) * dataset.dtype.itemsize,
            source=f"RadioML 2018 {label} dataset",
        )
    item_count = int(samples.shape[0])
    if item_count <= 0:
        raise ValueError("RadioML 2018 sample dataset must be non-empty")
    if labels.shape != (item_count, num_classes):
        raise ValueError(
            "RadioML 2018 sample and label shapes are incompatible: "
            f"{samples.shape} and {labels.shape}"
        )
    if snr.shape not in {(item_count,), (item_count, 1)}:
        raise ValueError(
            "RadioML 2018 sample and SNR shapes are incompatible: "
            f"{samples.shape} and {snr.shape}"
        )
    return sample_shape, iq_last, item_count


def _read_label_arrays(
    labels: HDF5RowReader,
    snr: HDF5RowReader,
    *,
    row_bytes: int,
    item_count: int,
    num_classes: int,
    batch_size: int,
) -> tuple[npt.NDArray[np.int16], npt.NDArray[np.int16]]:
    """Read and validate RadioML 2018 labels in bounded batches.

    Args:
        labels: Reader for the source one-hot label matrix.
        snr: Reader for the source SNR array.
        row_bytes: Combined size of one label row and one SNR row.
        item_count: Number of aligned items.
        num_classes: Expected number of one-hot columns.
        batch_size: Maximum number of items per HDF5 read.

    Returns:
        Modulation-class IDs and SNR values.

    Raises:
        ValueError: If labels are invalid, or the declared item count is too
            large to hold the label arrays in memory.
    """
    # Labels are small enough to retain in memory. Samples are streamed later
    # because they dominate the source file size.
    try:
        mod_class_id = np.empty((item_count,), dtype=np.int16)
        snr_db = np.empty((item_count,), dtype=np.int16)
    except MemoryError as exc:
        raise ValueError(
            f"RadioML 2018 input declares {item_count} items, too many to "
            "hold their labels in memory"
        ) from exc
    step = batch_items(row_bytes, batch_size)
    for start in range(0, item_count, step):
        stop = min(start + step, item_count)
        mod_class_id[start:stop] = _modulation_ids(
            labels.read(start, stop),
            num_classes=num_classes,
        )
        snr_db[start:stop] = _snr_values(snr.read(start, stop))
    return mod_class_id, snr_db


def _append_sample_batches(
    recording: SigMFRecording,
    samples: HDF5RowReader,
    *,
    item_bytes: int,
    item_count: int,
    batch_size: int,
    iq_last: bool,
) -> None:
    """Append RadioML 2018 samples in bounded HDF5 batches.

    Each batch holds at most `batch_size` items and at most the importer's
    batch byte budget, but always at least one item.

    Args:
        recording: Target recording with an `append_samples` method.
        samples: Reader for the source sample tensor.
        item_bytes: Size of one source sample item.
        item_count: Number of aligned items.
        batch_size: Maximum number of items per HDF5 read.
        iq_last: Whether the source stores I/Q on its final axis.
    """
    step = batch_items(item_bytes, batch_size)
    for start in range(0, item_count, step):
        stop = min(start + step, item_count)
        iq_batch = samples.read(start, stop)
        if iq_last:
            # SigMF-Zarr uses the canonical per-item layout (I/Q, time).
            iq_batch = np.moveaxis(iq_batch, 2, 1)
        recording.append_samples(iq_batch)


@remove_new_store_on_failure
def import_radioml2018_dataset(
    store_path: str | PathLike[str],
    source_path: str | PathLike[str],
    *,
    source_dataset: str | None = None,
    dataset_version: str = "2018",
    recording_name: str = "radioml2018",
    overwrite_store: bool = False,
    overwrite_recording: bool = False,
    modulation_classes: Sequence[str] = RADIOML2018_MODULATION_CLASSES,
    samples_dataset: str = "X",
    labels_dataset: str = "Y",
    snr_dataset: str = "Z",
    batch_size: int = 4096,
    global_metadata: JSONObject | None = None,
    iq_chunks: tuple[int, int, int] | None = None,
    sample_shards: ShardsLike | None = None,
    sample_compressor: CompressorLike = "auto",
    zarr_format: ZarrFormat | None = None,
) -> SigMFZarrStore:
    """Stream a RadioML 2018 HDF5 dataset into SigMF-Zarr.

    Args:
        store_path: Target SigMF-Zarr store.
        source_path: Source HDF5 path.
        source_dataset: Optional source dataset name for metadata.
        dataset_version: Dataset generation recorded in source metadata.
            Defaults to "2018".
        recording_name: Recording name to create.
        overwrite_store: Whether to replace an existing SigMF-Zarr store at
            `store_path`. The target must be absent, an empty directory, or
            a SigMF-Zarr store, and must not contain a source file.
        overwrite_recording: Whether to replace an existing recording.
        modulation_classes: Class names in one-hot column order.
        samples_dataset: HDF5 path containing samples.
        labels_dataset: HDF5 path containing modulation labels.
        snr_dataset: HDF5 path containing SNR labels.
        batch_size: HDF5 read batch size.
        global_metadata: Optional SigMF global metadata to merge.
        iq_chunks: Optional IQ sample chunks.
        sample_shards: Optional IQ sample shards.
        sample_compressor: Optional IQ sample compressor.
        zarr_format: Optional physical Zarr format requirement. Existing
            stores are auto-detected when omitted. New stores default to Zarr
            format 3.

    Returns:
        Updated SigMF-Zarr store.

    Raises:
        ValueError: If the source arrays or labels are invalid, or sample
            sharding is requested for Zarr format 2.
    """
    if zarr_format == 2 and sample_shards is not None:
        raise ValueError(
            "Sample sharding requires Zarr format 3. Omit sample_shards or "
            "create a Zarr format 3 store"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    classes = _validated_modulation_classes(modulation_classes)

    # Reject unlisted filters before reading, and keep HDF5 from loading a
    # plugin for one regardless.
    with hdf5_plugins_disabled(), h5py.File(Path(source_path), "r") as source:
        samples = _hdf5_dataset(source, samples_dataset, label="sample")
        labels = _hdf5_dataset(source, labels_dataset, label="label")
        snr = _hdf5_dataset(source, snr_dataset, label="SNR")
        if samples.dtype.kind != "f" or samples.dtype.itemsize not in {4, 8}:
            raise ValueError(
                "RadioML 2018 components must be float32 or float64"
            )
        datatype = f"cf{samples.dtype.itemsize * 8}_le"
        sample_shape, iq_last, item_count = _source_layout(
            samples,
            labels,
            snr,
            num_classes=len(classes),
        )
        sample_reader = HDF5RowReader(
            samples, description=f"HDF5 sample dataset {samples_dataset!r}"
        )
        # Validate every label before creating or replacing destination data.
        # A malformed late label must not be discovered after copying samples.
        mod_class_id, snr_db = _read_label_arrays(
            HDF5RowReader(
                labels, description=f"HDF5 label dataset {labels_dataset!r}"
            ),
            HDF5RowReader(
                snr, description=f"HDF5 SNR dataset {snr_dataset!r}"
            ),
            row_bytes=labels.dtype.itemsize * len(classes)
            + snr.dtype.itemsize * math.prod(snr.shape[1:]),
            item_count=item_count,
            num_classes=len(classes),
            batch_size=batch_size,
        )

        metadata = dict(global_metadata or {})
        if metadata.get("core:datatype", datatype) != datatype:
            raise ValueError(
                f"RadioML 2018 output requires core:datatype {datatype}"
            )
        metadata["core:datatype"] = datatype
        metadata["radioml:source_dataset"] = (
            source_dataset or Path(source_path).stem
        )
        metadata["radioml:dataset_version"] = dataset_version
        metadata = import_metadata(
            metadata,
            sources=[
                file_identity(Path(source_path), role="samples-and-truth")
            ],
            operation="import_radioml2018_dataset",
            source_namespace="radioml",
            parameters={
                "samples_dataset": samples_dataset,
                "labels_dataset": labels_dataset,
                "snr_dataset": snr_dataset,
                "source_rows": [0, item_count],
                "modulation_classes": list(classes),
                "source_iq_last": iq_last,
                "sample_conversion": "preserve-component-precision",
                "sample_axes": ["iq", "time"],
            },
        )
        store = SigMFZarrStore(
            store_path,
            overwrite=overwrite_store,
            zarr_format=zarr_format,
        )
        if recording_name in store.recordings and not overwrite_recording:
            raise ValueError(
                f"Recording {recording_name!r} already exists. Pass "
                "overwrite_recording=True to replace it"
            )
        with recording_import_transaction(
            store, recording_name, overwrite=overwrite_recording
        ):
            recording = store.recordings.create(
                recording_name,
                batched=True,
                sample_dtype=samples.dtype,
                sample_shape=sample_shape,
                sample_axes=("iq", "time"),
                global_metadata=metadata,
                sample_chunks=iq_chunks,
                sample_shards=sample_shards,
                sample_compressor=sample_compressor,
                overwrite=overwrite_recording,
            )
            _append_sample_batches(
                recording,
                sample_reader,
                item_bytes=math.prod(samples.shape[1:])
                * samples.dtype.itemsize,
                item_count=item_count,
                batch_size=batch_size,
                iq_last=iq_last,
            )

            recording.add_index(
                "mod_class_id",
                mod_class_id,
                axis="item",
                field="radioml:mod_class",
                labels=list(classes),
                overwrite=overwrite_recording,
            )
            recording.add_index(
                "snr_db",
                snr_db,
                axis="item",
                field="radioml:snr",
                unit="dB",
                overwrite=overwrite_recording,
            )
            recording.update_integrity()
            store._update_metadata_integrity_when_complete()
        return store


def load_modulation_classes(path: Path | None) -> tuple[str, ...]:
    """Load an explicit RadioML 2018 class order.

    A file with a `.json` suffix, or whose text begins with `[`, must hold a
    JSON string array. Text that begins with `classes` must be a Python-style
    assignment. Any other text names one class per non-empty line.

    Args:
        path: Optional JSON, Python-assignment, or line-delimited class file.

    Returns:
        Modulation classes in one-hot column order.

    Raises:
        ValueError: If the class file is invalid.
    """
    if path is None:
        return RADIOML2018_MODULATION_CLASSES
    text = path.read_text(encoding="utf-8")
    # Detect JSON by content as well, so that an array saved under another
    # suffix is not read as lines such as '["OOK",'.
    if path.suffix == ".json" or text.lstrip().startswith("["):
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON RadioML class file {str(path)!r}: {exc}"
            ) from exc
    elif text.lstrip().startswith("classes"):
        # Parse the distributed Python-style file without executing it.
        try:
            module = ast.parse(text, filename=str(path), mode="exec")
            statement = module.body[0] if len(module.body) == 1 else None
            if not (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == "classes"
            ):
                raise ValueError
            value = ast.literal_eval(statement.value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                "RadioML class assignment must have the form "
                "`classes = ['CLASS', ...]`"
            ) from exc
    else:
        value = [line.strip() for line in text.splitlines() if line.strip()]
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError("RadioML classes must be a list of strings")
    classes = tuple(value)
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("RadioML class names must be non-empty and unique")
    return classes


__all__ = [
    "RADIOML2018_MODULATION_CLASSES",
    "import_radioml2018_dataset",
    "load_modulation_classes",
]

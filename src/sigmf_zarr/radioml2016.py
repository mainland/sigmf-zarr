"""Import RadioML 2016 pickle datasets into SigMF-Zarr."""

from __future__ import annotations

import logging
import pickle
import warnings
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from zarr.core.array import CompressorLike, ShardsLike

from sigmf_zarr.integrity import calculate_array_sha512
from sigmf_zarr.json import JSONObject
from sigmf_zarr.provenance import import_metadata
from sigmf_zarr.store import SigMFZarrStore, ZarrFormat
from sigmf_zarr.store._transaction import recording_import_transaction

logger = logging.getLogger(__name__)

RadioML2016Key = tuple[str, int]
"""RadioML 2016 mapping key of `(modulation_class, snr_db)`."""

RadioML2016Value = npt.NDArray[np.floating[Any]]
"""RadioML 2016 IQ batch with shape `(N, 2, T)`."""

RadioML2016Dict = Mapping[RadioML2016Key, RadioML2016Value]
"""Canonical in-memory RadioML 2016 pickle mapping."""


def load_radioml2016_pickle(path: Path, *, encoding: str) -> object:
    """Load a legacy RadioML 2016 pickle file.

    Args:
        path: Pickle file path.
        encoding: String encoding for Python 2 pickle compatibility.

    Returns:
        Deserialized pickle object.
    """
    logger.warning(
        "Loading trusted pickle input %s. Pickle files can execute code",
        path,
    )
    with path.open("rb") as handle, warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"dtype\(\): align should be passed as Python or NumPy "
                r"boolean but got `align=0`"
            ),
            category=np.exceptions.VisibleDeprecationWarning,
        )
        return pickle.load(handle, encoding=encoding)


def as_radioml2016_dict(obj: object) -> RadioML2016Dict:
    """Validate a candidate RadioML 2016 pickle mapping.

    Args:
        obj: Candidate mapping.

    Returns:
        Validated RadioML 2016 mapping.

    Raises:
        TypeError: If the object does not have the expected mapping shape.
    """
    if not isinstance(obj, Mapping):
        raise TypeError(
            "Expected a mapping from (mod_class, snr_db) to IQ arrays."
        )
    if not obj:
        raise TypeError("Expected the RadioML mapping to be non-empty.")

    for key, value in obj.items():
        if not (
            isinstance(key, tuple)
            and len(key) == 2
            and isinstance(key[0], str)
            and isinstance(key[1], int)
        ):
            raise TypeError(
                "Expected keys of the form "
                "(mod_class: str, snr_db: int)."
            )
        if (
            not isinstance(value, np.ndarray)
            or value.ndim != 3
            or value.shape[1] != 2
        ):
            shape = getattr(value, "shape", None)
            raise TypeError(
                "Expected values to be 3D NumPy arrays with shape "
                f"(N, 2, T), got shape {shape}"
            )
    return cast(RadioML2016Dict, obj)


def flatten_radioml2016_dataset(
    dataset: RadioML2016Dict,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.int16],
    npt.NDArray[np.int16],
    list[str],
]:
    """Flatten a RadioML 2016 mapping into aligned arrays.

    Args:
        dataset: Mapping keyed by `(modulation_class, snr_db)`.

    Returns:
        IQ samples, modulation IDs, SNR labels, and modulation names.

    Raises:
        ValueError: If any batch has an incompatible shape.
    """
    # Pickle mappings preserve no portable class-ID convention. Sorting makes
    # both item order and the generated label table deterministic.
    mod_classes = sorted(set(mod_class for mod_class, _ in dataset))
    mod_class_to_id = {
        mod_class: index for index, mod_class in enumerate(mod_classes)
    }
    iq_batches: list[npt.NDArray[np.float32]] = []
    mod_labels: list[npt.NDArray[np.int16]] = []
    snr_labels: list[npt.NDArray[np.int16]] = []

    for mod_class, snr_value in sorted(dataset):
        batch = np.asarray(dataset[(mod_class, snr_value)], dtype=np.float32)
        if batch.ndim != 3 or batch.shape[1] != 2:
            raise ValueError(
                "Expected shape (N, 2, T) for "
                f"({mod_class}, {snr_value}), got {batch.shape}"
            )
        count = int(batch.shape[0])
        iq_batches.append(batch)
        mod_labels.append(
            np.full(
                (count,),
                mod_class_to_id[mod_class],
                dtype=np.int16,
            )
        )
        snr_labels.append(
            np.full((count,), snr_value, dtype=np.int16)
        )

    return (
        np.concatenate(iq_batches, axis=0),
        np.concatenate(mod_labels, axis=0),
        np.concatenate(snr_labels, axis=0),
        mod_classes,
    )


def _radioml2016_layout(
    dataset: RadioML2016Dict,
) -> tuple[tuple[int, ...], int, list[str]]:
    """Validate shapes and labels before creating destination storage.

    Args:
        dataset: Validated RadioML mapping.

    Returns:
        Per-item shape, total item count, and sorted modulation names.

    Raises:
        ValueError: If sample shapes or labels cannot be represented.
    """
    sample_shape = tuple(next(iter(dataset.values())).shape[1:])
    if any(size <= 0 for size in sample_shape):
        raise ValueError("RadioML sample dimensions must be positive")
    if any(batch.shape[1:] != sample_shape for batch in dataset.values()):
        raise ValueError("RadioML batches must have the same sample shape")
    item_count = sum(int(batch.shape[0]) for batch in dataset.values())
    if not item_count:
        raise ValueError("RadioML dataset must contain at least one item")
    mod_classes = sorted({mod_class for mod_class, _ in dataset})
    if len(mod_classes) > np.iinfo(np.int16).max + 1:
        raise ValueError("RadioML modulation IDs must fit in int16")
    for _, snr_value in dataset:
        if not np.iinfo(np.int16).min <= snr_value <= np.iinfo(np.int16).max:
            raise ValueError("RadioML SNR labels must fit in int16")
    return sample_shape, item_count, mod_classes


def import_radioml2016_dataset(
    store_path: str | PathLike[str],
    dataset: RadioML2016Dict | object,
    *,
    source_dataset: str | None = None,
    dataset_version: str = "2016",
    recording_name: str = "radioml2016",
    overwrite_store: bool = False,
    overwrite_recording: bool = False,
    batch_size: int = 4096,
    global_metadata: JSONObject | None = None,
    iq_chunks: tuple[int, int, int] | None = None,
    sample_shards: ShardsLike | None = None,
    sample_compressor: CompressorLike = "auto",
    zarr_format: ZarrFormat | None = None,
) -> SigMFZarrStore:
    """Import a pickle mapping in the RadioML 2016 format.

    Args:
        store_path: Target SigMF-Zarr store.
        dataset: RadioML 2016 mapping or candidate object.
        source_dataset: Optional source dataset name for metadata.
        dataset_version: Dataset generation recorded in source metadata.
            Defaults to "2016".
        recording_name: Recording name to create.
        overwrite_store: Whether to recreate the target store.
        overwrite_recording: Whether to replace an existing recording.
        batch_size: Maximum number of sample items copied per write.
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
        TypeError: If the object is not a supported pickle mapping.
        ValueError: If shapes, labels, or the batch size are invalid, or sample
            sharding is requested for Zarr format 2.
    """
    if zarr_format == 2 and sample_shards is not None:
        raise ValueError(
            "Sample sharding requires Zarr format 3. Omit sample_shards or "
            "create a Zarr format 3 store"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    dataset = as_radioml2016_dict(dataset)
    sample_shape, item_count, mod_classes = _radioml2016_layout(dataset)
    mod_class_to_id = {
        mod_class: index for index, mod_class in enumerate(mod_classes)
    }
    mod_class_id = np.empty(item_count, dtype=np.int16)
    snr_db = np.empty(item_count, dtype=np.int16)
    metadata = dict(global_metadata or {})
    if metadata.get("core:datatype", "cf32_le") != "cf32_le":
        raise ValueError("RadioML 2016 output requires core:datatype cf32_le")
    metadata["core:datatype"] = "cf32_le"
    metadata["radioml:source_dataset"] = source_dataset
    metadata["radioml:dataset_version"] = dataset_version

    sources: list[JSONObject] = []
    offset = 0
    for (modulation, snr), values in sorted(dataset.items()):
        sources.append({
            "kind": "logical-array", "role": "samples-and-labels",
            "sha512": calculate_array_sha512(values),
            "modulation": modulation, "snr_db": snr,
            "items": int(values.shape[0]), "target_start": offset,
        })
        offset += int(values.shape[0])
    metadata = import_metadata(
        metadata, sources=sources, operation="import_radioml2016_dataset",
        source_namespace="radioml",
        parameters={
            "order": "sorted-modulation-snr-then-source-row",
            "modulation_classes": list(mod_classes),
            "sample_conversion": "float32", "sample_axes": ["iq", "time"],
        },
    )

    store = SigMFZarrStore.create(
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
        recording = store.recordings.open(
            recording_name,
            create=True,
            batched=True,
            sample_dtype=np.float32,
            sample_shape=sample_shape,
            sample_axes=("iq", "time"),
            global_metadata=metadata,
            sample_chunks=iq_chunks,
            sample_shards=sample_shards,
            sample_compressor=sample_compressor,
            overwrite=overwrite_recording,
        )
        # Walk the mapping in the same sorted order used to assign class IDs.
        # offset tracks appended items across source batches so both indexes
        # stay aligned without concatenating another full sample tensor.
        offset = 0
        for mod_class, snr_value in sorted(dataset):
            batch = dataset[(mod_class, snr_value)]
            count = int(batch.shape[0])
            mod_class_id[offset:offset + count] = mod_class_to_id[mod_class]
            snr_db[offset:offset + count] = snr_value
            for start in range(0, count, batch_size):
                recording.append_samples(
                    np.asarray(
                        batch[start:start + batch_size], dtype=np.float32
                    )
                )
            offset += count
        recording.add_index(
            "mod_class_id",
            mod_class_id,
            axis="item",
            field="radioml:mod_class",
            labels=list(mod_classes),
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
        store.update_metadata_integrity()
    return store


__all__ = [
    "RadioML2016Dict",
    "RadioML2016Key",
    "RadioML2016Value",
    "as_radioml2016_dict",
    "flatten_radioml2016_dataset",
    "import_radioml2016_dataset",
    "load_radioml2016_pickle",
]

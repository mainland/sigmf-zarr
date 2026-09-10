"""Import RadioML 2016 pickle datasets into SigMF-Zarr."""

from __future__ import annotations

import pickle
import warnings
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from zarr.core.array import CompressorLike, ShardsLike

from sigmf_zarr._rfml import _with_rfml_profile
from sigmf_zarr.integrity import calculate_array_sha512
from sigmf_zarr.json import JSONObject
from sigmf_zarr.provenance import import_metadata
from sigmf_zarr.sample_storage import resolve_import_sample_storage
from sigmf_zarr.store import SigMFZarrStore, ZarrFormat
from sigmf_zarr.store._transaction import recording_import_transaction

RadioML2016Key = tuple[str, int]
"""RadioML 2016 mapping key of `(modulation_class, snr_db)`."""

RadioML2016Value = npt.NDArray[np.floating[Any]]
"""RadioML 2016 IQ batch with shape `(N, 2, T)`."""

RadioML2016Dict = Mapping[RadioML2016Key, RadioML2016Value]
"""Canonical in-memory RadioML 2016 pickle mapping."""

_NUMPY_PICKLE_GLOBALS = frozenset(
    {
        ("_codecs", "encode"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy._core.numeric", "_frombuffer"),
    }
)
"""Globals that NumPy array pickles reference, after module renaming."""


class _NumPyUnpickler(pickle.Unpickler):
    """Unpickler that reconstructs only containers and NumPy arrays."""

    def find_class(self, module: str, name: str) -> Any:
        """Resolve an allowed global used to rebuild NumPy arrays.

        Python 2 pickles name `numpy.core`, which NumPy 2 exposes only
        through a deprecated alias. Resolve those names in `numpy._core`.

        Args:
            module: Module named by the pickle.
            name: Global name within the module.

        Returns:
            Allowed global object.

        Raises:
            pickle.UnpicklingError: If the pickle names any other global.
        """
        if module.startswith("numpy.core."):
            module = "numpy._core." + module.removeprefix("numpy.core.")
        if (module, name) not in _NUMPY_PICKLE_GLOBALS:
            raise pickle.UnpicklingError(f"unsupported global {module}.{name}")
        return super().find_class(module, name)


def load_radioml2016_pickle(path: Path, *, encoding: str) -> object:
    """Load a legacy RadioML 2016 pickle file.

    Unpickling is restricted to built-in containers and NumPy arrays. A
    pickle that names any other global is rejected before that global is
    resolved, so loading cannot call arbitrary functions. A pickle stores
    array bytes verbatim, so the arrays in a mapping cannot hold more data
    than the file. A larger total means arrays share recorded data, which
    would let a small file expand into a large import, and is rejected.

    Args:
        path: Pickle file path.
        encoding: String encoding for Python 2 pickle compatibility.

    Returns:
        Deserialized pickle object.

    Raises:
        ValueError: If the pickle references an unsupported global, is
            malformed, or holds arrays with more data than the file.
    """
    with path.open("rb") as handle, warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"dtype\(\): align should be passed as Python or NumPy "
                r"boolean but got `align=0`"
            ),
            category=np.exceptions.VisibleDeprecationWarning,
        )
        try:
            result = _NumPyUnpickler(handle, encoding=encoding).load()
        except (EOFError, pickle.UnpicklingError) as exc:
            raise ValueError(f"Invalid RadioML pickle {path}: {exc}") from exc
    if isinstance(result, Mapping):
        array_bytes = sum(
            value.nbytes
            for value in result.values()
            if isinstance(value, np.ndarray)
        )
        if array_bytes > path.stat().st_size:
            raise ValueError(
                f"Invalid RadioML pickle {path}: arrays hold {array_bytes} "
                f"bytes, more than the {path.stat().st_size}-byte file"
            )
    return result


def as_radioml2016_dict(obj: object) -> RadioML2016Dict:
    """Validate a candidate RadioML 2016 pickle mapping.

    Args:
        obj: Candidate mapping.

    Returns:
        Validated RadioML 2016 mapping.

    Raises:
        TypeError: If the object does not have the expected mapping shape or
            an array is not real numeric data.
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
                "Expected keys of the form (mod_class: str, snr_db: int)."
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
        # Object arrays would hold arbitrary Python values, and complex or
        # structured values have no lossless float32 sample representation.
        if value.dtype.kind not in "iuf" or value.dtype.fields is not None:
            raise TypeError(
                f"Expected real numeric IQ arrays, got dtype {value.dtype}"
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
    mod_classes = sorted({mod_class for mod_class, _ in dataset})
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
        snr_labels.append(np.full((count,), snr_value, dtype=np.int16))

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
    automatic_sharding: bool = True,
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
        automatic_sharding: Whether to derive format-3 shards when explicit
            shards are not supplied.
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
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    dataset = as_radioml2016_dict(dataset)
    sample_shape, item_count, mod_classes = _radioml2016_layout(dataset)
    if any(not label for label in mod_classes):
        raise ValueError("Modulation class labels must be nonempty strings")
    mod_class_to_id = {
        mod_class: index for index, mod_class in enumerate(mod_classes)
    }
    mod_class_id = np.empty(item_count, dtype=np.int16)
    snr_db = np.empty(item_count, dtype=np.int16)
    metadata = _with_rfml_profile(global_metadata)
    if metadata.get("core:datatype", "cf32_le") != "cf32_le":
        raise ValueError("RadioML 2016 output requires core:datatype cf32_le")
    metadata["core:datatype"] = "cf32_le"
    metadata["radioml:source_dataset"] = source_dataset
    metadata["radioml:dataset_version"] = dataset_version

    sources: list[JSONObject] = []
    offset = 0
    for (modulation, snr), values in sorted(dataset.items()):
        sources.append(
            {
                "kind": "logical-array",
                "role": "samples-and-labels",
                "sha512": calculate_array_sha512(values),
                "modulation": modulation,
                "snr_db": snr,
                "items": int(values.shape[0]),
                "target_start": offset,
            }
        )
        offset += int(values.shape[0])
    metadata = import_metadata(
        metadata,
        sources=sources,
        operation="import_radioml2016_dataset",
        source_namespace="radioml",
        parameters={
            "order": "sorted-modulation-snr-then-source-row",
            "modulation_classes": list(mod_classes),
            "sample_conversion": "float32",
            "sample_axes": ["iq", "time"],
        },
    )

    store = SigMFZarrStore(
        store_path,
        overwrite=overwrite_store,
        zarr_format=zarr_format,
    )
    resolved_iq_chunks, resolved_sample_shards = resolve_import_sample_storage(
        np.float32,
        sample_shape,
        item_count,
        batched=True,
        zarr_format=store.zarr_format,
        sample_chunks=iq_chunks,
        sample_shards=sample_shards,
        automatic_sharding=automatic_sharding,
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
            sample_dtype=np.float32,
            sample_shape=sample_shape,
            sample_axes=("iq", "time"),
            global_metadata=metadata,
            sample_chunks=resolved_iq_chunks,
            sample_shards=resolved_sample_shards,
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
            mod_class_id[offset : offset + count] = mod_class_to_id[mod_class]
            snr_db[offset : offset + count] = snr_value
            for start in range(0, count, batch_size):
                recording.append_samples(
                    np.asarray(
                        batch[start : start + batch_size], dtype=np.float32
                    )
                )
            offset += count
        recording.add_index(
            "mod_class_id",
            mod_class_id,
            axis="item",
            field="rfml-dataset:modulation",
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
        store._update_metadata_integrity_when_complete()
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

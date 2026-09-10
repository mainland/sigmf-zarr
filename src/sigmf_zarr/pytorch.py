"""Optional CPU tensor adapter for dense, batched recordings."""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable, Mapping, Sequence
from operator import index as integer_index
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import numpy as np
import numpy.typing as npt
from torch import Tensor, from_numpy
from torch.utils.data import Dataset

from sigmf_zarr.indexes import validate_categorical_values
from sigmf_zarr.json import JSONObject, json_value
from sigmf_zarr.readonly import ReadOnlyArray
from sigmf_zarr.store import SigMFRecording, SigMFZarrStore


class RecordingItem(TypedDict):
    """One dense recording item, suitable for PyTorch default collation."""

    samples: Tensor
    """CPU samples in stored axis order, before optional transforms."""
    targets: dict[str, Tensor]
    """Explicitly selected numerical item targets."""

    item_index: int
    """Original position in the source recording."""


def _tensor_dtype(dtype: np.dtype[Any]) -> None:
    """Check that a numerical dtype has a lossless native PyTorch mapping.

    Args:
        dtype: Stored NumPy dtype.

    Raises:
        ValueError: If PyTorch cannot represent the numerical dtype.
    """
    if dtype.kind not in "biufc":
        raise ValueError(f"Unsupported tensor dtype: {dtype}")
    try:
        from_numpy(np.empty(0, dtype=dtype.newbyteorder("=")))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unsupported tensor dtype: {dtype}") from error


def _tensor(values: npt.ArrayLike) -> Tensor:
    """Copy values into independent native-endian CPU tensor storage.

    Args:
        values: Numerical sample or target values.

    Returns:
        CPU tensor preserving shape, signedness, and precision.
    """
    array = np.asarray(values)
    # from_numpy shares its input buffer. Copy before exposing a writable
    # tensor so in-place transforms cannot modify a cached or shared read.
    return from_numpy(
        np.array(array, dtype=array.dtype.newbyteorder("="), copy=True)
    )


def _target_labels(array: ReadOnlyArray) -> tuple[str, ...] | None:
    """Validate the dense target descriptor and its optional lookup table.

    Args:
        array: Selected index array.

    Returns:
        Source string labels, or None for an uncategorized target.

    Raises:
        ValueError: If alignment, dtype, validity, or labels are unsupported.
    """
    if array.attrs.get("axis") != "item" or array.ndim != 1:
        raise ValueError("Targets must be one-dimensional item indexes")
    if "validity" in array.attrs:
        raise ValueError("Nullable item targets are not supported")
    _tensor_dtype(array.dtype)
    if "labels" not in array.attrs:
        return None
    table = validate_categorical_values(
        np.empty(0, dtype=array.dtype), labels=array.attrs["labels"]
    )
    if not table or any(
        not isinstance(value, str) or not value for value in table
    ):
        raise ValueError("Target labels must be nonempty strings")
    labels = cast(list[str], table)
    if len(set(labels)) != len(labels):
        raise ValueError("Target labels must be unique")
    return tuple(labels)


def _source_selection(
    recording: SigMFRecording,
    split: tuple[str, str | Sequence[str]] | None,
) -> npt.NDArray[np.intp]:
    """Select checked split assignments in source order without group scans.

    Args:
        recording: Source batched recording.
        split: Explicit split index and partition labels, or all items.

    Returns:
        Original item positions in source order.

    Raises:
        ValueError: If split selection or stored assignments are invalid.
    """
    if split is None:
        return np.arange(len(recording), dtype=np.intp)
    if not isinstance(split, tuple) or len(split) != 2:
        raise ValueError("split must be None or an (index, partitions) tuple")
    name, partitions = split
    selected = (
        [partitions] if isinstance(partitions, str) else list(partitions)
    )
    view = recording.split(name)
    labels = view.labels
    if not selected or any(part not in labels for part in selected):
        raise ValueError("Select one or more named split partitions")
    ids = [labels.index(part) for part in selected]
    array = view.assignments
    positions = []
    for start in range(0, len(recording), 65536):
        values = np.asarray(array[start : start + 65536])
        validate_categorical_values(values, labels=list(labels))
        # flatnonzero returns batch-local offsets. Translate them back to
        # recording positions before concatenating the selected batches.
        positions.append(np.flatnonzero(np.isin(values, ids)) + start)
    return (
        np.concatenate(positions).astype(np.intp, copy=False)
        if positions
        else np.empty(0, dtype=np.intp)
    )


class RecordingDataset(Dataset[RecordingItem]):
    """Read dense recording items with explicit targets and split selection.

    The source must remain unchanged while this dataset or its loaders
    are in use. Handles open lazily per process. Remote URLs require
    spawn or forkserver workers. Import this module after installing the
    pytorch extra.
    """

    sample_axes: tuple[str, ...]
    """Stored per-item sample axes, before transforms."""

    _path: str
    """Serializable source path or URL."""

    _recording_name: str
    """Recording name relative to the store."""

    _targets: dict[str, str]
    """Output target names mapped to source indexes."""

    _options: JSONObject | None
    """Detached JSON backend configuration."""

    _source_indices: npt.NDArray[np.intp]
    """Selected source positions in recording order."""

    _labels: dict[str, tuple[str, ...] | None]
    """Validated source lookup tables by output name."""

    _sample_transform: Callable[[Tensor], Tensor] | None
    """Optional sample-only transform."""

    _target_transforms: dict[str, Callable[[Tensor], Tensor]]
    """Optional transforms for selected item targets."""

    _item_transform: Callable[[RecordingItem], RecordingItem] | None
    """Optional joint sample and target transform."""

    _pid: int | None
    """Process that owns the cached handles."""

    _store: SigMFZarrStore | None
    """Process-local read-only store, excluded from serialization."""

    _recording: SigMFRecording | None
    """Process-local recording, excluded from serialization."""

    _samples: ReadOnlyArray | None
    """Process-local sample handle, excluded from serialization."""

    _arrays: dict[str, ReadOnlyArray]
    """Process-local target handles, excluded from serialization."""

    def __init__(
        self,
        store: str | os.PathLike[str],
        *,
        recording: str,
        targets: Mapping[str, str],
        split: tuple[str, str | Sequence[str]] | None,
        storage_options: Mapping[str, object] | None = None,
        sample_transform: Callable[[Tensor], Tensor] | None = None,
        item_target_transforms: Mapping[str, Callable[[Tensor], Tensor]]
        | None = None,
        item_transform: Callable[[RecordingItem], RecordingItem] | None = None,
    ) -> None:
        """Inspect descriptors and select source positions.

        Args:
            store: Local path or remote URL, not an open store.
            recording: Batched recording name.
            targets: Output names mapped to item index names. May be empty.
            split: Split index and partition labels, or None for all items.
            storage_options: JSON backend configuration, without live clients.
            sample_transform: CPU sample transform, applied after conversion.
            item_target_transforms: CPU transforms keyed by output target name.
            item_transform: Joint CPU transform applied last.

        Raises:
            TypeError: If configuration has unsupported types.
            ValueError: If layout, selection, or descriptors are unsupported.
            KeyError: If the recording or a selected index does not exist.
            RuntimeError: If a remote store is used in a fork worker.
        """
        path = os.fspath(store)
        if not isinstance(path, str):
            raise TypeError("store must be a string path or URL")
        self._path = path if "://" in path else str(Path(path).absolute())
        self._recording_name = recording
        self._targets = dict(targets)
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(index, str)
            or not index
            for name, index in self._targets.items()
        ):
            raise ValueError(
                "Target names and index names must be nonempty strings"
            )
        self._options = (
            cast(
                JSONObject,
                json_value(dict(storage_options), name="storage_options"),
            )
            if storage_options is not None
            else None
        )
        self._sample_transform = sample_transform
        self._target_transforms = dict(item_target_transforms or {})
        self._item_transform = item_transform
        if self._target_transforms.keys() - self._targets.keys():
            raise ValueError("Target transforms must name selected targets")
        self._pid = None
        self._store = None
        self._recording = None
        self._samples = None
        self._arrays = {}
        self._guard_worker()
        # Inspect with a temporary handle. The dataset retains serializable
        # descriptors and positions, while each loader process opens its own
        # long-lived handles on first access.
        inspection = self._open_store()
        try:
            source = inspection.recordings.open(
                recording, create=False, validation="structural"
            )
            if not source.batched:
                raise ValueError(
                    "RecordingDataset requires a batched recording"
                )
            _tensor_dtype(source.samples.dtype)
            self.sample_axes = source.sample_axes
            self._labels = {
                output: _target_labels(source.index(name))
                for output, name in self._targets.items()
            }
            self._source_indices = _source_selection(source, split)
        finally:
            inspection._group.store.close()

    @property
    def source_indices(self) -> npt.NDArray[np.intp]:
        """Return a detached array of original recording item positions.

        Returns:
            Source positions corresponding to dataset positions.
        """
        return self._source_indices.copy()

    def labels(
        self, scope: Literal["item"], name: str
    ) -> tuple[str, ...] | None:
        """Return the source lookup table for a selected output target.

        Args:
            scope: Target scope. The dense adapter supports only item.
            name: Output target name.

        Returns:
            String lookup table, or None when the index has no labels.

        Raises:
            ValueError: If the scope is unsupported.
            KeyError: If the output target is not selected.
        """
        if scope != "item":
            raise ValueError("Only item targets are supported")
        return self._labels[name]

    def _guard_worker(self) -> None:
        """Reject remote reads in forked workers before opening backend
        handles.

        Raises:
            RuntimeError: If a remote URL is used in a fork worker.
        """
        remote = "://" in self._path and not self._path.startswith("file://")
        if (
            remote
            and mp.parent_process() is not None
            and mp.get_start_method() == "fork"
        ):
            raise RuntimeError(
                "Remote stores require multiprocessing_context='spawn' "
                "or 'forkserver' in DataLoader"
            )

    def _open_store(self) -> SigMFZarrStore:
        """Open a new read-only store from serializable configuration.

        Returns:
            Read-only store.
        """
        return SigMFZarrStore.open(
            self._path, mode="r", storage_options=self._options
        )

    def _ensure_open(self) -> SigMFRecording:
        """Open process-owned handles without scanning item JSON.

        Returns:
            Structurally validated recording in this process.
        """
        self._guard_worker()
        if self._pid != os.getpid():
            # Fork can copy Python objects without serializing the dataset.
            # Discard inherited handles rather than treating a copied cache
            # as a backend opened by this process.
            self._store = None
            self._recording = None
            self._samples = None
            self._arrays = {}
        if self._recording is None:
            store = self._open_store()
            try:
                recording = store.recordings.open(
                    self._recording_name, create=False, validation="structural"
                )
                arrays = {
                    output: recording.index(name)
                    for output, name in self._targets.items()
                }
            except BaseException:
                store._group.store.close()
                raise
            self._store = store
            self._recording = recording
            self._samples = recording.samples
            self._arrays = arrays
            self._pid = os.getpid()
        return self._recording

    def close(self) -> None:
        """Release this process's backend handles, allowing later lazy
        reopen.
        """
        if self._pid == os.getpid() and self._store is not None:
            self._store._group.store.close()
        self._pid = None
        self._store = None
        self._recording = None
        self._samples = None
        self._arrays = {}
    def __getstate__(self) -> dict[str, Any]:
        """Exclude all live backend handles from worker serialization.

        Returns:
            Dataset configuration and selection without process-owned state.
        """
        state = self.__dict__.copy()
        state.update(
            _pid=None, _store=None, _recording=None, _samples=None, _arrays={}
        )
        return state

    def __len__(self) -> int:
        """Return the number of selected source items.

        Returns:
            Dataset length.
        """
        return len(self._source_indices)
    def _source_index(self, position: int) -> int:
        """Normalize a dataset position and resolve its original source index.

        Args:
            position: Integer dataset position, with normal negative indexing.

        Returns:
            Original source position.

        Raises:
            TypeError: If the position is not an integer or is Boolean.
            IndexError: If the position is outside the dataset.
        """
        if isinstance(position, bool | np.bool_):
            raise TypeError("Dataset positions must not be Boolean")
        return int(self._source_indices[integer_index(position)])
    def metadata(self, position: int) -> JSONObject:
        """Read resolved shared and local JSON for one selected item.

        Args:
            position: Dataset position.

        Returns:
            Validated JSON metadata without implicit index projection.

        Raises:
            IndexError: If the position is outside the dataset.
            ValueError: If the requested JSON entry is malformed.
        """
        source = self._source_index(position)
        return self._ensure_open().resolved_item_metadata(source)
    def __getitem__(self, position: int) -> RecordingItem:
        """Read and transform one item using scalar storage selection.

        Args:
            position: Dataset position.

        Returns:
            Independent CPU tensors and original source position.

        Raises:
            IndexError: If the position is outside the dataset.
            ValueError: If target values or transform outputs are invalid.
        """
        source = self._source_index(position)
        self._ensure_open()
        assert self._samples is not None
        values = {
            name: np.asarray(array[source])
            for name, array in self._arrays.items()
        }
        return self._make_item(source, self._samples[source], values)

    def __getitems__(self, positions: list[int]) -> list[RecordingItem]:
        """Read a batch with orthogonal selection, preserving order and
        repeats.

        Args:
            positions: Requested dataset positions in collator order.

        Returns:
            Ordinary items with independent transforms and tensor storage.

        Raises:
            IndexError: If any position is outside the dataset.
            ValueError: If target values or transform outputs are invalid.
        """
        sources = np.asarray(
            [self._source_index(i) for i in positions], dtype=np.intp
        )
        if not len(sources):
            return []
        self._ensure_open()
        assert self._samples is not None
        # Read in collator order, including repeats. Sorting or deduplicating
        # sources here would change the correspondence with requested items.
        samples = np.asarray(self._samples.oindex[sources])
        targets = {
            name: np.asarray(array.oindex[sources])
            for name, array in self._arrays.items()
        }
        return [
            self._make_item(
                int(source),
                samples[i],
                {name: values[i] for name, values in targets.items()},
            )
            for i, source in enumerate(sources)
        ]

    def _make_item(
        self,
        source: int,
        samples: npt.ArrayLike,
        targets: Mapping[str, npt.ArrayLike],
    ) -> RecordingItem:
        """Convert values, apply transforms, and check the resulting item.

        Args:
            source: Original recording position.
            samples: Stored sample values.
            targets: Raw scalar target values by output name.

        Returns:
            Transformed item with independent CPU storage.

        Raises:
            ValueError: If values are missing or categorical IDs are invalid.
            TypeError: If transforms violate the dense CPU item contract.
        """
        converted = {}
        for name, values in targets.items():
            array = np.asarray(values)
            if not np.all(np.isfinite(array)):
                raise ValueError(
                    f"Target {name!r} contains a missing or nonfinite value"
                )
            labels = self._labels[name]
            if labels is not None:
                validate_categorical_values(
                    array.reshape(1), labels=list(labels)
                )
            converted[name] = _tensor(array)
        item = RecordingItem(
            samples=_tensor(samples), targets=converted, item_index=source
        )
        if self._sample_transform is not None:
            item["samples"] = self._sample_transform(item["samples"])
        for name, transform in self._target_transforms.items():
            item["targets"][name] = transform(item["targets"][name])
        # Check the per-field transforms before passing their output to the
        # joint transform, then check the joint transform's result as well.
        self._validate_item(item, source)
        if self._item_transform is not None:
            item = self._item_transform(item)
        self._validate_item(item, source)
        # Transforms may return reused buffers. Detach ownership between items.
        return RecordingItem(
            samples=item["samples"].clone(),
            targets={
                name: value.clone() for name, value in item["targets"].items()
            },
            item_index=source,
        )

    def _validate_item(self, item: RecordingItem, source: int) -> None:
        """Validate dense output structure and CPU placement after transforms.

        Args:
            item: Proposed transformed item.
            source: Expected original source position.

        Raises:
            TypeError: If item structure, identity, or placement is invalid.
        """
        if (
            not isinstance(item, dict)
            or set(item) != {"samples", "targets", "item_index"}
            or type(item["item_index"]) is not int
            or item["item_index"] != source
            or not isinstance(item["targets"], dict)
            or item["targets"].keys() != self._targets.keys()
        ):
            raise TypeError(
                "Transforms must preserve item structure and item_index"
            )
        for value in [item["samples"], *item["targets"].values()]:
            if not isinstance(value, Tensor) or value.device.type != "cpu":
                raise TypeError("Transforms must return CPU tensors")


__all__ = ["RecordingDataset", "RecordingItem"]

"""Named split indexes with explicit assignment and group validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from sigmf_zarr.indexes import validate_categorical_values
from sigmf_zarr.json import JSONObject, json_object
from sigmf_zarr.readonly import ReadOnlyArray
from sigmf_zarr.sources import iter_source_batches, source_descriptor

if TYPE_CHECKING:
    from sigmf_zarr.store import SigMFRecording

type SplitType = Literal["holdout", "kfold", "custom"]
"""Interpretation of a named partition assignment."""

type SplitMethod = Literal[
    "published", "random", "group_random", "chronological", "custom"
]
"""Provenance of assignments supplied by the caller."""


def _split_labels(attributes: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate split descriptors without reading assignments or groups.

    Args:
        attributes: Stored or proposed index attributes.

    Returns:
        Ordered partition names.

    Raises:
        ValueError: If a descriptor is malformed or requires later features.
    """
    if (
        attributes.get("axis"),
        attributes.get("field"),
        attributes.get("kind"),
    ) != ("item", "sigmf-zarr:split", "split"):
        raise ValueError(
            "A split must be an item-aligned sigmf-zarr:split index"
        )
    labels = attributes.get("labels")
    if not isinstance(labels, list) or len(labels) < 2:
        raise ValueError("A split requires at least two partition labels")
    if not all(isinstance(label, str) and label for label in labels):
        raise ValueError("Split labels must be nonempty strings")
    if len(set(labels)) != len(labels):
        raise ValueError("Split labels must be unique")
    if "group_sources" in attributes:
        source_descriptor(attributes["group_sources"])
    _validate_provenance(attributes)
    return tuple(labels)


def _validate_provenance(attributes: Mapping[str, Any]) -> None:
    """Check descriptive split metadata without accessing its sources.

    Args:
        attributes: Stored or proposed index attributes.

    Raises:
        ValueError: If provenance is invalid or requires unsupported features.
    """
    if "split_type" in attributes and attributes["split_type"] not in (
        "holdout",
        "kfold",
        "custom",
    ):
        raise ValueError("Unsupported split_type")
    if "method" in attributes and attributes["method"] not in (
        "published",
        "random",
        "group_random",
        "chronological",
        "custom",
    ):
        raise ValueError("Unsupported split method")
    for key in ("group_index", "generator", "description"):
        if key in attributes and not isinstance(attributes[key], str):
            raise ValueError(f"Split {key} must be a string")
    if "group_index" in attributes and not attributes["group_index"]:
        raise ValueError("Split group_index must not be empty")
    if attributes.get("method") == "group_random" and not (
        attributes.get("group_index") or attributes.get("group_sources")
    ):
        raise ValueError("group_random requires group_index or group_sources")
    if "seed" in attributes:
        seed = attributes["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Split seed must be a nonnegative integer")
    if "validity" in attributes:
        raise ValueError("Nullable splits are not supported")


def _encode_assignments(
    assignments: npt.ArrayLike, labels: Sequence[str]
) -> npt.NDArray[Any]:
    """Encode caller-supplied IDs or exact partition strings.

    Args:
        assignments: Vector of category IDs or partition labels.
        labels: Validated partition vocabulary.

    Returns:
        Compact unsigned integer assignments.

    Raises:
        ValueError: If assignments have invalid types, labels, or IDs.
    """
    # Preserve mixed Python types instead of coercing integers to strings.
    values = np.asarray(
        assignments,
        dtype=None if isinstance(assignments, np.ndarray) else object,
    )
    if values.ndim != 1:
        raise ValueError("Split assignments must be one-dimensional")
    dtype = np.min_scalar_type(len(labels) - 1)
    if values.dtype.kind in {"i", "u"}:
        validate_categorical_values(values, labels=list(labels))
        return values.astype(dtype)
    if all(isinstance(value, str) for value in values):
        lookup = {label: position for position, label in enumerate(labels)}
        try:
            return np.asarray([lookup[value] for value in values], dtype=dtype)
        except KeyError as exc:
            raise ValueError(f"Unknown split label {exc.args[0]!r}") from exc
    if all(
        isinstance(value, int | np.integer)
        and not isinstance(value, bool | np.bool_)
        for value in values
    ):
        if any(value < 0 or value >= len(labels) for value in values):
            raise ValueError("Split ID is outside the labels lookup table")
        return values.astype(dtype)
    raise ValueError("Split assignments must be integer IDs or label strings")


def _validate_assignments(
    recording: SigMFRecording,
    assignments: ReadOnlyArray | npt.NDArray[Any],
    attributes: Mapping[str, Any],
) -> None:
    """Check assignments and declared group isolation in bounded batches.

    Args:
        recording: Recording whose item identities define the split.
        assignments: Stored or proposed one-dimensional assignment array.
        attributes: Stored or proposed split descriptors.

    Raises:
        KeyError: If the declared grouping index does not exist.
        ValueError: If assignments or group isolation are invalid.
    """
    labels = _split_labels(attributes)
    if not recording.batched or assignments.shape != (len(recording),):
        raise ValueError(
            "Split assignments must match the recording item axis"
        )
    group_name = attributes.get("group_index")
    groups = None if group_name is None else recording.index(group_name)
    if groups is not None:
        if groups.attrs.get("axis") != "item":
            raise ValueError("Split group_index must be item-aligned")
        if groups.dtype.kind not in {"i", "u", "U", "O", "T"}:
            raise ValueError(
                "Split group identities must be integers or strings"
            )
    seen: set[int] = set()
    # Keep this map across batches to detect leakage between distant items.
    # Array reads are bounded, but retained state grows with distinct groups.
    group_partitions: dict[int | str, int] = {}
    for start in range(0, len(recording), 65536):
        stop = min(start + 65536, len(recording))
        values = np.asarray(assignments[start:stop])
        validate_categorical_values(values, labels=list(labels))
        seen.update(int(value) for value in np.unique(values))
        if groups is not None:
            identities = np.asarray(groups[start:stop])
            _check_groups(identities, values, group_partitions)
    missing = [label for i, label in enumerate(labels) if i not in seen]
    if missing:
        raise ValueError(f"Split has empty partitions: {missing}")
    if "group_sources" in attributes:
        _validate_sources(recording, assignments, attributes["group_sources"])


def _validate_sources(
    recording: SigMFRecording,
    assignments: ReadOnlyArray | npt.NDArray[Any],
    descriptor: JSONObject,
) -> None:
    """Check constituent source reuse across all assigned partitions.

    Args:
        recording: Recording containing constituent JSON and source indexes.
        assignments: Already validated partition assignments.
        descriptor: Explicit source field and identity key.

    Raises:
        ValueError: If a source is unknown or occurs in several partitions.
    """
    seen: dict[int | str, int] = {}
    for start, batch in iter_source_batches(recording, descriptor):
        values = assignments[start:start + len(batch)]
        for identities, partition in zip(batch, values, strict=True):
            for identity in identities:
                previous = seen.setdefault(identity, int(partition))
                if previous != partition:
                    raise ValueError(
                        f"Source {identity!r} occurs in multiple partitions"
                    )


def _check_groups(
    identities: npt.NDArray[Any],
    values: npt.NDArray[Any],
    seen: dict[int | str, int],
) -> None:
    """Check one batch against partitions seen for earlier identity batches.

    Args:
        identities: Selected scalar group identities.
        values: Selected partition IDs.
        seen: Previous partitions by identity, updated in place.

    Raises:
        ValueError: If an identity is invalid or occurs in several partitions.
    """
    for identity, partition in zip(identities, values, strict=True):
        if isinstance(identity, np.integer):
            identity = int(identity)
        if isinstance(identity, bool) or not isinstance(identity, int | str):
            raise ValueError(
                "Split group identities must be integers or strings"
            )
        # The first occurrence fixes the group's partition. Never overwrite
        # it, or a later conflicting assignment could hide earlier leakage.
        previous = seen.setdefault(identity, int(partition))
        if previous != partition:
            raise ValueError(
                f"Group {identity!r} occurs in multiple partitions"
            )


class SplitIndex:
    """Read-only named split view with no cached group-isolation guarantee.

    Descriptors are read on access. Keep the recording unchanged while
    consuming a view. Call `validate()` to check assignments against
    source groups again.
    """

    _recording: SigMFRecording
    """Recording containing the split."""

    _name: str
    """Index name relative to the recording's indexes group."""

    def __init__(self, recording: SigMFRecording, name: str) -> None:
        """Open a split view after checking its descriptors.

        Args:
            recording: Recording containing the split.
            name: Explicit index name.

        Raises:
            KeyError: If the index is absent.
            ValueError: If the index structure or descriptors are invalid.
        """
        self._recording = recording
        self._name = name
        self._index()

    def _index(self) -> ReadOnlyArray:
        """Read the index and validate its split descriptors.

        Returns:
            Read-only assignment array.

        Raises:
            KeyError: If the index is absent.
            ValueError: If its descriptors or dtype are invalid.
        """
        array = self._recording.index(self._name)
        _split_labels(array.attrs)
        if array.dtype.kind not in {"i", "u"}:
            raise ValueError("Split assignments must have integer dtype")
        return array

    @property
    def assignments(self) -> ReadOnlyArray:
        """Read stored partition IDs without checking group isolation.

        Returns:
            Read-only assignment array.
        """
        return self._index()

    @property
    def labels(self) -> tuple[str, ...]:
        """Read the ordered partition labels.

        Returns:
            Unique partition names in ID order.
        """
        return _split_labels(self._index().attrs)

    @property
    def provenance(self) -> JSONObject:
        """Read descriptive split attributes.

        Returns:
            JSON attributes detached from storage.
        """
        attributes = dict(self._index().attrs)
        for key in ("axis", "field", "kind", "labels"):
            attributes.pop(key, None)
        return json_object(attributes, name="split provenance")

    def validate(self) -> None:
        """Validate stored assignments and declared group isolation.

        Reads bounded batches and retains one partition per distinct group in
        memory. It leaves the store unchanged.

        Raises:
            KeyError: If the split or grouping index is absent.
            ValueError: If assignments, descriptors, or group isolation fail.
        """
        array = self._index()
        _validate_assignments(self._recording, array, array.attrs)


__all__ = ["SplitIndex", "SplitMethod", "SplitType"]

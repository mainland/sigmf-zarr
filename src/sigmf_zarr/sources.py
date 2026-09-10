"""Source reuse groups derived from per-item constituent metadata."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from sigmf_zarr.json import JSONObject, json_object

if TYPE_CHECKING:
    from sigmf_zarr.store import SigMFRecording


def source_descriptor(value: object) -> dict[str, str]:
    """Validate the explicitly selected component field and identity key.

    Args:
        value: Descriptor with field, identity, and optional fallback_index.

    Returns:
        Detached descriptor containing only nonempty strings.

    Raises:
        ValueError: If descriptor keys or values are invalid.
    """
    if (
        not isinstance(value, dict)
        or not {"field", "identity"} <= value.keys()
    ):
        raise ValueError("group_sources requires field and identity")
    if set(value) - {"field", "identity", "fallback_index"} or any(
        not isinstance(item, str) or not item for item in value.values()
    ):
        raise ValueError("Invalid group_sources descriptor")
    return dict(value)


def _identity(value: object) -> int | str:
    """Require a known, scalar source identity.

    Args:
        value: JSON or NumPy scalar source identity.

    Returns:
        Integer or nonempty string identity, with distinct types retained.

    Raises:
        ValueError: If the identity is absent, empty, or nonscalar.
    """
    if isinstance(value, np.integer):
        value = int(value)
    if (
        isinstance(value, bool)
        or not isinstance(value, int | str)
        or value == ""
    ):
        raise ValueError(
            "Source identities must be integers or nonempty strings"
        )
    return value


def iter_source_batches(
    recording: SigMFRecording,
    descriptor: JSONObject,
    *,
    batch_size: int = 65536,
) -> Iterator[tuple[int, list[tuple[int | str, ...]]]]:
    """Read source sets from resolved global fields in bounded item batches.

    A missing field uses the explicitly selected fallback index. An empty list
    means a known empty scene. Malformed components never use the fallback.
    Identities are scoped to this recording, not physical transmitters.

    Args:
        recording: Unchanged batched recording containing constituent JSON.
        descriptor: Component field, identity key, and optional fallback index.
        batch_size: Positive maximum number of item JSON entries per read.

    Yields:
        Source item offset and one tuple of source identities per item.

    Raises:
        ValueError: If metadata or selected identities are unknown or invalid.
        KeyError: If the explicit fallback index is absent.
    """
    spec = source_descriptor(descriptor)
    if not recording.batched or batch_size < 1:
        raise ValueError(
            "Source groups require batched data and positive batches"
        )
    fallback = (
        recording.index(spec["fallback_index"])
        if "fallback_index" in spec
        else None
    )
    if fallback is not None and (
        fallback.attrs.get("axis") != "item" or "validity" in fallback.attrs
    ):
        raise ValueError("Source fallback must be a dense item index")
    shared = recording.global_metadata
    for start in range(0, len(recording), batch_size):
        stop = min(start + batch_size, len(recording))
        entries = (
            recording.item_metadata_array[start:stop]
            if recording.has_item_metadata
            else ["{}"] * (stop - start)
        )
        fallback_values = None if fallback is None else fallback[start:stop]
        batch: list[tuple[int | str, ...]] = []
        for offset, entry in enumerate(entries):
            metadata = json_object(
                json.loads(str(entry)), name="item metadata"
            )
            local = json_object(metadata.get("global", {}), name="item global")
            merged = {**shared, **local}
            if spec["field"] not in merged:
                if fallback_values is None:
                    raise ValueError(
                        f"Unknown component sources for item {start + offset}"
                    )
                identities: tuple[int | str, ...] = (
                    _identity(fallback_values[offset]),
                )
            else:
                components = merged[spec["field"]]
                if not isinstance(components, list):
                    raise ValueError("Component sources must be a list")
                identities = tuple(
                    _identity(
                        json_object(component, name="component").get(
                            spec["identity"]
                        )
                    )
                    for component in components
                )
            batch.append(identities)
        yield start, batch


def source_groups(
    recording: SigMFRecording,
    descriptor: JSONObject,
    *,
    batch_size: int = 65536,
) -> npt.NDArray[np.intp]:
    """Assign connected items to deterministic groups through source reuse.

    If A shares a source with B and B shares another source with C, all three
    receive one group ID. Empty scenes receive independent groups. Memory grows
    with the number of items and distinct identities. No samples are read.

    Args:
        recording: Unchanged batched recording.
        descriptor: Explicit component field and identity selection.
        batch_size: Maximum item JSON entries per read.

    Returns:
        Compact group IDs ordered by each group's first source item position.
    """
    parents = np.arange(len(recording), dtype=np.intp)
    seen: dict[int | str, int] = {}

    def root(item: int) -> int:
        """Find a group's root while compressing the traversed parent path.

        Args:
            item: Source item position.

        Returns:
            Smallest item position in its connected component.
        """
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = int(parents[item])
        return item

    for start, batch in iter_source_batches(
        recording, descriptor, batch_size=batch_size
    ):
        for offset, identities in enumerate(batch):
            item = start + offset
            for identity in identities:
                left, right = root(item), root(seen.setdefault(identity, item))
                parents[max(left, right)] = min(left, right)
    roots = np.asarray([root(item) for item in range(len(recording))])
    return np.unique(roots, return_inverse=True)[1].astype(np.intp)

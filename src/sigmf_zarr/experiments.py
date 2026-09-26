"""Explicit dataset manifests for reproducible experiment inputs."""

from __future__ import annotations

import platform
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version

import numpy as np

from sigmf_zarr.json import JSONObject, json_object
from sigmf_zarr.provenance import capture_inputs
from sigmf_zarr.store import SigMFRecording


def _environment() -> JSONObject:
    """Identify numerical software without importing optional frameworks.

    Returns:
        Python, platform, and installed dependency versions.
    """
    result: JSONObject = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in ("sigmf-zarr", "numpy", "zarr", "torch"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            continue
    return result


def _split_inputs(provenance: JSONObject) -> tuple[set[str], JSONObject]:
    """Select declared grouping indexes and previously bound split inputs.

    Args:
        provenance: Stored split descriptors.

    Returns:
        Input index paths and optional earlier derivation binding.

    Raises:
        ValueError: If a stored binding has malformed index selection.
    """
    selected = set()
    if isinstance(provenance.get("group_index"), str):
        selected.add(str(provenance["group_index"]))
    sources = provenance.get("group_sources")
    if isinstance(sources, dict) and isinstance(
        sources.get("fallback_index"), str
    ):
        selected.add(str(sources["fallback_index"]))
    prior = json_object(provenance.get("inputs", {}), name="split inputs")
    if prior:
        indexes = json_object(prior.get("indexes"), name="split input indexes")
        selected.update(indexes)
    return selected, prior


def dataset_manifest(
    recording: SigMFRecording,
    *,
    targets: Mapping[str, str],
    split: str,
    partitions: Sequence[str],
    transforms: JSONObject,
    parameters: JSONObject,
) -> JSONObject:
    """Hash explicit dataset inputs and validate the selected split.

    Keep the recording unchanged throughout this scan and the experiment.
    Transform and experiment descriptions are supplied by the caller. Python
    callables are not serialized or inferred. The manifest records input
    identity and declared processing, not physical ground truth or numerical
    reproducibility across devices.

    Args:
        recording: Source batched recording.
        targets: Output target names mapped to dense item index paths.
        split: Explicit stored partition index name.
        partitions: Nonempty sequence of selected partition labels.
        transforms: JSON description of every applied data transformation.
        parameters: JSON experiment configuration, such as seed and code hash.

    Returns:
        Detached versioned manifest with hashes, descriptors, selection, and
        numerical software versions. No credentials or backend options.

    Raises:
        ValueError: If selection, target alignment, split isolation, or a
            previously stored split input binding is invalid.
    """
    view = recording.split(split)
    view.validate()
    if (
        isinstance(partitions, str)
        or not partitions
        or any(label not in view.labels for label in partitions)
    ):
        raise ValueError("Select one or more known partition labels")
    partitions = list(dict.fromkeys(partitions))
    for name in targets.values():
        array = recording.index(name)
        if array.attrs.get("axis") != "item" or "validity" in array.attrs:
            raise ValueError("Manifest targets must be dense item indexes")
    selected, prior = _split_inputs(view.provenance)
    selected.update([split, *targets.values()])
    inputs = capture_inputs(recording, indexes=sorted(selected))
    if prior:
        actual = json_object(inputs["indexes"], name="captured indexes")
        names = json_object(prior["indexes"], name="split input indexes")
        expected = {
            **inputs,
            "indexes": {name: actual[name] for name in names},
        }
        if expected != prior:
            raise ValueError(
                "Stored split input binding is stale or unsupported"
            )
    ids = [view.labels.index(label) for label in partitions]
    counts = {label: 0 for label in partitions}
    for start in range(0, len(recording), 65536):
        values = np.asarray(view.assignments[start : start + 65536])
        for label, identity in zip(partitions, ids, strict=True):
            counts[label] += int(np.count_nonzero(values == identity))
    return json_object(
        {
            "version": 1,
            "inputs": inputs,
            "targets": dict(targets),
            "selection": {
                "split": split,
                "partitions": list(partitions),
                "item_counts": counts,
                "order": "source-item-order",
            },
            "transforms": transforms,
            "parameters": parameters,
            "environment": _environment(),
            "source_truth_verified": False,
        },
        name="dataset manifest",
    )

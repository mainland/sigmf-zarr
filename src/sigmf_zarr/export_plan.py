"""Read-only export planning and explicit projection of selected item
indexes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from jsonschema import ValidationError, validate
from sigmf.error import SigMFFileError
from sigmf.sigmffile import SigMFFile

from sigmf_zarr.json import (
    JSONObject,
    json_object,
    json_object_list,
)
from sigmf_zarr.readonly import ReadOnlyArray
from sigmf_zarr.signals import _sample_start

if TYPE_CHECKING:
    from sigmf_zarr.store import SigMFRecording, SigMFZarrStore

INDEX_NAMESPACE = "sigmf-zarr-indexes"
"""Namespace preserving selected index values and their original
descriptors.
"""

@dataclass(frozen=True)
class SigMFExportPlan:
    """Detached preflight result, never a cached authorization to publish
    data.
    """

    metadata: JSONObject
    """Proposed metadata after scope resolution and explicit index
    projection.
    """

    preserved: tuple[str, ...]
    """Metadata sections whose supported values remain present."""

    translated: tuple[str, ...]
    """Coordinates and selected index representations that change."""

    regenerated: tuple[str, ...]
    """Storage-dependent values produced for the exported representation."""

    omitted: tuple[str, ...]
    """Native paths with no selected interchange representation."""

    rejected: tuple[str, ...]
    """Reasons the proposed export cannot proceed."""

    samples_verified: bool
    """Whether this inspection checked every selected sample's encoding."""

    @property
    def lossless(self) -> bool:
        """Return whether the inspected export preserves all selected content.

        Returns:
            True after sample verification, without omissions or errors.
        """
        return self.samples_verified and not self.omitted and not self.rejected

    def as_dict(self) -> JSONObject:
        """Return a JSON report suitable for review or command output.

        Returns:
            Detached metadata and disposition lists.
        """
        return json_object({
            "metadata": self.metadata,
            "preserved": list(self.preserved),
            "translated": list(self.translated),
            "regenerated": list(self.regenerated),
            "omitted": list(self.omitted),
            "rejected": list(self.rejected),
            "samples_verified": self.samples_verified,
            "lossless": self.lossless,
        }, name="export plan")


def export_omissions(
    recording: SigMFRecording, projected_indexes: Sequence[str] = (),
) -> tuple[str, ...]:
    """Identify omitted native paths without reading sample or index payloads.

    Args:
        recording: Source recording.
        projected_indexes: Explicitly selected item-index paths.

    Returns:
        Deterministically ordered omissions for the selected signal.
    """
    omitted = []
    if recording.indexes.attrs:
        omitted.append("indexes/@attributes")
    if len(recording.indexes):
        for name, member in recording.indexes.members(max_depth=None):
            if isinstance(member, ReadOnlyArray):
                if name not in projected_indexes:
                    omitted.append(f"indexes/{name}")
            elif member.attrs:
                omitted.append(f"indexes/{name}/@attributes")
    if len(recording.extensions) or recording.extensions.attrs:
        omitted.append("extension groups or arrays")
    if "channel" in recording.sample_axes:
        omitted.extend(
            f"channels/{index}/@attributes"
            for index in range(recording.num_channels)
            if recording.channel_metadata(index)
        )
    return tuple(sorted(omitted))


def _index_value(
    recording: SigMFRecording, name: str, item: int,
) -> JSONObject:
    """Preserve one scalar, its category interpretation, and its descriptors.

    Args:
        recording: Source recording.
        name: Explicit index path.
        item: Selected batch item.

    Returns:
        Self-describing JSON value bundle.

    Raises:
        ValueError: If the index is not a valid dense item target.
    """
    index = recording.index(name)
    if index.attrs.get("axis") != "item" or index.ndim != 1:
        raise ValueError(f"Projected index {name!r} must be item-aligned")
    attributes = json_object(dict(index.attrs), name="index attributes")
    if "validity" in attributes:
        raise ValueError("Nullable indexes require an explicit projection")
    raw = np.asarray(index[item]).item()
    value = json_object({"value": raw}, name=f"index {name}")["value"]
    labels = attributes.get("labels")
    result: JSONObject = {"value": value, "attributes": attributes}
    if labels is not None:
        if (
            not isinstance(labels, list) or not labels
            or not all(isinstance(label, str) and label for label in labels)
            or len(set(labels)) != len(labels)
            or type(raw) is not int or not 0 <= raw < len(labels)
        ):
            raise ValueError(f"Invalid category lookup for index {name!r}")
        result["label"] = labels[raw]
    return result


def project_item_indexes(
    recording: SigMFRecording, metadata: JSONObject,
    names: Sequence[str], item_index: int | None,
) -> JSONObject:
    """Project explicit indexes into a lossless, namespaced JSON bundle.

    Args:
        recording: Source recording.
        metadata: Prepared interchange metadata, copied before modification.
        names: Unique item-index paths to preserve.
        item_index: Selected item, required when names is nonempty.

    Returns:
        Metadata with values and descriptors in a declared extension.

    Raises:
        ValueError: If selection is invalid or extension JSON conflicts.
    """
    result = json_object(metadata, name="export metadata")
    if not names:
        return result
    if item_index is None or len(set(names)) != len(names):
        raise ValueError("Index projection requires an item and unique names")
    values: JSONObject = {
        name: _index_value(recording, name, item_index) for name in names
    }
    global_info = json_object(result["global"], name="global")
    field = f"{INDEX_NAMESPACE}:values"
    if field in global_info and global_info[field] != values:
        raise ValueError(f"Index projection conflicts with {field}")
    declaration: JSONObject = {
        "name": INDEX_NAMESPACE, "version": "0.1.0", "optional": True,
    }
    extensions = json_object_list(
        global_info.get("core:extensions", []), name="core:extensions"
    )
    existing = [e for e in extensions if e.get("name") == INDEX_NAMESPACE]
    if existing and existing != [declaration]:
        raise ValueError("Index projection extension declaration conflicts")
    if not existing:
        extensions.append(declaration)
    global_info["core:extensions"] = [e for e in extensions]
    global_info[field] = values
    result["global"] = global_info
    return result


def plan_sigmf_export(
    store: SigMFZarrStore, recording_name: str, *,
    item_index: int | None = None, project_indexes: Sequence[str] = (),
    force: bool = False, verify_samples: bool = True,
) -> SigMFExportPlan:
    """Inspect an export without creating files or changing the source.

    Export revalidates independently, so a plan is not safe to reuse after a
    source mutation. Sample verification streams the selected signal and also
    verifies declared recording integrity. Unknown field meanings are preserved
    as JSON, not certified by this operation.

    Args:
        store: Open source store.
        recording_name: Existing recording name.
        item_index: Selected batch item, or None for an unbatched recording.
        project_indexes: Explicit item indexes to preserve in extension JSON.
        force: Permit stale sample-span metadata, matching export behavior.
        verify_samples: Stream samples to check exact encoding and hashes.

    Returns:
        Structured dispositions, metadata, and any rejection diagnostics.
    """
    from sigmf_zarr.sigmf import (
        _export_metadata,
        _iter_sigmf_data_chunks,
        _validate_export_integrity,
    )

    metadata: JSONObject = {}
    omitted: tuple[str, ...] = ()
    rejected: tuple[str, ...] = ()
    verified = False
    try:
        recording = store.recordings[recording_name]
        metadata = _export_metadata(
            recording, item_index=item_index, force=force,
            project_indexes=project_indexes,
        )
        omitted = export_omissions(recording, project_indexes)
        checker = SigMFFile(metadata=json_object(metadata, name="metadata"))
        global_info = json_object(metadata["global"], name="global")
        if "core:version" in global_info:
            checker.set_global_field(
                "core:version", global_info["core:version"]
            )
        else:
            global_info["core:version"] = checker.get_global_field(
                "core:version"
            )
            metadata["global"] = global_info
        # Use SigMF's selected core schema. Extension payloads are preserved,
        # but checking their scientific meanings requires their own validator.
        validate(metadata, checker.get_schema())
        for section in ("captures", "annotations"):
            entries = json_object_list(metadata[section], name=section)
            starts = [_sample_start(entry) for entry in entries]
            if starts != sorted(starts):
                raise ValueError(f"{section} must be sorted by sample_start")
        if verify_samples:
            _validate_export_integrity(recording)
            global_info = json_object(metadata["global"], name="global")
            digest = hashlib.sha512()
            for chunk in _iter_sigmf_data_chunks(
                recording, item_index=item_index, global_metadata=global_info,
            ):
                digest.update(chunk.tobytes())
            declared = global_info.get("core:sha512")
            if (
                isinstance(declared, str)
                and declared.lower() != digest.hexdigest()
            ):
                raise ValueError("core:sha512 does not match exported samples")
            global_info["core:sha512"] = digest.hexdigest()
            metadata["global"] = global_info
            verified = True
    except (
        ValueError, TypeError, KeyError, ValidationError, SigMFFileError,
    ) as exc:
        rejected = (str(exc),)
    translated = [f"indexes/{name}" for name in project_indexes]
    if item_index is not None:
        translated.insert(0, "item capture coordinates")
    return SigMFExportPlan(
        metadata, ("global", "captures", "annotations") if metadata else (),
        tuple(translated), ("core:sha512", "dataset filename"), omitted,
        rejected, verified,
    )


__all__ = ["SigMFExportPlan", "plan_sigmf_export"]

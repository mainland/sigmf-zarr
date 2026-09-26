"""Explicit source identities and bindings for derived metadata."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

from sigmf_zarr.integrity import calculate_array_sha512, canonical_json_bytes
from sigmf_zarr.json import JSONObject, json_object

if TYPE_CHECKING:
    from sigmf_zarr.store import SigMFRecording


def file_identity(path: Path, *, role: str) -> JSONObject:
    """Hash an immutable input file with bounded reads.

    Args:
        path: Source file, unchanged throughout hashing and import.
        role: Source role, such as samples or truth.

    Returns:
        File name, byte count, SHA-512 digest, and caller-supplied role.
    """
    digest = hashlib.sha512()
    size = 0
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return {
        "kind": "file",
        "name": path.name,
        "bytes": size,
        "sha512": digest.hexdigest(),
        "role": role,
    }


def file_manifest_identity(
    paths: Sequence[Path], *, root: Path, role: str
) -> JSONObject:
    """Hash an ordered manifest without storing every file in metadata.

    Args:
        paths: Unique immutable input files under root.
        root: Root for portable relative file names.
        role: Role of the complete manifest.

    Returns:
        Versioned manifest digest, file count, and role. Reconstruct the
        manifest from the source tree to verify its identity.
    """
    records = []
    for path in sorted(set(paths), key=lambda path: path.relative_to(root)):
        record = file_identity(path, role=role)
        record["name"] = path.relative_to(root).as_posix()
        records.append(record)
    return {
        "kind": "file-manifest",
        "version": 1,
        "count": len(records),
        "sha512": hashlib.sha512(canonical_json_bytes(records)).hexdigest(),
        "role": role,
    }


def _declare_extension(metadata: JSONObject, namespace: str) -> None:
    """Declare a project-defined optional metadata namespace.

    Args:
        metadata: Detached global metadata to update.
        namespace: Namespace whose version-0.1.0 fields the writer supplies.

    Raises:
        ValueError: If an existing declaration conflicts with this writer.
    """
    declaration: JSONObject = {
        "name": namespace, "version": "0.1.0", "optional": True
    }
    extensions = metadata.setdefault("core:extensions", [])
    if not isinstance(extensions, list):
        raise ValueError("core:extensions must be a list")
    found = False
    for extension in extensions:
        if isinstance(extension, dict) and extension.get("name") == namespace:
            if extension != declaration:
                raise ValueError(f"Conflicting {namespace} extension")
            found = True
    if not found:
        extensions.append(declaration)


def import_metadata(
    metadata: JSONObject,
    *,
    sources: Sequence[JSONObject],
    operation: str,
    parameters: JSONObject,
    source_namespace: str | None = None,
) -> JSONObject:
    """Attach a versioned import record without changing source vocabulary.

    Args:
        metadata: Metadata to copy before adding the import record.
        sources: Immutable file or logical input identities.
        operation: Importer function name.
        parameters: Source ordering, class mapping, and sample conversions.
        source_namespace: Optional project-defined source metadata namespace.

    Returns:
        Detached metadata with an optional provenance extension declaration.

    Raises:
        ValueError: If caller metadata already contains import provenance or
            a conflicting extension declaration.
    """
    result = json_object(metadata, name="global metadata")
    namespace = "sigmf-zarr-provenance"
    field = f"{namespace}:import"
    if field in result:
        raise ValueError("Import provenance is generated from the input")
    _declare_extension(result, namespace)
    if source_namespace is not None:
        _declare_extension(result, source_namespace)
    result[field] = json_object(
        {
            "sources": list(sources),
            "operation": operation,
            "software": {
                "name": "sigmf-zarr",
                "version": version("sigmf-zarr"),
            },
            "parameters": parameters,
        },
        name="import provenance",
    )
    return result


def capture_inputs(
    recording: SigMFRecording, *, indexes: Sequence[str] = ()
) -> JSONObject:
    """Bind a derived result to the inputs explicitly selected by its caller.

    Keep the recording unchanged during this full sample and metadata scan.
    Store the returned object on an output index or in an external manifest.
    Unselected indexes do not participate, so adding an output index does not
    invalidate its own binding. This function makes no source-truth claim.

    Args:
        recording: Recording to scan without mutation.
        indexes: Explicit input index paths, including labels and descriptors.

    Returns:
        Logical hashes of samples, source metadata, and selected indexes.
    """
    selected: JSONObject = {}
    for name in sorted(set(indexes)):
        array = recording.index(name)
        selected[name] = {
            "sha512": calculate_array_sha512(array),
            "attributes": json_object(
                dict(array.attrs), name="index attributes"
            ),
        }
    return {
        "version": 1,
        "sample_sha512": recording.calculate_sample_sha512(),
        "source_metadata_sha512": recording.calculate_source_metadata_sha512(),
        "indexes": selected,
    }


def verify_inputs(recording: SigMFRecording, binding: JSONObject) -> bool:
    """Recompute an input binding without refreshing or invalidating outputs.

    Args:
        recording: Unchanged recording to scan.
        binding: Previously captured input binding.

    Returns:
        Whether every bound input still matches. Missing indexes return false.

    Raises:
        ValueError: If the binding format is unsupported or malformed.
    """
    indexes = binding.get("indexes")
    if type(binding.get("version")) is not int or binding["version"] != 1:
        raise ValueError("Unsupported input binding version")
    if not isinstance(indexes, dict):
        raise ValueError("Input binding indexes must be an object")
    try:
        return capture_inputs(recording, indexes=list(indexes)) == binding
    except KeyError:
        return False

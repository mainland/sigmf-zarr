"""Explicit source identities and bindings for derived metadata."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

from sigmf_zarr._version import __version__
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
        paths: Immutable input files under root. Paths are normalized
            lexically, so `.` and `..` components do not create duplicates.
        root: Root for portable relative file names.
        role: Role of the complete manifest.

    Returns:
        Versioned manifest digest, file count, and role. Reconstruct the
        manifest from the source tree to verify its identity.

    Raises:
        ValueError: If a normalized path is not below the normalized root.
    """
    base = Path(os.path.normpath(root))
    files: dict[Path, Path] = {}
    for path in paths:
        normalized = Path(os.path.normpath(path))
        try:
            relative = normalized.relative_to(base)
        except ValueError as exc:
            raise ValueError(
                f"Manifest file {str(path)!r} is not below root {str(root)!r}"
            ) from exc
        if relative == Path():
            raise ValueError(f"Manifest file {str(path)!r} names the root")
        files[relative] = normalized
    records = []
    for relative in sorted(files):
        record = file_identity(files[relative], role=role)
        record["name"] = relative.as_posix()
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
        "name": namespace,
        "version": "0.1.0",
        "optional": True,
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
                "version": __version__,
            },
            "parameters": parameters,
        },
        name="import provenance",
    )
    return result


_DIGEST = re.compile(r"[0-9a-f]{128}")
"""Lowercase hexadecimal SHA-512 digest."""


def _digest(value: object, *, name: str) -> None:
    """Require a lowercase hexadecimal SHA-512 digest.

    Args:
        value: Candidate digest.
        name: Field name for error messages.

    Raises:
        ValueError: If the value is not a digest string.
    """
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"Input binding {name} must be a SHA-512 digest")


def _validate_binding(binding: object) -> JSONObject:
    """Check that a value has the structure that capture_inputs() returns.

    Args:
        binding: Candidate version-1 input binding.

    Returns:
        Validated binding.

    Raises:
        ValueError: If the version is unsupported, a field is missing or
            unknown, or a digest or index entry is malformed.
    """
    result = json_object(binding, name="input binding")
    if type(result.get("version")) is not int or result["version"] != 1:
        raise ValueError("Unsupported input binding version")
    fields = {"version", "sample_sha512", "source_metadata_sha512", "indexes"}
    if set(result) != fields:
        raise ValueError(
            f"Input binding must contain exactly the fields {sorted(fields)}"
        )
    _digest(result["sample_sha512"], name="sample_sha512")
    _digest(result["source_metadata_sha512"], name="source_metadata_sha512")
    indexes = result["indexes"]
    if not isinstance(indexes, dict):
        raise ValueError("Input binding indexes must be an object")
    for name, entry in indexes.items():
        if (
            not isinstance(entry, dict)
            or set(entry) != {"sha512", "attributes"}
            or not isinstance(entry["attributes"], dict)
        ):
            raise ValueError(
                f"Input binding index {name!r} must contain sha512 and "
                "attributes"
            )
        _digest(entry["sha512"], name=f"index {name!r} sha512")
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
        Logical hashes of samples, source metadata, and selected indexes. The
        source metadata hash excludes the global `core:sha512` field.
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


def _same_binding(left: JSONObject, right: JSONObject) -> bool:
    """Compare bindings with JSON types kept distinct.

    Python equality treats `1`, `1.0`, and `True` as equal. Canonical JSON
    encodes them differently, as the format's hashes do.

    Args:
        left: First binding.
        right: Second binding.

    Returns:
        Whether both bindings have the same canonical JSON encoding.
    """
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def verify_inputs(recording: SigMFRecording, binding: JSONObject) -> bool:
    """Recompute an input binding without refreshing or invalidating outputs.

    Args:
        recording: Unchanged recording to scan.
        binding: Previously captured input binding.

    Returns:
        Whether every bound input still matches, with integers, floats, and
        Booleans distinct. Missing, invalidated, and structurally invalid
        indexes return false.

    Raises:
        ValueError: If the binding format is unsupported or malformed.
    """
    binding = _validate_binding(binding)
    indexes = cast(JSONObject, binding["indexes"])
    for name in indexes:
        try:
            recording.index(name)
        except (KeyError, ValueError):
            # An index that can no longer be read no longer matches.
            # Axis mutations mark aligned indexes invalid until repaired.
            return False
    try:
        current = capture_inputs(recording, indexes=list(indexes))
    except KeyError:
        return False
    return _same_binding(current, binding)

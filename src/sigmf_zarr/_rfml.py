"""Shared RFML profile declaration for dataset importers."""

from __future__ import annotations

from sigmf_zarr.json import (
    JSONObject,
    JSONValue,
    json_object,
    json_object_list,
)


def _with_rfml_profile(metadata: JSONObject | None) -> JSONObject:
    """Return detached metadata declaring the profile emitted by an importer.

    Args:
        metadata: Optional caller-supplied global metadata.

    Returns:
        Metadata with one rfml-dataset version 0.1.0 declaration. Other
        namespaces remain present. Prior declarations for this profile are
        replaced to describe the version written by the importer.

    Raises:
        ValueError: If metadata or the extension list is malformed.
    """
    result = json_object(metadata or {}, name="global metadata")
    declarations: list[JSONValue] = [
        entry
        for entry in json_object_list(
            result.get("core:extensions", []), name="core:extensions"
        )
        if entry.get("name") != "rfml-dataset"
    ]
    declarations.append(
        {"name": "rfml-dataset", "version": "0.1.0", "optional": True}
    )
    result["core:extensions"] = declarations
    return result

"""Conformance validation for complete SigMF-Zarr stores."""

from __future__ import annotations

import json
import math
import os
import warnings
from collections import Counter
from collections.abc import (
    AsyncIterator,
    Callable,
    Iterable,
    Iterator,
    Mapping,
)
from copy import deepcopy
from dataclasses import dataclass
from functools import cache
from os import PathLike
from pathlib import Path
from typing import Any, cast

import jsonschema
import numpy as np
import sigmf
from sigmf.schema import SCHEMA_COLLECTION, get_schema
from zarr.core.array import Array
from zarr.core.buffer import default_buffer_prototype
from zarr.core.group import Group
from zarr.core.sync import sync
from zarr.storage import FsspecStore, LocalStore, WrapperStore

from sigmf_zarr._consolidated import members, without_consolidated_metadata
from sigmf_zarr._text import escape_text
from sigmf_zarr.export_plan import validate_sigmf_schema
from sigmf_zarr.integrity import (
    INTEGRITY_ATTR,
    _decoded_chunk_nbytes,
    _nominal_itemsize,
    integrity_is_supported,
)
from sigmf_zarr.json import JSONObject, JSONValue, json_value
from sigmf_zarr.readonly import ReadOnlyArray, ReadOnlyGroup
from sigmf_zarr.store import SigMFRecording, SigMFZarrStore
from sigmf_zarr.store._common import (
    ZARR_METADATA_KEYS,
    has_invalid_name_character,
    store_location,
)


@dataclass(frozen=True)
class ValidationIssue:
    """One validation failure at a logical store path."""

    path: str
    """Logical Zarr path associated with the failure."""

    message: str
    """Human-readable failure description."""

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-compatible representation.

        Returns:
            Dictionary containing the issue path and message.
        """
        return {"path": self.path, "message": self.message}


@dataclass(frozen=True)
class ValidationReport:
    """Result of validating one complete store."""

    issues: tuple[ValidationIssue, ...]
    """Validation failures in deterministic path order."""

    recordings_checked: int
    """Number of recording groups inspected."""

    collections_checked: int
    """Number of collection groups inspected."""

    indexes_checked: int
    """Number of recording and store indexes inspected."""

    @property
    def valid(self) -> bool:
        """Return whether no validation failures were found.

        Returns:
            True when the store conforms to all requested checks.
        """
        return not self.issues

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible report.

        Returns:
            Stable validation summary and issue list.
        """
        return {
            "valid": self.valid,
            "recordings_checked": self.recordings_checked,
            "collections_checked": self.collections_checked,
            "indexes_checked": self.indexes_checked,
            "issues": [issue.as_dict() for issue in self.issues],
        }

    def format(self) -> str:
        """Return a human-readable validation report.

        Returns:
            Multiline validation summary.
        """
        summary = (
            f"valid={self.valid!r}, recordings={self.recordings_checked}, "
            f"collections={self.collections_checked}, "
            f"indexes={self.indexes_checked}"
        )
        if self.valid:
            return summary
        # Paths and messages can quote untrusted names and metadata.
        details = [
            escape_text(f"{issue.path}: {issue.message}")
            for issue in self.issues
        ]
        return "\n".join((summary, *details))


_READ_ERRORS = (
    KeyError,
    MemoryError,
    OSError,
    OverflowError,
    RecursionError,
    RuntimeError,
    TypeError,
    ValueError,
)
"""Failures reported as issues rather than raised.

Zarr format 3 reports a CRC32C mismatch as `ValueError`, while numcodecs
reports the Zarr format 2 filter mismatch as `RuntimeError`. Decoding a
deeply nested metadata document or listing deeply nested groups raises
`RecursionError`. An array whose declared shape is too large to read raises
`MemoryError`, or `OverflowError` when NumPy cannot represent the shape. An
unreadable file and a failed remote read raise `OSError`, because the store
boundary translates remote backend errors.
"""

_RECORDING_GROUPS = ("channels", "extensions", "indexes")
"""Child groups that every recording must contain."""

_RECORDING_FIELDS = tuple(
    f"{SigMFRecording.ZARR_EXTENSION_NAMESPACE}:{name}"
    for name in (
        "dtype",
        "sample-shape",
        "sample-axes",
        "num-channels",
        "checksum",
    )
)
"""Global fields that every recording must declare."""


def _issue_message(exc: BaseException) -> str:
    """Describe a failure that is reported as an issue.

    Args:
        exc: Failure raised while reading or hashing a resource.

    Returns:
        Nonempty issue message.
    """
    if isinstance(exc, MemoryError):
        return f"out of memory: {exc}" if str(exc) else "out of memory"
    return str(exc) or type(exc).__name__


def _logical_nbytes(array: Array[Any] | ReadOnlyArray) -> int:
    """Return the declared logical size of an array.

    Variable-length strings and objects count with the nominal element size
    that array hashing uses to choose slab lengths.

    Args:
        array: Zarr or read-only array.

    Returns:
        Logical size in bytes.
    """
    return math.prod(int(size) for size in array.shape) * _nominal_itemsize(
        array.dtype
    )


def _read_size_issue(
    path: str, array: Array[Any] | ReadOnlyArray, max_array_bytes: int
) -> ValidationIssue | None:
    """Report an array whose values are too large to read under a limit.

    Reading an array decodes its whole logical extent and at least one whole
    chunk or shard, which can be larger than the array.

    Args:
        path: Logical array path.
        array: Array to check.
        max_array_bytes: Greatest logical array size and decoded chunk or
            shard size to read.

    Returns:
        Issue naming the exceeded size, or None when both sizes are within
        the limit.
    """
    size = _logical_nbytes(array)
    if size > max_array_bytes:
        return ValidationIssue(
            path,
            f"integrity not verified: the array declares {size} bytes, "
            f"more than max_array_bytes={max_array_bytes}",
        )
    chunk_size = _decoded_chunk_nbytes(array)
    if chunk_size > max_array_bytes:
        return ValidationIssue(
            path,
            f"integrity not verified: one decoded chunk or shard holds "
            f"{chunk_size} bytes, more than max_array_bytes={max_array_bytes}",
        )
    return None


def _oversized_arrays(
    arrays: Iterable[tuple[str, Array[Any] | ReadOnlyArray]],
    max_array_bytes: int | None,
) -> list[ValidationIssue]:
    """Report arrays too large to hash under the validation size limit.

    Args:
        arrays: Logical path and array of each array a hash covers.
        max_array_bytes: Greatest logical array size and decoded chunk or
            shard size to hash, or ``None`` for no limit.

    Returns:
        One issue for each array whose declared size or decoded chunk or
        shard size exceeds the limit.
    """
    if max_array_bytes is None:
        return []
    issues = []
    for path, array in arrays:
        issue = _read_size_issue(path, array, max_array_bytes)
        if issue is not None:
            issues.append(issue)
    return issues


def _view_arrays(
    path: str, group: ReadOnlyGroup, *, exclude: frozenset[str] = frozenset()
) -> Iterator[tuple[str, ReadOnlyArray]]:
    """List every array below a group.

    Args:
        path: Logical path of `group`.
        group: Group to traverse.
        exclude: Relative array paths to omit.

    Yields:
        Logical path and array view of each array.
    """
    for name, member in sorted(group.members(max_depth=None)):
        if isinstance(member, ReadOnlyArray) and name not in exclude:
            yield f"{path}/{name}", member


async def _collect_keys(keys: AsyncIterator[str]) -> list[str]:
    """Collect keys from an asynchronous store listing.

    Args:
        keys: Asynchronous key iterator.

    Returns:
        Listed keys.
    """
    return [key async for key in keys]


def _child_group_names(group: Group) -> tuple[str, ...]:
    """List child groups, including children whose metadata is unreadable.

    Zarr parses every child's metadata document to list child groups, so one
    damaged document makes the whole listing fail. In that case, children
    are inspected one at a time. A child whose metadata cannot be read is
    listed, so that opening it reports the failure at its own path.

    Args:
        group: Recordings or collections group.

    Returns:
        Child names in sorted order.
    """
    try:
        return tuple(sorted(group.group_keys()))
    except _READ_ERRORS:
        pass
    keys = sync(_collect_keys(group.store.list_dir(group.path)))
    names = []
    for key in sorted(set(keys) - ZARR_METADATA_KEYS):
        try:
            child = group[key]
        except KeyError:
            # The key names a stray object rather than a Zarr node.
            continue
        except _READ_ERRORS:
            names.append(key)
            continue
        if isinstance(child, Group):
            names.append(key)
    return tuple(names)


def _child_array_names(group: Group) -> tuple[str, ...]:
    """List child arrays of the recordings or collections group.

    Args:
        group: Recordings or collections group.

    Returns:
        Array names in sorted order, or none when listing fails. A listing
        failure is reported for the affected child.
    """
    try:
        return tuple(sorted(group.array_keys()))
    except _READ_ERRORS:
        return ()


def _reserved_child_names(group: Group) -> tuple[str, ...]:
    """List child nodes named like a Zarr metadata document.

    These names are reserved, so only another tool can have written such a
    child. Zarr format 2 omits it from group listings, so the store is
    listed directly. A listed name is a child node when a node metadata
    document exists below it.

    Args:
        group: Recordings or collections group.

    Returns:
        Reserved names of child nodes, in sorted order.
    """
    store = group.store
    if not store.supports_listing:
        return ()
    parent = f"{group.path}/" if group.path else ""
    listed = set(sync(_collect_keys(store.list_dir(group.path))))
    return tuple(
        name
        for name in sorted(listed & ZARR_METADATA_KEYS)
        if any(
            sync(store.exists(f"{parent}{name}/{document}"))
            for document in ("zarr.json", ".zgroup", ".zarray")
        )
    )


_CONSOLIDATED_MESSAGE = (
    "group has consolidated metadata, which a SigMF-Zarr store must not "
    "contain"
)
"""Issue message for a group with consolidated metadata."""


def _root_has_consolidated_metadata(store: SigMFZarrStore) -> bool:
    """Return whether the root group has consolidated metadata.

    The store opens its root without consolidated metadata, so the root
    metadata documents are read directly.

    Args:
        store: Store to inspect.

    Returns:
        True when the root has a `.zmetadata` document in Zarr format 2, or
        a non-null `consolidated_metadata` field in Zarr format 3.
    """
    root = store._group.store_path
    if store.zarr_format == 2:
        return bool(sync((root / ".zmetadata").exists()))
    document = sync(
        (root / "zarr.json").get(prototype=default_buffer_prototype())
    )
    if document is None:
        return False
    metadata = json.loads(document.to_bytes())
    return (
        isinstance(metadata, dict)
        and metadata.get("consolidated_metadata") is not None
    )


def _has_consolidated_metadata(group: Group) -> bool:
    """Return whether a non-root group has consolidated metadata.

    Args:
        group: Group handle read from its own metadata document, before its
            consolidated metadata is removed.

    Returns:
        True when the group has a `.zmetadata` document in Zarr format 2, or
        a `consolidated_metadata` field in Zarr format 3.
    """
    if group.metadata.zarr_format == 2:
        return bool(sync((group.store_path / ".zmetadata").exists()))
    return group.metadata.consolidated_metadata is not None


def _readable_children(group: Group) -> list[tuple[str, Array[Any] | Group]]:
    """List the immediate children of a group whose metadata is readable.

    Zarr fails a whole listing when one child's metadata is damaged. In
    that case, children are read one at a time and unreadable children are
    skipped. Other checks report them.

    Args:
        group: Group without consolidated metadata.

    Returns:
        Child names paired with arrays and groups, which may carry their own
        consolidated metadata.
    """
    try:
        return list(group.members())
    except _READ_ERRORS:
        pass
    keys = sync(_collect_keys(group.store.list_dir(group.path)))
    children: list[tuple[str, Array[Any] | Group]] = []
    for key in sorted(set(keys) - ZARR_METADATA_KEYS):
        try:
            child = group[key]
        except _READ_ERRORS:
            continue
        children.append((key, child))
    return children


def _hierarchy_issues(store: SigMFZarrStore) -> list[ValidationIssue]:
    """Check rules that apply to every node of the store.

    Every group must lack consolidated metadata. Every node name below the
    root must not contain a control character or lone surrogate. Recording
    and collection names are checked when each resource opens. No array may
    have a structured or void dtype, because array hashes do not cover field
    names, and every array must store a chunk shape of positive integers.
    Nodes are listed one level at a time from their own metadata documents,
    so a consolidated copy cannot hide a node.

    Args:
        store: Store to inspect.

    Returns:
        One issue for each violation. Nothing below the root is checked when
        the store cannot list its keys.
    """
    issues = []
    try:
        if _root_has_consolidated_metadata(store):
            issues.append(ValidationIssue("/", _CONSOLIDATED_MESSAGE))
    except _READ_ERRORS as exc:
        issues.append(ValidationIssue("/", _issue_message(exc)))
    if not store._group.store.supports_listing:
        return issues
    # An explicit stack keeps deep hierarchies within the recursion limit.
    pending = [("", store._group)]
    while pending:
        path, group = pending.pop()
        try:
            issues.extend(_group_node_issues(path, group, pending))
        except _READ_ERRORS as exc:
            issues.append(ValidationIssue(path or "/", _issue_message(exc)))
    return issues


def _group_node_issues(
    path: str,
    group: Group,
    pending: list[tuple[str, Group]],
) -> list[ValidationIssue]:
    """Check the immediate children of one group for `_hierarchy_issues`.

    Args:
        path: Logical path of the group, empty for the root.
        group: Group without consolidated metadata.
        pending: Groups still to visit. Child groups are appended.

    Returns:
        One issue for each violation among the children.
    """
    issues = []
    for name, member in _readable_children(group):
        child_path = f"{path}/{name}" if path else name
        if path not in {"recordings", "collections"} and (
            has_invalid_name_character(name)
        ):
            issues.append(
                ValidationIssue(
                    child_path,
                    "name must not contain control characters or lone "
                    "surrogates",
                )
            )
        if isinstance(member, Group):
            if _has_consolidated_metadata(member):
                issues.append(
                    ValidationIssue(child_path, _CONSOLIDATED_MESSAGE)
                )
            pending.append((child_path, without_consolidated_metadata(member)))
        else:
            if _is_structured_dtype(member.dtype):
                issues.append(
                    ValidationIssue(
                        child_path,
                        "structured and void array dtypes are not permitted",
                    )
                )
            chunk_issue = _stored_chunk_shape_issue(child_path, member)
            if chunk_issue is not None:
                issues.append(chunk_issue)
    return issues


def _stored_chunk_shape_issue(
    path: str, array: Array[Any]
) -> ValidationIssue | None:
    """Check the chunk shape that an array's metadata document stores.

    Zarr 3.4 and later read a stored chunk size of 0 as 1 and only warn, so
    the parsed array can hide the violation. The document is read directly.

    Args:
        path: Logical path of the array.
        array: Array whose metadata document is checked.

    Returns:
        An issue if the stored chunk shape is not a list of positive
        integers, or None.
    """
    zarr_format = array.metadata.zarr_format
    name = "zarr.json" if zarr_format == 3 else ".zarray"
    document = sync(
        (array.store_path / name).get(prototype=default_buffer_prototype())
    )
    if document is None:
        return None
    metadata = json.loads(document.to_bytes())
    if not isinstance(metadata, dict):
        return None
    if zarr_format == 3:
        grid = metadata.get("chunk_grid")
        configuration = (
            grid.get("configuration") if isinstance(grid, dict) else None
        )
        chunks = (
            configuration.get("chunk_shape")
            if isinstance(configuration, dict)
            else None
        )
    else:
        chunks = metadata.get("chunks")
    if isinstance(chunks, list) and all(
        type(size) is int and size >= 1 for size in chunks
    ):
        return None
    return ValidationIssue(
        path, f"stored chunk shape {chunks!r} must hold positive integers"
    )


def _is_structured_dtype(dtype: object) -> bool:
    """Return whether an array dtype is a structured or void dtype.

    Args:
        dtype: Zarr or NumPy dtype object.

    Returns:
        True for NumPy kind `V`.
    """
    try:
        kind: str = np.dtype(cast(Any, dtype)).kind
    except TypeError:
        return False
    return kind == "V"


def _orphan_prefix_issues(store: SigMFZarrStore) -> list[ValidationIssue]:
    """Report names under `recordings/` and `collections/` without a node.

    An interrupted removal on a backend without a directory rename can
    delete a resource's metadata document but leave other keys under its
    name. Zarr omits such a name from group listings.

    Args:
        store: Store to inspect.

    Returns:
        One issue for each listed name that has no Zarr node metadata, or
        none when the store cannot list its keys.
    """
    documents = (
        ("zarr.json",) if store.zarr_format == 3 else (".zgroup", ".zarray")
    )
    issues = []
    for kind in ("recordings", "collections"):
        try:
            group = (
                store._recordings_group
                if kind == "recordings"
                else store._collections_group
            )
            issues.extend(_orphan_names(kind, group, documents))
        except _READ_ERRORS as exc:
            issues.append(ValidationIssue(kind, _issue_message(exc)))
    return issues


def _orphan_names(
    kind: str, group: Group, documents: tuple[str, ...]
) -> list[ValidationIssue]:
    """List names with keys but no node metadata for `_orphan_prefix_issues`.

    Args:
        kind: `recordings` or `collections`.
        group: Container group.
        documents: Node metadata document names for the Zarr format.

    Returns:
        One issue for each orphan name.
    """
    if not group.store.supports_listing:
        return []
    names = set(sync(_collect_keys(group.store.list_dir(group.path))))
    return [
        ValidationIssue(
            f"{kind}/{name}",
            "keys exist under this name, but it has no Zarr node metadata",
        )
        for name in sorted(names - ZARR_METADATA_KEYS)
        if not any(
            sync((group.store_path / name / document).exists())
            for document in documents
        )
    ]


def _attribute_issues(
    path: str, group: ReadOnlyGroup
) -> list[ValidationIssue]:
    """Check that every attribute in a group tree is finite, encodable JSON.

    Integrity hashes encode these attributes as canonical JSON, so values
    that another tool wrote, such as NaN, would otherwise surface only when a
    hash is calculated.

    Args:
        path: Logical path of `group`.
        group: Group whose attributes and descendants to check.

    Returns:
        One issue for each node with invalid attributes.
    """
    nodes: list[tuple[str, ReadOnlyArray | ReadOnlyGroup]] = [(path, group)]
    nodes.extend(
        (f"{path}/{name}", member)
        for name, member in sorted(group.members(max_depth=None))
    )
    issues = []
    for node_path, node in nodes:
        try:
            json_value(dict(node.attrs), name="attributes")
        except _READ_ERRORS as exc:
            issues.append(ValidationIssue(node_path, _issue_message(exc)))
    return issues


def _integrity_type_issue(
    path: str, attrs: Mapping[str, object]
) -> ValidationIssue | None:
    """Report an `integrity` attribute that is not a JSON object.

    Readers treat such a value as absent, so it would otherwise go unnoticed.

    Args:
        path: Logical path of the resource group.
        attrs: Resource group attributes.

    Returns:
        Issue for a malformed attribute, or None.
    """
    if INTEGRITY_ATTR in attrs and not isinstance(attrs[INTEGRITY_ATTR], dict):
        return ValidationIssue(
            path, f"{INTEGRITY_ATTR!r} attribute must be a JSON object"
        )
    return None


def _recording_layout_issues(
    path: str, recording: SigMFRecording
) -> list[ValidationIssue]:
    """Check recording requirements that opening a recording does not.

    Args:
        path: Logical recording path.
        recording: Opened recording.

    Returns:
        Issues for missing groups and fields, malformed descriptors, and
        core metadata that violates the SigMF schema.
    """
    issues = []
    group = recording.group
    for name in _RECORDING_GROUPS:
        if name not in group:
            issues.append(
                ValidationIssue(path, f"recording requires the {name!r} group")
            )
        elif not isinstance(group[name], ReadOnlyGroup):
            issues.append(ValidationIssue(path, f"{name!r} must be a group"))
    global_metadata = recording.global_metadata
    issues.extend(
        ValidationIssue(path, f"global metadata requires {field!r}")
        for field in _RECORDING_FIELDS
        if field not in global_metadata
    )
    integrity_issue = _integrity_type_issue(path, group.attrs)
    if integrity_issue is not None:
        issues.append(integrity_issue)
    if isinstance(group.get("indexes"), ReadOnlyGroup):
        for index_name, member in sorted(
            recording.indexes.members(max_depth=None)
        ):
            if not isinstance(member, ReadOnlyArray):
                continue
            unit = member.attrs.get("unit")
            if unit is not None and not isinstance(unit, str):
                issues.append(
                    ValidationIssue(
                        f"{path}/indexes/{index_name}",
                        "index 'unit' attribute must be a string",
                    )
                )
    schema_issue = _sigmf_schema_issue(path, recording)
    if schema_issue is not None:
        issues.append(schema_issue)
    return issues


def _json_path(parts: Iterable[str | int]) -> str:
    """Format a JSON location in the bracket notation of JSONPath.

    jsonschema formats `json_path` differently across releases, so issue
    messages build the path themselves.

    Args:
        parts: Object keys and array positions from the document root.

    Returns:
        Path such as `$.global['core:sample_rate']` or `$.captures[0]`.
    """
    path = "$"
    for part in parts:
        if isinstance(part, int):
            path += f"[{part}]"
        elif part.isidentifier():
            path += f".{part}"
        else:
            path += "[" + repr(part) + "]"
    return path


def _sigmf_schema_issue(
    path: str, recording: SigMFRecording
) -> ValidationIssue | None:
    """Check shared recording metadata against the SigMF core schema.

    Per-item metadata is not checked, because that requires reading every
    item.

    Args:
        path: Logical recording path.
        recording: Opened recording.

    Returns:
        Issue for the first schema violation, or None.
    """
    global_metadata = deepcopy(recording.global_metadata)
    # SigMF-Zarr does not require `core:datatype` until export. A
    # placeholder satisfies the schema requirement, so only the fields that
    # are present are checked.
    global_metadata.setdefault("core:datatype", "rf32_le")
    with warnings.catch_warnings():
        # sigmf warns about undeclared namespaces, which SigMF-Zarr permits.
        warnings.simplefilter("ignore", DeprecationWarning)
        captures: list[JSONValue] = [*recording.captures]
        annotations: list[JSONValue] = [*recording.annotations]
        try:
            validate_sigmf_schema(
                {
                    "global": global_metadata,
                    "captures": captures,
                    "annotations": annotations,
                }
            )
        except ValueError as exc:
            cause = exc.__cause__
            detail = (
                f"{_json_path(cause.absolute_path)}: {cause.message}"
                if isinstance(cause, jsonschema.ValidationError)
                else str(exc)
            )
            return ValidationIssue(
                path, f"metadata does not satisfy the SigMF schema: {detail}"
            )
    return None


_LOCAL_PROTOCOLS = frozenset({"file", "local"})
"""Fsspec protocols of the local file system."""

_CACHING_PROTOCOLS = frozenset(
    {"blockcache", "cached", "filecache", "simplecache"}
)
"""Fsspec protocols that cache another file system without changing paths."""

_EMPTY_NAMES: tuple[str, ...] = ()
"""Child names used after a listing failure."""


def _url_local_root(url: str) -> Path | None:
    """Return the local directory that a store URL names.

    Args:
        url: Store URL, possibly chained, as in `simplecache::file:///a`.

    Returns:
        Local directory for a `file` or `local` URL, optionally behind
        caching protocols, or ``None`` for any other URL.
    """
    *wrappers, target = url.split("::")
    protocol, separator, path = target.partition("://")
    if (
        not separator
        or protocol not in _LOCAL_PROTOCOLS
        or not all(wrapper in _CACHING_PROTOCOLS for wrapper in wrappers)
    ):
        return None
    return Path(path)


def _protocols(fs: object) -> frozenset[str]:
    """Return the protocol names of an fsspec file system.

    Args:
        fs: Fsspec file system.

    Returns:
        Protocol names, which fsspec gives as a string or a tuple.
    """
    protocol = getattr(fs, "protocol", ())
    if isinstance(protocol, str):
        return frozenset({protocol})
    return frozenset(name for name in protocol if isinstance(name, str))


def _backend_local_root(store: SigMFZarrStore) -> Path | None:
    """Return the local directory behind an opened store.

    Args:
        store: Opened store.

    Returns:
        Root directory of a local store or of an fsspec store over the local
        file system, or ``None`` for any other backend.
    """
    store_path = store._group.store_path
    backend = store_path.store
    while isinstance(backend, WrapperStore):
        backend = backend._store
    if isinstance(backend, LocalStore):
        root = Path(backend.root)
    elif isinstance(backend, FsspecStore):
        fs: object = backend.fs
        # Zarr wraps a synchronous file system for asynchronous use, and a
        # caching file system wraps the file system that holds the data.
        while True:
            wrapped = getattr(fs, "sync_fs", None)
            if wrapped is None and _protocols(fs) & _CACHING_PROTOCOLS:
                wrapped = getattr(fs, "fs", None)
            if wrapped is None:
                break
            fs = wrapped
        if not _protocols(fs) & _LOCAL_PROTOCOLS:
            return None
        root = Path(backend.path)
    else:
        return None
    return root / store_path.path if store_path.path else root


def _local_store_root(
    store_or_path: SigMFZarrStore | str | PathLike[str],
) -> Path | None:
    """Return the local directory of a store without reading its content.

    Args:
        store_or_path: Open store, local path, or URL.

    Returns:
        Local root directory, or ``None`` for a remote or in-memory store.
    """
    if isinstance(store_or_path, SigMFZarrStore):
        return _backend_local_root(store_or_path)
    location = store_location(store_or_path)
    if isinstance(location, Path):
        return location
    return _url_local_root(location)


def _local_file_issues(root: Path) -> list[ValidationIssue]:
    """Report entries below a local store root that are unsafe to read.

    The walk uses `lstat` information only and never follows a link or
    opens a file. Zarr follows symbolic links, so a writable open could
    modify or delete files outside the store, and reading a FIFO or device
    can block or never end. Zarr also resolves keys inside a directory that
    cannot be listed, so a link hidden there would go unreported.

    Args:
        root: Local store root. A root that the caller names through a
            symbolic link is followed.

    Returns:
        One issue for each symbolic link, each entry that is neither a
        regular file nor a directory, and each directory that cannot be
        listed. None when `root` is not a directory.
    """
    if not root.is_dir():
        return []
    issues = []
    pending = [""]
    while pending:
        relative = pending.pop()
        try:
            with os.scandir(root / relative) as listing:
                entries = sorted(listing, key=lambda entry: entry.name)
        except OSError as exc:
            issues.append(
                ValidationIssue(
                    relative or "/",
                    f"directory cannot be listed: {exc.strerror or exc}",
                )
            )
            continue
        for entry in entries:
            path = f"{relative}/{entry.name}" if relative else entry.name
            try:
                if entry.is_symlink():
                    message = "symbolic link below the store root"
                elif entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                    continue
                elif entry.is_file(follow_symlinks=False):
                    continue
                else:
                    message = (
                        "neither a regular file nor a directory below the "
                        "store root"
                    )
            except OSError as exc:
                message = f"cannot be inspected: {exc.strerror or exc}"
            issues.append(ValidationIssue(path, message))
    return issues


def _reported[T](
    path: str,
    issues: list[ValidationIssue],
    default: T,
    read: Callable[[], T],
) -> T:
    """Run one store-wide read, reporting a failure instead of raising it.

    Args:
        path: Logical path to report a failure at.
        issues: Issue list to extend.
        default: Result to use after a failure.
        read: Read to run.

    Returns:
        Result of `read`, or `default` after a reported failure.
    """
    try:
        return read()
    except _READ_ERRORS as exc:
        issues.append(ValidationIssue(path, _issue_message(exc)))
        return default


def _integrity_issue(
    *,
    path: str,
    integrity: JSONObject,
    calculators: Mapping[str, Callable[[], str]],
    verify: bool,
    require_integrity: bool,
    unverified: frozenset[str] = frozenset(),
) -> ValidationIssue | None:
    """Return an integrity issue for one resource when applicable.

    Each declared hash is verified separately. A managed edit removes only
    the hashes it affects, so the remaining hashes stay verifiable.

    Args:
        path: Logical resource path.
        integrity: Declared integrity object.
        calculators: Callback that recalculates each hash field that a
            complete integrity object declares.
        verify: Whether to recalculate and compare declared hashes.
        require_integrity: Whether absent hashes are failures.
        unverified: Hash fields not recalculated, because a covered array
            exceeds the validation size limit.

    Returns:
        Integrity issue, or None when the resource satisfies the policy.
    """
    if not integrity:
        if require_integrity:
            return ValidationIssue(path, "integrity metadata is required")
        return None
    if not integrity_is_supported(integrity):
        return ValidationIssue(
            path, "integrity metadata uses an unsupported algorithm or version"
        )
    declared = {
        field: integrity.get(field)
        for field in calculators
        if integrity.get(field) is not None
    }
    if verify and any(
        value != calculators[field]()
        for field, value in declared.items()
        if field not in unverified
    ):
        return ValidationIssue(
            path, "integrity hashes do not match current content"
        )
    if require_integrity and not all(
        isinstance(declared.get(field), str) for field in calculators
    ):
        return ValidationIssue(path, "integrity hashes are incomplete")
    return None


def _item_metadata_issues(
    path: str, recording: SigMFRecording, max_array_bytes: int | None
) -> tuple[list[ValidationIssue], bool]:
    """Validate the stored per-item metadata entries of one recording.

    Entries are checked one chunk at a time. Under a limit, a chunk whose
    nominal decoded size exceeds it is not read, and checking stops once
    the decoded entry text exceeds it.

    Args:
        path: Logical recording path.
        recording: Recording opened with structural validation.
        max_array_bytes: Greatest decoded chunk size and total entry text,
            in bytes, to read, or ``None`` for no limit.

    Returns:
        Issues, and whether every entry was read within the limit.
    """
    if not recording.has_item_metadata:
        return [], True
    array_path = f"{path}/item_metadata"
    if max_array_bytes is not None:
        chunk_size = _decoded_chunk_nbytes(recording.item_metadata_array)
        if chunk_size > max_array_bytes:
            return [
                ValidationIssue(
                    array_path,
                    f"entries not validated: one decoded chunk or shard "
                    f"holds {chunk_size} bytes, more than "
                    f"max_array_bytes={max_array_bytes}",
                )
            ], False
    try:
        complete = recording._validate_item_metadata_entries(
            max_text_bytes=max_array_bytes
        )
    except ValueError as exc:
        return [ValidationIssue(path, _issue_message(exc))], True
    if not complete:
        return [
            ValidationIssue(
                array_path,
                "entries not fully validated: they hold more than "
                f"max_array_bytes={max_array_bytes} bytes of text",
            )
        ], False
    return [], True


def _recording_integrity_issues(
    path: str,
    recording: SigMFRecording,
    *,
    verify_integrity: bool,
    require_integrity: bool,
    max_array_bytes: int | None,
    item_metadata_read: bool = True,
) -> list[ValidationIssue]:
    """Validate the hashes that one recording declares.

    Args:
        path: Logical recording path.
        recording: Opened recording.
        verify_integrity: Whether to verify hashes that are present,
            including `core:sha512`.
        require_integrity: Whether the recording must declare integrity.
        max_array_bytes: Greatest logical array size and decoded chunk or
            shard size to hash, or ``None`` for no limit.
        item_metadata_read: Whether the per-item metadata entries could be
            read within the limit. If not, the metadata hash, which covers
            them, is not verified.

    Returns:
        Integrity issues, including arrays too large to verify.
    """
    issues: list[ValidationIssue] = []
    samples = recording.samples
    sample_issues = _oversized_arrays(
        [(f"{path}/samples", samples)], max_array_bytes
    )
    metadata_issues = _oversized_arrays(
        _view_arrays(path, recording.group, exclude=frozenset({"samples"})),
        max_array_bytes,
    )
    integrity = recording.integrity
    if verify_integrity:
        if integrity.get("sample_sha512") is not None or (
            recording.sha512 is not None
        ):
            issues.extend(sample_issues)
        if integrity.get("metadata_sha512") is not None:
            issues.extend(metadata_issues)
    issue = _integrity_issue(
        path=path,
        integrity=integrity,
        calculators={
            "sample_sha512": recording.calculate_sample_sha512,
            "metadata_sha512": recording.calculate_metadata_sha512,
        },
        verify=verify_integrity,
        require_integrity=require_integrity,
        unverified=frozenset(
            field
            for field, skipped in (
                ("sample_sha512", bool(sample_issues)),
                (
                    "metadata_sha512",
                    bool(metadata_issues) or not item_metadata_read,
                ),
            )
            if skipped
        ),
    )
    if issue is not None:
        issues.append(issue)
    if (
        verify_integrity
        and not sample_issues
        and recording.sha512 is not None
        and not recording.verify_sha512()
    ):
        issues.append(
            ValidationIssue(path, "core:sha512 does not match sample data")
        )
    return issues


def _recording_index_issues(
    path: str, recording: SigMFRecording
) -> tuple[list[ValidationIssue], int]:
    """Validate each recording index separately.

    One invalid index does not prevent checking the others or the rest of
    the recording.

    Args:
        path: Logical recording path.
        recording: Opened recording.

    Returns:
        One issue for each invalid index, at the index path, and the number
        of indexes checked.
    """
    if not isinstance(recording.group.get("indexes"), ReadOnlyGroup):
        return [], 0
    issues = []
    indexes_checked = 0
    for index_name, member in sorted(
        recording.indexes.members(max_depth=None)
    ):
        if not isinstance(member, ReadOnlyArray):
            continue
        indexes_checked += 1
        # `_hierarchy_issues` reports a malformed name once for every node.
        if has_invalid_name_character(index_name):
            continue
        try:
            recording.index(index_name)
        except _READ_ERRORS as exc:
            issues.append(
                ValidationIssue(
                    f"{path}/indexes/{index_name}", _issue_message(exc)
                )
            )
    return issues, indexes_checked


def _recording_issues(
    store: SigMFZarrStore,
    recording_name: str,
    *,
    verify_integrity: bool,
    require_integrity: bool,
    max_array_bytes: int | None,
) -> tuple[list[ValidationIssue], int]:
    """Validate one recording and its indexes.

    Args:
        store: Store containing the recording.
        recording_name: Recording name to validate.
        verify_integrity: Whether to verify hashes that are present,
            including `core:sha512`.
        require_integrity: Whether the recording must declare integrity.
        max_array_bytes: Greatest logical array size to hash, or ``None``
            for no limit.

    Returns:
        Validation issues and number of indexes checked.
    """
    path = f"recordings/{recording_name}"
    issues: list[ValidationIssue] = []
    indexes_checked = 0
    # Treat each resource as an independent validation boundary so one broken
    # recording does not prevent a complete store report.
    try:
        # Entries are checked below under the read limit, not on open.
        recording = store.recordings.open(
            recording_name, validation="structural"
        )
        entry_issues, item_metadata_read = _item_metadata_issues(
            path, recording, max_array_bytes
        )
        issues.extend(entry_issues)
        issues.extend(_attribute_issues(path, recording.group))
        issues.extend(_recording_layout_issues(path, recording))
        index_issues, indexes_checked = _recording_index_issues(
            path, recording
        )
        issues.extend(index_issues)
        if verify_integrity or require_integrity:
            issues.extend(
                _recording_integrity_issues(
                    path,
                    recording,
                    verify_integrity=verify_integrity,
                    require_integrity=require_integrity,
                    max_array_bytes=max_array_bytes,
                    item_metadata_read=item_metadata_read,
                )
            )
    except _READ_ERRORS as exc:
        issues.append(ValidationIssue(path, _issue_message(exc)))
    return issues, indexes_checked


@cache
def _collection_schema() -> dict[str, Any]:
    """Return the SigMF collection schema of the installed sigmf package.

    Returns:
        Parsed JSON schema for `.sigmf-collection` documents.
    """
    schema: dict[str, Any] = get_schema(schema_file=SCHEMA_COLLECTION)
    return schema


def _collection_schema_issue(
    path: str, metadata: JSONObject
) -> ValidationIssue | None:
    """Check collection metadata against the SigMF collection schema.

    Args:
        path: Logical collection path.
        metadata: Collection `metadata` attribute.

    Returns:
        Issue for the first schema violation, or None.
    """
    # This applies the rule of the export check. Export derives
    # `core:streams` from `recording_ids`, and the schema that sigmf-python
    # bundles describes each stream as an array, contrary to the SigMF
    # specification, so `core:streams` is not checked. Export writes the
    # sigmf-python specification version when `core:version` is absent.
    collection = {
        key: value for key, value in metadata.items() if key != "core:streams"
    }
    collection.setdefault("core:version", sigmf.__specification__)
    try:
        jsonschema.validate({"collection": collection}, _collection_schema())
    except jsonschema.ValidationError as exc:
        location = _json_path(exc.absolute_path)
        return ValidationIssue(
            path,
            "metadata does not satisfy the SigMF collection schema: "
            f"{location}: {exc.message}",
        )
    return None


def _duplicate_recording_id_issue(
    path: str, recording_ids: Iterable[str]
) -> ValidationIssue | None:
    """Report recording identifiers that a collection lists more than once.

    Args:
        path: Logical collection path.
        recording_ids: Collection `recording_ids` attribute.

    Returns:
        Issue naming each repeated identifier, or None.
    """
    duplicates = sorted(
        name for name, count in Counter(recording_ids).items() if count > 1
    )
    if not duplicates:
        return None
    return ValidationIssue(
        path,
        f"recording_ids must be unique, got {duplicates!r} more than once",
    )


def _collection_issues(
    store: SigMFZarrStore,
    collection_name: str,
    *,
    recording_names: set[str],
    verify_integrity: bool,
    require_integrity: bool,
    max_array_bytes: int | None,
) -> list[ValidationIssue]:
    """Validate one collection and its recording references.

    Args:
        store: Store containing the collection.
        collection_name: Collection name to validate.
        recording_names: Recording names available in the store.
        verify_integrity: Whether to verify hashes that are present.
        require_integrity: Whether the collection must declare integrity.
        max_array_bytes: Greatest logical array size to hash, or ``None``
            for no limit.

    Returns:
        Validation issues for the collection.
    """
    path = f"collections/{collection_name}"
    issues: list[ValidationIssue] = []
    try:
        collection = store.collections.open(collection_name)
        issues.extend(_attribute_issues(path, collection.group))
        integrity_issue = _integrity_type_issue(path, collection.group.attrs)
        if integrity_issue is not None:
            issues.append(integrity_issue)
        schema_issue = _collection_schema_issue(path, collection.metadata)
        if schema_issue is not None:
            issues.append(schema_issue)
        duplicate_issue = _duplicate_recording_id_issue(
            path, collection.recording_ids
        )
        if duplicate_issue is not None:
            issues.append(duplicate_issue)
        missing = sorted(set(collection.recording_ids) - recording_names)
        if missing:
            issues.append(
                ValidationIssue(
                    path,
                    f"references missing recordings: {missing!r}",
                )
            )
        if verify_integrity or require_integrity:
            oversized = _oversized_arrays(
                _view_arrays(path, collection.group), max_array_bytes
            )
            if (
                verify_integrity
                and collection.integrity.get("metadata_sha512") is not None
            ):
                issues.extend(oversized)
            issue = _integrity_issue(
                path=path,
                integrity=collection.integrity,
                calculators={
                    "metadata_sha512": collection.calculate_metadata_sha512
                },
                verify=verify_integrity,
                require_integrity=require_integrity,
                unverified=(
                    frozenset({"metadata_sha512"})
                    if oversized
                    else frozenset()
                ),
            )
            if issue is not None:
                issues.append(issue)
    except _READ_ERRORS as exc:
        issues.append(ValidationIssue(path, _issue_message(exc)))
    return issues


def _store_index_issues(
    store: SigMFZarrStore,
) -> tuple[list[ValidationIssue], int]:
    """Validate every store-wide index.

    Args:
        store: Store whose indexes to validate.

    Returns:
        Validation issues and number of indexes checked.
    """
    try:
        issues = _attribute_issues("indexes", store.indexes)
        index_names = sorted(
            name
            for name, member in store.indexes.members(max_depth=None)
            if isinstance(member, ReadOnlyArray)
        )
    except _READ_ERRORS as exc:
        # Zarr parses every array document to list members, so one damaged
        # document prevents listing the others.
        return [ValidationIssue("indexes", _issue_message(exc))], 0
    for index_name in index_names:
        # `_hierarchy_issues` reports a malformed name once for every node.
        if has_invalid_name_character(index_name):
            continue
        try:
            store.index(index_name)
        except _READ_ERRORS as exc:
            issues.append(
                ValidationIssue(f"indexes/{index_name}", _issue_message(exc))
            )
    return issues, len(index_names)


def _store_integrity_issues(
    store: SigMFZarrStore,
    *,
    verify_integrity: bool,
    require_integrity: bool,
    max_array_bytes: int | None,
) -> list[ValidationIssue]:
    """Validate store-wide integrity metadata.

    Recordings and collections are verified separately, so this checks only
    the store-wide hash over root metadata and the declared child digests.

    Args:
        store: Store whose integrity metadata to validate.
        verify_integrity: Whether to verify a declared store-wide hash.
        require_integrity: Whether the store must declare integrity.
        max_array_bytes: Greatest logical array size to hash, or ``None``
            for no limit.

    Returns:
        Store-wide integrity issues.
    """
    issues = []
    try:
        oversized = _oversized_arrays(
            _root_arrays(store._refresh_root()), max_array_bytes
        )
        if (
            verify_integrity
            and store.integrity.get("metadata_sha512") is not None
        ):
            issues.extend(oversized)
        issue = _integrity_issue(
            path="/",
            integrity=store.integrity,
            calculators={"metadata_sha512": store.calculate_metadata_sha512},
            verify=verify_integrity,
            require_integrity=require_integrity,
            unverified=(
                frozenset({"metadata_sha512"}) if oversized else frozenset()
            ),
        )
    except _READ_ERRORS as exc:
        message = f"could not verify integrity: {_issue_message(exc)}"
        return [*issues, ValidationIssue("/", message)]
    return issues if issue is None else [*issues, issue]


def _root_arrays(root: Group) -> Iterator[tuple[str, Array[Any]]]:
    """List the arrays that the store-wide metadata hash covers.

    Arrays inside recordings and collections are covered by their own
    hashes and are omitted.

    Args:
        root: Root store group.

    Yields:
        Logical path and array of each array.
    """
    for name, member in sorted(members(root)):
        if isinstance(member, Array):
            yield name, member
        elif name in {"recordings", "collections"}:
            # Only the immediate arrays of these containers are root content.
            for child, array in sorted(members(member)):
                if isinstance(array, Array):
                    yield f"{name}/{child}", array
        else:
            for child, descendant in sorted(members(member, max_depth=None)):
                if isinstance(descendant, Array):
                    yield f"{name}/{child}", descendant


def validate_store(
    store_or_path: SigMFZarrStore | str | PathLike[str],
    *,
    verify_integrity: bool = True,
    require_integrity: bool = False,
    max_array_bytes: int | None = None,
) -> ValidationReport:
    """Validate the complete logical structure and optional integrity hashes.

    Validation also checks shared recording metadata against the SigMF core
    schema and collection metadata against the SigMF collection schema.
    Before reading a local store, validation walks it without following
    links and reports every symbolic link, every entry that is neither a
    regular file nor a directory, such as a FIFO or device, and every
    directory that cannot be listed. If it finds any, it reads no content
    and reports only those entries. A store opened through an fsspec local
    file system, as in `local:///data/a.zarr`, is a local store. Validation
    also reports consolidated metadata, names with control characters or
    lone surrogates, structured array dtypes, and keys under `recordings/`
    or `collections/` that belong to no node.

    Args:
        store_or_path: Open store, local path, or URL with an explicit
            scheme.
        verify_integrity: Whether to verify each hash that is present,
            including `core:sha512`. Each declared hash is checked
            separately, so an absent hash does not fail this check. Each
            sample array is read once per hash.
        require_integrity: Whether every resource must declare integrity.
        max_array_bytes: Greatest size, in bytes, of an array or of one
            decoded chunk or shard to read. A declared hash that covers a
            larger array, or an array with a larger chunk or shard, is not
            verified, and each such array is reported as an issue.
            Variable-length strings and objects count 16 bytes per element.
            The limit also bounds the per-item metadata entries that are
            checked: entries in a larger chunk are not read, and checking
            stops once the decoded entries exceed the limit. Each case is
            reported as an issue. ``None`` reads every array and entry.

    Returns:
        Complete validation report. Structural failures, including read
        failures, are reported rather than raised after the root store has
        opened successfully.

    Raises:
        ValueError: If the root is not a SigMF-Zarr store, or
            `max_array_bytes` is not a positive integer.
        OSError: If the root store cannot be read.
    """
    # Untyped callers can pass any value.
    if max_array_bytes is not None and (
        type(max_array_bytes) is not int or max_array_bytes <= 0
    ):
        raise ValueError("max_array_bytes must be a positive integer")
    local_root = _local_store_root(store_or_path)
    if local_root is not None:
        file_issues = _local_file_issues(local_root)
        if file_issues:
            # Reading content could block, exhaust memory, or follow a link
            # out of the store.
            return ValidationReport(
                issues=tuple(
                    sorted(
                        file_issues,
                        key=lambda issue: (issue.path, issue.message),
                    )
                ),
                recordings_checked=0,
                collections_checked=0,
                indexes_checked=0,
            )
    store = (
        store_or_path
        if isinstance(store_or_path, SigMFZarrStore)
        else SigMFZarrStore.open(store_or_path, mode="r")
    )
    # The local walk above has ruled out special files and links, so these
    # whole-store reads cannot block or leave the store.
    issues = _hierarchy_issues(store)
    issues.extend(_orphan_prefix_issues(store))
    integrity_issue = _integrity_type_issue("/", store.metadata())
    if integrity_issue is not None:
        issues.append(integrity_issue)
    # Store internals are used so that one unreadable child document does
    # not prevent listing the others.
    reserved_recordings = _reported(
        "recordings",
        issues,
        _EMPTY_NAMES,
        lambda: _reserved_child_names(store._recordings_group),
    )
    reserved_collections = _reported(
        "collections",
        issues,
        _EMPTY_NAMES,
        lambda: _reserved_child_names(store._collections_group),
    )
    recording_names = tuple(
        name
        for name in _reported(
            "recordings",
            issues,
            _EMPTY_NAMES,
            lambda: _child_group_names(store._recordings_group),
        )
        if name not in ZARR_METADATA_KEYS
    )
    collection_names = tuple(
        name
        for name in _reported(
            "collections",
            issues,
            _EMPTY_NAMES,
            lambda: _child_group_names(store._collections_group),
        )
        if name not in ZARR_METADATA_KEYS
    )
    issues.extend(
        ValidationIssue(
            f"{kind}/{name}",
            "name is reserved for Zarr metadata documents",
        )
        for kind, names in (
            ("recordings", reserved_recordings),
            ("collections", reserved_collections),
        )
        for name in names
    )
    issues.extend(
        ValidationIssue(f"{kind}/{name}", f"{kind} must contain only groups")
        for kind, group in (
            ("recordings", store._recordings_group),
            ("collections", store._collections_group),
        )
        for name in _child_array_names(group)
    )
    indexes_checked = 0

    for recording_name in recording_names:
        recording_issues, recording_indexes = _recording_issues(
            store,
            recording_name,
            verify_integrity=verify_integrity,
            require_integrity=require_integrity,
            max_array_bytes=max_array_bytes,
        )
        issues.extend(recording_issues)
        indexes_checked += recording_indexes

    available_recordings = set(recording_names)
    for collection_name in collection_names:
        issues.extend(
            _collection_issues(
                store,
                collection_name,
                recording_names=available_recordings,
                verify_integrity=verify_integrity,
                require_integrity=require_integrity,
                max_array_bytes=max_array_bytes,
            )
        )

    try:
        json_value(store.metadata(), name="attributes")
    except ValueError as exc:
        issues.append(ValidationIssue("/", str(exc)))
    index_issues, store_indexes = _store_index_issues(store)
    issues.extend(index_issues)
    indexes_checked += store_indexes

    if verify_integrity or require_integrity:
        issues.extend(
            _store_integrity_issues(
                store,
                verify_integrity=verify_integrity,
                require_integrity=require_integrity,
                max_array_bytes=max_array_bytes,
            )
        )

    return ValidationReport(
        # Stable ordering makes text output and automated comparisons
        # independent of the backing store's traversal order.
        issues=tuple(
            sorted(issues, key=lambda issue: (issue.path, issue.message))
        ),
        recordings_checked=len(recording_names) + len(reserved_recordings),
        collections_checked=len(collection_names) + len(reserved_collections),
        indexes_checked=indexes_checked,
    )


__all__ = ["ValidationIssue", "ValidationReport", "validate_store"]

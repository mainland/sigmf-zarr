"""Rollback boundaries for resource replacements and import operations."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from zarr.abc.store import Store
from zarr.core.buffer import default_buffer_prototype
from zarr.core.common import concurrent_map
from zarr.core.config import config as zarr_config
from zarr.core.group import Group
from zarr.core.sync import sync
from zarr.storage import LocalStore

_OPERATION: ContextVar[object | None] = ContextVar(
    "sigmf_zarr_operation", default=None
)
"""Marker identifying Zarr tasks started by the current transaction.

Zarr runs coroutines on its own event loop. Each task copies the context of
the thread that submitted it, and child tasks copy their parent's context,
so every task an operation starts carries the operation's marker.
"""


async def _wait_for_tasks(marker: object) -> None:
    """Wait until no task carrying `marker` is pending on the running loop.

    Args:
        marker: Operation marker that the tasks' contexts hold.
    """
    current = asyncio.current_task()
    while pending := {
        task
        for task in asyncio.all_tasks()
        if task is not current and task.get_context().get(_OPERATION) is marker
    }:
        await asyncio.wait(pending)


@contextmanager
def _settled_on_failure() -> Iterator[None]:
    """Wait for an operation's remaining Zarr writes when it fails.

    Zarr issues node, parent, and chunk writes concurrently. When one write
    fails, the caller receives the error while sibling writes continue on
    Zarr's event loop. An interrupt in the caller leaves the whole operation
    running. Rollback must not inspect or restore the store until those
    writes finish. A nested context shares the outermost marker, so waiting
    covers every task the enclosing transaction started.

    Yields:
        Control to the operation.
    """
    marker = _OPERATION.get()
    token = None
    if marker is None:
        marker = object()
        token = _OPERATION.set(marker)
    try:
        yield
    except BaseException:
        sync(_wait_for_tasks(marker))
        raise
    finally:
        if token is not None:
            _OPERATION.reset(token)


@contextmanager
def retained_backup_directory(parent: Path) -> Iterator[Path]:
    """Keep recovery files if restoring an import or export backup fails.

    Callers must remove all recovery files after a successful rollback.
    An exception with remaining recovery files preserves the
    directory and includes its location in the raised error.

    Args:
        parent: Directory on the same filesystem as the destination.

    Yields:
        Temporary directory for backup files.

    Raises:
        OSError: If an operation fails with unrecovered backup files.
    """
    backup_dir = Path(
        tempfile.mkdtemp(dir=parent, prefix=".sigmf-zarr-backup-")
    )
    try:
        yield backup_dir
    except BaseException as exc:
        if any(backup_dir.iterdir()):
            raise OSError(
                "Rollback failed. Recovery files are preserved in "
                f"{backup_dir}"
            ) from exc
        backup_dir.rmdir()
        raise
    else:
        shutil.rmtree(backup_dir)


async def _copy_prefix(
    source: Store, target: Store, source_prefix: str, target_prefix: str
) -> None:
    """Copy stored objects with at most one object in memory.

    Args:
        source: Backend supplying the saved objects.
        target: Backend receiving copies.
        source_prefix: Source directory prefix, including its final slash.
        target_prefix: Replacement prefix, including its final slash if any.

    Raises:
        OSError: If a listed object disappears during the copy.
    """
    async for key in source.list_prefix(source_prefix):
        value = await source.get(key, prototype=default_buffer_prototype())
        if value is None:
            raise OSError(f"Stored object disappeared during backup: {key}")
        await target.set(target_prefix + key[len(source_prefix) :], value)


def _backup_parent(backend: LocalStore) -> Path:
    """Return the directory that holds backups for a local store.

    A backup must be on the same filesystem as the store and outside it.
    The parent of a relative root such as `.` is `.`, which is the store
    itself, so the root is resolved first.

    Args:
        backend: Local directory store.

    Returns:
        Parent directory of the resolved store root.
    """
    return backend.root.resolve().parent


def _listable(backend: Store) -> bool:
    """Return whether a backend can list and delete objects.

    Args:
        backend: Store backend.

    Returns:
        True when the backend is writable and supports listing and deletes.
    """
    return (
        not backend.read_only
        and backend.supports_deletes
        and backend.supports_listing
    )


def _first_key_below(backend: Store, path: str) -> str | None:
    """Return one stored object below a node path, if any.

    Args:
        backend: Listable store backend.
        path: Node path without a trailing slash.

    Returns:
        A key below `path`, or `None` when no object is stored there.
    """

    async def first() -> str | None:
        async for key in backend.list_prefix(path + "/"):
            return key
        return None

    return sync(first())


_NODE_DOCUMENT_ORDER = (".zmetadata", ".zattrs", ".zarray", ".zgroup")
"""Metadata documents of a node, deleted in this order and before `zarr.json`.

A Zarr format 2 node exists while its `.zarray` or `.zgroup` document
exists, so that document goes last.
"""


async def _delete_tree(backend: Store, path: str) -> None:
    """Delete every object below a node, the node's own metadata last.

    Zarr deletes a prefix in listing order, which can remove the node's
    metadata document first. An interrupted deletion then leaves objects
    that belong to no visible node. Deleting the node's own metadata last
    keeps the node visible until nothing else remains, so a repeated
    deletion can find and finish it.

    Args:
        backend: Writable, listable store backend.
        path: Node path without a trailing slash.
    """
    order = {
        f"{path}/{name}": rank
        for rank, name in enumerate((*_NODE_DOCUMENT_ORDER, "zarr.json"))
    }
    keys = [key async for key in backend.list_prefix(path + "/")]
    await concurrent_map(
        [(key,) for key in keys if key not in order],
        backend.delete,
        zarr_config.get("async.concurrency"),
    )
    for key in sorted(
        (key for key in keys if key in order), key=order.__getitem__
    ):
        await backend.delete(key)
    # A local store keeps empty directories after deleting their files.
    await backend.delete_dir(path)


def delete_node(group: Group, name: str) -> None:
    """Delete a node and every object below it.

    On a writable, listable store, the node's own metadata document is
    deleted last, so a failure leaves the node visible and a repeated call
    finishes the deletion. When one delete fails, the call returns only
    after the concurrent deletes finish.

    Args:
        group: Parent group.
        name: Relative node name, including any nested path.
    """
    backend = group.store
    if not _listable(backend):
        del group[name]
        return
    with _settled_on_failure():
        sync(_delete_tree(backend, (group.store_path / name).path))


def delete_orphan(group: Group, name: str) -> bool:
    """Delete objects stored below a name that has no node metadata.

    A failed write or deletion can leave objects under a node path without
    the node's metadata document. Zarr then reports no node, so the objects
    are invisible to every operation that lists nodes.

    Args:
        group: Parent group.
        name: Relative name of the missing node.

    Returns:
        True when orphan objects were found and deleted, False when nothing
        is stored below `name` or the backend cannot list objects.
    """
    backend = group.store
    if not _listable(backend):
        return False
    path = (group.store_path / name).path
    if _first_key_below(backend, path) is None:
        return False
    with _settled_on_failure():
        sync(_delete_tree(backend, path))
    return True


@contextmanager
def _creation_transaction(
    group: Group, name: str, *, parents: Sequence[Group]
) -> Iterator[None]:
    """Remove a partially created node if its creation fails.

    Args:
        group: Parent that will contain the new node.
        name: Relative node name, including any nested path.
        parents: Groups whose attributes the creation may invalidate.

    Yields:
        Control to the creating operation.
    """
    attributes = [(parent, parent.attrs.asdict()) for parent in parents]
    try:
        yield
    except BaseException:
        # Leave the name available and restore any invalidated ancestor
        # hashes, because the store content is unchanged. Reopen parents to
        # observe hash changes made through other handles. Compare first to
        # avoid writes to an unchanged read-only store.
        if name in group:
            del group[name]
        for parent, attrs in attributes:
            current = Group.open(
                parent.store_path, zarr_format=parent.metadata.zarr_format
            )
            if current.attrs.asdict() != attrs:
                parent.attrs.put(attrs)
        raise


@contextmanager
def replacement_transaction(
    group: Group,
    name: str,
    *,
    overwrite: bool,
    parents: Sequence[Group],
) -> Iterator[None]:
    """Preserve an existing node and ancestor attributes during replacement.

    Callers must serialize access throughout the context. The old node is
    removed only after its recovery copy is available. Local directories move
    to a backup on the same filesystem, and rollback moves them back, so
    neither step needs free space. Other backends copy encoded objects to
    temporary disk storage, retaining at most one object in memory.
    Replacement is recoverable from Python exceptions, not process crashes.

    When the node is absent on entry, a failure removes any partially created
    node and restores the ancestor attributes instead, so a failed creation
    leaves the name available. With `overwrite`, objects stored below an
    absent node's path, such as those a failed write leaves behind, are
    replaced like an existing node. Without it, they make creation fail.

    Only creation and replacement use this context. Opening a node must not
    write or delete anything, even when the open fails.

    Args:
        group: Parent containing the target node.
        name: Relative node name, including any nested path.
        overwrite: Whether the operation replaces an existing node.
        parents: Groups whose attributes the operation may invalidate.

    Yields:
        Control with the old node absent and its backup retained.

    Raises:
        ValueError: If the backend is read-only or cannot support rollback,
            or objects that belong to no node are stored below the new
            node's path and `overwrite` is false.
        OSError: If rollback fails. The error identifies retained recovery
            files, including the saved ancestor attributes.
    """
    exists = name in group
    if exists and not overwrite:
        # The creating operation reports the existing node.
        yield
        return
    path = (group.store_path / name).path
    if exists or (
        overwrite
        and _listable(group.store)
        and _first_key_below(group.store, path) is not None
    ):
        with _backup_transaction(group.store, path, parents=parents):
            yield
        return
    with _creation_transaction(group, name, parents=parents):
        yield


@contextmanager
def _backup_transaction(
    backend: Store, path: str, *, parents: Sequence[Group]
) -> Iterator[None]:
    """Move stored objects to a backup and restore them on failure.

    Args:
        backend: Store containing the objects.
        path: Node path whose objects are replaced.
        parents: Groups whose attributes the operation may invalidate.

    Yields:
        Control with the objects removed and their backup retained.

    Raises:
        ValueError: If the backend is read-only or cannot list or delete.
        OSError: If rollback fails. The error identifies retained recovery
            files, including the saved ancestor attributes.
    """
    if not _listable(backend):
        raise ValueError(
            "Replacing or removing a node requires a writable, listable store"
        )
    attributes = [(parent, parent.attrs.asdict()) for parent in parents]
    local = backend.root / path if isinstance(backend, LocalStore) else None
    directory = (
        _backup_parent(backend)
        if isinstance(backend, LocalStore)
        else Path(tempfile.gettempdir())
    )
    with retained_backup_directory(directory) as backup:
        saved = backup / "node"
        attrs_file = backup / "parent-attributes.json"
        try:
            attrs_file.write_text(
                json.dumps(
                    {parent.path: attrs for parent, attrs in attributes}
                ),
                encoding="ascii",
            )
            if local is not None:
                local.rename(saved)
            else:
                with LocalStore(saved) as snapshot:
                    sync(_copy_prefix(backend, snapshot, path + "/", ""))
        except BaseException:
            # Backup failure precedes mutation of the original node.
            if saved.exists():
                shutil.rmtree(saved)
            attrs_file.unlink(missing_ok=True)
            raise
        try:
            with _settled_on_failure():
                if local is None:
                    sync(backend.delete_dir(path))
                yield
        except BaseException:
            _restore_node(backend, path, saved, local)
            for parent, attrs in attributes:
                parent.attrs.put(attrs)
            if local is None:
                shutil.rmtree(saved)
            attrs_file.unlink()
            raise


def _restore_node(
    backend: Store, path: str, saved: Path, local: Path | None
) -> None:
    """Replace a partially written node with its backup.

    Args:
        backend: Store containing the node.
        path: Node path in `backend`.
        saved: Backup directory, which is a local store for a nonlocal
            backend.
        local: Node directory for a local backend, otherwise `None`.
    """
    sync(backend.delete_dir(path))
    if local is None:
        with LocalStore(saved, read_only=True) as snapshot:
            sync(_copy_prefix(snapshot, backend, "", path + "/"))
        return
    # Moving the backup back needs no free space, unlike a copy, so rollback
    # also succeeds after the disk fills.
    if local.exists():
        shutil.rmtree(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    saved.rename(local)

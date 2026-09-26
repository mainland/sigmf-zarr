"""Rollback boundaries for resource replacements and import operations."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from zarr.abc.store import Store
from zarr.core.buffer import default_buffer_prototype
from zarr.core.group import Group
from zarr.core.sync import sync
from zarr.storage import LocalStore

if TYPE_CHECKING:
    pass


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
        await target.set(target_prefix + key[len(source_prefix):], value)


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
    to a backup on the same filesystem. Other backends copy encoded objects
    to temporary disk storage, retaining at most one object in memory.
    Replacement is recoverable from Python exceptions, not process crashes.

    Args:
        group: Parent containing the target node.
        name: Relative node name, including any nested path.
        overwrite: Whether the operation replaces an existing node.
        parents: Groups whose attributes the operation may invalidate.

    Yields:
        Control with the old node absent and its backup retained.

    Raises:
        ValueError: If the backend is read-only or cannot support rollback.
        OSError: If rollback fails. The error identifies retained recovery
            files, including the saved ancestor attributes.
    """
    if not overwrite or name not in group:
        yield
        return
    backend = group.store
    if backend.read_only or not (
        backend.supports_deletes and backend.supports_listing
    ):
        raise ValueError(
            "Replacing a node requires a writable, listable store"
        )
    path = group[name].path
    attributes = [(parent, parent.attrs.asdict()) for parent in parents]
    local = backend.root / path if isinstance(backend, LocalStore) else None
    directory = (
        backend.root.parent
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
            if local is None:
                sync(backend.delete_dir(path))
            yield
        except BaseException:
            sync(backend.delete_dir(path))
            if local is not None:
                shutil.copytree(saved, local)
            else:
                with LocalStore(saved, read_only=True) as snapshot:
                    sync(_copy_prefix(snapshot, backend, "", path + "/"))
            for parent, attrs in attributes:
                parent.attrs.put(attrs)
            shutil.rmtree(saved)
            attrs_file.unlink()
            raise

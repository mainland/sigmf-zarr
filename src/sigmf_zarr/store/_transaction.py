"""Rollback boundaries for recording and collection import operations."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from zarr.storage import LocalStore

from sigmf_zarr.store._container import SigMFZarrStore


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


@contextmanager
def recording_import_transaction(
    store: SigMFZarrStore,
    recording_name: str,
    *,
    overwrite: bool,
) -> Iterator[None]:
    """Restore a local recording when its replacement import fails.

    Callers must serialize writes to the store for the entire context. A
    backup stays outside the store until successful completion. Replacing an
    existing recording requires a local directory store.

    Args:
        store: Destination store.
        recording_name: Recording being created or replaced.
        overwrite: Whether to permit replacing an existing recording.

    Yields:
        Control to the importer after preserving any existing recording.

    Raises:
        ValueError: If a recording already exists and replacement is disabled
            or its backend does not support local rollback.
    """
    exists = recording_name in store.recordings
    if exists and not overwrite:
        raise ValueError(
            f"Recording {recording_name!r} already exists. Pass "
            "overwrite_recording=True to replace it"
        )
    if not exists:
        try:
            yield
        except BaseException:
            if recording_name in store.recordings:
                store.recordings.remove(recording_name)
            raise
        return

    with _existing_group_import_transaction(
        store,
        store._recordings_group[recording_name].path,
        backup_name="recording",
    ):
        yield


@contextmanager
def collection_import_transaction(
    store: SigMFZarrStore, collection_name: str
) -> Iterator[None]:
    """Restore an archive collection when any part of its import fails.

    Args:
        store: Destination store with serialized writes.
        collection_name: Collection being created or replaced.

    Yields:
        Control to the importer after preserving any existing collection.

    Raises:
        ValueError: If replacement uses a nonlocal backend.
    """
    if collection_name not in store.collections:
        try:
            yield
        except BaseException:
            if collection_name in store.collections:
                store._invalidate_metadata_integrity()
                del store._collections_group[collection_name]
            raise
        return
    with _existing_group_import_transaction(
        store,
        store._collections_group[collection_name].path,
        backup_name="collection",
    ):
        yield


@contextmanager
def _existing_group_import_transaction(
    store: SigMFZarrStore,
    group_path: str,
    *,
    backup_name: str,
) -> Iterator[None]:
    """Preserve one existing local group until its import completes.

    Args:
        store: Destination store with serialized writes.
        group_path: Existing recording or collection path in the store.
        backup_name: Directory name identifying the backed-up resource.

    Yields:
        Control to the importer after backup creation.

    Raises:
        ValueError: If the backend is not a local directory store.
    """
    backend = store._group.store
    if not isinstance(backend, LocalStore):
        raise ValueError(
            "Replacing an existing resource during import requires a "
            "local directory store"
        )
    recording_path = backend.root / group_path
    # Importers update store-level integrity as well as the resource. Restore
    # both together so rollback does not leave hashes for the failed import.
    original_attrs = store._group.attrs.asdict()
    with retained_backup_directory(backend.root.parent) as backup_dir:
        backup_path = backup_dir / backup_name
        attrs_path = backup_dir / "root-attributes.json"
        try:
            shutil.copytree(recording_path, backup_path)
            attrs_path.write_text(json.dumps(original_attrs), encoding="ascii")
        except BaseException:
            # No destination mutation has occurred during backup creation.
            if backup_path.exists():
                shutil.rmtree(backup_path)
            attrs_path.unlink(missing_ok=True)
            raise
        try:
            yield
        except BaseException:
            if recording_path.exists():
                shutil.rmtree(recording_path)
            # Retain the copy until both data and root metadata are restored.
            shutil.copytree(backup_path, recording_path)
            store._group.attrs.put(original_attrs)
            shutil.rmtree(backup_path)
            attrs_path.unlink()
            raise

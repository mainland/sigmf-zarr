"""Open the target store of a dataset import and clean up after failure."""

from __future__ import annotations

import functools
import inspect
import os
import shutil
import stat
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from os import PathLike
from pathlib import Path

import zarr
from zarr.core.sync import sync
from zarr.storage import FsspecStore

from sigmf_zarr.store import SigMFZarrStore, ZarrFormat
from sigmf_zarr.store._common import store_location

# Directories that `open_import_store` created during the innermost
# importer call, with the identity each had when it was created.
_created_targets: ContextVar[list[tuple[Path, os.stat_result]] | None] = (
    ContextVar("_created_targets", default=None)
)


def remove_new_store_on_failure[**P, R](
    importer: Callable[P, R],
) -> Callable[P, R]:
    """Remove a local target store that a failed import created.

    The wrapped importer names its target store with a `store_path`
    parameter. If nothing exists at that local path when the import starts
    and the import raises, the directory created at that path is removed
    before the exception propagates. An existing store or directory is never
    removed, because recording rollback inside the importer restores it. A
    store named by a URL is never removed.

    Args:
        importer: Import function with a `store_path` parameter.

    Returns:
        Import function with the same signature.
    """
    signature = inspect.signature(importer)

    @functools.wraps(importer)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        store_path = signature.bind(*args, **kwargs).arguments["store_path"]
        location = (
            store_location(store_path)
            if isinstance(store_path, str | PathLike)
            else None
        )
        new_target = location if isinstance(location, Path) else None
        if new_target is not None and (
            new_target.exists() or new_target.is_symlink()
        ):
            new_target = None
        try:
            return importer(*args, **kwargs)
        except BaseException as exc:
            if (
                new_target is not None
                and new_target.is_dir()
                and not new_target.is_symlink()
            ):
                try:
                    shutil.rmtree(new_target)
                except OSError as cleanup_exc:
                    exc.add_note(
                        "Could not remove the new store "
                        f"{str(new_target)!r}: {cleanup_exc}"
                    )
            raise

    return wrapper


def _remove_created_directory(
    path: Path, identity: os.stat_result, exc: BaseException
) -> None:
    """Remove a directory while it is the one that the import created.

    Args:
        path: Directory that `open_import_store` created.
        identity: `lstat` result recorded when the directory was created.
        exc: Import failure, which receives a note if removal fails.
    """
    try:
        current = path.lstat()
    except FileNotFoundError:
        return
    except OSError as cleanup_exc:
        exc.add_note(
            f"Could not inspect the new store {str(path)!r}: {cleanup_exc}"
        )
        return
    # Another process may have replaced the directory since its creation.
    if not os.path.samestat(current, identity):
        return
    try:
        shutil.rmtree(path)
    except OSError as cleanup_exc:
        exc.add_note(
            f"Could not remove the new store {str(path)!r}: {cleanup_exc}"
        )


def open_import_store[T](
    store_path: str | PathLike[str],
    *,
    overwrite_store: bool,
    zarr_format: ZarrFormat | None,
    sources: Sequence[str | PathLike[str]],
    resolve_storage: Callable[[ZarrFormat], T],
) -> tuple[SigMFZarrStore, T]:
    """Open or create the target store of an import.

    With `overwrite_store`, the target must be absent, an empty directory,
    or a SigMF-Zarr root, and a local target must neither equal nor contain
    an import source. `resolve_storage` runs before the target is replaced,
    so invalid storage options leave the target unchanged. A local target
    directory that this function creates is recorded for removal by
    `remove_new_store_on_failure`.

    Args:
        store_path: Target store path or URL.
        overwrite_store: Whether to replace an existing SigMF-Zarr store.
        zarr_format: Optional physical Zarr format requirement.
        sources: Local input paths that the import reads.
        resolve_storage: Function that validates and resolves sample storage
            options for the physical Zarr format of the target.

    Returns:
        Open store and the result of `resolve_storage`.

    Raises:
        ValueError: If `overwrite_store` names a target that may not be
            replaced, `resolve_storage` rejects the options, or an existing
            target is not a compatible SigMF-Zarr store.
    """
    location = store_location(store_path)
    if overwrite_store:
        storage = resolve_storage(3 if zarr_format is None else zarr_format)
        if _store_to_replace(location, sources):
            store = SigMFZarrStore(
                store_path, overwrite=True, zarr_format=zarr_format
            )
            return store, storage
        if isinstance(location, Path):
            _create_target_directory(location)
        return SigMFZarrStore(store_path, zarr_format=zarr_format), storage
    if isinstance(location, Path):
        _create_target_directory(location)
    store = SigMFZarrStore(store_path, zarr_format=zarr_format)
    return store, resolve_storage(store.zarr_format)


def _create_target_directory(location: Path) -> None:
    """Create a missing local target directory and record its identity.

    Args:
        location: Local target store path.
    """
    if os.path.lexists(location):
        return
    location.parent.mkdir(parents=True, exist_ok=True)
    try:
        location.mkdir()
    except FileExistsError:
        # Another process created the target after the check above.
        return
    created = _created_targets.get()
    if created is not None:
        created.append((location, location.lstat()))


def _store_to_replace(
    location: str | Path, sources: Sequence[str | PathLike[str]]
) -> bool:
    """Check that `overwrite_store` may replace whatever is at the target.

    Args:
        location: Local target path or store URL.
        sources: Local input paths that the import reads.

    Returns:
        Whether a SigMF-Zarr root exists at the target. False means that
        the target is absent or an empty directory, which the import
        initializes without deleting anything.

    Raises:
        ValueError: If the target is a symbolic link, is not a directory,
            equals or contains a source, or is a nonempty directory that is
            not a SigMF-Zarr root.
    """
    if not isinstance(location, Path):
        return _url_store_to_replace(location)
    try:
        mode = location.lstat().st_mode
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(mode):
        raise ValueError(
            f"Target store {str(location)!r} is a symbolic link. "
            "overwrite_store replaces only a directory"
        )
    if not stat.S_ISDIR(mode):
        raise ValueError(
            f"Target store {str(location)!r} is not a directory. "
            "overwrite_store replaces only a directory"
        )
    target = location.resolve()
    for source in sources:
        resolved = Path(source).resolve()
        if resolved == target or target in resolved.parents:
            raise ValueError(
                f"Target store {str(location)!r} contains the import source "
                f"{os.fspath(source)!r}. overwrite_store would delete it"
            )
    with os.scandir(location) as entries:
        if next(entries, None) is None:
            return False
    if not _is_sigmf_zarr_root(location):
        raise ValueError(
            f"Target {str(location)!r} is not a SigMF-Zarr store. "
            "overwrite_store replaces only a SigMF-Zarr store or an empty "
            "directory"
        )
    return True


def _url_store_to_replace(location: str) -> bool:
    """Check that `overwrite_store` may replace the content at a store URL.

    Args:
        location: Store URL.

    Returns:
        Whether a SigMF-Zarr root exists at the URL. False means that the URL
        names no content.

    Raises:
        ValueError: If the URL names content other than a SigMF-Zarr root.
    """
    if _is_sigmf_zarr_root(location):
        return True
    target = FsspecStore.from_url(location, read_only=True)
    if not sync(target.is_empty("")):
        raise ValueError(
            f"Target {location!r} is not a SigMF-Zarr store. "
            "overwrite_store replaces only a SigMF-Zarr store or an empty "
            "location"
        )
    return False


def _is_sigmf_zarr_root(location: str | Path) -> bool:
    """Return whether a location holds a root group marked as SigMF-Zarr.

    Args:
        location: Local path or store URL.

    Returns:
        True when the root group has the SigMF-Zarr schema name. A store
        with an unsupported schema version still counts.
    """
    try:
        group = zarr.open_group(location, mode="r", use_consolidated=False)
    except (OSError, ValueError, TypeError, KeyError):
        # A missing, unreadable, or malformed root group is not a store.
        return False
    return group.attrs.get("schema_name") == SigMFZarrStore.SCHEMA_NAME

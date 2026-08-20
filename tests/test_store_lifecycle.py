"""Tests for opening, creating, and closing stores and collections."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from zarr.storage import LocalStore

from sigmf_zarr.store import SigMFCollection, SigMFZarrStore, ZarrFormat


def test_create_refuses_existing_store(tmp_path: Path) -> None:
    """Creation must not adopt or replace an existing store implicitly.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    path = tmp_path / "store.zarr"
    SigMFZarrStore.create(path).collections.create("kept")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        SigMFZarrStore.create(path)
    assert SigMFZarrStore.open(path).list_collections() == ("kept",)
    assert SigMFZarrStore.create(path, overwrite=True).list_collections() == ()


def test_create_accepts_empty_directory(tmp_path: Path) -> None:
    """An existing empty directory has no hierarchy to protect.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path)
    assert store.zarr_format == 3
    assert store.list_recordings() == ()


def test_open_never_creates_a_store(tmp_path: Path) -> None:
    """Opening a missing target must fail without writing anything.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    path = tmp_path / "missing.zarr"
    for mode in ("r", "r+"):
        with pytest.raises(FileNotFoundError):
            SigMFZarrStore.open(path, mode=cast(Any, mode))
    assert not path.exists()


@pytest.mark.parametrize("mode", ["a", "w", "w-"])
def test_open_rejects_creating_modes(tmp_path: Path, mode: str) -> None:
    """Open accepts only modes that require an existing store.

    Args:
        tmp_path: Pytest temporary path fixture.
        mode: Rejected Zarr access mode.
    """
    path = tmp_path / "store.zarr"
    with pytest.raises(ValueError, match="mode"):
        SigMFZarrStore.open(path, mode=cast(Any, mode))
    assert not path.exists()


def test_open_r_plus_writes_existing_store(tmp_path: Path) -> None:
    """Mode `r+` opens an existing store for managed writes.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    path = tmp_path / "store.zarr"
    SigMFZarrStore.create(path)
    SigMFZarrStore.open(path, mode="r+").collections.create("new")
    assert SigMFZarrStore.open(path).list_collections() == ("new",)


@pytest.mark.parametrize(("existing", "requested"), [(3, 2), (2, 3)])
def test_requested_format_must_match_existing_store(
    tmp_path: Path, existing: int, requested: int
) -> None:
    """A format request must not add a second hierarchy to a store.

    Args:
        tmp_path: Pytest temporary path fixture.
        existing: Format of the existing store.
        requested: Conflicting requested format.
    """
    path = tmp_path / "store.zarr"
    SigMFZarrStore.create(path, zarr_format=cast(ZarrFormat, existing))
    before = sorted(entry.name for entry in path.iterdir())
    for mode in ("a", "r", "r+"):
        with pytest.raises(ValueError, match="Zarr format"):
            SigMFZarrStore(
                path,
                mode=cast(Any, mode),
                zarr_format=cast(ZarrFormat, requested),
            )
    assert sorted(entry.name for entry in path.iterdir()) == before
    reopened = SigMFZarrStore(path, zarr_format=cast(ZarrFormat, existing))
    assert reopened.zarr_format == existing


def test_append_mode_initializes_empty_target_as_format_3(
    tmp_path: Path,
) -> None:
    """The constructor's default mode creates a format-3 store when absent.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    path = tmp_path / "store.zarr"
    assert SigMFZarrStore(path).zarr_format == 3
    assert SigMFZarrStore(path).zarr_format == 3


def test_context_manager_closes_backend(tmp_path: Path) -> None:
    """Leaving a store context must close its backend.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    backend = LocalStore(tmp_path / "store.zarr")
    with SigMFZarrStore.create(backend) as store:
        assert backend._is_open
        store.collections.create("kept")
    assert not backend._is_open


def test_collections_open_and_create_are_strict(tmp_path: Path) -> None:
    """Collections must be opened only when present and created explicitly.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr")
    with pytest.raises(KeyError):
        store.collections.open("missing")
    with pytest.raises(KeyError):
        store.collections["missing"]
    assert store.list_collections() == ()

    store.collections.create("group", metadata={"author": "first"})
    with pytest.raises(ValueError, match="already exists"):
        store.collections.create("group")
    assert store.collections["group"].metadata == {"author": "first"}

    replaced = store.collections.create(
        "group", metadata={"author": "second"}, overwrite=True
    )
    assert replaced.metadata == {"author": "second"}
    with pytest.raises(ValueError, match="requires create=True"):
        SigMFCollection(store, "group", overwrite=True)

"""Tests for opening, creating, and closing stores and collections."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from zarr.storage import LocalStore

from sigmf_zarr.cli.context import StoreContext
from sigmf_zarr.store import SigMFCollection, SigMFZarrStore, ZarrFormat
from sigmf_zarr.store._common import is_store_url, store_location
from sigmf_zarr.validation import validate_store


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


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_path_location_is_never_parsed_as_url(
    tmp_path: Path, zarr_format: int
) -> None:
    """A path-like location names its directory, including `#` and `?`.

    Zarr parses a string as a URL and drops the fragment and query, which
    would select and replace the sibling directory `run`.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format.
    """
    sibling = tmp_path / "run"
    sibling.mkdir()
    (sibling / "precious.txt").write_text("unrelated data")
    path = tmp_path / "run#2.zarr?x=1"

    SigMFZarrStore.create(
        path, overwrite=True, zarr_format=cast(ZarrFormat, zarr_format)
    ).collections.create("kept")

    assert sorted(child.name for child in sibling.iterdir()) == [
        "precious.txt"
    ]
    assert (path / "collections").is_dir()
    with SigMFZarrStore.open(path) as store:
        assert store.zarr_format == zarr_format
        assert store.list_collections() == ("kept",)
    assert validate_store(path).issues == ()
    assert StoreContext().open(path).list_collections() == ("kept",)
    assert sorted(child.name for child in sibling.iterdir()) == [
        "precious.txt"
    ]


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize(
    ("location", "sibling", "overwrite"),
    [("run#2.zarr", "run", True), ("a?b.zarr", "a", False)],
)
def test_string_location_without_scheme_is_local_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    zarr_format: int,
    location: str,
    sibling: str,
    overwrite: bool,
) -> None:
    """A string without a URL scheme names its directory literally.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Working-directory fixture.
        zarr_format: Physical Zarr format.
        location: Relative store path with a URL delimiter.
        sibling: Directory that URL parsing would select instead.
        overwrite: Whether creation may replace existing content.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / sibling).mkdir()
    (tmp_path / sibling / "precious.txt").write_text("unrelated data")

    SigMFZarrStore.create(
        location,
        overwrite=overwrite,
        zarr_format=cast(ZarrFormat, zarr_format),
    ).collections.create("kept")

    assert sorted(child.name for child in tmp_path.iterdir()) == sorted(
        [location, sibling]
    )
    assert [child.name for child in (tmp_path / sibling).iterdir()] == [
        "precious.txt"
    ]
    assert (tmp_path / location / "collections" / "kept").is_dir()
    with SigMFZarrStore.open(location) as store:
        assert store.zarr_format == zarr_format
        assert store.list_collections() == ("kept",)
    assert validate_store(location).issues == ()
    assert StoreContext().open(location).list_collections() == ("kept",)


@pytest.mark.parametrize(
    ("location", "url"),
    [
        ("s3://bucket/store.zarr", True),
        ("file:///data/store.zarr", True),
        ("memory://store", True),
        ("simplecache::s3://bucket/store.zarr", True),
        ("run#2.zarr", False),
        ("a?b.zarr", False),
        ("/data/run#2.zarr", False),
        ("memory:store", False),
        ("C://data/store.zarr", False),
        ("data/s3://store.zarr", False),
    ],
)
def test_only_explicit_schemes_make_store_urls(
    location: str, url: bool
) -> None:
    """A string is a store URL only when it starts with a scheme.

    Args:
        location: Store location string.
        url: Whether Zarr should parse the location as a URL.
    """
    assert is_store_url(location) is url
    assert isinstance(store_location(location), str) is url


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_url_locations_reach_zarr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, zarr_format: ZarrFormat
) -> None:
    """File and memory URLs keep their URL meaning.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Working-directory fixture.
        zarr_format: Physical Zarr format.
    """
    monkeypatch.chdir(tmp_path)
    file_url = (tmp_path / "file.zarr").as_uri()
    SigMFZarrStore.create(
        file_url, zarr_format=zarr_format
    ).collections.create("kept")
    assert (tmp_path / "file.zarr" / "collections" / "kept").is_dir()
    with SigMFZarrStore.open(file_url) as store:
        assert store.zarr_format == zarr_format
        assert store.list_collections() == ("kept",)

    # zarr 3.2.0 gives each opening of a memory URL a new empty store, so
    # use one handle. Later releases route the URL through fsspec. Either
    # way, the URL must not become a local directory.
    memory_url = f"memory://{tmp_path.name}-{zarr_format}"
    with SigMFZarrStore.create(memory_url, zarr_format=zarr_format) as store:
        store.collections.create("kept")
        assert not isinstance(store._group.store, LocalStore)
        assert store.list_collections() == ("kept",)
        assert validate_store(store).issues == ()
    assert [child.name for child in tmp_path.iterdir()] == ["file.zarr"]


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

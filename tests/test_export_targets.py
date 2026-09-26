"""Tests for export source selection and target file names."""

from __future__ import annotations

import errno
import os
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import sigmf_zarr.sigmf as sigmf_module
from sigmf_zarr import SigMFZarrStore, export_sigmf, export_sigmf_archive


@pytest.fixture
def store(tmp_path: Path) -> SigMFZarrStore:
    """Create a writable, hashed store with three exportable recordings.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Store containing recordings `a`, `b`, and `c.v2`.
    """
    result = SigMFZarrStore.create(tmp_path / "store.zarr")
    for name in ("a", "b", "c.v2"):
        recording = result.recordings.create(
            name,
            sample_shape=(4,),
            global_metadata={"core:datatype": "rf32_le"},
            captures=[{"core:sample_start": 0}],
        )
        recording.set_samples(np.arange(4, dtype=np.float32))
    result.update_integrity()
    return result


def test_export_of_missing_recording_raises_key_error(
    store: SigMFZarrStore, tmp_path: Path
) -> None:
    """Exporting a missing recording must not try to create it.

    Args:
        store: Writable source store.
        tmp_path: Pytest temporary path fixture.
    """
    with pytest.raises(KeyError):
        export_sigmf(store, "missing", tmp_path / "out")
    assert "missing" not in store.recordings
    assert store.verify_integrity()


def test_export_retains_dots_in_target_stems(
    store: SigMFZarrStore, tmp_path: Path
) -> None:
    """A target stem keeps its dots, and SigMF suffixes are normalized.

    Args:
        store: Source store.
        tmp_path: Pytest temporary path fixture.
    """
    meta = export_sigmf(store, "c.v2", tmp_path / "out" / "c.v2")
    assert meta == tmp_path / "out" / "c.v2.sigmf-meta"
    assert (tmp_path / "out" / "c.v2.sigmf-data").is_file()
    data_target = export_sigmf(store, "a", tmp_path / "out" / "a.sigmf-data")
    assert data_target == tmp_path / "out" / "a.sigmf-meta"


def test_archive_export_of_missing_collection_raises_key_error(
    store: SigMFZarrStore, tmp_path: Path
) -> None:
    """Archive export must not create a missing collection in its source.

    Args:
        store: Writable source store.
        tmp_path: Pytest temporary path fixture.
    """
    with pytest.raises(KeyError):
        export_sigmf_archive(
            store, tmp_path / "bundle.sigmf", collection_name="typo"
        )
    assert store.list_collections() == ()
    assert store.verify_integrity()
    assert not (tmp_path / "bundle.sigmf").exists()


def test_collection_archive_contains_only_its_recordings(
    store: SigMFZarrStore, tmp_path: Path
) -> None:
    """A collection export contains its members and no other recording.

    Args:
        store: Source store.
        tmp_path: Pytest temporary path fixture.
    """
    store.collections.create("pair", recording_ids=["b", "a"])
    archive = export_sigmf_archive(
        store, tmp_path / "pair", collection_name="pair"
    )
    assert archive == tmp_path / "pair.sigmf"
    with tarfile.open(archive) as members:
        names = sorted(Path(name).name for name in members.getnames())
    assert names == [
        "a.sigmf-data",
        "a.sigmf-meta",
        "b.sigmf-data",
        "b.sigmf-meta",
        "pair",
        "pair.sigmf-collection",
    ]


def test_archive_retains_dots_in_target_stem(
    store: SigMFZarrStore, tmp_path: Path
) -> None:
    """An archive target stem keeps its dots.

    Args:
        store: Source store.
        tmp_path: Pytest temporary path fixture.
    """
    archive = export_sigmf_archive(
        store, tmp_path / "bundle.2026-10-06", recording_names=["a"]
    )
    assert archive == tmp_path / "bundle.2026-10-06.sigmf"
    assert archive.is_file()


@pytest.mark.parametrize("suffix", [".sigmf-meta", ".sigmf-data"])
def test_overwrite_rejects_directory_at_recording_target(
    store: SigMFZarrStore, tmp_path: Path, suffix: str
) -> None:
    """Overwrite replaces target files but never deletes a directory.

    Args:
        store: Source store.
        tmp_path: Pytest temporary path fixture.
        suffix: Target whose path holds a directory.
    """
    directory = (tmp_path / "out").with_suffix(suffix)
    directory.mkdir()
    (directory / "precious.txt").write_text("keep")
    with pytest.raises(ValueError, match="not a regular file"):
        export_sigmf(store, "a", tmp_path / "out", overwrite=True)
    assert (directory / "precious.txt").read_text() == "keep"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        directory.name,
        "store.zarr",
    ]


def test_overwrite_replaces_symbolic_link_at_recording_target(
    store: SigMFZarrStore, tmp_path: Path
) -> None:
    """Overwrite replaces a symbolic link itself, not its target.

    Args:
        store: Source store.
        tmp_path: Pytest temporary path fixture.
    """
    linked = tmp_path / "linked"
    linked.mkdir()
    link = tmp_path / "out.sigmf-data"
    link.symlink_to(linked, target_is_directory=True)
    export_sigmf(store, "a", tmp_path / "out", overwrite=True)
    assert not link.is_symlink()
    assert link.is_file()
    assert linked.is_dir()


def test_archive_overwrite_rejects_directory_target(
    store: SigMFZarrStore, tmp_path: Path
) -> None:
    """Archive overwrite never replaces a directory at the archive path.

    Args:
        store: Source store.
        tmp_path: Pytest temporary path fixture.
    """
    directory = tmp_path / "bundle.sigmf"
    directory.mkdir()
    (directory / "precious.txt").write_text("keep")
    with pytest.raises(ValueError, match="not a regular file"):
        export_sigmf_archive(store, directory, overwrite=True)
    assert (directory / "precious.txt").read_text() == "keep"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "bundle.sigmf",
        "store.zarr",
    ]


@pytest.mark.parametrize(
    "link_name", ["out.sigmf-meta", "out.sigmf-data", "out.sigmf"]
)
def test_export_treats_dangling_link_as_existing_target(
    store: SigMFZarrStore, tmp_path: Path, link_name: str
) -> None:
    """Export without overwrite keeps a symbolic link to a missing file.

    Args:
        store: Source store.
        tmp_path: Pytest temporary path fixture.
        link_name: Target occupied by a dangling link.
    """
    link = tmp_path / link_name
    link.symlink_to(tmp_path / "missing")
    if link.suffix == ".sigmf":
        with pytest.raises(ValueError, match="already exists"):
            export_sigmf_archive(store, tmp_path / "out")
    else:
        with pytest.raises(ValueError, match="already exist"):
            export_sigmf(store, "a", tmp_path / "out")
    assert link.is_symlink()
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [link_name, "store.zarr"]
    )


@pytest.mark.parametrize("hard_links", [True, False])
@pytest.mark.parametrize("planted", ["out.sigmf-meta", "out.sigmf-data"])
def test_export_never_replaces_target_created_during_export(
    store: SigMFZarrStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hard_links: bool,
    planted: str,
) -> None:
    """Publication refuses a file created after the existence check.

    Args:
        store: Source store.
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
        hard_links: Whether the file system supports hard links.
        planted: Target that another writer creates during export.
    """
    if not hard_links:

        def no_link(*args: Any, **kwargs: Any) -> None:
            """Fail as a file system without hard links does.

            Args:
                *args: Ignored positional arguments.
                **kwargs: Ignored keyword arguments.

            Raises:
                OSError: Always.
            """
            raise OSError(errno.EPERM, "Operation not permitted")

        monkeypatch.setattr(os, "link", no_link)
    write = sigmf_module._write_sigmf_data_file

    def write_and_plant(*args: Any, **kwargs: Any) -> str:
        """Write the data file while another writer claims a target.

        Args:
            *args: Positional `_write_sigmf_data_file` arguments.
            **kwargs: Keyword `_write_sigmf_data_file` arguments.

        Returns:
            Digest of the written data.
        """
        (tmp_path / planted).write_text("other writer")
        return write(*args, **kwargs)

    monkeypatch.setattr(
        sigmf_module, "_write_sigmf_data_file", write_and_plant
    )
    with pytest.raises(ValueError, match="already exist"):
        export_sigmf(store, "a", tmp_path / "out")
    assert (tmp_path / planted).read_text() == "other writer"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        planted,
        "store.zarr",
    ]


def test_archive_export_never_replaces_target_created_during_export(
    store: SigMFZarrStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Archive publication refuses a file created after the existence check.

    Args:
        store: Source store.
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    normalize = sigmf_module._reproducible_member
    target = tmp_path / "bundle.sigmf"

    def normalize_and_plant(member: tarfile.TarInfo) -> tarfile.TarInfo:
        """Normalize a member while another writer claims the target.

        Args:
            member: Archive member header.

        Returns:
            Normalized member header.
        """
        target.write_text("other writer")
        return normalize(member)

    monkeypatch.setattr(
        sigmf_module, "_reproducible_member", normalize_and_plant
    )
    with pytest.raises(ValueError, match="already exists"):
        export_sigmf_archive(store, target, recording_names=["a"])
    assert target.read_text() == "other writer"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "bundle.sigmf",
        "store.zarr",
    ]


@pytest.mark.parametrize(
    "target", ["", ".", "..", "out/..", "out/.", "out/", "./.", "out//"]
)
def test_export_rejects_targets_without_a_file_name(
    store: SigMFZarrStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    """An empty or dot target does not name an output file.

    Args:
        store: Source store.
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
        target: Target without a file name.
    """
    work = tmp_path / "work"
    (work / "out").mkdir(parents=True)
    monkeypatch.chdir(work)
    with pytest.raises(ValueError, match="must name a file"):
        export_sigmf(store, "a", target)
    with pytest.raises(ValueError, match="must name a file"):
        export_sigmf_archive(store, target)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "store.zarr",
        "work",
    ]
    assert [path.name for path in work.iterdir()] == ["out"]
    assert list((work / "out").iterdir()) == []


def test_archive_export_rejects_a_collection_with_repeated_recordings(
    store: SigMFZarrStore, tmp_path: Path
) -> None:
    """Collection recording identifiers must be unique.

    Args:
        store: Source store.
        tmp_path: Pytest temporary path fixture.
    """
    collection = store.collections.create("pair", recording_ids=["b", "a"])
    # The API rejects repeated identifiers, so write them as another tool
    # could.
    collection._raw_group.attrs["recording_ids"] = ["b", "a", "b"]
    with pytest.raises(ValueError, match="more than once"):
        export_sigmf_archive(store, tmp_path / "pair", collection_name="pair")
    assert not (tmp_path / "pair.sigmf").exists()


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"core:version": 7}, "7 is not of type 'string'"),
        ({"core:author": ["a"]}, "is not of type 'string'"),
    ],
)
def test_archive_export_checks_the_collection_schema(
    store: SigMFZarrStore,
    tmp_path: Path,
    metadata: dict[str, Any],
    message: str,
) -> None:
    """A collection that violates the SigMF schema is not exported.

    Args:
        store: Source store.
        tmp_path: Pytest temporary path fixture.
        metadata: Collection metadata.
        message: Expected error text.
    """
    store.collections.create("pair", recording_ids=["a"], metadata=metadata)
    with pytest.raises(ValueError, match=message):
        export_sigmf_archive(store, tmp_path / "pair", collection_name="pair")
    assert not (tmp_path / "pair.sigmf").exists()


def test_replacement_keeps_a_directory_created_during_export(
    store: SigMFZarrStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory that appears at a target after the check is restored.

    Args:
        store: Source store.
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    write = sigmf_module._write_sigmf_data_file
    planted = tmp_path / "out.sigmf-meta"

    def write_and_plant(*args: Any, **kwargs: Any) -> str:
        """Write the data file while another writer creates a directory.

        Args:
            *args: Positional `_write_sigmf_data_file` arguments.
            **kwargs: Keyword `_write_sigmf_data_file` arguments.

        Returns:
            Digest of the written data.
        """
        planted.mkdir()
        (planted / "precious.txt").write_text("other writer")
        return write(*args, **kwargs)

    monkeypatch.setattr(
        sigmf_module, "_write_sigmf_data_file", write_and_plant
    )
    with pytest.raises(ValueError, match="changed during export"):
        export_sigmf(store, "a", tmp_path / "out", overwrite=True)
    assert (planted / "precious.txt").read_text() == "other writer"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "out.sigmf-meta",
        "store.zarr",
    ]


def test_interrupted_replacement_restores_the_previous_pair(
    store: SigMFZarrStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupt during publication rolls back and then propagates.

    Args:
        store: Source store.
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    export_sigmf(store, "a", tmp_path / "out")
    before = {
        name: (tmp_path / name).read_bytes()
        for name in ("out.sigmf-meta", "out.sigmf-data")
    }
    replace = Path.replace

    def interrupt_install(self: Path, target: Path) -> Path:
        """Interrupt installation of the new data file.

        Args:
            self: Path being moved.
            target: Destination path.

        Returns:
            Destination path.

        Raises:
            KeyboardInterrupt: When the new data file is installed.
        """
        if Path(
            target
        ).name == "out.sigmf-data" and ".sigmf-zarr-export-" in str(self):
            raise KeyboardInterrupt
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", interrupt_install)
    with pytest.raises(KeyboardInterrupt):
        export_sigmf(store, "b", tmp_path / "out", overwrite=True)
    monkeypatch.undo()
    assert {name: (tmp_path / name).read_bytes() for name in before} == before
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "out.sigmf-data",
        "out.sigmf-meta",
        "store.zarr",
    ]


def test_interrupted_new_publication_leaves_no_data_file(
    store: SigMFZarrStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupt between the two new files removes the data file.

    Args:
        store: Source store.
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    link = os.link

    def interrupt_metadata(source: Any, target: Any, **kwargs: Any) -> None:
        """Interrupt publication of the metadata file.

        Args:
            source: Existing file.
            target: New link.
            **kwargs: Link options.

        Raises:
            KeyboardInterrupt: When the metadata file is published.
        """
        if str(target).endswith("out.sigmf-meta"):
            raise KeyboardInterrupt
        link(source, target, **kwargs)

    monkeypatch.setattr(os, "link", interrupt_metadata)
    with pytest.raises(KeyboardInterrupt):
        export_sigmf(store, "a", tmp_path / "out")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["store.zarr"]

"""Archive conversion preserves metadata and rejects omitted members."""

import hashlib
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest

from sigmf_zarr.sigmf import (
    _check_archive_members,
    export_sigmf_archive,
    import_sigmf_archive,
)


def _archive(tmp_path: Path, *, extra: str | None = None) -> Path:
    """Build a recording and collection without dependency normalization.

    Args:
        tmp_path: Temporary directory fixture.
        extra: Optional unsupported archive content.

    Returns:
        Source archive path.
    """
    root = tmp_path / "source"
    root.mkdir()
    meta = root / "rec.sigmf-meta"
    meta.write_text(
        json.dumps(
            {
                "global": {
                    "core:datatype": "rf32_le",
                    "core:version": "1.0.0",
                },
                "captures": [{"core:sample_start": 0}],
                "annotations": [],
            }
        )
    )
    meta.with_suffix(".sigmf-data").write_bytes(
        np.arange(4, dtype="<f4").tobytes()
    )
    stream = {
        "name": "rec",
        "hash": hashlib.sha512(meta.read_bytes()).hexdigest(),
    }
    if extra == "stream":
        stream["test:role"] = "reference"
    if extra == "stream-path":
        stream["name"] = "../../elsewhere/rec"
    document: dict[str, object] = {
        "collection": {
            "core:version": "1.0.0",
            "core:streams": [stream],
            "test:nested": {"receivers": [1, 2]},
        }
    }
    if extra == "top-level":
        document["x:extra"] = {"receivers": [1, 2]}
    collection = root / "paired.sigmf-collection"
    collection.write_text(json.dumps(document))
    if extra == "attachment":
        (root / "calibration.txt").write_text("calibration data")
    if extra == "collection":
        (root / "second.sigmf-collection").write_bytes(collection.read_bytes())
    archive = tmp_path / "source.sigmf"
    with tarfile.open(archive, "w") as output:
        output.add(root, arcname="source")
    return archive


def test_collection_preserves_version_and_extension_metadata(
    tmp_path: Path,
) -> None:
    """Retain collection provenance while regenerating metadata-file hashes.

    Args:
        tmp_path: Temporary directory fixture.
    """
    archive = _archive(tmp_path)
    store = import_sigmf_archive(tmp_path / "store.zarr", archive)
    collection = store.collections.open("paired")
    before = dict(collection.metadata)
    assert before["core:version"] == "1.0.0"
    output = export_sigmf_archive(
        store, tmp_path / "output.sigmf", collection_name="paired"
    )
    with tarfile.open(output) as result:
        collection_file = result.extractfile("output/paired.sigmf-collection")
        assert collection_file is not None
        document = json.load(collection_file)
        metadata_file = result.extractfile("output/rec.sigmf-meta")
        assert metadata_file is not None
        metadata = metadata_file.read()
    info = document["collection"]
    assert info["core:version"] == "1.0.0"
    assert info["test:nested"] == {"receivers": [1, 2]}
    assert info["core:streams"] == [
        {"name": "rec", "hash": hashlib.sha512(metadata).hexdigest()}
    ]
    assert collection.metadata == before


@pytest.mark.parametrize(
    "extra",
    ["attachment", "collection", "stream", "stream-path", "top-level"],
)
def test_archive_rejects_unrepresented_content(
    tmp_path: Path, extra: str
) -> None:
    """Reject content loss before creating the destination store.

    Args:
        tmp_path: Temporary directory fixture.
        extra: Unsupported archive content.
    """
    archive = _archive(tmp_path, extra=extra)
    destination = tmp_path / "store.zarr"
    with pytest.raises(
        ValueError,
        match=r"auxiliary|Multiple|core:streams|path components|top-level",
    ):
        import_sigmf_archive(destination, archive)
    assert not destination.exists()


@pytest.mark.parametrize("mode", ["w:gz", "w:bz2", "w:xz"])
def test_archive_import_rejects_compressed_archives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: Literal["w:gz", "w:bz2", "w:xz"],
) -> None:
    """A `.sigmf` archive must not expand through decompression.

    Args:
        tmp_path: Temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
        mode: Compressed tarfile write mode.
    """
    plain = _archive(tmp_path)
    archive = tmp_path / "compressed.sigmf"
    with (
        tarfile.open(plain) as source,
        tarfile.open(archive, mode) as output,
    ):
        for member in source.getmembers():
            output.addfile(member, source.extractfile(member))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    destination = tmp_path / "store.zarr"
    with pytest.raises(ValueError, match=r"compressed\. A \.sigmf archive"):
        import_sigmf_archive(destination, archive)
    assert not destination.exists()
    assert list(scratch.iterdir()) == []


def _write_member_archive(path: Path, kind: str) -> None:
    """Write an archive whose member must be rejected before extraction.

    Args:
        path: Archive path.
        kind: Kind of rejected member.
    """
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as output:
        if kind == "sparse":
            # A PAX 1.0 sparse file with one 4-byte region and a declared
            # size of 1 GiB.
            data = b"1\n0\n4\n".ljust(512, b"\0") + b"abcd"
            member = tarfile.TarInfo("source/GNUSparseFile.0/rec.sigmf-data")
            member.size = len(data)
            member.pax_headers = {
                "GNU.sparse.major": "1",
                "GNU.sparse.minor": "0",
                "GNU.sparse.name": "source/rec.sigmf-data",
                "GNU.sparse.realsize": str(1 << 30),
            }
            output.addfile(member, io.BytesIO(data))
            return
        member = tarfile.TarInfo("source/rec.sigmf-data")
        if kind == "absolute":
            member.name = "/source/rec.sigmf-data"
        elif kind == "hard-link":
            member.type = tarfile.LNKTYPE
            member.linkname = "source/other"
        elif kind == "fifo":
            member.type = tarfile.FIFOTYPE
        elif kind == "truncated":
            member.size = 1 << 20
        if member.size:
            # tarfile requires the declared bytes, so remove them afterward.
            output.addfile(member, io.BytesIO(bytes(member.size)))
        else:
            output.addfile(member)
    if kind == "truncated":
        with path.open("r+b") as handle:
            handle.truncate(4096)


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("sparse", "is a sparse file"),
        ("absolute", "outside the archive root"),
        ("hard-link", "not a regular file or directory"),
        ("fifo", "not a regular file or directory"),
        ("truncated", "unexpected end of data"),
    ],
)
def test_archive_import_checks_members_before_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, message: str
) -> None:
    """Reject unsafe members without relying on the tarfile data filter.

    Args:
        tmp_path: Temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
        kind: Kind of rejected member.
        message: Expected error text.
    """
    archive = tmp_path / "source.sigmf"
    _write_member_archive(archive, kind)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    destination = tmp_path / "store.zarr"
    with pytest.raises(ValueError, match=message):
        import_sigmf_archive(destination, archive)
    assert not destination.exists()
    assert list(scratch.iterdir()) == []


def test_archive_export_is_reproducible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exports at different times and by different accounts are identical.

    Args:
        tmp_path: Temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    store = import_sigmf_archive(tmp_path / "store.zarr", _archive(tmp_path))
    first = export_sigmf_archive(
        store, tmp_path / "first" / "paired", collection_name="paired"
    )
    gettarinfo = tarfile.TarFile.gettarinfo

    def later_and_elsewhere(
        self: tarfile.TarFile, *args: Any, **kwargs: Any
    ) -> tarfile.TarInfo:
        """Describe a file as written a day later by another account.

        Args:
            self: Archive being written.
            *args: Positional `gettarinfo` arguments.
            **kwargs: Keyword `gettarinfo` arguments.

        Returns:
            Member header with a later time and a different owner.
        """
        member = gettarinfo(self, *args, **kwargs)
        member.mtime += 86400
        member.uid += 1000
        member.gid += 1000
        member.uname = "someone"
        member.gname = "else"
        return member

    monkeypatch.setattr(tarfile.TarFile, "gettarinfo", later_and_elsewhere)
    # A different umask changes the modes of the files that export writes.
    previous = os.umask(0o077)
    try:
        second = export_sigmf_archive(
            store, tmp_path / "second" / "paired", collection_name="paired"
        )
    finally:
        os.umask(previous)
    assert second.read_bytes() == first.read_bytes()
    with tarfile.open(first) as archive:
        members = archive.getmembers()
    assert [(member.name, member.mode) for member in members] == [
        ("paired", 0o755),
        ("paired/paired.sigmf-collection", 0o644),
        ("paired/rec.sigmf-data", 0o644),
        ("paired/rec.sigmf-meta", 0o644),
    ]
    for member in members:
        assert (member.uid, member.gid, member.mtime) == (0, 0, 0)
        assert (member.uname, member.gname) == ("", "")


def test_archive_member_sizes_must_fit_in_the_archive() -> None:
    """Declared member sizes bound extraction by the archive size."""
    members = [tarfile.TarInfo("source"), tarfile.TarInfo("source/a")]
    members[0].type = tarfile.DIRTYPE
    members[1].size = 4096
    _check_archive_members(members, 4096)
    with pytest.raises(ValueError, match="more than the 4095-byte archive"):
        _check_archive_members(members, 4095)

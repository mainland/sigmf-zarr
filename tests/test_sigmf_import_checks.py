"""SigMF import rejects input that it cannot store and export again."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import sigmf_zarr.cli.sigmf as sigmf_cli
from sigmf_zarr import SigMFZarrStore, import_sigmf, import_sigmf_archive
from sigmf_zarr.sigmf import _check_archive_members, _read_archive_members

# The JSON decoder raises `RecursionError` at this depth on some interpreters.
# Others decode it and reach the library's own depth check.
DEEP_MESSAGE = "nests too deeply|nested at most 128 levels deep"

SAMPLES = np.arange(8, dtype="<f4")
"""Samples of every test recording."""


def _metadata(**global_fields: Any) -> dict[str, Any]:
    """Return metadata for an eight-sample `rf32_le` recording.

    Args:
        **global_fields: Extra global fields.

    Returns:
        Standard SigMF metadata.
    """
    return {
        "global": {
            "core:datatype": "rf32_le",
            "core:version": "1.2.0",
            **global_fields,
        },
        "captures": [{"core:sample_start": 0}],
        "annotations": [],
    }


def _write_pair(directory: Path, metadata: dict[str, Any]) -> Path:
    """Write a recording pair named `rec`.

    Args:
        directory: Existing directory.
        metadata: Standard SigMF metadata.

    Returns:
        Metadata file path.
    """
    meta = directory / "rec.sigmf-meta"
    meta.write_text(json.dumps(metadata))
    (directory / "rec.sigmf-data").write_bytes(SAMPLES.tobytes())
    return meta


def _member(
    output: tarfile.TarFile,
    name: str,
    data: bytes = b"",
    *,
    directory: bool = False,
) -> None:
    """Add one regular file or directory member.

    Args:
        output: Archive being written.
        name: Member name, written as given.
        data: File contents.
        directory: Whether to add a directory member.
    """
    member = tarfile.TarInfo(name)
    if directory:
        member.type = tarfile.DIRTYPE
        member.mode = 0o755
        output.addfile(member)
        return
    member.size = len(data)
    output.addfile(member, io.BytesIO(data))


def _recording_archive(
    path: Path,
    *,
    collection: object = None,
    extra: tuple[tuple[str, bytes | None], ...] = (),
) -> Path:
    """Write an archive with recording `a/rec` and optional extra members.

    Args:
        path: Archive path.
        collection: Optional `.sigmf-collection` document or raw text.
        extra: Additional members as names and contents. `None` contents
            add a directory member.

    Returns:
        Archive path.
    """
    meta = json.dumps(_metadata()).encode()
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as output:
        _member(output, "a", directory=True)
        _member(output, "a/rec.sigmf-meta", meta)
        _member(output, "a/rec.sigmf-data", SAMPLES.tobytes())
        if collection is not None:
            text = (
                collection
                if isinstance(collection, str)
                else json.dumps(collection)
            )
            _member(output, "a/c.sigmf-collection", text.encode())
        for name, data in extra:
            _member(output, name, data or b"", directory=data is None)
    return path


def _collection(**fields: Any) -> dict[str, Any]:
    """Return a collection document that references recording `rec`.

    Args:
        **fields: Collection fields that replace the defaults.

    Returns:
        Collection document.
    """
    digest = hashlib.sha512(json.dumps(_metadata()).encode()).hexdigest()
    collection = {
        "core:version": "1.2.0",
        "core:streams": [{"name": "rec", "hash": digest}],
        **fields,
    }
    return {"collection": collection}


class _LoopingArchive:
    """Archive reader whose headers stop advancing, as in CVE-2025-8194."""

    def __init__(self, offsets: Iterator[int]) -> None:
        """Initialize the reader.

        Args:
            offsets: Header offsets to report.
        """
        self._offsets = offsets

    def next(self) -> tarfile.TarInfo | None:
        """Return the next member header.

        Returns:
            Member at the next reported offset, or `None` at the end.
        """
        offset = next(self._offsets, None)
        if offset is None:
            return None
        member = tarfile.TarInfo("loop")
        member.offset = offset
        return member


def test_archive_headers_must_advance() -> None:
    """A header that points back at itself must stop parsing."""

    def offsets() -> Iterator[int]:
        """Yield one header offset and then repeat it without end.

        Yields:
            Header offsets.
        """
        yield 0
        while True:
            yield 512

    archive = _LoopingArchive(offsets())
    with pytest.raises(ValueError, match="does not follow"):
        _read_archive_members(archive, 10240)  # type: ignore[arg-type]


def test_archive_member_count_is_bounded_by_its_size() -> None:
    """An archive cannot hold more headers than 512-byte blocks."""
    archive = _LoopingArchive(iter(range(0, 10**9, 512)))
    with pytest.raises(ValueError, match="more members"):
        _read_archive_members(archive, 4096)  # type: ignore[arg-type]


def test_implied_directories_count_against_the_archive_size() -> None:
    """Parents that extraction creates count as extracted entries."""
    member = tarfile.TarInfo("/".join(["d"] * 50) + "/x")
    with pytest.raises(ValueError, match="more files and directories"):
        _check_archive_members([member], 4096)
    _check_archive_members([member], 51 * 512)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ((("", b"x"),), "archive root as a file"),
        (((".", b"x"),), "archive root as a file"),
        ((("./a/rec.sigmf-data", b"0000"),), "more than one member"),
        ((("a/q.sigmf-meta", None),), "SigMF file suffixes"),
        ((("a/rec.sigmf-data", None),), "both files and directories"),
        ((("a/rec.sigmf-data/x", b"x"),), "both files and directories"),
        (((("d" * 300) + "/x", b"x"),), "File name too long"),
    ],
)
def test_archive_member_names_raise_value_error(
    tmp_path: Path, extra: tuple[tuple[str, bytes | None], ...], message: str
) -> None:
    """Unusable member names are invalid input, not file-system errors.

    Args:
        tmp_path: Pytest temporary path fixture.
        extra: Additional archive members.
        message: Expected error text.
    """
    archive = _recording_archive(tmp_path / "in.sigmf", extra=extra)
    with pytest.raises(ValueError, match=message) as error:
        import_sigmf_archive(tmp_path / "out.zarr", archive)
    assert not isinstance(error.value, OSError)
    assert not (tmp_path / "out.zarr").exists()


@pytest.mark.parametrize(
    ("collection", "message"),
    [
        ({"collection": []}, "to be a JSON object"),
        (_collection(**{"core:version": 7}), "collection schema"),
        (
            {
                "collection": {
                    "core:streams": _collection()["collection"]["core:streams"]
                }
            },
            "'core:version' is a required property",
        ),
        (
            _collection(
                **{
                    "core:streams": 2
                    * _collection()["collection"]["core:streams"]
                }
            ),
            "more than once",
        ),
        (
            '{"collection": {"core:version": "1.2.0", "x:deep": '
            + "[" * 100000
            + "]" * 100000
            + "}}",
            DEEP_MESSAGE,
        ),
    ],
)
def test_archive_collection_is_checked(
    tmp_path: Path, collection: object, message: str
) -> None:
    """The collection must satisfy the schema and name each recording once.

    Args:
        tmp_path: Pytest temporary path fixture.
        collection: Collection document or raw text.
        message: Expected error text.
    """
    archive = _recording_archive(tmp_path / "in.sigmf", collection=collection)
    with pytest.raises(ValueError, match=message):
        import_sigmf_archive(tmp_path / "out.zarr", archive)
    assert not (tmp_path / "out.zarr").exists()


def test_archive_import_keeps_an_existing_collection(tmp_path: Path) -> None:
    """Replacing a collection requires overwrite_collection.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    archive = _recording_archive(
        tmp_path / "in.sigmf", collection=_collection()
    )
    store_path = tmp_path / "out.zarr"
    store = SigMFZarrStore.create(store_path)
    store.recordings.create("old", sample_shape=(2,))
    store.collections.create(
        "c", recording_ids=("old",), metadata={"x:note": "curated by hand"}
    )
    with pytest.raises(ValueError, match="overwrite_collection"):
        import_sigmf_archive(store_path, archive)
    reopened = SigMFZarrStore.open(store_path)
    assert reopened.list_recordings() == ("old",)
    kept = reopened.collections.open("c")
    assert kept.recording_ids == ("old",)
    assert kept.metadata == {"x:note": "curated by hand"}

    replaced = import_sigmf_archive(
        store_path, archive, overwrite_collection=True
    )
    assert replaced.collections.open("c").recording_ids == ("rec",)


def test_cli_passes_overwrite_collection(tmp_path: Path) -> None:
    """The command line exposes collection replacement for archives.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    archive = _recording_archive(
        tmp_path / "in.sigmf", collection=_collection()
    )
    store_path = tmp_path / "out.zarr"
    store = SigMFZarrStore.create(store_path)
    store.collections.create("c", metadata={"x:note": "old"})
    arguments = ["import", "sigmf", str(archive), str(store_path)]
    with pytest.raises(SystemExit) as exit_info:
        sigmf_cli.SigMFCommand().run(arguments)
    assert exit_info.value.code == 2
    assert SigMFZarrStore.open(store_path).list_recordings() == ()
    command = sigmf_cli.SigMFCommand()
    assert command.run([*arguments, "--overwrite-collection"]) == 0
    assert SigMFZarrStore.open(store_path).collections.open(
        "c"
    ).recording_ids == ("rec",)


def test_cli_rejects_overwrite_collection_for_one_recording(
    tmp_path: Path,
) -> None:
    """Collection replacement does not apply to a single recording.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    meta = _write_pair(tmp_path, _metadata())
    with pytest.raises(SystemExit) as exit_info:
        sigmf_cli.SigMFCommand().run(
            [
                "import",
                "sigmf",
                str(meta),
                str(tmp_path / "out.zarr"),
                "--overwrite-collection",
            ]
        )
    assert exit_info.value.code == 2
    assert not (tmp_path / "out.zarr").exists()


def test_deeply_nested_metadata_raises_value_error(tmp_path: Path) -> None:
    """JSON nesting beyond the parser's limit is invalid input.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    meta = _write_pair(tmp_path, _metadata())
    meta.write_text(
        '{"global": {"core:datatype": "rf32_le", "x:deep": '
        + "[" * 100000
        + "]" * 100000
        + '}, "captures": [], "annotations": []}'
    )
    with pytest.raises(ValueError, match=DEEP_MESSAGE):
        import_sigmf(tmp_path / "out.zarr", meta)


def test_metadata_recursion_error_raises_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A decoder that exhausts the stack is reported as invalid input.

    Interpreters differ in the depth at which the JSON decoder raises
    `RecursionError`, so this test forces the error.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    meta = _write_pair(tmp_path, _metadata())

    def exhaust_stack(*args: object, **kwargs: object) -> object:
        raise RecursionError

    monkeypatch.setattr(json, "load", exhaust_stack)
    with pytest.raises(ValueError, match="nests too deeply"):
        import_sigmf(tmp_path / "out.zarr", meta)


@pytest.mark.parametrize(
    ("section", "entry", "global_fields", "message"),
    [
        ("captures", {"core:sample_start": 0.0}, {}, "must be an integer"),
        (
            "annotations",
            {"core:sample_start": 0, "core:sample_count": 2.0},
            {},
            "must be an integer",
        ),
        (
            "captures",
            {"core:sample_start": 0},
            {"core:offset": 0.0},
            "integer",
        ),
        (
            "annotations",
            {"core:sample_start": 4, "core:sample_count": 5},
            {},
            "beyond sample end 8",
        ),
        (
            "captures",
            {"core:sample_start": 9},
            {},
            "outside sample range 0..8",
        ),
        (
            "captures",
            {"core:sample_start": 0, "core:sample_count": 3},
            {},
            "describes 3 samples",
        ),
    ],
)
def test_import_applies_the_checks_of_default_export(
    tmp_path: Path,
    section: str,
    entry: dict[str, Any],
    global_fields: dict[str, Any],
    message: str,
) -> None:
    """Import rejects metadata that default export would reject.

    Args:
        tmp_path: Pytest temporary path fixture.
        section: Metadata section of `entry`.
        entry: Capture or annotation entry.
        global_fields: Extra global fields.
        message: Expected error text.
    """
    metadata = _metadata(**global_fields)
    metadata[section] = [entry]
    if section == "annotations":
        metadata["captures"] = [{"core:sample_start": 0}]
    meta = _write_pair(tmp_path, metadata)
    with pytest.raises(ValueError, match=message):
        import_sigmf(tmp_path / "out.zarr", meta)
    assert not (tmp_path / "out.zarr").exists()


def test_import_rejects_a_dataset_file_that_is_a_link(tmp_path: Path) -> None:
    """A `core:dataset` file must be a regular file, not a link.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "secret").write_bytes(SAMPLES.tobytes())
    source = tmp_path / "source"
    source.mkdir()
    (source / "samples.bin").symlink_to(elsewhere / "secret")
    meta = source / "rec.sigmf-meta"
    meta.write_text(json.dumps(_metadata(**{"core:dataset": "samples.bin"})))
    with pytest.raises(ValueError, match="not a regular file"):
        import_sigmf(tmp_path / "out.zarr", meta)

    (source / "samples.bin").unlink()
    (source / "samples.bin").mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        import_sigmf(tmp_path / "out.zarr", meta)
    assert not (tmp_path / "out.zarr").exists()


def test_import_follows_a_link_to_the_conforming_dataset(
    tmp_path: Path,
) -> None:
    """The caller names the conforming pair, so its links are followed.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    (tmp_path / "real.bin").write_bytes(SAMPLES.tobytes())
    meta = tmp_path / "rec.sigmf-meta"
    meta.write_text(json.dumps(_metadata()))
    (tmp_path / "rec.sigmf-data").symlink_to(tmp_path / "real.bin")
    recording = import_sigmf(tmp_path / "out.zarr", meta)
    np.testing.assert_array_equal(recording.samples[:], SAMPLES)

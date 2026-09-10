"""Tests for complete-store validation and integrity CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import zarr

from sigmf_zarr.cli.sigmf import SigMFCommand
from sigmf_zarr.integrity import INTEGRITY_ATTR
from sigmf_zarr.json import json_object, json_object_list, json_value
from sigmf_zarr.store import SigMFRecording, SigMFZarrStore
from sigmf_zarr.store._common import ZarrFormat
from sigmf_zarr.validation import (
    ValidationIssue,
    ValidationReport,
    validate_store,
)


def _validation_store(tmp_path: Path) -> tuple[object, SigMFZarrStore]:
    """Create a small conforming store for validation tests.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Store path and opened writable store.
    """
    store_path = tmp_path / "validation.zarr"
    store = SigMFZarrStore.create(store_path, overwrite=True)
    # One chunk holds the whole sample array, so tests of the read limit
    # see only the logical array size.
    recording = store.recordings.create(
        "rec",
        sample_shape=(4,),
        sample_axes=("time",),
        global_metadata={"core:datatype": "rf32_le"},
        sample_chunks=(4,),
    )
    recording.set_samples(np.arange(4, dtype=np.float32))
    recording.add_index(
        "quality",
        np.arange(4, dtype=np.int8),
        axis="time",
        field="test:quality",
    )
    store.update_integrity()
    return store_path, store


def test_validate_store_accepts_complete_hashed_store(
    tmp_path: Path,
) -> None:
    """A conforming store with all hashes should pass strict validation.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    _, store = _validation_store(tmp_path)

    report = validate_store(store, require_integrity=True)

    assert report.valid is True
    assert report.recordings_checked == 1
    assert report.indexes_checked == 1
    assert report.issues == ()


def test_validate_store_reports_dtype_and_index_corruption(
    tmp_path: Path,
) -> None:
    """Validation should aggregate resource paths for malformed content.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    _, store = _validation_store(tmp_path)
    recording = store.recordings.open("rec")
    metadata = dict(recording.global_metadata)
    metadata["sigmf-zarr:dtype"] = "<f8"
    recording._raw_group.attrs["global"] = metadata
    recording._indexes_group["quality"].attrs["axis"] = "missing"

    report = validate_store(store)

    assert report.valid is False
    assert any(issue.path == "recordings/rec" for issue in report.issues)
    assert "dtype" in report.format()


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_validation_rejects_big_endian_sample_dtype(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    """A big-endian `sigmf-zarr:dtype` written by another tool is invalid.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, zarr_format=zarr_format)
    store.recordings.create("rec", sample_shape=(4,), sample_checksum=None)
    # Another tool writes a self-consistent big-endian declaration and, in
    # Zarr format 2, a big-endian array.
    root = zarr.open_group(store_path, mode="r+")
    group = cast(zarr.Group, root["recordings/rec"])
    if zarr_format == 2:
        del group["samples"]
        group.create_array("samples", shape=(4,), dtype=">f4")
    metadata = json_object(group.attrs["global"], name="global")
    metadata["sigmf-zarr:dtype"] = ">f4"
    group.attrs["global"] = metadata

    report = validate_store(store_path, verify_integrity=False)

    assert [issue.path for issue in report.issues] == ["recordings/rec"]
    assert "little-endian" in report.issues[0].message


def test_validation_and_integrity_commands_support_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI commands should validate and refresh hashes for scripts.

    Args:
        tmp_path: Pytest temporary path fixture.
        capsys: Pytest output capture fixture.
    """
    store_path, store = _validation_store(tmp_path)
    recording = store.recordings.open("rec")
    recording.set_global_field("core:description", "changed")

    status = SigMFCommand().run(
        [
            "store",
            str(store_path),
            "integrity",
            "update",
            "--format",
            "json",
        ]
    )
    update_result = json.loads(capsys.readouterr().out)

    assert status == 0
    assert update_result["valid"] is True

    status = SigMFCommand().run(
        ["store", str(store_path), "validate", "--format", "json"]
    )
    validate_result = json.loads(capsys.readouterr().out)

    assert status == 0
    assert validate_result["valid"] is True


def test_validation_reports_missing_stale_and_unsupported_integrity(
    tmp_path: Path,
) -> None:
    """Integrity policy failures should produce stable resource issues.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    _, store = _validation_store(tmp_path)
    recording = store.recordings.open("rec")
    recording._samples_array[0] = np.float32(99.0)

    stale = validate_store(store, require_integrity=True)

    assert stale.valid is False
    assert any(
        issue.path == "recordings/rec" and "do not match" in issue.message
        for issue in stale.issues
    )
    assert stale.as_dict()["valid"] is False

    integrity = dict(recording.integrity)
    integrity["version"] = 999
    recording._raw_group.attrs[INTEGRITY_ATTR] = integrity

    unsupported = validate_store(store)

    assert any(
        issue.path == "recordings/rec" and "unsupported" in issue.message
        for issue in unsupported.issues
    )

    recording._raw_group.attrs.pop(INTEGRITY_ATTR)
    missing = validate_store(store, require_integrity=True)

    assert any(
        issue.path == "recordings/rec" and "required" in issue.message
        for issue in missing.issues
    )


def test_validation_reports_missing_collection_recordings(
    tmp_path: Path,
) -> None:
    """Collection references and invalid store indexes should be reported.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    _, store = _validation_store(tmp_path)
    collection = store.collections.create("broken")
    # The managed API rejects dangling references. Write one directly, as an
    # external writer could.
    with collection.mutate() as group:
        group.attrs["recording_ids"] = ["missing"]
    store.add_index("fold", np.array([0], dtype=np.int8))
    try:
        with store.mutate_index("fold") as index:
            index[0] = np.int8(1)
            raise RuntimeError("stop")
    except RuntimeError:
        pass

    report = validate_store(store, verify_integrity=False)

    assert any(
        issue.path == "collections/broken"
        and "missing recordings" in issue.message
        for issue in report.issues
    )
    assert any(issue.path == "indexes/fold" for issue in report.issues)


def _corrupt_first_sample_chunk(store_path: Path) -> None:
    """Flip one byte in the first stored sample chunk.

    Args:
        store_path: Local store containing recording `rec`.
    """
    samples = store_path / "recordings" / "rec" / "samples"
    chunk = next(
        path
        for path in sorted(samples.rglob("*"))
        if path.is_file()
        and path.name not in {".zarray", ".zattrs", "zarr.json"}
    )
    data = bytearray(chunk.read_bytes())
    data[len(data) // 2] ^= 0xFF
    chunk.write_bytes(bytes(data))


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_validation_reports_corrupted_sample_chunks(
    tmp_path: Path, zarr_format: int
) -> None:
    """A checksum failure in either format must be reported, not raised.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(
        store_path, zarr_format=cast(ZarrFormat, zarr_format)
    )
    recording = store.recordings.create(
        "rec", sample_shape=(2, 4096), sample_chunks=(2, 1024)
    )
    recording.set_samples(np.ones((2, 4096), dtype=np.float32))
    store.update_integrity()
    _corrupt_first_sample_chunk(store_path)

    report = validate_store(store_path)

    assert not report.valid
    assert any(
        issue.path == "recordings/rec" and "checksum" in issue.message
        for issue in report.issues
    )


def test_json_helpers_reject_non_json_metadata() -> None:
    """Metadata validation should reject malformed and non-finite values."""
    for value in (float("nan"), {1: "value"}, object()):
        try:
            json_value(value, name="test value")
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid JSON value {value!r}")

    for helper, value in ((json_object, []), (json_object_list, {})):
        try:
            helper(value, name="test value")
        except ValueError:
            pass
        else:
            raise AssertionError("Expected invalid JSON container")


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_nested_store_index_failure_cannot_be_rehashed(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    """Reject incomplete nested indexes until a writer repairs them.

    Args:
        tmp_path: Pytest temporary directory.
        zarr_format: Physical Zarr format to exercise.
    """
    store = SigMFZarrStore.create(
        tmp_path / "store.zarr", zarr_format=zarr_format
    )
    store.add_index("nested/fold", [1, 2])

    def interrupt() -> None:
        """Write one value, then abandon the index mutation."""
        with store.mutate_index("nested/fold") as index:
            index[0] = 3
            raise RuntimeError("incomplete")

    with pytest.raises(RuntimeError, match="incomplete"):
        interrupt()

    report = validate_store(store)
    assert not report.valid
    assert report.indexes_checked == 1
    assert report.issues[0].path == "indexes/nested/fold"
    with pytest.raises(ValueError, match=r"nested/fold.*invalid"):
        store.update_integrity()
    assert not store.verify_integrity()

    with store.mutate_index("nested/fold") as index:
        index[:] = [3, 4]
    store.update_integrity()
    assert store.verify_integrity()
    assert validate_store(store, require_integrity=True).valid


@pytest.mark.parametrize("field", ["metadata", "recording_ids"])
def test_collection_requires_declared_attributes(
    tmp_path: Path, field: str
) -> None:
    """Report absent collection fields instead of substituting defaults.

    Args:
        tmp_path: Pytest temporary directory.
        field: Required collection attribute to remove.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr")
    collection = store.collections.create("all")
    # A failed managed mutation restores the field, so remove it the way an
    # external writer would.
    del collection._raw_group.attrs[field]
    report = validate_store(store)
    assert not report.valid
    assert report.collections_checked == 1
    assert report.issues[0].path == "collections/all"
    assert field in report.issues[0].message


def _damage_metadata_document(group_path: Path, damage: str) -> None:
    """Make one group's attribute document unreadable.

    Args:
        group_path: Local Zarr group directory.
        damage: Kind of damage to apply.
    """
    v3_document = group_path / "zarr.json"
    document = v3_document if v3_document.exists() else group_path / ".zattrs"
    text = document.read_text()
    if damage == "truncated":
        document.write_text(text[: len(text) // 2])
        return
    value = "9" * 5000 if damage == "long-integer" else "[" * 100_000
    if damage == "deep":
        value += "]" * 100_000
    decoded = json.loads(text)
    decoded.pop("attributes" if document is v3_document else "x:bad", None)
    encoded = json.dumps(decoded)
    document.write_text(encoded[:-1] + f', "x:bad": {value}}}')


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("damage", ["truncated", "long-integer", "deep"])
@pytest.mark.parametrize("kind", ["recordings", "collections"])
def test_validation_reports_unreadable_child_documents(
    tmp_path: Path, zarr_format: int, damage: str, kind: str
) -> None:
    """One unreadable child document must not hide the other resources.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
        damage: Kind of damage to apply.
        kind: Resource container holding the damaged child.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(
        store_path, zarr_format=cast(ZarrFormat, zarr_format)
    )
    for name in ("bad", "good"):
        store.recordings.create(name, sample_shape=(4,))
    for name in ("bad", "good"):
        store.collections.create(name, recording_ids=["good"])
    _damage_metadata_document(store_path / kind / "bad", damage)

    report = validate_store(store_path)

    assert [issue.path for issue in report.issues] == [f"{kind}/bad"]
    assert report.recordings_checked == 2
    assert report.collections_checked == 2


@pytest.mark.parametrize(
    ("zarr_format", "name"),
    [
        (2, ".zarray"),
        (2, ".zmetadata"),
        (2, "zarr.json"),
        (3, ".zarray"),
        (3, ".zattrs"),
        (3, ".zgroup"),
        (3, ".zmetadata"),
    ],
)
@pytest.mark.parametrize("kind", ["recordings", "collections"])
def test_validation_reports_reserved_child_names(
    tmp_path: Path, zarr_format: int, name: str, kind: str
) -> None:
    """A child named like a Zarr metadata document is reported.

    The API rejects these names, so another tool must write such a child.
    Zarr format 2 omits it from group listings.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
        name: Reserved name that the format can hold as a child group.
        kind: Resource container holding the child.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(
        store_path, zarr_format=cast(ZarrFormat, zarr_format)
    )
    for recording_name in ("other", "good"):
        store.recordings.create(recording_name, sample_shape=(4,))
    for collection_name in ("other", "good"):
        store.collections.create(collection_name, recording_ids=["good"])
    (store_path / kind / "other").rename(store_path / kind / name)

    report = validate_store(store_path)

    assert [(issue.path, issue.message) for issue in report.issues] == [
        (f"{kind}/{name}", "name is reserved for Zarr metadata documents")
    ]
    assert report.recordings_checked == 2
    assert report.collections_checked == 2


def test_validation_verifies_each_declared_hash(tmp_path: Path) -> None:
    """A metadata edit leaves the current sample hash verifiable.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    _, store = _validation_store(tmp_path)
    recording = store.recordings.open("rec")
    recording.set_global_field("example:name", "edited")
    assert set(recording.integrity) == {
        "algorithm",
        "version",
        "sample_sha512",
    }

    assert validate_store(store).valid is True
    strict = validate_store(store, require_integrity=True)
    assert {issue.as_dict()["message"] for issue in strict.issues} == {
        "integrity metadata is required",
        "integrity hashes are incomplete",
    }

    recording._samples_array[0] = np.float32(99.0)
    stale = validate_store(store)
    assert [issue.as_dict() for issue in stale.issues] == [
        {
            "path": "recordings/rec",
            "message": "integrity hashes do not match current content",
        }
    ]


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_validation_reports_non_finite_channel_and_index_attributes(
    tmp_path: Path, zarr_format: int
) -> None:
    """Attributes that hashing cannot encode must be reported.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(
        store_path, zarr_format=cast(ZarrFormat, zarr_format)
    )
    recording = store.recordings.create(
        "rec", sample_shape=(2, 4), sample_axes=("channel", "time")
    )
    recording.add_index(
        "quality", np.zeros(4, dtype=np.int8), axis="time", field="x:q"
    )
    store.add_index("fold", np.zeros(1, dtype=np.int8))
    # Another tool can write non-finite attributes directly.
    root = zarr.open_group(store_path, mode="r+")
    root["recordings/rec/channels/1"].attrs["x:gain"] = float("nan")
    root["recordings/rec/indexes/quality"].attrs["x:limit"] = float("inf")
    root["indexes/fold"].attrs["x:limit"] = float("-inf")

    report = validate_store(store_path, verify_integrity=False)

    assert [issue.path for issue in report.issues] == [
        "indexes/fold",
        "recordings/rec/channels/1",
        "recordings/rec/indexes/quality",
    ]
    assert all("finite" in issue.message for issue in report.issues)


def test_validation_reports_invalid_item_capture_starts(
    tmp_path: Path,
) -> None:
    """Capture starts written by another tool must be checked.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path)
    recording = store.recordings.create("rec", batched=True, sample_shape=(4,))
    recording.append_samples(
        np.zeros((2, 4), dtype=np.float32), item_metadata=[None, None]
    )
    item_metadata = zarr.open_array(
        store_path, path="recordings/rec/item_metadata", mode="r+"
    )
    item_metadata[1] = json.dumps({"captures": [{"core:sample_start": -5}]})

    report = validate_store(store_path, verify_integrity=False)

    assert [issue.path for issue in report.issues] == ["recordings/rec"]
    assert "core:sample_start" in report.issues[0].message


def test_report_format_escapes_control_characters() -> None:
    """Formatted issue paths and messages contain no control characters."""
    report = ValidationReport(
        issues=(ValidationIssue("recordings/a\x1b[2K", "bad\r\x07\u202e"),),
        recordings_checked=1,
        collections_checked=0,
        indexes_checked=0,
    )
    lines = report.format().split("\n")
    assert lines[1] == "recordings/a\\x1b[2K: bad\\r\\x07\\u202e"


def _edit_array_document(
    array_path: Path, zarr_format: ZarrFormat, **changes: object
) -> None:
    """Edit the metadata document of one array as another tool would.

    Args:
        array_path: Local directory of the array.
        zarr_format: Physical Zarr format of the array.
        **changes: Top-level metadata fields to replace.
    """
    document = array_path / ("zarr.json" if zarr_format == 3 else ".zarray")
    metadata = json.loads(document.read_text())
    metadata.update(changes)
    document.write_text(json.dumps(metadata))


# Zarr format 3 cannot represent a null string fill value.
@pytest.mark.parametrize(
    ("zarr_format", "fill_value"), [(2, "{}"), (3, "{}"), (2, None), (3, "x")]
)
def test_item_metadata_fill_value_must_be_empty(
    tmp_path: Path, zarr_format: ZarrFormat, fill_value: str | None
) -> None:
    """A declared item axis cannot make missing chunks read as entries.

    With the fill value `{}`, every missing chunk would decode as valid empty
    objects, so a small store declaring 2**40 items would take hours to open.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
        fill_value: Fill value written by another tool.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, zarr_format=zarr_format)
    recording = store.recordings.create(
        "rec", batched=True, sample_shape=(4,), sample_chunks=(1, 4)
    )
    recording.append_samples(
        np.zeros((1, 4), dtype=np.float32), item_metadata=[None]
    )
    recording_path = store_path / "recordings" / "rec"
    for name, changes in (
        ("samples", {"shape": [2**40, 4]}),
        ("item_metadata", {"shape": [2**40], "fill_value": fill_value}),
    ):
        _edit_array_document(recording_path / name, zarr_format, **changes)

    reopened = SigMFZarrStore.open(store_path)
    for validation in ("full", "structural"):
        with pytest.raises(ValueError, match="fill value must be the empty"):
            reopened.recordings.open("rec", validation=cast(Any, validation))
    report = validate_store(store_path)
    assert [issue.path for issue in report.issues] == ["recordings/rec"]
    assert "fill value" in report.issues[0].message


def test_writer_item_metadata_fill_value_is_empty(tmp_path: Path) -> None:
    """Writers produce the fill value that readers require.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    zarr_format: ZarrFormat
    for zarr_format in (2, 3):
        store = SigMFZarrStore.create(
            tmp_path / f"v{zarr_format}.zarr", zarr_format=zarr_format
        )
        recording = store.recordings.create(
            "rec", batched=True, sample_shape=(4,)
        )
        recording.append_samples(
            np.zeros((2, 4), dtype=np.float32), item_metadata=[None, None]
        )
        assert recording.item_metadata_array.fill_value == ""


def test_validation_reports_memory_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running out of memory while hashing is an issue, not a crash.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    _, store = _validation_store(tmp_path)

    def exhaust(self: SigMFRecording) -> str:
        """Fail as a huge declared array would.

        Args:
            self: Recording being hashed.

        Raises:
            MemoryError: Always.
        """
        del self
        raise MemoryError("Unable to allocate 7.28 TiB")

    monkeypatch.setattr(SigMFRecording, "calculate_metadata_sha512", exhaust)

    report = validate_store(store)

    assert [(issue.path, issue.message) for issue in report.issues] == [
        ("recordings/rec", "out of memory: Unable to allocate 7.28 TiB")
    ]


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_validation_limits_hashed_array_size(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    """Arrays above the size limit are reported instead of hashed.

    An extension array declaring 10**12 elements without stored chunks would
    otherwise be hashed as fill values for hours.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, zarr_format=zarr_format)
    recording = store.recordings.create(
        "rec",
        sample_shape=(4,),
        global_metadata={"core:datatype": "rf32_le"},
        sample_chunks=(4,),
    )
    recording.add_extension_array("ext", np.arange(4))
    store.add_index("names", np.array(["a", "b"], dtype=object))
    store.update_integrity()
    _edit_array_document(
        store_path / "recordings" / "rec" / "extensions" / "ext",
        zarr_format,
        shape=[10**12],
    )

    report = validate_store(store_path, max_array_bytes=24)

    assert [(issue.path, issue.message) for issue in report.issues] == [
        (
            "indexes/names",
            "integrity not verified: the array declares 32 bytes, more than "
            "max_array_bytes=24",
        ),
        (
            "recordings/rec/extensions/ext",
            "integrity not verified: the array declares 8000000000000 "
            "bytes, more than max_array_bytes=24",
        ),
    ]
    # The sample hash is still verified, and nothing else is reported.
    assert validate_store(
        store_path, verify_integrity=False, max_array_bytes=24
    ).valid
    for limit in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            validate_store(store_path, max_array_bytes=cast(Any, limit))


def test_validation_limit_skips_only_oversized_samples(
    tmp_path: Path,
) -> None:
    """A skipped sample hash leaves the metadata hash verified.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    _, store = _validation_store(tmp_path)
    recording = store.recordings.open("rec")
    recording.update_sha512()
    store.update_integrity()
    # Make both stored sample digests stale without changing metadata.
    recording._samples_array[0] = 99

    report = validate_store(store, max_array_bytes=8)

    assert [(issue.path, issue.message) for issue in report.issues] == [
        (
            "recordings/rec/samples",
            "integrity not verified: the array declares 16 bytes, more than "
            "max_array_bytes=8",
        )
    ]
    assert not validate_store(store, max_array_bytes=16).valid


def test_validate_command_accepts_array_size_limit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command line exposes the validation size limit.

    Args:
        tmp_path: Pytest temporary path fixture.
        capsys: Pytest output capture fixture.
    """
    store_path, _ = _validation_store(tmp_path)
    command = ["store", str(store_path), "validate", "--format", "json"]

    status = SigMFCommand().run([*command, "--max-array-bytes", "8"])
    result = json.loads(capsys.readouterr().out)

    assert status == 1
    assert [issue["path"] for issue in result["issues"]] == [
        "recordings/rec/samples"
    ]
    assert SigMFCommand().run([*command, "--max-array-bytes", "16"]) == 0
    capsys.readouterr()
    # The limit also bounds structural reads, so the command accepts it
    # without hash verification and reports a result instead of a usage
    # error.
    status = SigMFCommand().run(
        [*command, "--max-array-bytes", "8", "--skip-integrity"]
    )
    assert status in {0, 1}
    assert "valid" in json.loads(capsys.readouterr().out)
    with pytest.raises(SystemExit) as exit_info:
        SigMFCommand().run([*command, "--max-array-bytes", "0"])
    assert exit_info.value.code == 2
    assert "--max-array-bytes" in capsys.readouterr().err


def test_validation_reports_symbolic_links(tmp_path: Path) -> None:
    """Links planted below a local store root are reported, not followed.

    A writable open would write chunks through the linked directory.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    _validation_store(tmp_path)
    store_path = tmp_path / "validation.zarr"
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "notes.txt").write_text("keep")
    chunks = store_path / "recordings" / "rec" / "samples" / "c"
    chunks.rename(victim / "c")
    chunks.symlink_to(victim / "c", target_is_directory=True)
    document = store_path / "indexes" / "zarr.json"
    document.rename(victim / "zarr.json")
    document.symlink_to(victim / "zarr.json")

    report = validate_store(store_path)

    assert [(issue.path, issue.message) for issue in report.issues] == [
        ("indexes/zarr.json", "symbolic link below the store root"),
        ("recordings/rec/samples/c", "symbolic link below the store root"),
    ]


def _set_attribute(
    group: zarr.Group, key: str, field: str | None, value: object
) -> None:
    """Replace one attribute, or one field of a global metadata attribute.

    Args:
        group: Group to edit.
        key: Attribute name.
        field: Field of a `global` attribute to replace, or None to replace
            the whole attribute.
        value: New value.
    """
    if field is None:
        group.attrs[key] = cast(Any, value)
        return
    metadata = json_object(group.attrs[key], name=key)
    metadata[field] = cast(Any, value)
    group.attrs[key] = metadata


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize(
    ("key", "field", "value", "message"),
    [
        (
            "global",
            "core:sample_rate",
            "1e6",
            "$.global['core:sample_rate']: '1e6' is not of type 'number'",
        ),
        ("global", "core:datatype", "cf33_le", "$.global['core:datatype']"),
        ("global", "core:version", "1.2", "$.global['core:version']"),
        (
            "global",
            "core:sha512",
            12345,
            "$.global['core:sha512']: 12345 is not of type 'string'",
        ),
        (
            "captures",
            None,
            [{"core:sample_start": "0"}],
            "$.captures[0]['core:sample_start']",
        ),
        (
            "annotations",
            None,
            [{"core:sample_start": 0, "core:sample_count": "2"}],
            "$.annotations[0]['core:sample_count']",
        ),
        (
            "captures",
            None,
            [{"core:sample_start": 2}, {"core:sample_start": 0}],
            "captures has incorrect sample start ordering",
        ),
    ],
)
def test_validation_applies_sigmf_core_schema(
    tmp_path: Path,
    zarr_format: ZarrFormat,
    key: str,
    field: str | None,
    value: object,
    message: str,
) -> None:
    """Core metadata that export would reject is reported.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
        key: Recording attribute to edit.
        field: Global metadata field to edit, or None.
        value: Value written by another tool.
        message: Expected part of the issue message.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, zarr_format=zarr_format)
    store.recordings.create(
        "rec",
        sample_shape=(4,),
        global_metadata={"core:datatype": "rf32_le", "core:version": "1.2.0"},
    )
    assert validate_store(store_path).valid
    group = zarr.open_group(store_path, path="recordings/rec", mode="r+")
    _set_attribute(group, key, field, value)

    report = validate_store(store_path)

    assert [issue.path for issue in report.issues] == ["recordings/rec"]
    assert report.issues[0].message.startswith(
        "metadata does not satisfy the SigMF schema: "
    )
    assert message in report.issues[0].message


def test_validation_schema_permits_absent_datatype_and_namespaces(
    tmp_path: Path,
) -> None:
    """Fields that SigMF-Zarr leaves optional are not schema issues.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path)
    store.recordings.create(
        "rec",
        sample_shape=(4,),
        global_metadata={"undeclared:field": 1},
        captures=[{"core:sample_start": 0}, {"core:sample_start": 0}],
    )

    assert validate_store(store_path).issues == ()


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize(
    "damage",
    [
        "checksum",
        "extensions",
        "indexes",
        "integrity",
        "unit",
        "checksum-list",
        "checksum-object",
    ],
)
def test_validation_reports_documented_requirements(
    tmp_path: Path, zarr_format: ZarrFormat, damage: str
) -> None:
    """States that the format calls invalid are reported.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
        damage: Requirement that another tool violates.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, zarr_format=zarr_format)
    recording = store.recordings.create(
        "rec", sample_shape=(4,), sample_checksum=None
    )
    recording.add_index("snr", np.zeros(4), axis="time", field="x:snr")
    group = zarr.open_group(store_path, path="recordings/rec", mode="r+")
    path = "recordings/rec"
    if damage == "checksum":
        metadata = json_object(group.attrs["global"], name="global")
        del metadata["sigmf-zarr:checksum"]
        group.attrs["global"] = metadata
        expected = "global metadata requires 'sigmf-zarr:checksum'"
    elif damage in {"extensions", "indexes"}:
        del group[damage]
        expected = f"recording requires the {damage!r} group"
    elif damage == "integrity":
        group.attrs[INTEGRITY_ATTR] = ["sha512"]
        expected = "'integrity' attribute must be a JSON object"
    elif damage == "unit":
        cast(zarr.Array[Any], group["indexes/snr"]).attrs["unit"] = 5
        path = "recordings/rec/indexes/snr"
        expected = "index 'unit' attribute must be a string"
    else:
        value = ["crc32c"] if damage == "checksum-list" else {"a": 1}
        _set_attribute(group, "global", "sigmf-zarr:checksum", value)
        expected = (
            "Recording 'rec' has unsupported 'sigmf-zarr:checksum' "
            f"metadata: {value!r}"
        )

    report = validate_store(store_path)

    assert [(issue.path, issue.message) for issue in report.issues] == [
        (path, expected)
    ]


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("kind", ["recordings", "collections"])
def test_validation_reports_arrays_among_resources(
    tmp_path: Path, zarr_format: ZarrFormat, kind: str
) -> None:
    """Recording and collection containers hold only groups.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
        kind: Resource container holding the array.
    """
    store_path = tmp_path / "store.zarr"
    SigMFZarrStore.create(store_path, zarr_format=zarr_format)
    group = zarr.open_group(store_path, path=kind, mode="r+")
    group.create_array("junk", shape=(3,), dtype="i4")

    report = validate_store(store_path)

    assert [(issue.path, issue.message) for issue in report.issues] == [
        (f"{kind}/junk", f"{kind} must contain only groups")
    ]


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_validation_reports_non_object_store_integrity(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    """Root and collection `integrity` attributes must be objects.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, zarr_format=zarr_format)
    store.recordings.create("rec", sample_shape=(4,))
    store.collections.create("col", recording_ids=["rec"])
    zarr.open_group(store_path, mode="r+").attrs[INTEGRITY_ATTR] = "sha512"
    zarr.open_group(store_path, path="collections/col", mode="r+").attrs[
        INTEGRITY_ATTR
    ] = 1

    report = validate_store(store_path)

    assert [(issue.path, issue.message) for issue in report.issues] == [
        ("/", "'integrity' attribute must be a JSON object"),
        ("collections/col", "'integrity' attribute must be a JSON object"),
    ]

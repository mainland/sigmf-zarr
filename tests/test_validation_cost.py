"""Tests that opening and validating untrusted stores has bounded cost."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from aiohttp import ClientPayloadError
from zarr.core.array import Array
from zarr.storage import FsspecStore

import sigmf_zarr.json as json_module
from sigmf_zarr.integrity import _decoded_chunk_nbytes
from sigmf_zarr.json import JSONObject, json_value
from sigmf_zarr.store import SigMFRecording, SigMFZarrStore, ZarrFormat
from sigmf_zarr.validation import validate_store
from tests.http_store import serve_directory


def _issues(report: Any) -> list[tuple[str, str]]:
    """Return the path and message of each issue in a report.

    Args:
        report: Validation report.

    Returns:
        Issue paths and messages in report order.
    """
    return [(issue.path, issue.message) for issue in report.issues]


def _batched_store(
    tmp_path: Path, entries: list[dict[str, Any]] | None = None
) -> tuple[Path, SigMFZarrStore]:
    """Create a store with one batched recording and per-item metadata.

    Args:
        tmp_path: Pytest temporary path fixture.
        entries: Per-item metadata. Two empty entries by default.

    Returns:
        Store path and opened writable store.
    """
    items = entries if entries is not None else [{}, {}]
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path)
    recording = store.recordings.create(
        "rec", batched=True, sample_shape=(4,), sample_chunks=(2, 4)
    )
    recording.append_samples(
        np.zeros((len(items), 4), dtype=np.float32),
        item_metadata=items,
    )
    return store_path, store


def test_json_value_names_values_only_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Value names are built for errors only.

    Building a name for every value repeats each ancestor key, so long keys
    above a long list made validation quadratic in the input size.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    names: list[str] = []
    original = json_module._value_name

    def counted(base: str, parts: list[str | int]) -> str:
        """Record each name that validation builds.

        Args:
            base: Name of the outermost value.
            parts: Path to the named value.

        Returns:
            Formatted name.
        """
        name = original(base, parts)
        names.append(name)
        return name

    monkeypatch.setattr(json_module, "_value_name", counted)
    value: object = [0] * 1000
    for _ in range(20):
        value = {"k" * 1000: value}

    json_value(value, name="value")

    assert names == []
    with pytest.raises(
        ValueError, match=r"^Expected value\.a\[1\]\.b numbers to be finite$"
    ):
        json_value({"a": [0, {"b": float("nan")}]}, name="value")
    with pytest.raises(ValueError, match=r"Expected value\.a object keys"):
        json_value({"a": {1: 0}}, name="value")


def test_recording_open_reads_no_item_metadata_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A default open does not decode per-item metadata entries.

    A few hundred bytes of compressed entries can decode to millions of
    items, so a full scan on every open made each open slow.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    _, store = _batched_store(tmp_path)
    store.recordings.open("rec")._item_metadata_array[1] = "not JSON"
    original = Array.__getitem__

    def guarded_read(self: Array[Any], selection: Any) -> Any:
        """Fail on any per-item metadata read.

        Args:
            self: Array being read.
            selection: Zarr selection.

        Returns:
            Selected values.
        """
        assert not self.path.endswith("item_metadata")
        return original(self, selection)

    with monkeypatch.context() as patch:
        patch.setattr(Array, "__getitem__", guarded_read)
        recording = store.recordings.open("rec")
    assert recording.get_item_metadata(0) == {}
    with pytest.raises(ValueError, match="not valid JSON"):
        recording.get_item_metadata(1)
    with pytest.raises(ValueError, match="not valid JSON"):
        store.recordings.open("rec", validation="full")


def test_validation_skips_item_metadata_chunks_over_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Entries in a chunk larger than the read limit are not decoded.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    store_path, store = _batched_store(tmp_path)
    store.update_integrity()
    chunk_bytes = (
        16 * store.recordings.open("rec").item_metadata_array.chunks[0]
    )
    limit = chunk_bytes - 1
    calls: list[int | None] = []
    original = SigMFRecording._validate_item_metadata_entries

    def counted(
        self: SigMFRecording, *, max_text_bytes: int | None = None
    ) -> bool:
        """Record each entry scan.

        Args:
            self: Recording being validated.
            max_text_bytes: Text limit passed by validation.

        Returns:
            Whether every entry was checked.
        """
        calls.append(max_text_bytes)
        return original(self, max_text_bytes=max_text_bytes)

    monkeypatch.setattr(
        SigMFRecording, "_validate_item_metadata_entries", counted
    )

    report = validate_store(store_path, max_array_bytes=limit)

    assert calls == []
    assert (
        "recordings/rec/item_metadata",
        f"entries not validated: one decoded chunk or shard holds "
        f"{chunk_bytes} bytes, more than max_array_bytes={limit}",
    ) in _issues(report)
    assert validate_store(store_path, max_array_bytes=chunk_bytes).valid
    assert calls == [chunk_bytes]


def test_validation_stops_at_item_metadata_text_limit(tmp_path: Path) -> None:
    """Entry checks stop once the decoded entry text exceeds the limit.

    The metadata hash covers the unchecked entries, so it is not verified.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    text = "x" * 40_000
    store_path, store = _batched_store(
        tmp_path, [{"global": {"x:text": text}}] * 2
    )
    store.update_integrity()
    item_metadata = zarr.open_array(
        store_path / "recordings" / "rec" / "item_metadata", mode="r+"
    )
    # A tampered entry after the limit is neither checked nor hashed.
    item_metadata[1] = "not JSON" + text
    chunk_bytes = 16 * item_metadata.chunks[0]
    limit = max(chunk_bytes, 50_000)

    report = validate_store(store_path, max_array_bytes=limit)

    assert _issues(report) == [
        (
            "recordings/rec/item_metadata",
            "entries not fully validated: they hold more than "
            f"max_array_bytes={limit} bytes of text",
        )
    ]
    unlimited = _issues(validate_store(store_path))
    assert ("recordings/rec", "stored item_metadata[1] is not valid JSON") in (
        unlimited
    )


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_validation_counts_decoded_chunk_size(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    """A chunk larger than its array counts against the read limit.

    Reading an eight-element index with a 2**20-element chunk decodes the
    whole chunk, so the logical size alone does not bound the read.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, zarr_format=zarr_format)
    recording = store.recordings.create(
        "rec", sample_shape=(8,), sample_chunks=(8,)
    )
    recording.add_index(
        "snr", np.arange(8, dtype=np.int16), axis="time", field="x:snr"
    )
    store.update_integrity()
    index_path = store_path / "recordings" / "rec" / "indexes" / "snr"
    document = index_path / ("zarr.json" if zarr_format == 3 else ".zarray")
    metadata = json.loads(document.read_text())
    if zarr_format == 3:
        metadata["chunk_grid"]["configuration"]["chunk_shape"] = [2**20]
    else:
        metadata["chunks"] = [2**20]
    document.write_text(json.dumps(metadata))
    for chunk in index_path.glob("c/*" if zarr_format == 3 else "0"):
        chunk.unlink()

    report = validate_store(store_path, max_array_bytes=1024)

    assert _issues(report) == [
        (
            "recordings/rec/indexes/snr",
            "integrity not verified: one decoded chunk or shard holds "
            "2097152 bytes, more than max_array_bytes=1024",
        )
    ]


def test_decoded_chunk_size_counts_whole_shard() -> None:
    """A sharded array counts the shard rather than the inner chunk."""
    sharded = zarr.create_array(
        {}, shape=(8,), chunks=(4,), shards=(1024,), dtype="int16"
    )
    plain = zarr.create_array({}, shape=(8,), chunks=(4,), dtype="int16")
    strings = zarr.create_array({}, shape=(8,), chunks=(64,), dtype=str)

    assert _decoded_chunk_nbytes(sharded) == 2048
    assert _decoded_chunk_nbytes(plain) == 8
    assert _decoded_chunk_nbytes(strings) == 64 * 16


def test_validation_reports_special_files_without_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FIFOs, sockets, and links are reported before any content is read.

    Reading a FIFO blocks and reading a link to `/dev/zero` never ends, so
    content checks are skipped once such an entry is found.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    store_path, store = _batched_store(tmp_path)
    store.update_integrity()
    extensions = store_path / "recordings" / "rec" / "extensions"
    os.mkfifo(extensions / "fifo")
    monkeypatch.chdir(extensions)
    # A relative name keeps the socket path within the platform limit.
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind("socket")
    (extensions / "link").symlink_to(tmp_path)
    # Content checks would report this damaged attribute document.
    (store_path / "recordings" / "rec" / "zarr.json").write_text("{")

    report = validate_store(store_path)

    special = "neither a regular file nor a directory below the store root"
    assert _issues(report) == [
        ("recordings/rec/extensions/fifo", special),
        (
            "recordings/rec/extensions/link",
            "symbolic link below the store root",
        ),
        ("recordings/rec/extensions/socket", special),
    ]
    assert report.recordings_checked == 0


@pytest.mark.skipif(
    os.name != "posix" or os.geteuid() == 0,
    reason="requires POSIX permissions enforced for the current user",
)
def test_validation_reports_unreadable_directories(tmp_path: Path) -> None:
    """A directory that cannot be listed can hide a link that Zarr follows.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path, store = _batched_store(tmp_path)
    store.update_integrity()
    hidden = store_path / "recordings" / "rec" / "extensions" / "hidden"
    hidden.mkdir()
    (hidden / "0").symlink_to(tmp_path)
    hidden.chmod(0o111)
    try:
        report = validate_store(store_path)
    finally:
        hidden.chmod(0o755)

    assert _issues(report) == [
        (
            "recordings/rec/extensions/hidden",
            "directory cannot be listed: Permission denied",
        )
    ]


@pytest.mark.parametrize("scheme", ["local://", "simplecache::file://"])
def test_validation_treats_fsspec_local_stores_as_local(
    tmp_path: Path, scheme: str
) -> None:
    """Stores opened through fsspec local file systems are walked as well.

    Args:
        tmp_path: Pytest temporary path fixture.
        scheme: URL prefix naming an fsspec local file system.
    """
    # Zarr opens synchronous fsspec file systems only with fsspec 2024.12
    # or later.
    pytest.importorskip("fsspec.implementations.asyn_wrapper")
    store_path, store = _batched_store(tmp_path)
    store.update_integrity()
    (store_path / "recordings" / "rec" / "extensions" / "link").symlink_to(
        tmp_path
    )
    url = f"{scheme}{store_path.resolve()}"
    expected = [
        (
            "recordings/rec/extensions/link",
            "symbolic link below the store root",
        )
    ]

    assert _issues(validate_store(url)) == expected
    assert _issues(validate_store(SigMFZarrStore.open(url))) == expected


@pytest.mark.skipif(
    os.name != "posix" or os.geteuid() == 0,
    reason="requires POSIX permissions enforced for the current user",
)
def test_validation_reports_unreadable_files(tmp_path: Path) -> None:
    """A permission failure is reported rather than raised.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path, store = _batched_store(tmp_path)
    store.update_integrity()
    chunk = store_path / "recordings" / "rec" / "item_metadata" / "c" / "0"
    chunk.chmod(0)
    try:
        report = validate_store(store_path)
    finally:
        chunk.chmod(0o644)

    assert [path for path, _ in _issues(report)] == ["recordings/rec"]
    assert "Permission denied" in report.issues[0].message


# Zarr 3.4 and later read a stored chunk size of 0 as 1 and warn.
@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
def test_validation_reports_store_index_metadata_failures(
    tmp_path: Path,
) -> None:
    """Damaged store-wide index metadata and huge shapes are reported.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path, store = _batched_store(tmp_path)
    store.add_index("sidx", np.arange(3, dtype=np.int32))
    document = store_path / "indexes" / "sidx" / "zarr.json"
    metadata = json.loads(document.read_text())
    metadata["chunk_grid"]["configuration"]["chunk_shape"] = [0]
    document.write_text(json.dumps(metadata))
    samples = store_path / "recordings" / "rec" / "samples" / "zarr.json"
    sample_metadata = json.loads(samples.read_text())
    sample_metadata["shape"][0] = 2**64
    samples.write_text(json.dumps(sample_metadata))

    report = validate_store(store_path, verify_integrity=False)

    paths = {path for path, _ in _issues(report)}
    assert "recordings/rec" in paths
    # Zarr before 3.4 fails to read the index, and later versions repair its
    # chunk shape, which the hierarchy check reports from the document.
    assert paths - {"recordings/rec"} in (
        {"indexes"},
        {"indexes/sidx"},
        {"indexes", "indexes/sidx"},
    )


def test_validation_reports_failed_remote_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A truncated HTTP response is reported as an issue.

    The store boundary translates HTTP library failures into `OSError`.

    Args:
        tmp_path: Pytest temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    _, store = _batched_store(tmp_path)
    store.update_integrity()
    original = FsspecStore.get

    async def truncated(
        self: FsspecStore, key: str, *args: Any, **kwargs: Any
    ) -> Any:
        """Fail sample chunk reads as a truncated response would.

        Args:
            self: Store being read.
            key: Store key.
            *args: Remaining read arguments.
            **kwargs: Remaining read keyword arguments.

        Returns:
            Stored value for any other key.

        Raises:
            ClientPayloadError: For a sample chunk.
        """
        if "/samples/c/" in key:
            raise ClientPayloadError("Response payload is not completed")
        return await original(self, key, *args, **kwargs)

    monkeypatch.setattr(FsspecStore, "get", truncated)
    with serve_directory(tmp_path) as url:
        report = validate_store(f"{url}/store.zarr")
        with pytest.raises(OSError, match="ClientPayloadError"):
            SigMFZarrStore.open(f"{url}/store.zarr").recordings.open(
                "rec"
            ).samples[0]

    assert _issues(report) == [
        (
            "recordings/rec",
            "ClientPayloadError: Response payload is not completed",
        )
    ]


def test_channel_group_check_does_not_enumerate_declared_channels(
    tmp_path: Path,
) -> None:
    """A huge declared channel count is compared without listing it.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path)
    store.recordings.create(
        "rec", sample_shape=(2, 4), sample_axes=("channel", "time")
    )
    recording_path = store_path / "recordings" / "rec"
    document = recording_path / "zarr.json"
    metadata = json.loads(document.read_text())
    global_metadata = metadata["attributes"]["global"]
    global_metadata["sigmf-zarr:sample-shape"] = [10**6, 4]
    global_metadata["sigmf-zarr:num-channels"] = 10**6
    document.write_text(json.dumps(metadata))
    samples = recording_path / "samples" / "zarr.json"
    sample_metadata = json.loads(samples.read_text())
    sample_metadata["shape"] = [10**6, 4]
    samples.write_text(json.dumps(sample_metadata))

    with pytest.raises(ValueError, match="channel groups") as error:
        SigMFZarrStore.open(store_path).recordings.open("rec")

    assert str(error.value) == (
        "Recording 'rec' channel groups ['0', '1'] do not match channels "
        "'0' to '999999'"
    )


def test_channel_group_messages_truncate_long_lists(tmp_path: Path) -> None:
    """Extra and noncanonical channel group names are rejected briefly.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr")
    recording = store.recordings.create(
        "rec", sample_shape=(2, 4), sample_axes=("channel", "time")
    )
    for name in ("01", *(str(index) for index in range(2, 12))):
        recording._channels_group.create_group(name)

    with pytest.raises(ValueError, match="channel groups") as error:
        store.recordings.open("rec")
    assert str(error.value) == (
        "Recording 'rec' channel groups ['0', '01', '1', '10', '11', '2', "
        "'3', '4'] and 5 more do not match channels '0' to '1'"
    )

    for name in ("01", *(str(index) for index in range(3, 12))):
        del recording._channels_group[name]
    recording._channels_group.create_group("01")
    del recording._channels_group["1"]
    with pytest.raises(ValueError, match=r"\['0', '01', '2'\]"):
        store.recordings.open("rec")


def test_item_offsets_and_capture_starts_fit_int64(tmp_path: Path) -> None:
    """Per-item sample indexes are bounded like the SigMF schema bounds them.

    A larger `core:offset` made timestamp projection overflow.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path, store = _batched_store(tmp_path)
    recording = store.recordings.open("rec")
    largest = 2**63 - 1
    recording.set_item_metadata_entry(
        0,
        {
            "global": {"core:offset": largest},
            "captures": [{"core:sample_start": largest}],
        },
    )
    entries: tuple[JSONObject, ...] = (
        {"global": {"core:offset": largest + 1}},
        {"global": {"core:offset": "0"}},
        {"captures": [{"core:sample_start": largest + 1}]},
    )
    for entry in entries:
        with pytest.raises(ValueError, match="at most 9223372036854775807"):
            recording.set_item_metadata_entry(1, entry)
        with pytest.raises(ValueError, match="at most 9223372036854775807"):
            recording.append_samples(
                np.zeros((1, 4), dtype=np.float32), item_metadata=[entry]
            )

    item_metadata = zarr.open_array(
        store_path / "recordings" / "rec" / "item_metadata", mode="r+"
    )
    item_metadata[1] = json.dumps({"global": {"core:offset": 10**400}})
    report = validate_store(store_path, verify_integrity=False)
    assert _issues(report) == [
        (
            "recordings/rec",
            "stored item_metadata[1] global metadata core:offset must be a "
            "nonnegative integer at most 9223372036854775807",
        )
    ]

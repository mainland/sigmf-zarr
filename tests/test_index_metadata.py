"""Independent metadata, index decoding, and bounded opening contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from zarr.core.array import Array

from sigmf_zarr import SigMFRecording, SigMFZarrStore, validate_store
from sigmf_zarr.indexes import validate_categorical_values
from sigmf_zarr.store import ZarrFormat


@pytest.fixture(params=[2, 3])
def recording(
    tmp_path: Path, request: pytest.FixtureRequest
) -> SigMFRecording:
    """Create a three-item recording in each physical format.

    Args:
        tmp_path: Temporary directory fixture.
        request: Parametrized physical format.

    Returns:
        Recording with JSON metadata and independently stored category IDs.
    """
    store = SigMFZarrStore.create(
        tmp_path / "data.zarr", zarr_format=cast(ZarrFormat, request.param)
    )
    result = store.recordings.open(
        "rec",
        batched=True,
        sample_shape=(2, 4),
        sample_axes=("iq", "time"),
        global_metadata={"example:class": "shared"},
    )
    result.append_samples(
        np.zeros((3, 2, 4), dtype=np.float32),
        item_metadata=[{"global": {"example:class": "JSON"}}, None, None],
    )
    result.add_index(
        "class_id",
        np.array([0, 1, 0], dtype=np.int16),
        axis="item",
        field="example:class",
        labels=["BPSK", "QPSK"],
        chunks=(1,),
    )
    return result


def test_discovery_reads_only_descriptors(
    recording: SigMFRecording, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Find all nested alternatives without reading values or certifying them.

    Args:
        recording: Example recording.
        monkeypatch: Patch fixture.
    """
    recording.add_index(
        "quality/alternative",
        [1, 0, 1],
        axis="item",
        field="example:class",
        labels=["BPSK", "QPSK"],
    )
    recording.add_index(
        "time_class",
        [0, 0, 1, 1],
        axis="time",
        field="example:class",
    )
    recording._indexes_group["quality/alternative"].attrs[
        "sigmf-zarr:valid"
    ] = False

    def no_read(*args: object, **kwargs: object) -> None:
        """Reject array payload reads during discovery."""
        raise AssertionError("Discovery read an array payload")

    monkeypatch.setattr(Array, "__getitem__", no_read)
    assert recording.find_indexes("example:class") == (
        "class_id",
        "quality/alternative",
        "time_class",
    )
    assert recording.find_indexes("example:class", axis="item") == (
        "class_id",
        "quality/alternative",
    )
    assert recording.find_indexes("missing") == ()
    with pytest.raises(ValueError, match="invalid"):
        recording.index("quality/alternative")


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        (1, ["QPSK"]),
        (-1, ["BPSK"]),
        ([2, 1, 2, 0], ["BPSK", "QPSK", "BPSK", "BPSK"]),
        ((-1, 1), ["BPSK", "QPSK"]),
        ([], []),
        (slice(None, None, -1), ["BPSK", "QPSK", "BPSK"]),
        (slice(0, 3, 2), ["BPSK", "BPSK"]),
        (slice(2, 1), []),
    ],
)
def test_decode_selection(
    recording: SigMFRecording, selection: Any, expected: list[str]
) -> None:
    """Decode bounded selections while preserving order and shape.

    Args:
        recording: Example recording.
        selection: Requested positions.
        expected: Corresponding labels.
    """
    result = recording.decode_index("class_id", selection=selection)
    assert result.ndim == 1
    assert result.tolist() == expected
    assert recording.index("class_id").dtype == np.dtype("int16")


@pytest.mark.parametrize("selection", [3, -4, [0, 2**100]])
def test_decode_rejects_out_of_range_positions(
    recording: SigMFRecording, selection: Any
) -> None:
    """Reject invalid positions before narrowing their integer dtype.

    Args:
        recording: Example recording.
        selection: Invalid position selection.
    """
    with pytest.raises(IndexError):
        recording.decode_index("class_id", selection=selection)


@pytest.mark.parametrize("selection", [True, [0, True], [1.0], "1"])
def test_decode_rejects_noninteger_positions(
    recording: SigMFRecording, selection: Any
) -> None:
    """Do not interpret Boolean masks or floats as positional selections.

    Args:
        recording: Example recording.
        selection: Invalid selection type.
    """
    with pytest.raises(TypeError):
        recording.decode_index("class_id", selection=selection)


def test_decoding_checks_selected_values_only(
    recording: SigMFRecording,
) -> None:
    """An invalid category outside the selection does not trigger a full scan.

    Args:
        recording: Example recording.
    """
    with recording.mutate_index("class_id") as index:
        index[1] = 99
    assert recording.decode_index("class_id", selection=[2, 0]).tolist() == [
        "BPSK",
        "BPSK",
    ]
    with pytest.raises(ValueError, match="outside"):
        recording.decode_index("class_id", selection=1)
    assert recording.index("class_id")[1] == 99


@pytest.mark.parametrize(
    ("values", "labels"),
    [
        ([True], ["a"]),
        ([0.0], ["a"]),
        ([[0]], ["a"]),
        ([-1], ["a"]),
        ([1], ["a"]),
        ([0], None),
        ([0], "a"),
        ([0], [float("nan")]),
        ([0], []),
        (np.array([2**64 - 1], dtype=np.uint64), ["a"]),
    ],
)
def test_categorical_validation(values: Any, labels: object) -> None:
    """Reject malformed category values and lookup tables without storage.

    Args:
        values: Invalid category vector.
        labels: Lookup metadata to validate.
    """
    with pytest.raises(ValueError):
        validate_categorical_values(values, labels=labels)


def test_json_lookup_values_are_independent(recording: SigMFRecording) -> None:
    """Nested JSON labels retain a one-dimensional result with detached copies.

    Args:
        recording: Example recording.
    """
    with recording.mutate_index("class_id") as index:
        index.attrs["labels"] = [[1, 2], {"mode": "QPSK"}]
    result = recording.decode_index("class_id", selection=[0, 0, 1])
    assert result.shape == (3,)
    result[0].append(3)
    result[2]["mode"] = "changed"
    assert result[1] == [1, 2]
    assert recording.index("class_id").attrs["labels"] == [
        [1, 2],
        {"mode": "QPSK"},
    ]


def test_metadata_and_indexes_remain_independent(
    recording: SigMFRecording,
) -> None:
    """JSON and sample edits neither synchronize nor invalidate category IDs.

    Args:
        recording: Example recording.
    """
    recording.set_item_metadata(
        [{"global": {"example:class": "new"}}, {}, {}], overwrite=True
    )
    with recording.mutate_samples() as samples:
        samples[:] = 1
    assert (
        recording.resolved_item_metadata(0)["global"]["example:class"] == "new"
    )
    assert (
        recording.resolved_item_metadata(1)["global"]["example:class"]
        == "shared"
    )
    assert recording.decode_index("class_id", selection=0).tolist() == ["BPSK"]
    assert validate_store(recording._store, verify_integrity=False).valid


@pytest.mark.parametrize(
    "key",
    [
        "axis",
        "field",
        "labels",
        "unit",
        "kind",
        "sigmf-zarr:valid",
        "sigmf-zarr:invalid-reason",
    ],
)
def test_attributes_cannot_override_index_contract(
    recording: SigMFRecording, key: str
) -> None:
    """Invalid replacement attributes fail before touching the existing index.

    Args:
        recording: Example recording.
        key: Attribute owned by the index model.
    """
    recording.update_integrity()
    before = recording.integrity
    with pytest.raises(ValueError, match="override"):
        recording.add_index(
            "class_id",
            [1, 1, 1],
            axis="item",
            field="example:class",
            overwrite=True,
            attributes={key: "override"},
        )
    assert recording.index("class_id")[:].tolist() == [0, 1, 0]
    assert recording.integrity == before


def test_descriptive_attributes(recording: SigMFRecording) -> None:
    """Persist detached provenance without introducing a binding.

    Args:
        recording: Example recording.
    """
    attributes = {"measurement": {"definition": "source"}}
    index = recording.add_index(
        "snr",
        [1, 2, 3],
        axis="item",
        field="example:snr",
        attributes=attributes,
    )
    attributes["measurement"]["definition"] = "changed"
    assert index.attrs["measurement"] == {"definition": "source"}


def test_structural_open_defers_only_entry_validation(
    recording: SigMFRecording, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Open without JSON reads, then reject corrupt selected entries and scans.

    Args:
        recording: Example recording.
        monkeypatch: Patch fixture.
    """
    recording._item_metadata_array[1] = "not JSON"
    original = Array.__getitem__

    def guarded_read(self: Array, selection: Any) -> Any:
        """Detect a metadata payload read during structural opening."""
        assert not self.path.endswith("item_metadata")
        return original(self, selection)

    with monkeypatch.context() as patch:
        patch.setattr(Array, "__getitem__", guarded_read)
        opened = recording._store.recordings.open(
            "rec",
            create=False,
            validation="structural",
        )
        assert opened.decode_index("class_id", selection=0).tolist() == [
            "BPSK"
        ]
    assert opened.get_item_metadata(0)["global"] == {"example:class": "JSON"}
    with pytest.raises(ValueError, match="not valid JSON"):
        opened.get_item_metadata(1)
    with pytest.raises(ValueError, match="not valid JSON"):
        recording._store.recordings.open("rec", create=False)
    assert not validate_store(recording._store, verify_integrity=False).valid


def test_structural_open_rejects_misaligned_metadata(
    recording: SigMFRecording,
) -> None:
    """Skipping entry scans must retain structural checks.

    Args:
        recording: Example recording.
    """
    recording._item_metadata_array.resize((2,))
    with pytest.raises(ValueError, match="item axis shape"):
        recording._store.recordings.open("rec", validation="structural")


@pytest.mark.parametrize("defect", ["dtype", "checksum"])
def test_structural_open_validates_storage_descriptors(
    recording: SigMFRecording, defect: str
) -> None:
    """Reject wrong string storage and missing checksums without a JSON scan.

    Args:
        recording: Example recording.
        defect: Invalid metadata storage descriptor.
    """
    from zarr.core.dtype import VariableLengthUTF8

    group = recording._raw_group
    del group["item_metadata"]
    dtype = "int32" if defect == "dtype" else VariableLengthUTF8()
    group.create_array("item_metadata", shape=(3,), dtype=dtype)
    message = "variable-length UTF-8" if defect == "dtype" else "CRC32C"
    with pytest.raises(ValueError, match=message):
        recording._store.recordings.open("rec", validation="structural")


def test_invalid_mode_fails_before_creation(recording: SigMFRecording) -> None:
    """Reject an unknown validation mode before any store mutation.

    Args:
        recording: Example recording.
    """
    with pytest.raises(ValueError, match="validation"):
        recording._store.recordings.open(
            "new", sample_shape=(4,), validation=cast(Any, "none")
        )
    assert "new" not in recording._store.recordings


def test_invalid_attributes_fail_before_replacement(
    recording: SigMFRecording,
) -> None:
    """Reject non-JSON provenance before deleting an existing array.

    Args:
        recording: Example recording.
    """
    with pytest.raises(ValueError, match="finite"):
        recording.add_index(
            "class_id",
            [1, 1, 1],
            axis="item",
            field="example:class",
            overwrite=True,
            attributes={"measurement": float("nan")},
        )
    assert recording.index("class_id")[:].tolist() == [0, 1, 0]


@pytest.mark.parametrize("key", ["field", "kind"])
def test_required_descriptors_are_checked_on_read_and_mutation(
    recording: SigMFRecording, key: str
) -> None:
    """Missing descriptors must prevent reads, validation, and certification.

    Args:
        recording: Example recording in either physical format.
        key: Required descriptor to remove.
    """
    original = recording.index("class_id").attrs[key]
    with (
        pytest.raises(ValueError, match=key),
        recording.mutate_index("class_id") as index,
    ):
        del index.attrs[key]
    with pytest.raises(ValueError, match="invalid"):
        recording.index("class_id")

    # A separately produced store may omit descriptors without an invalid
    # marker. Its reader must apply the same structural contract.
    array = recording._indexes_group["class_id"]
    del array.attrs["sigmf-zarr:valid"]
    with pytest.raises(ValueError, match=key):
        recording.index("class_id")
    report = validate_store(recording._store, require_integrity=True)
    assert not report.valid
    assert any(key in issue.message for issue in report.issues)
    with pytest.raises(ValueError, match=key):
        recording._store.update_integrity()

    with recording.mutate_index("class_id") as index:
        index.attrs[key] = original
    recording._store.update_integrity()
    assert validate_store(recording._store, require_integrity=True).valid


@pytest.mark.parametrize("key", ["field", "kind"])
@pytest.mark.parametrize("value", ["", None, 12])
def test_invalid_descriptors_do_not_replace_an_index(
    recording: SigMFRecording, key: str, value: Any
) -> None:
    """Reject malformed descriptors before changing an existing index.

    Args:
        recording: Example recording in either physical format.
        key: Required descriptor to replace.
        value: Invalid descriptor value.
    """
    recording._store.update_integrity()
    attributes = dict(recording.index("class_id").attrs)
    descriptors = {"field": "example:class", "kind": "metadata", key: value}
    with pytest.raises(ValueError, match=key):
        recording.add_index(
            "class_id", [1, 1, 1], axis="item", overwrite=True, **descriptors
        )
    np.testing.assert_array_equal(recording.index("class_id")[:], [0, 1, 0])
    assert dict(recording.index("class_id").attrs) == attributes
    assert recording._store.verify_integrity()

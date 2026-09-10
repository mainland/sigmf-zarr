"""Tests for format rules that creation, opening, and validation enforce."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import zarr
from zarr.errors import ZarrUserWarning

from sigmf_zarr.json import json_object
from sigmf_zarr.store import SigMFZarrStore
from sigmf_zarr.store._common import ZarrFormat
from sigmf_zarr.validation import ValidationReport, validate_store

FORMATS = pytest.mark.parametrize("zarr_format", [2, 3])
"""Run a test once for each physical Zarr format."""


def _consolidate(store_path: Path, *, path: str = "") -> None:
    """Write consolidated metadata as another Zarr tool would.

    Args:
        store_path: Store root.
        path: Group whose descendants to consolidate, relative to the root.
    """
    with warnings.catch_warnings():
        # Zarr warns that Zarr format 3 does not specify consolidation.
        warnings.simplefilter("ignore", ZarrUserWarning)
        zarr.consolidate_metadata(store_path, path=path)


def _issues(report: ValidationReport) -> list[tuple[str, str]]:
    """Return the paths and messages of a report.

    Args:
        report: Validation report.

    Returns:
        Issue paths paired with messages.
    """
    return [(issue.path, issue.message) for issue in report.issues]


def _raw_group(store_path: Path, path: str) -> zarr.Group:
    """Open a group for editing as another Zarr tool would.

    Args:
        store_path: Store root.
        path: Group path relative to the root.

    Returns:
        Writable group.
    """
    return zarr.open_group(store_path, path=path, mode="r+")


def _set_global_field(group: zarr.Group, field: str, value: object) -> None:
    """Replace one field of a recording's `global` attribute.

    Args:
        group: Recording group.
        field: Global metadata field.
        value: New value.
    """
    metadata = json_object(group.attrs["global"], name="global")
    metadata[field] = cast(Any, value)
    group.attrs["global"] = metadata


def _metadata_document(store_path: Path, path: str, zarr_format: int) -> Path:
    """Return the file that holds a group's attributes.

    Args:
        store_path: Store root.
        path: Group path relative to the root.
        zarr_format: Physical Zarr format.

    Returns:
        Path of `zarr.json` or `.zattrs`.
    """
    return store_path / path / ("zarr.json" if zarr_format == 3 else ".zattrs")


def _edit_sample_rate(store_path: Path, zarr_format: int, rate: float) -> None:
    """Change `core:sample_rate` in a recording's own metadata document.

    Args:
        store_path: Store root.
        zarr_format: Physical Zarr format.
        rate: New sample rate.
    """
    document = _metadata_document(store_path, "recordings/rec", zarr_format)
    content = json.loads(document.read_text())
    attrs = content["attributes"] if zarr_format == 3 else content
    attrs["global"]["core:sample_rate"] = rate
    document.write_text(json.dumps(content))


def _hashed_store(store_path: Path, zarr_format: ZarrFormat) -> None:
    """Create a store with one batched recording and current hashes.

    Args:
        store_path: Store root.
        zarr_format: Physical Zarr format.
    """
    store = SigMFZarrStore.create(store_path, zarr_format=zarr_format)
    recording = store.recordings.create(
        "rec",
        sample_shape=(2, 4),
        batched=True,
        global_metadata={"core:sample_rate": 1.0},
    )
    recording.append_samples(np.ones((3, 2, 4), dtype=np.float32))
    store.update_integrity()


_CONSOLIDATED = (
    "group has consolidated metadata, which a SigMF-Zarr store must not "
    "contain"
)


@FORMATS
def test_consolidated_metadata_does_not_hide_tampering(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    """Reads use the node documents, and validation reports the copy.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
    """
    store_path = tmp_path / "store.zarr"
    _hashed_store(store_path, zarr_format)
    _consolidate(store_path)
    _edit_sample_rate(store_path, zarr_format, 999.0)

    store = SigMFZarrStore.open(store_path, mode="r")

    assert store.recordings["rec"].global_metadata["core:sample_rate"] == 999.0
    assert store.verify_integrity() is False
    assert _issues(validate_store(store_path)) == [
        ("/", _CONSOLIDATED),
        ("recordings/rec", "integrity hashes do not match current content"),
    ]


@FORMATS
def test_consolidated_metadata_does_not_discard_writes(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    """Writes through a store with consolidated metadata stay visible.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
    """
    store_path = tmp_path / "store.zarr"
    _hashed_store(store_path, zarr_format)
    _consolidate(store_path)

    store = SigMFZarrStore.open(store_path, mode="r+")
    recording = store.recordings["rec"]
    recording.append_samples(np.zeros((2, 2, 4), dtype=np.float32))
    recording.set_global_field("x:note", "written")
    store.update_integrity()

    assert len(recording) == 5
    reopened = SigMFZarrStore.open(store_path, mode="r").recordings["rec"]
    assert len(reopened) == 5
    assert reopened.global_metadata["x:note"] == "written"
    assert reopened.verify_integrity() is True


def test_nested_consolidated_metadata_is_ignored(tmp_path: Path) -> None:
    """A child group's consolidated copy does not hide its descendants.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path = tmp_path / "store.zarr"
    _hashed_store(store_path, 3)
    _consolidate(store_path, path="recordings")
    _edit_sample_rate(store_path, 3, 999.0)

    store = SigMFZarrStore.open(store_path, mode="r")

    assert store.recordings["rec"].global_metadata["core:sample_rate"] == 999.0
    assert store.verify_integrity() is False
    assert _issues(validate_store(store_path)) == [
        ("recordings", _CONSOLIDATED),
        ("recordings/rec", "integrity hashes do not match current content"),
    ]


_STRUCTURED = np.array(
    [(10, -20), (11, -21)], dtype=[("lat", "<i4"), ("lon", "<i4")]
)
"""Structured values that array hashes cannot describe."""


@FORMATS
@pytest.mark.parametrize("values", [_STRUCTURED, np.zeros(2, dtype="V4")])
@pytest.mark.filterwarnings("ignore::zarr.errors.UnstableSpecificationWarning")
def test_structured_and_void_arrays_are_rejected(
    tmp_path: Path, zarr_format: ZarrFormat, values: np.ndarray[Any, Any]
) -> None:
    """Writers reject, and validation reports, structured array dtypes.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
        values: Structured or void values.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, zarr_format=zarr_format)
    recording = store.recordings.create("rec", sample_shape=(2,))
    store.update_integrity()

    with pytest.raises(ValueError, match="Structured and void dtypes"):
        recording.add_extension_array("x/position", values)
    with pytest.raises(ValueError, match="Structured and void dtypes"):
        recording.add_index("pos", values, axis="time", field="x:pos")
    with pytest.raises(ValueError, match="Structured and void dtypes"):
        store.add_index("pos", values)
    assert store.verify_integrity() is True

    _raw_group(store_path, "recordings/rec/extensions").create_array(
        "position", data=values
    )
    assert _issues(validate_store(store_path, verify_integrity=False)) == [
        (
            "recordings/rec/extensions/position",
            "structured and void array dtypes are not permitted",
        )
    ]


def _single_recording(store_path: Path, zarr_format: ZarrFormat) -> None:
    """Create a store with one small unbatched recording.

    Args:
        store_path: Store root.
        zarr_format: Physical Zarr format.
    """
    SigMFZarrStore.create(
        store_path, zarr_format=zarr_format
    ).recordings.create("rec", sample_shape=(4,), sample_dtype=np.int8)


@FORMATS
def test_schema_version_must_be_integer(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    """The floating-point value `1.0` is not the schema version `1`.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
    """
    store_path = tmp_path / "store.zarr"
    _single_recording(store_path, zarr_format)
    _raw_group(store_path, "").attrs["schema_version"] = 1.0

    with pytest.raises(ValueError, match="schema version"):
        validate_store(store_path)


@FORMATS
@pytest.mark.parametrize(
    "spelling", ["int8", "i1", "=i1", ">i1", "<i1", "b", "byte"]
)
def test_sample_dtype_must_be_canonical(
    tmp_path: Path, zarr_format: ZarrFormat, spelling: str
) -> None:
    """Only the NumPy `.str` form of the sample dtype is accepted.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
        spelling: Non-canonical spelling of `|i1`.
    """
    store_path = tmp_path / "store.zarr"
    _single_recording(store_path, zarr_format)
    group = _raw_group(store_path, "recordings/rec")
    _set_global_field(group, "sigmf-zarr:dtype", spelling)

    assert _issues(validate_store(store_path)) == [
        (
            "recordings/rec",
            f"Recording 'rec' 'sigmf-zarr:dtype' value {spelling!r} is not "
            "the canonical dtype string '|i1'",
        )
    ]


@FORMATS
@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        (
            "sigmf-zarr:valid",
            1,
            "Index 'snr' attribute 'sigmf-zarr:valid' must be a Boolean, "
            "got 1",
        ),
        (
            "sigmf-zarr:valid",
            "true",
            "Index 'snr' attribute 'sigmf-zarr:valid' must be a Boolean, "
            "got 'true'",
        ),
        (
            "sigmf-zarr:invalid-reason",
            5,
            "Index 'snr' attribute 'sigmf-zarr:invalid-reason' must be a "
            "string, got 5",
        ),
    ],
)
def test_index_markers_must_have_their_types(
    tmp_path: Path,
    zarr_format: ZarrFormat,
    attribute: str,
    value: object,
    message: str,
) -> None:
    """`sigmf-zarr:valid` is a Boolean and the reason is a string.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
        attribute: Index marker attribute to replace.
        value: Value of the wrong type.
        message: Expected issue message.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, zarr_format=zarr_format)
    recording = store.recordings.create("rec", sample_shape=(4,))
    recording.add_index("snr", np.zeros(4), axis="time", field="x:snr")
    store.add_index("snr", np.zeros(3))
    for path in ("recordings/rec/indexes/snr", "indexes/snr"):
        _raw_group(store_path, "")[path].attrs[attribute] = cast(Any, value)

    report = validate_store(store_path, verify_integrity=False)

    repair = ". Repair it with mutate_index() or replace it with add_index()"
    assert _issues(report) == [
        ("indexes/snr", message + repair),
        ("recordings/rec/indexes/snr", message + repair),
    ]


def test_invalid_index_does_not_stop_recording_validation(
    tmp_path: Path,
) -> None:
    """Each index is reported at its own path, and hashes are still checked.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path)
    recording = store.recordings.create("rec", sample_shape=(8,))
    recording.add_index("a", np.zeros(8), axis="time", field="x:a")
    recording.add_index("b", np.zeros(8), axis="time", field="x:b")
    store.update_integrity()
    group = _raw_group(store_path, "recordings/rec")
    group["indexes/a"].attrs["sigmf-zarr:valid"] = False
    group["indexes/b"].attrs["axis"] = "nope"
    cast(zarr.Array[Any], group["samples"])[:] = 1.0

    report = validate_store(store_path)

    assert _issues(report) == [
        ("recordings/rec", "integrity hashes do not match current content"),
        (
            "recordings/rec/indexes/a",
            "Index 'a' is invalid: unknown reason. Repair it with "
            "mutate_index() or replace it with add_index()",
        ),
        ("recordings/rec/indexes/b", "Index 'b' has invalid axis 'nope'"),
    ]
    assert report.indexes_checked == 2


def _damage(group: zarr.Group, damage: str) -> None:
    """Make a `(channel, iq, time)` recording violate one format rule.

    Args:
        group: Recording group with shape `(2, 2, 4)`.
        damage: Rule to violate.
    """
    if damage == "second-time-axis":
        _set_global_field(
            group, "sigmf-zarr:sample-axes", ["time", "iq", "time"]
        )
    elif damage == "iq-length":
        _set_global_field(
            group, "sigmf-zarr:sample-axes", ["channel", "time", "iq"]
        )
    elif damage == "duplicate-axes":
        _set_global_field(
            group, "sigmf-zarr:sample-axes", ["channel", "channel", "time"]
        )
    elif damage == "missing-channels":
        del group["channels"]
    elif damage == "num-channels":
        _set_global_field(group, "sigmf-zarr:num-channels", 3)
    elif damage == "unbatched-item-metadata":
        group.create_array("item_metadata", shape=(2,), dtype=str)
    elif damage == "multidimensional-index":
        del group["indexes/snr"]
        index = group.create_array("indexes/snr", shape=(4, 1), dtype="f8")
        index.attrs.update({"axis": "time", "field": "x:snr", "kind": "x"})
    elif damage == "missing-samples":
        del group["samples"]
    else:
        assert damage == "dtype-mismatch"
        _set_global_field(group, "sigmf-zarr:dtype", "<f8")


@FORMATS
@pytest.mark.parametrize(
    ("damage", "path", "message"),
    [
        ("second-time-axis", "recordings/rec", "must not contain duplicates"),
        ("iq-length", "recordings/rec", "'iq' axis must have length 2"),
        ("duplicate-axes", "recordings/rec", "must not contain duplicates"),
        (
            "missing-channels",
            "recordings/rec",
            "missing required 'channels' group",
        ),
        (
            "num-channels",
            "recordings/rec",
            "does not match channel axis length 2",
        ),
        (
            "unbatched-item-metadata",
            "recordings/rec",
            "has item_metadata but is not batched",
        ),
        (
            "multidimensional-index",
            "recordings/rec/indexes/snr",
            "Index 'snr' must be one-dimensional",
        ),
        ("missing-samples", "recordings/rec", "'samples'"),
        ("dtype-mismatch", "recordings/rec", "does not match samples dtype"),
    ],
)
def test_validation_rejects_layout_violations(
    tmp_path: Path,
    zarr_format: ZarrFormat,
    damage: str,
    path: str,
    message: str,
) -> None:
    """Each documented layout requirement is reported in both formats.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
        damage: Rule that another tool violates.
        path: Expected issue path.
        message: Expected part of the issue message.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, zarr_format=zarr_format)
    recording = store.recordings.create(
        "rec", sample_shape=(2, 2, 4), sample_axes=("channel", "iq", "time")
    )
    recording.add_index("snr", np.zeros(4), axis="time", field="x:snr")
    assert validate_store(store_path).valid
    _damage(_raw_group(store_path, "recordings/rec"), damage)

    report = validate_store(store_path)

    assert [issue.path for issue in report.issues] == [path]
    assert message in report.issues[0].message


@FORMATS
def test_complex_dtype_with_iq_axis_is_rejected(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    """A complex element already holds I and Q, so `iq` is contradictory.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, zarr_format=zarr_format)
    message = "must not contain an 'iq' axis"
    with pytest.raises(ValueError, match=message):
        store.recordings.create(
            "bad",
            sample_shape=(2, 8),
            sample_axes=("iq", "time"),
            sample_dtype=np.complex64,
        )
    assert "bad" not in store.recordings
    store.recordings.create(
        "rec",
        sample_shape=(2, 8),
        sample_axes=("channel", "time"),
        sample_dtype=np.complex64,
    )
    group = _raw_group(store_path, "recordings/rec")
    _set_global_field(group, "sigmf-zarr:sample-axes", ["iq", "time"])
    _set_global_field(group, "sigmf-zarr:num-channels", 1)
    channels = group["channels"]
    assert isinstance(channels, zarr.Group)
    for name in channels.group_keys():
        del group[f"channels/{name}"]

    with pytest.raises(ValueError, match=message):
        SigMFZarrStore.open(store_path, mode="r").recordings.open("rec")
    report = validate_store(store_path)
    assert [issue.path for issue in report.issues] == ["recordings/rec"]
    assert message in report.issues[0].message


_BAD_NAMES = ["line\nfeed", "escape\x1b[2J", "bell\x07", "surrogate\udc80"]
"""Names with control characters or a lone surrogate."""


@pytest.mark.parametrize("name", _BAD_NAMES)
def test_writers_reject_names_with_control_characters(
    tmp_path: Path, name: str
) -> None:
    """Resource and array names must be printable Unicode text.

    Args:
        tmp_path: Pytest temporary path fixture.
        name: Invalid name.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr")
    recording = store.recordings.create("rec", sample_shape=(4,))
    message = "must not contain control characters or lone surrogates"

    with pytest.raises(ValueError, match=message):
        store.recordings.create(name, sample_shape=(4,))
    with pytest.raises(ValueError, match=message):
        store.collections.create(name, recording_ids=["rec"])
    with pytest.raises(ValueError, match=message):
        recording.add_index(f"q/{name}", np.zeros(4), axis="time", field="x")
    with pytest.raises(ValueError, match=message):
        recording.add_extension_array(name, np.zeros(2))
    with pytest.raises(ValueError, match=message):
        store.add_index(name, np.zeros(2))
    assert store.list_recordings() == ("rec",)


@FORMATS
@pytest.mark.parametrize("name", _BAD_NAMES)
def test_validation_reports_names_with_control_characters(
    tmp_path: Path, zarr_format: ZarrFormat, name: str
) -> None:
    """Names that another tool wrote are reported once at their own path.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
        name: Invalid name.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, zarr_format=zarr_format)
    store.recordings.create("rec", sample_shape=(4,))
    root = _raw_group(store_path, "")
    root.create_group(f"recordings/{name}")
    index = root.create_array(
        f"recordings/rec/indexes/{name}", shape=(4,), dtype="f8"
    )
    index.attrs.update({"axis": "time", "field": "x:q", "kind": "metadata"})

    issues = _issues(validate_store(store_path, verify_integrity=False))

    assert (
        f"recordings/{name}",
        f"Recording name {name!r} must not contain control characters or "
        "lone surrogates",
    ) in issues
    assert (
        f"recordings/rec/indexes/{name}",
        "name must not contain control characters or lone surrogates",
    ) in issues
    assert len(issues) == 2


def _remove_node_document(
    store_path: Path, path: str, zarr_format: int
) -> None:
    """Delete a node's metadata document but keep its other keys.

    Args:
        store_path: Store root.
        path: Node path relative to the root.
        zarr_format: Physical Zarr format.
    """
    document = "zarr.json" if zarr_format == 3 else ".zgroup"
    (store_path / path / document).unlink()
    (store_path / path / ".zattrs").unlink(missing_ok=True)


@FORMATS
@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
def test_validation_reports_orphan_prefixes(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    """Keys under a resource name without node metadata are reported.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, zarr_format=zarr_format)
    store.recordings.create("rec", sample_shape=(4,))
    store.recordings.create("gone", sample_shape=(4,))
    store.collections.create("col", recording_ids=["rec"])
    _remove_node_document(store_path, "recordings/gone", zarr_format)
    _remove_node_document(store_path, "collections/col", zarr_format)

    assert _issues(validate_store(store_path)) == [
        (
            "collections/col",
            "keys exist under this name, but it has no Zarr node metadata",
        ),
        (
            "recordings/gone",
            "keys exist under this name, but it has no Zarr node metadata",
        ),
    ]


@FORMATS
@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (
            {"core:version": 7},
            "$.collection['core:version']: 7 is not of type 'string'",
        ),
        (
            {"core:description": ["a"]},
            "$.collection['core:description']: ['a'] is not of type 'string'",
        ),
    ],
)
def test_validation_applies_sigmf_collection_schema(
    tmp_path: Path,
    zarr_format: ZarrFormat,
    metadata: dict[str, Any],
    message: str,
) -> None:
    """Collection metadata that export would reject is reported.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
        metadata: Collection metadata.
        message: Expected issue detail.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, zarr_format=zarr_format)
    store.recordings.create("rec", sample_shape=(4,))
    store.collections.create("col", recording_ids=["rec"], metadata=metadata)

    prefix = "metadata does not satisfy the SigMF collection schema: "
    assert _issues(validate_store(store_path)) == [
        ("collections/col", prefix + message)
    ]


def test_collection_schema_permits_absent_version(tmp_path: Path) -> None:
    """Export supplies `core:version`, so a collection may omit it.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path)
    store.recordings.create("rec", sample_shape=(4,))
    store.collections.create(
        "col",
        recording_ids=["rec"],
        metadata={"core:description": "pair", "x:field": [1, 2]},
    )

    assert validate_store(store_path).issues == ()


@FORMATS
def test_collection_recording_ids_must_be_unique(
    tmp_path: Path, zarr_format: ZarrFormat
) -> None:
    """Writers reject, and validation reports, repeated recording IDs.

    Args:
        tmp_path: Pytest temporary path fixture.
        zarr_format: Physical Zarr format to create.
    """
    store_path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(store_path, zarr_format=zarr_format)
    store.recordings.create("rec", sample_shape=(4,))
    collection = store.collections.create("col", recording_ids=["rec"])
    message = "recording IDs must be unique"
    with pytest.raises(ValueError, match=message):
        store.collections.create("dup", recording_ids=["rec", "rec"])
    with pytest.raises(ValueError, match=message):
        collection.set_recording_ids(["rec", "rec"])
    _raw_group(store_path, "collections/col").attrs["recording_ids"] = [
        "rec",
        "rec",
    ]

    assert _issues(validate_store(store_path)) == [
        (
            "collections/col",
            "recording_ids must be unique, got ['rec'] more than once",
        )
    ]

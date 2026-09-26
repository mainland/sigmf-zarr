"""Named split creation and caller-managed group validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from zarr.core.array import Array

from sigmf_zarr import SigMFRecording, SigMFZarrStore, validate_store
from sigmf_zarr.store import ZarrFormat


@pytest.fixture(params=[2, 3])
def recording(
    tmp_path: Path, request: pytest.FixtureRequest
) -> SigMFRecording:
    """Create a recording with repeated source identities.

    Args:
        tmp_path: Temporary store directory.
        request: Physical format parameter.

    Returns:
        Six-item recording with scalar session IDs.
    """
    store = SigMFZarrStore.create(
        tmp_path / "splits.zarr", zarr_format=cast(ZarrFormat, request.param)
    )
    result = store.recordings.open(
        "rec", batched=True, sample_shape=(4,), sample_axes=("time",)
    )
    result.append_samples(np.zeros((6, 4), dtype=np.float32))
    result.add_index(
        "session", [0, 0, 1, 1, 2, 2], axis="item", field="example:session"
    )
    return result


@pytest.mark.parametrize("strings", [False, True])
def test_named_splits_round_trip(
    recording: SigMFRecording, strings: bool
) -> None:
    """Store explicit schemes with detached provenance and compact IDs.

    Args:
        recording: Example recording.
        strings: Whether assignments are label strings or integer IDs.
    """
    assignments = [0, 0, 1, 1, 2, 2]
    labels = ["train", "validation", "test"]
    supplied = [labels[i] for i in assignments] if strings else assignments
    split = recording.add_split(
        "splits/session",
        supplied,
        labels=labels,
        group_index="session",
        method="group_random",
        seed=42,
        split_type="holdout",
        generator="example 1",
        description="session isolation",
        chunks=(2,),
    )
    recording.add_split(
        "official", assignments, labels=labels, method="published"
    )
    assert recording.find_indexes("sigmf-zarr:split") == (
        "official",
        "splits/session",
    )
    assert split.assignments[:].tolist() == assignments
    assert split.assignments.dtype == np.dtype("uint8")
    assert split.labels == tuple(labels)
    assert split.provenance["seed"] == 42
    provenance = split.provenance
    provenance["seed"] = 0
    assert split.provenance["seed"] == 42
    with pytest.raises(TypeError):
        split.assignments[:] = 0
    reopened = recording._store.recordings.open("rec", create=False)
    reopened.validate_split("splits/session")
    assert reopened.decode_index("official", selection=[4, 0, 4]).tolist() == [
        "test",
        "train",
        "test",
    ]
    assert validate_store(recording._store, verify_integrity=False).valid


def test_source_edit_requires_explicit_revalidation(
    recording: SigMFRecording, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep assignments readable after a group-isolation claim becomes false.

    Args:
        recording: Example recording.
        monkeypatch: Payload-read guard fixture.
    """
    recording.add_split(
        "holdout",
        [0, 0, 1, 1, 1, 1],
        labels=["train", "test"],
        group_index="session",
    )
    with recording.mutate_index("session") as groups:
        groups[5] = 0

    def no_read(*args: object, **kwargs: object) -> None:
        """Reject payload reads during split-view construction."""
        raise AssertionError("Split view scanned assignments or sources")

    with monkeypatch.context() as patch:
        patch.setattr(Array, "__getitem__", no_read)
        view = recording.split("holdout")
        assert view.labels == ("train", "test")
        assert view.provenance["group_index"] == "session"
    assert view.assignments[:].tolist() == [0, 0, 1, 1, 1, 1]
    recording.update_integrity()
    before = recording.integrity
    with pytest.raises(ValueError, match="multiple partitions"):
        recording.validate_split("holdout")
    assert recording.integrity == before
    assert recording.verify_integrity()
    assert validate_store(recording._store, verify_integrity=False).valid
    with recording.mutate_index("session") as groups:
        groups[5] = 2
    view.validate()


@pytest.mark.parametrize(
    "assignments",
    [
        [-1, 0, 1, 1, 1, 1],
        [0, 0, 2, 1, 1, 1],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [False] * 6,
        [0, "test", 1, 1, 1, 1],
        ["other"] * 6,
        [0] * 6,
        [0, 1],
        [[0, 0, 0], [1, 1, 1]],
        np.array([0, 0, 1, 1, 1, 2**64 - 1], dtype=np.uint64),
    ],
)
def test_invalid_assignments_do_not_replace_existing_index(
    recording: SigMFRecording, assignments: Any
) -> None:
    """Preflight encoding and partition checks before replacement or hashing.

    Args:
        recording: Example recording.
        assignments: Malformed proposed assignment.
    """
    recording.add_split(
        "holdout", [0, 0, 1, 1, 1, 1], labels=["train", "test"]
    )
    recording.update_integrity()
    before = recording.integrity
    with pytest.raises(ValueError):
        recording.add_split(
            "holdout", assignments, labels=["train", "test"], overwrite=True
        )
    assert recording.split("holdout").assignments[:].tolist() == [
        0,
        0,
        1,
        1,
        1,
        1,
    ]
    assert recording.integrity == before


@pytest.mark.parametrize(
    "options",
    [
        {"labels": "ab"},
        {"labels": ["a"]},
        {"labels": ["a", "a"]},
        {"labels": ["a", ""]},
        {"labels": ["a", 1]},
        {"labels": ["a", "b", "unused"]},
        {"seed": -1},
        {"seed": True},
        {"split_type": "unknown"},
        {"method": "group_random"},
        {"method": "unknown"},
        {"group_index": ""},
        {"description": 1},
        {"group_index": "holdout"},
    ],
)
def test_invalid_descriptors_do_not_create_split(
    recording: SigMFRecording, options: dict[str, Any]
) -> None:
    """Reject malformed or unsupported provenance before creating storage.

    Args:
        recording: Example recording.
        options: Invalid public writer arguments.
    """
    arguments: dict[str, Any] = {"labels": ["a", "b"], **options}
    with pytest.raises(ValueError):
        recording.add_split("holdout", [0, 0, 1, 1, 1, 1], **arguments)
    assert "holdout" not in recording.indexes


@pytest.mark.parametrize("strings", [False, True])
def test_group_isolation_checked_before_write(
    recording: SigMFRecording, strings: bool
) -> None:
    """Reject assignments that split an integer or string group.

    Args:
        recording: Example recording.
        strings: Whether group identities are strings.
    """
    if strings:
        recording.add_index(
            "session",
            ["a", "a", "b", "b", "c", "c"],
            axis="item",
            field="example:session",
            overwrite=True,
        )
    with pytest.raises(ValueError, match="multiple partitions"):
        recording.add_split(
            "bad",
            [0, 1, 0, 1, 0, 1],
            labels=["train", "test"],
            group_index="session",
        )
    assert "bad" not in recording.indexes
    recording.add_split(
        "good",
        [0, 0, 1, 1, 1, 1],
        labels=["train", "test"],
        group_index="session",
    ).validate()


def test_generic_mutation_does_not_run_split_validation(
    recording: SigMFRecording,
) -> None:
    """Keep generic shape checks separate from domain-level partition checks.

    Args:
        recording: Example recording.
    """
    view = recording.add_split(
        "holdout", [0, 0, 1, 1, 1, 1], labels=["a", "b"]
    )
    with recording.mutate_index("holdout") as array:
        array[:] = 0
    assert view.assignments[:].tolist() == [0] * 6
    with pytest.raises(ValueError, match="empty partitions"):
        view.validate()
    with recording.mutate_index("holdout") as array:
        array[5] = 99
    with pytest.raises(ValueError, match="outside"):
        view.validate()


@pytest.mark.parametrize("attribute", ["validity"])
def test_deferred_features_are_not_silently_validated(
    recording: SigMFRecording, attribute: str
) -> None:
    """Leave generic indexes readable while rejecting unsupported split claims.

    Args:
        recording: Example recording.
        attribute: Descriptor belonging to a later stage.
    """
    recording.add_index(
        "future",
        [0, 0, 1, 1, 1, 1],
        axis="item",
        field="sigmf-zarr:split",
        kind="split",
        labels=["a", "b"],
        attributes={attribute: []},
    )
    assert recording.index("future")[0] == 0
    with pytest.raises(ValueError, match="not supported"):
        recording.validate_split("future")


def test_validation_carries_groups_across_batches(tmp_path: Path) -> None:
    """Detect overlap across validation batches with bounded array reads.

    Args:
        tmp_path: Temporary store directory.
    """
    store = SigMFZarrStore.create(tmp_path / "large.zarr")
    rec = store.recordings.open("rec", batched=True, sample_shape=(1,))
    rec.append_samples(np.zeros((65538, 1), dtype=np.float32))
    groups = np.arange(65538, dtype=np.int64)
    groups[-1] = 0
    rec.add_index("groups", groups, axis="item", field="example:groups")
    assignments = np.zeros(65538, dtype=np.uint8)
    assignments[-2:] = 1
    with pytest.raises(ValueError, match="multiple partitions"):
        rec.add_split(
            "holdout", assignments, labels=["a", "b"], group_index="groups"
        )


@pytest.mark.parametrize("defect", ["missing", "axis", "float", "invalid"])
def test_bad_group_sources_fail_before_creation(
    recording: SigMFRecording, defect: str
) -> None:
    """Validate referenced scalar group structure and dtype before a write.

    Args:
        recording: Example recording.
        defect: Group index failure mode.
    """
    name = "session"
    if defect == "missing":
        name = "absent"
    elif defect == "axis":
        recording.add_index(
            name,
            [0, 0, 1, 1],
            axis="time",
            field="example:session",
            overwrite=True,
        )
    elif defect == "float":
        recording.add_index(
            name,
            np.zeros(6, dtype=np.float64),
            axis="item",
            field="example:session",
            overwrite=True,
        )
    else:
        recording._indexes_group[name].attrs["sigmf-zarr:valid"] = False
    with pytest.raises((KeyError, ValueError)):
        recording.add_split(
            "bad", [0, 0, 1, 1, 1, 1], labels=["a", "b"], group_index=name
        )
    assert "bad" not in recording.indexes


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("method", []),
        ("split_type", {}),
        ("group_index", 1),
        ("labels", ["a", "a"]),
        ("kind", "metadata"),
    ],
)
def test_view_rejects_malformed_descriptors(
    recording: SigMFRecording, key: str, value: Any
) -> None:
    """Report malformed stored descriptors with a domain validation error.

    Args:
        recording: Example recording.
        key: Corrupted descriptor.
        value: Invalid replacement value.
    """
    view = recording.add_split(
        "holdout", [0, 0, 1, 1, 1, 1], labels=["a", "b"]
    )
    with recording.mutate_index("holdout") as array:
        array.attrs[key] = value
    with pytest.raises(ValueError):
        view.validate()
    assert recording.index("holdout")[0] == 0


def test_descriptor_only_views_do_not_accept_float_ids(
    recording: SigMFRecording,
) -> None:
    """Reject noninteger assignment storage even without a value scan.

    Args:
        recording: Example recording.
    """
    recording.add_index(
        "float",
        np.zeros(6),
        axis="item",
        field="sigmf-zarr:split",
        kind="split",
        labels=["a", "b"],
    )
    with pytest.raises(ValueError, match="integer dtype"):
        recording.split("float")

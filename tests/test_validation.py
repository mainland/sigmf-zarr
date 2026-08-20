"""Tests for complete-store validation and integrity CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sigmf_zarr.cli.sigmf import SigMFCommand
from sigmf_zarr.integrity import INTEGRITY_ATTR
from sigmf_zarr.json import json_object, json_object_list, json_value
from sigmf_zarr.store import SigMFZarrStore
from sigmf_zarr.store._common import ZarrFormat
from sigmf_zarr.validation import validate_store


def _validation_store(tmp_path) -> tuple[object, SigMFZarrStore]:
    """Create a small conforming store for validation tests.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Store path and opened writable store.
    """
    store_path = tmp_path / "validation.zarr"
    store = SigMFZarrStore.create(store_path, overwrite=True)
    recording = store.recordings.open(
        "rec",
        create=True,
        sample_shape=(4,),
        sample_axes=("time",),
        global_metadata={"core:datatype": "rf32_le"},
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


def test_validate_store_accepts_complete_hashed_store(tmp_path) -> None:
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


def test_validate_store_reports_dtype_and_index_corruption(tmp_path) -> None:
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


def test_validation_and_integrity_commands_support_json(
    tmp_path, capsys
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
    tmp_path,
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
        issue.path == "recordings/rec"
        and "do not match" in issue.message
        for issue in stale.issues
    )
    assert stale.as_dict()["valid"] is False

    integrity = dict(recording.integrity)
    integrity["version"] = 999
    recording._raw_group.attrs[INTEGRITY_ATTR] = integrity

    unsupported = validate_store(store)

    assert any(
        issue.path == "recordings/rec"
        and "unsupported" in issue.message
        for issue in unsupported.issues
    )

    recording._raw_group.attrs.pop(INTEGRITY_ATTR)
    missing = validate_store(store, require_integrity=True)

    assert any(
        issue.path == "recordings/rec" and "required" in issue.message
        for issue in missing.issues
    )


def test_validation_reports_missing_collection_recordings(tmp_path) -> None:
    """Collection references and invalid store indexes should be reported.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    _, store = _validation_store(tmp_path)
    store.collections.open(
        "broken",
        create=True,
        recording_ids=("missing",),
    )
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
    with (
        pytest.raises(RuntimeError, match="incomplete"),
        store.mutate_index("nested/fold") as index,
    ):
        index[0] = 3
        raise RuntimeError("incomplete")

    report = validate_store(store)
    assert not report.valid
    assert report.indexes_checked == 1
    assert report.issues[0].path == "indexes/nested/fold"
    with pytest.raises(ValueError, match="nested/fold.*invalid"):
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
    collection = store.collections.open("all", create=True)
    with (
        pytest.raises(ValueError, match=f"missing required '{field}'"),
        collection.mutate() as group,
    ):
        del group.attrs[field]
    report = validate_store(store)
    assert not report.valid
    assert report.collections_checked == 1
    assert report.issues[0].path == "collections/all"
    assert field in report.issues[0].message

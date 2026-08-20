"""End-to-end tests for store inspection, integrity, and export commands."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sigmf_zarr import SigMFZarrStore
from sigmf_zarr.cli.sigmf import SigMFCommand


def run(arguments: list[str]) -> int:
    """Run the command line in process.

    Args:
        arguments: Command-line arguments after the program name.

    Returns:
        Command exit status.
    """
    return int(SigMFCommand().run(arguments) or 0)


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    """Create a store with three recordings and one collection.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Store path.
    """
    path = tmp_path / "store.zarr"
    store = SigMFZarrStore.create(path)
    for name in ("a", "b", "c"):
        recording = store.recordings.create(
            name,
            sample_shape=(4,),
            global_metadata={"core:datatype": "rf32_le"},
            captures=[{"core:sample_start": 0}],
        )
        recording.set_samples(np.arange(4, dtype=np.float32))
    store.collections.create("pair", recording_ids=["a", "b"])
    return path


def test_listing_and_info_commands(
    store_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Listing and information commands report store resources.

    Args:
        store_path: Example store path.
        capsys: Pytest capture fixture.
    """
    assert run(["store", str(store_path), "recordings"]) == 0
    assert capsys.readouterr().out.split() == ["a", "b", "c"]
    assert run(["store", str(store_path), "collections"]) == 0
    assert capsys.readouterr().out.split() == ["pair"]

    assert run(["store", str(store_path), "recording", "a", "info"]) == 0
    assert "name='a'" in capsys.readouterr().out
    assert run(["store", str(store_path), "info", "--format", "json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["recordings"] == ["a", "b", "c"]

    assert (
        run(
            [
                "store",
                str(store_path),
                "collection",
                "pair",
                "info",
                "--format",
                "json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["recording_ids"] == ["a", "b"]


def test_missing_recording_is_reported_by_name(
    store_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing resource exits with a readable command-line error.

    Args:
        store_path: Example store path.
        capsys: Pytest capture fixture.
    """
    with pytest.raises(SystemExit) as exit_info:
        run(["store", str(store_path), "recording", "missing", "info"])
    assert exit_info.value.code == 2
    assert "not found: 'missing'" in capsys.readouterr().err


def test_integrity_commands_update_and_verify(
    store_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verification fails before an update and succeeds after it.

    Args:
        store_path: Example store path.
        capsys: Pytest capture fixture.
    """
    assert run(["store", str(store_path), "integrity", "verify"]) == 1
    assert "integrity metadata is required" in capsys.readouterr().out
    assert run(["store", str(store_path), "integrity", "update"]) == 0
    assert run(["store", str(store_path), "integrity", "verify"]) == 0
    assert capsys.readouterr().out.startswith("valid=True")


def test_validate_skip_integrity_ignores_declared_hashes(
    store_path: Path,
) -> None:
    """Structural validation does not verify a stale `core:sha512`.

    Args:
        store_path: Example store path.
    """
    store = SigMFZarrStore.open(store_path, mode="r+")
    store.recordings["a"].set_global_field("core:sha512", "0" * 128)
    assert run(["store", str(store_path), "validate"]) == 1
    assert run(["store", str(store_path), "validate", "--skip-integrity"]) == 0

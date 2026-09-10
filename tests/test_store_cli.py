"""End-to-end tests for store inspection, integrity, and export commands."""

from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path

import numpy as np
import pytest

from sigmf_zarr import SigMFZarrStore
from sigmf_zarr.cli.sigmf import SigMFCommand
from sigmf_zarr.store import ZarrFormat


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
    for kind, command in (
        ("recording", "info"),
        ("recording", "export"),
        ("recording", "validate-rfml"),
        ("collection", "info"),
        ("collection", "export"),
    ):
        argv = ["store", str(store_path), kind, "missing", command]
        if command == "export":
            argv.append(str(store_path.parent / "out"))
        with pytest.raises(SystemExit) as exit_info:
            run(argv)
        assert exit_info.value.code == 2
        err = capsys.readouterr().err
        assert f"error: {kind} not found: 'missing'" in err
        # The usage line belongs to the selected subcommand.
        assert f" {kind} " in err.splitlines()[0]


def test_untrusted_names_cannot_forge_output_lines(
    store_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Line feeds in names print as escapes, never as new output lines.

    Args:
        store_path: Example store path.
        capsys: Pytest capture fixture.
    """
    forged = "a\nforged-recording"
    recordings = store_path / "recordings"
    (recordings / "c").rename(recordings / forged)
    assert run(["store", str(store_path), "recordings"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "a",
        "a\\nforged-recording",
        "b",
    ]


def test_option_conflict_uses_subcommand_usage(
    store_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An option error prints the usage line of the selected subcommand.

    Args:
        store_path: Example store path.
        capsys: Pytest capture fixture.
    """
    with pytest.raises(SystemExit) as exit_info:
        run(["export", str(store_path), str(store_path.parent / "out")])
    assert exit_info.value.code == 2
    usage, message = capsys.readouterr().err.split("sigmf-zarr: error: ")
    assert usage.startswith("usage: sigmf-zarr export ")
    assert "--recording-name" in usage
    assert message.startswith("--recording-name is required")


def test_export_errors_name_command_line_options(
    store_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Errors name command-line options rather than Python parameters.

    Args:
        store_path: Example store path.
        tmp_path: Pytest temporary path fixture.
        capsys: Pytest capture fixture.
    """
    output = str(tmp_path / "a.sigmf-meta")
    export = ["store", str(store_path), "recording", "a", "export", output]
    assert run(export) == 0
    capsys.readouterr()
    with pytest.raises(SystemExit):
        run(export)
    err = capsys.readouterr().err
    assert "Pass --overwrite " in err
    assert "overwrite=True" not in err
    with pytest.raises(SystemExit):
        run([*export, "--overwrite", "--item-index", "0"])
    err = capsys.readouterr().err
    assert "error: --item-index " in err
    assert "item_index" not in err


def test_collection_export_contains_only_its_recordings(
    store_path: Path, tmp_path: Path
) -> None:
    """The collection export command archives only collection members.

    Args:
        store_path: Example store path.
        tmp_path: Pytest temporary path fixture.
    """
    output = tmp_path / "pair.sigmf"
    assert (
        run(
            [
                "store",
                str(store_path),
                "collection",
                "pair",
                "export",
                str(output),
            ]
        )
        == 0
    )
    with tarfile.open(output) as archive:
        names = {Path(name).name for name in archive.getnames()}
    assert "c.sigmf-meta" not in names
    assert {"a.sigmf-meta", "b.sigmf-meta", "pair.sigmf-collection"} <= names


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


def test_archive_import_command_round_trips_collection(
    store_path: Path, tmp_path: Path
) -> None:
    """Archive import through the command line restores a collection.

    Args:
        store_path: Example store path.
        tmp_path: Pytest temporary path fixture.
    """
    archive = tmp_path / "pair.sigmf"
    run(
        [
            "store",
            str(store_path),
            "collection",
            "pair",
            "export",
            str(archive),
        ]
    )
    imported = tmp_path / "imported.zarr"
    assert run(["import", "sigmf", str(archive), str(imported)]) == 0
    store = SigMFZarrStore.open(imported)
    assert store.list_recordings() == ("a", "b")
    assert store.collections["pair"].recording_ids == ("a", "b")
    assert store.verify_integrity()


SPOOFED_NAME = (
    "\r\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K"
    "valid=True, recordings=1, collections=0, indexes=0\x1b[8m"
)
"""Recording name that rewrites earlier terminal lines when printed raw."""


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_text_output_escapes_control_characters(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], zarr_format: ZarrFormat
) -> None:
    """Names with terminal control sequences print as escapes.

    Writers reject such names, so the test renames a recording as another
    tool could.

    Args:
        tmp_path: Pytest temporary path fixture.
        capsys: Pytest capture fixture.
        zarr_format: Zarr format of the store.
    """
    path = tmp_path / "store.zarr"
    with SigMFZarrStore.create(path, zarr_format=zarr_format) as store:
        recording = store.recordings.create(
            "plain",
            sample_shape=(4,),
            global_metadata={"core:datatype": "rf32_le"},
        )
        recording.set_samples(np.zeros(4, dtype=np.float32))
    recordings = path / "recordings"
    (recordings / "plain").rename(recordings / SPOOFED_NAME)
    escaped = "\\r\\x1b[2K\\x1b[1A"

    assert run(["store", str(path), "integrity", "verify"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("valid=False")
    assert escaped in out
    assert "\x1b" not in out
    assert "\r" not in out

    assert run(["store", str(path), "recordings"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(escaped)
    assert "\x1b" not in out

    assert run(["store", str(path), "info"]) == 0
    assert "\x1b" not in capsys.readouterr().out


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_undecodable_recording_name_prints_escaped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], zarr_format: ZarrFormat
) -> None:
    """A directory name that is not UTF-8 prints instead of failing.

    Args:
        tmp_path: Pytest temporary path fixture.
        capsys: Pytest capture fixture.
        zarr_format: Zarr format of the store.
    """
    path = tmp_path / "store.zarr"
    with SigMFZarrStore.create(path, zarr_format=zarr_format) as store:
        recording = store.recordings.create(
            "plain",
            sample_shape=(4,),
            global_metadata={"core:datatype": "rf32_le"},
        )
        recording.set_samples(np.zeros(4, dtype=np.float32))
    recordings = path / "recordings"
    try:
        (recordings / "plain").rename(recordings / os.fsdecode(b"bad\xff"))
    except UnicodeEncodeError:
        pytest.skip("The file system does not accept undecodable names")

    assert run(["store", str(path), "recordings"]) == 0
    assert capsys.readouterr().out.split() == ["bad\\udcff"]
    assert run(["store", str(path), "integrity", "verify"]) == 1
    assert "recordings/bad\\udcff: " in capsys.readouterr().out


def test_recording_json_counts_samples_and_items(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Recording JSON reports time samples and items separately.

    Args:
        tmp_path: Pytest temporary path fixture.
        capsys: Pytest capture fixture.
    """
    path = tmp_path / "store.zarr"
    with SigMFZarrStore.create(path) as store:
        single = store.recordings.create(
            "single",
            sample_shape=(400,),
            global_metadata={"core:datatype": "rf32_le"},
        )
        single.set_samples(np.zeros(400, dtype=np.float32))
        batch = store.recordings.create(
            "batch",
            batched=True,
            sample_shape=(5,),
            global_metadata={"core:datatype": "rf32_le"},
        )
        batch.append_samples(np.zeros((3, 5), dtype=np.float32))

    counts = {}
    for name in ("single", "batch"):
        arguments = ["store", str(path), "recording", name, "info"]
        assert run([*arguments, "--format", "json"]) == 0
        result = json.loads(capsys.readouterr().out)
        assert "num_samples" not in result
        counts[name] = (result["sample_count"], result["item_count"])
    assert counts == {"single": (400, 1), "batch": (5, 3)}

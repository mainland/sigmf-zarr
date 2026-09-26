"""Export options reach the converter through both command-line forms."""

import json
from pathlib import Path

import numpy as np
import pytest

from sigmf_zarr.cli.sigmf import SigMFCommand
from sigmf_zarr.store import SigMFZarrStore


@pytest.mark.parametrize("resource_form", [False, True])
def test_cli_exports_selected_item_with_explicit_loss(
    tmp_path: Path, resource_form: bool
) -> None:
    """Exercise item selection and loss warnings through the public parser.

    Args:
        tmp_path: Temporary directory fixture.
        resource_form: Whether to use the resource-oriented export command.
    """
    source = tmp_path / "source.zarr"
    with SigMFZarrStore.create(source) as store:
        recording = store.recordings.open(
            "rec", batched=True, sample_shape=(4,), sample_axes=("time",),
            global_metadata={"core:datatype": "rf32_le"},
        )
        recording.append_samples(
            np.arange(8, dtype=np.float32).reshape(2, 4),
            item_captures=[None, [{"core:datetime": "2026-01-01T00:00:00Z"}]],
        )
        recording.add_index("label", [1, 2], axis="item", field="test:label")
    output = tmp_path / "item.sigmf-meta"
    args = (
        ["store", str(source), "recording", "rec", "export", str(output)]
        if resource_form else
        ["export", str(source), str(output), "--recording-name", "rec"]
    )
    with pytest.warns(UserWarning, match="omits indexes"):
        assert SigMFCommand().run(
            [*args, "--item-index", "1", "--allow-lossy"]
        ) == 0
    np.testing.assert_array_equal(
        np.fromfile(output.with_suffix(".sigmf-data"), dtype="<f4"),
        np.arange(4, 8, dtype=np.float32),
    )
    assert json.loads(output.read_text())["captures"] == [{
        "core:sample_start": 0, "core:datetime": "2026-01-01T00:00:00Z"
    }]


@pytest.mark.parametrize("resource_form", [False, True])
def test_cli_export_preflight_and_projection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], resource_form: bool,
) -> None:
    """Both CLI forms report and preserve explicitly selected target indexes.

    Args:
        tmp_path: Temporary directory.
        capsys: Captured command output.
        resource_form: Whether to use resource-oriented syntax.
    """
    source = tmp_path / "source.zarr"
    with SigMFZarrStore.create(source) as store:
        recording = store.recordings.open(
            "rec", batched=True, sample_shape=(4,), sample_axes=("time",),
            global_metadata={"core:datatype": "rf32_le"},
        )
        recording.append_samples(np.ones((1, 4)))
        recording.add_index("label", [0], axis="item", field="test:label")
    output = tmp_path / "item.sigmf-meta"
    args = (
        ["store", str(source), "recording", "rec", "export", str(output)]
        if resource_form else
        ["export", str(source), str(output), "--recording-name", "rec"]
    )
    args += ["--item-index", "0", "--project-index", "label"]
    assert SigMFCommand().run([*args, "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["lossless"]
    assert not output.exists()
    assert SigMFCommand().run(args) == 0
    assert output.exists()

"""RFML class adoption without changing source SNR or duplicating indexes."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from sigmf_zarr import (
    SigMFZarrStore,
    import_panoradio_dataset,
    import_radioml2016_dataset,
    import_radioml2018_dataset,
)
from sigmf_zarr.json import JSONObject


@pytest.fixture(params=["2016", "2022", "2018", "panoradio"])
def importer(
    tmp_path: Path, request: pytest.FixtureRequest
) -> tuple[str, Callable[..., SigMFZarrStore]]:
    """Prepare tiny source datasets for each adopting importer.

    Args:
        tmp_path: Temporary source directory.
        request: Importer family parameter.

    Returns:
        Source family and configured import function.
    """
    kind = request.param
    if kind in ("2016", "2022"):
        dataset = {
            ("QPSK", 4): np.zeros((1, 2, 8), dtype=np.float32),
            ("BPSK", -2): np.ones((2, 2, 8), dtype=np.float32),
        }
        return kind, partial(
            import_radioml2016_dataset,
            dataset=dataset,
            dataset_version=kind,
            recording_name="rec",
        )
    if kind == "2018":
        source = tmp_path / "source.h5"
        with h5py.File(source, "w") as handle:
            handle["X"] = np.zeros((3, 8, 2), dtype=np.float32)
            handle["Y"] = np.eye(2, dtype=np.uint8)[[1, 0, 1]]
            handle["Z"] = np.array([-2, 4, -2], dtype=np.int16)
        return kind, partial(
            import_radioml2018_dataset,
            source_path=source,
            modulation_classes=("QPSK", "BPSK"),
            recording_name="rec",
        )
    source = tmp_path / "source.npy"
    np.save(source, np.zeros((3, 8), dtype=np.complex64))
    tags = tmp_path / "tags.csv"
    tags.write_text(
        "idx,mode,snr\n0,psk31,-2\n1,USB,4\n2,psk31,-2\n", encoding="ascii"
    )
    return kind, partial(
        import_panoradio_dataset,
        source_path=source,
        tags_path=tags,
        recording_name="rec",
    )


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_import_declares_profile_and_preserves_values(
    tmp_path: Path,
    importer: tuple[str, Callable[..., SigMFZarrStore]],
    zarr_format: int,
) -> None:
    """Adopt class meanings while preserving source metadata and raw IDs.

    Args:
        tmp_path: Destination directory.
        importer: Configured source importer.
        zarr_format: Physical Zarr format.
    """
    kind, import_dataset = importer
    metadata: JSONObject = {
        "core:description": "source description",
        "core:extensions": [
            {"name": "example", "version": "2.0.0", "optional": False},
            {"name": "rfml-dataset", "version": "0.0.1", "optional": False},
            {"name": "rfml-dataset", "version": "0.1.0", "optional": True},
        ],
    }
    original = deepcopy(metadata)
    store = import_dataset(
        tmp_path / "output.zarr",
        global_metadata=metadata,
        zarr_format=zarr_format,
    )
    assert metadata == original
    recording = store.recordings["rec"]
    declarations = recording.global_metadata["core:extensions"]
    assert [
        entry for entry in declarations if entry["name"] == "rfml-dataset"
    ] == [{"name": "rfml-dataset", "version": "0.1.0", "optional": True}]
    assert {
        "name": "example",
        "version": "2.0.0",
        "optional": False,
    } in declarations
    assert any(entry["name"] == "sigmf-zarr" for entry in declarations)
    assert (
        recording.global_metadata["core:description"] == "source description"
    )
    name = "mode_id" if kind == "panoradio" else "mod_class_id"
    field = (
        "rfml-dataset:class"
        if kind == "panoradio"
        else "rfml-dataset:modulation"
    )
    assert recording.find_indexes(field, axis="item") == (name,)
    assert recording.find_indexes("radioml:mod_class") == ()
    assert recording.find_indexes("panoradio:mode") == ()
    assert set(recording.indexes.array_keys()) == {name, "snr_db"}
    index = recording.index(name)
    if kind in ("2016", "2022"):
        assert index[:].tolist() == [0, 0, 1]
        assert index.attrs["labels"] == ["BPSK", "QPSK"]
        assert recording.index("snr_db")[:].tolist() == [-2, -2, 4]
    else:
        assert index[:].tolist() == [1, 0, 1]
        assert index.attrs["labels"] == (
            ["USB", "psk31"] if kind == "panoradio" else ["QPSK", "BPSK"]
        )
        assert recording.index("snr_db")[:].tolist() == [-2, 4, -2]
    snr = recording.index("snr_db")
    assert snr.attrs["field"] == (
        "panoradio:snr" if kind == "panoradio" else "radioml:snr"
    )
    assert snr.attrs["unit"] == "dB"
    assert "measurement" not in snr.attrs
    assert recording.find_indexes("rfml-dataset:snr") == ()
    assert recording.has_item_metadata is False
    assert store.verify_integrity()


@pytest.mark.parametrize("extensions", [None, "bad", [1]])
def test_invalid_declaration_fails_before_store_creation(
    tmp_path: Path,
    importer: tuple[str, Callable[..., SigMFZarrStore]],
    extensions: Any,
) -> None:
    """Reject malformed metadata before an importer creates its destination.

    Args:
        tmp_path: Destination directory.
        importer: Configured source importer.
        extensions: Malformed extension list.
    """
    _, import_dataset = importer
    destination = tmp_path / "absent.zarr"
    with pytest.raises(ValueError, match="core:extensions"):
        import_dataset(
            destination, global_metadata={"core:extensions": extensions}
        )
    assert not destination.exists()


@pytest.mark.parametrize("dataset_version", ["2016", "2022"])
def test_empty_modulation_is_rejected(
    tmp_path: Path, dataset_version: str
) -> None:
    """Refuse an empty profile class name before writing samples.

    Args:
        tmp_path: Destination directory.
        dataset_version: Pickle dataset generation.
    """
    destination = tmp_path / "absent.zarr"
    with pytest.raises(ValueError, match="nonempty"):
        import_radioml2016_dataset(
            destination,
            {("", 0): np.zeros((1, 2, 8), dtype=np.float32)},
            dataset_version=dataset_version,
        )
    assert not destination.exists()

"""RadioML HDF5 component precision determines its SigMF encoding."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from sigmf_zarr.json import json_object
from sigmf_zarr.provenance import file_identity
from sigmf_zarr.radioml2018 import import_radioml2018_dataset


@pytest.mark.parametrize("dtype", ["<f4", ">f8"])
def test_radioml2018_declares_component_precision(
    tmp_path: Path,
    dtype: str,
) -> None:
    """Retain precision while declaring a portable interchange byte order.

    Args:
        tmp_path: Temporary directory.
        dtype: HDF5 component dtype.
    """
    source = tmp_path / "source.h5"
    with h5py.File(source, "w") as handle:
        handle["X"] = np.ones((1, 8, 2), dtype=dtype)
        handle["Y"] = np.ones((1, 1), dtype="u1")
        handle["Z"] = np.zeros(1, dtype="i2")
    with import_radioml2018_dataset(
        tmp_path / "rml.zarr", source, modulation_classes=("BPSK",)
    ) as store:
        recording = store.recordings["radioml2018"]
        provenance = json_object(
            recording.global_metadata["sigmf-zarr-provenance:import"],
            name="provenance",
        )
        assert provenance["sources"] == [
            file_identity(source, role="samples-and-truth")
        ]
        parameters = json_object(provenance["parameters"], name="parameters")
        assert parameters["source_rows"] == [0, 1]
        assert recording.global_metadata["core:datatype"] == (
            f"cf{np.dtype(dtype).itemsize * 8}_le"
        )
    with pytest.raises(ValueError, match="core:datatype"):
        import_radioml2018_dataset(
            tmp_path / "bad.zarr",
            source,
            modulation_classes=("BPSK",),
            global_metadata={"core:datatype": "rf32_le"},
        )
    assert not (tmp_path / "bad.zarr").exists()


def test_radioml2018_big_endian_source_has_one_logical_dtype(
    tmp_path: Path,
) -> None:
    """Big-endian HDF5 samples import identically in both Zarr formats.

    Args:
        tmp_path: Temporary directory.
    """
    components = np.array(
        [[[1.5, -2.0], [np.pi, 1e-40], [0.0, -0.0], [3.0, 4.0]]],
        dtype=">f4",
    )
    source = tmp_path / "source.h5"
    with h5py.File(source, "w") as handle:
        handle["X"] = components
        handle["Y"] = np.ones((1, 1), dtype="u1")
        handle["Z"] = np.zeros(1, dtype="i2")
    metadata_hashes = set()
    for zarr_format in (2, 3):
        with import_radioml2018_dataset(
            tmp_path / f"rml-{zarr_format}.zarr",
            source,
            modulation_classes=("BPSK",),
            zarr_format=zarr_format,
        ) as store:
            recording = store.recordings["radioml2018"]
            assert recording.global_metadata["sigmf-zarr:dtype"] == "<f4"
            assert np.dtype(recording.samples.dtype).str == "<f4"
            assert recording.global_metadata["core:datatype"] == "cf32_le"
            np.testing.assert_array_equal(
                recording.samples[:], components.transpose(0, 2, 1)
            )
            metadata_hashes.add(recording.calculate_metadata_sha512())
    assert len(metadata_hashes) == 1

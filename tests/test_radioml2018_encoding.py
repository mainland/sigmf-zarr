"""RadioML HDF5 component precision determines its SigMF encoding."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from sigmf_zarr.provenance import file_identity
from sigmf_zarr.radioml2018 import import_radioml2018_dataset


@pytest.mark.parametrize("dtype", ["<f4", ">f8"])
def test_radioml2018_declares_component_precision(
    tmp_path: Path, dtype: str,
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
        provenance = recording.global_metadata["sigmf-zarr-provenance:import"]
        assert provenance["sources"] == [
            file_identity(source, role="samples-and-truth")
        ]
        assert provenance["parameters"]["source_rows"] == [0, 1]
        assert recording.global_metadata["core:datatype"] == (
            f"cf{np.dtype(dtype).itemsize * 8}_le"
        )
    with pytest.raises(ValueError, match="core:datatype"):
        import_radioml2018_dataset(
            tmp_path / "bad.zarr", source, modulation_classes=("BPSK",),
            global_metadata={"core:datatype": "rf32_le"},
        )
    assert not (tmp_path / "bad.zarr").exists()

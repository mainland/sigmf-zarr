"""Export preserves values or rejects an incompatible declared encoding."""

from pathlib import Path

import numpy as np
import pytest

from sigmf_zarr.sigmf import export_sigmf
from sigmf_zarr.store import SigMFZarrStore


@pytest.mark.parametrize(
    ("datatype", "samples", "axes", "message"),
    [
        ("ri8", [0.5, 128, -129], ("time",), "cannot be represented"),
        ("rf32_le", [1.0 + 2**-40], ("time",), "losslessly"),
        ("rf32_le", [[1, 2], [3, 4]], ("iq", "time"), "complexity"),
        ("cf32_le", [1, 2], ("time",), "complexity"),
        ("rf64_le", np.array([2**53 + 1], dtype="i8"), ("time",),
         "losslessly"),
    ],
)
def test_export_rejects_sample_conversion(
    tmp_path: Path, datatype: str, samples: object,
    axes: tuple[str, ...], message: str,
) -> None:
    """Reject numerical loss even when metadata omission is authorized.

    Args:
        tmp_path: Temporary directory.
        datatype: Requested interchange encoding.
        samples: Values incompatible with that encoding.
        axes: Sample coordinates.
        message: Expected diagnostic.
    """
    values = np.asarray(samples)
    with SigMFZarrStore.create(tmp_path / "source.zarr") as store:
        recording = store.recordings.open(
            "rec", sample_dtype=values.dtype, sample_shape=values.shape,
            sample_axes=axes, global_metadata={"core:datatype": datatype},
        )
        with recording.mutate_samples() as destination:
            destination[:] = values
        with pytest.raises(ValueError, match=message):
            export_sigmf(store, "rec", tmp_path / "out", allow_lossy=True)
        assert not (tmp_path / "out.sigmf-data").exists()
        assert not (tmp_path / "out.sigmf-meta").exists()


def test_exact_integer_encoding_from_float_components(tmp_path: Path) -> None:
    """Allow a different memory dtype when every value is exactly
    representable.

    Args:
        tmp_path: Temporary directory.
    """
    values = np.array([-128.0, -0.0, 127.0])
    with SigMFZarrStore.create(tmp_path / "source.zarr") as store:
        recording = store.recordings.open(
            "rec", sample_dtype="f8", sample_shape=(3,),
            sample_axes=("time",), global_metadata={"core:datatype": "ri8"},
        )
        with recording.mutate_samples() as destination:
            destination[:] = values
        output = export_sigmf(store, "rec", tmp_path / "out")
        assert output.with_suffix(".sigmf-data").read_bytes() == bytes(
            [128, 0, 127]
        )

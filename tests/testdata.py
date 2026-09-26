"""Small, independently defined SigMF fixtures used by the test suite."""

import io
import json
import tarfile
from pathlib import Path

import numpy as np
from sigmf import SigMFFile, __specification__

TEST_FLOAT32_DATA = np.arange(16, dtype=np.float32)
TEST_METADATA = {
    SigMFFile.ANNOTATION_KEY: [
        {"core:sample_count": 16, "core:sample_start": 0}
    ],
    SigMFFile.CAPTURE_KEY: [{"core:sample_start": 0}],
    SigMFFile.GLOBAL_KEY: {
        "core:datatype": "rf32_le",
        "core:sha512": (
            "f4984219b318894fa7144519185d1ae81ea721c6113243a52b51e444512a"
            "39d74cf41a4cec3c5d000bd7277cc71232c04d7a946717497e18619bdbe"
            "94bfeadd6"
        ),
        "core:num_channels": 1,
        "core:offset": 0,
        "core:version": __specification__,
    },
}


INVALID_SIGMF_SOURCES = (
    "corrupt-archive",
    "traversal-archive",
    "absolute-link-archive",
    "hash-mismatch",
    "unknown-datatype",
    "missing-dataset",
)
"""Kinds of untrusted SigMF input that import must reject."""


def write_invalid_sigmf_source(directory: Path, kind: str) -> Path:
    """Write one invalid SigMF recording or archive.

    Args:
        directory: Existing directory that receives the source files.
        kind: One of `INVALID_SIGMF_SOURCES`.

    Returns:
        Path of the `.sigmf` archive or `.sigmf-meta` file.

    Raises:
        ValueError: If `kind` is unknown.
    """
    if kind.endswith("-archive"):
        archive = directory / "source.sigmf"
        if kind == "corrupt-archive":
            archive.write_bytes(b"not a tar archive\n" * 64)
            return archive
        with tarfile.open(archive, "w") as output:
            if kind == "traversal-archive":
                member = tarfile.TarInfo("../escape.sigmf-meta")
                member.size = 2
                output.addfile(member, io.BytesIO(b"{}"))
            elif kind == "absolute-link-archive":
                member = tarfile.TarInfo("source/link.sigmf-meta")
                member.type = tarfile.SYMTYPE
                member.linkname = "/etc/passwd"
                output.addfile(member)
            else:
                raise ValueError(f"Unknown invalid SigMF source: {kind}")
        return archive
    global_info: dict[str, object] = {
        "core:datatype": "rf32_le",
        "core:version": __specification__,
    }
    if kind == "hash-mismatch":
        global_info["core:sha512"] = "0" * 128
    elif kind == "unknown-datatype":
        global_info["core:datatype"] = "ci12_le"
    elif kind == "missing-dataset":
        global_info["core:dataset"] = "absent.bin"
    else:
        raise ValueError(f"Unknown invalid SigMF source: {kind}")
    meta = directory / "source.sigmf-meta"
    meta.write_text(
        json.dumps(
            {
                "global": global_info,
                "captures": [{"core:sample_start": 0}],
                "annotations": [],
            }
        ),
        encoding="utf-8",
    )
    if kind != "missing-dataset":
        meta.with_suffix(".sigmf-data").write_bytes(
            TEST_FLOAT32_DATA.tobytes()
        )
    return meta

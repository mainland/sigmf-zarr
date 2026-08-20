"""Types shared by SigMF-Zarr store implementations."""

from typing import Literal

type ChecksumName = Literal["crc32c"]
"""Supported chunk-level sample checksum names."""

type ZarrFormat = Literal[2, 3]
"""Supported physical Zarr storage formats."""

__all__ = ["ChecksumName", "ZarrFormat"]

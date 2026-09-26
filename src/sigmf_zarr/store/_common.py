"""Types shared by SigMF-Zarr store implementations."""

from typing import Literal

type ChecksumName = Literal["crc32c"]
"""Supported chunk-level sample checksum names."""

type ZarrFormat = Literal[2, 3]
"""Supported physical Zarr storage formats."""

DEFAULT_SAMPLE_CHUNK_BYTES = 1024 * 1024
"""Target logical chunk size for recordings created without explicit chunks."""


def _validate_resource_name(name: str, *, kind: str) -> None:
    """Require a resource identifier to name one child group.

    Args:
        name: Recording or collection identifier.
        kind: Resource kind used in the error message.

    Raises:
        ValueError: If the identifier is empty or contains path components.
    """
    # Untyped callers can pass any value.
    if (
        not isinstance(name, str)  # type: ignore[redundant-expr]
        or not name
        or name in {".", ".."}
        or any(character in name for character in ("/", "\\", "\x00"))
    ):
        raise ValueError(
            f"{kind} name must be a single nonempty Zarr group name"
        )


__all__ = ["ChecksumName", "ZarrFormat"]

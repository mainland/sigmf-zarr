"""Tests for Chad Spooner's ``.tim`` reader."""

from __future__ import annotations

import struct

import numpy as np

from sigmf_zarr.tim import decode_tim, read_tim


def _tim_bytes(
    samples: np.ndarray,
    *,
    byte_order: str,
) -> bytes:
    """Encode test samples using Spooner's binary layout.

    Args:
        samples: One-dimensional real or complex samples.
        byte_order: ``struct`` and NumPy byte-order prefix.

    Returns:
        Encoded test file contents.
    """
    values = np.asarray(samples)
    real_complex = 2 if np.iscomplexobj(values) else 1
    if real_complex == 2:
        components = np.empty((values.size * 2,), dtype=np.float32)
        components[0::2] = values.real
        components[1::2] = values.imag
    else:
        components = np.asarray(values, dtype=np.float32)
    payload = components.astype(f"{byte_order}f4").tobytes()
    return struct.pack(
        f"{byte_order}ii",
        real_complex,
        values.size,
    ) + payload


def test_decode_tim_auto_detects_little_endian_complex_samples() -> None:
    """The decoder should demultiplex an interleaved complex payload."""
    expected = np.array([1 + 2j, -3 + 4.5j], dtype=np.complex64)

    decoded = decode_tim(_tim_bytes(expected, byte_order="<"))

    assert decoded.dtype == np.dtype(np.complex64)
    np.testing.assert_array_equal(decoded, expected)


def test_decode_tim_auto_detects_big_endian_real_samples() -> None:
    """The decoder should support big-endian real files."""
    expected = np.array([1.25, -2.5, 3.75], dtype=np.float32)

    decoded = decode_tim(_tim_bytes(expected, byte_order=">"))

    assert decoded.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(decoded, expected)


def test_read_tim_returns_samples(tmp_path) -> None:
    """The path reader should return the decoded array.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    path = tmp_path / "signal_1.tim"
    expected = np.array([1 - 1j, 2 - 2j], dtype=np.complex64)
    path.write_bytes(_tim_bytes(expected, byte_order="<"))

    samples = read_tim(path)

    np.testing.assert_array_equal(samples, expected)


def test_decode_tim_rejects_invalid_real_complex_flag() -> None:
    """The decoder should reject unknown real/complex header values.

    Raises:
        AssertionError: If the invalid header is accepted.
    """
    data = struct.pack("<ii", 3, 1) + struct.pack("<f", 1.0)

    try:
        decode_tim(data)
    except ValueError as exc:
        assert "Invalid .tim header" in str(exc)
    else:
        raise AssertionError("Expected an invalid .tim flag to fail")


def test_decode_tim_rejects_truncated_payload() -> None:
    """The decoder should require exactly the declared payload length.

    Raises:
        AssertionError: If a truncated payload is accepted.
    """
    data = struct.pack("<ii", 2, 2) + struct.pack("<fff", 1.0, 2.0, 3.0)

    try:
        decode_tim(data)
    except ValueError as exc:
        assert "payload length" in str(exc)
    else:
        raise AssertionError("Expected a truncated .tim payload to fail")

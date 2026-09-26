"""Periodic FFT windows shared by raw and reduced signal analysis."""

from typing import Literal

import numpy as np
import numpy.typing as npt

WindowFunction = Literal[
    "hann", "hamming", "blackman", "blackmanharris", "rectangular"
]

WINDOW_LABELS: dict[WindowFunction, str] = {
    "hann": "Hann",
    "hamming": "Hamming",
    "blackman": "Blackman",
    "blackmanharris": "Blackman-Harris",
    "rectangular": "Rectangular",
}
"""Display names for the supported periodic FFT windows."""


def fft_window(
    length: int, kind: WindowFunction = "hann"
) -> npt.NDArray[np.float64]:
    """Return periodic window coefficients for spectral analysis.

    Args:
        length: Positive number of coefficients.
        kind: Supported window name. A rectangular window uses unit weights.
            Blackman-Harris uses the minimum four-term coefficients.

    Returns:
        Float64 coefficients. A one-sample window has unit weight.

    Raises:
        ValueError: If the length or window name is invalid.

    References:
        Fredric J. Harris, "On the Use of Windows for Harmonic Analysis with
        the Discrete Fourier Transform", Proceedings of the IEEE, vol. 66,
        no. 1, pp. 51-83, January 1978. Equation (33), p. 64, defines the
        periodic form. The coefficient table on p. 65 gives the minimum
        four-term (-92 dB) window:
        https://www.fceia.unr.edu.ar/prodivoz/Harris_1978.pdf

        Julius O. Smith, "Spectral Audio Signal Processing", section
        "Blackman-Harris Window Family", explains the cosine-sum construction:
        https://www.dsprelated.com/freebooks/sasp/Blackman_Harris_Window_Family.html
    """
    if type(length) is not int or length < 1:
        raise ValueError("Window length must be a positive integer")
    functions = {
        "hann": np.hanning,
        "hamming": np.hamming,
        "blackman": np.blackman,
    }
    if kind not in WINDOW_LABELS:
        raise ValueError(f"Unknown FFT window: {kind}")
    if length == 1 or kind == "rectangular":
        return np.ones(length, dtype=np.float64)
    if kind == "blackmanharris":
        # Harris (1978), Eq. (33) and the "4-Term (-92 dB)" column on p. 65.
        # The denominator is length for periodic FFT analysis, rather than
        # length - 1 for a symmetric window used in filter design.
        phase = 2 * np.pi * np.arange(length) / length
        return (
            0.35875
            - 0.48829 * np.cos(phase)
            + 0.14128 * np.cos(2 * phase)
            - 0.01168 * np.cos(3 * phase)
        )
    return functions[kind](length + 1)[:-1]

"""Import Chad Spooner's CSPB machine-learning datasets."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Literal

from sigmf_zarr.json import JSONObject, JSONValue

logger = logging.getLogger(__name__)

CSPBTruthFormat = Literal["cspb-ml", "psk-mixtures"]
"""Supported Spooner truth-file layouts."""

CSPB_MODULATION_CLASSES: tuple[str, ...] = (
    "BPSK",
    "QPSK",
    "8PSK",
    "DQPSK",
    "16QAM",
    "64QAM",
    "256QAM",
    "MSK",
    "SQPSK",
    "GMSK",
)
"""Stable modulation label order covering the published CSPB datasets."""

_SIGNAL_INDEX_PATTERN = re.compile(r"(\d+)\.tim$", re.IGNORECASE)


@dataclass(frozen=True)
class CSPBSignalMetadata:
    """Truth metadata for one signal present in a dataset item."""

    modulation: str
    """Normalized modulation class name."""

    source_signal_index: int | None = None
    """Source single-signal index for mixture datasets."""

    symbol_rate: float | None = None
    """Normalized symbol rate in symbols per sample."""

    base_symbol_period: float | None = None
    """Base symbol period in samples."""

    carrier_offset: float | None = None
    """Normalized carrier-frequency offset in cycles per sample."""

    excess_bandwidth: float | None = None
    """Square-root raised-cosine excess-bandwidth parameter."""

    upsample_factor: int | None = None
    """Source resampling upsample factor."""

    downsample_factor: int | None = None
    """Source resampling downsample factor."""

    inband_snr_db: float | None = None
    """In-band signal-to-noise ratio in decibels."""

    noise_spectral_density_db: float | None = None
    """Noise spectral density in decibels."""

    modulation_variant: int | None = None
    """PSK Mixtures modulation-variant code."""

    modulation_type: int | None = None
    """PSK Mixtures modulation-type code."""

    signal_power_db: float | None = None
    """Signal power in decibels."""

    def to_json(self) -> JSONObject:
        """Return this signal's metadata as a JSON object.

        Returns:
            Namespaced JSON metadata with absent values omitted.
        """
        values: tuple[tuple[str, JSONValue], ...] = (
            ("cspb:modulation", self.modulation),
            ("cspb:source_signal_index", self.source_signal_index),
            ("cspb:symbol_rate", self.symbol_rate),
            ("cspb:base_symbol_period", self.base_symbol_period),
            ("cspb:carrier_offset", self.carrier_offset),
            ("cspb:excess_bandwidth", self.excess_bandwidth),
            ("cspb:upsample_factor", self.upsample_factor),
            ("cspb:downsample_factor", self.downsample_factor),
            ("cspb:inband_snr_db", self.inband_snr_db),
            (
                "cspb:noise_spectral_density_db",
                self.noise_spectral_density_db,
            ),
            ("cspb:modulation_variant", self.modulation_variant),
            ("cspb:modulation_type", self.modulation_type),
            ("cspb:signal_power_db", self.signal_power_db),
        )
        return {key: value for key, value in values if value is not None}


@dataclass(frozen=True)
class CSPBTruthRecord:
    """Truth metadata aligned with one ``.tim`` dataset item."""

    signal_index: int
    """Numeric index embedded in the corresponding ``.tim`` filename."""

    signals: tuple[CSPBSignalMetadata, ...]
    """One or more signals present in the item."""

    def to_item_metadata(self) -> JSONObject:
        """Return a SigMF-Zarr per-item metadata bundle.

        Returns:
            Item metadata containing the component-signal descriptions.
        """
        return {
            "global": {
                "cspb:signal_index": self.signal_index,
                "cspb:signals": [signal.to_json() for signal in self.signals],
            }
        }


@dataclass(frozen=True)
class CSPBTruth:
    """Parsed Spooner truth file."""

    format: CSPBTruthFormat
    """Detected truth-file layout."""

    records: dict[int, CSPBTruthRecord]
    """Truth records keyed by signal-file index."""


def _parse_int(value: str, *, field: str, line_number: int) -> int:
    """Parse one integer truth value with context.

    Args:
        value: Text value.
        field: Field name used in errors.
        line_number: One-based source line number.

    Returns:
        Parsed integer.

    Raises:
        ValueError: If the value is not an integer.
    """
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Truth line {line_number} has invalid {field}: {value!r}"
        ) from exc


def _parse_float(value: str, *, field: str, line_number: int) -> float:
    """Parse one finite floating-point truth value with context.

    Args:
        value: Text value.
        field: Field name used in errors.
        line_number: One-based source line number.

    Returns:
        Parsed finite float.

    Raises:
        ValueError: If the value is not a finite number.
    """
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(
            f"Truth line {line_number} has invalid {field}: {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(
            f"Truth line {line_number} has non-finite {field}: {value!r}"
        )
    return parsed


def _parse_cspb_ml_record(
    fields: list[str],
    *,
    line_number: int,
) -> CSPBTruthRecord:
    """Parse a CSPB.ML.2018 or CSPB.ML.2022 truth row.

    Args:
        fields: Whitespace-separated row fields.
        line_number: One-based source line number.

    Returns:
        Parsed truth record.

    Raises:
        ValueError: If the row does not contain the documented nine fields.
    """
    if len(fields) != 9:
        raise ValueError(
            f"Truth line {line_number} has {len(fields)} fields. "
            "CSPB.ML rows require 9"
        )
    signal_index = _parse_int(
        fields[0], field="signal index", line_number=line_number
    )
    base_symbol_period = _parse_float(
        fields[2], field="base symbol period", line_number=line_number
    )
    upsample_factor = _parse_int(
        fields[5], field="upsample factor", line_number=line_number
    )
    downsample_factor = _parse_int(
        fields[6], field="downsample factor", line_number=line_number
    )
    if base_symbol_period <= 0 or upsample_factor <= 0:
        raise ValueError(
            f"Truth line {line_number} requires positive symbol-period and "
            "upsample values"
        )
    symbol_rate = 1.0 / base_symbol_period
    if downsample_factor > 0:
        # The published truth describes rational resampling after symbol
        # generation, so express the final rate in symbols per output sample.
        symbol_rate *= downsample_factor / upsample_factor

    signal = CSPBSignalMetadata(
        modulation=fields[1].upper(),
        symbol_rate=symbol_rate,
        base_symbol_period=base_symbol_period,
        carrier_offset=_parse_float(
            fields[3], field="carrier offset", line_number=line_number
        ),
        excess_bandwidth=_parse_float(
            fields[4], field="excess bandwidth", line_number=line_number
        ),
        upsample_factor=upsample_factor,
        downsample_factor=downsample_factor,
        inband_snr_db=_parse_float(
            fields[7], field="in-band SNR", line_number=line_number
        ),
        noise_spectral_density_db=_parse_float(
            fields[8],
            field="noise spectral density",
            line_number=line_number,
        ),
    )
    return CSPBTruthRecord(signal_index=signal_index, signals=(signal,))


def _psk_mixtures_modulation(
    modulation_variant: int,
    modulation_type: int,
    *,
    line_number: int,
) -> str:
    """Decode the PSK Mixtures type and variant pair.

    Args:
        modulation_variant: Variant code from the truth file.
        modulation_type: Type code from the truth file.
        line_number: One-based source line number.

    Returns:
        Common modulation name.

    Undocumented pairs are retained under an explicit synthetic label so a
    published truth file can be imported without guessing its modulation.
    """
    names = {
        (1, 1): "BPSK",
        (1, 2): "QPSK",
        (1, 3): "8PSK",
        (2, 2): "QPSK",
        (2, 4): "16QAM",
        (2, 6): "64QAM",
        (3, 1): "SQPSK",
        (3, 2): "MSK",
        (3, 3): "GMSK",
    }
    name = names.get((modulation_type, modulation_variant))
    if name is not None:
        return name
    logger.warning(
        "Truth line %d has undocumented modulation type/variant pair "
        "(%d, %d)",
        line_number,
        modulation_type,
        modulation_variant,
    )
    return f"UNKNOWN-{modulation_type}-{modulation_variant}"


def _parse_psk_signal(
    fields: list[str],
    *,
    source_signal_index: int,
    line_number: int,
) -> CSPBSignalMetadata:
    """Parse one five-field PSK Mixtures signal description.

    Args:
        fields: Symbol rate, carrier offset, variant, type, and power.
        source_signal_index: Index of the originating single-signal file.
        line_number: One-based source line number.

    Returns:
        Parsed component-signal metadata.
    """
    modulation_variant = _parse_int(
        fields[2], field="modulation variant", line_number=line_number
    )
    modulation_type = _parse_int(
        fields[3], field="modulation type", line_number=line_number
    )
    return CSPBSignalMetadata(
        modulation=_psk_mixtures_modulation(
            modulation_variant,
            modulation_type,
            line_number=line_number,
        ),
        source_signal_index=source_signal_index,
        symbol_rate=_parse_float(
            fields[0], field="symbol rate", line_number=line_number
        ),
        carrier_offset=_parse_float(
            fields[1], field="carrier offset", line_number=line_number
        ),
        modulation_variant=modulation_variant,
        modulation_type=modulation_type,
        signal_power_db=_parse_float(
            fields[4], field="signal power", line_number=line_number
        ),
    )


def _parse_psk_mixtures_record(
    fields: list[str],
    *,
    line_number: int,
) -> CSPBTruthRecord:
    """Parse a CSPB.ML.2023 single- or two-signal truth row.

    Args:
        fields: Whitespace-separated row fields.
        line_number: One-based source line number.

    Returns:
        Parsed truth record.

    Raises:
        ValueError: If the row shape or redundant indexes are invalid.
    """
    match = re.fullmatch(r"Index_(\d+)", fields[0], re.IGNORECASE)
    if match is None:
        raise ValueError(
            f"Truth line {line_number} has invalid Index_N field "
            f"{fields[0]!r}"
        )
    signal_index = int(match.group(1))
    if len(fields) == 8:
        first_index = _parse_int(
            fields[1], field="source index", line_number=line_number
        )
        repeated_index = _parse_int(
            fields[2], field="repeated source index", line_number=line_number
        )
        if first_index != signal_index or repeated_index != signal_index:
            raise ValueError(
                f"Truth line {line_number} has inconsistent single-signal "
                "indexes"
            )
        signals: tuple[CSPBSignalMetadata, ...] = (
            _parse_psk_signal(
                fields[3:8],
                source_signal_index=first_index,
                line_number=line_number,
            ),
        )
    elif len(fields) == 13:
        first_index = _parse_int(
            fields[1], field="first source index", line_number=line_number
        )
        second_index = _parse_int(
            fields[2], field="second source index", line_number=line_number
        )
        signals = (
            _parse_psk_signal(
                fields[3:8],
                source_signal_index=first_index,
                line_number=line_number,
            ),
            _parse_psk_signal(
                fields[8:13],
                source_signal_index=second_index,
                line_number=line_number,
            ),
        )
    else:
        raise ValueError(
            f"Truth line {line_number} has {len(fields)} fields. PSK "
            "Mixtures rows require 8 or 13"
        )
    return CSPBTruthRecord(signal_index=signal_index, signals=signals)


def load_cspb_truth(path: str | PathLike[str]) -> CSPBTruth:
    """Load a published CSPB truth/label text file.

    The layout is detected from the first non-empty row. Numeric nine-field
    rows are the CSPB.ML.2018/2022 format. Rows beginning with ``Index_N``
    are the CSPB.ML.2023 PSK Mixtures format.

    Args:
        path: Truth text file.

    Returns:
        Parsed truth format and records keyed by signal index.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the file is empty, mixes layouts, contains malformed
            rows, or repeats an index.
    """
    lines = [
        (line_number, line.split())
        for line_number, raw_line in enumerate(
            Path(path).read_text(encoding="utf-8").splitlines(),
            start=1,
        )
        if (line := raw_line.strip()) and not line.startswith("#")
    ]
    if not lines:
        raise ValueError(f"CSPB truth file {str(path)!r} is empty")

    # Choose one layout for the whole file. Per-row detection could silently
    # combine incompatible challenge generations.
    truth_format: CSPBTruthFormat = (
        "psk-mixtures" if lines[0][1][0].lower().startswith("index_")
        else "cspb-ml"
    )
    records: dict[int, CSPBTruthRecord] = {}
    for line_number, fields in lines:
        record = (
            _parse_psk_mixtures_record(fields, line_number=line_number)
            if truth_format == "psk-mixtures"
            else _parse_cspb_ml_record(fields, line_number=line_number)
        )
        row_is_psk = fields[0].lower().startswith("index_")
        if row_is_psk != (truth_format == "psk-mixtures"):
            raise ValueError(
                f"Truth line {line_number} uses a different layout from "
                "the first row"
            )
        if record.signal_index in records:
            raise ValueError(
                f"Truth line {line_number} repeats signal index "
                f"{record.signal_index}"
            )
        records[record.signal_index] = record
    return CSPBTruth(format=truth_format, records=records)


__all__ = [
    "CSPBSignalMetadata",
    "CSPBTruth",
    "CSPBTruthRecord",
    "load_cspb_truth",
]

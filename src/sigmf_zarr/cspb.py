"""Import Chad Spooner's CSPB machine-learning datasets."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Literal, cast
from zipfile import BadZipFile, ZipFile

import numpy as np
import numpy.typing as npt
from zarr.core.array import CompressorLike, ShardsLike

from sigmf_zarr.json import JSONObject, JSONValue
from sigmf_zarr.store import SigMFRecording, SigMFZarrStore, ZarrFormat
from sigmf_zarr.tim import TimSamples, decode_tim

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


@dataclass(frozen=True)
class _TimEntry:
    """Location of one direct or ZIP-contained ``.tim`` file."""

    container: Path
    """Direct ``.tim`` path or containing ZIP path."""

    member: str | None
    """ZIP member name, or ``None`` for a direct file."""

    source_name: str
    """Human-readable source filename stored in the output index."""

    signal_index: int
    """Numeric signal identifier parsed from the filename."""


@dataclass(frozen=True)
class _CSPBSampleSpec:
    """Sample layout established by the first CSPB input file."""

    first_sample: npt.NDArray[np.float32]
    """First sample converted to the stored axis convention."""

    sample_shape: tuple[int, ...]
    """Per-item sample shape."""

    complex_samples: bool
    """Whether the source contains complex-valued samples."""

    sample_axes: tuple[str, ...]
    """Stored sample-axis names."""

    datatype: str
    """Corresponding standard SigMF datatype name."""


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


def _signal_index(name: str) -> int:
    """Extract the trailing numeric index from a ``.tim`` filename.

    Args:
        name: Direct path or ZIP member name.

    Returns:
        Parsed positive signal index.

    Raises:
        ValueError: If the filename does not end in ``N.tim``.
    """
    match = _SIGNAL_INDEX_PATTERN.search(Path(name).name)
    if match is None:
        raise ValueError(
            f"CSPB .tim filename must end in a numeric index: {name!r}"
        )
    signal_index = int(match.group(1))
    if signal_index <= 0:
        raise ValueError(f"CSPB signal index must be positive: {name!r}")
    return signal_index


def _zip_tim_entries(
    path: Path,
    *,
    source_root: Path | None,
) -> list[_TimEntry]:
    """List ``.tim`` members in one ZIP archive.

    Args:
        path: ZIP archive path.
        source_root: Optional directory used to make source names relative.

    Returns:
        Discovered entries.

    Raises:
        ValueError: If the ZIP archive is invalid or a filename lacks an
            index.
    """
    try:
        with ZipFile(path) as archive:
            # Discover names without extracting archives to disk. Archives are
            # reopened once and retained only while samples are streamed.
            members = [
                info.filename
                for info in archive.infolist()
                if not info.is_dir() and info.filename.lower().endswith(".tim")
            ]
    except BadZipFile as exc:
        raise ValueError(f"Invalid ZIP archive {str(path)!r}") from exc

    archive_name = (
        str(path.relative_to(source_root))
        if source_root is not None
        else path.name
    )
    return [
        _TimEntry(
            container=path,
            member=member,
            source_name=f"{archive_name}:{member}",
            signal_index=_signal_index(member),
        )
        for member in members
    ]


def _direct_tim_entry(path: Path, *, source_root: Path | None) -> _TimEntry:
    """Describe one direct ``.tim`` file.

    Args:
        path: Direct ``.tim`` path.
        source_root: Optional directory used to make the source name relative.

    Returns:
        Discovered entry.
    """
    source_name = (
        str(path.relative_to(source_root))
        if source_root is not None
        else path.name
    )
    return _TimEntry(
        container=path,
        member=None,
        source_name=source_name,
        signal_index=_signal_index(path.name),
    )


def _directory_tim_entries(source: Path) -> list[_TimEntry]:
    """Discover direct and archived ``.tim`` files in a directory tree.

    Args:
        source: Source directory.

    Returns:
        Discovered entries in path order.
    """
    entries: list[_TimEntry] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".tim":
            entries.append(_direct_tim_entry(path, source_root=source))
        elif suffix == ".zip":
            entries.extend(_zip_tim_entries(path, source_root=source))
    return entries


def _sort_unique_tim_entries(
    entries: Sequence[_TimEntry],
) -> list[_TimEntry]:
    """Validate signal indexes and sort entries by index.

    Args:
        entries: Entries to validate.

    Returns:
        Entries sorted by numeric signal index.

    Raises:
        ValueError: If two entries use the same signal index.
    """
    by_index: dict[int, _TimEntry] = {}
    for entry in entries:
        previous = by_index.get(entry.signal_index)
        if previous is not None:
            raise ValueError(
                f"Duplicate CSPB signal index {entry.signal_index}: "
                f"{previous.source_name!r} and {entry.source_name!r}"
            )
        by_index[entry.signal_index] = entry
    # Numeric order, rather than filename order, preserves alignment with the
    # truth files when identifiers have different digit counts.
    return [by_index[index] for index in sorted(by_index)]


def _discover_tim_entries(source: Path) -> list[_TimEntry]:
    """Discover direct and ZIP-contained ``.tim`` files.

    Args:
        source: A ``.tim`` file, ZIP batch, or directory tree.

    Returns:
        Entries sorted by numeric signal index.

    Raises:
        FileNotFoundError: If the source does not exist.
        ValueError: If no files are found or signal indexes are duplicated.
    """
    if not source.exists():
        raise FileNotFoundError(source)

    if source.is_file() and source.suffix.lower() == ".tim":
        entries = [_direct_tim_entry(source, source_root=None)]
    elif source.is_file() and source.suffix.lower() == ".zip":
        entries = _zip_tim_entries(source, source_root=None)
    elif source.is_dir():
        entries = _directory_tim_entries(source)
    else:
        raise ValueError(
            "CSPB source must be a .tim file, ZIP archive, or directory"
        )

    if not entries:
        raise ValueError(f"No .tim files found under {str(source)!r}")

    return _sort_unique_tim_entries(entries)


def _read_entry(
    entry: _TimEntry,
    *,
    archives: dict[Path, ZipFile],
) -> TimSamples:
    """Read one discovered ``.tim`` entry.

    Args:
        entry: Direct or archived entry.
        archives: Open ZIP archives keyed by path.
    Returns:
        Decoded signal.
    """
    data = (
        entry.container.read_bytes()
        if entry.member is None
        else archives[entry.container].read(entry.member)
    )
    return decode_tim(data)


def _zarr_sample(samples: npt.NDArray[np.generic]) -> npt.NDArray[np.float32]:
    """Convert one decoded signal to the SigMF-Zarr axis convention.

    Args:
        samples: One-dimensional real or complex signal.

    Returns:
        Float32 array with shape ``(time,)`` or ``(iq, time)``.
    """
    if np.iscomplexobj(samples):
        return np.stack((samples.real, samples.imag), axis=0).astype(
            np.float32,
            copy=False,
        )
    return np.asarray(samples, dtype=np.float32)


def cspb_sample_shape(
    source_path: str | PathLike[str],
) -> tuple[int, ...]:
    """Inspect the first CSPB file and return its Zarr sample shape.

    Args:
        source_path: Source ``.tim`` file, ZIP archive, or directory.
    Returns:
        Per-item shape using an explicit I/Q axis for complex samples.

    Raises:
        FileNotFoundError: If the source does not exist.
        ValueError: If no valid, non-empty ``.tim`` file is found.
    """
    entry = _discover_tim_entries(Path(source_path))[0]
    if entry.member is None:
        samples = decode_tim(entry.container.read_bytes())
    else:
        with ZipFile(entry.container) as archive:
            samples = decode_tim(archive.read(entry.member))
    sample = _zarr_sample(samples)
    if sample.size == 0:
        raise ValueError("CSPB .tim files must contain samples")
    return tuple(int(dim) for dim in sample.shape)


def _merge_truth_files(
    truth_paths: Sequence[str | PathLike[str]],
) -> tuple[dict[int, CSPBTruthRecord], tuple[CSPBTruthFormat, ...]]:
    """Load and merge one or more truth files.

    Args:
        truth_paths: Truth text files to merge.

    Returns:
        Records keyed by signal index and the encountered layout names.

    Raises:
        ValueError: If files contain duplicate signal indexes.
    """
    records: dict[int, CSPBTruthRecord] = {}
    formats: list[CSPBTruthFormat] = []
    for path in truth_paths:
        truth = load_cspb_truth(path)
        if truth.format not in formats:
            formats.append(truth.format)
        for signal_index, record in truth.records.items():
            if signal_index in records:
                raise ValueError(
                    f"Truth files repeat signal index {signal_index}"
                )
            records[signal_index] = record
    return records, tuple(formats)


def _single_signal_values(
    records: Sequence[CSPBTruthRecord],
    getter: Callable[[CSPBSignalMetadata], JSONValue],
) -> list[JSONValue] | None:
    """Return one scalar truth value per item when all are available.

    Args:
        records: Item-aligned truth records.
        getter: Function selecting one component field.

    Returns:
        Values when every item contains exactly one non-null value,
        otherwise ``None``.
    """
    values: list[JSONValue] = []
    for record in records:
        # Dense scalar indexes cannot represent cochannel mixtures or missing
        # fields. Those records remain available through per-item metadata.
        if len(record.signals) != 1:
            return None
        value = getter(record.signals[0])
        if value is None:
            return None
        values.append(value)
    return values


def _add_truth_indexes(
    recording: SigMFRecording,
    records: Sequence[CSPBTruthRecord],
) -> None:
    """Add dense indexes that can represent the selected truth records.

    Args:
        recording: Target recording.
        records: Item-aligned truth records.
    """
    recording.add_index(
        "signal_count",
        np.asarray([len(record.signals) for record in records], dtype=np.int8),
        axis="item",
        field="cspb:signal_count",
    )

    modulations = _single_signal_values(
        records,
        lambda signal: signal.modulation,
    )
    if modulations is not None:
        present = {cast(str, value) for value in modulations}
        # Retain the published stable order, then append unknown labels in a
        # deterministic order so integer IDs remain reproducible.
        labels = [
            value for value in CSPB_MODULATION_CLASSES if value in present
        ]
        labels.extend(sorted(present.difference(labels)))
        label_ids = {label: index for index, label in enumerate(labels)}
        recording.add_index(
            "mod_class_id",
            np.asarray(
                [label_ids[cast(str, value)] for value in modulations],
                dtype=np.int16,
            ),
            axis="item",
            field="cspb:modulation",
            labels=cast(list[JSONValue], labels),
        )

    scalar_indexes: tuple[
        tuple[
            str,
            str,
            str | None,
            Callable[[CSPBSignalMetadata], JSONValue],
        ],
        ...,
    ] = (
        (
            "symbol_rate",
            "cspb:symbol_rate",
            "symbols/sample",
            lambda signal: signal.symbol_rate,
        ),
        (
            "base_symbol_period",
            "cspb:base_symbol_period",
            "samples",
            lambda signal: signal.base_symbol_period,
        ),
        (
            "carrier_offset",
            "cspb:carrier_offset",
            "cycles/sample",
            lambda signal: signal.carrier_offset,
        ),
        (
            "excess_bandwidth",
            "cspb:excess_bandwidth",
            None,
            lambda signal: signal.excess_bandwidth,
        ),
        (
            "upsample_factor",
            "cspb:upsample_factor",
            None,
            lambda signal: signal.upsample_factor,
        ),
        (
            "downsample_factor",
            "cspb:downsample_factor",
            None,
            lambda signal: signal.downsample_factor,
        ),
        (
            "inband_snr_db",
            "cspb:inband_snr_db",
            "dB",
            lambda signal: signal.inband_snr_db,
        ),
        (
            "noise_spectral_density_db",
            "cspb:noise_spectral_density_db",
            "dB",
            lambda signal: signal.noise_spectral_density_db,
        ),
        (
            "modulation_variant",
            "cspb:modulation_variant",
            None,
            lambda signal: signal.modulation_variant,
        ),
        (
            "modulation_type",
            "cspb:modulation_type",
            None,
            lambda signal: signal.modulation_type,
        ),
        (
            "signal_power_db",
            "cspb:signal_power_db",
            "dB",
            lambda signal: signal.signal_power_db,
        ),
    )
    for index_name, field, unit, getter in scalar_indexes:
        values = _single_signal_values(records, getter)
        if values is None:
            continue
        recording.add_index(
            index_name,
            np.asarray(values),
            axis="item",
            field=field,
            unit=unit,
        )


def _select_truth_records(
    entries: Sequence[_TimEntry],
    truth: dict[int, CSPBTruthRecord],
) -> list[CSPBTruthRecord]:
    """Align available truth records with discovered input entries.

    Args:
        entries: Discovered input entries.
        truth: Truth records keyed by signal index.

    Returns:
        Item-aligned truth records, or an empty list when none were supplied.

    Raises:
        ValueError: If truth data omits an imported signal index.
    """
    if not truth:
        return []
    missing = [
        entry.signal_index
        for entry in entries
        if entry.signal_index not in truth
    ]
    if missing:
        preview = ", ".join(str(index) for index in missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(
            f"Truth data is missing {len(missing)} imported signal "
            f"indexes: {preview}{suffix}"
        )
    return [truth[entry.signal_index] for entry in entries]


def _sample_spec(first_samples: TimSamples) -> _CSPBSampleSpec:
    """Derive the stored layout from the first decoded CSPB file.

    Args:
        first_samples: Decoded first input file.

    Returns:
        Stored sample specification.

    Raises:
        ValueError: If the file contains no samples.
    """
    first_sample = _zarr_sample(first_samples)
    if first_sample.size == 0:
        raise ValueError("CSPB .tim files must contain samples")
    complex_samples = np.iscomplexobj(first_samples)
    return _CSPBSampleSpec(
        first_sample=first_sample,
        sample_shape=tuple(int(dim) for dim in first_sample.shape),
        complex_samples=complex_samples,
        sample_axes=("iq", "time") if complex_samples else ("time",),
        datatype="cf32_le" if complex_samples else "rf32_le",
    )


def _cspb_global_metadata(
    global_metadata: JSONObject | None,
    *,
    source_dataset: str,
    sample_spec: _CSPBSampleSpec,
    truth_formats: Sequence[CSPBTruthFormat],
) -> JSONObject:
    """Build global metadata for an imported CSPB recording.

    Args:
        global_metadata: Optional source global metadata.
        source_dataset: Published source dataset name.
        sample_spec: Stored sample specification.
        truth_formats: Encountered truth-file layouts.

    Returns:
        Merged global metadata.

    Raises:
        ValueError: If a declared SigMF datatype conflicts with the input.
    """
    metadata = dict(global_metadata or {})
    declared_datatype = metadata.get("core:datatype")
    if declared_datatype not in {None, sample_spec.datatype}:
        raise ValueError(
            f"global core:datatype {declared_datatype!r} does not match "
            f"the .tim data type {sample_spec.datatype!r}"
        )
    metadata["core:datatype"] = sample_spec.datatype
    metadata.setdefault("core:sample_rate", 1.0)
    metadata.setdefault("core:version", "1.2.0")
    metadata["cspb:source_dataset"] = source_dataset
    metadata["cspb:source_format"] = "tim"
    if truth_formats:
        metadata["cspb:truth_formats"] = list(truth_formats)
    return metadata


def _validate_sample(
    entry: _TimEntry,
    samples: TimSamples,
    sample: npt.NDArray[np.float32],
    *,
    sample_spec: _CSPBSampleSpec,
) -> None:
    """Validate one CSPB sample against the first input file.

    Args:
        entry: Source entry for error reporting.
        samples: Decoded source samples.
        sample: Converted sample array.
        sample_spec: Expected sample specification.

    Raises:
        ValueError: If sample complexity or shape changes.
    """
    if np.iscomplexobj(samples) != sample_spec.complex_samples:
        raise ValueError(
            f"CSPB file {entry.source_name!r} changes between real and "
            "complex samples"
        )
    if sample.shape != sample_spec.first_sample.shape:
        raise ValueError(
            f"CSPB file {entry.source_name!r} has sample shape "
            f"{sample.shape}. Expected {sample_spec.first_sample.shape}"
        )


def _append_cspb_batches(
    recording: SigMFRecording,
    entries: Sequence[_TimEntry],
    *,
    archives: dict[Path, ZipFile],
    batch_size: int,
    sample_spec: _CSPBSampleSpec,
    selected_truth: Sequence[CSPBTruthRecord],
) -> None:
    """Read, validate, and append CSPB files in bounded batches.

    Args:
        recording: Target recording.
        entries: Item-aligned source entries.
        archives: Open ZIP archives keyed by path.
        batch_size: Number of files per Zarr write.
        sample_spec: Expected sample specification.
        selected_truth: Optional item-aligned truth records.
    """
    # Single-signal truth is represented efficiently by dense indexes. Store
    # full bundles only when mixtures require nested component descriptions.
    use_item_metadata = any(
        len(record.signals) > 1 for record in selected_truth
    )
    logger.info("Importing %d CSPB .tim files", len(entries))
    for start in range(0, len(entries), batch_size):
        stop = min(start + batch_size, len(entries))
        batch_samples: list[npt.NDArray[np.float32]] = []
        batch_metadata: list[JSONObject] = []
        for position, entry in enumerate(entries[start:stop], start=start):
            if position == 0:
                # The first entry established the layout before recording
                # creation, so reuse it instead of reading it twice.
                sample = sample_spec.first_sample
            else:
                samples = _read_entry(entry, archives=archives)
                sample = _zarr_sample(samples)
                _validate_sample(
                    entry,
                    samples,
                    sample,
                    sample_spec=sample_spec,
                )
            batch_samples.append(sample)
            if use_item_metadata:
                batch_metadata.append(
                    selected_truth[position].to_item_metadata()
                )
        recording.append_samples(
            np.stack(batch_samples, axis=0),
            item_metadata=batch_metadata if use_item_metadata else None,
        )
        batch_number = start // batch_size + 1
        if batch_number % 100 == 0 or stop == len(entries):
            logger.info("Imported %d/%d CSPB files", stop, len(entries))


def _add_source_indexes(
    recording: SigMFRecording,
    entries: Sequence[_TimEntry],
) -> None:
    """Add source identity indexes for imported CSPB items.

    Args:
        recording: Target recording.
        entries: Item-aligned source entries.
    """
    recording.add_index(
        "signal_id",
        np.asarray(
            [entry.signal_index for entry in entries], dtype=np.int64
        ),
        axis="item",
        field="cspb:signal_index",
    )
    recording.add_index(
        "source_file",
        np.asarray([entry.source_name for entry in entries], dtype=object),
        axis="item",
        field="cspb:source_file",
    )


def import_cspb_dataset(
    store_path: str | PathLike[str],
    source_path: str | PathLike[str],
    *,
    truth_paths: Sequence[str | PathLike[str]] = (),
    source_dataset: str | None = None,
    recording_name: str = "cspb",
    overwrite_store: bool = False,
    overwrite_recording: bool = False,
    batch_size: int = 64,
    global_metadata: JSONObject | None = None,
    sample_chunks: tuple[int, ...] | None = None,
    sample_shards: ShardsLike | None = None,
    sample_compressor: CompressorLike = "auto",
    zarr_format: ZarrFormat | None = None,
) -> SigMFZarrStore:
    """Stream Spooner ``.tim`` datasets into a SigMF-Zarr recording.

    The source may be one ``.tim`` file, one ZIP batch, or a directory
    containing extracted files, ZIP batches, or both. Truth files are optional.
    CSPB.ML.2018/2022 and CSPB.ML.2023 layouts are detected automatically.

    Args:
        store_path: Target SigMF-Zarr store.
        source_path: Source ``.tim`` file, ZIP archive, or directory.
        truth_paths: Optional truth/label text files.
        source_dataset: Optional published dataset name.
        recording_name: Recording name to create.
        overwrite_store: Whether to recreate the target store.
        overwrite_recording: Whether to replace an existing recording.
        batch_size: Number of files appended per Zarr write.
        global_metadata: Optional SigMF global metadata to merge.
        sample_chunks: Optional sample-array chunk shape.
        sample_shards: Optional sample-array shard shape.
        sample_compressor: Optional sample-array compressor.
        zarr_format: Optional physical Zarr format requirement. Existing
            stores are auto-detected when omitted. New stores default to Zarr
            format 3.

    Returns:
        Updated SigMF-Zarr store.

    Raises:
        FileNotFoundError: If a source path does not exist.
        ValueError: If files, truth rows, shapes, encodings, or options are
            incompatible.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if zarr_format == 2 and sample_shards is not None:
        raise ValueError(
            "Sample sharding requires Zarr format 3. Omit sample_shards or "
            "create a Zarr format 3 store"
        )

    source = Path(source_path)
    entries = _discover_tim_entries(source)
    truth, truth_formats = _merge_truth_files(truth_paths)
    selected_truth = _select_truth_records(entries, truth)

    with ExitStack() as stack:
        # Keep each ZIP archive open across all batch reads and use the same
        # stack to roll back a partial recording on any later failure.
        archives = {
            path: stack.enter_context(ZipFile(path))
            for path in {entry.container for entry in entries if entry.member}
        }
        first_samples = _read_entry(
            entries[0],
            archives=archives,
        )
        sample_spec = _sample_spec(first_samples)
        metadata = _cspb_global_metadata(
            global_metadata,
            source_dataset=source_dataset or source.stem,
            sample_spec=sample_spec,
            truth_formats=truth_formats,
        )

        store = SigMFZarrStore.create(
            store_path,
            overwrite=overwrite_store,
            zarr_format=zarr_format,
        )
        if recording_name in store.recordings and not overwrite_recording:
            raise ValueError(
                f"Recording {recording_name!r} already exists. Pass "
                "overwrite_recording=True to replace it"
            )
        resolved_sample_chunks = sample_chunks
        if resolved_sample_chunks is None:
            resolved_sample_chunks = store.default_sample_chunks(
                np.float32,
                sample_spec.sample_shape,
                1 if sample_shards is not None else len(entries),
                batched=True,
                target_chunk_bytes=4 * 1024 * 1024,
            )
        recording = store.recordings.open(
            recording_name,
            create=True,
            batched=True,
            sample_dtype=np.float32,
            sample_shape=sample_spec.sample_shape,
            sample_axes=sample_spec.sample_axes,
            global_metadata=metadata,
            sample_chunks=resolved_sample_chunks,
            sample_shards=sample_shards,
            sample_compressor=sample_compressor,
            overwrite=overwrite_recording,
        )
        import_state = {"complete": False}

        def cleanup_failed_import() -> None:
            """Remove a partial CSPB recording when import fails."""
            if (
                not import_state["complete"]
                and recording_name in store.recordings
            ):
                store.recordings.remove(recording_name)

        # Register cleanup before the first sample write. Marking completion
        # suppresses it only after all indexes and integrity hashes are stored.
        stack.callback(cleanup_failed_import)

        _append_cspb_batches(
            recording,
            entries,
            archives=archives,
            batch_size=batch_size,
            sample_spec=sample_spec,
            selected_truth=selected_truth,
        )
        _add_source_indexes(recording, entries)
        if selected_truth:
            _add_truth_indexes(recording, selected_truth)
        recording.update_integrity()
        store.update_metadata_integrity()
        import_state["complete"] = True
        return store


__all__ = [
    "CSPB_MODULATION_CLASSES",
    "CSPBSignalMetadata",
    "CSPBTruth",
    "CSPBTruthFormat",
    "CSPBTruthRecord",
    "cspb_sample_shape",
    "import_cspb_dataset",
    "load_cspb_truth",
]

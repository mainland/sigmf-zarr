"""Core query syntax, exact comparisons, and bounded index-only execution."""

from __future__ import annotations

import operator
import sys
from concurrent.futures import CancelledError
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from zarr.core.array import Array

from sigmf_zarr.filtering import filter_items
from sigmf_zarr.query import IndexQuery, compile_query
from sigmf_zarr.store import SigMFRecording, SigMFZarrStore, ZarrFormat


@pytest.fixture(params=[2, 3])
def recording(
    tmp_path: Path, request: pytest.FixtureRequest
) -> SigMFRecording:
    """Create dense indexes with independent item JSON.

    Args:
        tmp_path: Temporary directory.
        request: Physical Zarr format parameter.

    Returns:
        Five-item recording.
    """
    store = SigMFZarrStore.create(
        tmp_path / "data.zarr", zarr_format=cast(ZarrFormat, request.param)
    )
    rec = store.recordings.open(
        "rec", batched=True, sample_shape=(2, 8), sample_axes=("iq", "time")
    )
    rec.append_samples(np.zeros((5, 2, 8), np.float32))
    rec.add_index(
        "snr",
        [-10.0, 0.0, 10.0, 20.0, np.nan],
        axis="item",
        field="test:value",
    )
    rec.add_index(
        "class_id",
        [0, 1, 0, 1, 0],
        axis="item",
        field="test:value",
        labels=["BPSK", "QPSK"],
    )
    rec.add_index(
        "splits/example", [0, 0, 1, 1, 1], axis="item", field="test:value"
    )
    rec.add_index(
        "flag",
        [True, False, True, False, True],
        axis="item",
        field="test:value",
    )
    rec.set_item_metadata([{"global": {"snr": 1000}}] * 5)
    return rec


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ('label(class_id) in ["QPSK"] and snr >= 0', [1, 3]),
        ('label("class_id") == "BPSK"', [0, 2, 4]),
        ('label(class_id) != "unknown"', [0, 1, 2, 3, 4]),
        ('label(class_id) == "unknown"', []),
        ("label(class_id) in []", []),
        ('index("splits/example") == 1', [2, 3, 4]),
        ("class_id == 0 or class_id == 1 and snr > 10", [0, 2, 3, 4]),
        ("(class_id == 0 or class_id == 1) and snr > 10", [3]),
        ("not (snr >= 10)", [0, 1]),
        ("not not (snr >= 10)", [2, 3]),
        ("snr < 0 or flag == true", [0, 2, 4]),
        ("not (snr < 0 or flag == true)", [1, 3]),
        ("not (snr < 0 and flag == false)", [0, 1, 2, 3, 4]),
        ("snr in []", []),
        ("not (snr in [])", [0, 1, 2, 3]),
        ("snr >= -1e1 and snr <= 1.0E+1", [0, 1, 2]),
        ("flag != true", [1, 3]),
    ],
)
def test_queries(
    recording: SigMFRecording, expression: str, expected: list[int]
) -> None:
    """Precedence, membership, and unknown values must have stable semantics.

    Args:
        recording: Source recording.
        expression: Query text.
        expected: Original matching positions.
    """
    assert (
        compile_query(expression).select(recording, batch_size=2).tolist()
        == expected
    )


@pytest.mark.parametrize(
    "expression",
    [
        "",
        " ",
        "snr >",
        "snr > 0 garbage",
        "snr == null",
        "snr == NaN",
        "snr == 1e1000",
        "snr == [1]",
        "snr in 1",
        "snr in [1, [2]]",
        'snr == "unterminated',
        "snr + 1 > 0",
        '__import__("os")',
        "snr > 1 && snr < 3",
        "snr == 01",
        "snr == +1",
    ],
)
def test_invalid_syntax(expression: str) -> None:
    """Unsupported syntax must fail during compilation.

    Args:
        expression: Invalid query text.
    """
    with pytest.raises(ValueError):
        compile_query(expression)


def test_error_location() -> None:
    """Syntax diagnostics should identify the offending source position."""
    with pytest.raises(ValueError, match="line 2, column"):
        compile_query("snr > 0 and\n@")


def test_reads_once_per_index_batch(
    recording: SigMFRecording, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated predicates must share reads and avoid JSON and samples.

    Args:
        recording: Source recording.
        monkeypatch: Patch fixture.
    """
    original = Array.__getitem__
    reads = []

    def read(array: Array, selection: Any, *args: Any, **kwargs: Any) -> Any:
        """Record allowed index reads and reject other payload access.

        Args:
            array: Source array.
            selection: Batch selection.
            *args: Additional read arguments.
            **kwargs: Additional read options.

        Returns:
            Stored index values.
        """
        assert "/indexes/" in array.path
        reads.append((array.path.rsplit("/", 1)[-1], selection))
        return original(array, selection, *args, **kwargs)

    monkeypatch.setattr(Array, "__getitem__", read)
    monkeypatch.setitem(sys.modules, "jmespath", None)
    query = compile_query(
        "snr >= 0 and snr <= 20 and "
        '(class_id == 0 or label(class_id) == "QPSK")'
    )
    assert query.index_names == ("class_id", "snr")
    assert query.select(recording, batch_size=2).tolist() == [1, 2, 3]
    assert len(reads) == 6
    for name in query.index_names:
        assert [s for n, s in reads if n == name] == [
            slice(0, 2),
            slice(2, 4),
            slice(4, 5),
        ]


@pytest.mark.parametrize(
    "expression",
    [
        'snr == "10"',
        "label(class_id) == 0",
        'label(snr) == "x"',
        "flag > true",
        "class_id == true",
        'snr in [0, "x"]',
    ],
)
def test_bad_types_before_reads(
    recording: SigMFRecording, expression: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validate operand types before starting a scan.

    Args:
        recording: Source recording.
        expression: Incompatible query.
        monkeypatch: Patch fixture.
    """

    def no_read(*args: Any, **kwargs: Any) -> Any:
        """Reject payload reads during validation."""
        raise AssertionError("Payload read before validation")

    monkeypatch.setattr(Array, "__getitem__", no_read)
    with pytest.raises(ValueError):
        compile_query(expression).select(recording)
    with pytest.raises(KeyError):
        compile_query("absent == 0").select(recording)


@pytest.mark.parametrize(
    "dtype,values,operand",
    [
        ("uint64", [0, 2**53, 2**53 + 1, 2**63, 2**64 - 1], 2**53 + 1),
        ("uint64", [0, 1, 2, 2**63, 2**64 - 1], -1),
        ("uint64", [0, 1, 2, 2**63, 2**64 - 1], 2**64),
        ("uint64", [0, 2**53, 2**53 + 1, 2**63, 2**64 - 1], float(2**53)),
        ("int64", [-(2**63), -1, 0, 2**53 + 1, 2**63 - 1], 2**64),
        ("int16", [-20, -1, 0, 1, 20], 0.5),
        ("float64", [0, 1, 2**53, 2**54, 2**63], 2**53 + 1),
        ("float32", [-20, -0.1, 0, 0.1, 20], 0.1),
    ],
)
def test_numeric_precision(
    recording: SigMFRecording, dtype: str, values: list[Any], operand: Any
) -> None:
    """Native comparisons and fallbacks must agree with Python numbers.

    Args:
        recording: Source recording.
        dtype: Index numeric type.
        values: Stored values.
        operand: Query literal.
    """
    array = np.asarray(values, dtype=dtype)
    recording.add_index("number", array, axis="item", field="test:value")
    for symbol, operation in [
        ("==", operator.eq),
        ("!=", operator.ne),
        ("<", operator.lt),
        ("<=", operator.le),
        (">", operator.gt),
        (">=", operator.ge),
    ]:
        expected = [
            i
            for i, value in enumerate(array.tolist())
            if operation(value, operand)
        ]
        assert (
            compile_query(f"number {symbol} {operand}")
            .select(recording, batch_size=3)
            .tolist()
            == expected
        )
    assert compile_query(f"number in [{operand}]").select(
        recording
    ).tolist() == [
        i for i, value in enumerate(array.tolist()) if value == operand
    ]


def test_empty_and_cancelled(recording: SigMFRecording) -> None:
    """Cancellation never returns a partial selection.

    Args:
        recording: Source recording.
    """
    progress = []
    query = compile_query("class_id == 0")
    with pytest.raises(CancelledError):
        query.select(
            recording,
            batch_size=2,
            cancelled=lambda: len(progress) > 1,
            progress=lambda done, total: progress.append((done, total)),
        )
    assert progress == [(0, 5), (2, 5)]
    assert IndexQuery().select(recording).tolist() == list(range(5))
    for size in (0, -1, True):
        with pytest.raises(ValueError, match="batch_size"):
            query.select(recording, batch_size=size)


def test_query_rebinds_label_tables(recording: SigMFRecording) -> None:
    """Compiled syntax must not retain a recording's label-to-ID mapping.

    Args:
        recording: Source recording.
    """
    query = compile_query('label(class_id) == "QPSK"')
    assert query.select(recording).tolist() == [1, 3]
    recording.add_index(
        "class_id",
        [0, 1, 0, 1, 0],
        axis="item",
        field="test:value",
        labels=["QPSK", "BPSK"],
        overwrite=True,
    )
    assert query.select(recording).tolist() == [0, 2, 4]


def test_query_with_separate_json_filter(recording: SigMFRecording) -> None:
    """The optional JSON predicate remains independent of the index query.

    Args:
        recording: Source recording.
    """
    pytest.importorskip("jmespath")
    assert filter_items(
        recording,
        index_query="snr >= 10",
        metadata_query="global.snr == `1000`",
    ).tolist() == [2, 3]


def test_strings_quoted_names_and_bad_categories(
    recording: SigMFRecording,
) -> None:
    """String literals and quoted names must retain their exact meaning.

    Args:
        recording: Source recording.
    """
    recording.add_index(
        "or",
        np.asarray(['a"b', "B", "A", "B", "A"]),
        axis="item",
        field="test:text",
    )
    assert compile_query(r'index("or") == "a\"b"').select(
        recording
    ).tolist() == [0]
    assert compile_query('index("or") in ["A", "B"]').select(
        recording
    ).tolist() == [1, 2, 3, 4]
    recording.add_index(
        "class_id",
        [0, 1, 0, 99, 0],
        axis="item",
        field="test:class",
        labels=["BPSK", "QPSK"],
        overwrite=True,
    )
    with pytest.raises(ValueError):
        compile_query('label(class_id) == "BPSK"').select(recording)


def test_empty_recording_and_unbatched_rejection(tmp_path: Path) -> None:
    """Empty selections must validate descriptors and report completion.

    Args:
        tmp_path: Temporary directory.
    """
    store = SigMFZarrStore.create(tmp_path / "empty.zarr")
    rec = store.recordings.open(
        "empty", batched=True, sample_shape=(2, 8), sample_axes=("iq", "time")
    )
    rec.add_index(
        "x", np.empty(0, dtype=np.int16), axis="item", field="test:x"
    )
    progress = []
    result = compile_query("x > 0").select(
        rec, progress=lambda done, total: progress.append((done, total))
    )
    assert result.dtype == np.int64
    assert result.size == 0
    assert progress == [(0, 0)]
    with pytest.raises(KeyError):
        compile_query("missing > 0").select(rec)
    continuous = store.recordings.open(
        "continuous", batched=False, sample_shape=(8,), sample_axes=("time",)
    )
    with pytest.raises(ValueError, match="batched"):
        IndexQuery().select(continuous)

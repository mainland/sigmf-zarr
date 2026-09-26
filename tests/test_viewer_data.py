"""Viewer data access must remain bounded, read-only, and independent of Qt."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from zarr.core.array import Array

from sigmf_zarr.store import SigMFZarrStore, ZarrFormat
from sigmf_zarr.viewer import DatasetSource


def test_peak_selection_decodes_named_axes(source: DatasetSource) -> None:
    """Envelope reads must honor item, channel, and time-axis selections.

    Args:
        source: Recording with reversed time/IQ order and two channels.
    """
    raw = source.read_window(
        "rec", item=2, start=3, count=4, coordinates={"channel": 1}
    )
    peaks = source.read_peaks(
        "rec", item=2, start=3, count=4, coordinates={"channel": 1}
    )
    np.testing.assert_array_equal(peaks, np.abs(raw.samples))


@pytest.fixture(params=[2, 3])
def source(tmp_path: Path, request: pytest.FixtureRequest) -> DatasetSource:
    """Create a recording with reversed time/IQ order and two receiver
    channels.

    Args:
        tmp_path: Temporary directory.
        request: Physical Zarr format.

    Returns:
        Read-only viewer data source.
    """
    path = tmp_path / "data.zarr"
    store = SigMFZarrStore.create(
        path, zarr_format=cast(ZarrFormat, request.param)
    )
    recording = store.recordings.open(
        "rec",
        batched=True,
        sample_dtype="float32",
        sample_shape=(8, 2, 2),
        sample_axes=("time", "channel", "iq"),
        global_metadata={"test:class": "shared"},
        channel_metadata=[{"test:receiver": "A"}, {"test:receiver": "B"}],
    )
    recording.append_samples(
        np.arange(96, dtype=np.float32).reshape(3, 8, 2, 2)
    )
    recording.add_index(
        "class",
        [0, 1, 0],
        axis="item",
        field="test:class",
        labels=["BPSK", "QPSK"],
    )
    recording.set_item_metadata(
        [{"global": {"test:class": "JSON"}}, None, None]
    )
    unbatched = store.recordings.open(
        "continuous",
        batched=False,
        sample_dtype="complex64",
        sample_shape=(8,),
        sample_axes=("time",),
    )
    unbatched.set_samples(np.full(8, 3 + 4j, dtype=np.complex64))
    store.collections.open(
        "group",
        recording_ids=["rec", "continuous", "rec", "missing"],
        metadata={"description": "Example collection"},
    )
    store.collections.open("empty", recording_ids=[])
    return DatasetSource(path)


def test_descriptor_reads_do_not_scan_payload(
    source: DatasetSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opening the browser must read only descriptors, even with item JSON.

    Args:
        source: Example data source.
        monkeypatch: Patch fixture.
    """

    def no_read(*args: Any, **kwargs: Any) -> Any:
        """Observe access during the test."""
        raise AssertionError("Unexpected payload scan")

    monkeypatch.setattr(Array, "__getitem__", no_read)
    assert source.recordings() == ("continuous", "rec")
    catalog = source.catalog()
    assert catalog.recordings == ("continuous", "rec")
    assert catalog.collections == {
        "empty": (),
        "group": ("rec", "continuous", "rec", "missing"),
    }
    metadata = source.collection_metadata("group")
    assert metadata["metadata"] == {"description": "Example collection"}
    assert metadata["recording_ids"] == ["rec", "continuous", "rec", "missing"]
    info = source.describe("rec")
    assert info.item_count == 3
    assert info.axes == ("item", "time", "channel", "iq")
    assert info.indexes["class"]["field"] == "test:class"


def test_window_uses_named_axes_and_preserves_metadata(
    source: DatasetSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read one channel and a bounded time range without projecting labels.

    Args:
        source: Example data source.
        monkeypatch: Patch fixture.
    """
    original = Array.__getitem__
    reads = []

    def read(array: Array, selection: Any, *args: Any, **kwargs: Any) -> Any:
        """Observe access during the test.

        Args:
            array: Array being observed.
            selection: Array selection.

        Returns:
            Observed operation result.
        """
        if array.path.endswith("samples"):
            reads.append(selection)
        return original(array, selection, *args, **kwargs)

    monkeypatch.setattr(Array, "__getitem__", read)
    window = source.read_window(
        "rec",
        item=0,
        start=2,
        count=3,
        coordinates={"channel": 1},
        indexes=["class"],
    )
    np.testing.assert_array_equal(
        window.samples, [10 + 11j, 14 + 15j, 18 + 19j]
    )
    assert reads == [(0, slice(2, 5), 1, slice(None))]
    assert window.metadata["global"]["test:class"] == "JSON"
    assert window.recording_metadata["global"]["test:class"] == "shared"
    assert window.item_metadata == {"global": {"test:class": "JSON"}}
    assert window.channel_metadata["test:receiver"] == "B"
    assert "indexes" not in window.metadata
    assert "test:receiver" not in window.metadata["global"]
    window.metadata["global"]["test:class"] = "changed"
    assert window.item_metadata["global"]["test:class"] == "JSON"
    assert window.recording_metadata["global"]["test:class"] == "shared"
    other = source.read_window("rec", item=1, coordinates={"channel": 0})
    assert other.item_metadata == {}
    assert other.channel_metadata["test:receiver"] == "A"
    continuous = source.read_window("continuous")
    assert continuous.item_metadata is None
    assert continuous.channel_metadata is None
    assert window.indexes["class"]["label"] == "BPSK"
    assert not window.samples.flags.writeable
    with pytest.raises(ValueError):
        window.samples[0] = 0
    assert source.read_window(
        "rec", item=0, start=7, count=99, coordinates={"channel": 0}
    ).samples.shape == (1,)


def test_overview_preserves_named_axes_and_metadata(
    source: DatasetSource,
) -> None:
    """Reduced data must preserve item, channel, and index provenance.

    Args:
        source: Example source in each physical Zarr format.
    """
    view = source.read_overview(
        "rec",
        item=0,
        start=2,
        count=3,
        coordinates={"channel": 1},
        indexes=["class"],
        fft_size=2,
    )
    np.testing.assert_array_equal(
        view.signal.iq, [10 + 11j, 14 + 15j, 18 + 19j]
    )
    assert view.context.samples.size == 0
    assert view.context.metadata["global"]["test:class"] == "JSON"
    assert view.context.recording_metadata["global"]["test:class"] == "shared"
    assert view.context.channel_metadata["test:receiver"] == "B"
    assert view.context.indexes["class"]["label"] == "BPSK"


def test_page_preserves_original_positions_and_avoids_json(
    source: DatasetSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pages should preserve order and repeats without scanning JSON or
    samples.

    Args:
        source: Example data source.
        monkeypatch: Patch fixture.
    """
    original = Array.__getitem__

    def read(array: Array, selection: Any, *args: Any, **kwargs: Any) -> Any:
        """Observe access during the test.

        Args:
            array: Array being observed.
            selection: Array selection.

        Returns:
            Observed operation result.
        """
        assert not array.path.endswith(("samples", "item_metadata"))
        return original(array, selection, *args, **kwargs)

    monkeypatch.setattr(Array, "__getitem__", read)
    rows = source.page("rec", [2, 0, 2, 1], ["class"])
    assert [row["item"] for row in rows] == [2, 0, 2, 1]
    assert rows[-1]["indexes"]["class"] == {"value": 1, "label": "QPSK"}
    with pytest.raises(IndexError):
        source.page("rec", [-1])
    with pytest.raises(ValueError):
        source.page("rec", [0] * 4097)


def test_unbatched_and_selection_errors(source: DatasetSource) -> None:
    """Continuous recordings use bounded windows.

    Args:
        source: Example data source.
    """
    assert source.describe("continuous").item_count == 1
    np.testing.assert_array_equal(
        source.read_window("continuous").samples, [3 + 4j] * 8
    )
    with pytest.raises(ValueError, match="batched"):
        source.filter("continuous")
    for kwargs in (
        {"item": 0},
        {"count": 65537},
        {"count": 0},
        {"coordinates": {"channel": 0}},
    ):
        with pytest.raises(ValueError):
            source.read_window("continuous", **kwargs)
    with pytest.raises(IndexError):
        source.read_window("continuous", start=8)
    with pytest.raises(ValueError, match="Select an item"):
        source.read_window("rec", coordinates={"channel": 0})


def test_optional_imports_and_cli_help() -> None:
    """Base imports, data access, and CLI help must not import GUI packages."""
    code = """
import sys
import sigmf_zarr
import sigmf_zarr.filtering
import sigmf_zarr.viewer
from sigmf_zarr.cli.sigmf import SigMFCommand
SigMFCommand().build_parser()
optional = {"PySide6", "matplotlib", "torch", "jmespath"}
assert not optional.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
    subprocess.run(
        [sys.executable, "-m", "sigmf_zarr.cli.sigmf", "view", "--help"],
        check=True,
        capture_output=True,
    )


def test_view_command_reports_missing_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing desktop extra should produce an installation hint.

    Args:
        monkeypatch: Patch fixture.
    """
    import argparse

    import sigmf_zarr.cli.view as view

    monkeypatch.setattr(view, "find_spec", lambda name: None)
    with pytest.raises(ValueError, match="sigmf-zarr\\[viewer\\]"):
        view.ViewCommand().handle(argparse.Namespace(store="unused"))


def test_view_command_does_not_require_jmespath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The viewer can launch when only the optional JSON dependency is absent.

    Args:
        monkeypatch: Patch fixture.
    """
    import argparse
    from types import ModuleType

    import sigmf_zarr.cli.view as view

    launched = []
    module = ModuleType("sigmf_zarr.viewer.qt")

    def launch(path: str) -> int:
        """Record the requested store without starting a Qt event loop.

        Args:
            path: Store location supplied by the command.

        Returns:
            Successful exit status.
        """
        launched.append(path)
        return 0

    monkeypatch.setattr(module, "launch", launch, raising=False)
    monkeypatch.setitem(sys.modules, "sigmf_zarr.viewer.qt", module)
    monkeypatch.setattr(
        view,
        "find_spec",
        lambda name: None if name == "jmespath" else object(),
    )
    assert view.ViewCommand().handle(argparse.Namespace(store="example")) == 0
    assert launched == ["example"]

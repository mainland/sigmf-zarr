"""Rollback must wait for Zarr writes still running after a failure."""

import asyncio
import signal
import threading
import time
from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from zarr.abc.store import Store
from zarr.core.buffer import Buffer, default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import MemoryStore

from sigmf_zarr.store import SigMFZarrStore, ZarrFormat
from sigmf_zarr.validation import validate_store

_DELAY = 0.2
"""Seconds that sibling writes remain in flight after the failure."""


class _RacingWrites:
    """Fail one backend write while a sibling write is still in flight.

    Every write delays before reaching the backend. The first write that
    starts while another is pending triggers the failure. Writes that start
    after the failure, such as rollback writes, run without delay.
    """

    def __init__(
        self,
        backend: Store,
        monkeypatch: pytest.MonkeyPatch,
        *,
        interrupt: bool,
    ) -> None:
        """Wrap the writing methods of one backend instance.

        Args:
            backend: Store backend receiving the writes.
            monkeypatch: Fixture that restores the methods after the test.
            interrupt: Whether to interrupt the main thread instead of
                failing the write.
        """
        self._interrupt = interrupt
        self._in_flight = 0
        self.failed = False
        self.completed_after_failure = 0
        for name in ("set", "set_if_not_exists"):
            original = getattr(backend, name)
            monkeypatch.setattr(backend, name, self._wrap(original))

    def _wrap(
        self, original: Callable[[str, Buffer], Awaitable[None]]
    ) -> Callable[[str, Buffer], Awaitable[None]]:
        """Return a write method that injects the race.

        Args:
            original: Bound backend write method.

        Returns:
            Replacement write method.
        """

        async def write(key: str, value: Buffer) -> None:
            if self.failed:
                await original(key, value)
                return
            if self._in_flight:
                self.failed = True
                if not self._interrupt:
                    raise OSError(f"injected failure writing {key}")
                main = threading.main_thread().ident
                assert main is not None
                signal.pthread_kill(main, signal.SIGINT)
            self._in_flight += 1
            try:
                await asyncio.sleep(_DELAY)
                await original(key, value)
            finally:
                self._in_flight -= 1
            if self.failed:
                self.completed_after_failure += 1

        return write


async def _read_all(backend: Store) -> dict[str, bytes]:
    """Read every stored object.

    Args:
        backend: Store backend to read.

    Returns:
        Stored bytes by key.
    """
    result = {}
    async for key in backend.list():
        value = await backend.get(key, prototype=default_buffer_prototype())
        assert value is not None
        result[key] = value.to_bytes()
    return result


@pytest.fixture(
    params=[("local", 2), ("local", 3), ("memory", 2), ("memory", 3)]
)
def store(tmp_path: Path, request: pytest.FixtureRequest) -> SigMFZarrStore:
    """Create a hashed store with one recording.

    Args:
        tmp_path: Temporary directory.
        request: Backend and physical-format parameters.

    Returns:
        Writable example store.
    """
    backend, version = request.param
    result = SigMFZarrStore.create(
        tmp_path / "data.zarr" if backend == "local" else MemoryStore(),
        zarr_format=cast(ZarrFormat, version),
    )
    recording = result.recordings.create(
        "rec", batched=True, sample_shape=(4,)
    )
    recording.append_samples(np.arange(12, dtype=np.float32).reshape(3, 4))
    recording.add_index("quality", [1, 2, 3], axis="item", field="test:q")
    result.update_integrity()
    return result


_CREATE: dict[str, Any] = {
    "batched": True,
    "sample_shape": (2, 3),
    "sample_axes": ("channel", "time"),
    "channel_metadata": [{"test:a": 1}, {"test:a": 2}],
}
"""Creation options that write several nodes concurrently."""


def test_failed_creation_waits_for_sibling_writes(
    store: SigMFZarrStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A late sibling write must not leave a partial recording.

    Args:
        store: Hashed example store.
        monkeypatch: Write failure injection.
    """
    backend = store._group.store
    before = sync(_read_all(backend))
    racing = _RacingWrites(backend, monkeypatch, interrupt=False)
    with pytest.raises(OSError, match="injected failure"):
        store.recordings.create("new", **_CREATE)
    assert racing.completed_after_failure > 0
    assert sync(_read_all(backend)) == before
    assert validate_store(store).valid
    created = store.recordings.create("new", **_CREATE)
    assert created.sample_shape == (2, 3)


def test_failed_replacement_waits_for_sibling_writes(
    store: SigMFZarrStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A late sibling write must not overwrite the restored recording.

    Args:
        store: Hashed example store.
        monkeypatch: Write failure injection.
    """
    backend = store._group.store
    before = sync(_read_all(backend))
    racing = _RacingWrites(backend, monkeypatch, interrupt=False)
    with pytest.raises(OSError, match="injected failure"):
        store.recordings.create(
            "rec", overwrite=True, sample_dtype=np.int16, **_CREATE
        )
    assert racing.completed_after_failure > 0
    assert sync(_read_all(backend)) == before
    assert validate_store(store).valid
    assert store.verify_integrity()


@pytest.mark.skipif(
    not hasattr(signal, "pthread_kill"), reason="requires POSIX signals"
)
@pytest.mark.parametrize("name", ["new", "rec"])
def test_interrupted_creation_waits_for_running_operation(
    store: SigMFZarrStore, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """An interrupt must roll back only after Zarr stops writing.

    Args:
        store: Hashed example store.
        monkeypatch: Interrupt injection.
        name: New or replaced recording name.
    """
    backend = store._group.store
    before = sync(_read_all(backend))
    racing = _RacingWrites(backend, monkeypatch, interrupt=True)
    # A shell starts background jobs with SIGINT ignored, and Python then
    # keeps that disposition. Install Python's handler for this test.
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with pytest.raises(KeyboardInterrupt):
            store.recordings.create(name, overwrite=True, **_CREATE)
    finally:
        signal.signal(signal.SIGINT, previous)
    assert racing.completed_after_failure > 0
    assert sync(_read_all(backend)) == before
    assert validate_store(store).valid


@pytest.mark.parametrize(
    "name", [".zarray", ".zattrs", ".zgroup", ".zmetadata", "zarr.json"]
)
def test_zarr_metadata_names_are_rejected_before_writing(
    store: SigMFZarrStore, name: str
) -> None:
    """A Zarr metadata document name is never a resource or array name.

    In either Zarr format, such a name can address the parent's own
    metadata document, so it is rejected before any write.

    Args:
        store: Hashed example store.
        name: Reserved Zarr metadata document name.
    """
    recording = store.recordings["rec"]
    backend = store._group.store
    before = sync(_read_all(backend))
    operations: list[Callable[[], object]] = [
        lambda: store.recordings.create(name, sample_shape=(4,)),
        lambda: store.recordings.create(
            name, overwrite=True, sample_shape=(4,)
        ),
        lambda: store.collections.create(name, overwrite=True),
        lambda: store.recordings.open(name),
        lambda: store.collections.open(name),
    ]
    for array_name in (name, f"group/{name}", f"{name}/values"):
        operations += [
            partial(
                store.add_index,
                array_name,
                np.zeros(4, np.int8),
                overwrite=True,
            ),
            partial(
                recording.add_index,
                array_name,
                [1, 2, 3],
                axis="item",
                field="test:q",
            ),
            partial(
                recording.add_split,
                array_name,
                [0, 1, 0],
                labels=["train", "test"],
            ),
            partial(
                recording.add_extension_array,
                array_name,
                np.zeros(3),
                overwrite=True,
            ),
            partial(store.index, array_name),
            partial(recording.index, array_name),
            partial(recording.split, array_name),
        ]
    for operation in operations:
        with pytest.raises(ValueError, match="reserved for Zarr metadata"):
            operation()
    assert name not in store.recordings
    assert sync(_read_all(backend)) == before
    assert validate_store(store, require_integrity=True).valid


@pytest.mark.parametrize("overwrite", [False, True])
def test_names_below_arrays_are_rejected_before_writing(
    store: SigMFZarrStore, overwrite: bool
) -> None:
    """A name below an existing array must not touch that array's chunks.

    Zarr format 3 stores the chunks of array `ids` under `ids/c/`, so a
    failed creation of `ids/c` must not delete that prefix during rollback.

    Args:
        store: Hashed example store.
        overwrite: Whether the rejected call requests replacement.
    """
    recording = store.recordings["rec"]
    store.add_index("ids", np.arange(1, 6))
    recording.add_extension_array("ext", np.arange(1, 6))
    store.update_integrity()
    backend = store._group.store
    before = sync(_read_all(backend))
    operations: list[Callable[[], object]] = []
    for suffix in ("c", "c/0", "0"):
        operations += [
            partial(
                store.add_index,
                f"ids/{suffix}",
                np.zeros(4, np.int8),
                overwrite=overwrite,
            ),
            partial(
                recording.add_index,
                f"quality/{suffix}",
                [1, 2, 3],
                axis="item",
                field="test:q",
                overwrite=overwrite,
            ),
            partial(
                recording.add_split,
                f"quality/{suffix}",
                [0, 1, 0],
                labels=["train", "test"],
                overwrite=overwrite,
            ),
            partial(
                recording.add_extension_array,
                f"ext/{suffix}",
                np.zeros(3),
                overwrite=overwrite,
            ),
        ]
    for operation in operations:
        with pytest.raises(ValueError, match="below the existing array"):
            operation()
    assert sync(_read_all(backend)) == before
    np.testing.assert_array_equal(store.index("ids")[:], np.arange(1, 6))
    assert validate_store(store, require_integrity=True).valid


def _add_resizable_recordings(store: SigMFZarrStore) -> None:
    """Add hashed batched and unbatched recordings with small chunks.

    One-item and one-sample chunks make each write span several chunks.

    Args:
        store: Example store.
    """
    items = store.recordings.create(
        "items", batched=True, sample_shape=(4,), sample_chunks=(1, 4)
    )
    items.append_samples(np.zeros((3, 4), np.float32), item_metadata=[{}] * 3)
    items.add_index("q", [1, 2, 3], axis="item", field="test:q")
    continuous = store.recordings.create(
        "time", sample_shape=(3,), sample_chunks=(1,)
    )
    continuous.set_samples(np.arange(3, dtype=np.float32))
    continuous.add_index("t", [1, 2, 3], axis="time", field="test:t")
    store.update_integrity()


def _append_items(store: SigMFZarrStore) -> None:
    """Append three items, written as concurrent chunk writes.

    Args:
        store: Example store.
    """
    store.recordings["items"].append_samples(
        np.ones((3, 4), np.float32), capture={"test:c": 1}
    )


def _append_time(store: SigMFZarrStore) -> None:
    """Append samples to an unbatched recording across several chunks.

    Args:
        store: Example store.
    """
    store.recordings["time"].append_samples(
        np.ones(6, np.float32), capture={"test:c": 1}
    )


def _replace_samples(store: SigMFZarrStore) -> None:
    """Replace and lengthen the samples of an unbatched recording.

    Args:
        store: Example store.
    """
    store.recordings["time"].set_samples(np.ones(9, np.float32))


def _mutate_samples(store: SigMFZarrStore) -> None:
    """Write several sample chunks through the mutation context.

    Args:
        store: Example store.
    """
    with store.recordings["time"].mutate_samples() as samples:
        samples.resize((9,))
        samples[:] = np.ones(9, np.float32)


def _mutate_item_metadata(store: SigMFZarrStore) -> None:
    """Write several item metadata chunks through the mutation context.

    Args:
        store: Example store.
    """
    with store.recordings["items"].mutate_item_metadata() as array:
        array.resize((5,))
        array[:] = np.full(5, '{"global": {"test:x": 2}}', dtype=object)


@pytest.mark.skipif(
    not hasattr(signal, "pthread_kill"), reason="requires POSIX signals"
)
@pytest.mark.parametrize("interrupt", [False, True])
@pytest.mark.parametrize("operation", [_append_items, _append_time])
def test_append_rollback_waits_for_running_writes(
    store: SigMFZarrStore,
    monkeypatch: pytest.MonkeyPatch,
    interrupt: bool,
    operation: Callable[[SigMFZarrStore], None],
) -> None:
    """Append rollback must follow every late chunk write.

    A late chunk write that lands after truncation would leave a chunk
    outside the restored shape.

    Args:
        store: Hashed example store.
        monkeypatch: Failure or interrupt injection.
        interrupt: Whether to interrupt the main thread instead of failing a
            write.
        operation: Append that writes several chunks concurrently.
    """
    _add_resizable_recordings(store)
    backend = store._group.store
    before = sync(_read_all(backend))
    racing = _RacingWrites(backend, monkeypatch, interrupt=interrupt)
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with pytest.raises(KeyboardInterrupt if interrupt else OSError):
            operation(store)
    finally:
        signal.signal(signal.SIGINT, previous)
    # Give late writes time to land, so the checks below see any damage
    # that a rollback without waiting would leave.
    time.sleep(3 * _DELAY)
    assert racing.failed
    assert racing.completed_after_failure > 0
    report = validate_store(store)
    assert report.valid, report.issues
    # An append replaces no values, so rollback restores every object.
    assert sync(_read_all(backend)) == before


@pytest.mark.skipif(
    not hasattr(signal, "pthread_kill"), reason="requires POSIX signals"
)
@pytest.mark.parametrize(
    ("operation", "array"),
    [
        (_append_items, "items/samples"),
        (_append_time, "time/samples"),
        (_replace_samples, "time/samples"),
        (_mutate_samples, "time/samples"),
        (_mutate_item_metadata, "items/item_metadata"),
    ],
)
def test_rollback_follows_late_resize(
    store: SigMFZarrStore,
    monkeypatch: pytest.MonkeyPatch,
    operation: Callable[[SigMFZarrStore], None],
    array: str,
) -> None:
    """A resize interrupted before its metadata write lands is undone.

    The interrupt arrives while Zarr writes the resized array metadata. If
    rollback restored the shape without waiting, the late write would land
    afterward and leave a shape that disagrees with the recording metadata.

    Args:
        store: Hashed example store.
        monkeypatch: Interrupt injection.
        operation: Mutation that resizes `array`.
        array: Resized array path below `recordings/`.
    """
    _add_resizable_recordings(store)
    backend = store._group.store
    document = f"recordings/{array}/" + (
        "zarr.json" if store.zarr_format == 3 else ".zarray"
    )
    original_set = backend.set
    state = {"armed": True}

    async def late_set(key: str, value: Buffer) -> None:
        """Interrupt the main thread, then write the array metadata late.

        Args:
            key: Object key to write.
            value: Object contents.
        """
        if state["armed"] and key == document:
            state["armed"] = False
            main = threading.main_thread().ident
            assert main is not None
            signal.pthread_kill(main, signal.SIGINT)
            await asyncio.sleep(_DELAY)
        await original_set(key, value)

    monkeypatch.setattr(backend, "set", late_set)
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with pytest.raises(KeyboardInterrupt):
            operation(store)
    finally:
        signal.signal(signal.SIGINT, previous)
    time.sleep(3 * _DELAY)
    assert not state["armed"]
    report = validate_store(store)
    assert report.valid, report.issues
    assert store.recordings["time"].sample_shape == (3,)
    assert len(store.recordings["items"]) == 3

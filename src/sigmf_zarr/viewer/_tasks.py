"""Background operations with cooperative cancellation and stale-result
checks.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Event

from PySide6.QtCore import QObject, QTimer

Progress = Callable[[int, int], None]
Work = Callable[[Event, Progress], object]


@dataclass
class _Task:
    """One replaceable operation tracked by the GUI thread."""

    cancelled: Event
    """Cancellation request visible to worker code."""

    done: Callable[[object], None]
    """GUI-thread completion callback."""

    future: Future[object] | None = None
    """Executor future, assigned after construction."""

    progress: tuple[int, int] | None = None
    """Latest progress snapshot.

    Assignment replaces the complete tuple.
    """

    reported: tuple[int, int] | None = field(default=None)
    """Last progress snapshot displayed by the GUI."""

    def report(self, examined: int, total: int) -> None:
        """Replace the pending progress snapshot without touching Qt widgets.

        Args:
            examined: Number of examined items.
            total: Total number of items.
        """
        self.progress = (examined, total)


class Tasks(QObject):
    """Run bounded concurrent work and deliver only the latest keyed result."""

    _executor: ThreadPoolExecutor
    """Worker pool.

    Workers do not own Qt objects.
    """

    _pending: dict[str, _Task]
    """Latest operation for each independent task key."""

    _timer: QTimer
    """GUI-thread completion poller."""

    _error: Callable[[str], None]
    """GUI-thread error callback."""

    _progress: Callable[[str, int, int], None]
    """GUI-thread progress callback."""

    def __init__(
        self,
        parent: QObject,
        error: Callable[[str], None],
        progress: Callable[[str, int, int], None],
    ) -> None:
        """Create a worker pool and a completion timer.

        Args:
            parent: Owning GUI object.
            error: Error display callback.
            progress: Progress display callback.
        """
        super().__init__(parent)
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._pending = {}
        self._error = error
        self._progress = progress
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.poll)
        self._timer.start(40)

    def submit(
        self, key: str, work: Work, done: Callable[[object], None]
    ) -> None:
        """Replace a keyed task and suppress its superseded result.

        Args:
            key: Independent operation name.
            work: Worker accepting cancellation and progress callbacks.
            done: Completion callback on the GUI thread.
        """
        self.cancel(key)
        task = _Task(Event(), done)
        task.future = self._executor.submit(work, task.cancelled, task.report)
        self._pending[key] = task

    def cancel(self, key: str) -> None:
        """Cancel a task and remove its result from consideration.

        Args:
            key: Task name to cancel if present.
        """
        # A running backend read may ignore cancellation until it returns.
        # Removing its task now prevents even a successful late result from
        # reaching the GUI after a newer request has replaced it.
        task = self._pending.pop(key, None)
        if task is not None:
            task.cancelled.set()
            if task.future is not None:
                task.future.cancel()

    def poll(self) -> None:
        """Deliver completed current operations on the GUI thread."""
        for key, task in list(self._pending.items()):
            # Completion callbacks can submit or cancel other keys during this
            # poll. Recheck identity before using the iteration snapshot.
            if self._pending.get(key) is not task:
                continue
            progress = task.progress
            if progress is not None and progress != task.reported:
                self._progress(key, *progress)
                task.reported = progress
            assert task.future is not None
            if not task.future.done():
                continue
            # Remove before delivery: the callback may submit the next task
            # under this same key, which must survive the current poll.
            del self._pending[key]
            try:
                result = task.future.result()
                task.done(result)
            except CancelledError:
                pass
            except Exception as exc:
                self._error(str(exc))

    def close(self) -> None:
        """Cancel work and stop delivery without blocking the GUI thread.

        In-flight backend I/O finishes under its backend timeout.
        Cancellation prevents further filter batches and delivery of
        obsolete results.
        """
        self._timer.stop()
        for key in list(self._pending):
            self.cancel(key)
        self._executor.shutdown(wait=False, cancel_futures=True)

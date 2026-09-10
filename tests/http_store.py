"""Local HTTP fixture with byte-range reads for remote Zarr tests."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path
from threading import Thread

from aiohttp import web


@contextmanager
def serve_directory(path: Path) -> Iterator[str]:
    """Serve a directory over HTTP, including byte-range requests.

    Args:
        path: Directory containing the store.

    Yields:
        Local HTTP root URL.
    """
    started: Future[tuple[str, asyncio.AbstractEventLoop, asyncio.Event]] = (
        Future()
    )

    async def serve() -> None:
        """Run the HTTP server until the caller signals cleanup."""
        app = web.Application()
        app.router.add_static("/", path, show_index=True)
        runner = web.AppRunner(app)
        try:
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = runner.addresses[0][1]
            stopped = asyncio.Event()
            started.set_result(
                (
                    f"http://127.0.0.1:{port}",
                    asyncio.get_running_loop(),
                    stopped,
                )
            )
            await stopped.wait()
        finally:
            await runner.cleanup()

    def run() -> None:
        """Publish startup exceptions to the caller instead of hanging."""
        try:
            asyncio.run(serve())
        except BaseException as error:
            if not started.done():
                started.set_exception(error)
            else:
                raise

    thread = Thread(target=run, daemon=True)
    thread.start()
    url, loop, stopped = started.result(timeout=10)
    try:
        yield url
    finally:
        loop.call_soon_threadsafe(stopped.set)
        thread.join(timeout=10)
        if thread.is_alive():
            raise RuntimeError("HTTP test server failed to stop")

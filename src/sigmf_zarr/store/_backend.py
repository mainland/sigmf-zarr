"""Store backends whose remote failures surface as `OSError`."""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator, AsyncIterator, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zarr.storage import FsspecStore, WrapperStore

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr.abc.buffer import Buffer
    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import BufferPrototype

try:
    from aiohttp import (  # type: ignore[import-not-found, unused-ignore]
        ClientError as _ClientError,
    )
except ImportError:  # pragma: no cover - fsspec[http] installs aiohttp.
    _BACKEND_ERRORS: tuple[type[Exception], ...] = ()
else:
    _BACKEND_ERRORS = (_ClientError,)
"""Remote library failures that are not already `OSError`.

Fsspec reports most failures as `OSError` subclasses, such as
`FileNotFoundError`, but its HTTP file system passes `aiohttp` failures
through, such as a truncated response body.
"""

_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*")
"""First protocol of a store URL, as in `s3` or `simplecache`."""


@contextmanager
def _translated() -> Iterator[None]:
    """Re-raise a remote library failure as `OSError`.

    Yields:
        Nothing.

    Raises:
        OSError: If the body raises a remote library failure.
    """
    try:
        yield
    except _BACKEND_ERRORS as exc:
        raise OSError(f"{type(exc).__name__}: {exc}") from exc


async def _translated_keys(keys: AsyncIterator[str]) -> AsyncIterator[str]:
    """Yield listed keys, translating remote library failures.

    Args:
        keys: Keys listed by the wrapped store.

    Yields:
        Each listed key.
    """
    with _translated():
        async for key in keys:
            yield key


class OSErrorStore(WrapperStore[FsspecStore]):
    """Fsspec store whose remote library failures raise `OSError`.

    Callers then handle a failed remote read like a failed local read,
    without depending on the libraries behind each fsspec protocol.
    """

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        """Read one value.

        Args:
            key: Store key.
            prototype: Buffer prototype for the result.
            byte_range: Optional byte range to read.

        Returns:
            Value, or ``None`` when the key is absent.
        """
        with _translated():
            return await self._store.get(key, prototype, byte_range)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        """Read several byte ranges.

        Args:
            prototype: Buffer prototype for the results.
            key_ranges: Keys and optional byte ranges.

        Returns:
            Values in request order, ``None`` for absent keys.
        """
        with _translated():
            return await self._store.get_partial_values(prototype, key_ranges)

    async def _get_many(
        self,
        requests: Iterable[tuple[str, BufferPrototype, ByteRequest | None]],
    ) -> AsyncGenerator[tuple[str, Buffer | None], None]:
        """Read several values.

        Args:
            requests: Keys, buffer prototypes, and optional byte ranges.

        Yields:
            Each key with its value, or ``None`` when absent.
        """
        with _translated():
            async for result in self._store._get_many(requests):
                yield result

    async def exists(self, key: str) -> bool:
        """Return whether a key exists.

        Args:
            key: Store key.

        Returns:
            True when the key exists.
        """
        with _translated():
            return await self._store.exists(key)

    async def getsize(self, key: str) -> int:
        """Return the stored size of one value.

        Args:
            key: Store key.

        Returns:
            Size in bytes.
        """
        with _translated():
            return await self._store.getsize(key)

    async def getsize_prefix(self, prefix: str) -> int:
        """Return the total stored size of the values below a prefix.

        Args:
            prefix: Key prefix.

        Returns:
            Size in bytes.
        """
        with _translated():
            return await self._store.getsize_prefix(prefix)

    async def is_empty(self, prefix: str) -> bool:
        """Return whether no key starts with a prefix.

        Args:
            prefix: Key prefix.

        Returns:
            True when no key starts with `prefix`.
        """
        with _translated():
            return await self._store.is_empty(prefix)

    async def set(self, key: str, value: Buffer) -> None:
        """Write one value.

        Args:
            key: Store key.
            value: Value to write.
        """
        with _translated():
            await self._store.set(key, value)

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        """Write one value unless the key exists.

        Args:
            key: Store key.
            value: Value to write.
        """
        with _translated():
            await self._store.set_if_not_exists(key, value)

    async def _set_many(self, values: Iterable[tuple[str, Buffer]]) -> None:
        """Write several values.

        Args:
            values: Keys and values to write.
        """
        with _translated():
            await self._store._set_many(values)

    async def delete(self, key: str) -> None:
        """Delete one key.

        Args:
            key: Store key.
        """
        with _translated():
            await self._store.delete(key)

    async def delete_dir(self, prefix: str) -> None:
        """Delete every key below a prefix.

        Args:
            prefix: Key prefix.
        """
        with _translated():
            await self._store.delete_dir(prefix)

    async def clear(self) -> None:
        """Delete every key."""
        with _translated():
            await self._store.clear()

    def list(self) -> AsyncIterator[str]:
        """List every key.

        Returns:
            Asynchronous key iterator.
        """
        return _translated_keys(self._store.list())

    def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        """List the keys that start with a prefix.

        Args:
            prefix: Key prefix.

        Returns:
            Asynchronous key iterator.
        """
        return _translated_keys(self._store.list_prefix(prefix))

    def list_dir(self, prefix: str) -> AsyncIterator[str]:
        """List the immediate children of a prefix.

        Args:
            prefix: Key prefix.

        Returns:
            Asynchronous key iterator.
        """
        return _translated_keys(self._store.list_dir(prefix))


def backend_store(
    location: str | Path,
    *,
    read_only: bool,
    storage_options: dict[str, Any] | None,
) -> str | Path | OSErrorStore:
    """Open the fsspec backend for a store URL with translated failures.

    Zarr opens a local path or `file` URL as a `LocalStore`, whose failures
    are already `OSError`, and a `memory` URL in memory. Every other URL
    uses an fsspec backend, which is wrapped so that remote library failures
    raise `OSError`.

    Args:
        location: Local path or store URL.
        read_only: Whether the store permits reads only.
        storage_options: Optional backend-specific storage options.

    Returns:
        `location` for a local path or a `file` or `memory` URL, otherwise a
        wrapped fsspec store.
    """
    if isinstance(location, Path):
        return location
    scheme = _SCHEME.match(location)
    if scheme is None or scheme.group().lower() in {"file", "memory"}:
        return location
    return OSErrorStore(
        FsspecStore.from_url(
            location, storage_options=storage_options, read_only=read_only
        )
    )

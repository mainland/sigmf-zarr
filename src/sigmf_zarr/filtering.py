"""Explicit item filtering over independent indexes and JSON metadata."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import CancelledError
from typing import Any

import numpy as np
import numpy.typing as npt

from sigmf_zarr import query as _query
from sigmf_zarr.query import IndexFilter, IndexQuery, compile_query
from sigmf_zarr.store import SigMFRecording

FilterOperator = _query.FilterOperator
"""Comparison operator of an `IndexFilter`.

The values are `"=="`, `"!="`, `"<"`, `"<="`, `">"`, `">="`, and `"in"`.
This alias is `sigmf_zarr.query.FilterOperator`.
"""


def _compile_query(expression: str | None) -> Any:
    """Compile an optional JSON predicate without importing GUI packages.

    Args:
        expression: JMESPath expression, or None to omit JSON reads.

    Returns:
        Compiled expression, or None.

    Raises:
        ImportError: If JMESPath is not installed.
        ValueError: If the expression is empty or syntactically invalid.
    """
    if expression is None:
        return None
    try:
        import jmespath
    except ImportError as exc:
        raise ImportError("JSON filtering requires sigmf-zarr[query]") from exc
    if not expression.strip():
        raise ValueError("Metadata query must not be empty")
    try:
        return jmespath.compile(expression)
    except jmespath.exceptions.JMESPathError as exc:
        raise ValueError(f"Invalid metadata query: {exc}") from exc


def _metadata_matches(query: Any, metadata: object, item: int) -> bool:
    """Evaluate a JSON predicate with explicit Boolean result semantics.

    Args:
        query: Compiled JMESPath expression.
        metadata: Resolved item JSON.
        item: Original position used in diagnostics.

    Returns:
        True only for Boolean true. Null is treated as no match.

    Raises:
        ValueError: If evaluation fails or produces a non-Boolean value.
    """
    import jmespath

    try:
        result = query.search(metadata)
    except jmespath.exceptions.JMESPathError as exc:
        raise ValueError(
            f"Metadata query failed at item {item}: {exc}"
        ) from exc
    if result is not None and not isinstance(result, bool):
        raise ValueError(
            f"Metadata query at item {item} must return Boolean or null"
        )
    return result is True


def filter_items(
    recording: SigMFRecording,
    filters: Sequence[IndexFilter] = (),
    *,
    index_query: str | IndexQuery | None = None,
    metadata_query: str | None = None,
    batch_size: int = 4096,
    cancelled: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> npt.NDArray[np.int64]:
    """Select original item positions using index and JSON predicates.

    Index conditions are combined with AND before JSON is read for surviving
    candidates. Samples are never read. Results are ascending positions in the
    recording, not a persisted split or a claim of semantic freshness.

    Args:
        recording: Batched recording, preferably opened structurally.
        filters: Explicit comparisons against named item indexes.
        index_query: Optional core index query, combined with filters by AND.
        metadata_query: Optional JMESPath Boolean expression evaluated against
            resolved item metadata. Null means no match. Other result types
            and evaluation errors fail the operation.
        batch_size: Maximum number of index values evaluated per batch.
        cancelled: Optional cooperative cancellation callback.
        progress: Optional callback receiving examined and total item counts.

    Returns:
        Ascending original item positions. Memory is proportional to the
        matches, one batch and one stored chunk of index values, the encoded
        item JSON of one stored chunk, the recording's shared metadata, and
        one decoded item. Each stored chunk is decoded once. Shared metadata
        is not copied for each item.

    Raises:
        ValueError: If arguments, selected indexes, or the query are invalid.
        ImportError: If JSON filtering is requested without JMESPath.
        concurrent.futures.CancelledError: If cancellation is requested. No
            partial result is returned.
    """
    query = _compile_query(metadata_query)
    compiled = (
        compile_query(index_query)
        if isinstance(index_query, str)
        else index_query or IndexQuery()
    )
    batches = compiled._iter_batches(
        recording,
        filters=filters,
        batch_size=batch_size,
        cancelled=cancelled,
        progress=progress,
    )
    if query is None:
        matches = [candidates for candidates in batches if candidates.size]
    else:
        matches = list(
            _json_matches(recording, query, batches, cancelled=cancelled)
        )
    _check_cancelled(cancelled)
    return np.concatenate(matches) if matches else np.empty(0, dtype=np.int64)


def _json_matches(
    recording: SigMFRecording,
    query: Any,
    batches: Iterable[npt.NDArray[np.int64]],
    *,
    cancelled: Callable[[], bool] | None,
) -> Iterator[npt.NDArray[np.int64]]:
    """Evaluate a JSON predicate over index candidates, chunk by chunk.

    Zarr decodes a whole `item_metadata` chunk to return any entry in it.
    Candidates are therefore held until every candidate in their chunk is
    known, so each chunk is decoded once even when it spans many batches.

    Args:
        recording: Batched source recording.
        query: Compiled JMESPath expression.
        batches: Ascending index candidates for contiguous item batches.
        cancelled: Optional cooperative cancellation callback.

    Yields:
        Ascending accepted positions, in nonempty arrays.

    Raises:
        ValueError: If item JSON is malformed or the predicate fails.
        concurrent.futures.CancelledError: If cancellation is requested.
    """
    chunk = (
        max(int(recording.item_metadata_array.chunks[0]), 1)
        if recording.has_item_metadata
        else 1
    )

    def accept(candidates: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        """Return the candidates whose resolved metadata matches.

        Args:
            candidates: Ascending candidate positions.

        Returns:
            Accepted positions.
        """
        accepted = []
        # Resolve one item at a time. JMESPath does not modify its input, so
        # the resolved views share the recording metadata instead of copying
        # it for every item.
        for item, metadata in recording._resolved_item_metadata_views(
            candidates.tolist()
        ):
            _check_cancelled(cancelled)
            if _metadata_matches(query, metadata, item):
                accepted.append(item)
        return np.asarray(accepted, dtype=np.int64)

    pending: list[npt.NDArray[np.int64]] = []
    for candidates in batches:
        if not candidates.size:
            continue
        last = int(candidates[-1]) // chunk
        if pending and int(pending[0][0]) // chunk == last:
            pending.append(candidates)
            continue
        # Earlier chunks are complete. Concatenation happens only when the
        # candidates reach a new chunk, so its total cost stays linear.
        waiting = np.concatenate([*pending, candidates])
        split = int(np.searchsorted(waiting, last * chunk))
        accepted = accept(waiting[:split])
        if accepted.size:
            yield accepted
        pending = [waiting[split:]]
    if pending:
        accepted = accept(np.concatenate(pending))
        if accepted.size:
            yield accepted


def _check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    """Raise when a caller requests cooperative cancellation.

    Args:
        cancelled: Optional cancellation callback.

    Raises:
        concurrent.futures.CancelledError: If cancellation was requested.
    """
    if cancelled is not None and cancelled():
        raise CancelledError("Filtering cancelled")


__all__ = ["FilterOperator", "IndexFilter", "filter_items"]

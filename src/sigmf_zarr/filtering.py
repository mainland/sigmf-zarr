"""Explicit item filtering over independent indexes and JSON metadata."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import CancelledError
from typing import Any

import numpy as np
import numpy.typing as npt

from sigmf_zarr.query import (
    FilterOperator,
    IndexFilter,
    IndexQuery,
    compile_query,
)
from sigmf_zarr.store import SigMFRecording


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
        raise ImportError(
            "JSON filtering requires sigmf-zarr[query]"
        ) from exc
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
        batch_size: Maximum number of index values read per batch.
        cancelled: Optional cooperative cancellation callback.
        progress: Optional callback receiving examined and total item counts.

    Returns:
        Ascending original item positions. Memory is proportional to matches
        plus one batch, rather than all decoded metadata.

    Raises:
        ValueError: If arguments, selected indexes, or the query are invalid.
        ImportError: If JSON filtering is requested without JMESPath.
        CancelledError: If cancellation is requested. No partial result is
            returned.
    """
    query = _compile_query(metadata_query)
    compiled = (
        compile_query(index_query)
        if isinstance(index_query, str)
        else index_query or IndexQuery()
    )
    matches: list[npt.NDArray[np.int64]] = []
    for candidates in compiled._iter_batches(
        recording,
        filters=filters,
        batch_size=batch_size,
        cancelled=cancelled,
        progress=progress,
    ):
        if query is not None:
            accepted = []
            # Resolve only surviving index candidates. Keeping JSON work inside
            # this batch avoids materializing metadata for the whole recording.
            metadata_batch = recording.resolved_item_metadata_batch(
                [int(item) for item in candidates]
            )
            for item, metadata in zip(candidates, metadata_batch, strict=True):
                _check_cancelled(cancelled)
                if _metadata_matches(
                    query,
                    metadata,
                    int(item),
                ):
                    accepted.append(item)
            candidates = np.asarray(accepted, dtype=np.int64)
        if candidates.size:
            matches.append(candidates)
    _check_cancelled(cancelled)
    return np.concatenate(matches) if matches else np.empty(0, dtype=np.int64)


def _check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    """Raise when a caller requests cooperative cancellation.

    Args:
        cancelled: Optional cancellation callback.

    Raises:
        CancelledError: If cancellation was requested.
    """
    if cancelled is not None and cancelled():
        raise CancelledError("Filtering cancelled")


__all__ = ["FilterOperator", "IndexFilter", "filter_items"]

"""Adapt shared signal coordinates to viewer regions."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from sigmf_zarr.signals import (
    AxisMode as AxisMode,
)
from sigmf_zarr.signals import (
    CaptureSegment as CaptureSegment,
)
from sigmf_zarr.signals import (
    _item_capture_index,
    _offset,
)
from sigmf_zarr.signals import (
    capture_segments as capture_segments,
)
from sigmf_zarr.signals import (
    item_capture_metadata as item_capture_metadata,
)
from sigmf_zarr.signals import (
    project_samples as project_samples,
)
from sigmf_zarr.signals import (
    sample_rate as sample_rate,
)
from sigmf_zarr.viewer.regions import Region, Span


def sigmf_regions(
    metadata: Mapping[str, object],
    length: int,
    *,
    scope: str = "recording",
    source_metadata: Mapping[str, object] | None = None,
    capture_item: int | None = None,
) -> tuple[Region, ...]:
    """Adapt capture and annotation entries into named-axis regions.

    Args:
        metadata: SigMF metadata bundle in original coordinates.
        length: Stored signal length.
        scope: Source prefix used in stable region identifiers.
        source_metadata: Original scope supplying entries. Timing and offsets
            still come from the resolved metadata.
        capture_item: Item position when source captures use item offsets.
            Only the applicable shared capture produces a region.

    Returns:
        Regions preserving original metadata. Entries outside the recording
        or with invalid intervals have no drawable region.
    """
    segments = capture_segments(metadata, length)
    offset = _offset(metadata)
    result = []
    for kind, field in (
        ("capture", "captures"),
        ("annotation", "annotations"),
    ):
        # Timing uses resolved metadata, but region identity and inspected JSON
        # come from the original scope to distinguish shared and item entries.
        entries = (
            metadata if source_metadata is None else source_metadata
        ).get(field, [])
        selected = (
            _item_capture_index(entries, capture_item)
            if kind == "capture" and capture_item is not None
            else None
        )
        for index, entry in enumerate(
            entries if isinstance(entries, list) else []
        ):
            if (
                not isinstance(entry, dict)
                or type(entry.get("core:sample_start")) is not int
            ):
                continue
            if kind == "capture" and capture_item is not None:
                if index != selected:
                    continue
                start = 0
            else:
                start = entry["core:sample_start"] - offset
            end = next(
                (
                    segment.stop
                    for segment in segments
                    if segment.start <= max(0, start) < segment.stop
                ),
                length,
            )
            count = entry.get("core:sample_count")
            if kind == "annotation" and type(count) is int:
                end = start + count
            lo, hi = max(0, start), min(length, end)
            if lo >= hi:
                continue
            bounds = {"sample": Span(lo, hi)}
            lower, upper = (
                entry.get("core:freq_lower_edge"),
                entry.get("core:freq_upper_edge"),
            )
            if (
                kind == "annotation"
                and isinstance(lower, (int, float))
                and isinstance(upper, (int, float))
                and np.isfinite([lower, upper]).all()
                and lower < upper
            ):
                reference = (
                    "RF"
                    if any(
                        s.frequency not in (None, 0)
                        and s.start <= max(0, start) < s.stop
                        for s in segments
                    )
                    else "baseband"
                )
                bounds["frequency"] = Span(lower, upper, "Hz", reference)
            label = str(
                entry.get(
                    "core:description",
                    entry.get(
                        "core:label", f"{kind.capitalize()} {index + 1}"
                    ),
                )
            )
            result.append(
                Region(
                    f"{scope}/{field}/{index}",
                    bounds,
                    label,
                    kind,
                    dict(entry),
                )
            )
    return tuple(result)

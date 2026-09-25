"""Named-axis regions independent of signal formats, storage, and GUI APIs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Span:
    """Half-open interval in an explicitly identified coordinate system."""

    start: int | float
    """Inclusive lower coordinate."""

    stop: int | float
    """Exclusive upper coordinate."""

    unit: str = "sample"
    """Coordinate unit, such as sample or Hz."""

    reference: str = "recording"
    """Coordinate reference, such as recording, RF, or baseband."""

    def __post_init__(self) -> None:
        """Reject reversed or empty intervals.

        Raises:
            ValueError: If the endpoints do not define a nonempty interval.
        """
        if not self.start < self.stop:
            raise ValueError("Region intervals must be nonempty")


@dataclass(frozen=True)
class Region:
    """Labeled region with optional bounds on arbitrary named axes.

    An omitted axis is unrestricted. Source metadata is retained for
    inspection and is never rewritten to reflect display clipping or
    projection.
    """

    key: str
    """Source identifier, unique within the displayed signal."""

    bounds: Mapping[str, Span]
    """Bounds by axis name."""

    label: str = ""
    """Human-readable label."""

    kind: str = "annotation"
    """Consumer-defined category used for styling and visibility."""

    metadata: Mapping[str, object] = field(default_factory=dict)
    """Original source attributes for inspection."""

    def intersects(
        self, axis: str, start: int | float, stop: int | float
    ) -> bool:
        """Test overlap with an interval on a named axis.

        Args:
            axis: Axis to inspect.
            start: Inclusive query coordinate.
            stop: Exclusive query coordinate.

        Returns:
            Whether the region overlaps or is unrestricted on this axis.
        """
        span = self.bounds.get(axis)
        return span is None or span.start < stop and start < span.stop

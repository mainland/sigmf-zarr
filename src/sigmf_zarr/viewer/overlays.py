"""Matplotlib rendering of regions in projected display coordinates."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, cast

from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Bbox, IdentityTransform

from sigmf_zarr.viewer.regions import Region


class RegionPatch(Rectangle):
    """Rectangle retaining its source region for inspection and hover."""

    region: Region
    """Projected region with unmodified source metadata."""

    def draw(self, renderer: RendererBase) -> None:
        """Draw annotation outlines one pixel wide at the rendering scale.

        Args:
            renderer: Renderer defining the conversion from points to pixels.
        """
        if self.region.kind != "capture":
            # Line widths use points. Recompute at draw time so one pixel stays
            # one pixel after a DPI change or rendering to a different backend.
            self.set_linewidth(1 / cast(float, renderer.points_to_pixels(1)))
        super().draw(renderer)


class RegionLabel(Text):
    """Description drawn only when it fits inside the visible region."""

    patch: RegionPatch
    """Region whose visible bounds determine label placement."""

    def draw(self, renderer: RendererBase) -> None:
        """Place the label inside the clipped region when space permits.

        Args:
            renderer: Renderer providing actual text and region dimensions.
        """
        if self.axes is None or not self.patch.get_visible():
            return
        # Measure in display pixels on every draw: zoom changes the visible
        # part of the patch, and font metrics depend on the active renderer.
        bounds = Bbox.intersection(
            self.patch.get_window_extent(renderer), self.axes.bbox
        )
        if bounds is None:
            return
        self.set_position((bounds.x0 + 5, bounds.y1 - 5))
        text_bounds = self.get_window_extent(renderer)
        if (
            bounds.width >= text_bounds.width + 10
            and bounds.height >= text_bounds.height + 10
        ):
            super().draw(renderer)


def draw_regions(
    axes: Axes,
    regions: Iterable[Region],
    *,
    orientation: Literal["horizontal", "vertical"] = "horizontal",
) -> None:
    """Draw pickable regions without changing plot limits.

    Args:
        axes: Target axes.
        regions: Regions with a projected `position` bound and optional
            `frequency` bound already expressed in axis coordinates.
        orientation: Direction of the position axis.
    """
    xlim, ylim = axes.get_xlim(), axes.get_ylim()
    for region in regions:
        span = region.bounds.get("position")
        if span is None:
            continue
        color = "tab:blue" if region.kind == "capture" else "tab:orange"
        frequency = region.bounds.get("frequency")
        # Without a frequency bound, use a blended transform: position remains
        # in data units while the other coordinate spans the full axes (0..1).
        low, high = (frequency.start, frequency.stop) if frequency else (0, 1)
        if orientation == "vertical":
            xy = (low, span.start)
            width, height = high - low, span.stop - span.start
            transform = (
                axes.transData if frequency else axes.get_yaxis_transform()
            )
        else:
            xy = (span.start, low)
            width, height = span.stop - span.start, high - low
            transform = (
                axes.transData if frequency else axes.get_xaxis_transform()
            )
        annotation = region.kind != "capture"
        artist = RegionPatch(
            xy,
            width,
            height,
            transform=transform,
            facecolor=to_rgba(color, 0.1 if annotation else 0.04),
            edgecolor=to_rgba(color, 1.0 if annotation else 0.95),
            linewidth=1 if annotation else 0.8,
            zorder=3 if annotation else 2,
        )
        artist.region = region
        axes.add_patch(artist)
        if annotation and region.label:
            label = RegionLabel(
                text=region.label,
                color="white",
                fontsize=8,
                verticalalignment="top",
                transform=IdentityTransform(),
                clip_on=True,
                zorder=4,
                bbox={
                    "facecolor": "black",
                    "alpha": 0.8,
                    "edgecolor": "none",
                    "pad": 2,
                },
            )
            label.patch = artist
            axes.add_artist(label)
        artist.set_gid(region.key)
        artist.set_picker(True)
        if region.kind == "capture":
            if orientation == "vertical":
                axes.axhline(span.start, color=color, linewidth=0.7)
            else:
                axes.axvline(span.start, color=color, linewidth=0.7)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)

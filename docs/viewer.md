# Dataset viewer

The optional desktop viewer uses Qt Widgets through PySide6 and embeds
Matplotlib figures. It opens datasets read-only and requires no schema changes.

## Installation and launch

```bash
pip install 'sigmf-zarr[viewer]'
sigmf-zarr view radioml2016.10a.zarr
```

For development in the repository environment:

```bash
uv sync --extra dev --extra docs --extra viewer
uv run --extra viewer sigmf-zarr view radioml2016.10a.zarr
```

A local graphical session is required. Store URLs use the same backend support
as the library. Install any additional backend dependencies separately.

## Browse records and individual items

Use the tree on the left to expand **Collections** or **Recordings**. Each
collection expands to its recording references in collection order. The
**Recordings** branch lists every recording, including recordings outside a
collection. A recording may appear under several collections. Those entries
refer to the same recording. Missing references are marked and disabled.
Selecting the **Collections** or **Recordings** heading clears the previous
recording display.

Recordings are leaves in the tree. Selecting a recording opens its item browser
and plots. Use the tabs to inspect metadata and indexes. Switching between
references to the same recording preserves its selected item and filters.
Selecting a collection displays its metadata and member references. Collection
and category selections hide recording controls, item navigation, and waveform
panels. These tree
operations read collection attributes and recording descriptors, not samples or
per-item JSON. Individual items are selected with a slider and do not become
tree nodes.

For batched recordings, drag the horizontal slider to select one item in the
leading `item` axis. Use **Previous item**, **Next item**, or **Alt+Left** and
**Alt+Right** to step through individual items. The slider also supports arrow
keys and Home/End. Enter a one-based **Match** position to jump precisely within
the filtered selection. The original, zero-based item position appears beside
the matching item count. There are no browser pages.

Select named indexes under **Displayed indexes** to show their values above the
plots. Raw category IDs are followed by their decoded labels. Displayed index
values and JSON metadata are independent snapshots. A matching `field` attribute
does not cause one representation to replace the other.

Slider movement updates the selected position immediately. Sample reads begin
after a 100 ms pause, so dragging does not load every intermediate item. Old
plots and values are cleared when the selection changes, and obsolete worker
results are discarded. Visualization tabs remain hidden until a sample window
is available. The selected visualization tab is restored after loading unless
another tab is selected while the read is pending. An empty selection clears the display and disables item
navigation. A single item or continuous recording needs no item slider movement.

Select positions for additional named axes, such as `channel`. Each newly
selected item or channel initially displays its entire signal. The viewer
recognizes either native complex samples or a real `iq` axis of length two,
interpreted as I followed by Q. It uses axis names instead of assuming a tensor
layout. Continuous, unbatched recordings support the same window controls and
appear as one browsable record. Item filters apply only to batched recordings.

## Select a time interval

The timeline below the plots initially shows the entire signal as a
peak-magnitude overview. Its highlighted selection controls all four views.
The overview has an independent visible interval that can be zoomed and panned
without changing the selected samples. The item slider selects which signal
to inspect. The timeline selects the interval within it.
The timeline appears only for sample visualization panels, including while a
new interval loads. Metadata and index tabs hide it. Switching tabs preserves
the selected interval. The item slider remains available for browsing per-item
metadata.

- Drag either edge to resize the interval.
- Drag inside the interval to move it without changing its duration.
- Click outside the interval to center it on that position.
- Use the mouse wheel to zoom the overview around the pointer, or `+` and `-`
  to zoom around its midpoint.
- Use the scrollbar below the overview to pan its visible interval.
- Select **Zoom to selection** to fill the overview with the selected interval.
- Select **Fit recording**, or press Home with the timeline focused, to show
  the entire recording without changing the selection.
- Use arrow keys to move the selection or Shift+arrows to resize its right edge.
- Select **Select full recording** to display all samples in the waveform panels.

The label below the timeline reports exact sample positions using an inclusive
start and exclusive stop. The selection may extend outside the visible
overview interval. Reads begin after a 100 ms pause in navigation. Completing
a time-axis pan or zoom in the Time or Spectrogram panel updates the scrubber
selection and loads that interval for all panels. Both panels share the visible
time range. Frequency-only zoom does not change the sample selection. Moving
the scrubber preserves the spectrogram's frequency zoom. Plot navigation pans
or expands the overview only when needed to reveal the selected interval.

In capture timestamp mode, navigation selects samples from every intersecting
capture. If the matching samples form disjoint ranges, the scrubber selects
their smallest containing interval. A gesture entirely inside a timestamp gap
restores the previous sample interval and reports that no samples match.

Overview navigation requests up to 1024 peak bins for the visible interval.
Every finite sample in that interval contributes to the envelope. The scan
uses blocks of at most 65536 samples and does not compute spectra or metrics.
This provides finer detail when zooming in on short bursts. Background reads
are cancelled and obsolete results discarded when the viewport, item, or
recording changes. **Fit recording** reuses the cached full-signal overview.

The timeline retains the 128 most recently used detail envelopes with strong
references. Revisiting an exact cached interval restores its envelope
immediately without a delay or sample read. Uncached intervals display the
finest cached envelope that covers them, or the full-recording envelope, while
finer detail loads. Detail retention uses at most 4 MiB of array storage. The
full-recording envelope is retained separately. Selecting another recording,
item, or channel clears the cache, including when the signal length is unchanged.

Selections of at most 65536 samples use the original samples. Larger selections
use an explicit **Reduced overview**, computed in a worker from consecutive
blocks of at most 65536 samples. The calculation scans the whole selected
interval and does not load it all into memory:

- The Time view preserves per-bin I and Q minima and maxima.
- The Spectrogram view averages linear windowed FFT bin power into at most
  512 time bins before converting it to dB.
- The Frequency view averages linear power over all finite complete FFT frames.
- The I/Q view displays at most 4096 uniformly selected original samples and
  labels the sampled view.
- The built-in scalar metrics include every source sample.

The full-signal overview is cached for the selected item and channel. Its
initial scan can take time for large or remote recordings. A narrower interval
can load while that scan continues. Selecting another item cancels the previous
scan between blocks. The timeline and plots remain hidden for collection and
category selections.

## Captures, annotations, and axis modes

**Plot axis** selects the coordinate system used by the Time and Spectrogram
panels:

- **Sample index** uses continuous stored sample positions and is the default.
- **Elapsed sample time** divides stored sample positions by the sample rate.
  It does not represent acquisition gaps.
- **Capture timestamp (UTC)** uses the timestamp declared by each capture.
  Gaps, overlapping captures, and backward jumps are preserved. Captures with
  unknown timestamps are omitted and the plot reports missing timing.

Time modes require a known sample rate. Timestamp mode also requires at least
one timestamp. Legacy timestamp strings without timezone information are
interpreted as UTC. The scrubber always selects stored sample-index intervals,
independent of the displayed plot axis. Capture fields do not carry forward
to later captures. Duplicate capture starts use the last entry in the resolved
metadata for timing, including item-level entries.

**Captures** and **Annotations** independently control overlays. The overview
has a blue capture band and an orange annotation band. Time plots show sample
spans. Spectrograms show time-frequency rectangles when frequency bounds are
available, or full-height spans otherwise. Absolute RF bounds are converted
using each applicable capture's center frequency. RF rectangles with unknown
center frequency, or frequency bounds without a known sample rate, remain
available in the inspector but are omitted from the spectrogram.

Annotations use a solid, one-pixel outline and a fill with 10% opacity.
Hover over a region to see its original JSON.
Descriptions appear inside annotation boxes when the visible area is large
enough to fit the text. Labels fall back to `core:label` when
`core:description` is absent. Label visibility updates with zoom and resizing.

Double-click an overview band or plot overlay to open **Captures and
annotations**. A single click does not open the inspector.
The inspector lists region labels and shows original JSON with its recording
or item source identifier. Double-click a list entry or select **Select interval**
to display the region's sample range in the plots. Overlapping regions remain
individually accessible in the list. This interface is read-only.

Rendering splits annotations at capture boundaries without splitting or
rewriting their source metadata. Waveform lines and FFT frames do not cross
capture boundaries, even when overlays are hidden or the sample axis is
selected. Reduced time and spectrogram bins spanning a capture boundary are
omitted from those plots. Zoom in for finer detail at a boundary. Scalar
metrics and the overview peak envelope still include every selected sample.

## Inspect metadata scopes

The Indexes tab is hidden when the recording has no indexes. The Captures and
annotations tab is hidden when the selected item has no capture or annotation
regions. Stored recording regions also count toward this visibility check.

The metadata tabs preserve the stored scopes:

| Tab | Contents |
| --- | --- |
| Recording metadata | Stored recording attributes, including shared `global`, `captures`, and `annotations` |
| Item metadata | The selected item's stored JSON bundle, without shared fields |
| Indexes | Explicitly selected index values, decoded labels, and array attributes |
| Channel metadata | Attributes for the channel selected by the coordinate control |
| Collection metadata | The selected collection's metadata and recording references |

Item metadata appears only for batched recordings. An absent item metadata
array displays **No stored item metadata.** An empty stored bundle displays
`{}`. Channel metadata appears only for recordings with an explicit `channel`
axis. Collection selection displays its own metadata tab. Recording metadata
remains available when no items match a filter.

Enable **Show resolved item metadata** to inspect the derived **Resolved item
metadata** tab. Item `global` fields shallowly override shared fields in that
view. Item captures and annotations append to shared lists. This view does not
include collection metadata, channel metadata, or index values.

## Filter items

Enter an expression in the multiline **Index query** editor, then select
**Apply query** or press **Ctrl+Enter** in the editor. Enter inserts a newline.
For example:

```text
label(mod_class_id) in ["BPSK", "QPSK"]
and snr_db >= 10
```

The core [index query language](query.md) supports comparisons, membership,
Boolean operators, and parentheses. Queries can reference any supported item
index, independently of the **Displayed indexes** selection. Use
`index("splits/example")` for names containing punctuation. Compilation and
evaluation run in a worker.

Viewer queries evaluate index arrays in batches without resolving per-item
JSON metadata. Nonfinite index values remain unknown under comparisons and
negation. Invalid expressions fail the operation and preserve the previous
completed selection. Index predicates never fall back to JSON.

The applied definition appears below the query editor. Editing an expression
does not change the completed selection until the next successful application.
Clear the editor, then apply to restore all items. **Cancel** preserves the
previous completed selection and discards the pending result. Cancellation is
checked between index batches. In-flight backend reads finish under their backend
configuration and timeout.

## Plots and metrics

The default panels are **Spectrogram**, **Time**, **Frequency**, and **I/Q**,
in that order. **Time** shows the real and imaginary components. **Frequency**
shows a two-sided spectrum. **I/Q** plots Q against I in a square plot area with
equal amplitude scales on both axes, including after resizing or zooming.
Matplotlib's toolbar provides pan, zoom, and figure export. Source samples are
not normalized, resampled, or synchronized to symbol timing.

For raw selections, the spectrum applies the selected window $w[n]$ before
the discrete Fourier transform (DFT), normalized by the sum of its coefficients:

$$
P[k] = \left|\frac{\operatorname{DFT}(xw)[k]}{\sum_{n=0}^{N-1} w[n]}\right|^2.
$$

The sample axis follows **Plot axis**. Frequency axes use offsets in Hz when
the selected metadata provides `core:sample_rate`, or cycles per sample
otherwise. Bin power is displayed as
$10\log_{10}(\max(P[k], 10^{-30}))$, relative to one stored amplitude unit
squared. This is bin power, not a calibrated power spectral density. The viewer
does not infer physical power, SNR, or sample rate from a dataset label.

The **Spectrogram** panel displays the selected sample or time coordinate
horizontally and frequency increasing upward. Center frequency is not added
to the frequency axis. Windows spanning several captures use independent
spectrogram images. The Frequency panel averages windowed frame powers
when the raw selection spans multiple captures, rather than transforming
across an acquisition boundary.

Use **FFT** and **Overlap** to adjust frequency and time resolution. **Auto**
chooses a power-of-two FFT length up to 256, reduced for short items. Explicit
FFT lengths are clipped to the available samples. The calculation uses complete
frames without zero padding. **Window** selects periodic Hann (the default),
Hamming, Blackman, Blackman-Harris, or rectangular coefficients for both
spectral panels. Blackman-Harris uses the minimum four-term window, identified
as `blackmanharris` in the Python API.
Rectangular applies equal weight to every sample. Each transform is
divided by the sum of the window coefficients before converting squared
magnitude to dB in stored amplitude units squared. Nonfinite frames are masked.

**Range** sets the color range below **Maximum**. **Auto level** tracks the
selected window's peak bin power. Disable it to keep a fixed scale while
browsing items. All-zero windows remain dark with the default scale. Color-scale
controls redraw the loaded data without reading the dataset again. FFT,
overlap, and window controls also work locally for raw selections. For reduced
overviews, changing any of these settings starts a new bounded scan of the
selected interval. Cached spectral summaries include the selected window.
FFT, overlap, window, and color-scale changes preserve the visible time and frequency
ranges. Reduced overviews remain visible while replacement FFT results load.
The toolbar's Home button restores the full plot after a settings change.
FFT frames may cross read-block boundaries but never capture boundaries.

**Colormap** offers `magma`, `viridis`, `inferno`, `cividis`, and `gray`.
The default is `magma`. Changing it preserves the
zoom and color limits without reading samples or recomputing FFTs. The reusable
`spectrogram()` and `overview_spectrogram()` functions accept a `cmap` keyword
with a registered name or a Matplotlib colormap instance.

Metrics report mean squared magnitude, root mean square magnitude, and peak
magnitude over the selected window. They use stored amplitude units. JSON
captures and annotations remain in their original coordinates and are not
clipped or projected into the selected window.

## Reuse filtering without Qt

Numeric index filtering requires only the base package. JSON filtering remains
available through the Python API. Install `sigmf-zarr[query]` for optional
JMESPath support. The viewer extra does not include JMESPath.

```python
from sigmf_zarr import SigMFZarrStore
from sigmf_zarr.filtering import IndexFilter, filter_items

store = SigMFZarrStore.open("radioml2016.10a.zarr")
recording = store.recordings.open("radioml2016", validation="structural")
positions = filter_items(
    recording,
    [
        IndexFilter("mod_class_id", "in", ["BPSK", "QPSK"], decode=True),
        IndexFilter("snr_db", ">=", 0),
    ],
)
print(positions)
```

Use the actual recording and index names shown by the browser. `filter_items()`
returns ascending original positions and does not persist a split or modify
metadata. It scans selected indexes in bounded batches and reads JSON only for
survivors. It never reads samples. Its `progress` and `cancelled` callbacks have
no GUI dependencies. The returned positions are an independent selection
snapshot. They do not update when a recording changes.

`SigMFRecording.resolved_item_metadata_batch()` resolves explicitly selected
items in one array selection, preserving order and duplicates. The filter uses
this method to avoid repeatedly decoding the same physical metadata chunk.

## Extend the desktop application

`DatasetSource` provides the collection/recording catalog, descriptor discovery,
browser pages, filtering, and
sample-window reads without importing Qt or Matplotlib. Each operation owns a
read-only backend handle and closes it before returning.

`DatasetSource.read_overview()` returns an `OverviewWindow` for a large interval.
Its `context` contains coordinates and metadata with an empty sample vector.
Its `signal` contains the reduced data. It accepts cooperative `cancelled` and
`progress` callbacks, checked between read blocks.

`DatasetViewer` is an embeddable Qt widget. `add_panel(title, widget, update)`
accepts any Qt widget and an update callback. The callback receives a
`SampleWindow`, or `None` when the selection is cleared. `MatplotlibPanel`
provides a figure, canvas, and toolbar around a drawing function. No plugin
registration files or base panel subclasses are required.

Pass `overview_update=` to `add_panel()` to handle `OverviewWindow` separately.
Raw callbacks receive `None` when the selection is cleared and are called with
a `SampleWindow` only for raw selections. Plot navigation and FFT updates keep
the previous data visible until replacement results arrive. Other selection
changes clear the panels first. Panels without an overview callback are
hidden for large selections. `MatplotlibPanel(draw, draw_overview=...)` provides
separate drawing callbacks. Register `panel.set_overview` as its overview
callback.

`SpectrogramPanel` includes the spectrogram controls and can be embedded in
another Qt application. The drawing function
`sigmf_zarr.viewer.plots.spectrogram()` also accepts a Matplotlib figure and
`SampleWindow` directly, without importing Qt.

The following components can be reused without dataset access:

- `sigmf_zarr.viewer.regions.Region` and `Span` describe labeled half-open
  intervals on named axes. An omitted axis is unrestricted. Each span records
  its units and coordinate reference. These types have no GUI or SigMF
  dependency.
- `sigmf_zarr.viewer.captures` adapts SigMF metadata into source regions and
  acquisition segments and provides piecewise sample-to-time projection.
  `sigmf_zarr.viewer.presentation.PlotOptions` configures standalone plot
  axis modes and overlay visibility. `sample_interval()` in the same module
  maps visible coordinates back to a containing stored sample interval.
- `sigmf_zarr.viewer.overlays.draw_regions()` renders projected regions in
  Matplotlib. Picked patches expose their source identifiers through `gid`.
  `RegionPatch.region` retains the metadata used by hover inspection.
  `RegionLabel` decides whether a description fits during each render.
  `sigmf_zarr.viewer.region_qt.RegionBrowser` provides a reusable read-only
  Qt inspector and emits `interval_selected` for sample-range navigation.

- `sigmf_zarr.viewer.timeline.RangeSelector` is a standalone Qt widget over a
  nonnegative integer extent. It exposes `set_extent()`, `set_range()`,
  `selection`, and `fit()` for sample selection. Overview navigation uses
  `set_visible_range()`, `visible_range`, `fit_view()`, and
  `zoom_to_selection()`. `reveal_selection()` pans or expands the overview
  only when the selection is outside it. The independent `range_changed` and
  `visible_range_changed` signals emit Python integer start and stop
  positions, including values beyond 32 bits. Consumers supply peak data
  using `set_overview(peaks, start=..., stop=...)`. Peak updates do not change
  either range. The widget has no SigMF, Zarr, or Matplotlib dependency.
  `has_overview(start, stop)` identifies exact cache hits so consumers can
  avoid reads. The `cache_size` constructor argument sets the number of retained
  detail envelopes. A value of zero disables detail retention. `set_extent()`
  clears cached summaries for a new signal.
  `set_regions()` supplies pickable sample bands, and `region_selected` emits
  the original region when a band is selected.
- `sigmf_zarr.viewer.overview.summarize_peaks()` reduces consecutive NumPy
  blocks to peak magnitudes without spectral analysis. `DatasetSource.read_peaks()`
  adapts this function to a recording, item, channel, and time interval.
- `sigmf_zarr.viewer.overview.summarize_signal()` accepts an iterable of
  consecutive NumPy blocks and a total sample count. It returns a
  `SignalOverview` containing read-only envelopes, spectral powers, sampled
  I/Q points, and full-interval scalar metrics. It has no storage or GUI
  dependency.

For example, an application can use the range selector directly:

```python
from PySide6.QtWidgets import QApplication
from sigmf_zarr.viewer.timeline import RangeSelector

app = QApplication([])
selector = RangeSelector()
selector.set_extent(1000000)
selector.set_range(200000, 400000)
selector.set_overview([0.1, 0.4, 1.0, 0.6, 0.2])
selector.range_changed.connect(lambda start, stop: print(start, stop))
selector.resize(800, 100)
selector.show()
raise SystemExit(app.exec())
```

Save this example as `custom_viewer.py` and run
`python custom_viewer.py radioml2016.10a.zarr`:

```python
import argparse

import numpy as np
from matplotlib.figure import Figure
from PySide6.QtWidgets import QApplication

from sigmf_zarr.viewer import DatasetSource, SampleWindow
from sigmf_zarr.viewer.qt import DatasetViewer, MatplotlibPanel


def envelope(figure: Figure, window: SampleWindow) -> None:
    """Draw sample magnitude against original sample positions."""
    axes = figure.subplots()
    axes.plot(
        window.start + np.arange(len(window.samples)),
        np.abs(window.samples),
    )
    axes.set(xlabel="Sample position", ylabel="Magnitude")


def median_magnitude(window: SampleWindow) -> float:
    """Calculate the selected window's median magnitude."""
    return float(np.median(np.abs(window.samples)))


parser = argparse.ArgumentParser()
parser.add_argument("store")
args = parser.parse_args()
app = QApplication([])
viewer = DatasetViewer(
    DatasetSource(args.store),
    metrics={"Median magnitude": median_magnitude},
)
panel = MatplotlibPanel(envelope)
viewer.add_panel("Envelope", panel, panel.set_window)
viewer.resize(1280, 900)
viewer.show()
raise SystemExit(app.exec())
```

Supplying `metrics` replaces the default metrics. Set `default_panels=False`
to construct a viewer containing only consumer-supplied visualizations and the
metadata tabs. Registered visualization tabs are visible when data compatible
with their callbacks is available. `window_changed` emits the selected raw
window or `None`. `overview_changed` emits the selected reduced interval or
`None`. Callbacks should treat window samples and metadata as read-only.
The sample array is a detached, non-writable NumPy vector.

`SampleWindow.recording_metadata`, `item_metadata`, and `channel_metadata`
expose the stored scopes separately. `item_metadata` is `None` when no item
metadata array exists or the recording is unbatched. `channel_metadata` is
`None` when no explicit channel axis exists. `SampleWindow.metadata` retains
the resolved item JSON for batched recordings and recording attributes for
continuous recordings. `SampleWindow.indexes` remains independent of JSON.

Reads and metrics run in workers. Metric callbacks must not access Qt widgets.
Custom raw-sample metrics run only on selections of at most 65536 samples.
Larger selections display **Zoom in for raw-sample metrics** instead of applying
those callbacks to sampled or aggregated data. Consumers can calculate their
own overview metrics from `SignalOverview` through the overview callback.
Panel callbacks run on the GUI thread and should keep drawing bounded. A custom
panel performing expensive analysis should manage that work in a worker.
Selection changes discard obsolete results and clear old displays while a new
window loads. Close embedded viewers when their containing application closes
to cancel background work.

## Scope and limits

The first viewer provides item inspection and explicit filtering. It does not
compute dataset-wide statistics, edit metadata, save query definitions, or
introduce signal tables. Additional-axis and matching-item position controls
support values up to $2^{31}-1$. The timeline uses Python integer coordinates.
Sample reads are logically bounded, but a backend may fetch an entire
physical chunk or shard. Filters return positions in memory and require memory
proportional to the number of matches.

Keep the dataset unchanged during browsing. The viewer does not provide a
transactional snapshot across operations or monitor external writes. Backend
errors and invalid selected indexes are reported without silently choosing a
different metadata source.

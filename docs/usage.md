# Usage

Use SigMF-Zarr as a Python API for building Zarr-backed signal stores or as a
command-line interface (CLI) for importing and exporting standard SigMF files
and supported datasets.

## Installation

Install a published release with pip:

```bash
python -m pip install sigmf-zarr
```

Install the package from the repository in editable mode during development:

```bash
uv sync --extra dev --extra docs
```

Run commands through `uv` so they use the repository environment:

```bash
uv run pytest
uv run sigmf-zarr --help
```

## CLI

The package installs the `sigmf-zarr` command. Its resource-oriented commands
inspect a store and its recordings and collections:

```bash
sigmf-zarr store store.zarr info
sigmf-zarr store store.zarr recordings
sigmf-zarr store store.zarr recording myrec info --format json
sigmf-zarr store store.zarr collections
sigmf-zarr store store.zarr collection paired info
sigmf-zarr store store.zarr validate
sigmf-zarr store store.zarr integrity verify
```

Inspection commands produce human-readable text by default and accept
`--format json` for scripts. Store locations remain strings so compatible Zarr
URLs are not coerced into local filesystem paths.

All importers are subcommands of `sigmf-zarr import`. List the supported
formats with:

```bash
sigmf-zarr import --help
```

Import a single standard SigMF recording:

```bash
sigmf-zarr import sigmf input.sigmf-meta store.zarr
```

Import a standard SigMF archive:

```bash
sigmf-zarr import sigmf input.sigmf store.zarr
```

All import commands share `SOURCE`, `STORE`, `--recording-name`,
`--overwrite-store`, `--overwrite-recording`, and `--zarr-format`. When the
target store already exists, its Zarr format is detected automatically. New
stores use Zarr format 3 by default. Pass `--zarr-format 2` when creating a new
store for a downstream tool that requires the Zarr format 2 layout:

```bash
sigmf-zarr import sigmf input.sigmf-meta store.zarr --zarr-format 2
```

Supplying `--zarr-format` for an existing store verifies that the store has the
requested format. The same behavior applies to both RadioML import commands.
Zarr format 2 does not support sample sharding, so it cannot be combined with
`--sample-shard-batch`.

Import a RadioML 2016 pickle mapping:

```bash
sigmf-zarr import radioml2016 RML2016.10a.pkl store.zarr
```

Python pickle loading can execute code. Use `sigmf-zarr import radioml2016`
only with dataset files obtained from a trusted source.

Import a RadioML 2018 HDF5 dataset:

```bash
sigmf-zarr import radioml2018 \
  GOLD_XYZ_OSC.0001_1024.hdf5 store.zarr
```

See [RadioML](radioml.md) for dataset-specific behavior, class ordering, CLI
options, and Python examples.

Import Chad Spooner's CSPB `.tim` files or ZIP batches:

```bash
sigmf-zarr import cspb CSPB.ML.2018R2 cspb.zarr \
  --truth-file signal_record_C_2023.txt \
  --source-dataset CSPB.ML.2018R2
```

See [Chad Spooner CSPB Datasets](cspb.md) for the `.tim` encoding, supported
truth layouts, ZIP handling, indexes, and Python API.

Export one unbatched recording back to standard SigMF:

```bash
sigmf-zarr export store.zarr output.sigmf-meta --recording-name myrec
```

The equivalent resource-oriented form is:

```bash
sigmf-zarr store store.zarr recording myrec export output.sigmf-meta
```

Export a standard SigMF archive:

```bash
sigmf-zarr export store.zarr output.sigmf --archive --recording rec1
```

The `sigmf-zarr import sigmf` command detects standard SigMF metadata files
and archives from the file suffix. Use `--format` when the suffix is ambiguous.

## Create a recording

Create a store with one unbatched real-valued recording:

```python
import numpy as np

from sigmf_zarr import SigMFZarrStore

store = SigMFZarrStore.create("example.zarr", overwrite=True)
recording = store.recordings.open(
    "tone",
    create=True,
    batched=False,
    sample_dtype=np.float32,
    sample_shape=(1024,),
    sample_axes=("time",),
    global_metadata={
        "core:datatype": "rf32_le",
        "core:sample_rate": 1_000_000.0,
        "core:version": "1.2.0",
    },
    captures=[
        {
            "core:sample_start": 0,
            "core:frequency": 915_000_000.0,
        }
    ],
)

samples = np.zeros((1024,), dtype=np.float32)
recording.set_samples(samples)
```

`SigMFZarrStore.create` defaults to Zarr format 3. To create the same logical
schema using the Zarr format 2 physical layout, pass `zarr_format=2`:

```python
store = SigMFZarrStore.create(
    "example-v2.zarr",
    overwrite=True,
    zarr_format=2,
)
```

`SigMFZarrStore.open` auto-detects both formats. This compatibility uses
zarr-python 3.2 or newer. It is not compatibility with the old zarr-python 2.x
package.

The recording stores SigMF-like `global`, `captures`, and `annotations`
metadata as Zarr group attributes. SigMF-Zarr adds storage metadata such as
`sigmf-zarr:sample-shape` and `sigmf-zarr:sample-axes` to the recording
`global` object.

## Sample axes

Use `sample_axes` to make the tensor layout explicit. Standard SigMF serializes
a `cf*` signal as interleaved complex samples in its flat `.sigmf-data` byte
stream. SigMF-Zarr instead stores the components in one real-valued Zarr array
with an explicit `iq` axis of length `2`: index `0` is I and index `1` is Q.
They are not separate arrays. Import splits standard complex samples along this
axis, and export combines them into the standard interleaved representation.

```python
recording = store.recordings.open(
    "array",
    create=True,
    batched=False,
    sample_dtype=np.float32,
    sample_shape=(4, 2, 2048),
    sample_axes=("channel", "iq", "time"),
    global_metadata={
        "core:datatype": "cf32_le",
        "core:num_channels": 4,
        "core:sample_rate": 2_000_000.0,
        "core:version": "1.2.0",
    },
    overwrite=True,
)

samples = np.zeros((4, 2, 2048), dtype=np.float32)
recording.set_samples(samples)

assert recording.sample_axes == ("channel", "iq", "time")
assert recording.runtime_axes == ("channel", "iq", "time")
assert recording.axis_index("time") == 2
assert recording.sample_count == 2048
```

For a batched recording, `sample_shape` describes one logical item and the
leading runtime axis is inferred as `item`:

```python
recording = store.recordings.open(
    "snippets",
    create=True,
    batched=True,
    sample_dtype=np.float32,
    sample_shape=(2, 128),
    sample_axes=("iq", "time"),
    overwrite=True,
)

batch = np.zeros((16, 2, 128), dtype=np.float32)
recording.append_samples(batch)

assert recording.batched is True
assert recording.runtime_axes == ("item", "iq", "time")
```

Batching is derived from the stored array rank and
`sigmf-zarr:sample-shape`. There is no separate stored batching flag.

## Append samples with captures

Use `append_samples(..., capture=...)` to grow a recording and attach capture
metadata to the appended segment.

```python
segment = np.ones((4, 2, 128), dtype=np.float32)

recording.append_samples(
    segment,
    capture={
        "core:frequency": 916_000_000.0,
        "core:datetime": "2026-06-02T12:00:00Z",
    },
)
```

For unbatched recordings, appends extend the `time` axis. If
`core:sample_start` is omitted from the capture metadata, SigMF-Zarr fills it
with the old time sample count. For batched recordings, appends extend the
leading `item` axis and `core:sample_start` defaults to the old item count.
A shared capture timestamp applies only to the item at that offset. It does
not establish timestamps for later items.

For independently acquired items, supply `item_captures` with one capture list
per appended item. Use multiple captures within a list for transitions or
acquisition gaps within one item:

```python
import numpy as np

from sigmf_zarr import SigMFZarrStore

with SigMFZarrStore.create("captured.zarr", overwrite=True) as captured_store:
    captured = captured_store.recordings.open(
        "snippets",
        batched=True,
        sample_dtype=np.complex64,
        sample_shape=(1024,),
        sample_axes=("time",),
        global_metadata={"core:sample_rate": 1_000_000.0},
    )
    captured.append_samples(
        np.zeros((2, 1024), dtype=np.complex64),
        capture={"core:frequency": 915_000_000.0},
        item_captures=[
            [{"core:datetime": "2026-09-25T12:00:10.123456789Z"}],
            [
                {"core:datetime": "2026-09-25T12:00:02Z"},
                {
                    "core:sample_start": 512,
                    "core:datetime": "2026-09-25T12:00:03Z",
                },
            ],
        ],
    )
```

The example stores a separate timestamp for each item and a later capture
within the second item. The first capture in each list defaults to time-sample
position zero, or the item's resolved `core:offset` when present. Later
captures must provide explicit starts. The writer does not infer missing
capture timestamps.

Use `None` for an item with no capture list to add. Use `item_metadata`
alongside `item_captures` for per-item globals or annotations. For a given
item, supply captures through only one of those arguments. Both arguments
must have one entry per appended item. All metadata is validated before
storage changes.

## Metadata indexes

Indexes are dense one-dimensional arrays aligned with a runtime sample axis.
They are useful for labels, SNR values, quality scores, frequency bins, or
other metadata that belongs to each item, channel, or time sample.

```python
recording = store.recordings.open(
    "snippets",
    create=False,
)

recording.add_index(
    "snr_db",
    np.array([0, 2, 4, 6] * 4, dtype=np.int16),
    axis="item",
    field="radioml:snr",
    unit="dB",
    overwrite=True,
)

recording.add_index(
    "mod_class_id",
    np.zeros((16,), dtype=np.int16),
    axis="item",
    field="radioml:mod_class",
    labels=["BPSK"],
    overwrite=True,
)
```

Index length must match the selected runtime axis. For example, an `item`
index on a batched recording with `16` items must have length `16`.

## Resolved signal access

`recording.signal()` resolves an unbatched recording.
`recording.signal(item_index=0)` resolves one explicit batch item. The resulting
`SignalView` exposes named axes, shape, dtype, sample rate, source offset,
capture segments, and detached metadata. Its `read_samples(start, stop)` method
reads a bounded range in stored time coordinates and preserves per-signal axis
order. It never reads neighboring items. Starts and stops use local stored
positions even when `core:offset` is nonzero.

The view resolves per-item overrides and translates the applicable shared
capture from item offsets to the signal's source sample coordinates. It retains
explicit source anchors without inferring timestamps or global indices. The
recording's raw metadata remains available separately. Keep the recording open
and unchanged while using a view. Detached metadata does not freeze sample
storage.

## Per-item metadata

Use per-item metadata for irregular or nested metadata that cannot be
represented efficiently as a typed index. Each entry can supplement shared
`global`, `captures`, and `annotations` metadata:

```python
item_batch = np.zeros((2, 2, 128), dtype=np.float32)

recording.append_samples(
    item_batch,
    item_metadata=[
        None,
        {
            "global": {"core:frequency": 915_000_000.0},
            "captures": [
                {
                    "core:sample_start": 0,
                    "core:datetime": "2026-08-19T12:00:00Z",
                }
            ],
        },
    ],
)

metadata = recording.get_item_metadata(1)
resolved = recording.resolved_item_metadata(1)
```

For individual or contiguous updates, use the validated setters. They encode
and validate only the supplied entries:

```python
recording.set_item_metadata_entry(
    1,
    {"global": {"core:frequency": 433_920_000.0}},
)
recording.set_item_metadata_slice(
    slice(10, 12),
    [
        {"global": {"example:fold": "validation"}},
        None,
    ],
)
```

The low-level `mutate_item_metadata()` context remains available for bulk
encoded-array changes. On exit it reads and validates complete Zarr chunks,
not individual scalar entries. All `item_metadata` chunks use CRC32C so normal
reads detect physical corruption before JSON is accepted.

The metadata list must contain one entry per appended item. If a recording
already has item metadata, appending without `item_metadata` or `item_captures`
adds empty objects to keep the array aligned. Prefer typed indexes for dense
scalar values such as class IDs and SNRs.

## Channel metadata

Supply one metadata object for each explicit sample channel when creating a
recording:

```python
recording = store.recordings.open(
    "array",
    create=True,
    batched=False,
    sample_shape=(2, 2, 2048),
    sample_axes=("channel", "iq", "time"),
    channel_metadata=[
        {"antenna:element": 0, "antenna:polarization": "H"},
        {"antenna:element": 1, "antenna:polarization": "V"},
    ],
)

print(recording.channel_metadata(0))
recording.set_channel_metadata(
    1,
    {"antenna:element": 1, "antenna:gain": 12.5},
)
```

Channel metadata is stored in numbered groups under `channels/`. A recording
without an explicit channel axis has one implicit channel and an empty
`channels/` group.

## Integrity

New sample arrays use per-chunk CRC32C checksums by default. Zarr validates the
checksum automatically whenever a chunk is read. Format 3 represents it in the
codec pipeline, while Zarr format 2 represents it as a numcodecs filter. Disable
checksums only when a caller has a specific reason to accept unprotected sample
chunks:

```python
recording = store.recordings.open(
    "unchecked",
    create=True,
    batched=False,
    sample_shape=(1024,),
    sample_checksum=None,
)
```

For an unbatched recording, calculate or verify standard SigMF
`core:sha512` over the exact `.sigmf-data` byte stream that export would
produce:

```python
digest = recording.calculate_sha512()  # Does not change metadata.
recording.update_sha512()              # Stores core:sha512.
assert recording.verify_sha512()
```

`set_samples()` and `append_samples()` automatically remove a stored digest.
Public sample views are read-only. Use a mutation context for same-shape value
updates. It invalidates hashes before granting writable access:

```python
with recording.mutate_samples() as samples:
    samples[100:200] = replacement
```

Use `set_samples()` or `append_samples()` for shape changes. SHA-512 is a
whole-recording portability and identity check. CRC32C remains the efficient
per-chunk corruption check.

SigMF-Zarr can additionally maintain logical SHA-512 hashes for samples and
metadata. Unlike `core:sha512`, these native hashes describe the logical Zarr
data model and therefore also work for batched recordings and metadata arrays:

```python
store.update_integrity()

print(recording.sample_sha512)
print(recording.metadata_sha512)
print(store.metadata_sha512)
assert store.verify_integrity()
```

Managed mutations invalidate the affected hashes. Recording-level and
store-wide indexes are also read-only and have explicit writable contexts:

```python
with recording.mutate_index("snr_db") as index:
    index[100:200] = new_snr

with store.mutate_index("split") as index:
    index[:] = new_split
```

The same rule applies to other mutable recording and collection structures.
Use `mutate_extensions()`, `mutate_item_metadata()`,
`set_channel_metadata()`, `SigMFCollection.mutate()`, and the collection
setters instead of reaching through a public Zarr handle. These paths validate
JSON metadata and invalidate the recording, collection, and root digests they
affect.

An index is marked invalid before the context body runs and restored to valid
only after successful structural validation. Resizing a sample axis similarly
invalidates indexes aligned with that axis until they are repaired or replaced.
Re-run `store.update_integrity()` after completing a series of changes.

## Import and export SigMF

The reusable Python helpers mirror the CLI:

```python
from sigmf_zarr import SigMFZarrStore, export_sigmf, import_sigmf

recording = import_sigmf(
    "store.zarr",
    "input.sigmf-meta",
    recording_name="imported",
    overwrite_store=True,
)

export_sigmf(
    SigMFZarrStore.open("store.zarr"),
    "imported",
    "roundtrip.sigmf-meta",
    overwrite=True,
)
```

Standard SigMF export supports only unbatched recordings. Batched recordings
represent multiple independent items, so exporting them to standard SigMF
requires an explicit split strategy that the implementation does not provide.

When importing standard SigMF, `core:num_channels` is mapped to an explicit
SigMF-Zarr `channel` axis. A real two-channel SigMF stream imports as
`("channel", "time")`. A complex two-channel stream imports as
`("channel", "iq", "time")`.

Complex floating-point precision is preserved: `cf32_le` and `cf32_be` use
32-bit I/Q components, while `cf64_le` and `cf64_be` use 64-bit components.
Export restores the endianness declared by `core:datatype` rather than using
the host machine's native byte order.

Import verifies that the reconstructed standard dataset bytes match the source
`core:sha512` and stores that digest. If conversion cannot preserve the source
bytes exactly, import fails instead of retaining a misleading digest. Export
adds `core:sha512` when it is absent and rejects a stored digest that does not
match the bytes being written.

Archive import also verifies `.sigmf-collection` stream hashes against the
referenced metadata files. A standalone `.sigmf-meta` file has no standard
self-hash for its metadata. After a successful import, SigMF-Zarr calculates
its own sample and metadata hashes. Export verifies any stored native hashes,
calculates `core:sha512` from the emitted data bytes, and, for archives,
calculates collection stream hashes from the emitted metadata bytes.

For multi-channel unbatched recordings, standard SigMF export validates
`core:num_channels` against the declared `channel` axis. If the field is
missing and the recording has more than one channel, export fills it from the
axis length. If the field is present but does not match the axis length, export
raises `ValueError`. Single-channel exports may still include
`core:num_channels: 1` because that is normal standard SigMF metadata.

## Mutation and failure behavior

Replacing a recording, collection, index, extension array, or item metadata
array with `overwrite=True` preserves the old resource and ancestor metadata
until creation, data writes, and validation finish. If any of these steps
raises an exception, the operation restores the old resource. If restoration
also fails, an `OSError` identifies retained recovery files. Local replacements
move the old directory to a backup on the same filesystem. Other writable,
listable backends copy encoded objects to temporary disk storage, using memory
proportional to one encoded object.

Callers must serialize access during replacement. This rollback boundary does
not provide concurrent-reader snapshots or recovery from process crashes.
Replacing an entire store with `overwrite=True` remains destructive.

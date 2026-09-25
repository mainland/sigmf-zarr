# Usage

Use SigMF-Zarr as a Python API for building Zarr-backed signal stores.

## Installation

Install a published release with pip:

```bash
python -m pip install sigmf-zarr
```

Install the package from the repository in editable mode during development:

```bash
uv sync
```

Run commands from the repository environment:

```bash
.venv/bin/python -m pytest
```

## Create a recording

A store location string is a URL only when it starts with an explicit scheme
followed by `://`, such as `s3://`, `http://`, `file://`, or `memory://`, or a
chained scheme followed by `::`, such as `simplecache::s3://bucket/store.zarr`.
Any other string, and every `pathlib.Path` or other path-like object, names a
local directory. SigMF-Zarr does not parse a local path as a URL, so characters
such as `#` and `?` are part of the directory name. The command line applies
the same rule to store arguments.

Create a store with one unbatched real-valued recording:

```python
import numpy as np

from sigmf_zarr import SigMFZarrStore

store = SigMFZarrStore.create("example.zarr", overwrite=True)
recording = store.recordings.create(
    "tone",
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

`SigMFZarrStore.create` refuses a target that already contains a Zarr
hierarchy unless `overwrite=True`. `SigMFZarrStore.open` never creates a store.
It opens read-only by default and accepts `mode="r+"` for writes. The
`SigMFZarrStore` constructor follows Zarr access modes. Its default mode, `"a"`,
opens an existing store or initializes an empty target. Requesting a Zarr
format that differs from an existing store raises `ValueError`.

Recordings and collections follow the same rule. `store.recordings.open()`
raises `KeyError` for a missing recording, and `store.recordings.create()`
raises `ValueError` for an existing one unless `overwrite=True`. Use a store as
a context manager, or call `close()`, to release its backend.

When `sample_chunks` is omitted, new recordings use logical chunks of about
1 MiB along the `item` axis of a batched recording or the `time` axis of an
unbatched recording. Pass `sample_chunks` to choose a different layout.

`sample_dtype` must be a Boolean, integer, floating-point, or complex dtype.
Recording creation raises `ValueError` for text, byte-string, date-time,
time-delta, object, void, and structured dtypes. Each element of a complex
dtype holds one I/Q pair, so recording creation also raises `ValueError` for a
complex dtype with an `iq` axis. Pass `sample_axes` explicitly for a complex
recording whose shape would infer an `iq` axis, such as `(2, 1024)`.

Extension arrays, indexes, and store-wide arrays must not use a structured or
void dtype. Array hashes do not cover structured field names, so
`add_extension_array()`, `add_index()`, and `SigMFZarrStore.create_array()`
raise `ValueError` for such values. Store each field as a separate array.

`sample_compressor`, and the `compressor` and `compressors` arguments of
`add_extension_array()` and `SigMFZarrStore.create_array()`, accept these
forms in both Zarr formats:

- `"auto"` for the default Zstandard compressor
- `None` for no compression
- A numcodecs `Blosc`, `Zstd`, or `GZip` instance
- A `zarr.codecs` `BloscCodec`, `ZstdCodec`, or `GzipCodec` instance

SigMF-Zarr translates each codec to the equivalent codec of the store's
format. Zarr format 2 also accepts any other numcodecs codec. Zarr format 3
also accepts any other bytes-to-bytes codec and its JSON mapping. Any other
compressor raises `ValueError` before any write. `compressors` also accepts a
sequence of codecs, applied in order. Zarr format 2 accepts at most one.

The recording stores SigMF-like `global`, `captures`, and `annotations`
metadata as Zarr group attributes. SigMF-Zarr adds storage metadata such as
`sigmf-zarr:sample-shape` and `sigmf-zarr:sample-axes` to the recording
`global` object.

## Sample axes

Use `sample_axes` to make the tensor layout explicit. Standard SigMF serializes
a `cf*` signal as interleaved complex samples in its flat `.sigmf-data` byte
stream. SigMF-Zarr instead stores the components in one real-valued Zarr array
with an explicit `iq` axis of length `2`: index `0` is I and index `1` is Q.
They are not separate arrays.

```python
recording = store.recordings.create(
    "array",
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
recording = store.recordings.create(
    "snippets",
    batched=True,
    sample_dtype=np.float32,
    sample_shape=(2, 128),
    sample_axes=("iq", "time"),
    overwrite=True,
)

batch = np.zeros((12, 2, 128), dtype=np.float32)
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

An append of zero items or samples validates its arguments and changes
nothing, so every hash remains current. Such an append must not pass `capture`,
because a capture describes at least one sample. `append_samples()` raises
`ValueError` before any write in that case.

For unbatched recordings, appends extend the `time` axis. If
`core:sample_start` is omitted from the capture metadata, SigMF-Zarr fills it
with `core:offset`, or zero when `core:offset` is absent, plus the old time
sample count. For batched recordings, appends extend the
leading `item` axis and `core:sample_start` defaults to the old item count.

A shared capture must pass the checks that `validate_store()` applies. Its
`core:sample_start` must be a nonnegative integer that does not precede the
start of the last stored capture, and the capture must satisfy the SigMF
schema. `append_samples()` raises `ValueError` before any write otherwise.

For independently acquired items, supply `item_captures` with one capture list
per appended item. Use multiple captures within a list for transitions or
acquisition gaps within one item:

```python
import numpy as np

from sigmf_zarr import SigMFZarrStore

with SigMFZarrStore.create("captured.zarr", overwrite=True) as captured_store:
    captured = captured_store.recordings.create(
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
recording = store.recordings.open("snippets")

recording.add_index(
    "snr_db",
    np.array([0, 2, 4, 6] * 4, dtype=np.int16),
    axis="item",
    field="example:snr",
    unit="dB",
    overwrite=True,
)

recording.add_index(
    "mod_class_id",
    np.zeros((16,), dtype=np.int16),
    axis="item",
    field="example:mod_class",
    labels=["BPSK"],
    overwrite=True,
)
```

Index length must match the selected runtime axis. For example, an `item`
index on a batched recording with `16` items must have length `16`.

## Per-item metadata

Use per-item metadata for irregular or nested metadata that cannot be
represented efficiently as a typed index. Each entry can supplement shared
`global`, `captures`, and `annotations` metadata:

```python
recording = store.recordings.create(
    "irregular",
    batched=True,
    sample_dtype=np.float32,
    sample_shape=(2, 128),
    sample_axes=("iq", "time"),
    overwrite=True,
)
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
    slice(0, 2),
    [
        {"global": {"example:fold": "validation"}},
        None,
    ],
)
```

The low-level `mutate_item_metadata()` context remains available for bulk
encoded-array changes. On exit it reads and validates complete Zarr chunks,
not individual scalar entries. Like `mutate_samples()`, it cannot resize the
array. If the body changes the shape, the context restores the original shape,
even when the body raises. It then raises `ValueError`, unless the body raised
its own exception. Entries outside a shrunken extent are lost and become empty
metadata objects. Values written inside the context remain, and the metadata
hashes remain absent. `SigMFCollection.mutate()` restores the original
collection attributes if the body raises or the result is invalid. All
`item_metadata` chunks use CRC32C so normal reads detect physical corruption
before JSON is accepted.

The metadata list must contain one entry per appended item. If a recording
already has item metadata, appending without `item_metadata` or `item_captures`
adds empty objects to keep the array aligned. Prefer typed indexes for dense
scalar values such as class IDs and SNRs.

## Channel metadata

Supply one metadata object for each explicit sample channel when creating a
recording:

```python
recording = store.recordings.create(
    "dual-polarized",
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
checksum automatically whenever a chunk is read. Zarr format 3 represents it in
the codec pipeline, while Zarr format 2 represents it as a numcodecs filter.
Disable checksums only when a caller has a specific reason to accept
unprotected sample chunks:

```python
recording = store.recordings.create(
    "unchecked",
    batched=False,
    sample_shape=(1024,),
    sample_checksum=None,
)
```

`set_samples()` and `append_samples()` automatically remove a stored
`core:sha512` digest. Public sample views are read-only. Use a mutation context
for same-shape value updates. It invalidates hashes before granting writable
access:

```python
replacement = np.ones(100, dtype=np.float32)
with recording.mutate_samples() as samples:
    samples[100:200] = replacement
```

Use `set_samples()` or `append_samples()` for shape changes.

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

The store-wide hash covers root metadata, store-wide indexes, and the digests
that each recording and collection declares. `store.update_metadata_integrity()`
therefore requires current hashes for every recording and collection.
`store.verify_metadata_integrity()` checks only the store-wide hash, without
reading any recording arrays. `store.verify_integrity()` also recalculates
every recording and collection hash.

Managed mutations invalidate the affected hashes. Recording-level and
store-wide indexes are also read-only and have explicit writable contexts:

```python
snippets = store.recordings.open("snippets")
with snippets.mutate_index("snr_db") as index:
    index[:4] = np.full(4, 10, dtype=np.int16)

store.add_index("fold", np.zeros(4, dtype=np.int8), overwrite=True)
with store.mutate_index("fold") as index:
    index[:] = np.array([0, 1, 0, 1], dtype=np.int8)
```

The same rule applies to other mutable recording and collection structures.
Use `mutate_extensions()`, `mutate_item_metadata()`,
`set_channel_metadata()`, `SigMFCollection.mutate()`, and the collection
setters instead of reaching through a public Zarr handle. These paths validate
JSON metadata and invalidate the recording, collection, and root digests they
affect. `set_global_field()` rejects the `sigmf-zarr` storage fields, which the
recording maintains from its sample array. It also rejects a value that
violates the SigMF schema that `validate_store()` applies, such as a string
`core:sample_rate`. Updates to `core:extensions` retain one canonical
SigMF-Zarr declaration and preserve other namespace declarations.

Recording and collection names must identify immediate child groups. Names
must be nonempty and must not contain path separators, `.` or `..` path
components, or a NUL character.

Recording, collection, index, split, and extension-array names must not contain
control characters or lone surrogates, which are the Unicode categories `Cc`
and `Cs`. Such a name could forge terminal output or fail to encode as UTF-8.
The API raises `ValueError` for such a name, and `validate_store()` reports one
that another tool stored.

Index, split, and extension-array names must identify one array below their
group. A name may contain nested groups separated by `/`. It must not start or
end with `/`, contain an empty, `.`, or `..` segment, or contain a NUL
character. A name that identifies a group is not an index. Lookups and
`mutate_index()` raise `KeyError` for it, and `overwrite=True` does not replace
the group. A name below an existing array is rejected with `ValueError` before
any write. For example, with index `ids` present, `ids/c` is rejected, because
Zarr format 3 stores the chunks of `ids` under that prefix.

The Zarr metadata document names `.zarray`, `.zattrs`, `.zgroup`, `.zmetadata`,
and `zarr.json` are reserved in both Zarr formats. A recording or collection
name must not be a reserved name, and no segment of an index, split, or
extension-array name may be one. The API raises `ValueError` for a reserved
name before any write, and `validate_store()` reports a recording or collection
that another tool stored under a reserved name.

Collection creation and `set_recording_ids()` reject identifiers that do not
name an existing recording or that appear more than once.
`store.recordings.remove()` rejects a recording that a collection references.
Update the collection first.

An index is marked invalid before the context body runs and restored to valid
only after successful structural validation. Resizing a sample axis similarly
invalidates indexes aligned with that axis until they are repaired or replaced.
Re-run `store.update_integrity()` after completing a series of changes.

## Mutation and failure behavior

Replacing a recording, collection, index, extension array, or item metadata
array with `overwrite=True` preserves the old resource and ancestor metadata
until creation, data writes, and validation finish. If any of these steps
raises an exception, the operation restores the old resource. Zarr performs
some writes concurrently, so restoration first waits for every write that the
failed step started. After a `KeyboardInterrupt`, the exception propagates only
after the interrupted writes finish. If restoration also fails, an `OSError`
identifies retained recovery files. Local replacements move the old directory
to a backup on the same filesystem, and restoration moves it back, so neither
step needs free disk space. Other writable, listable backends copy
encoded objects to temporary disk storage, using memory proportional to one
encoded object.

In a local directory store, `store.recordings.remove()` follows the same
boundary. If the store hash update fails, the recording moves back and the
store hash is restored. Other backends remove the store hash and then delete
the recording directly, without a backup, so removal does not copy the
recording. They delete the recording group's own metadata last. If the
deletion fails, the recording remains listed with part of its content.
`validate_store()` reports such a partial recording. Call `remove()` again to
finish the removal.

A failed write outside these boundaries, or another tool, can leave objects
under `recordings/<name>/` or `collections/<name>/` without the group's
metadata. Such orphan objects form no resource. Opening the name raises
`KeyError` and never writes or deletes anything. `store.recordings.remove()`
deletes orphan objects under a recording name when the store can list them.

Creating a recording, collection, index, extension array, or item metadata
array follows the same boundary. If creation fails, the operation removes the
partial resource and restores any ancestor hashes it invalidated, so the name
remains available. On a store that can list its objects, creation requires
that no orphan objects exist under the new name, so rollback never deletes an
object that the operation did not write. Without `overwrite=True`, orphan
objects make creation raise `ValueError`. With `overwrite=True`, creation
replaces them like an existing resource.

Callers must serialize access during replacement. This rollback boundary does
not provide concurrent-reader snapshots or recovery from process crashes.
Replacing an entire store with `overwrite=True` remains destructive.

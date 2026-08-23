# Usage

Use SigMF-Zarr as a Python API for building Zarr-backed signal stores or from
the command line to import and export standard SigMF files and supported
datasets.

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
.venv/bin/sigmf-zarr --help
```

## Command line

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
`--format json` for scripts. A store argument is a Zarr URL only when it starts
with an explicit scheme, such as `s3://` or `file://`. Any other store argument
names a local directory, including one whose name contains `#` or `?`.

`validate` checks store structure and verifies every declared hash, including
`core:sha512`. It reads each sample array once for each declared sample hash.
Pass `--skip-integrity` to check structure only, or `--require-integrity` to
also require hashes for the store and every recording and collection. For an
untrusted store, pass `--max-array-bytes` to bound the data that validation
decodes. Validation does not verify a hash that covers an array whose declared
logical size, or whose decoded chunk or shard size, exceeds the limit. It does
not check per-item metadata entries in a chunk that exceeds the limit, and it
stops checking entries once their decoded text exceeds the limit. Validation
reports each skipped array and each unchecked set of entries as an issue.
Variable-length strings count 16 bytes per element. The limit therefore bounds
the number of strings in one decoded chunk, but not their length, so a small
compressed chunk of long strings can still decode to more than the limit. The
limit also applies with `--skip-integrity`.

Before it reads a local store, `validate` lists the directory tree without
following links. If it finds a symbolic link, an entry that is neither a
regular file nor a directory, such as a FIFO or device, or a directory that it
cannot list, it reports each one and reads no store content. A store opened
through an fsspec local file system, as in `local:///data/store.zarr` or
`simplecache::file:///data/store.zarr`, is a local store. A failed read,
including a failed remote read, is reported as an issue.

`integrity update` recalculates every hash, then checks structure and hash
presence without reading the samples again.

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
requested format. Zarr format 3 imports automatically place logical chunks in
physical shards targeting approximately 4 MiB. RadioML and CSPB chunks target
approximately 256 KiB, while standard SigMF chunks target approximately 1 MiB.
Pass `--no-sample-sharding` to disable this behavior. Format-2 imports use
logical chunks targeting approximately 4 MiB because format 2 does not support
sharding. The RadioML and CSPB commands also accept `--sample-shard-batch` to
override the automatic item count for each format-3 shard.

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

See [RadioML](radioml.md) for dataset-specific behavior, class ordering,
command-line options, and Python examples.

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

Export targets without a `.sigmf-meta` or `.sigmf` suffix receive one, and dots
in the target stem are retained. An empty target, a target that ends with a
path separator, and a target whose final component is `.` or `..` are errors.

With `--overwrite`, export moves each existing target file into a backup
directory before it publishes the new files. If another process creates a
directory at a target path after the initial check, export moves that
directory back, raises `ValueError`, and leaves the previous files in place.
Any exception during publication, including `KeyboardInterrupt`, restores the
previous files before it propagates.

Archive export writes every member with UID and GID 0, empty owner and group
names, mode `0o755` for directories and `0o644` for files, and modification
time 0, the Unix epoch. Exports of the same content to the same target name
therefore produce identical archive bytes.

The `sigmf-zarr import sigmf` command detects standard SigMF metadata files
and archives from the file suffix. Use `--format` when the suffix is ambiguous.

### Text and JSON output

Text output escapes every nonprintable character in store names, metadata,
validation messages, warnings, log messages, error messages, and `--debug`
tracebacks. Control characters, such as the escape character and carriage
return, appear as Python-style backslash escapes, for example `\x1b` and `\r`.
A line feed in a name or message appears as `\n`, so each name, issue, and
message occupies one output line. A file name that is not valid UTF-8
appears with escapes such as `\udcff`. Backslashes in names are not escaped,
so text output is for display only. JSON output escapes every non-ASCII
character, and its strings decode to the stored names.

With `--format json`, each `store` command prints one JSON object:

- `store STORE info` prints `schema_name`, `schema_version`, `zarr_format`,
  `metadata_sha512`, and the `recordings`, `collections`, and `indexes` name
  lists.
- `store STORE validate` and both `integrity` commands print `valid`,
  `recordings_checked`, `collections_checked`, `indexes_checked`, and
  `issues`. Each issue has a `path` and a `message`.
- `store STORE recordings` prints `{"recordings": [...]}`, and
  `store STORE collections` prints `{"collections": [...]}`.
- `store STORE recording NAME info` prints `name`, `zarr_format`,
  `sample_count`, `item_count`, `batched`, `samples_shape`, `samples_chunks`,
  `sample_axes`, `sample_dtype`, `sample_checksum`, `sha512`,
  `sample_sha512`, `metadata_sha512`, `num_channels`, `has_item_metadata`,
  `global_keys`, `num_captures`, `num_annotations`, `extensions`, and
  `indexes`. `sample_count` is the length of the `time` axis, which is the
  item length in a batched recording. `item_count` is the number of batch
  items, or `1` for an unbatched recording.
- `store STORE recording NAME validate-rfml` prints `valid`, `profile`,
  `version`, `indexes_checked`, `unvalidated_fields`,
  `source_truth_verified`, and `issues`.
- `store STORE recording NAME export` prints `recording` and `output`. With
  `--dry-run`, it prints the `SigMFExportPlan.as_dict()` report in both
  output formats.
- `store STORE collection NAME info` prints `name`, `recording_ids`,
  `metadata_sha512`, and `metadata_keys`.
- `store STORE collection NAME export` prints `collection` and `output`.

### Exit status

Every command exits with one of these statuses:

- `0`: The command succeeded.
- `1`: A check failed. Validation or integrity verification found issues,
  RFML profile validation found violations, or an export dry run found
  rejections or omissions that `--allow-lossy` does not permit.
- `2`: A usage or input error occurred. Examples are an invalid option, a
  missing or unreadable store, a missing recording, a failed remote read, and
  an invalid import source. The command prints the usage line of the selected
  subcommand and a message that starts with `sigmf-zarr: error:` to standard
  error. Where a library message names a Python parameter that has a
  command-line option, such as `overwrite=True`, the message names the
  option, such as `--overwrite`. A missing recording or collection is named
  with its kind, as in `recording not found: 'name'`.

With `--debug`, an input error prints its Python traceback to standard error
instead, and the command exits with status 1.

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
They are not separate arrays. Import splits standard complex samples along this
axis, and export combines them into the standard interleaved representation.

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
A shared capture timestamp applies only to the item at that offset. It does
not establish timestamps for later items.

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

Signal resolution requires a nonnegative integer `core:sample_start` in every
shared and per-item capture. `item_captures` fills in the start of the first
capture of each item, but captures written through `item_metadata` keep the
fields they are given. `signal()` raises `ValueError` and names the capture
when a start is missing.

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

For an unbatched recording, calculate or verify standard SigMF
`core:sha512` over the exact `.sigmf-data` byte stream that export would
produce:

```python
digest = recording.calculate_sha512()  # Does not change metadata.
recording.update_sha512()              # Stores core:sha512.
assert recording.verify_sha512()
```

`set_samples()` and `append_samples()` automatically remove a stored digest.
Public sample views are read-only. They prevent accidental mutation but are not
a security boundary. Use a mutation context for same-shape value updates. It
invalidates hashes before granting writable access:

```python
replacement = np.ones(100, dtype=np.float32)
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

## Import and export SigMF

The reusable Python helpers mirror the command line:

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
Complex integer datatypes retain integer I/Q components. Import stores
components in little-endian byte order. Export interleaves these components
using the width and byte order declared by `core:datatype`. Export also
restores the declared byte order for floating-point samples.

Standard SigMF import reads sample data in bounded blocks. Sample conversion
does not require materializing the complete dataset. Export regenerates the
dataset reference for its output files.

Import verifies that the reconstructed standard dataset bytes match the source
`core:sha512` and stores that digest. If the declared digest is stale or
conversion cannot preserve the source bytes exactly, import raises `ValueError`
instead of retaining a misleading digest. Import also raises `ValueError` for
an unreadable archive, an archive member that the `tarfile` `data` extraction
filter rejects, and a recording that sigmf-python cannot read, such as one
whose `core:dataset` file is missing. Export adds `core:sha512` when it is
absent and rejects a stored digest that does not match the bytes being written.

A `.sigmf` archive must be an uncompressed tar file. Archive import raises
`ValueError` for a gzip, bzip2, xz, or Zstandard stream, and checks every member
before it extracts any. It reads member headers itself and requires each header
to follow the previous one, so a malformed header cannot make parsing loop. It
rejects sparse files, members other than regular files and directories, such as
symbolic links, hard links, and device files, members with an absolute name, a
`..` component, or a NUL character, two regular files with the same name, a
name that is both a regular file and a directory, and a directory with a SigMF
file suffix. A member name that the file system cannot hold, such as one that
is too long, is also a `ValueError`. Archive import also rejects an archive
whose regular files declare more bytes than the archive holds, and an archive
that would create more files and directories, including implied parent
directories, than it has 512-byte blocks. The extracted file data is therefore
no larger than the archive. Each directory can still occupy a file-system block
and an inode, so a directory needs up to about eight times its 512-byte header
on a file system with 4 KiB blocks.

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

Serialize all writes to a store, including integrity updates. The Python API
does not coordinate concurrent writers. Readers must not assume a consistent
snapshot while another process modifies the store.

Managed sample replacement and append operations validate shapes, convert
sample values, and validate supplied metadata before changing storage. A value
is converted to the stored dtype only when the conversion preserves it exactly.
Overflow, rounding, NaN or infinity stored as an integer, and a nonzero
imaginary part stored as a real value are rejected. Rejected inputs preserve
the recording. Writable mutation contexts invalidate affected
hashes before exposing an array or group. These contexts do not roll back
partial writes after a storage I/O failure. Index mutation contexts leave
incomplete indexes marked invalid until repaired. A context that restores a
shape or attributes on failure, such as `mutate_samples()`,
`mutate_item_metadata()`, or `SigMFCollection.mutate()`, first waits for the
writes that the failed body started.

An append resizes the sample array and any item metadata and updates several
attributes. If an exception interrupts these writes, the append truncates the
arrays to their original lengths and restores the attributes, index markers,
and hashes, so the recording matches its state before the append. As with
replacement, the restoration first waits for every write that the append
started.
`set_samples()` restores lengths and attributes the same way, but it may
already have replaced sample values. It therefore leaves the sample hashes and
`core:sha512` absent. `mutate_samples()` restores the original sample array
shape if the caller resizes it, then raises `ValueError`.

Import rejects an existing recording unless replacement is explicitly enabled.
Replacing a recording through an importer requires a local directory store.
The importer keeps a temporary backup and restores the recording if creation,
conversion, or integrity calculation fails. If restoration itself fails, the
error identifies a retained backup for recovery. Import cleanup removes a
partially created new recording. A failed import of a new recording may leave
the root integrity hash absent, requiring recalculation. If nothing existed at
a local target path before a failed import, the importer also removes the
directory that it created there. It records the device and inode numbers of
that directory when it creates it, and removes the directory only while the
path still names the same directory. Import never removes a store or directory
that existed before the import or that another process created.

An explicit `overwrite_store=True` replaces an existing SigMF-Zarr store at the
destination and does not preserve its previous contents. The destination must
be absent, an empty directory, or a directory whose root group is a SigMF-Zarr
store. Any other directory, a regular file, and a symbolic link are errors. A
local destination that equals or contains an input file is also an error,
because replacement would delete the input. Importers validate sample chunk and
shard options before they replace a store, so an invalid option leaves the
store unchanged. A chunk or shard extent must be a positive integer, and a
chunk or shard shape must have one extent per sample-array dimension. Import
rollback and export replacement handle operation failures. They do not provide
a transaction across process termination, power loss, or concurrent access.

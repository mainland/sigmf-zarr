# Format model

SigMF-Zarr is a SigMF-inspired Zarr container. It keeps SigMF metadata as JSON
objects on Zarr groups, stores sample data in chunked Zarr arrays, and adds
axis-aware metadata indexes for ML and analysis workflows.

The format is intentionally not a byte-for-byte translation of a SigMF archive.
Recordings remain the primary organizational unit, while Zarr provides
hierarchical storage, chunking, compression, and efficient array access.

## Root layout

A SigMF-Zarr store is a Zarr group with two required root attributes:

- The `schema_name` attribute must be the string `sigmf-zarr`.
- The `schema_version` attribute must be the integer `1`. Neither the JSON
  value `true` nor the floating-point value `1.0` is the integer `1`.

The root, recording, and collection groups may also carry an `integrity`
attribute as described in the logical integrity section below.

The root group has three required child groups:

```text
/
|-- recordings/
|-- collections/
`-- indexes/
```

These groups have distinct roles:

- The `recordings/` group contains one Zarr group per recording.
- The `collections/` group contains named groups that reference related
  recordings.
- The `indexes/` group contains optional store-wide index arrays.

The `recordings/` and `collections/` groups must contain only groups.
Validation reports an array in either group.

A key below `recordings/<name>/` or `collections/<name>/` belongs to a resource
only when `<name>` has Zarr group metadata. An interrupted removal on a backend
without a directory rename can delete that metadata and leave other keys. Zarr
omits such a name from group listings. When the store can list its keys,
validation reports each name under either group that has keys but no node
metadata.

A store in a local directory must contain only regular files and directories
below its root. Zarr follows a symbolic link, so a write through the store
could modify or delete files outside it, and a read of a FIFO or device could
block or exhaust memory. Validation lists the store without following links
and reports each symbolic link, each other non-regular entry, and each
directory that it cannot list. If it finds one, it reads no store content.

A store must not contain consolidated metadata. In Zarr format 2, consolidated
metadata is a `.zmetadata` document. In Zarr format 3, it is the
`consolidated_metadata` field of a group's `zarr.json`. It copies the metadata
of a group's descendants, and Zarr readers use the copy in place of the node
metadata documents. A copy that differs from those documents would show other
readers content that the integrity hashes do not cover, and writes made through
the node documents would not be visible through it. SigMF-Zarr readers ignore
consolidated metadata at every level of the hierarchy, and validation reports
each group that has it. SigMF-Zarr writers do not create or update consolidated
metadata. When a writer rewrites the `zarr.json` of a Zarr format 3 group, the
new document has no `consolidated_metadata` field.

Recording and collection identifiers must be nonempty names of immediate child
groups. Identifiers must not be `.` or `..`, contain `/` or `\`, or contain a
NUL character.

Recording and collection identifiers, and every `/`-separated segment of an
index, split, or extension-array name, must be Unicode text without control
characters or lone surrogates. These are the code points in Unicode categories
`Cc` and `Cs`. A lone surrogate has no UTF-8 encoding. Python uses one to
represent a file-name byte that is not valid UTF-8. A control character, such
as a line feed or an escape, can forge terminal output when a name is printed.
Writers reject such names. Validation reports such a name for any node below
the root.

The Zarr metadata document names `.zarray`, `.zattrs`, `.zgroup`, `.zmetadata`,
and `zarr.json` are reserved in both Zarr formats. A recording or collection
identifier must not be a reserved name. No `/`-separated segment of an index,
split, or extension-array name may be a reserved name. In each Zarr format, a
child with such a name either replaces its parent's metadata document or is
omitted from group listings. Validation reports a recording or collection with
a reserved name.

### Physical Zarr format

The logical SigMF-Zarr schema can be stored using Zarr format 2 or Zarr format
3. New stores use Zarr format 3 by default. Readers auto-detect either physical
format. A store has one physical format. Writers must reject a request for the
other format rather than create a second hierarchy beside the existing one.
The Python implementation uses zarr-python 3.2 or newer for both. Zarr format 2
compatibility does not require or support the older zarr-python 2.x runtime.

The logical groups, arrays, attributes, and SigMF-Zarr metadata fields are the
same in both formats. Their physical array encodings differ:

- Zarr format 3 uses serializers and codec pipelines.
- Zarr format 2 uses numcodecs filters and a single compressor. Variable-length
  string arrays use the VLenUTF8 filter. The `item_metadata` array places a
  CRC32C filter after VLenUTF8 and before compression.
- Array sharding is available only in Zarr format 3. Writers must reject shard
  settings when creating a Zarr format 2 store.

The reference importers create new stores in Zarr format 3 and use physical
sample shards targeting approximately 4 MiB by default. Batched dataset
imports use logical chunks targeting approximately 256 KiB. Unbatched imports
use logical chunks targeting approximately 1 MiB. When writing format 2, they
use logical chunks targeting approximately 4 MiB instead.

A recording created through the Python API without explicit chunks uses logical
chunks of approximately 1 MiB along its growth axis: `item` for a batched
recording and `time` otherwise. Each chunk holds complete items, or complete
I/Q and channel planes. The chunk length does not depend on the initial
recording length, because appends extend that axis. With an explicit shard
shape, the chunk length is the largest divisor of the shard length that does
not exceed that size, so every shard holds whole chunks.

## Recording layout

Each recording lives at `recordings/<recording_name>/`:

```text
recordings/<recording_name>/
|-- samples
|-- channels/
|-- extensions/
|-- indexes/
`-- item_metadata (optional)
```

The recording group stores core metadata as Zarr group attributes:

- The `global` attribute contains the SigMF-like global metadata object.
- The `captures` attribute contains a list of SigMF-like capture metadata
  objects.
- The `annotations` attribute contains a list of SigMF-like annotation metadata
  objects.

The `channels/`, `extensions/`, and `indexes/` groups are required, even when
they are empty. The `samples` array stores the recording's sample tensor. The
`extensions/` group is reserved for extension-owned data that does not fit the
generic index model. The `indexes/` group stores dense metadata arrays aligned
with sample axes. The `channels/` group stores per-channel metadata when the
sample tensor declares a `channel` axis. The optional `item_metadata` array
stores structured metadata for individual items in a batched recording.

## SigMF-Zarr metadata extension

The normative namespace definition is
[`sigmf-zarr.sigmf-ext.md`](sigmf-zarr.sigmf-ext.md). This chapter specifies
the larger Zarr container model around those SigMF metadata fields.

The `global`, `captures`, and `annotations` attributes must satisfy the SigMF
core schema for every field that they contain. In particular, `core:datatype`,
`core:version`, `core:sample_rate`, `core:sha512`, `core:sample_start`, and
`core:sample_count` must have the types and forms that the schema defines, and
captures and annotations must be sorted by `core:sample_start`. A recording may
omit `core:datatype` until export. `validate_store()` checks the shared
recording metadata against the schema of the installed sigmf package, but not
per-item metadata.

SigMF-Zarr-specific recording metadata is represented as a SigMF extension in
the recording `global` object. A recording must declare exactly one
`sigmf-zarr` extension in `global["core:extensions"]`, with the version and
optional flag shown here:

```json
{
  "name": "sigmf-zarr",
  "version": "0.1.0",
  "optional": true
}
```

The extension defines these `global` fields. Every field is required:

`sigmf-zarr:dtype`
: NumPy dtype string for the element type stored in `samples`, such as
  `<f4`. The dtype must be a Boolean, signed integer, unsigned integer,
  floating-point, or complex type, which are the NumPy kinds `b`, `i`, `u`,
  `f`, and `c`. Text, byte-string, date-time, time-delta, object, void, and
  structured dtypes are not sample dtypes. The value must use little-endian
  byte order (`<`), or `|` for a dtype with no byte order, such as `|i1`. The
  value must be the canonical NumPy dtype string, which is the `str` attribute
  of the NumPy dtype, such as `<f4`, `<c8`, or `|i1`. Other spellings of the
  same dtype, such as `float32`, `f4`, `=f4`, or `>i1`, are not permitted,
  because they would change the metadata hash of identical content. In Zarr
  format 2, where the array dtype carries a byte order, the `samples` array
  dtype must equal this value. Writers convert big-endian sample values to
  little-endian byte order, so one source records the same dtype and metadata
  hashes in both Zarr formats. `validate_store()` reports a dtype of another
  kind, a big-endian value, and a non-canonical spelling.

`sigmf-zarr:sample-shape`
: Shape of one logical sample tensor. For an unbatched recording, this is the
  full `samples` shape. For a homogeneous batched recording, this excludes the
  leading runtime `item` axis. Every dimension must be positive.

`sigmf-zarr:sample-axes`
: Axis names corresponding to `sigmf-zarr:sample-shape`.

`sigmf-zarr:num-channels`
: Number of simultaneous sample channels. This is the declared `channel` axis
  length, or `1` when no explicit channel axis exists.

`sigmf-zarr:checksum`
: Chunk-level checksum applied to sample data. Its value is `crc32c` or `null`.
  A recording without sample checksums must declare the field with the value
  `null` rather than omit it.

These fields describe the stored `samples` array, and a writer must keep them
consistent with it. The Python API maintains them during creation and appends
and rejects direct assignment through `set_global_field()`.

A recording is treated as batched when
`samples.ndim == len(sigmf-zarr:sample-shape) + 1`. The leading runtime axis is
then named `item`.

SigMF-Zarr extension fields are stripped when exporting a recording back to
standard SigMF metadata. Other extension declarations and supported metadata
values are retained, subject to the file-reference and hash adjustments in the
[interchange contract](usage.md). Export rejects unsupported native metadata
unless the caller explicitly permits its omission with a warning.

## Complex sample representation

Standard SigMF and SigMF-Zarr represent the same logical complex samples
differently:

| Representation | Standard SigMF | SigMF-Zarr |
| --- | --- | --- |
| Container | Flat `.sigmf-data` byte stream | Chunked `samples` Zarr array |
| Complex sample | Interleaved I and Q components described by a `cf*` datatype | Real-valued elements with an explicit length-2 `iq` axis |
| Single channel | Time-ordered complex samples | `(iq, time)` |
| Multiple channels | Time-major, channel-interleaved complex samples | `(channel, iq, time)` |

SigMF-Zarr does not store I and Q in separate arrays. Index `0` of the `iq`
axis contains the in-phase component and index `1` contains the quadrature
component. A standard SigMF import splits each complex sample into those two
planes. Export combines the planes into complex samples again and writes the
interleaved byte stream required by the declared `core:datatype`.

A writer may instead create a recording with a complex sample dtype, such as
`<c8` or `<c16`. Each element of such an array holds one complex sample, with
the in-phase component as the real part and the quadrature component as the
imaginary part. Each element then holds the I/Q pair that an `iq` axis would
otherwise hold, so a single channel uses `(time)` and multiple channels use
`(channel, time)`. A recording with a complex sample dtype must not declare an
`iq` axis, which would give a second place for the I and Q components. Writers
reject the combination at creation, readers reject it when the recording opens,
and `validate_store()` reports it. Export writes these values as the same
interleaved byte stream, so `<c8` samples with `core:datatype` `cf32_le` export
as interleaved 32-bit I/Q components. Standard SigMF import always produces the
real-valued `iq` axis representation.

Export requires matching real or complex interpretation and exact numerical
representability in the declared datatype. It rejects discarded quadrature,
fractional integer components, overflow, and precision loss. Permission to omit
metadata does not authorize sample conversion. Quantization and normalization
must be performed explicitly before export.

For floating-point complex data, `cf32_*` maps to a real-valued Zarr array with
32-bit components and `cf64_*` maps to one with 64-bit components. Import and
export preserve the declared byte order in the standard SigMF representation.
The logical Zarr representation uses the little-endian array dtype recorded by
`sigmf-zarr:dtype`. For example, a `cf32_be` dataset imports with
`sigmf-zarr:dtype` `<f4`, retains `core:datatype` `cf32_be`, and exports the
original big-endian bytes.

## Sample axes

Sample axes make the tensor layout explicit. The stored
`sigmf-zarr:sample-axes` list must:

- Have the same length as `sigmf-zarr:sample-shape`
- Contain no duplicate axis names
- Contain exactly one `time` axis
- Not contain the reserved `item` axis
- Contain at most one `iq` axis
- Contain at most one `channel` axis
- Use an `iq` axis of length `2` when `iq` is present
- Contain no `iq` axis when `sigmf-zarr:dtype` is complex

Common unbatched RF layouts include:

```text
(time)
(iq, time)
(channel, iq, time)
```

Common homogeneous batch layouts include:

```text
(item, iq, time)
(item, channel, iq, time)
```

In batched layouts, `item` is a runtime axis derived from the `samples` array
shape. It is not written into `sigmf-zarr:sample-axes`.

## Per-item metadata

A batched recording may contain an `item_metadata` array with shape `(N,)`,
where `N` is the length of the runtime `item` axis. The array uses
variable-length UTF-8 strings, and every element is a JSON object. An empty
object means that the item uses only shared recording metadata.

An item metadata object may contain these keys:

`global`
: Object whose fields shallowly override the shared recording `global` object.
  It must not contain `core:extensions` or `sigmf-zarr:*` storage fields.

`captures`
: List of capture objects appended to the shared `captures` list for the item.

`annotations`
: List of annotation objects appended to the shared `annotations` list for the
  item.

No other top-level keys are permitted. The array is optional, and its presence
is derived directly from the recording group rather than duplicated in global
metadata. It is valid only for batched recordings and must always remain
aligned with the item axis. Appending samples extends an existing metadata
array with `{}` entries unless metadata for the new items is supplied
explicitly.

Every `item_metadata` chunk is protected by CRC32C. In Zarr format 3, CRC32C
is the final bytes-to-bytes codec after compression. In Zarr format 2, the
filter pipeline is VLenUTF8 followed by CRC32C, followed by the array
compressor. Readers must reject an `item_metadata` array without CRC32C.

The fill value of the `item_metadata` array must be the empty string. Zarr
reads a missing chunk as fill values, and the empty string is not a JSON
object, so a missing chunk is invalid. Readers must reject any other fill
value, including `null`, before reading entries. Otherwise a small store that
declares a very long item axis could make a reader decode one chunk of valid
fill values after another without bound.

The fill-value rule bounds reads of missing chunks only. A stored chunk can
compress many entries into a few bytes, so a reader that decodes every entry
must bound its own work.

Dense scalar fields such as modulation class or SNR should remain typed
recording indexes. `item_metadata` is intended for irregular or nested
per-example metadata.

## Channel metadata

The required `channels/` group is empty when no explicit `channel` axis is
declared. When a channel axis exists, it contains exactly one numbered subgroup
per channel:

```text
channels/
|-- 0/
|-- 1/
`-- ...
```

Each subgroup's attributes form a free-form JSON metadata object for that
channel.
Applications should use namespaced fields for domain-specific values such as
antenna position, polarization, receiver identity, and calibration. The group
names must be the contiguous decimal indexes from `0` through
`sigmf-zarr:num-channels - 1`.

The `item` and `channel` axes are orthogonal. Items are independent examples.
Channels are simultaneous receivers that share the same time axis.

The implementation can infer default axes for common shapes:

- Rank 1: `(time)`
- Rank 2 with first dimension `2`: `(iq, time)`
- Rank 2 with second dimension `2`: `(time, iq)`
- Rank 2 otherwise: `(channel, time)`
- Rank 3 with middle dimension `2`: `(channel, iq, time)`
- Rank 3 with first dimension `2`: `(iq, channel, time)`
- Rank 3 with last dimension `2`: `(time, channel, iq)`

For other shapes, writers should provide `sample_axes` explicitly.

When importing standard SigMF, `core:num_channels` is mapped into an explicit
`channel` axis. Real multi-channel streams import as `(channel, time)`.
Complex multi-channel streams import as `(channel, iq, time)`. The flat
standard SigMF stream is interpreted as time-major, channel-interleaved data.
Complex `cf32_*` datasets use 32-bit floating-point I/Q components in Zarr,
while `cf64_*` datasets retain 64-bit components. Export applies the byte order
declared by `core:datatype`, so conforming `cf64_le` and `cf64_be` dataset bytes
round-trip exactly through either Zarr physical format.

When exporting an unbatched recording to standard SigMF, the declared
`channel` axis is mapped to `core:num_channels`. If `core:num_channels` is
already present, it must be a positive integer that matches the `channel` axis
length, or `1` when no `channel` axis is declared. If it is absent and the
recording has more than one channel, export fills it from the axis length.

## Captures and appends

Capture and annotation metadata follows the SigMF model and is stored in the
recording `captures` and `annotations` attributes.

When `append_samples(..., capture=...)` receives capture metadata, the writer
adds a capture entry to `captures`. If the entry does not already include
`core:sample_start`, SigMF-Zarr fills it in:

- For batched recordings, `core:sample_start` defaults to the old `item` count
- For unbatched recordings, `core:sample_start` defaults to `core:offset`
  (zero when absent) plus the old `time` sample count

In a batched recording, shared capture starts index items. The capture with
the greatest `core:sample_start` at or before an item supplies its shared
capture fields until the next shared capture. Duplicate starts use the last
entry. A shared `core:datetime` timestamps the first stored time sample of the
item at that offset. A shared `core:global_index` also applies only to that
item. Neither source anchor extends to subsequent items.
Items are independent examples and need not be consecutive in time.

Per-item capture starts are absolute sample indices. `core:offset` gives the
index of the item's first stored time sample. Captures describe transitions
within the signal and take precedence over shared captures at duplicate time
positions. The resolved signal view uses shared capture fields as defaults for
each per-item capture, except for `core:datetime`, `core:global_index`, and
`core:sample_start`. Each per-item source anchor must be supplied explicitly.
Shared and per-item annotations retain time-sample
coordinates. Stored metadata and the JSON resolution rules remain unchanged.

For independently acquired items, `append_samples(..., item_captures=...)`
accepts one capture list or `None` per appended item. Each list is stored in
that item's `item_metadata` bundle. An item may have multiple captures with
distinct timestamps and acquisition settings. The writer does not broadcast
timestamps or infer temporal continuity between items or captures.

The first capture in each supplied list defaults `core:sample_start` to the
item's resolved `core:offset`, or zero when absent. Later captures must specify
`core:sample_start`. Starts must be integers from 0 to $2^{63} - 1$ in the
item's source time-sample coordinates. A per-item `core:offset` must be an
integer in the same range, the bound that the SigMF schema places on the
recording-level field. Explicit starts are preserved. A `None` entry adds
no captures, and an empty list supplies an empty per-item capture list.

`item_captures` is valid only for batched recordings and must have the same
length as the appended batch. It may accompany shared `capture` metadata and
per-item `global` or `annotations` metadata. An item must not supply captures
through both `item_captures` and `item_metadata`. A `None` entry in
`item_captures` leaves captures supplied through `item_metadata` unchanged.
The writer validates and copies the supplied metadata before changing samples,
metadata arrays, indexes, or integrity state.

Unbatched appends extend the explicit `time` axis and update
`sigmf-zarr:sample-shape`. Batched appends extend the leading runtime `item`
axis and keep `sigmf-zarr:sample-shape` unchanged.

## Metadata indexes

Indexes are dense one-dimensional Zarr arrays that materialize metadata values
along a sample axis. They are useful for labels, quality metrics, frequency
bins, time-varying parameters, and ML dataset metadata.

SigMF-Zarr has two index scopes:

- Recording-level indexes under `recordings/<recording_name>/indexes/`
- Store-wide indexes under root `indexes/`

### Recording-level indexes

A recording-level index lives at:

```text
recordings/<recording_name>/indexes/<index_name>
```

The array must be one-dimensional. Its length must match the length of the
runtime sample axis named by its `axis` attribute. Runtime axes are:

- For unbatched recordings: `sigmf-zarr:sample-axes`
- For batched recordings: `("item", *sigmf-zarr:sample-axes)`

An index name may be a nested relative path, such as `quality/snr_db`. Each
`/`-separated segment must be nonempty, must not be `.` or `..`, and must not
be a reserved Zarr metadata document name. The name must not contain NUL. An
index name identifies an array, never a group.
Validation, axis invalidation, and integrity checks include indexes at every
depth below the `indexes/` group.

Each recording-level index has these attributes:

`axis`
: Required. Runtime sample axis indexed by the array, such as `item`, `time`,
  `channel`, or a custom sample axis.

`field`
: Required. Metadata field represented by the index, such as
  `radioml:snr`, `radioml:mod_class`, or an application-specific namespaced
  field.

`kind`
: Required. Describes the index kind. The default value is `metadata`.

`unit`
: Optional string. Unit for numeric values, such as `dB` or `Hz`.

`labels`
: Optional. Lookup values for encoded indexes. For example, an integer label
  index can store human-readable class names in `labels`.

`sigmf-zarr:valid`
: Optional Boolean. The value `false` marks an index whose mutation did not
  complete or whose aligned axis changed. Absence or `true` means valid.
  Readers must reject an invalid index until a writer repairs or replaces it.
  Readers must also reject an index whose `sigmf-zarr:valid` value is not a
  Boolean, such as `0` or `"false"`.

`sigmf-zarr:invalid-reason`
: Optional string explanation accompanying `sigmf-zarr:valid=false`. Readers
  must reject an index whose `sigmf-zarr:invalid-reason` value is not a string.

`validity`
: Reserved. Its presence declares that some index values are missing and that
  a validity mask identifies the present values. Schema version 1 does not
  define the mask representation. A reader that requires one value for every
  position, such as queries, splits, the PyTorch adapter, and index projection
  during export, must reject an index that declares `validity`.

An index array is a materialized metadata vector, not an automatically
maintained secondary index. If a writer appends samples after creating an
axis-aligned index, the index becomes invalid and the writer must update or
replace it before use. Integrity recalculation must reject a store containing
an invalid index rather than hashing it as current.

## Logical integrity

Sample chunks use CRC32C by default. In Zarr format 3, CRC32C is the final
bytes-to-bytes codec for each logical chunk, after compression, so reads detect
corruption before decompression. In Zarr format 2, CRC32C is a numcodecs filter
applied before the single compressor and is verified after decompression. For
sharded Zarr format 3 arrays, CRC32C is part of the inner logical-chunk codec
pipeline. The shard index retains its own Zarr checksum independently. Zarr
format 2 does not support sharding.

Writers may disable sample checksums explicitly, in which case
`sigmf-zarr:checksum` is `null`. Readers must verify that this metadata field
agrees with the sample array codec pipeline.

An unbatched recording may also carry standard SigMF `core:sha512`. The digest
covers the exact byte stream produced for its `.sigmf-data` file, after applying
the layout and byte order declared by `core:datatype`. It does not cover Zarr
metadata, compressed chunk bytes, or a `.sigmf` archive container. This makes
the digest independent of whether the physical store uses Zarr format 2 or
Zarr format 3.

`core:sha512` is optional because it is an O(N) recording-level check and cannot
be updated incrementally after arbitrary writes. Managed sample replacement and
append operations must remove it before changing samples. A writer may then
recalculate it after the mutation is complete. Public sample and index handles
are read-only. Writable access is available only through mutation contexts,
which invalidate affected digests before returning the backing array. Chunk
CRC32C remains the automatic, local integrity mechanism for normal Zarr reads.
SHA-512 provides portable identity for the reconstructed standard SigMF
dataset.

SigMF-Zarr also defines optional native logical SHA-512 hashes. These hashes
cover the data model rather than serialized Zarr metadata or encoded chunk
bytes, so they have the same values for equivalent Zarr format 2 and Zarr
format 3 stores. Each participating group has an `integrity` attribute with
`algorithm="sha512"`, `version=1`, and one or more scope-specific fields. The
`version` value must be the JSON integer `1`, not `1.0`. The attribute, when
present, must be a JSON object:

- A recording has `sample_sha512` over the logical `samples` array and
  `metadata_sha512` over all recording metadata.
- A collection has `metadata_sha512` over its attributes, arrays, and child
  groups.
- The root has `metadata_sha512` over root attributes, store-wide indexes, any
  other root content, and the digests that each recording and collection
  declares.

A hash excludes only the `integrity` attribute of the group that it covers.
An attribute named `integrity` on any other group, such as a channel group or
the `recordings` group, is ordinary content and is hashed. JSON values use the
canonical encoding defined below.

### Array hashes

An array hash divides the logical array into slabs along one axis, the slab
axis. A sample array uses its growth axis: `item` for a batched recording and
`time` otherwise. Every other array uses its first axis. Each slab holds $L$
consecutive slab-axis positions with the complete extent of every other axis.
The final slab may be shorter. The slab length is

$$
L = \max\left(1, \left\lfloor \frac{2^{24}}{s P} \right\rfloor\right),
$$

where $s$ is the NumPy item size in bytes and $P$ is the product of the other
dimensions, or 1 when that product is zero. Variable-length strings and objects
use $s = 16$. A fixed-width Unicode string of width $n$ uses $s = 4n$.

Each slab digest is SHA-512 over the slab's values in C order:

- Boolean, integer, floating-point, complex, date-time, and time-delta values
  use their little-endian bytes. A Boolean value is one byte, 0 or 1.
- A fixed-width byte string uses its raw bytes, padded with NUL bytes to the
  declared width.
- Every other value is converted to a JSON value, encoded as canonical JSON,
  and preceded by its encoded length as an 8-byte big-endian unsigned integer.
  Each element of a variable-length or fixed-width Unicode string array is a
  JSON string. For example, the string `bc` contributes the length
  `00 00 00 00 00 00 00 04` followed by the four bytes `"bc"`. A fixed-width
  Unicode value excludes trailing U+0000 code points, as NumPy reads it. Each
  element of an object array is the JSON value that it holds.

The array digest is SHA-512 over the ASCII bytes `sigmf-zarr-array-v1`, a NUL
byte, the canonical JSON digest descriptor, a NUL byte, and the concatenated
64-byte slab digests. The digest descriptor has these fields:

- `dtype` identifies the logical element type as described below.
- `shape` is the logical shape.
- `slab_axis` is the slab axis, or `null` for a zero-dimensional array.
- `slab_length` is $L$, or 1 for a zero-dimensional array.
- `version` is the integrity version, 1.

The `dtype` field of the digest descriptor depends only on the element type,
not on the NumPy release, host byte order, or stored byte order:

- Boolean, integer, floating-point, complex, date-time, and time-delta types
  use the NumPy type string with little-endian byte order, such as `<i4`,
  `<f8`, `<c8`, and `<M8[ns]`. One-byte types use `|`, as in `|b1`, `|i1`, and
  `|u1`.
- Fixed-width byte strings use `|S` followed by the width in bytes, such as
  `|S3`.
- Fixed-width Unicode strings use `<U` followed by the width in code points,
  such as `<U2`, for either stored byte order.
- Variable-length UTF-8 strings use `string`, the Zarr format 3 data type name.
  This descriptor applies to Zarr format 3 `string` arrays, to Zarr format 2
  arrays with dtype `|O` and the `vlen-utf8` filter, and to NumPy `StringDType`
  arrays. It does not depend on how NumPy spells `StringDType`, which is `|T16`
  in NumPy 2.0 and `StringDType()` in later releases.
- Object arrays use `|O`. An object array that holds strings therefore hashes
  differently from a variable-length string array with the same values.

No array in a store may have a structured or void dtype, which is NumPy kind
`V`. The `dtype` descriptor does not describe field names, field dtypes, or
field offsets, so a hash could not detect a change to them. Writers reject such
arrays, and `validate_store()` reports them.

A zero-dimensional array has one slab containing its value with the array's
declared dtype. For example, a zero-dimensional `|S3` array holding `ab`
contributes the bytes `ab` and one NUL byte. An array without elements has no
slabs. Slabs depend only on the logical dtype and shape, so a reader can hash
one slab at a time and decode each physical chunk once, for any chunk layout.

### Metadata hashes

A recording or collection metadata hash covers group and array structure,
group and array attributes, array dtype and shape, and the array hash of each
metadata array, such as an index. Sample array values are excluded from the
recording metadata hash because the sample hash covers them. The `samples`
entry still records the array's `attrs`, `dtype`, and `shape`. The hash input
is the ASCII bytes `sigmf-zarr-metadata-v1`, a NUL byte, and the canonical
JSON payload.

The payload of a group is a JSON object with these members:

- `arrays` maps each child array name to an object with four members. `attrs`
  holds the array attributes, `dtype` is the `dtype` field of the array's
  digest descriptor, such as `<f4` or `string`, `shape` is the logical shape,
  and `sha512` is the array hash. The recording `samples` entry omits `sha512`
  rather than setting it to `null`.
- `attrs` holds the group attributes. The hashed group omits its own
  `integrity` attribute.
- `groups` maps each child group name to the payload of that group, built by
  the same rule. A child group payload keeps every attribute, including one
  named `integrity`.

The store-wide metadata hash uses the same payload for root attributes,
store-wide indexes, and any other root content. In place of each recording and
collection group, the payload records the digests that the group declares:
`metadata_sha512` and `sample_sha512` for a recording, and `metadata_sha512`
for a collection. Specifically, `groups.recordings.groups` maps each recording
name to an object with the members `metadata_sha512` and `sample_sha512`, and
`groups.collections.groups` maps each collection name to an object with the
member `metadata_sha512`. The `recordings` and `collections` payloads keep
their own `arrays` and `attrs`, including any `integrity` attribute. The hash
input is the ASCII bytes `sigmf-zarr-store-v1`, a NUL byte, and the canonical
JSON payload. Updating the store-wide hash therefore requires current hashes
for every recording and collection, but it does not reread their arrays.
Verifying the store-wide hash alone confirms root content and the declared
digests. Full verification also recalculates every recording and collection
hash.

### Hash maintenance

Managed API mutations and writable mutation contexts remove every affected
native hash before writing. A failed index mutation leaves the index explicitly
invalid. A full `store.update_integrity()` pass recalculates recording,
collection, and root hashes, while `store.verify_integrity()` requires all of
those hashes to be present and current. Importers hash the resource they write.
They also update the store-wide hash when every other recording and collection
has current hashes, and otherwise leave it absent. These whole-content hashes
complement chunk CRC32C. They do not authenticate a store against an attacker
who can alter both content and stored hashes or bypass the API to modify the
backing Zarr store directly.

At the standard SigMF boundary, import verifies every available standard hash
and creates native hashes for content that passes its checks. `core:sha512`
checks each `.sigmf-data` byte stream. A `.sigmf-collection` additionally checks
its declared hashes for the referenced `.sigmf-meta` files. Standalone
`.sigmf-meta` files have no standard field that hashes their own metadata, so
there is nothing external to verify in that case. Export always calculates
`core:sha512` from the emitted data bytes. Archive export calculates collection
stream hashes from the emitted metadata bytes. If a native internal hash is
present but stale, export fails rather than silently blessing changed content.

### Canonical JSON encoding

Version 1 hashes encode JSON values with these rules, which an independent
implementation must reproduce exactly:

- Object members are sorted by key, comparing keys as sequences of Unicode code
  points. This order differs from RFC 8785, which compares UTF-16 code units,
  when keys contain characters outside the Basic Multilingual Plane.
- No whitespace appears between tokens. Members and elements are separated by
  `,`, and keys are separated from values by `:`.
- Strings are encoded as UTF-8. `"` and `\` are escaped as `\"` and `\\`.
  Backspace, form feed, line feed, carriage return, and tab are escaped as
  `\b`, `\f`, `\n`, `\r`, and `\t`. Other characters from U+0000 through
  U+001F are escaped as `\u00XX` with lowercase hexadecimal digits. All other
  characters, including `/`, U+007F, and non-ASCII characters, appear
  unescaped.
- Integers are written in decimal without an exponent or leading zeros.
- Floating-point numbers use the shortest decimal string that rounds to the
  same binary64 value, as produced by Python `repr()`. The string always
  contains a decimal point or an exponent. Values whose decimal exponent is
  less than -4 or at least 16 use exponent notation with a sign and at least
  two exponent digits, such as `1e-05`, `2.5e-07`, and `1e+16`. Other values use
  positional notation, such as `0.0001`, `1.0`, and `-0.0`.
- Integers and floating-point numbers remain distinct, so `1` and `1.0` hash
  differently.
- Literals are written as `true`, `false`, and `null`. Non-finite numbers are
  rejected.

## Store-wide indexes

Store-wide indexes live under root `indexes/`:

```text
indexes/<index_name>
```

These arrays are also one-dimensional, but schema version 1 does not require
standard attributes for them. They are intended for cross-recording lookup
data, manifests, or other container-level query aids. String values are stored
as variable-length UTF-8 arrays, and JSON object indexes can be stored as JSON
strings and decoded by the Python API. Store-wide indexes use the same optional
`sigmf-zarr:valid` and `sigmf-zarr:invalid-reason` mutation markers as
recording-level indexes.

## Collections

Collections live at `collections/<collection_name>/`. A collection group has
two required attributes and may have the optional `integrity` attribute:

- The `metadata` attribute is a JSON object that describes the collection.
- The `recording_ids` attribute is a list of recording names that the collection
  references.

The `metadata` object must satisfy the SigMF collection schema for every field
that it contains. Export supplies `core:version` when it is absent and derives
`core:streams` from `recording_ids`, so the object may omit both.
`validate_store()` checks the object, except `core:streams`, against the
collection schema of the installed sigmf package.

Each recording identifier must name a recording in the store and must appear at
most once in `recording_ids`. Writers reject a repeated identifier, and
`validate_store()` reports one. The Python API rejects creating a collection
that references a missing recording and rejects removing a recording that a
collection references. Readers should tolerate dangling references written by
other tools, and `validate_store()` reports them.

Collections are useful for representing relationships among recordings without
moving or duplicating sample arrays.

## Dataset-specific metadata

Dataset-specific metadata should use namespaced fields. Prefer generic
recording-level indexes when the data is dense and aligned with a sample axis.
Use `extensions/` only when the data is structured in a way that does not fit a
one-dimensional axis-aligned index.

The RadioML importer follows this pattern:

- The `global["radioml:source_dataset"]` field records the source dataset name.
- The `indexes/mod_class_id` array stores integer modulation class IDs with
  `axis="item"`, `field="radioml:mod_class"`, and `labels=[...]`.
- The `indexes/snr_db` array stores SNR values with `axis="item"`,
  `field="radioml:snr"`, and `unit="dB"`.

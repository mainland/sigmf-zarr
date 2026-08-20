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
- The `schema_version` attribute must be the integer `1`.

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

Recording and collection identifiers must be nonempty names of immediate child
groups. Identifiers must not be `.` or `..`, contain `/` or `\`, or contain a
NUL character.

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

The `samples` array stores the recording's sample tensor. The `extensions/`
group is reserved for extension-owned structured data that does not fit the
generic index model. The `indexes/` group stores dense metadata arrays aligned
with sample axes. The required `channels/` group stores per-channel metadata
when the sample tensor declares a `channel` axis. The optional `item_metadata`
array stores structured metadata for individual items in a batched recording.

## SigMF-Zarr metadata extension

The normative namespace definition is
[`sigmf-zarr.sigmf-ext.md`](sigmf-zarr.sigmf-ext.md). This chapter specifies
the larger Zarr container model around those SigMF metadata fields.

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

The extension defines these `global` fields:

`sigmf-zarr:dtype`
: NumPy dtype string for the element type stored in `samples`, such as
  `<f4`.

`sigmf-zarr:sample-shape`
: Shape of one logical sample tensor. For an unbatched recording, this is the
  full `samples` shape. For a homogeneous batched recording, this excludes the
  leading runtime `item` axis.

`sigmf-zarr:sample-axes`
: Axis names corresponding to `sigmf-zarr:sample-shape`.

`sigmf-zarr:num-channels`
: Number of simultaneous sample channels. This is the declared `channel` axis
  length, or `1` when no explicit channel axis exists.

`sigmf-zarr:checksum`
: Chunk-level checksum applied to sample data. Its value is `crc32c` or `null`.

These fields describe the stored `samples` array, and a writer must keep them
consistent with it. The Python API maintains them during creation and rejects
direct assignment through `set_global_field()`.

A recording is treated as batched when
`samples.ndim == len(sigmf-zarr:sample-shape) + 1`. The leading runtime axis is
then named `item`.

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
aligned with the item axis.

Every `item_metadata` chunk is protected by CRC32C. In Zarr format 3, CRC32C
is the final bytes-to-bytes codec after compression. In Zarr format 2, the
filter pipeline is VLenUTF8 followed by CRC32C, followed by the array
compressor. Readers must reject an `item_metadata` array without CRC32C.

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

## Collections

Collections live at `collections/<collection_name>/`. A collection group has
two required attributes:

- The `metadata` attribute is a JSON object that describes the collection.
- The `recording_ids` attribute is a list of recording names that the collection
  references.

Each recording identifier must name a recording in the store. The Python API
rejects creating a collection that references a missing recording and rejects
removing a recording that a collection references. Readers should tolerate
dangling references written by other tools.

Collections are useful for representing relationships among recordings without
moving or duplicating sample arrays.

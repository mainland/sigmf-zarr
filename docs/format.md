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

### Physical Zarr format

The logical SigMF-Zarr schema can be stored using Zarr format 2 or Zarr format
3. New stores use Zarr format 3 by default. Readers auto-detect either physical
format. The Python implementation uses zarr-python 3.2 or newer for both. Zarr
Zarr format 2 compatibility does not require or support the older zarr-python
2.x runtime.

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
use logical chunks targeting approximately 4 MiB instead. Sample compression
defaults to Zstandard. The command-line importers also support Zstandard, LZ4,
and high-compression LZ4 through Blosc.

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
the recording `global` object. A recording declares the extension in
`global["core:extensions"]`:

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

A recording is treated as batched when
`samples.ndim == len(sigmf-zarr:sample-shape) + 1`. The leading runtime axis is
then named `item`.

SigMF-Zarr extension fields are stripped when exporting a recording back to
standard SigMF metadata. Other extension declarations and non-`sigmf-zarr:*`
fields are preserved.

## Complex sample representation

Standard SigMF and SigMF-Zarr represent the same logical complex samples
differently:

| Representation | Standard SigMF | SigMF-Zarr |
| --- | --- | --- |
| Container | Flat `.sigmf-data` byte stream | Chunked `samples` Zarr array |
| Complex sample | Interleaved I and Q components described by a `cf*` datatype | Real-valued elements with an explicit length-2 `iq` axis |
| Single channel | Time-ordered complex samples | `(iq, time)` |
| Multiple channels | Time-major, channel-interleaved complex samples | `(channel, iq, time)` |

SigMF-Zarr does not use a complex Zarr dtype, and it does not store I and Q in
separate arrays. Index `0` of the `iq` axis contains the in-phase component and
index `1` contains the quadrature component. A standard SigMF import splits
each complex sample into those two planes. Export combines the planes into
complex samples again and writes the interleaved byte stream required by the
declared `core:datatype`.

For floating-point complex data, `cf32_*` maps to a real-valued Zarr array with
32-bit components and `cf64_*` maps to one with 64-bit components. Import and
export preserve the declared byte order in the standard SigMF representation.
the logical Zarr representation uses the array dtype recorded by
`sigmf-zarr:dtype`.

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
aligned with the item axis. Appending samples extends an existing metadata
array with `{}` entries unless metadata for the new items is supplied
explicitly.

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
channels are simultaneous receivers that share the same time axis.

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
- For unbatched recordings, `core:sample_start` defaults to the old `time`
  sample count

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

An index name may be a nested relative path, such as `quality/snr_db`.
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
: Optional. Unit string for numeric values, such as `dB` or `Hz`.

`labels`
: Optional. Lookup values for encoded indexes. For example, an integer label
  index can store human-readable class names in `labels`.

`sigmf-zarr:valid`
: Optional. The value `false` marks an index whose mutation did not complete or
  whose aligned axis changed. Absence means valid. Readers must reject an
  invalid index until a writer repairs or replaces it.

`sigmf-zarr:invalid-reason`
: Optional explanation accompanying `sigmf-zarr:valid=false`.

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
Zarr format 2 does not support sharding.

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
`algorithm="sha512"`, `version=1`, and one or more scope-specific fields:

- A recording has `sample_sha512` over the logical `samples` array and
  `metadata_sha512` over all recording metadata.
- A collection has `metadata_sha512` over its attributes, arrays, and child
  groups.
- The root has `metadata_sha512` over all store metadata.

The sample hash includes the array's logical dtype, shape, and values in C
order. Numeric values are normalized to little-endian bytes. The metadata hash
includes group and array structure, group and array attributes, array dtype and
shape, and the logical values of metadata arrays such as indexes. Sample array
values are excluded from metadata hashes because the recording sample hash
covers them. The sample array descriptor remains included. Every `integrity`
attribute is excluded from hash input to avoid self-reference. JSON objects use
UTF-8, lexicographically sorted keys, compact separators, and no non-finite
floating-point values.

Managed API mutations and writable mutation contexts remove every affected
native hash before writing. A failed index mutation leaves the index explicitly
invalid. A full `store.update_integrity()` pass recalculates recording,
collection, and root hashes, while `store.verify_integrity()` requires all of
those hashes to be present and current. These whole-content hashes complement
chunk CRC32C. They do not authenticate a store against an attacker who can
alter both content and stored hashes or bypass the API to modify the backing
Zarr store directly.

At the standard SigMF boundary, import verifies every available standard hash
and creates native hashes for content that passes its checks. `core:sha512`
checks each `.sigmf-data` byte stream. A `.sigmf-collection` additionally checks
its declared hashes for the referenced `.sigmf-meta` files. Standalone
`.sigmf-meta` files have no standard field that hashes their own metadata, so
there is nothing external to verify in that case. Export always calculates
`core:sha512` from the emitted data bytes. Archive export calculates collection
stream hashes from the emitted metadata bytes. If a native internal hash is
present but stale, export fails rather than silently blessing changed content.

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

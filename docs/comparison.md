# Relationship to standard SigMF

SigMF-Zarr preserves the metadata vocabulary and signal semantics of standard
SigMF while using Zarr as its storage model. It is intended to complement
standard SigMF, not redefine its file formats. Standard SigMF remains the
interchange boundary. SigMF-Zarr provides an array-oriented representation for
analysis, machine learning, and datasets that benefit from incremental access.

## Shared concepts

Both representations organize metadata into the same principal objects:

- The `global` object describes the recording as a whole.
- The `captures` list describes changes in acquisition state along the sample
  timeline.
- The `annotations` list describes sample regions and points of interest.
- The `core:datatype` field identifies the logical sample encoding.
- The `core:extensions` field declares additional metadata namespaces.
- Collections relate multiple recordings.

SigMF-Zarr stores these objects as JSON-compatible Zarr group attributes. It
adds namespaced fields such as `sigmf-zarr:sample-shape` and
`sigmf-zarr:sample-axes` to describe its array representation. Those storage
fields are removed when exporting standard SigMF metadata.

## Representation comparison

| Area | Standard SigMF | SigMF-Zarr |
| --- | --- | --- |
| Recording | `.sigmf-meta` plus a flat `.sigmf-data` stream | Recording group containing metadata and arrays |
| Archive or dataset | Optional tar-based `.sigmf` archive | Hierarchical Zarr store |
| Samples | Flat binary sequence described by `core:datatype` | Typed, chunked N-dimensional `samples` array |
| Complex samples | Interleaved I and Q components | One real-valued array with an explicit `iq` axis |
| Channels | Time-major channel interleaving with `core:num_channels`, or separate recordings | Explicit `channel` axis with per-channel metadata groups |
| Multiple examples | Usually separate recordings or an application convention | Optional leading `item` axis for homogeneous batches |
| Dense labels | JSON annotations, extensions, or external data | Typed indexes aligned with `item`, `time`, or another axis |
| Compression | External or container-level | Independent array-chunk compression |
| Partial access | Byte-offset access to an unpacked data file | Array slicing that reads intersecting chunks |
| Mutation | Files or archives are commonly rewritten | Recordings, metadata, and arrays can be updated in place |
| Integrity | Whole-file SHA-512 fields | Standard hashes plus logical hashes and per-chunk checksums |

These are storage differences, not changes to the meaning of the underlying
signal. For example, importing a standard complex recording splits every
complex sample across the SigMF-Zarr `iq` axis. Export combines that axis and
writes the interleaved standard byte stream again.

## Sample access and chunking

An unpacked standard `.sigmf-data` file is simple and efficient for sequential
I/O. A reader that knows the datatype, channel count, and sample offset can
also seek directly to a byte range. The file itself does not define chunks,
however, so compression and application-level access units must be managed
outside the sample stream.

Zarr divides an array into independently addressable chunks. Reading a time
slice or batch loads and decompresses only the chunks intersecting that slice.
Chunk shapes can favor sequential scans, time windows, individual channels,
or batches of examples. Zarr format 3 can additionally place multiple chunks
in a shard, reducing the number of storage objects while retaining indexed
access within the shard. SigMF-Zarr importers use format 3 and approximately
4 MiB shards by default. Batched imports place approximately 256 KiB logical
chunks inside each shard. Format-2 imports use approximately 4 MiB chunks
without sharding.

Chunking introduces a design choice: a chunk shape that suits one workload may
be inefficient for another. Very small recordings may also gain little from
chunking and can have proportionally greater metadata overhead than a flat
file.

## Complex samples and channels

Standard SigMF serializes a multi-channel stream in time-major,
channel-interleaved order. `core:num_channels` supplies the information needed
to interpret that sequence. A reader can select one channel, but it must
de-interleave the corresponding values from the stream.

SigMF-Zarr makes channel and component dimensions explicit. Common layouts
are:

```text
(time)                       real, single channel
(iq, time)                   complex, single channel
(channel, time)              real, multiple channels
(channel, iq, time)          complex, multiple channels
```

The declared axis names, rather than their positions, define their semantics.
Applications can slice a channel directly, and the `channels/` group provides
a place for per-channel properties such as receiver identity, antenna
position, polarization, or calibration. All channels still share the
recording's capture timeline.

## Batches and axis-aligned metadata

SigMF recordings primarily describe a signal timeline. Collections group
related recordings, but they do not define a homogeneous tensor of training
examples.

SigMF-Zarr can store equal-shaped examples in one recording with a leading
runtime `item` axis. Dense values such as modulation class, SNR, or a model
output can be stored as typed indexes aligned with that axis. Indexes can also
align with `time` or `channel`. Irregular per-example metadata can use the
optional `item_metadata` array.

The batch model does not replace collections. Items represent independent,
equal-shaped examples inside one recording. Channels represent simultaneous
signals. Collections relate complete recordings.

## Extensions and additional arrays

SigMF's namespace extension mechanism is well suited to additional JSON
metadata. Large label tensors, derived features, calibration matrices, and
model outputs require an application convention or additional files because
they do not naturally fit in a metadata document.

SigMF-Zarr retains namespaced JSON metadata and also provides two structured
locations. Typed, axis-aligned values belong in `indexes/`. Extension-specific
groups and arrays belong in `extensions/`. New arrays or attributes can be
added without rewriting existing sample chunks. Consumers still need to
understand the relevant extension namespace, and changes to the common logical
layout require a new `schema_version` rather than an unannounced structural
change.

## Hierarchy, backends, and packaging

A standard SigMF recording uses a metadata/data file pair. A standard SigMF
archive packages one or more recordings and an optional collection in a tar
container. This is a compact, portable interchange form, although changing an
archive generally requires rewriting it and member-oriented access depends on
the archive reader.

A SigMF-Zarr store uses named groups and arrays for recordings, collections,
indexes, and extension data. Zarr can operate over compatible directory,
key-value, and object-storage backends. This permits the same logical schema to
be used locally or remotely, subject to the capabilities and consistency model
of the selected backend.

Directory and object-store layouts may contain many keys. Zarr format 3
sharding can reduce that count. Single-file packaging can be convenient for
transport, but its update and concurrency characteristics may differ from a
directory or object-store deployment.

## Compression and integrity

Standard SigMF's `core:sha512` covers the bytes of a recording's data file.
Collection stream hashes can cover metadata files. These hashes provide strong
whole-file identity and corruption detection, but verifying one requires
reading the complete corresponding file.

SigMF-Zarr uses several complementary integrity layers:

- CRC32C detects corruption when individual sample chunks are read.
- A logical sample SHA-512 covers the recording's sample values independently
  of Zarr format, chunks, compression, or sharding.
- A logical metadata SHA-512 covers the recording metadata model.
- A store metadata SHA-512 covers store-level logical metadata.
- Standard SigMF hashes are verified during import and regenerated during
  export.

Managed mutations invalidate hashes and derived indexes that may no longer
describe the changed data. Applications must still choose appropriate locking
or transaction behavior when multiple writers use the same storage backend.
the schema does not make arbitrary concurrent mutations atomic.

## Import, export, and interoperability

The package imports standard recording pairs and standard SigMF archives. The
conversion copies SigMF metadata into Zarr attributes, maps complex and channel
interleaving to explicit axes, and rechunks the samples. Available standard
hashes are checked during this process.

An unbatched SigMF-Zarr recording can be exported as a standard SigMF
recording. Export restores standard sample order and byte encoding, removes
SigMF-Zarr storage fields, validates sample-indexed metadata, and calculates
hashes over the files it writes. Batched recordings have no direct standard
SigMF recording equivalent and must first be separated or otherwise mapped by
the application.

Conversion is therefore semantic rather than a byte-for-byte wrapper around
the original files. Standard SigMF tools consume exported recordings without
needing to understand the SigMF-Zarr schema.

## Choosing a representation

Standard SigMF is often the better fit when:

- Broad interoperability with existing SigMF tools is the priority.
- A recording is written once and read mostly as a sequential stream.
- A small, simple file pair is preferable.
- The dataset is primarily exchanged or archived.

SigMF-Zarr is often the better fit when:

- Workloads repeatedly select time ranges, channels, or training examples.
- Samples benefit from chunk-level compression and validation.
- Recordings or metadata must be extended without rewriting a container.
- Channel, batch, label, or derived-array dimensions should be explicit.
- A hierarchical dataset must contain many recordings and related arrays.
- The storage backend is a directory or object store.

Many workflows can use both: standard SigMF for interchange and SigMF-Zarr as
the working representation used for analysis and model training.

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
- Zarr format 2 uses numcodecs filters and a single compressor.

## Collections

Collections live at `collections/<collection_name>/`. A collection group has
two required attributes:

- The `metadata` attribute is a JSON object that describes the collection.
- The `recording_ids` attribute is a list of recording names that the collection
  references.

Collections are useful for representing relationships among recordings without
moving or duplicating sample arrays.

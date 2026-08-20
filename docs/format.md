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
- Zarr format 2 uses numcodecs filters and a single compressor.

## Collections

Collections live at `collections/<collection_name>/`. A collection group has
two required attributes:

- The `metadata` attribute is a JSON object that describes the collection.
- The `recording_ids` attribute is a list of recording names that the collection
  references.

Collections are useful for representing relationships among recordings without
moving or duplicating sample arrays.

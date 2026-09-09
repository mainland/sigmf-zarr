# Security policy

## Supported versions

Only the newest published alpha is supported during initial format development.

## Reporting a vulnerability

Report security problems privately to `mainland@drexel.edu`. Do not include
sensitive recordings or credentials in a public issue. Include the affected
version, input format, impact, and a minimal reproducer when possible.

## Untrusted inputs

RadioML 2016 and RML22 files are Python pickle files, and `pickle.load` can
execute arbitrary code. The `sigmf-zarr import radioml2016` and
`sigmf-zarr import rml22` commands and `load_radioml2016_pickle()` do not use
`pickle.load` or NumPy's unpickling functions. Their loader interprets only the
pickle opcodes that NumPy array mappings use in protocols 0 through 5,
including Python 2 pickles. It provides the following guarantees:

- The loader never calls a function that the file names. A pickle may name
  only `numpy.ndarray`, `numpy.dtype`, the NumPy array reconstruction
  functions, and `_codecs.encode`, which the loader replaces with inert
  placeholders. The loader rejects any other global before resolving it.
- Each array has a little-endian or big-endian integer or floating-point dtype
  of 1, 2, 4, or 8 bytes. The loader builds the dtype from a validated type
  code and byte order, and rejects dtype state with subarrays, fields, object
  references, flags, or a nondefault item size. It does not pass dtype or
  array state from the file to NumPy.
- The result is a dictionary of at most 65,536 entries whose values are
  arrays. Each key is a `(modulation, snr)` pair of a string of at most 256
  characters and an integer in the `int16` range other than a Boolean.
  Because a pickle can refer to one stored label from many keys, the labels
  of all keys must hold no more characters in total than the file has bytes.
- Before it allocates an array, the loader checks that the recorded data
  matches the declared shape and dtype, and that the arrays built so far hold
  no more bytes than the file. The bytes that `_codecs.encode` produces have
  the same bound. A dictionary that refers to one array more than once must
  also stay within this bound.
- The loader limits a pickle to 1,048,576 opcodes, 16 open marks, and tuples
  of at most 32 items nested at most 2 levels deep. Text integers must fit in
  8 bytes and text memo indexes in 4 bytes, the ranges of the binary opcodes,
  so a pickle cannot choose many keys or memo indexes with one hash value.
  Apart from arrays and other values read from the file, the loader
  allocates at most about 120 bytes for each opcode, or about 120 MiB at the
  opcode limit. Nesting cannot exhaust the interpreter stack.

The importers also reject a dictionary entry with no items and a dictionary
of more than 4,096 entries, and combine consecutive entries into writes of at
most `batch_size` items. Provenance metadata records about 400 bytes for each
entry plus its label, and the store records each distinct label in the class
list and the `mod_class_id` index. With 4,096 entries and 256-character
labels, the metadata is therefore at most about 40 MB. A pickle that breaks
any of these rules raises `ValueError`. Pickle files are therefore subject to
the same guidance as the other untrusted inputs below.

SigMF archives, HDF5 files, ZIP files, `.tim` files, JSON metadata, and Zarr
stores should be treated as untrusted data. Run imports with operating
system permissions appropriate for the output location and apply external
resource limits when processing unknown or unusually large datasets.

The RadioML 2018 importer reads only the named HDF5 file and rejects external
raw storage, virtual datasets, and external links. The RadioML 2018, CSPB, and
Panoradio importers reject an item larger than 64 MiB before reading it and
read batches of at most 64 MiB of samples.

RadioML 2018 HDF5 inputs may use only the gzip, shuffle, and fletcher32
filters. The importer rejects any other filter before reading data, and
empties the HDF5 plugin search path during the import. HDF5's own gzip filter
inflates a chunk until its compressed stream ends, so the importer reads each
gzip chunk raw, inflates it at most one byte past the chunk's declared size,
and rejects a chunk that is larger, truncated, or fails its Fletcher-32
checksum. Label and SNR elements must be scalar numbers. The declared item
count can still exceed the file size when datasets hold only fill values. The
label arrays then use 4 bytes per declared item, and the import reads every
declared item.

CSPB directory import rejects symbolic links anywhere in the source tree, so
an extracted dataset cannot pull other local files into a store. A source
path that the caller names may itself be a link. SigMF import follows the
paths that the caller names, but rejects a `core:dataset` file that is a link
or not a regular file.

With `overwrite_store`, an importer replaces only an absent target, an empty
directory, or a SigMF-Zarr store, and refuses a local target that equals or
contains an input file. A RadioML 2016 or RML22 import receives a loaded
mapping, so only the first rule protects its pickle file.

Untrusted stores must not be opened for writing. Zarr follows symbolic links
inside a local store, so a write could modify or delete files outside the
store. Read-only views prevent accidental mutation through the public API, but
they are not a security boundary.

Validate an untrusted store before opening it in any other way. Before
reading a local store, `validate_store()` lists it without following links.
It reports every symbolic link, every FIFO, socket, or device, and every
directory that it cannot list, and then reads no store content. A FIFO or
device in a store can make any other read block or exhaust memory.

SigMF-Zarr never reads Zarr consolidated metadata, so a consolidated copy
cannot change what SigMF-Zarr reads, hashes, or writes. Other Zarr readers use
such a copy in place of the node metadata documents, so a store that contains
one can show them content that the integrity hashes do not cover.
`validate_store()` reports consolidated metadata.

Reading any part of a chunk or shard fetches and decodes all of it. A small
store can declare large or highly compressible chunks, so the memory and time
of a read are bounded by the chunk size, not by the requested selection.
Recording opens are structural by default and decode no per-item metadata.
For an untrusted store, pass `max_array_bytes` (`--max-array-bytes`) to
validation. Validation then reads no array, chunk, or shard whose nominal
decoded size exceeds the limit, and stops checking per-item metadata once the
decoded text exceeds it. Variable-length strings count 16 bytes per element,
so one small compressed chunk of long strings can still decode to more than
the limit.

For a remote store, the backend buffers each fetched object in memory, and an
HTTP server can stream an unbounded response for one chunk. SigMF-Zarr does
not limit response sizes or download time. When reading remote or untrusted
stores, apply an address-space or container memory limit, and configure
backend timeouts and response-size limits in `storage_options` where the
backend supports them.

Some imports hold or extract a whole input before writing it:

- The CSPB importer decompresses each `.tim` ZIP member in memory. It
  accepts only stored and deflate members that are not encrypted. It first
  requires the member's declared size to equal the size its `.tim` header
  declares, and rejects a header that declares more than 64 MiB of samples.
  It inflates a member at most one byte past that size.
- SigMF archive import extracts the complete archive into the system temporary
  directory, which must have space for the extracted files. Import accepts
  only an uncompressed tar file and rejects sparse members, so the extracted
  file data is no larger than the archive. It also rejects an archive whose
  regular files declare more bytes than the archive holds, and an archive
  that would create more files and directories, including implied parent
  directories, than it has 512-byte blocks. Each directory can still occupy
  one file-system block and one inode, so directories need up to about eight
  times their header size on a file system with 4 KiB blocks.

Before extracting a SigMF archive, import rejects every member other than a
regular file or directory, including symbolic links, hard links, and device
files, and every member with an absolute name or a `..` component. Extraction
also applies the `tarfile` `data` filter. The member checks do not depend on
the filter, which has known bypasses in CPython releases before
3.12.11 and 3.13.4 (CVE-2024-12718, CVE-2025-4138, CVE-2025-4330, and
CVE-2025-4517).

Import reads archive member headers itself. It requires each header to start
after the previous one and caps the member count at one per 512-byte block. A
header with a negative size therefore cannot make parsing loop, independently
of the interpreter. CPython fixed this `tarfile` loop in 3.12.12 and 3.13.6
(CVE-2025-8194).

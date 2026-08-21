# Security policy

## Supported versions

Only the newest published alpha is supported during initial format development.

## Reporting a vulnerability

Report security problems privately to `mainland@drexel.edu`. Do not include
sensitive recordings or credentials in a public issue. Include the affected
version, input format, impact, and a minimal reproducer when possible.

## Untrusted inputs

RadioML 2016 files are Python pickle files, and `pickle.load` can execute
arbitrary code. The `sigmf-zarr import radioml2016` command and
`load_radioml2016_pickle()` do not use `pickle.load`. Their loader resolves
only the NumPy globals needed to rebuild arrays and rejects a pickle that
references any other global before resolving it. The importer accepts only
real numeric arrays. The loader also rejects a pickle whose arrays hold more
data than the file, so a small file cannot expand into a large import. Pickle
files are therefore subject to the same guidance as the other untrusted inputs
below. The remaining risk is NumPy's validation of the array state recorded in
the file.

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

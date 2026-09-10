# Changelog

All notable changes to this project will be documented in this file. The
project follows Semantic Versioning for public releases and PEP 440 for Python
package versions.

## 1.0.0a2 - unreleased

This is the first public alpha of the initial SigMF-Zarr format draft.

The format and Python API may change incompatibly during the alpha series.
Data written by one alpha may require migration before a later alpha can read
it. Stable compatibility begins only when the initial format draft is declared
complete.

### Storage model

- Define the initial logical store, recording, collection, index, channel, and
  item-metadata model.
- Support Zarr format 2 and Zarr format 3 through zarr-python 3. A store uses
  one physical format.
- Open stores, recordings, and collections without creating them. Create them
  only through `create()`, which refuses an existing target unless
  `overwrite=True`.
- Record `sigmf-zarr:dtype` and create `samples` arrays in little-endian byte
  order in both Zarr formats. Recording creation and every importer convert
  big-endian sample values exactly, `sigmf_dtype_to_zarr()` returns the
  little-endian dtype, and validation rejects a big-endian `sigmf-zarr:dtype`.
- Restrict sample dtypes to Boolean, integer, floating-point, and complex
  kinds. Recording creation rejects text, byte-string, date-time, time-delta,
  object, void, and structured dtypes, and validation reports a stored one.
  Document complex sample dtypes as an alternative to the `iq` axis.
- Accept numcodecs `Blosc`, `Zstd`, and `GZip` and `zarr.codecs`
  `BloscCodec`, `ZstdCodec`, and `GzipCodec` compressors in both Zarr formats,
  translated to the store's format. Raise `ValueError` that names the
  accepted forms for any other compressor without an equivalent.
- Choose default chunks for extension and generic arrays of every rank,
  including zero-dimensional arrays.
- Store sample values only when conversion to the stored dtype preserves them
  exactly. Appends and replacements reject overflow, rounding, nonfinite values
  stored as integers, signedness wrapping, and discarded imaginary parts.
- Use default sample chunks of about 1 MiB along the growth axis. Batched
  Zarr format 3 imports use approximately 256 KiB chunks and 4 MiB shards,
  unbatched Zarr format 3 imports use approximately 1 MiB chunks, and Zarr
  format 2 imports use 4 MiB chunks.
- Append distinct per-item capture lists with independent timestamps through
  `item_captures`, preserving shared item-offset captures and validating all
  metadata before storage changes.
- Derive per-item metadata presence from the array, protect its chunks with
  CRC32C, and provide validated entry and slice updates.
- Support structural recording opens that defer per-item JSON validation
  until access, while retaining full validation by default.
- Require recording and collection names to identify immediate child groups.
  Require collection recording identifiers to name existing recordings, and
  reject removal of a referenced recording.
- Require index, split, and extension-array names to identify one array below
  their group. Reject empty, `.`, and `..` segments, and never treat or replace
  a group as an index.
- Treat a store location string as a URL only when it starts with an explicit
  scheme, such as `s3://`, `file://`, or `memory://`. Any other string, and
  every path-like object, names a local directory, so `#` and `?` are literal.
- Reserve the Zarr metadata document names `.zarray`, `.zattrs`, `.zgroup`,
  `.zmetadata`, and `zarr.json` in both Zarr formats. Recording and collection
  names and segments of index, split, and extension-array names must not use
  them, and validation reports a recording or collection that does.
- Reject direct assignment of `sigmf-zarr` storage fields and keys with lone
  surrogates through `set_global_field()`, and maintain the canonical storage
  extension declaration in `core:extensions`.
- Raise `TypeError` for Boolean and non-integer item positions in the item
  metadata getters and setters before any read or write.
- Check array targets and encoding options before invalidating hashes, so a
  rejected call leaves every hash in place.
- Roll back failed creations and replacements of recordings, collections,
  indexes, extension arrays, and item metadata, failed appends and sample
  replacements, and failed collection mutations. Retain recovery files if
  restoration fails. Local rollback moves the backup back into place and
  needs no free disk space.
- Remove recordings from local directory stores transactionally. A failed
  store hash update restores the recording and the store hash. Other
  backends delete directly, and a failed deletion can leave a partial
  recording that validation reports.
- Restore the `item_metadata` shape when `mutate_item_metadata()` resizes it,
  as `mutate_samples()` does for samples.
- Require the empty string as the `item_metadata` fill value, and reject any
  other fill value when opening a recording. A missing chunk is therefore
  invalid, and a small store cannot make a full open decode fill values for
  every item that its shape declares.
- Reject a Boolean `schema_version`, and reject a non-string index `unit` in
  `add_index()`.
- Make an `append_samples()` call with zero items or samples a no-op that
  keeps every hash, and reject one that passes `capture`, because a capture
  must describe at least one sample.
- Provide `SignalView` for bounded time reads and resolved metadata of selected
  items and continuous recordings.
- Export the `SigMFRecordings` and `SigMFCollections` proxy view types from
  `sigmf_zarr.store`.
- Require the canonical NumPy dtype string for `sigmf-zarr:dtype`, the JSON
  integer `1` for `schema_version`, a Boolean `sigmf-zarr:valid`, and a
  string `sigmf-zarr:invalid-reason`.
- Forbid a complex sample dtype with an `iq` axis.
- Reject resource and array names that contain control characters or lone
  surrogates (Unicode categories `Cc` and `Cs`).
- Require collection recording identifiers to be unique.
- Require per-item `core:offset` values and capture `core:sample_start`
  values to be integers from 0 to 2**63 - 1.
- Report fsspec HTTP client failures, such as a truncated response, as
  `OSError`.

### Integrity, validation, and provenance

- Maintain logical sample, metadata, recording, collection, and store SHA-512
  hashes that do not depend on the physical Zarr encoding. Array hashes cover
  fixed logical slabs, so verification decodes each chunk once. The store hash
  covers root content and the digests that recordings and collections declare.
- Specify the canonical JSON encoding, array descriptors, and metadata
  payloads used by logical integrity hashes.
- Identify variable-length UTF-8 string arrays by the fixed hash descriptor
  `string` instead of the NumPy dtype string, which differs between NumPy
  releases, and identify fixed-width Unicode arrays with little-endian byte
  order. Hash zero-dimensional arrays with their declared dtype. Exclude only
  the hashed group's own `integrity` attribute from a metadata hash. These
  rules change hash values within integrity version 1. Call
  `update_integrity()` once on a store hashed by an earlier development build.
- Validate store structure, metadata, indexes, and declared hashes. Report
  Zarr format 2 CRC32C failures as validation issues.
- Report shared recording metadata that violates the SigMF core schema, a
  missing `sigmf-zarr:checksum` or other required field, a missing
  `extensions/` or `indexes/` group, an array under `recordings/` or
  `collections/`, a non-object `integrity` attribute, a non-string index
  `unit`, and symbolic links below the root of a local store.
- Report `MemoryError` during validation as an issue. The optional
  `max_array_bytes` argument of `validate_store()` and the
  `--max-array-bytes` option of `sigmf-zarr store validate` skip hash
  verification for arrays whose declared logical size exceeds the limit and
  report each such array.
- Record file and importer provenance, and bind derived outputs to logical
  samples, source metadata, and explicitly selected indexes.
- Exclude the global `core:sha512` field from the version-1 input-binding
  source metadata digest, so `update_sha512()` does not make bindings stale.
  Bindings captured by an earlier development build on a recording with
  `core:sha512` must be captured again.
- Compare input bindings by canonical JSON, so integers, floats, and Booleans
  are distinct, and reject malformed bindings in `verify_inputs()` and
  `add_split()`.
- Normalize `.` and `..` components in file manifest paths before removing
  duplicates, and reject paths outside the manifest root.
- Never read Zarr consolidated metadata. Every group, including nested
  Zarr format 3 groups, is read from its own metadata document, so reads,
  hashes, and writes cannot diverge from a consolidated copy. A store must
  not contain consolidated metadata, and validation reports each group that
  has it.
- Forbid structured and void array dtypes, whose field names array hashes do
  not cover.
- Report every array whose stored chunk shape is not a list of positive
  integers. Some zarr releases read a zero chunk length as one.
- Open recordings with structural validation by default. Per-item metadata
  entries are validated when read, by `validation="full"`, and by
  `validate_store()`.
- Before reading a local store, walk it without following links. Report
  every symbolic link, every entry that is neither a regular file nor a
  directory, and every directory that cannot be listed, and read no content
  when any is found. Stores opened through an fsspec local file system count
  as local.
- Make `max_array_bytes` also bound the decoded size of one chunk or shard
  and the per-item metadata text that validation checks, and report each
  skipped check.
- Report permission errors, overflowing shapes, damaged store-wide index
  metadata, deeply nested groups, and failed remote reads as issues instead
  of raising. Report each invalid recording index at its own path and
  continue with the remaining checks.
- Check collection metadata against the SigMF collection schema, apart from
  `core:streams`, and report repeated recording identifiers and names under
  `recordings/` or `collections/` that have keys but no node metadata.
- Validate JSON metadata in time linear in its size, and check channel
  groups without building one name per declared channel.

### SigMF interchange

- Import and export standard SigMF recordings, archives, and collections.
  Preserve and verify standard SigMF dataset hashes.
- Preserve exact dataset encodings, source recording and collection versions,
  and absolute sample coordinates based on `core:offset`.
- Export explicitly selected batch items with resolved capture scopes,
  independent timestamps, and per-item dataset hashes. Item export checks the
  recording metadata hash and relies on chunk CRC32C for the selected samples.
- Reject unsupported input metadata and archive files before destination
  creation. Reject sample axes that standard SigMF cannot represent.
- Check imported metadata against the SigMF schema that export applies, and
  reject `sigmf-zarr:*` fields in imported captures and annotations and
  datasets without a complete sample, so every imported recording can be
  exported.
- Accept only uncompressed tar files as `.sigmf` archives. Before extraction,
  reject sparse members, members other than regular files and directories,
  members outside the archive root, and members that declare more bytes than
  the archive holds.
- Name a recording imported through its `.sigmf-data` path by its SigMF base
  name, and reject an empty `recording_name` and export targets that are empty
  or end in `.` or `..`.
- Require explicit permission, and warn, before omitting native indexes,
  extension arrays, channel metadata, or recording nodes outside the defined
  layout during standard export.
- Check exported metadata against the SigMF schema before writing output.
- Write archive members with UID and GID 0, empty owner names, fixed modes, and
  modification time 0, so identical exports produce identical archives.
- Without `overwrite`, treat a dangling symbolic link as an existing export
  target, and never replace a target created while export runs.
- Plan exports without writing output, and project explicitly selected indexes
  into the `sigmf-zarr-indexes` extension. Planning raises `KeyError` for a
  missing recording, as export does.
- Move replaced recordings and collections to a backup during archive import
  instead of copying them.
- Rename public Python API parameters to match the other importers and
  exporters: `sigmf_path` of `import_sigmf()` to `source_path`,
  `archive_source` and `overwrite_recordings` of `import_sigmf_archive()` to
  `source_path` and `overwrite_recording`, and `archive_target` of
  `export_sigmf_archive()` to `output_path`. The old names are not accepted.
- Accept `sample_chunks` and `sample_shards` in `import_sigmf_archive()`. They
  apply to every recording in the archive, with the same meaning and defaults
  as in `import_sigmf()`.
- Read archive member headers with an offset and count bound, so a crafted
  archive cannot make `tarfile` loop (CVE-2025-8194). Count implied parent
  directories against the archive size. Report invalid member names and
  layouts as `ValueError`.
- Require `overwrite_collection=True` (`--overwrite-collection`) before
  archive import replaces a collection with the same name.
- Apply the sample-index checks of default export at import, so integral
  floating-point indexes and captures or annotations past the data are
  rejected. Reject a `core:dataset` file that is a symbolic link or not a
  regular file.
- Check collections against the SigMF collection schema at archive import
  and export, and reject a collection that names a recording more than
  once.
- Reject a complex dtype on an `iq` axis in export and export plans.
- Reject export targets that end in `/` or `/.`. Roll back export
  publication after an interrupt, and never delete a directory that appears
  at a target during export.
- Raise `KeyError` for a missing projected index before any other export
  check.

### Dataset importers

- Import RadioML 2016, RadioML 2018, RML22, Chad Spooner CSPB, and Panoradio HF
  datasets in bounded batches.
- Limit each RadioML 2018, CSPB, and Panoradio batch to about 64 MiB of
  samples as well as `batch_size` items. Reject an item, or a RadioML 2018
  HDF5 chunk, larger than 64 MiB before reading it.
- Read RadioML 2018 data only from the named HDF5 file. Reject datasets with
  external raw storage, virtual datasets, and paths that use external links.
- Record `radioml:source_dataset` from `import_radioml2018_dataset()` only when
  `source_dataset` is given, as the other RadioML importers do. The command
  line still defaults it to the source file name without its suffix.
- Require ASCII decimal numbers in Panoradio tags and in CSPB truth files and
  `.tim` file names, rejecting underscores and other Unicode digits.
- Provide shared `auto`, `none`, `zstd`, `lz4`, and `lz4hc` sample compression
  choices for every import command.
- Load RadioML pickles with a restricted opcode interpreter that never calls a
  function named by the file, accepts only plain integer and floating-point
  arrays, checks declared array sizes against the file size before
  allocating, and bounds opcode count and nesting depth. Reject pickle entries
  with no items. Check each CSPB ZIP member's declared size against its `.tim`
  header before reading it.
- Export `load_radioml2016_pickle()` from `sigmf_zarr`, and accept `str` paths
  as well as path-like objects.
- Rename the public Python API parameter `iq_chunks` of
  `import_radioml2016_dataset()` and `import_radioml2018_dataset()` to
  `sample_chunks`, as in the other importers. The old name is not accepted.
- Replace an existing target with `overwrite_store` only when it is an empty
  directory or a SigMF-Zarr store, and refuse a target that equals or
  contains an input. Check sample chunk and shard options before replacing
  anything. A failed import removes only a directory that it created,
  identified by device and inode.
- Accept only `(str, int)` pickle keys with a label of at most 256
  characters, an `int16` SNR other than a Boolean, and total label text no
  longer than the file. Limit text integers and memo indexes to the ranges
  of the binary pickle opcodes and dictionaries to 65,536 entries.
- Combine RadioML 2016 and RML22 buckets into writes of up to `batch_size`
  items, and reject mappings with more than 4,096 buckets.
- Accept only stored and deflate CSPB ZIP members, and inflate a member at
  most one byte past the size that its `.tim` header declares. Reject
  symbolic links in CSPB source directories. Require CSPB truth integers to
  fit `int64`, and at most 32,768 modulation labels.
- Allow only the gzip, shuffle, and fletcher32 HDF5 filters in RadioML 2018
  files, inflate gzip chunks with a bound, verify Fletcher-32 checksums, and
  disable HDF5 plugin loading during import. Require scalar numeric label
  and SNR elements, and at most 32,768 classes.

### Indexes

- Discover indexes by descriptive field, and decode selected category IDs
  explicitly without synchronizing independent JSON metadata.
- Define the reserved index attribute `validity`, which declares missing
  values, and reject indexes that declare it wherever dense values are
  required.
- Accept additional descriptive index attributes with protected base fields,
  and require nonempty `field` and `kind` descriptors.

### Command line

- Provide the `sigmf-zarr` command for store inspection, validation, integrity
  updates, SigMF interchange, and dataset import.

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
- Use default sample chunks of about 1 MiB along the growth axis.
- Append distinct per-item capture lists with independent timestamps through
  `item_captures`, preserving shared item-offset captures and validating all
  metadata before storage changes.
- Derive per-item metadata presence from the array, protect its chunks with
  CRC32C, and provide validated entry and slice updates.
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
- Check exported metadata against the SigMF schema before writing output.

### Dataset importers

- Import RadioML 2016 and RadioML 2018 datasets in bounded batches.
- Restrict RadioML pickle loading to built-in containers and real numeric
  NumPy arrays, and reject arrays that hold more data than the file.

### Command line

- Provide the `sigmf-zarr` command for store inspection, validation,
  integrity updates, and SigMF interchange.

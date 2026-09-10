# Changelog

All notable changes to this project will be documented in this file. The
project follows Semantic Versioning for public releases and PEP 440 for Python
package versions.

## 1.0.0a1 - unreleased

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
- Reject direct assignment of `sigmf-zarr` storage fields through
  `set_global_field()`, and maintain the canonical storage extension
  declaration in `core:extensions`.
- Roll back failed creations and replacements of recordings, collections,
  indexes, extension arrays, and item metadata, and failed appends and sample
  replacements. Retain recovery files if restoration fails.
- Provide `SignalView` for bounded time reads and resolved metadata of selected
  items and continuous recordings.

### Integrity, validation, and provenance

- Maintain logical sample, metadata, recording, collection, and store SHA-512
  hashes that do not depend on the physical Zarr encoding. Array hashes cover
  fixed logical slabs, so verification decodes each chunk once. The store hash
  covers root content and the digests that recordings and collections declare.
- Specify the canonical JSON encoding used by logical integrity hashes.
- Validate store structure, metadata, indexes, and declared hashes. Report
  Zarr format 2 CRC32C failures as validation issues.
- Record file and importer provenance, and bind derived outputs to logical
  samples, source metadata, and explicitly selected indexes.

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
- Require explicit permission, and warn, before omitting native indexes,
  extension arrays, or channel metadata during standard export.
- Check exported metadata against the SigMF schema before writing output.
- Plan exports without writing output, and project explicitly selected indexes
  into the `sigmf-zarr-indexes` extension.
- Move replaced recordings and collections to a backup during archive import
  instead of copying them.

### Dataset importers

- Import RadioML 2016, RadioML 2018, RML22, Chad Spooner CSPB, and Panoradio HF
  datasets in bounded batches.
- Provide shared `auto`, `none`, `zstd`, `lz4`, and `lz4hc` sample compression
  choices for every import command.
- Restrict RadioML pickle loading to built-in containers and real numeric
  NumPy arrays, and reject arrays that hold more data than the file. Check each
  CSPB ZIP member's declared size against its `.tim` header before reading it.

### Indexes and splits

- Discover indexes by descriptive field, and decode selected category IDs
  explicitly without synchronizing independent JSON metadata.
- Accept additional descriptive index attributes with protected base fields,
  and require nonempty `field` and `kind` descriptors.
- Store multiple named split schemes with compact assignments, read-only
  views, provenance, and explicit group-isolation validation, including
  isolation of reused constituent sources.

### Machine learning

- Add an optional dense PyTorch dataset with explicit targets and split
  selection, dtype-preserving CPU tensors, transforms, batched reads, and
  process-local read-only handles.
- Record dataset manifests with target descriptors, item order, transform
  descriptions, software versions, and input hashes.
- Define the draft `rfml-dataset` dense metadata profile.
- Add a RadioML VT-CNN2 training example with seeded modulation/SNR-stratified
  splits, CPU target transforms, spawn workers, and validation accuracy by SNR.

### Command line

- Provide the `sigmf-zarr` command for store inspection, validation, integrity
  updates, SigMF interchange, and dataset import.

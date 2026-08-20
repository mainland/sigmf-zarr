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
- Use default sample chunks of about 1 MiB along the growth axis.
- Append distinct per-item capture lists with independent timestamps through
  `item_captures`, preserving shared item-offset captures and validating all
  metadata before storage changes.
- Derive per-item metadata presence from the array, protect its chunks with
  CRC32C, and provide validated entry and slice updates.
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
- Check exported metadata against the SigMF schema before writing output.

### Dataset importers

- Import RadioML 2016 and RadioML 2018 datasets in bounded batches.
- Restrict RadioML pickle loading to built-in containers and real numeric
  NumPy arrays, and reject arrays that hold more data than the file.

### Command line

- Provide the `sigmf-zarr` command for store inspection, validation,
  integrity updates, and SigMF interchange.

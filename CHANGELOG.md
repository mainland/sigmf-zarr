# Changelog

All notable changes to this project will be documented in this file. The
project follows Semantic Versioning for public releases and PEP 440 for Python
package versions.

## 1.0.0a1 - unreleased

This is the first public alpha of the initial SigMF-Zarr format draft.

- Append distinct per-item capture lists with independent timestamps through
  `item_captures`, preserving shared item-offset captures and validating all
  metadata before storage changes.
- Define the initial logical store, recording, collection, index, channel, and
  item-metadata model.
- Support physical Zarr formats 2 and 3 through zarr-python 3.
- Use approximately 256 KiB logical chunks and 4 MiB physical shards for
  batched format-3 imports. Use approximately 1 MiB chunks for unbatched
  format-3 imports and 4 MiB chunks for format-2 imports.
- Provide shared `auto`, `none`, `zstd`, `lz4`, and `lz4hc` sample compression
  choices for every import command.
- Import and export standard SigMF recordings and archives.
- Preserve source recording and collection versions during interchange, and
  validate absolute sample coordinates using `core:offset`.
- Export explicitly selected batch items with resolved capture scopes,
  independent timestamps, and per-item dataset hashes.
- Reject unsupported input metadata and archive files before destination
  creation. Require explicit permission and warn before omitting native indexes,
  extension arrays, or channel metadata during standard export.
- Import RadioML 2016, RadioML 2018, RML22, and Chad Spooner CSPB datasets.
- Import Panoradio HF complex NumPy arrays and aligned CSV tags with bounded
  sample conversion and typed per-item indexes.
- Share bounded sample writes between the RadioML 2016 and RML22 pickle
  importers while retaining dataset-specific subcommands and metadata.
- Preserve and verify standard SigMF dataset hashes.
- Maintain logical sample, metadata, recording, collection, and store hashes.
- Restore existing recordings and collections when replacement creation or
  validation fails, retaining recovery files if restoration fails.
- Roll back failed index, extension-array, and item-metadata replacements,
  including descriptor writes and codec failures.
- Derive per-item metadata presence from the array, protect its chunks with
  CRC32C, and provide efficient validated entry and slice updates.
- Discover indexes by descriptive field and explicitly decode selected
  category IDs without synchronizing independent JSON metadata.
- Accept additional descriptive index attributes with protected base fields.
- Require nonempty `field` and `kind` descriptors when creating, reading,
  validating, or completing a mutation of a recording index.
- Support structural recording opens that defer per-item JSON validation
  until access while retaining full validation by default.
- Store multiple named split schemes with compact assignments, read-only
  views, provenance, and explicit scalar group-isolation validation.

- Add an optional dense PyTorch dataset with explicit targets and split
  selection, dtype-preserving CPU tensors, transforms, batched reads, and
  process-local read-only handles.

- Add a RadioML VT-CNN2 training example with seeded modulation/SNR-stratified
  splits, CPU target transforms, spawn workers, and validation accuracy by SNR.

- Define the draft `rfml-dataset` dense metadata profile and document
  source-field adoption decisions without changing existing importer output.

The format and Python API may change incompatibly during the alpha series.
Data written by one alpha may require migration before a later alpha can read
it. Stable compatibility begins only when the initial format draft is declared
complete.

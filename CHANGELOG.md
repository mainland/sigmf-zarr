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
- Import and export standard SigMF recordings and archives.
- Preserve and verify standard SigMF dataset hashes.
- Maintain logical sample, metadata, recording, collection, and store hashes.
- Restore existing recordings and collections when replacement creation or
  validation fails, retaining recovery files if restoration fails.
- Roll back failed index, extension-array, and item-metadata replacements,
  including descriptor writes and codec failures.
- Derive per-item metadata presence from the array, protect its chunks with
  CRC32C, and provide efficient validated entry and slice updates.

The format and Python API may change incompatibly during the alpha series.
Data written by one alpha may require migration before a later alpha can read
it. Stable compatibility begins only when the initial format draft is declared
complete.

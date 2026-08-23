# Changelog

All notable changes to this project will be documented in this file. The
project follows Semantic Versioning for public releases and PEP 440 for Python
package versions.

## 1.0.0a1 - unreleased

This is the first public alpha of the initial SigMF-Zarr format draft.

- Define the initial logical store, recording, collection, index, channel, and
  item-metadata model.
- Support physical Zarr formats 2 and 3 through zarr-python 3.
- Use approximately 256 KiB logical chunks and 4 MiB physical shards for
  batched format-3 imports. Use approximately 1 MiB chunks for unbatched
  format-3 imports and 4 MiB chunks for format-2 imports.
- Import and export standard SigMF recordings and archives.
- Import RadioML 2016, RadioML 2018, and Chad Spooner CSPB datasets.
- Preserve and verify standard SigMF dataset hashes.
- Maintain logical sample, metadata, recording, collection, and store hashes.
- Derive per-item metadata presence from the array, protect its chunks with
  CRC32C, and provide efficient validated entry and slice updates.

The format and Python API may change incompatibly during the alpha series.
Data written by one alpha may require migration before a later alpha can read
it. Stable compatibility begins only when the initial format draft is declared
complete.

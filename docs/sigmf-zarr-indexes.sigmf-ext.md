# SigMF-Zarr index projection extension

Namespace: `sigmf-zarr-indexes`

Version: `0.1.0`

This project-defined extension preserves explicitly selected item indexes in
the standard SigMF recording exported for that item. It adds no dataset bytes
and is optional for sample interpretation. It does not map dataset-native
labels to canonical SigMF signal classes or certify measurement definitions.

## Global fields

`sigmf-zarr-indexes:values` is an object keyed by the original index path. Each
entry contains:

- A `value` containing the original JSON-compatible scalar at the selected item
- An `attributes` object containing the original index descriptors
- A `label` containing the decoded category string when `attributes.labels`
  supplies a category lookup

The numeric category ID remains in `value`. Its interpretation remains scoped
to the preserved lookup table. Units, measurement descriptors, field names, and
split provenance remain in `attributes`. Readers must not interpret a source
field as a canonical measurement merely because it appears in this extension.

Writers must declare this namespace in `core:extensions` with version `0.1.0`
and `optional: true`. Projection must reject conflicting existing values or
namespace declarations. Only dense item indexes with JSON-compatible values
are supported. A validity mask requires a separately defined projection and
must not be silently discarded.

Standard SigMF import preserves this JSON without reconstructing native Zarr
indexes. The exported signal represents one selected item. Other items and
their index values are outside its scope.

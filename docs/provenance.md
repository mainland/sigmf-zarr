# Provenance and input bindings

Provenance identifies inputs and describes processing. Logical integrity hashes
check stored content. Neither establishes that a label is physically correct.
Keep inputs unchanged while hashing, importing, or using a derived result.

## Import records

The project-defined `sigmf-zarr-provenance` extension has version `0.1.0` and is
optional for sample interpretation. Writers must declare it in
`core:extensions` with `optional: true`.

The global `sigmf-zarr-provenance:import` object contains:

- `sources`: An ordered array of immutable input identities
- `operation`: The importer function name
- `software`: An object containing the package `name` and `version`
- `parameters`: A JSON object describing ordering, class mappings, and conversions

A file identity has `kind="file"`, a portable `name`, a byte count in `bytes`,
a `sha512` digest over the original bytes, and a `role`. File names are locators.
The digest and byte count identify content. Separate truth files must have their
own identities. Callers must not supply an existing import record to an importer
that generates one from its inputs.

A version-1 file manifest has `kind="file-manifest"`, `version=1`, `count`,
`sha512`, and `role`. Its digest hashes the canonical JSON array of individual
file identities, sorted by relative path components. Each entry's name is its
POSIX path relative to the selected root. Input paths and the root are
normalized lexically before duplicates are removed, so `sub/../a.bin` and
`./a.bin` both name `a.bin`. `file_manifest_identity()` raises `ValueError` for
a path that does not lie below the root after normalization. Normalization does
not resolve symbolic links.
Reconstruct the manifest from the source tree to verify it. Hashing ZIP
containers identifies the downloaded bytes, including archive packaging.

In-memory array identities use the logical array hash defined by the
[format specification](format.md). Source labels and output
ranges accompany those identities. This identifies decoded input values rather
than the bytes of an earlier serialization.

## Bind derived metadata to its inputs

`capture_inputs(recording, indexes=[...])` returns a version-1 object containing
`sample_sha512`, `source_metadata_sha512`, and an `indexes` object. Each selected
index entry includes its logical `sha512` and complete `attributes`, including
lookup tables and measurement descriptors. No index is selected implicitly.

The source metadata digest includes recording metadata, per-item JSON, sample
descriptors, channel metadata, and extension data. It excludes sample values,
the entire recording-level `indexes` group, and the global `core:sha512` field.
Sample values have a separate digest. `core:sha512` is a digest of the sample
values, so `update_sha512()` leaves a binding current. Selected input indexes
are bound explicitly so an output index can store this binding without
referring to itself.

Store the binding on a derived index or in an external JSON manifest.
`verify_inputs(recording, binding)` recomputes the hashes and compares all
bound inputs by their canonical JSON encoding. Integers, floats, and Booleans
are therefore distinct. Index labels changed from `[0, 1]` to `[false, true]`,
or an attribute changed from `1` to `1.0`, make the binding stale. It returns
false after an input changes or a selected index disappears. It also returns
false when a selected index is invalid, for example after an item append marks
it invalid. It raises `ValueError` for an unsupported or malformed binding. A
well-formed binding has exactly the fields `version`, `sample_sha512`,
`source_metadata_sha512`, and `indexes`. Each digest is 128 lowercase
hexadecimal characters, and each index entry has exactly `sha512` and an
`attributes` object. `add_split()` applies the same check to `input_binding`.
It does not refresh, invalidate, or delete derived values. Adding an unrelated
index leaves the binding unchanged. Both operations scan data and should be
requested explicitly rather than run during ordinary index reads.

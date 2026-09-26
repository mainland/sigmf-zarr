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

Some imports hold or extract a whole input before writing it:

- The CSPB importer decompresses each `.tim` ZIP member in memory. It first
  requires the member's declared size to equal the size its `.tim` header
  declares, but a header may declare up to 16 GiB.
- SigMF archive import extracts the complete archive into the system temporary
  directory, which must have space for the extracted files.

# Security policy

## Supported versions

Only the newest published alpha is supported during initial format development.

## Reporting a vulnerability

Report security problems privately to `mainland@drexel.edu`. Do not include
sensitive recordings or credentials in a public issue. Include the affected
version, input format, impact, and a minimal reproducer when possible.

## Untrusted inputs

RadioML 2016 and RML22 files are Python pickle files. Loading a pickle can
execute code, so the `sigmf-zarr import radioml2016` and
`sigmf-zarr import rml22` commands must only be used with files from a trusted
source.

SigMF archives, HDF5 files, ZIP files, `.tim` files, JSON metadata, and Zarr
stores should also be treated as untrusted data. Run imports with operating
system permissions appropriate for the output location and apply external
resource limits when processing unknown or unusually large datasets.

# Security policy

## Supported versions

Only the newest published alpha is supported during initial format development.

## Reporting a vulnerability

Report security problems privately to `mainland@drexel.edu`. Do not include
sensitive recordings or credentials in a public issue. Include the affected
version, input format, impact, and a minimal reproducer when possible.

## Untrusted inputs

SigMF archives, JSON metadata, and Zarr stores should be treated as untrusted
data. Run imports with operating system permissions
appropriate for the output location and apply external resource limits when
processing unknown or unusually large datasets.

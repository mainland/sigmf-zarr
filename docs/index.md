# SigMF-Zarr

SigMF-Zarr is a Python package for storing SigMF-style signal recordings in
Zarr. It keeps the core SigMF metadata model while using chunked, typed Zarr
arrays for sample data and related recording metadata.

The package is organized around `sigmf_zarr.store.SigMFZarrStore`, with helpers
for importing and exporting SigMF datasets.

```{toctree}
:maxdepth: 2
:caption: Contents

usage
comparison
format
sigmf-zarr.sigmf-ext
development
api
```

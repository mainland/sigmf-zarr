# sigmf-zarr

SigMF-Zarr is a Python package for storing SigMF-style signal recordings in
Zarr. It keeps SigMF metadata close to the standard model while using chunked,
typed Zarr arrays for sample data and related recording metadata.

## Documentation

- [Usage](docs/usage.md)
- [RadioML](docs/radioml.md)
- [Format model](docs/format.md)
- [Development](docs/development.md)
- [API reference](docs/api.md)

## Quick start

Install the current release from PyPI:

```bash
python -m pip install sigmf-zarr
sigmf-zarr --version
```

For development from a checkout:

```bash
uv sync --extra dev --extra docs
uv run pytest
```

Build the documentation locally:

```bash
uv run --extra docs sphinx-build -b html -W docs docs/_build/html
```

## CLI

The package installs the `sigmf-zarr` command for inspecting stores and
importing or exporting supported data:

```bash
sigmf-zarr store store.zarr info
sigmf-zarr store store.zarr recordings --format json
sigmf-zarr store store.zarr validate
sigmf-zarr store store.zarr integrity verify
sigmf-zarr import sigmf input.sigmf-meta store.zarr
sigmf-zarr store store.zarr recording myrec export output.sigmf-meta
```

New stores use Zarr format 3 by default. Import commands auto-detect the format
of existing stores and accept `--zarr-format 2` for tools that require the
Zarr format 2 physical layout. Reading either format uses the same zarr-python
3.2-or-newer runtime.

## Alpha status

Version 1.0.0a1 is an alpha of the initial format draft. The storage format and
Python API may change incompatibly until the initial draft is complete. See the
[changelog](CHANGELOG.md) and the normative
[`sigmf-zarr` extension](docs/sigmf-zarr.sigmf-ext.md).

RadioML 2016 uses Python pickle input. Only import pickle files obtained from a
trusted source because loading a malicious pickle can execute code.

## Project information

- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [MIT license](LICENSE)

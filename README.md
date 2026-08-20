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

Install a published release from PyPI:

```bash
python -m pip install sigmf-zarr
sigmf-zarr --version
```

For development from a checkout, create the environment and run the tests:

```bash
uv sync
.venv/bin/python -m pytest
```

Build the documentation locally:

```bash
.venv/bin/python -m sphinx -W --keep-going -b html docs docs/_build/html
```

## Command line

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

Version 1.0.0a2 is an alpha of the initial format draft. The storage format and
Python API may change incompatibly until the initial draft is complete. See the
[changelog](https://github.com/mainland/sigmf-zarr/blob/main/CHANGELOG.md) and the normative
[`sigmf-zarr` extension](https://github.com/mainland/sigmf-zarr/blob/main/docs/sigmf-zarr.sigmf-ext.md).

RadioML 2016 uses Python pickle input. Only import pickle files obtained from a
trusted source because loading a malicious pickle can execute code.

## Project information

- [Contributing](https://github.com/mainland/sigmf-zarr/blob/main/CONTRIBUTING.md)
- [Security policy](https://github.com/mainland/sigmf-zarr/blob/main/SECURITY.md)
- [MIT license](https://github.com/mainland/sigmf-zarr/blob/main/LICENSE)

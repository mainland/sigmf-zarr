# sigmf-zarr

SigMF-Zarr is a Python package for storing SigMF-style signal recordings in
Zarr. It keeps SigMF metadata close to the standard model while using chunked,
typed Zarr arrays for sample data and related recording metadata.

## Documentation

The documentation is published at <https://sigmf-zarr.readthedocs.io/>. Its
pages are also available in the repository:

- [Usage](https://github.com/mainland/sigmf-zarr/blob/main/docs/usage.md)
- [Relationship to standard SigMF](https://github.com/mainland/sigmf-zarr/blob/main/docs/comparison.md)
- [RadioML](https://github.com/mainland/sigmf-zarr/blob/main/docs/radioml.md)
- [Chad Spooner CSPB datasets](https://github.com/mainland/sigmf-zarr/blob/main/docs/cspb.md)
- [Panoradio HF dataset](https://github.com/mainland/sigmf-zarr/blob/main/docs/panoradio.md)
- [PyTorch datasets](https://github.com/mainland/sigmf-zarr/blob/main/docs/pytorch.md)
- [Index queries](https://github.com/mainland/sigmf-zarr/blob/main/docs/query.md)
- [Dataset viewer](https://github.com/mainland/sigmf-zarr/blob/main/docs/viewer.md)
- [RFML dataset profile](https://github.com/mainland/sigmf-zarr/blob/main/docs/rfml.md)
- [Format model](https://github.com/mainland/sigmf-zarr/blob/main/docs/format.md)
- [Provenance and input bindings](https://github.com/mainland/sigmf-zarr/blob/main/docs/provenance.md)
- [The `sigmf-zarr` SigMF extension](https://github.com/mainland/sigmf-zarr/blob/main/docs/sigmf-zarr.sigmf-ext.md)
- [SigMF-Zarr index projection extension](https://github.com/mainland/sigmf-zarr/blob/main/docs/sigmf-zarr-indexes.sigmf-ext.md)
- [Development](https://github.com/mainland/sigmf-zarr/blob/main/docs/development.md)
- [API reference](https://github.com/mainland/sigmf-zarr/blob/main/docs/api.md)

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

Subcommands under `sigmf-zarr import` import RadioML 2016, RadioML 2018.01A,
RML22, CSPB, and Panoradio HF datasets. Run `sigmf-zarr import --help` to list
the supported formats. For example, import an RML22 pickle mapping with:

```bash
sigmf-zarr import rml22 RML22.01A.pkl store.zarr
```

See the [RadioML documentation](https://github.com/mainland/sigmf-zarr/blob/main/docs/radioml.md) for supported input layouts,
storage options, and Python examples.

Import a Panoradio HF NumPy file with its CSV tags:

```bash
sigmf-zarr import panoradio dataset_panoradio_hf.npy panoradio.zarr \
  --tags-file dataset_panoradio_hf_tags.csv
```

See the [Panoradio documentation](https://github.com/mainland/sigmf-zarr/blob/main/docs/panoradio.md) for the sample layout,
tag indexes, and Python API.

## Alpha status

Version 1.0.0a2 is an alpha of the initial format draft. The storage format and
Python API may change incompatibly until the initial draft is complete. See the
[changelog](https://github.com/mainland/sigmf-zarr/blob/main/CHANGELOG.md) and the normative
[`sigmf-zarr` extension](https://github.com/mainland/sigmf-zarr/blob/main/docs/sigmf-zarr.sigmf-ext.md).

RadioML 2016 and RML22 use Python pickle input. The importers restrict
unpickling to built-in containers and real numeric NumPy arrays, reject any
other global, and reject arrays that hold more data than the file. Use
`load_radioml2016_pickle()` rather than `pickle.load` to decode these files in
Python.

## Project information

- [Contributing](https://github.com/mainland/sigmf-zarr/blob/main/CONTRIBUTING.md)
- [Security policy](https://github.com/mainland/sigmf-zarr/blob/main/SECURITY.md)
- [MIT license](https://github.com/mainland/sigmf-zarr/blob/main/LICENSE)

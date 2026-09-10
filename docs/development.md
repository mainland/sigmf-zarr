# Development

This project uses `uv` for local environment management and dependency locking.
The `.python-version` file pins the local interpreter request to Python 3.12,
and the resolved dependency set is recorded in `uv.lock`.

## Layout

- Package source: `src/sigmf_zarr`
- Test suite: `tests`
- Sphinx documentation: `docs`
- Design and planning notes: `notes`

Reusable RadioML conversion functions live in `sigmf_zarr.radioml2016`
and `sigmf_zarr.radioml2018`. The `sigmf-zarr import rml22` command calls
`import_radioml2016_dataset()` with `dataset_version="2022"` to reuse its
pickle format support.
Command-line modules parse arguments and call the public import functions.
Importers share the recording rollback boundary in
`sigmf_zarr.store._transaction`. Schema validation and read-only views remain
independent of the command-line interface.

## Environment

Create or update the local environment with development and documentation
dependencies:

```bash
uv sync --extra dev --extra docs
```

For adapter development, include PyTorch:

```bash
uv sync --extra dev --extra docs --extra pytorch
uv run --extra pytorch pytest
```

On Linux, the default PyPI build includes CUDA support. For CPU-only testing,
install from the PyTorch CPU index after syncing the core environment, then
invoke tools directly so another sync does not replace that installation:

```bash
uv pip install --python .venv/bin/python 'torch>=2.6' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pytest
.venv/bin/mypy src
```

Core-only test environments skip the adapter tests. The PyTorch CI job runs
the full suite with the CPU dependency installed.

## Checks

Run project checks through the locked environment:

```bash
uv run pytest
uv run --extra pytorch mypy src
uv run ruff check src tests docs benchmarks
uv run docformatter --check --recursive src tests benchmarks
uv build
uv run twine check dist/*
uv run tox
```

## Pre-commit checks

Install the pre-commit hooks after creating the development environment:

```bash
uv run pre-commit install
```

The hooks check changed files for common repository errors and run Ruff and
docformatter. They also run mypy when source code or its configuration changes.
Run every hook against the complete repository with:

```bash
uv run pre-commit run --all-files
```

## Documentation

Build the HTML documentation with Sphinx:

```bash
uv run --extra docs sphinx-build -b html -W docs docs/_build/html
```

The tox documentation environment runs the same warning-as-error Sphinx build:

```bash
uv run tox -e doc
```

The test suite treats deprecation warnings as errors and enforces branch-aware
coverage. Continuous integration runs the supported Python versions, builds the
wheel and source distribution, installs the wheel into a clean environment,
and smoke-tests the `sigmf-zarr` command and its subcommands.

Performance-sensitive tests assert deterministic I/O complexity instead of
wall-clock limits, which vary across CI hosts. In particular, item-metadata
validation must perform one array read per Zarr chunk, and validated entry and
slice setters must not scan the complete metadata array.

## Release process

1. Make every locked and clean-environment check pass.
2. Update `CHANGELOG.md` and remove `Unreleased` from the release heading.
3. Commit the complete source tree and create an annotated PEP 440-compatible
   tag, such as `v1.0.0a1`.
4. Build and test the artifacts from that exact tag.
5. Publish to TestPyPI and smoke-test installation.
6. Publish the unchanged artifacts to PyPI using Trusted Publishing.

Publishing and tagging are deliberate maintainer actions. Ordinary test and
build commands never publish a package.

## Dense loading benchmark

Run the reproducible local and loopback HTTP comparison:

```bash
.venv/bin/python -m benchmarks.pytorch_loading
```

Install the development and PyTorch extras first. The benchmark compares
scalar reads with `__getitems__()` for 256-item sequential and random requests,
using 8,192 float32 items with shape `(2, 128)`. Zarr format 2 uses 256-item
chunks. Zarr format 3 uses the same chunks within 4,096-item shards. It reports
constructor time, Python allocation peak during construction, and median read
times over three warm-cache trials. Allocation tracking excludes native
allocations. HTTP runs use a local server and do not model WAN latency or S3.

One Python 3.12 run with PyTorch 2.14 measured batched reads at 16.5-35.8 times
scalar throughput for local storage and 24.9-80.1 times for loopback HTTP.
These measurements support using orthogonal batch selection. They are not
performance guarantees or timing assertions in the test suite.

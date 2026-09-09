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

Editable installs load source changes directly, but their installed version
metadata is captured when the package is built. After changing commits or
rewriting history, refresh that metadata before recording software versions
in derived metadata:

```bash
uv pip install --python .venv/bin/python --no-deps \
  --reinstall-package sigmf-zarr --editable .
.venv/bin/sigmf-zarr --version
```

This command rebuilds the local package without changing installed dependency
versions.

## Checks

Run project checks through the locked environment:

```bash
uv run pytest
uv run mypy src
uv run ruff check src tests docs
uv run docformatter --check --recursive src tests
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

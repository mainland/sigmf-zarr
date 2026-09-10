# Development

This project uses `uv` for local environment management and dependency locking.
The `.python-version` file pins the local interpreter request to Python 3.12,
and the resolved dependency set is recorded in `uv.lock`.

## Layout

- Package source: `src/sigmf_zarr`
- Test suite: `tests`
- Sphinx documentation: `docs`

Reusable RadioML conversion functions live in `sigmf_zarr.radioml2016`
and `sigmf_zarr.radioml2018`. The `sigmf-zarr import rml22` command calls
`import_radioml2016_dataset()` with `dataset_version="2022"` to reuse its
pickle format support.
Command-line modules parse arguments and call the public import functions.
Importers share the recording rollback boundary in
`sigmf_zarr.store._transaction`. Schema validation and read-only views remain
independent of the command-line interface.

## Environment

Create or update the development environment:

```bash
uv sync
```

`uv sync` creates `.venv` from `uv.lock` and installs the package in editable
mode with the `dev` dependency group. That group includes the `test` and `docs`
groups, every optional extra, and the checkers at the versions that the lock
pins. Run tools from the environment as `.venv/bin/python -m <tool>`, which
works however the environment was built.

Where uv is unavailable, pip 25.1 or later can build an equivalent environment.
This path ignores `uv.lock`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e . --group dev
```

The locked environment installs CPU PyTorch from the PyTorch CPU index, which
`[tool.uv.sources]` in `pyproject.toml` selects. The setting affects only uv's
development environment. Package installations and the pip path resolve `torch`
from PyPI, whose Linux wheels include CUDA support.

Editable installs load source changes directly, but their installed version
metadata is captured when the package is built. After changing commits or
rewriting history, refresh that metadata before recording software versions in
derived metadata:

```bash
uv pip install --python .venv/bin/python --no-deps \
  --reinstall-package sigmf-zarr --editable .
.venv/bin/sigmf-zarr --version
```

This command rebuilds the local package without changing installed dependency
versions.

## Checks

Install the pre-commit hooks once per clone:

```bash
.venv/bin/pre-commit install
```

The hooks run Ruff lint and format checks, strict mypy over `src`, `tests`,
`examples`, `benchmarks`, and `docs/conf.py`, sp-repo-review, codespell, rumdl,
zizmor, and basic file checks. The Ruff, mypy, and sp-repo-review hooks run
from `.venv`, so they use the versions that the lock pins. The hooks check
files and never rewrite them. Format Python sources with Ruff:

```bash
.venv/bin/python -m ruff format
```

Run every hook against the complete repository:

```bash
.venv/bin/pre-commit run --all-files
```

Run the test suite with branch coverage:

```bash
.venv/bin/python -m pytest --cov
```

The test suite treats every warning as an error. A coverage run fails below 80
percent. Environments without the optional integrations skip the PyTorch tests,
so measure coverage only with every extra installed.

Tox runs the tests without the lock, so a runtime dependency missing from
`pyproject.toml` fails. Its environments are:

- `3.12`, `3.13`, and `3.14`: The core tests on each supported Python version.
- `full`: The tests with every optional integration and coverage.
- `min`: The tests with the lowest supported direct dependency versions.
- `lint`: Every pre-commit hook. It requires `.venv`.
- `docs`: The documentation build.
- `linkcheck`: The external link check. It needs the network and is not part of
  CI.

Run one environment with `.venv/bin/tox -e <name>`.

CI runs every tox environment except `linkcheck`. It also checks that `uv.lock`
matches `pyproject.toml`, audits the locked dependencies with pip-audit, builds
the wheel and source distribution, installs the wheel into a clean environment,
and smoke-tests the `sigmf-zarr` command and its subcommands.

Performance-sensitive tests assert deterministic I/O complexity instead of
wall-clock limits, which vary across CI hosts. In particular, item-metadata
validation must perform one array read per Zarr chunk, and validated entry and
slice setters must not scan the complete metadata array.

## Documentation

Build the HTML documentation with Sphinx:

```bash
.venv/bin/python -m sphinx -W --keep-going -b html docs docs/_build/html
```

The build fails on any warning. Nitpicky mode makes every unresolved
cross-reference a warning. `docs/conf.py` lists the references that cannot
resolve, grouped by cause.

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

The development environment includes PyTorch. The benchmark compares scalar
reads with `__getitems__()` for 256-item sequential and random requests, using
8,192 float32 items with shape `(2, 128)`. Zarr format 2 uses 256-item chunks.
Zarr format 3 uses the same chunks within 4,096-item shards. It reports
constructor time, Python allocation peak during construction, and median read
times over three warm-cache trials. Allocation tracking excludes native
allocations. HTTP runs use a local server and do not model WAN latency or S3.

One Python 3.12 run with PyTorch 2.14 measured batched reads at 16.5-35.8 times
scalar throughput for local storage and 24.9-80.1 times for loopback HTTP.
These measurements support using orthogonal batch selection. They are not
performance guarantees or timing assertions in the test suite.

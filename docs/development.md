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
`sigmf_zarr.store._transaction`. The `remove_new_store_on_failure` decorator in
`sigmf_zarr._import_target` removes a target store that a failed import
created. Schema validation and read-only views remain independent of the
command-line interface.

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
`examples`, and `docs/conf.py`, sp-repo-review, codespell, rumdl, zizmor, and
basic file checks. The Ruff, mypy, and sp-repo-review hooks run from `.venv`,
so they use the versions that the lock pins. The hooks check files and never
rewrite them. Format Python sources with Ruff:

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
- `full`: The tests with every optional integration and coverage, on the
  Python version that runs tox.
- `min`: The tests with the lowest supported direct dependency versions.
- `lint`: Every pre-commit hook. It requires `.venv`.
- `docs`: The documentation build.
- `linkcheck`: The external link check. It needs the network and is not part of
  CI.

Run one environment with `.venv/bin/tox -e <name>`.

CI runs every tox environment except `linkcheck`. It runs `full` on Python
3.12, 3.13, and 3.14 by starting tox under each version. CI also checks that
`uv.lock` matches `pyproject.toml`, builds the wheel and source distribution,
and installs the wheel into a clean environment. The smoke test in that
environment runs `--help` for every importer, creates a store, and runs
`store` integrity update, validation, recording information, `export`, import
of the exported recording, and integrity verification.

A separate audit workflow checks the locked dependencies with pip-audit on
every push and pull request and once a week. The lock pins CPU PyTorch with a
local version label, such as `2.15.0+cpu`, which is not on PyPI. The workflow
removes the label so that pip-audit checks the matching PyPI release instead of
skipping it.

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

setuptools-scm derives the package version from the most recent tag that is
an ancestor of the built commit. The release tag must therefore be on the
release commit itself.

1. Update `CHANGELOG.md` and remove `Unreleased` from the release heading.
2. Commit the complete source tree, push it, and make every CI check pass on
   that commit.
3. Create an annotated PEP 440-compatible tag, such as `v1.0.0a2`, on the
   release commit and push it. The tag must be `v` followed by the version
   in the release heading of `CHANGELOG.md`.
4. Publish a GitHub release from the tag.
5. Install the published release from PyPI into a clean environment and run
   `sigmf-zarr --version`.

Publishing the GitHub release starts the `Publish release` workflow. It
checks out the release tag, builds the wheel and source distribution with a
pinned uv version, checks them with a pinned twine version, and uploads them
to PyPI with Trusted Publishing from the `pypi` environment. The workflow
builds the artifacts again from the tag. It does not run the tests or upload
the CI artifacts, which is why CI must pass on the tagged commit first.
Before it publishes, the workflow fails unless the release tag is `v`
followed by the built version and a CI run succeeded on the tagged commit.
Running the workflow manually builds and checks the artifacts without
publishing them. The project does not publish to TestPyPI.

Publishing and tagging are deliberate maintainer actions. Ordinary test and
build commands never publish a package.

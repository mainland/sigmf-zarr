# Contributing

Contributions are welcome through the repository issue tracker and pull request
workflow.

Use Python 3.12 or newer and the repository `uv.lock` file:

```bash
uv sync
.venv/bin/pre-commit install
.venv/bin/python -m pytest
.venv/bin/pre-commit run --all-files
.venv/bin/python -m sphinx -W --keep-going -b html docs docs/_build/html
uv build
```

The pre-commit hooks run Ruff lint and format checks, strict mypy,
sp-repo-review, codespell, rumdl, zizmor, and basic file checks. CI runs the
same hooks on every file. The hooks never rewrite files. Run
`.venv/bin/python -m ruff format` to format Python sources. The
[development guide](docs/development.md) describes the environment and every
check.

New Python functions must have type annotations and Google-style docstrings.
Add tests for behavior changes, including failure paths where data may be
created or replaced. Keep user-facing text and source files ASCII unless a
domain term requires otherwise.

Format changes must update the canonical extension document, the format model,
the validator, and conformance tests in the same change.

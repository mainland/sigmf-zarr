# Contributing

Contributions are welcome through the repository issue tracker and pull request
workflow.

Use Python 3.12 or newer and the repository `uv.lock` file:

```bash
uv sync --extra dev --extra docs
uv run pre-commit install
uv run pytest
uv run ruff check src tests docs
uv run docformatter --check --recursive src tests
uv run mypy src
uv run --extra docs sphinx-build -b html -W docs docs/_build/html
uv build
```

Run every pre-commit hook against the repository before submitting a change:

```bash
uv run pre-commit run --all-files
```

New Python functions must have type annotations and Google-style docstrings.
Add tests for behavior changes, including failure paths where data may be
created or replaced. Keep user-facing text and source files ASCII unless a
domain term requires otherwise.

Format changes must update the canonical extension document, the format model,
the validator, and conformance tests in the same change.

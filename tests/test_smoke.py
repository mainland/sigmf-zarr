"""Smoke tests for the sigmf_zarr package."""

import sigmf_zarr


def test_package_imports() -> None:
    """Import the top-level package."""
    assert sigmf_zarr.__version__

"""Smoke tests for the sigmf_zarr package."""

def test_package_imports() -> None:
    """Import the top-level package."""
    import sigmf_zarr

    assert sigmf_zarr.__version__

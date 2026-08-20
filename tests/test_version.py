"""Tests for the single package version source."""

from __future__ import annotations

import importlib
import importlib.metadata

import pytest

import sigmf_zarr
import sigmf_zarr._version as version_module


def test_version_matches_installed_metadata() -> None:
    """The package version comes from installed distribution metadata."""
    assert sigmf_zarr.__version__ == importlib.metadata.version("sigmf-zarr")


def test_version_falls_back_without_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source tree without metadata reports an explicit unknown version.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    def missing(name: str) -> str:
        """Report that a distribution is not installed.

        Args:
            name: Distribution name.

        Raises:
            PackageNotFoundError: Always.
        """
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    try:
        reloaded = importlib.reload(version_module)
        assert reloaded.__version__ == reloaded.UNKNOWN_VERSION == "0+unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(version_module)

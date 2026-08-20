"""Sphinx configuration for the SigMF-Zarr documentation."""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

project = "SigMF-Zarr"
author = "Geoffrey Mainland"
copyright = "2026, Geoffrey Mainland"

try:
    release = package_version("sigmf-zarr")
except PackageNotFoundError:
    release = "0.0.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "SigMF-Zarr"

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_attr_annotations = True
napoleon_preprocess_types = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

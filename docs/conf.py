"""Sphinx configuration for the SigMF-Zarr documentation."""

from __future__ import annotations

from importlib.metadata import version as package_version

project = "SigMF-Zarr"
author = "Geoffrey Mainland"
copyright = "2026, Geoffrey Mainland"

release = package_version("sigmf-zarr")
nitpicky = True

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

# Each entry is a reference that cannot resolve, grouped by cause.
nitpick_ignore = [
    # NumPy documents its scalar types as attributes, not classes.
    ("py:class", "numpy.complex64"),
    ("py:class", "numpy.float32"),
    ("py:class", "numpy.int16"),
    ("py:class", "numpy.int64"),
    # zarr does not document these configuration types and aliases.
    ("py:class", "zarr.core.array.CompressorLike"),
    ("py:class", "zarr.core.array.CompressorsLike"),
    ("py:class", "zarr.core.array.SerializerLike"),
    ("py:class", "zarr.core.array.ShardsLike"),
    ("py:class", "zarr.core.array_spec.ArrayConfig"),
    ("py:class", "zarr.core.array_spec.ArrayConfigParams"),
    # Autodoc cannot evaluate annotations that name imports made only for
    # type checking, so it leaves every name in them unqualified.
    ("py:class", "AccessModeLiteral"),
    ("py:class", "Array"),
    ("py:class", "CompressorLike"),
    ("py:class", "Figure"),
    ("py:class", "Group"),
    ("py:class", "PathLike"),
    ("py:class", "ShardsLike"),
    ("py:class", "StoreLike"),
    ("py:class", "np.intp"),
    ("py:class", "npt.DTypeLike"),
    ("py:class", "npt.NDArray"),
    # Public signatures name these types by their private defining module.
    # The API reference documents their public names.
    ("py:class", "argparse._SubParsersAction"),
    ("py:class", "sigmf_zarr.query._Node"),
    ("py:class", "sigmf_zarr.readonly._ReadOnlyIndexer"),
    ("py:class", "sigmf_zarr.store._common.ChecksumName"),
    ("py:class", "sigmf_zarr.store._common.ZarrFormat"),
    ("py:class", "sigmf_zarr.store._container.SigMFCollections"),
    ("py:class", "sigmf_zarr.store._container.SigMFRecordings"),
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable/", None),
}

# NumPy's public typing aliases resolve to private modules whose paths change
# between NumPy releases.
nitpick_ignore_regex = [("py:class", r"numpy\._typing\..*")]

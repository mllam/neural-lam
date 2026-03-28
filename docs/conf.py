import importlib.metadata
from pathlib import Path

project = "Neural-LAM"
author = "ML-LAM Team"
copyright = "2024, ML-LAM Team"

try:
    version = importlib.metadata.version("neural-lam")
except importlib.metadata.PackageNotFoundError:
    version = "unknown"

release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx_autodoc_typehints",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_method = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = None
html_theme_options = {
    "display_version": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

master_doc = "index"
language = "en"
highlight_language = "python"

html_sidebars = {
    "**": [
        "versions.html",
        "searchbox.html",
        "navigation.html",
        "relations.html",
        "sourcelink.html",
    ]
}

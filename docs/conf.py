# Standard library
import os
import sys

# Add the project root to sys.path so sphinx can find neural_lam
sys.path.insert(0, os.path.abspath(".."))

project = "Neural-LAM"
copyright = "2024–2026, MLLAM Community"
author = "MLLAM Community"

# General Sphinx configuration
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

extensions = [
    "autoapi.extension",
    "myst_nb",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinxext.opengraph",
]

# AutoAPI settings
autoapi_dirs = ["../neural_lam"]
autoapi_root = "api"
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_python_class_content = "both"
autoapi_member_order = "groupwise"
autoapi_python_use_implicit_namespaces = False
autoapi_keep_files = True
autoapi_add_toctree_entry = True
autoapi_ignore = ["**/tests/**", "**/conftest.py"]

# Napoleon settings
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True

# Intersphinx settings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "pytorch_lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
    "torch_geometric": (
        "https://pytorch-geometric.readthedocs.io/en/latest/",
        None,
    ),
}

# MyST / Notebook settings
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "linkify",
    "substitution",
    "tasklist",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "smartquotes",
    "attrs_inline",
]
nb_execution_mode = "off"

# HTML Theme
html_theme = "sphinx_book_theme"
html_logo = "_static/logo.png"
html_static_path = ["_static"]

html_theme_options = {
    "repository_url": "https://github.com/mllam/neural-lam",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "repository_branch": "main",
    "path_to_docs": "docs",
    "show_navbar_depth": 2,
    "show_toc_level": 2,
    "extra_footer": (
        '<p>Built with <a href="https://www.sphinx-doc.org/">Sphinx</a> | '
        '<a href="https://github.com/mllam/neural-lam">Source</a></p>'
    ),
}

# OpenGraph settings
ogp_site_url = "https://neural-lam.readthedocs.io/en/latest/"
ogp_image = "_static/logo.png"
ogp_use_first_image = True

suppress_warnings = [
    "autoapi.python_import_resolution",
    "myst.xref_missing",
]

# Linkcheck settings
linkcheck_ignore = [
    r"https://kutt\.to/mllam",  # Returns 403 Forbidden for bots
    r"https://docs\.pytorch\.org/.*",  # Flaky anchors in intersphinx
]

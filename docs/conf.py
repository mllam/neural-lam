# Standard library
import sys
from pathlib import Path

# Add the project root to sys.path so sphinx can find neural_lam
sys.path.insert(0, str(Path("..").resolve()))

project = "Neural-LAM"
copyright = "2024–2026, MLLAM Community"
author = "MLLAM Community"

# General Sphinx configuration
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

extensions = [
    "autoapi.extension",
    "myst_parser",
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

# MyST settings
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
html_baseurl = "https://mllam.github.io/neural-lam/"
ogp_site_url = "https://mllam.github.io/neural-lam/"
ogp_image = "_static/logo.png"
ogp_use_first_image = True

# Linkcheck settings
linkcheck_report_timeouts_as_broken = True
linkcheck_ignore = [
    r"https://kutt\.to/mllam",  # Returns 403 Forbidden for bots
    r"https://docs\.pytorch\.org/.*",  # Flaky anchors in intersphinx
    r"https://mllam\.github\.io/neural-lam/.*",  # Own site, not yet deployed
]


def use_tracked_source(app, pagename, templatename, context, doctree):
    """Point edit links of generated pages at their tracked source file."""
    if pagename.startswith("notebooks/"):
        candidates = [f"docs/{pagename}.py"]
    elif pagename.startswith("api/neural_lam/"):
        module = pagename.removeprefix("api/").removesuffix("/index")
        candidates = [f"{module}.py", f"{module}/__init__.py"]
    else:
        return
    root = Path(app.srcdir).parent
    sources = [c for c in candidates if (root / c).is_file()]
    if not sources:
        return
    context["edit_page_provider_name"] = "GitHub"
    context["edit_source_path"] = sources[0]
    # pydata-sphinx-theme requires `file_name` to appear in the template
    context["edit_page_url_template"] = (
        "{{ github_url }}/{{ github_user }}/{{ github_repo }}/edit/"
        "{{ github_version }}/{{ edit_source_path or doc_path ~ file_name }}"
    )


def setup(app):
    """Configure source links before the theme creates its header buttons."""
    app.connect("html-page-context", use_tracked_source, priority=500)

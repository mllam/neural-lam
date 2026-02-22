# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath("../../"))

# Mock heavy scientific and ML dependencies so documentation
# can be built without requiring the full runtime environment.
MOCK_MODULES = [
    "torch",
    "torch.nn",
    "torch.optim",
    "torch.utils",
    "torch_geometric",
    "pytorch_lightning",
    "mlflow",
    "numpy",
    "scipy",
    "pandas",
    "xarray",
    "netCDF4",
    "zarr",
    "matplotlib",
    "matplotlib.pyplot",
    "seaborn",
    "tueplots",
    "wandb",
    "sklearn",
    "tqdm",
    "yaml",
    "loguru",
    "dataclass_wizard",
    "cartopy",
    "cartopy.crs",
    "mllam_data_prep",
    "dask",
    "parse",
]

project = 'Neural-LAM'
copyright = '2026, MLLAM Contributors'
author = 'MLLAM Contributors'
release = '0.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
autosummary_generate = True
autodoc_preserve_defaults = True
autodoc_mock_imports = MOCK_MODULES
suppress_warnings = ["autodoc.import_object"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
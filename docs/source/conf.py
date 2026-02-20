# Configuration file for the Sphinx documentation builder.

import os
import sys

# Make project importable for autodoc
sys.path.insert(0, os.path.abspath("../../"))

# Mock heavy ML dependencies so docs build without full environment
autodoc_mock_imports = [
    # Deep Learning
    "torch",
    "torch.nn",
    "torch.optim",
    "torch.utils",
    "torch_geometric",
    "torch_geometric.nn",
    "pytorch_lightning",
    "mlflow",

    # Scientific Python stack
    "numpy",
    "pandas",
    "scipy",
    "xarray",
    "netCDF4",
    "zarr",

    # Plotting / visualization
    "matplotlib",
    "matplotlib.pyplot",
    "seaborn",
    "tueplots",

    # Misc utilities often used in ML repos
    "sklearn",
    "tqdm",
    "yaml",
]

# -- Project information -----------------------------------------------------

project = "Neural-LAM"
copyright = "2026, MLLAM contributors"
author = "MLLAM contributors"
release = "0.1"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "myst_parser",
]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = []
language = "en"

# -- Options for HTML output 

html_theme = "alabaster"
html_static_path = ["_static"]
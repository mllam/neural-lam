import os
import sys
from unittest.mock import MagicMock

# Point Sphinx to the parent directory containing the neural_lam source code
sys.path.insert(0, os.path.abspath('..'))

project = 'neural-lam'
copyright = '2026, MLLAM Contributors'
author = 'Gaurav Sharma'
release = '0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
]

# --- TWO-TIER MOCKING STRATEGY ---

# TIER 1: Aggressive Mocking (Fixes Type/Math execution crashes)
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

HARD_MOCK_MODULES = [
    "tueplots", 
    "mlflow", 
    "mlflow.pytorch", 
]
sys.modules.update((mod_name, Mock()) for mod_name in HARD_MOCK_MODULES)

# TIER 2: Native Sphinx Mocking (Preserves Class Inheritance)
autodoc_mock_imports = [
    "torch",
    "pytorch_lightning",
    "torch_geometric",
    "xarray",
    "tqdm",
    "matplotlib",
    "cartopy",
    "dgl",
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'special-members': '__init__',
    'inherited-members': True,
    'show-inheritance': True,
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
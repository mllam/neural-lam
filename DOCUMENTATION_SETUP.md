# Documentation System Setup - Complete Implementation

## Overview

A complete end-to-end automatic documentation generation system has been set up for neural-lam. The system automatically generates API documentation from Python docstrings and deploys to GitHub Pages.

## What Has Been Implemented

### 1. **Sphinx Documentation Infrastructure**
   - Complete Sphinx configuration (`docs/conf.py`)
   - Auto-documentation enabled with Napoleon extension for Google-style docstrings
   - ReadTheDocs theme for professional appearance
   - Cross-reference support for Python, NumPy, and PyTorch

### 2. **Documentation Structure**
   - **Overview**: Project introduction and features
   - **Installation**: Setup instructions for different environments
   - **Quick Start**: Practical examples and basic usage
   - **API Reference**: Complete auto-generated documentation from docstrings
     - Core modules (config, utils, metrics, loggers, interactions)
     - Models (BaseGraphModel, GraphLAM, HiLAM, HiLAMParallel)
     - Data storage (base, NetCDF, NPY files)
     - Utilities (graph creation, visualization, loss weighting)
   - **Model Architectures**: Detailed model documentation
   - **Data Handling**: Guide for data preparation and loading
   - **Training Guide**: Training workflows and best practices
   - **Utilities**: Helper tools and utilities

### 3. **CI/CD Integration**
   - GitHub Actions workflow (`.github/workflows/docs.yml`)
   - Automatically builds documentation on:
     - Push to main branch
     - Pull requests (for validation, no deployment)
   - Automatically deploys to GitHub Pages on successful builds

### 4. **Developer Tools**
   - `scripts/make-docs.sh` - Convenient local build script
   - `docs/Makefile` - Standard Sphinx build commands
   - `CONTRIBUTING.md` - Documentation contribution guidelines
   - `docs/README.md` - Documentation maintainer guide

### 5. **Dependencies Management**
   - Added docs dependency group in `pyproject.toml`
   - `docs/requirements.txt` for direct installation
   - Updated `.gitignore` to exclude build artifacts

## Files Created

```
docs/
├── conf.py                  # Sphinx configuration
├── index.rst               # Main documentation page
├── overview.rst            # Project overview
├── installation.rst        # Installation guide
├── quickstart.rst          # Quick start examples
├── models.rst              # Model documentation
├── data.rst                # Data handling guide
├── training.rst            # Training guide
├── utilities.rst           # Utility tools
├── api.rst                 # API reference index
├── api/
│   ├── core.rst           # Core modules API
│   ├── models.rst         # Model architectures API
│   ├── datastore.rst      # Data storage API
│   └── utilities.rst      # Utilities API
├── Makefile               # Build commands
├── requirements.txt       # Sphinx dependencies
├── _static/               # Static files directory
└── README.md              # Docs maintainer guide

.github/workflows/
└── docs.yml               # Documentation CI/CD workflow

scripts/
└── make-docs.sh           # Local build script

Root:
├── CONTRIBUTING.md        # Contribution guidelines
└── .gitignore            # Updated with docs build artifacts
```

## How to Use

### Local Development

1. **Install documentation dependencies:**
   ```bash
   pip install -e ".[docs]"
   ```

2. **Build documentation:**
   ```bash
   cd docs && make html
   ```
   Or use the convenience script:
   ```bash
   bash scripts/make-docs.sh
   ```

3. **View documentation:**
   - Open `docs/_build/html/index.html` in a browser
   - Or serve locally:
     ```bash
     cd docs/_build/html && python -m http.server 8000
     # Visit http://localhost:8000
     ```

### Automatic Deployment

Documentation is automatically built and deployed whenever you:

1. Push commits to the `main` branch
2. The GitHub Actions workflow builds the docs and deploys to GitHub Pages

**Note:** GitHub Pages deployment requires repository configuration (see below)

### Adding Documentation

1. **Update existing page:** Edit `.rst` files in `docs/`
2. **Create new page:** 
   - Add new `.rst` file in `docs/`
   - Include in `docs/index.rst` toctree
3. **API documentation:** Automatically generated from Python docstrings
   - Use Google-style docstrings (supported by Napoleon)
   - Docstrings are extracted and displayed in `api/` pages

## GitHub Pages Configuration

To enable documentation publishing, configure GitHub Pages:

1. **Go to Repository Settings** → Pages
2. **Source:** Select "Deploy from branch"
3. **Branch:** Select `gh-pages` (automatically created by GitHub Actions)
4. **Folder:** Select `/ (root)`
5. **Save**

After configuration, documentation is accessible at:
```
https://mllam.github.io/neural-lam/
```

Deployed docs are automatically updated when commits are pushed to main.

## Key Features

### Auto-Generated API Docs
All public modules, classes, and functions are documented automatically from their docstrings:
- Constructor parameters and return values clearly shown
- Type hints displayed inline
- Cross-links to related documentation
- Code examples included from docstrings

### Professional Theme
ReadTheDocs theme provides:
- Responsive mobile-friendly layout
- Full-text search capability
- Navigation sidebar for easy browsing
- Code syntax highlighting
- Cross-references to Python, NumPy, PyTorch docs

### Continuous Updates
- Docs rebuild on every push to main
- Changes visible in GitHub Pages within minutes
- PRs validated with build checks (no deployment)

### Multiple Build Formats
Besides HTML, can generate:
- PDF (requires LaTeX)
- Man pages
- JSON
- EPUB

## Docstring Standards

Write docstrings in Google style (supported by Napoleon):

```python
def train_model(config: ModelConfig, data: DataLoader) -> None:
    """
    Train a neural network model.

    This function trains the model using the provided configuration
    and data loader. It handles all training logistics including
    checkpointing and logging.

    Args:
        config: Model configuration object
        data: PyTorch DataLoader with training data

    Returns:
        None

    Raises:
        ValueError: If config is invalid
        RuntimeError: If training fails

    Examples:
        >>> config = ModelConfig(hidden_dim=128)
        >>> loader = DataLoader(dataset)
        >>> train_model(config, loader)
    """
```

## Build Status

Documentation build status is available in GitHub Actions:
- Workflow: `Build and Deploy Documentation`
- Status badge can be added to README

## Troubleshooting

### Build Failures
1. Check GitHub Actions logs for errors
2. Run locally: `make -C docs html` 
3. Verify all dependencies installed: `pip install -e ".[docs]"`

### Missing API Documentation
- Verify module is importable
- Check docstring format (should use Google style)
- Ensure module is listed in appropriate `api/*.rst` file

### Warnings During Build
- Most warnings don't prevent successful builds
- Fix docstring formatting issues in source code when possible
- Some duplicate object warnings are harmless

## Maintenance

### Regular Tasks
- Update docstrings when code changes
- Keep `.rst` files synchronized with functionality
- Review GitHub Pages deployment status after pushes

### Adding New Modules
1. Create module in `neural_lam/`
2. Add documentation in appropriate `api/` file:
   ```rst
   .. automodule:: neural_lam.new_module
      :members:
      :undoc-members:
      :show-inheritance:
   ```
3. Rebuild docs to verify

## Next Steps

1. **Enable GitHub Pages** (see section above)
2. **Test the workflow** by pushing a commit to main
3. **View deployed docs** at repository URL
4. **Share documentation link** with users and team members

## Documentation at a Glance

| Component | Location | Purpose |
|-----------|----------|---------|
| Sphinx Config | `docs/conf.py` | Build configuration and extensions |
| Main Page | `docs/index.rst` | Documentation entry point |
| User Guides | `docs/*.rst` | Installation, quick start, training, etc. |
| API Docs | `docs/api/*.rst` | Auto-generated API reference |
| Build Script | `scripts/make-docs.sh` | Local build convenience tool |
| CI/CD Workflow | `.github/workflows/docs.yml` | Automatic build and deployment |
| Guidelines | `CONTRIBUTING.md` | How to contribute documentation |
| Maintenance | `docs/README.md` | For documentation maintainers |

## Support

For Sphinx documentation help:
- https://www.sphinx-doc.org/
- https://sphinx-rtd-theme.readthedocs.io/
- https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

For this implementation, refer to:
- `CONTRIBUTING.md` - Contributing guide
- `docs/README.md` - Docs maintainer guide
- `.github/workflows/docs.yml` - Workflow configuration

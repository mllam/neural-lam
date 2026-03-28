# Documentation Build Guide

## Building Documentation Locally

### Prerequisites

Ensure you have the documentation dependencies installed:

```bash
pip install -e ".[docs]"
```

Or manually install:

```bash
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints sphinx-doctest
```

### Building HTML Documentation

From the project root:

```bash
cd docs
make html
```

Or use the convenience script:

```bash
bash scripts/make-docs.sh
```

The built documentation will be in `docs/_build/html/`. Open `docs/_build/html/index.html` in your browser to view it.

### Serving Documentation Locally

To view the documentation in a web browser with a local server:

```bash
cd docs/_build/html
python -m http.server 8000
```

Then open http://localhost:8000 in your browser.

### Building Other Formats

Generate PDF documentation (requires LaTeX):

```bash
cd docs
make latexpdf
```

Generate man pages:

```bash
cd docs
make man
```

## Documentation Structure

```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Main documentation page
├── overview.rst         # Project overview
├── installation.rst     # Installation guide
├── quickstart.rst       # Quick start examples
├── models.rst           # Model documentation
├── data.rst             # Data handling guide
├── training.rst         # Training guide
├── utilities.rst        # Utility tools
├── api.rst              # API reference index
├── api/
│   ├── core.rst         # Core modules
│   ├── models.rst       # Model APIs
│   ├── datastore.rst    # Data storage APIs
│   └── utilities.rst    # Utility APIs
├── _build/              # Build output (generated)
├── _static/             # Static files (CSS, JS)
└── _templates/          # Template overrides
```

## Documentation Standards

When contributing to the documentation:

1. **Docstrings**: Write clear, concise docstrings for all public modules, classes, and functions
2. **Format**: Use Google-style docstrings (supported by Napoleon extension)
3. **Examples**: Include practical examples where helpful
4. **Cross-references**: Link to related API documentation using Sphinx references

### Docstring Example

```python
def my_function(param1: str, param2: int) -> bool:
    """
    Brief description of what the function does.

    Longer description if needed. Can span multiple lines.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When something is invalid
        TypeError: When types don't match

    Examples:
        >>> my_function("example", 42)
        True
    """
    pass
```

## Continuous Integration

Documentation is automatically built and deployed when changes are pushed to the main branch via GitHub Actions. The workflow is defined in `.github/workflows/docs.yml`.

View the deployed documentation at: https://mllam.github.io/neural-lam/

## Troubleshooting

### Build Warnings

Some warnings may appear during builds due to docstring formatting issues. These are typically:

- Missing blank lines in docstrings
- Indentation issues in definition lists
- Duplicate object descriptions (can use `:no-index:` to suppress)

These don't prevent documentation generation but should be fixed in source code docstrings when possible.

### Import Errors

If imports fail during documentation build:

1. Ensure all dependencies are installed: `pip install -e .`
2. Check that the Python path is correct
3. Verify module docstrings don't have import errors

### Build Failures

If the documentation build fails completely:

1. Run `make clean` in the docs directory
2. Regenerate the build: `make html`
3. Check for syntax errors in `.rst` files
4. Review Sphinx error messages carefully

For more help, see the [Sphinx documentation](https://www.sphinx-doc.org/).

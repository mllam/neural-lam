# Documentation

This directory contains the Sphinx-based documentation for Neural-LAM.

## Quick Start

### Building Documentation

Install documentation dependencies:

```bash
pip install -e ".[docs]"
```

Or directly install from docs/requirements.txt:

```bash
pip install -r docs/requirements.txt
```

Build HTML documentation:

```bash
make html
```

View the documentation:

```bash
open _build/html/index.html
```

Or serve locally:

```bash
cd _build/html && python -m http.server 8000
```

Then visit http://localhost:8000

## Continuous Deployment

Documentation is automatically built and deployed to GitHub Pages when changes are pushed to main.

The deployment workflow is triggered by `.github/workflows/docs.yml` which:
1. Builds documentation with Sphinx
2. Uploads the built HTML to GitHub Pages
3. Makes it available at https://mllam.github.io/neural-lam/

## Documentation Structure

- **index.rst** - Main documentation page with navigation
- **overview.rst** - Project overview and features
- **installation.rst** - Installation instructions
- **quickstart.rst** - Quick start guide with code examples
- **models.rst** - Model architecture documentation
- **data.rst** - Data handling and preparation guide
- **training.rst** - Training workflow and best practices
- **utilities.rst** - Utility tools and functions
- **api/** - Complete API reference generated from docstrings
  - **core.rst** - Core modules (config, utils, metrics, etc.)
  - **models.rst** - Model API references
  - **datastore.rst** - Data storage implementations
  - **utilities.rst** - Helper utilities
- **conf.py** - Sphinx configuration
- **Makefile** - Build commands

## Sphinx Configuration

### Extensions

- `sphinx.ext.autodoc` - Automatically documents Python modules
- `sphinx.ext.napoleon` - Supports Google/NumPy style docstrings
- `sphinx_autodoc_typehints` - Displays type hints in documentation
- `sphinx.ext.intersphinx` - Cross-references external documentation
- `sphinx_rtd_theme` - ReadTheDocs theme for professional appearance

### Theme

Uses [Read the Docs Theme](https://sphinx-rtd-theme.readthedocs.io/) providing:
- Responsive mobile-friendly layout
- Full-text search
- Version switching capability
- Built-in analytics support

## Building Other Formats

Generate PDF documentation (requires LaTeX):

```bash
make latexpdf
```

Generate man pages:

```bash
make man
```

Generate Epub:

```bash
make epub
```

## Adding Documentation

### New Pages

Create `.rst` files in the docs directory and include them in `index.rst`:

```rst
.. toctree::
   :maxdepth: 2

   new_page
```

### API Documentation

API docs are auto-generated from docstrings. To document a function:

```python
def my_function(param1: str, param2: int) -> bool:
    """
    Brief one-line description.

    Longer description spanning multiple lines if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When something is wrong

    Examples:
        >>> my_function("test", 42)
        True
    """
```

The autodoc extension will automatically include this in the documentation.

### Cross-References

Link to other documentation:

```rst
See :doc:`quickstart` for examples.
Link to :py:func:`neural_lam.utils.my_function`.
Reference :py:class:`neural_lam.models.GraphLAM`.
```

## Troubleshooting

### Build Errors

If Sphinx build fails:

1. Clean the build: `make clean`
2. Try again: `make html`
3. Check for Python import errors
4. Verify all dependencies are installed

### Missing Cross-References

If cross-references appear broken:

1. Verify the module is importable: `python -c "import neural_lam.module"`
2. Check the exact name of the symbol
3. Use `make clean` and rebuild

### Warnings During Build

Some warnings are normal (e.g., docstring formatting issues in source code). These won't fail the build but should be addressed in source docstrings when possible.

## Contributing

When modifying code, ensure docstrings are updated with:

- Clear description of functionality
- Parameter documentation with types
- Return value documentation
- Examples where helpful
- Raises section for exceptions

See `CONTRIBUTING.md` in the project root for detailed guidelines.

## References

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [Napoleon Extension](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
- [Read the Docs Theme](https://sphinx-rtd-theme.readthedocs.io/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html)

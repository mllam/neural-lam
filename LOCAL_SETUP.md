# Local Setup Guide

This guide explains how to set up the Neural-LAM repository on your local machine after cloning it.

### 1. Prerequisites
- **Python >= 3.10** (e.g., Python 3.11 or 3.12).
- Git.

### 2. Create and Activate a Virtual Environment
It is highly recommended to isolate your project dependencies using a virtual environment. From the root of the cloned repository (`neural-lam`), run:

```bash
# Create a virtual environment named '.venv'
python3 -m venv .venv

# Activate it (on macOS/Linux):
source .venv/bin/activate
```
*(You will need to activate the environment every time you open a new terminal.)*

### 3. Upgrade Pip and Install the Project
Neural-LAM requires up-to-date installation tools to build correctly.

```bash
# Upgrade pip to the latest version
pip install --upgrade pip

# Install the neural-lam package in editable mode along with its main dependencies
pip install -e .
```

### 4. Running the Model
Once installed, you can execute standard python scripts using the `neural_lam` module from the root of the repository. (Refer to the main `README.md` for specific PyTorch Lightning training and evaluation commands).

### 5. Building the Documentation (Optional)
If you want to build the Sphinx documentation locally, there are two important things to note:
1. Because `pyproject.toml` currently uses the new `dependency-groups` feature, standard `pip` may not automatically install the docs extras. Install them manually with:
   ```bash
   pip install "sphinx>=7.0" sphinx-book-theme sphinx-autodoc-typehints "sphinx-copybutton>=0.5" "myst-parser>=2.0"
   ```
2. **You must run the `make html` command from inside the `docs/` directory**:
   ```bash
   cd docs
   make html
   ```

After a successful build, the generated HTML will be located at `docs/_build/html/index.html`. You can open this file in your browser.

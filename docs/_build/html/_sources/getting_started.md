# Getting Started

## Installation

### From PyPI

```bash
python -m pip install neural_lam
```

### From Source (using `uv`)

1. Clone the repository and navigate to the root directory.
2. Install `uv` if you don't have it ([install instructions](https://docs.astral.sh/uv/getting-started/installation)):
   ```bash
   pip install uv
   ```

:::{note}
If you are happy using the latest version of `torch` with GPU support (expecting the
latest CUDA is installed), you can skip to step 5.
:::

3. Create a virtual environment:
   ```bash
   uv venv --no-project
   ```

4. Install a specific version of `torch`:
   ```bash
   # CPU-only
   uv pip install torch --index-url https://download.pytorch.org/whl/cpu

   # Or for CUDA 11.1
   uv pip install torch --index-url https://download.pytorch.org/whl/cu111
   ```
   Find the correct URL for your CUDA version on the [PyTorch website](https://pytorch.org/get-started/locally/).

5. Install Neural-LAM:
   ```bash
   # Standard installation
   uv pip install .

   # Development installation (recommended for contributors)
   uv pip install --group dev -e .
   ```

### From Source (using `pip`)

1. Clone the repository and navigate to the root directory.
2. (Optional) Install a specific version of `torch`:
   ```bash
   python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```
3. Install Neural-LAM:
   ```bash
   # Standard
   python -m pip install .

   # Development
   python -m pip install --group dev -e .
   ```

## Quick Start

Once installed, the general workflow is:

1. **Prepare your data** — set up a datastore configuration (see {doc}`user_guide/data`)
2. **Create a graph** — generate the graph structure for your model (see {doc}`user_guide/graphs`)
3. **Train** — train a model (see {doc}`user_guide/training`)
4. **Evaluate** — evaluate model performance (see {doc}`user_guide/evaluation`)

### Configuration

Neural-LAM uses a YAML configuration file (`config.yaml`) that defines:

1. The datastore kind and path to its config
2. State feature weighting for the loss function
3. Valid numerical ranges for output clamping

Example `config.yaml`:

```yaml
datastore:
  kind: mdp
  config_path: danra.datastore.yaml
training:
  state_feature_weighting:
    __config_class__: ManualStateFeatureWeighting
    weights:
      u100m: 1.0
      v100m: 1.0
      t2m: 1.0
      r2m: 1.0
  output_clamping:
    lower:
      t2m: 0.0
      r2m: 0
    upper:
      r2m: 1.0
```

### Folder Structure

```
data/
├── config.yaml           - Configuration file for neural-lam
├── danra.datastore.yaml  - Configuration file for the datastore
└── graphs/               - Directory containing generated graphs
```

The path to `config.yaml` sets the root directory — all other paths are resolved
relative to its parent directory.

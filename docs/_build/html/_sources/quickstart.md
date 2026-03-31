# Quickstart

This page gives a brief overview of how to get started with neural-lam.

## Configuration

neural-lam uses a YAML configuration file to specify the datastore and
training settings. See the [repository README](https://github.com/mllam/neural-lam)
for full details on configuration options.

## Training a model
```bash
python -m neural_lam.train_model --config_path path/to/config.yaml
```

## Graph generation

Before training, generate the graph components:
```bash
python -m neural_lam.create_graph --config_path path/to/config.yaml
```

A full step-by-step tutorial is coming soon in the HelloWorld notebook.
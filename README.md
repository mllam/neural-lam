![Linting](https://github.com/mllam/neural-lam/actions/workflows/pre-commit.yml/badge.svg?branch=main)
[![test (pdm install, gpu)](https://github.com/mllam/neural-lam/actions/workflows/ci-pdm-install-and-test-gpu.yml/badge.svg)](https://github.com/mllam/neural-lam/actions/workflows/ci-pdm-install-and-test-gpu.yml)
[![test (pdm install, cpu)](https://github.com/mllam/neural-lam/actions/workflows/ci-pdm-install-and-test-cpu.yml/badge.svg)](https://github.com/mllam/neural-lam/actions/workflows/ci-pdm-install-and-test-cpu.yml)

<p align="middle">
    <img src="figures/neural_lam_header.png" width="700">
</p>

Neural-LAM is a repository of graph-based neural weather prediction models for Limited Area Modeling (LAM).
The code uses [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/pytorch-lightning).
Graph Neural Networks are implemented using [PyG](https://pyg.org/) and logging is set up through [Weights & Biases](https://wandb.ai/).

The repository contains LAM versions of:

* The graph-based model from [Keisler (2022)](https://arxiv.org/abs/2202.07575).
* GraphCast, by [Lam et al. (2023)](https://arxiv.org/abs/2212.12794).
* The hierarchical model from [Oskarsson et al. (2023)](https://arxiv.org/abs/2309.17370).

For more information see our paper: [*Graph-based Neural Weather Prediction for Limited Area Modeling*](https://arxiv.org/abs/2309.17370).
If you use Neural-LAM in your work, please cite:
```
@inproceedings{oskarsson2023graphbased,
    title={Graph-based Neural Weather Prediction for Limited Area Modeling},
    author={Oskarsson, Joel and Landelius, Tomas and Lindsten, Fredrik},
    booktitle={NeurIPS 2023 Workshop on Tackling Climate Change with Machine Learning},
    year={2023}
}
```
As the code in the repository is continuously evolving, the latest version might feature some small differences to what was used in the paper.
See the branch [`ccai_paper_2023`](https://github.com/joeloskarsson/neural-lam/tree/ccai_paper_2023) for a revision of the code that reproduces the workshop paper.

We plan to continue updating this repository as we improve existing models and develop new ones.
Collaborations around this implementation are very welcome.
If you are working with Neural-LAM feel free to get in touch and/or submit pull requests to the repository.

# Modularity
The Neural-LAM code is designed to modularize the different components involved in training and evaluating neural weather prediction models.
Models, graphs and data are stored separately and it should be possible to swap out individual components.
Still, some restrictions are inevitable:

* The graph used has to be compatible with what the model expects. E.g. a hierarchical model requires a hierarchical graph.
* The graph and data are specific to the limited area under consideration. This is of course true for the data, but also the graph should be created with the exact geometry of the area in mind.

<p align="middle">
  <img src="figures/neural_lam_setup.png" width="600"/>
</p>


## A note on the limited area setting
Currently we are using these models on a limited area covering the Nordic region, the so called MEPS area (see [paper](https://arxiv.org/abs/2309.17370)).
There are still some parts of the code that is quite specific for the MEPS area use case.
This is in particular true for the mesh graph creation (`python -m neural_lam.create_mesh`) and some of the constants set in a `data_config.yaml` file (path specified in `python -m neural_lam.train_model --data_config <data-config-filepath>` ).
If there is interest to use Neural-LAM for other areas it is not a substantial undertaking to refactor the code to be fully area-agnostic.
We would be happy to support such enhancements.
See the issues https://github.com/joeloskarsson/neural-lam/issues/2, https://github.com/joeloskarsson/neural-lam/issues/3 and https://github.com/joeloskarsson/neural-lam/issues/4 for some initial ideas on how this could be done.

# Using Neural-LAM
Below follows instructions on how to use Neural-LAM to train and evaluate models.

## Installation

When installing `neural-lam` you have a choice of either installing with
directly `pip` or using the `pdm` package manager.
We recommend using `pdm` as it makes it easy to add/remove packages while
keeping versions consistent (it automatically updates the `pyproject.toml`
file), makes it easy to handle virtual environments and includes the
development toolchain packages installation too.

**regarding `torch` installation**: because `torch` creates different package
variants for different CUDA versions and cpu-only support you will need to install
`torch` separately if you don't want the most recent GPU variant that also
expects the most recent version of CUDA on your system.

We cover all the installation options in our [github actions ci/cd
setup](.github/workflows/) which you can use as a reference.

### Using `pdm`

1. Clone this repository and navigate to the root directory.
2. Install `pdm` if you don't have it installed on your system (either with `pip install pdm` or [following the install instructions](https://pdm-project.org/latest/#installation)).
> If you are happy using the latest version of `torch` with GPU support (expecting the latest version of CUDA is installed on your system) you can skip to step 5.
3. Create a virtual environment for pdm to use with `pdm venv create --with-pip`.
4. Install a specific version of `torch` with `pdm run python -m pip install torch --index-url https://download.pytorch.org/whl/cpu` for a CPU-only version or `pdm run python -m pip install torch --index-url https://download.pytorch.org/whl/cu111` for CUDA 11.1 support (you can find the correct URL for the variant you want on [PyTorch webpage](https://pytorch.org/get-started/locally/)).
5. Install the dependencies with `pdm install` (by default this in include the). If you will be developing `neural-lam` we recommend to install the development dependencies with `pdm install --group dev`. By default `pdm` installs the `neural-lam` package in editable mode, so you can make changes to the code and see the effects immediately.

### Using `pip`

1. Clone this repository and navigate to the root directory.
> If you are happy using the latest version of `torch` with GPU support (expecting the latest version of CUDA is installed on your system) you can skip to step 3.
2. Install a specific version of `torch` with `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu` for a CPU-only version or `python -m pip install torch --index-url https://download.pytorch.org/whl/cu111` for CUDA 11.1 support (you can find the correct URL for the variant you want on [PyTorch webpage](https://pytorch.org/get-started/locally/)).
3. Install the dependencies with `python -m pip install .`. If you will be developing `neural-lam` we recommend to install in editable mode and install the development dependencies with `python -m pip install -e ".[dev]"` so you can make changes to the code and see the effects immediately.


## Data
Datasets should be stored in a directory called `data`.
See the [repository format section](#format-of-data-directory) for details on the directory structure.

The full MEPS dataset can be shared with other researchers on request, contact us for this.
A tiny subset of the data (named `meps_example`) is available in `example_data.zip`, which can be downloaded from [here](https://liuonline-my.sharepoint.com/:f:/g/personal/joeos82_liu_se/EuiUuiGzFIFHruPWpfxfUmYBSjhqMUjNExlJi9W6ULMZ1w?e=97pnGX).
Download the file and unzip in the neural-lam directory.
All graphs used in the paper are also available for download at the same link (but can as easily be re-generated using `python -m neural_lam.create_mesh`).
Note that this is far too little data to train any useful models, but all pre-processing and training steps can be run with it.
It should thus be useful to make sure that your python environment is set up correctly and that all the code can be ran without any issues.

## Pre-processing
An overview of how the different pre-processing steps, training and files depend on each other is given in this figure:
<p align="middle">
  <img src="figures/component_dependencies.png"/>
</p>
In order to start training models at least three pre-processing steps have to be run:

* `python -m neural_lam.create_mesh`
* `python -m neural_lam.create_grid_features`
* `python -m neural_lam.create_parameter_weights`

### Create graph
Run `python -m neural_lam.create_mesh` with suitable options to generate the graph you want to use (see `python neural_lam.create_mesh --help` for a list of options).
The graphs used for the different models in the [paper](https://arxiv.org/abs/2309.17370) can be created as:

* **GC-LAM**: `python -m neural_lam.create_mesh --graph multiscale`
* **Hi-LAM**: `python -m neural_lam.create_mesh --graph hierarchical --hierarchical 1` (also works for Hi-LAM-Parallel)
* **L1-LAM**: `python -m neural_lam.create_mesh --graph 1level --levels 1`

The graph-related files are stored in a directory called `graphs`.

### Create remaining static features
To create the remaining static files run `python -m neural_lam.create_grid_features` and `python -m neural_lam.create_parameter_weights`.

## Weights & Biases Integration
The project is fully integrated with [Weights & Biases](https://www.wandb.ai/) (W&B) for logging and visualization, but can just as easily be used without it.
When W&B is used, training configuration, training/test statistics and plots are sent to the W&B servers and made available in an interactive web interface.
If W&B is turned off, logging instead saves everything locally to a directory like `wandb/dryrun...`.
The W&B project name is set to `neural-lam`, but this can be changed in the flags of `python -m neural_lam.train_model` (using argsparse).
See the [W&B documentation](https://docs.wandb.ai/) for details.

If you would like to login and use W&B, run:
```
wandb login
```
If you would like to turn off W&B and just log things locally, run:
```
wandb off
```

## Train Models
Models can be trained using `python -m neural_lam.train_model`.
Run `python neural_lam.train_model --help` for a full list of training options.
A few of the key ones are outlined below:

* `--dataset`: Which data to train on
* `--model`: Which model to train
* `--graph`: Which graph to use with the model
* `--processor_layers`: Number of GNN layers to use in the processing part of the model
* `--ar_steps`: Number of time steps to unroll for when making predictions and computing the loss

Checkpoints of trained models are stored in the `saved_models` directory.
The implemented models are:

### Graph-LAM
This is the basic graph-based LAM model.
The encode-process-decode framework is used with a mesh graph in order to make one-step pedictions.
This model class is used both for the L1-LAM and GC-LAM models from the [paper](https://arxiv.org/abs/2309.17370), only with different graphs.

To train 1L-LAM use
```
python -m neural_lam.train_model --model graph_lam --graph 1level ...
```

To train GC-LAM use
```
python -m neural_lam.train_model --model graph_lam --graph multiscale ...
```

### Hi-LAM
A version of Graph-LAM that uses a hierarchical mesh graph and performs sequential message passing through the hierarchy during processing.

To train Hi-LAM use
```
python -m neural_lam.train_model --model hi_lam --graph hierarchical ...
```

### Hi-LAM-Parallel
A version of Hi-LAM where all message passing in the hierarchical mesh (up, down, inter-level) is ran in parallel.
Not included in the paper as initial experiments showed worse results than Hi-LAM, but could be interesting to try in more settings.

To train Hi-LAM-Parallel use
```
python -m neural_lam.train_model --model hi_lam_parallel --graph hierarchical ...
```

Checkpoint files for our models trained on the MEPS data are available upon request.

## Evaluate Models
Evaluation is also done using `python -m neural_lam.train_model`, but using the `--eval` option.
Use `--eval val` to evaluate the model on the validation set and `--eval test` to evaluate on test data.
Most of the training options are also relevant for evaluation (not `ar_steps`, evaluation always unrolls full forecasts).
Some options specifically important for evaluation are:

* `--load`: Path to model checkpoint file (`.ckpt`) to load parameters from
* `--n_example_pred`: Number of example predictions to plot during evaluation.

**Note:** While it is technically possible to use multiple GPUs for running evaluation, this is strongly discouraged. If using multiple devices the `DistributedSampler` will replicate some samples to make sure all devices have the same batch size, meaning that evaluation metrics will be unreliable. This issue stems from PyTorch Lightning. See for example [this draft PR](https://github.com/Lightning-AI/torchmetrics/pull/1886) for more discussion and ongoing work to remedy this.

# Repository Structure
Except for training and pre-processing scripts all the source code can be found in the `neural_lam` directory.
Model classes, including abstract base classes, are located in `neural_lam/models`.

## Format of data directory
It is possible to store multiple datasets in the `data` directory.
Each dataset contains a set of files with static features and a set of samples.
The samples are split into different sub-directories for training, validation and testing.
The directory structure is shown with examples below.
Script names within parenthesis denote the script used to generate the file.
```
data
├── dataset1
│   ├── samples                             - Directory with data samples
│   │   ├── train                           - Training data
│   │   │   ├── nwp_2022040100_mbr000.npy  - A time series sample
│   │   │   ├── nwp_2022040100_mbr001.npy
│   │   │   ├── ...
│   │   │   ├── nwp_2022043012_mbr001.npy
│   │   │   ├── nwp_toa_downwelling_shortwave_flux_2022040100.npy   - Solar flux forcing
│   │   │   ├── nwp_toa_downwelling_shortwave_flux_2022040112.npy
│   │   │   ├── ...
│   │   │   ├── nwp_toa_downwelling_shortwave_flux_2022043012.npy
│   │   │   ├── wtr_2022040100.npy          - Open water features for one sample
│   │   │   ├── wtr_2022040112.npy
│   │   │   ├── ...
│   │   │   └── wtr_202204012.npy
│   │   ├── val                             - Validation data
│   │   └── test                            - Test data
│   └── static                              - Directory with graph information and static features
│       ├── nwp_xy.npy                      - Coordinates of grid nodes (part of dataset)
│       ├── surface_geopotential.npy        - Geopotential at surface of grid nodes (part of dataset)
│       ├── border_mask.npy                 - Mask with True for grid nodes that are part of border (part of dataset)
│       ├── grid_features.pt                - Static features of grid nodes (neural_lam.create_grid_features)
│       ├── parameter_mean.pt               - Means of state parameters (neural_lam.create_parameter_weights)
│       ├── parameter_std.pt                - Std.-dev. of state parameters (neural_lam.create_parameter_weights)
│       ├── diff_mean.pt                    - Means of one-step differences (neural_lam.create_parameter_weights)
│       ├── diff_std.pt                     - Std.-dev. of one-step differences (neural_lam.create_parameter_weights)
│       ├── flux_stats.pt                   - Mean and std.-dev. of solar flux forcing (neural_lam.create_parameter_weights)
│       └── parameter_weights.npy           - Loss weights for different state parameters (neural_lam.create_parameter_weights)
├── dataset2
├── ...
└── datasetN
```

## Format of graph directory
The `graphs` directory contains generated graph structures that can be used by different graph-based models.
The structure is shown with examples below:
```
graphs
├── graph1                                  - Directory with a graph definition
│   ├── m2m_edge_index.pt                   - Edges in mesh graph (neural_lam.create_mesh)
│   ├── g2m_edge_index.pt                   - Edges from grid to mesh (neural_lam.create_mesh)
│   ├── m2g_edge_index.pt                   - Edges from mesh to grid (neural_lam.create_mesh)
│   ├── m2m_features.pt                     - Static features of mesh edges (neural_lam.create_mesh)
│   ├── g2m_features.pt                     - Static features of grid to mesh edges (neural_lam.create_mesh)
│   ├── m2g_features.pt                     - Static features of mesh to grid edges (neural_lam.create_mesh)
│   └── mesh_features.pt                    - Static features of mesh nodes (neural_lam.create_mesh)
├── graph2
├── ...
└── graphN
```

### Mesh hierarchy format
To keep track of levels in the mesh graph, a list format is used for the files with mesh graph information.
In particular, the files
```
│   ├── m2m_edge_index.pt                   - Edges in mesh graph (neural_lam.create_mesh)
│   ├── m2m_features.pt                     - Static features of mesh edges (neural_lam.create_mesh)
│   ├── mesh_features.pt                    - Static features of mesh nodes (neural_lam.create_mesh)
```
all contain lists of length `L`, for a hierarchical mesh graph with `L` layers.
For non-hierarchical graphs `L == 1` and these are all just singly-entry lists.
Each entry in the list contains the corresponding edge set or features of that level.
Note that the first level (index 0 in these lists) corresponds to the lowest level in the hierarchy.

In addition, hierarchical mesh graphs (`L > 1`) feature a few additional files with static data:
```
├── graph1
│   ├── ...
│   ├── mesh_down_edge_index.pt             - Downward edges in mesh graph (neural_lam.create_mesh)
│   ├── mesh_up_edge_index.pt               - Upward edges in mesh graph (neural_lam.create_mesh)
│   ├── mesh_down_features.pt               - Static features of downward mesh edges (neural_lam.create_mesh)
│   ├── mesh_up_features.pt                 - Static features of upward mesh edges (neural_lam.create_mesh)
│   ├── ...
```
These files have the same list format as the ones above, but each list has length `L-1` (as these edges describe connections between levels).
Entries 0 in these lists describe edges between the lowest levels 1 and 2.

# Development and Contributing
Any push or Pull-Request to the main branch will trigger a selection of pre-commit hooks.
These hooks will run a series of checks on the code, like formatting and linting.
If any of these checks fail the push or PR will be rejected.
To test whether your code passes these checks before pushing, run
``` bash
pre-commit run --all-files
```
from the root directory of the repository.

Furthermore, all tests in the ```tests``` directory will be run upon pushing changes by a github action. Failure in any of the tests will also reject the push/PR.

# Contact
If you are interested in machine learning models for LAM, have questions about our implementation or ideas for extending it, feel free to get in touch.
You can open a github issue on this page, or (if more suitable) send an email to [joel.oskarsson@liu.se](mailto:joel.oskarsson@liu.se).

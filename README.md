![Linting](https://github.com/mllam/neural-lam/actions/workflows/pre-commit.yml/badge.svg)
![Automatic tests](https://github.com/mllam/neural-lam/actions/workflows/run_tests.yml/badge.svg)

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


# Using Neural-LAM
Below follows instructions on how to use Neural-LAM to train and evaluate models.

## Installation
Follow the steps below to create the necessary python environment.

1. Install GEOS for your system. For example with `sudo apt-get install libgeos-dev`. This is necessary for the Cartopy requirement.
2. Use python 3.9.
3. Install version 2.0.1 of PyTorch. Follow instructions on the [PyTorch webpage](https://pytorch.org/get-started/previous-versions/) for how to set this up with GPU support on your system.
4. Install required packages specified in `requirements.txt`.
5. Install PyTorch Geometric version 2.2.0. This can be done by running
```
TORCH="2.0.1"
CUDA="cu117"

pip install pyg-lib==0.2.0 torch-scatter==2.1.1 torch-sparse==0.6.17 torch-cluster==1.6.1\
    torch-geometric==2.3.1 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
```
You will have to adjust the `CUDA` variable to match the CUDA version on your system or to run on CPU. See the [installation webpage](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for more information.

## Data
The repository is set up to work with `yaml` configuration files. These files are used to specify the dataset properties and location. An example of a dataset configuration file is stored in `neural_lam/data_config.yaml` and outlined below.

## Pre-processing
An overview of how the different scripts and files depend on each other is given in this figure:
<p align="middle">
  <img src="figures/component_dependencies.png"/>
</p>
In order to start training models at least one pre-processing script has to be ran:

* `create_mesh.py`

If not provided directly by the user, the following scripts also has to be ran:

* `calculate_statistics.py`
* `create_boundary_mask.py`

The following script is optional, but can be used to create additional features:

* `create_forcing.py`

### Create graph
Run `create_mesh.py` with suitable options to generate the graph you want to use (see `python create_mesh.py --help` for a list of options).
The graphs used for the different models in the [paper](https://arxiv.org/abs/2309.17370) can be created as:

* **GC-LAM**: `python create_mesh.py --graph multiscale`
* **Hi-LAM**: `python create_mesh.py --graph hierarchical --hierarchical 1` (also works for Hi-LAM-Parallel)
* **L1-LAM**: `python create_mesh.py --graph 1level --levels 1`

The graph-related files are stored in a directory called `graphs`.

## Weights & Biases Integration
The project is fully integrated with [Weights & Biases](https://www.wandb.ai/) (W&B) for logging and visualization, but can just as easily be used without it.
When W&B is used, training configuration, training/test statistics and plots are sent to the W&B servers and made available in an interactive web interface.
If W&B is turned off, logging instead saves everything locally to a directory like `wandb/dryrun...`.
The W&B project name is set to `neural-lam`, but this can be changed in the flags of `train_model.py` (using argsparse).
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
Models can be trained using `train_model.py`.
Run `python train_model.py --help` for a full list of training options.
A few of the key ones are outlined below:

* `--data_config`: Path to the data configuration file
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
python train_model.py --model graph_lam --graph 1level ...
```

To train GC-LAM use
```
python train_model.py --model graph_lam --graph multiscale ...
```

### Hi-LAM
A version of Graph-LAM that uses a hierarchical mesh graph and performs sequential message passing through the hierarchy during processing.

To train Hi-LAM use
```
python train_model.py --model hi_lam --graph hierarchical ...
```

### Hi-LAM-Parallel
A version of Hi-LAM where all message passing in the hierarchical mesh (up, down, inter-level) is ran in parallel.
Not included in the paper as initial experiments showed worse results than Hi-LAM, but could be interesting to try in more settings.

To train Hi-LAM-Parallel use
```
python train_model.py --model hi_lam_parallel --graph hierarchical ...
```

Checkpoint files for our models trained on the MEPS data are available upon request.

## Evaluate Models
Evaluation is also done using `train_model.py`, but using the `--eval` option.
Use `--eval val` to evaluate the model on the validation set and `--eval test` to evaluate on test data.
Most of the training options are also relevant for evaluation (not `ar_steps`, evaluation always unrolls full forecasts).
Some options specifically important for evaluation are:

* `--load`: Path to model checkpoint file (`.ckpt`) to load parameters from
* `--n_example_pred`: Number of example predictions to plot during evaluation.

**Note:** While it is technically possible to use multiple GPUs for running evaluation, this is strongly discouraged. If using multiple devices the `DistributedSampler` will replicate some samples to make sure all devices have the same batch size, meaning that evaluation metrics will be unreliable. This issue stems from PyTorch Lightning. See for example [this draft PR](https://github.com/Lightning-AI/torchmetrics/pull/1886) for more discussion and ongoing work to remedy this.

# Repository Structure
Except for training and pre-processing scripts all the source code can be found in the `neural_lam` directory.
Model classes, including abstract base classes, are located in `neural_lam/models`.
Notebooks for visualization and analysis are located in `docs`.


## Format of data directory
The new workflow uses YAML configuration files to specify dataset properties and locations.
Below is an example of how to structure your data directory and a condensed version of the YAML configuration file. The community decided for now, that a zarr-based approach is the most flexible and efficient way to store the data. Please make sure that your dataset is stored as zarr, contains the necessary dimensions, and is structured as described below. For optimal performance chunking the dataset along the time dimension only is recommended.
```
name: danra
state:                                    # State variables vary in time and are predicted by the model
  zarrs:
    - path:                               # Path to the zarr file
      dims:                               # Only the following dimensions will be mapped: time, level, x, y, grid
        time: time                        # Required
        level: null                       # Optional
        x: x                              # Either x and y or grid must be specified
        y: y
        grid: null                        # Grid has precedence over x and y
      lat_lon_names:                      # Required to map grid- projection to lat/lon
        lon: lon
        lat: lat
    - path:
      ...                                 # Additional zarr files are allowed
  surface_vars:                           # Single level variables to include in the state (in this order)
    - var1
    - var2
  surface_units:                          # Units for the surface variables
    - unit1
    - unit2
  atmosphere_vars:                        # Multi-level variables to include in the state (in this order)
    - var1
    ...
  atmosphere_units:                       # Units for the atmosphere variables
    - unit1
    ...
  levels:                                 # Selection of vertical levels to include in the state (pressure/height/model level)
    - 100
    - 200
    ...
forcing:                                  # Forcing variables vary in time but are not predicted by the model
  ...                                     # Same structure as state, multiple zarr files allowed
  window: 3                               # Number of time steps to use for forcing (odd number)
static:                                   # Static variables are not predicted by the model and do not vary in time
  zarrs:
    ...
      dims:                               # Same structure as state but no "time" dimension
        level: null
        x: x
        y: y
        grid: null
    ...
boundary:                                 # Boundary variables are not predicted by the model and do not vary in time
    ...                                   # They are used to inform the model about the surrounding weather conditions
    ...                                   # The boundaries are often used from a separate model, specified identically to the state
  mask:                                   # Boundary mask to indicate where the model should not make predictions
    path: "data/boundary_mask.zarr"
    dims:
      x: x
      y: y
  window: 3                               # Windowing of the boundary variables (odd number), may differ from forcing window
utilities:                                # Additional utilities to be used in the model
  normalization:                          # Normalization statistics for the state, forcing, and one-step differences
    zarrs:                                # Zarr files containing the normalization statistics, multiple allowed
      - path: "data/normalization.zarr"        # Path to the zarr file, default locaton of `calculate_statistics.py`
        stats_vars:                       # The variables to use for normalization, predefined and required
          state_mean: name_in_dataset1
          state_std: name_in_dataset2
          forcing_mean: name_in_dataset3
          forcing_std: name_in_dataset4
          diff_mean: name_in_dataset5
          diff_std: name_in_dataset6
    combined_stats:                       # For some variables the statistics can be retrieved jointly
      - vars:                             # List of variables that should end of with the same statistics
        - vars1
        - vars2
      - vars:
        ...
grid_shape_state:                         # Shape of the state grid, used for reshaping the model output
  y: 589                                  # Number of grid points in the y-direction (lat)
  x: 789                                  # Number of grid points in the x-direction (lon)
splits:                                   # Train, validation, and test splits based on time-sampling
  train:
    start: 1990-09-01T00
    end: 1990-09-11T00
  val:
    start: 1990-09-11T03
    end: 1990-09-13T09
  test:
    start: 1990-09-11T03
    end: 1990-09-13T09
projection:                               # Projection of the grid (only used for plotting)
  class: LambertConformal                 # Name of class in cartopy.crs
  kwargs:
    central_longitude: 6.22
    central_latitude: 56.0
    standard_parallels: [47.6, 64.4]

```

## Format of graph directory
The `graphs` directory contains generated graph structures that can be used by different graph-based models.
The structure is shown with examples below:
```
graphs
├── graph1                                  - Directory with a graph definition
│   ├── m2m_edge_index.pt                   - Edges in mesh graph (create_mesh.py)
│   ├── g2m_edge_index.pt                   - Edges from grid to mesh (create_mesh.py)
│   ├── m2g_edge_index.pt                   - Edges from mesh to grid (create_mesh.py)
│   ├── m2m_features.pt                     - Static features of mesh edges (create_mesh.py)
│   ├── g2m_features.pt                     - Static features of grid to mesh edges (create_mesh.py)
│   ├── m2g_features.pt                     - Static features of mesh to grid edges (create_mesh.py)
│   └── mesh_features.pt                    - Static features of mesh nodes (create_mesh.py)
├── graph2
├── ...
└── graphN
```

### Mesh hierarchy format
To keep track of levels in the mesh graph, a list format is used for the files with mesh graph information.
In particular, the files
```
│   ├── m2m_edge_index.pt                   - Edges in mesh graph (create_mesh.py)
│   ├── m2m_features.pt                     - Static features of mesh edges (create_mesh.py)
│   ├── mesh_features.pt                    - Static features of mesh nodes (create_mesh.py)
```
all contain lists of length `L`, for a hierarchical mesh graph with `L` layers.
For non-hierarchical graphs `L == 1` and these are all just singly-entry lists.
Each entry in the list contains the corresponding edge set or features of that level.
Note that the first level (index 0 in these lists) corresponds to the lowest level in the hierarchy.

In addition, hierarchical mesh graphs (`L > 1`) feature a few additional files with static data:
```
├── graph1
│   ├── ...
│   ├── mesh_down_edge_index.pt             - Downward edges in mesh graph (create_mesh.py)
│   ├── mesh_up_edge_index.pt               - Upward edges in mesh graph (create_mesh.py)
│   ├── mesh_down_features.pt               - Static features of downward mesh edges (create_mesh.py)
│   ├── mesh_up_features.pt                 - Static features of upward mesh edges (create_mesh.py)
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

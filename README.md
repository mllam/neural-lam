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

For more information see our preprint: [*Graph-based Neural Weather Prediction for Limited Area Modeling*](https://arxiv.org/abs/2309.17370).
If you use Neural-LAM in your work, please cite:
```
@article{oskarsson2023graphbased,
      title={Graph-based Neural Weather Prediction for Limited Area Modeling},
      author={Joel Oskarsson and Tomas Landelius and Fredrik Lindsten},
      year={2023},
      journal={arXiv preprint arXiv:2309.17370}
}
```

We plan to continue updating this repository as we improve existing models and develop new ones.
Collaborations around this implementation are very welcome.
If you are working with Neural-LAM feel free to get in touch and/or submit pull requests to the repository.

<span style="color:blue;">Additions relevant to the COSMO Neural-LAM implementation are highlighted in __blue__.</span>
# Quick Start
<span style="color:blue;">
Follow the steps below to get started with Neural-LAM on Balfrin.cscs.ch.
Don't worry everything is carried out on a small subset of data for a limited number of epochs.
</span>

```{bash}
# Clone the repository
git clone https://github.com/MeteoSwiss/neural-lam/
cd neural-lam

# Link the data folder containing the COSMO zarr archives
ln -s /scratch/mch/sadamov/pyprojects_data/neural_lam/data
mkdir lightning_logs

# Create the conda environment (~10min)
mamba env create -f environment.yml
mamba activate neural-lam

# Run the preprocessing/training scripts
sbatch slurm_train.sh

# Run the evaluation script and generate plots and gif for TQV
# (don't execute preprocessing scripts at the same time as training)
sbatch slurm_eval.sh

```


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
This is in particular true for the mesh graph creation (`create_mesh.py`) and some of the constants used (`neural_lam/constants.py`).
If there is interest to use Neural-LAM for other areas it is not a substantial undertaking to refactor the code to be fully area-agnostic.
We would be happy to support such enhancements.
See the issues https://github.com/joeloskarsson/neural-lam/issues/2, https://github.com/joeloskarsson/neural-lam/issues/3 and https://github.com/joeloskarsson/neural-lam/issues/4 for some initial ideas on how this could be done.

<span style="color:blue;">

For the COSMO implementation some additional settings can be defined in `neural_lam/constants`. Most of the code should take user input either from `neural_lam/constants` or directly from command-line argument parsing. Would certainly be worth the effort to make the code fully area-agnostic.

</span>

# Using Neural-LAM
Below follows instructions on how to use Neural-LAM to train and evaluate models.

## Installation

<span style="color:blue;">

For COSMO we use conda to avoid the Cartopy installation issues and because conda environments usually work well on the vCluster called Balfrin.cscs.ch.

1. Simply run `conda env create -f environment.yml` to create the environment.
2. Activate the environment with `conda activate neural-lam`.
3. Happy Coding \o/

Note that only the cuda version is pinned to 11.8, otherwise all the latest libraries are installed. This might break in the future and must be adjusted to the users conda version.

</span>

\
Follow the steps below to create the neccesary python environment.

1. Install GEOS for your system. For example with `sudo apt-get install libgeos-dev`. This is neccesary for the Cartopy requirement.
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
Datasets should be stored in a directory called `data`.
See the [repository format section](#format-of-data-directory) for details on the directory structure.

The full MEPS dataset can be shared with other researchers on request, contact us for this.
A tiny subset of the data (named `meps_example`) is available in `example_data.zip`, which can be downloaded from [here](https://liuonline-my.sharepoint.com/:f:/g/personal/joeos82_liu_se/EuiUuiGzFIFHruPWpfxfUmYBSjhqMUjNExlJi9W6ULMZ1w?e=97pnGX).
Download the file and unzip in the neural-lam directory.
All graphs used in the paper are also available for download at the same link (but can as easily be re-generated using `create_mesh.py`).
Note that this is far too little data to train any useful models, but all scripts can be ran with it.
It should thus be useful to make sure that your python environment is set up correctly and that all the code can be ran without any issues.

<span style="color:blue;">

For COSMO the data is stored in the `data` folder with the same structure, but called `cosmo`. The data will be open-source someday but for now we cannot share the data outside of our vCluster. A tiny example dataset could probably be made available.

</span>

## Pre-processing
An overview of how the different scripts and files depend on each other is given in this figure:
<p align="middle">
  <img src="figures/component_dependencies.png"/>
</p>
In order to start training models at least three pre-processing scripts have to be ran:

* `create_mesh.py`
* `create_grid_features.py`
* `create_parameter_weights.py`

<span style="color:blue;">

For COSMO also run `create_static_features.py` to create the static features for the graph nodes.

</span>

### Create graph
Run `create_mesh.py` with suitable options to generate the graph you want to use (see `python create_mesh.py --help` for a list of options).
The graphs used for the different models in the [paper](https://arxiv.org/abs/2309.17370) can be created as:

* **GC-LAM**: `python create_mesh.py --graph multiscale`
* **Hi-LAM**: `python create_mesh.py --graph hierarchical --hierarchical 1` (also works for Hi-LAM-Parallel)
* **L1-LAM**: `python create_mesh.py --graph 1level --levels 1`

The graph-related files are stored in a directory called `graphs`.

### Create remaining static features
To create the remaining static files run the scripts `create_grid_features.py` and `create_parameter_weights.py`.
The main option to set for these is just which dataset to use.

## Weights & Biases Integration
The project is fully integrated with [Weights & Biases](https://www.wandb.ai/) (W&B) for logging and visualization, but can just as easily be used without it.
When W&B is used, training configuration, training/test statistics and plots are sent to the W&B servers and made available in an interactive web interface.
If W&B is turned off, logging instead saves everything locally to a directory like `wandb/dryrun...`.
The W&B project name is set to `neural-lam`, but this can be changed in `neural_lam/constants.py`.
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

* `--dataset`: Which data to train on
* `--model`: Which model to train
* `--graph`: Which graph to use with the model
* `--processor_layers`: Number of GNN layers to use in the processing part of the model
* `--ar_steps`: Number of time steps to unroll for when making predictions and computing the loss

<span style="color:blue;">

For COSMO three simple slurm sbatch scripts are available for training/evaluating the model. You can launch either of these jobs respectively with:

```
sbatch slurm_train.sh
sbatch slurm_eval.sh
```

This will train the model using the same seed and data as seen in the figures on wandb.

</span>


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
A version of Hi-LAM where all message passing in the hierarchical mesh (up, down, inter-level) is ran in paralell.
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
│       ├── grid_features.pt                - Static features of grid nodes (create_grid_features.py)
│       ├── parameter_mean.pt               - Means of state parameters (create_parameter_weights.py)
│       ├── parameter_std.pt                - Std.-dev. of state parameters (create_parameter_weights.py)
│       ├── diff_mean.pt                    - Means of one-step differences (create_parameter_weights.py)
│       ├── diff_std.pt                     - Std.-dev. of one-step differences (create_parameter_weights.py)
│       ├── flux_stats.pt                   - Mean and std.-dev. of solar flux forcing (create_parameter_weights.py)
│       └── parameter_weights.npy           - Loss weights for different state parameters (create_parameter_weights.py)
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

# Contact
If you are interested in machine learning models for LAM, have questions about our implementation or ideas for extending it, feel free to get in touch.
You can open a github issue on this page, or (if more suitable) send an email to [joel.oskarsson@liu.se](mailto:joel.oskarsson@liu.se).

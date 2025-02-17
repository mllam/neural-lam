[![slack](https://img.shields.io/badge/slack-join-brightgreen.svg?logo=slack)](https://join.slack.com/t/ml-lam/shared_invite/zt-2t112zvm8-Vt6aBvhX7nYa6Kbj_LkCBQ)
[![Linting](https://github.com/mllam/neural-lam/actions/workflows/pre-commit.yml/badge.svg?branch=main)](https://github.com/mllam/neural-lam/actions/workflows/pre-commit.yml)
[![CPU+GPU testing](https://github.com/mllam/neural-lam/actions/workflows/install-and-test.yml/badge.svg?branch=main)](https://github.com/mllam/neural-lam/actions/workflows/install-and-test.yml)

<p align="middle">
    <img src="figures/neural_lam_header.png" width="700">
</p>

Neural-LAM is a repository of graph-based neural weather prediction models for Limited Area Modeling (LAM).
Also global forecasting is possible, but currently on a [different branch](https://github.com/mllam/neural-lam/tree/prob_model_global) ([planned to be merged with main](https://github.com/mllam/neural-lam/issues/63)).
The code uses [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/pytorch-lightning).
Graph Neural Networks are implemented using [PyG](https://pyg.org/) and logging is set up through [Weights & Biases](https://wandb.ai/).

The repository contains LAM versions of:

* The graph-based model from [Keisler (2022)](https://arxiv.org/abs/2202.07575).
* GraphCast, by [Lam et al. (2023)](https://arxiv.org/abs/2212.12794).
* The hierarchical model from [Oskarsson et al. (2023)](https://arxiv.org/abs/2309.17370).

# Publications
For a more in-depth scientific introduction to machine learning for LAM weather forecasting see the publications listed here.
As the code in the repository is continuously evolving, the latest version might feature some small differences to what was used for these publications.
We retain some paper-specific branches for reproducibility purposes.


*If you use Neural-LAM in your work, please cite the relevant paper(s)*.

#### [Graph-based Neural Weather Prediction for Limited Area Modeling](https://arxiv.org/abs/2309.17370)
```
@inproceedings{oskarsson2023graphbased,
    title={Graph-based Neural Weather Prediction for Limited Area Modeling},
    author={Oskarsson, Joel and Landelius, Tomas and Lindsten, Fredrik},
    booktitle={NeurIPS 2023 Workshop on Tackling Climate Change with Machine Learning},
    year={2023}
}
```
See the branch [`ccai_paper_2023`](https://github.com/joeloskarsson/neural-lam/tree/ccai_paper_2023) for a revision of the code that reproduces this workshop paper.

#### [Probabilistic Weather Forecasting with Hierarchical Graph Neural Networks](https://arxiv.org/abs/2406.04759)
```
@inproceedings{oskarsson2024probabilistic,
  title = {Probabilistic Weather Forecasting with Hierarchical Graph Neural Networks},
  author = {Oskarsson, Joel and Landelius, Tomas and Deisenroth, Marc Peter and Lindsten, Fredrik},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {37},
  year = {2024},
}
```
See the branches [`prob_model_lam`](https://github.com/mllam/neural-lam/tree/prob_model_lam) and [`prob_model_global`](https://github.com/mllam/neural-lam/tree/prob_model_global) for revisions of the code that reproduces this paper.
The global and probabilistic models from this paper are not yet fully merged with `main` (see issues [62](https://github.com/mllam/neural-lam/issues/62) and [63](https://github.com/mllam/neural-lam/issues/63)).

# Modularity
The Neural-LAM code is designed to modularize the different components involved in training and evaluating neural weather prediction models.
Models, graphs and data are stored separately and it should be possible to swap out individual components.
Still, some restrictions are inevitable:

* The graph used has to be compatible with what the model expects. E.g. a hierarchical model requires a hierarchical graph.
* The graph and data are specific to the limited area under consideration. This is of course true for the data, but also the graph should be created with the exact geometry of the area in mind.

<p align="middle">
  <img src="figures/neural_lam_setup.png" width="600"/>
</p>


# Installing Neural-LAM

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

## Using `pdm`

1. Clone this repository and navigate to the root directory.
2. Install `pdm` if you don't have it installed on your system (either with `pip install pdm` or [following the install instructions](https://pdm-project.org/latest/#installation)).
> If you are happy using the latest version of `torch` with GPU support (expecting the latest version of CUDA is installed on your system) you can skip to step 5.
3. Create a virtual environment for pdm to use with `pdm venv create --with-pip`.
4. Install a specific version of `torch` with `pdm run python -m pip install torch --index-url https://download.pytorch.org/whl/cpu` for a CPU-only version or `pdm run python -m pip install torch --index-url https://download.pytorch.org/whl/cu111` for CUDA 11.1 support (you can find the correct URL for the variant you want on [PyTorch webpage](https://pytorch.org/get-started/locally/)).
5. Install the dependencies with `pdm install` (by default this in include the). If you will be developing `neural-lam` we recommend to install the development dependencies with `pdm install --group dev`. By default `pdm` installs the `neural-lam` package in editable mode, so you can make changes to the code and see the effects immediately.

## Using `pip`

1. Clone this repository and navigate to the root directory.
> If you are happy using the latest version of `torch` with GPU support (expecting the latest version of CUDA is installed on your system) you can skip to step 3.
2. Install a specific version of `torch` with `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu` for a CPU-only version or `python -m pip install torch --index-url https://download.pytorch.org/whl/cu111` for CUDA 11.1 support (you can find the correct URL for the variant you want on [PyTorch webpage](https://pytorch.org/get-started/locally/)).
3. Install the dependencies with `python -m pip install .`. If you will be developing `neural-lam` we recommend to install in editable mode and install the development dependencies with `python -m pip install -e ".[dev]"` so you can make changes to the code and see the effects immediately.


# Using Neural-LAM

Once `neural-lam` is installed you will be able to train/evaluate models. For this you will in general need two things:

1. **Data to train/evaluate the model**. To represent this data we use a concept of
   *datastores* in Neural-LAM (see the [Data](#data-the-datastore-and-weatherdataset-classes) section for more details).
   In brief, a datastore implements the process of loading data from disk in a
   specific format (for example zarr or numpy files) by implementing an
   interface that provides the data in a data-structure that can be used within
   neural-lam. A datastore is used to create a `pytorch.Dataset`-derived
   class that samples the data in time to create individual samples for
   training, validation and testing.

2. **The graph structure** is used to define message-passing GNN layers,
   that are trained to emulate fluid flow in the atmosphere over time. The
   graph structure is created for a specific datastore.

Any command you run in neural-lam will include the path to a configuration file
to be used (usually called `config.yaml`). This configuration file defines the
path to the datastore configuration you wish to use and allows you to configure
different aspects about the training and evaluation of the model.

The path you provide to the neural-lam config (`config.yaml`) also sets the
root directory relative to which all other paths are resolved, as in the parent
directory of the config becomes the root directory. Both the datastore and
graphs you generate are then stored in subdirectories of this root directory.
Exactly how and where a specific datastore expects its source data to be stored
and where it stores its derived data is up to the implementation of the
datastore.

In general the folder structure assumed in Neural-LAM is as follows (we will
assume you placed `config.yaml` in a folder called `data`):

```
data/
├── config.yaml           - Configuration file for neural-lam
├── danra.datastore.yaml  - Configuration file for the datastore, referred to from config.yaml
└── graphs/               - Directory containing graphs for training
```

And the content of `config.yaml` could in this case look like:
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

For now the neural-lam config only defines few things:

1. The kind of datastore and the path to its config
2. The weighting of different features in
the loss function. If you don't define the state feature weighting it will default to
weighting all features equally.
3. Valid numerical range for output of each feature.The numerical range of all features default to $]-\infty, \infty[$.

(This example is taken from the `tests/datastore_examples/mdp` directory.)


Below follows instructions on how to use Neural-LAM to train and evaluate
models, with details first given for each kind of datastore implemented
and later the graph generation. Once `neural-lam` has been installed the
general process is:

1. Run any pre-processing scripts to generate the necessary derived data that your chosen datastore requires
2. Run graph-creation step
3. Train the model

## Data (the `DataStore` and `WeatherDataset` classes)

To enable flexibility in what input-data sources can be used with neural-lam,
the input-data representation is split into two parts:

1. A "datastore" (represented by instances of
   [neural_lam.datastore.BaseDataStore](neural_lam/datastore/base.py)) which
   takes care of loading a given category (state, forcing or static) and split
   (train/val/test) of data from disk and returning it as a `xarray.DataArray`.
   The returned data-array is expected to have the spatial coordinates
   flattened into a single `grid_index` dimension and all variables and vertical
   levels stacked into a feature dimension (named as `{category}_feature`). The
   datastore also provides information about the number, names and units of
   variables in the data, the boundary mask, normalisation values and grid
   information.

2. A `pytorch.Dataset`-derived class (called
   `neural_lam.weather_dataset.WeatherDataset`) which takes care of sampling in
   time to create individual samples for training, validation and testing. The
   `WeatherDataset` class is also responsible for normalising the values and
   returning `torch.Tensor`-objects.

There are currently two different datastores implemented in the codebase:

1. `neural_lam.datastore.MDPDatastore` which represents loading of
   *training-ready* datasets in zarr format created with the
   [mllam-data-prep](https://github.com/mllam/mllam-data-prep) package.
   Training-ready refers to the fact that this data has been transformed
   (variables have been stacked, spatial coordinates have been flattened,
   statistics for normalisation have been calculated, etc) to be ready for
   training. `mllam-data-prep` can combine any number of datasets that can be
   read with [xarray](https://github.com/pydata/xarray) and the processing can
   either be done at run-time or as a pre-processing step before calling
   neural-lam.

2. `neural_lam.datastore.NpyFilesDatastoreMEPS` which reads MEPS data from
   `.npy`-files in the format introduced in neural-lam `v0.1.0`. Note that this
   datastore is specific to the format of the MEPS dataset, but can act as an
   example for how to create similar numpy-based datastores.

If neither of these options fit your need you can create your own datastore by
subclassing the `neural_lam.datastore.BaseDataStore` class or
`neural_lam.datastore.BaseRegularGridDatastore` class (if your data is stored on
a regular grid) and implementing the abstract methods.


### MDP (mllam-data-prep) Datastore - `MDPDatastore`

With `MDPDatastore` (the mllam-data-prep datastore) all the selection,
transformation and pre-calculation steps that are needed to go from
for example gridded weather data to a format that is optimised for training
in neural-lam, are done in a separate package called
[mllam-data-prep](https://github.com/mllam/mllam-data-prep) rather than in
neural-lam itself.
Specifically, the `mllam-data-prep` datastore configuration (for example
[danra.datastore.yaml](tests/datastore_examples/mdp/danra.datastore.yaml))
specifies a) what source datasets to read from, b) what variables to select, c)
what transformations of dimensions and variables to make, d) what statistics to
calculate (for normalisation) and e) how to split the data into training,
validation and test sets (see full details about the configuration specification
in the [mllam-data-prep README](https://github.com/mllam/mllam-data-prep)).

From a datastore configuration `mllam-data-prep` returns the transformed
dataset as an `xr.Dataset` which is then written in zarr-format to disk by
`neural-lam` when the datastore is first initiated (the path of the dataset is
derived from the datastore config, so that from a config named `danra.datastore.yaml` the resulting dataset is stored in `danra.datastore.zarr`).
You can also run `mllam-data-prep` directly to create the processed dataset by providing the path to the datastore configuration file:

```bash
python -m mllam_data_prep --config data/danra.datastore.yaml
```

If you will be working on a large dataset (on the order of 10GB or more) it
could be beneficial to produce the processed `.zarr` dataset before using it
in neural-lam so that you can do the processing across multiple CPU cores in parallel. This is done by including the `--dask-distributed-local-core-fraction` argument when calling mllam-data-prep to set the fraction of your system's CPU cores that should be used for processing (see the
[mllam-data-prep
README for details](https://github.com/mllam/mllam-data-prep?tab=readme-ov-file#creating-large-datasets-with-daskdistributed)).

For example:

```bash
python -m mllam_data_prep --config data/danra.datastore.yaml --dask-distributed-local-core-fraction 0.5
```

### NpyFiles MEPS Datastore - `NpyFilesDatastoreMEPS`

Version `v0.1.0` of Neural-LAM was built to train from numpy-files from the
MEPS weather forecasts dataset.
To enable this functionality to live on in later versions of neural-lam we have
built a datastore called `NpyFilesDatastoreMEPS` which implements functionality
to read from these exact same numpy-files. At this stage this datastore class
is very much tied to the MEPS dataset, but the code is written in a way where
it quite easily could be adapted to work with numpy-based weather
forecast/analysis files in future.

The full MEPS dataset can be shared with other researchers on request, contact us for this.
A tiny subset of the data (named `meps_example`) is available in
`example_data.zip`, which can be downloaded from
[here](https://liuonline-my.sharepoint.com/:f:/g/personal/joeos82_liu_se/EuiUuiGzFIFHruPWpfxfUmYBSjhqMUjNExlJi9W6ULMZ1w?e=97pnGX).

Download the file and unzip in the neural-lam directory.
Graphs used in the initial paper are also available for download at the same link (but can as easily be re-generated using `python -m neural_lam.create_graph`).
Note that this is far too little data to train any useful models, but all pre-processing and training steps can be run with it.
It should thus be useful to make sure that your python environment is set up correctly and that all the code can be ran without any issues.

The following datastore configuration works with the MEPS dataset:

```yaml
# meps.datastore.yaml
dataset:
  name: meps_example
  num_forcing_features: 16
  var_longnames:
  - pres_heightAboveGround_0_instant
  - pres_heightAboveSea_0_instant
  - nlwrs_heightAboveGround_0_accum
  - nswrs_heightAboveGround_0_accum
  - r_heightAboveGround_2_instant
  - r_hybrid_65_instant
  - t_heightAboveGround_2_instant
  - t_hybrid_65_instant
  - t_isobaricInhPa_500_instant
  - t_isobaricInhPa_850_instant
  - u_hybrid_65_instant
  - u_isobaricInhPa_850_instant
  - v_hybrid_65_instant
  - v_isobaricInhPa_850_instant
  - wvint_entireAtmosphere_0_instant
  - z_isobaricInhPa_1000_instant
  - z_isobaricInhPa_500_instant
  var_names:
  - pres_0g
  - pres_0s
  - nlwrs_0
  - nswrs_0
  - r_2
  - r_65
  - t_2
  - t_65
  - t_500
  - t_850
  - u_65
  - u_850
  - v_65
  - v_850
  - wvint_0
  - z_1000
  - z_500
  var_units:
  - Pa
  - Pa
  - W/m\textsuperscript{2}
  - W/m\textsuperscript{2}
  - "-"
  - "-"
  - K
  - K
  - K
  - K
  - m/s
  - m/s
  - m/s
  - m/s
  - kg/m\textsuperscript{2}
  - m\textsuperscript{2}/s\textsuperscript{2}
  - m\textsuperscript{2}/s\textsuperscript{2}
  num_timesteps: 65
  num_ensemble_members: 2
  step_length: 3
  remove_state_features_with_index: [15]
grid_shape_state:
- 268
- 238
projection:
  class_name: LambertConformal
  kwargs:
    central_latitude: 63.3
    central_longitude: 15.0
    standard_parallels:
    - 63.3
    - 63.3
```

Which you can then use in a neural-lam configuration file like this:

```yaml
# config.yaml
datastore:
  kind: npyfilesmeps
  config_path: meps.datastore.yaml
training:
  state_feature_weighting:
    __config_class__: ManualStateFeatureWeighting
    values:
      u100m: 1.0
      v100m: 1.0
```

For npy-file based datastores you must separately run the command that creates the variables used for standardization:

```bash
python -m neural_lam.datastore.npyfilesmeps.compute_standardization_stats <path-to-datastore-config>
```

### Graph creation

Run `python -m neural_lam.create_mesh` with suitable options to generate the graph you want to use (see `python neural_lam.create_mesh --help` for a list of options).
The graphs used for the different models in the [paper](#graph-based-neural-weather-prediction-for-limited-area-modeling) can be created as:

* **GC-LAM**: `python -m neural_lam.create_graph --config_path <neural-lam-config-path> --name multiscale`
* **Hi-LAM**: `python -m neural_lam.create_graph --config_path <neural-lam-config-path> --name hierarchical --hierarchical` (also works for Hi-LAM-Parallel)
* **L1-LAM**: `python -m neural_lam.create_graph --config_path <neural-lam-config-path> --name 1level --levels 1`

The graph-related files are stored in a directory called `graphs`.

## Logging your experiments

### Weights & Biases Integration
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

### MLFlow Integration
The project is also integrated with [MLFlow](https://mlflow.org/) for logging and storing artefacts.

MLFlow is not used by default, but can be switched to by setting `--logger mlflow` in the training command. With MLFlow enabled, training configuration, training/test statistics and plots are logged to the MLFlow server. MLFlow is self-hosted and can be run locally or on a server. See the [MLFlow documentation](https://mlflow.org/docs/latest/index.html) for details.

Use the environment variable `MLFLOW_TRACKING_URI` to set the URI of the MLFlow server. If not set the logging can not be used. An example of setting the URI to a server is and running a training command is `MLFLOW_TRACKING_URI=http://localhost:5000 python -m neural_lam.train_model --config_path <config_path> --logger mlflow`.

## Train Models
Models can be trained using `python -m neural_lam.train_model --config_path <config_path>`.
Run `python neural_lam.train_model --help` for a full list of training options.
A few of the key ones are outlined below:

* `--config_path`: Path to the configuration for neural-lam (for example in `data/myexperiment/config.yaml`).
* `--model`: Which model to train
* `--graph`: Which graph to use with the model
* `--epochs`: Number of epochs to train for
* `--processor_layers`: Number of GNN layers to use in the processing part of the model
* `--ar_steps_train`: Number of time steps to unroll for when making predictions and computing the loss
* `--ar_steps_eval`: Number of time steps to unroll for during validation steps

Checkpoints of trained models are stored in the `saved_models` directory.
The implemented models are:

### Graph-LAM
This is the basic graph-based LAM model.
The encode-process-decode framework is used with a mesh graph in order to make one-step pedictions.
This model class is used both for the L1-LAM and GC-LAM models from the [paper](#graph-based-neural-weather-prediction-for-limited-area-modeling), only with different graphs.

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

### High Performance Computing

The training script can be run on a cluster with multiple GPU-nodes. Neural LAM is set up to use PyTorch Lightning's `DDP` backend for distributed training.
The code can be used on systems both with and without slurm. If the cluster has multiple nodes, set the `--num_nodes` argument accordingly.

Using SLURM, the job can be started with `sbatch slurm_job.sh` with a shell script like the following.
```
#!/bin/bash -l
#SBATCH --job-name=Neural-LAM
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres:gpu=4
#SBATCH --partition=normal
#SBATCH --mem=444G
#SBATCH --no-requeue
#SBATCH --exclusive
#SBATCH --output=lightning_logs/neurallam_out_%j.log
#SBATCH --error=lightning_logs/neurallam_err_%j.log

# Load necessary modules or activate environment, for example:
conda activate neural-lam

srun -ul python -m neural_lam.train_model \
    --config_path /path/to/config.yaml \
    --num_nodes $SLURM_JOB_NUM_NODES
```

When using on a system without SLURM, where all GPU's are visible, it is possible to select a subset of GPU's to use for training with the `devices` cli argument, e.g. `--devices 0 1` to use the first 2 GPU's.

## Evaluate Models
Evaluation is also done using `python -m neural_lam.train_model --config_path <config-path>`, but using the `--eval` option.
Use `--eval val` to evaluate the model on the validation set and `--eval test` to evaluate on test data.
Most of the training options are also relevant for evaluation.
Some options specifically important for evaluation are:

* `--load`: Path to model checkpoint file (`.ckpt`) to load parameters from
* `--n_example_pred`: Number of example predictions to plot during evaluation.
* `--ar_steps_eval`: Number of time steps to unroll for during evaluation

**Note:** While it is technically possible to use multiple GPUs for running evaluation, this is strongly discouraged. If using multiple devices the `DistributedSampler` will replicate some samples to make sure all devices have the same batch size, meaning that evaluation metrics will be unreliable.
A possible workaround is to just use batch size 1 during evaluation.
This issue stems from PyTorch Lightning. See for example [this PR](https://github.com/Lightning-AI/torchmetrics/pull/1886) for more discussion.

# Repository Structure
Except for training and pre-processing scripts all the source code can be found in the `neural_lam` directory.
Model classes, including abstract base classes, are located in `neural_lam/models`.
Notebooks for visualization and analysis are located in `docs`.

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
If you are interested in machine learning models for LAM, have questions about the implementation or ideas for extending it, feel free to get in touch.
There is an open [mllam slack channel](https://join.slack.com/t/ml-lam/shared_invite/zt-2t112zvm8-Vt6aBvhX7nYa6Kbj_LkCBQ) that anyone can join (after following the link you have to request to join, this is to avoid spam bots).
You can also open a github issue on this page, or (if more suitable) send an email to [joel.oskarsson@liu.se](mailto:joel.oskarsson@liu.se).

# Configuration and Datastores

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
├── era5.datastore.yaml   - Optional second datastore, here providing boundary forcing
└── graph/                - Directory containing graphs for training
```

And the content of `config.yaml` could in this case look like:
```yaml
datastores:
  danra:
    kind: mdp
    config_path: danra.datastore.yaml
  era5:  # optional; no `state` data, so used for input (boundary) only
    kind: mdp
    config_path: era5.datastore.yaml
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

1. A named mapping of datastores (`datastores`), each giving the kind of
   datastore and the path to its config. Which one is the interior and which is
   the boundary follows from the categories each provides, see [Data
   categories](#data-categories) below.
2. The weighting of different features in
the loss function. If you don't define the state feature weighting it will default to
weighting all features equally.
3. Valid numerical range for output of each feature. The numerical range of all features default to $]-\infty, \infty[$.

(This example is taken from the
`tests/datastore_examples/mdp/era5_1000hPa_danra_100m_winds` directory; the
`tests/datastore_examples/mdp/danra_100m_winds` one next to it is the
single-datastore version.)


Below follows instructions for each kind of datastore implemented (see
{doc}`graphs` for the graph generation and {doc}`training` for training and
evaluation). Once `neural-lam` has been installed the
general process is:

1. Run any pre-processing scripts to generate the necessary derived data that your chosen datastore requires
2. Run graph-creation step
3. Train the model

(data-the-datastore-and-weatherdataset-classes)=
## Data (the `DataStore` and `WeatherDataset` classes)

To enable flexibility in what input-data sources can be used with neural-lam,
the input-data representation is split into two parts:

1. A "datastore" (represented by instances of
   [neural_lam.datastore.BaseDatastore](https://github.com/mllam/neural-lam/blob/main/neural_lam/datastore/base.py)) which
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

Each interior `MDPDatastore` still has its own boundary mask - a rim of
`n_boundary_points` grid cells (default 30) excluded from the loss and
overwritten with ground truth in the prediction. This is independent of any
pairing with a separate boundary datastore (below).
Set `datastores.<name>.n_boundary_points` in `config.yaml` to 0 once boundary
forcing comes from elsewhere. If there is no separate datastore with boundary
forcing, `n_boundary_points > 0` can still be used to take boundary input from the
interior state. Using both `n_boundary_points > 0` and a separate boundary datastore
corresponds to taking two kinds of boundary inputs (from interior state and boundary
datastore), and is not recommended unless you have a very specific setup.
`NpyFilesDatastoreMEPS`'s boundary mask is fixed at data-preparation time and
does not support this option.

(data-categories)=
### Data categories

Each variable in a datastore belongs to one of three *categories*, which fix
whether it is fed to the model as input, predicted as output, or both:

| Category  | Model input | Model output | Description |
|-----------|:-----------:|:------------:|-------------|
| `state`   | ✓           | ✓            | Prognostic variables the model both reads and predicts (autoregressed forward in time). |
| `forcing` | ✓           |              | Time-varying inputs known in advance (e.g. solar radiation, boundary forcing). |
| `static`  | ✓           |              | Time-invariant inputs (e.g. orography, land-sea mask). |

The categories also fix each datastore's role. A datastore with `state` data is the interior domain; one without is input-only, e.g. ERA5 boundary forcing for a LAM domain, whose forcing is windowed into an extra tensor per training sample. Exactly one interior datastore is required and at most one boundary datastore is supported.

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
subclassing the `neural_lam.datastore.BaseDatastore` class and implementing the abstract methods.

Datastores whose grid points form a complete 2D `x`/`y` rectilinear grid (the
spacing need not be uniform) set `is_on_rectilinear_grid` to `True` and
implement `grid_shape_state`. This can depend on the data, e.g. a
domain-cropped `MDPDatastore` is not on a rectilinear grid.


### MDP (mllam-data-prep) Datastore - `MDPDatastore`

With `MDPDatastore` (the mllam-data-prep datastore) all the selection,
transformation and pre-calculation steps that are needed to go from
for example gridded weather data to a format that is optimised for training
in neural-lam, are done in a separate package called
[mllam-data-prep](https://github.com/mllam/mllam-data-prep) rather than in
neural-lam itself.
Specifically, the `mllam-data-prep` datastore configuration (for example
[danra.datastore.yaml](https://github.com/mllam/neural-lam/blob/main/tests/datastore_examples/mdp/danra_100m_winds/danra.datastore.yaml))
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

The full MEPS dataset is available for download [here](https://nextcloud.liu.se/s/meps). See instructions in the file `README.txt` for how to download and extract the full dataset.
Note that while this data works with the `NpyFilesDatastoreMEPS` datastore, the exact data loading in this setup differs somewhat from that used in earlier published papers using the dataset.
For research reproducibility, see the exact corresponding branch for each publication (under [Publications](https://github.com/mllam/neural-lam#publications) in the README).

A tiny subset of the MEPS data (named `meps_example`) is also available in `example_data.zip`, which can be downloaded from [here](https://drive.google.com/drive/folders/1N6ZT_mkfbdVloVsNs9T5YOrMtxd3jG-j?usp=sharing).
Download the files and unzip in the neural-lam directory.
Graphs used in the initial paper are also available for download at the same link (but can as easily be re-generated using `python -m neural_lam.create_graph`).
Note that this subset is far too small to train any useful models, but all scripts can be run with it.
It should thus be useful to make sure that your python environment is set up correctly and that all the code can be run without any issues.

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
datastores:
  meps:
    kind: npyfilesmeps
    config_path: meps.datastore.yaml
```

For npy-file based datastores you must separately run the command that creates the variables used for standardization:

```bash
python -m neural_lam.datastore.npyfilesmeps.compute_standardization_stats <path-to-datastore-config>
```

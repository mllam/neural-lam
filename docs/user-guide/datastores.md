# Configuration and Datastores

## Configuration file

Every Neural-LAM command takes a `--config_path` argument pointing to a YAML
configuration file (typically `config.yaml`). The directory containing this
file becomes the root directory — all other paths (datastore config, graphs)
are resolved relative to it.

```
data/
├── config.yaml           - Neural-LAM configuration
├── danra.datastore.yaml  - Datastore configuration (referred to from config.yaml)
└── graphs/               - Generated graph files
```

A minimal `config.yaml` looks like this:

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

The config defines:

1. The kind of datastore and the path to its config.
2. The weighting of different features in the loss function (defaults to equal weighting).
3. Valid numerical ranges for each output feature (defaults to no clamping).

For the full configuration reference see {py:class}`neural_lam.config.NeuralLAMConfig`.

## Datastores

A datastore loads data from disk in a specific format and presents it through
a common interface that Neural-LAM uses for training and evaluation. There are
currently two datastores implemented.

### MDP datastore — `MDPDatastore`

The recommended datastore. Uses training-ready datasets in zarr format produced
by [mllam-data-prep](https://github.com/mllam/mllam-data-prep), which can
combine any number of xarray-readable sources and handles variable selection,
dimension flattening, and normalisation statistics.

Set `kind: mdp` in `config.yaml` and point `config_path` at a
`mllam-data-prep` configuration file (e.g. `danra.datastore.yaml`). Neural-LAM
writes the processed dataset to a `.zarr` file alongside that config on first
run. To pre-process a large dataset (≥10 GB) before training, run:

```bash
python -m mllam_data_prep --config data/danra.datastore.yaml
```

For parallel processing across multiple CPU cores:

```bash
python -m mllam_data_prep --config data/danra.datastore.yaml \
    --dask-distributed-local-core-fraction 0.5
```

See the [mllam-data-prep README](https://github.com/mllam/mllam-data-prep) for
full configuration details.

### NpyFiles MEPS datastore — `NpyFilesDatastoreMEPS`

Reads MEPS data from `.npy` files in the format introduced in Neural-LAM
`v0.1.0`. Set `kind: npyfilesmeps` in `config.yaml`.

The full MEPS dataset is available for [download here](https://nextcloud.liu.se/s/meps).
A small example subset (`meps_example`) is available in `example_data.zip`
from [Google Drive](https://drive.google.com/drive/folders/1N6ZT_mkfbdVloVsNs9T5YOrMtxd3jG-j?usp=sharing)
— download and unzip in the neural-lam directory. This subset is too small to
train useful models but is sufficient to verify your environment.

Before training you must generate normalisation statistics:

```bash
python -m neural_lam.datastore.npyfilesmeps.compute_standardization_stats \
    <path-to-datastore-config>
```

Example `meps.datastore.yaml` and corresponding `config.yaml`:

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
  - W/m²
  - W/m²
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
  - kg/m²
  - m²/s²
  - m²/s²
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

```yaml
# config.yaml
datastore:
  kind: npyfilesmeps
  config_path: meps.datastore.yaml
training:
  state_feature_weighting:
    __config_class__: ManualStateFeatureWeighting
    weights:
      u100m: 1.0
      v100m: 1.0
```

### Custom datastores

Implement your own datastore by subclassing
{py:class}`neural_lam.datastore.BaseDataStore` (or
{py:class}`neural_lam.datastore.BaseRegularGridDatastore` for regular grids)
and implementing the abstract methods. See
[`neural_lam/datastore/base.py`](https://github.com/mllam/neural-lam/blob/main/neural_lam/datastore/base.py)
for the full interface.

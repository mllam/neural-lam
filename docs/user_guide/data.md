# Data: DataStores and WeatherDataset

Neural-LAM uses a two-part data representation for flexibility:

1. **DataStore** — loads data from disk and returns `xarray.DataArray` objects
2. **WeatherDataset** — a PyTorch `Dataset` that samples in time, normalizes, and produces tensors

## DataStore Architecture

A datastore (subclass of {class}`~neural_lam.datastore.BaseDatastore`) handles loading
a given category (`state`, `forcing`, or `static`) and split (`train`/`val`/`test`)
of data. The returned data-array has:

- Spatial coordinates flattened into a single `grid_index` dimension
- All variables and vertical levels stacked into a `{category}_feature` dimension

The datastore also provides:
- Variable names, long names, and units
- Boundary mask
- Normalization statistics (mean and std)
- Grid coordinate information

## Available DataStores

### MDPDatastore (`mllam-data-prep`)

{class}`~neural_lam.datastore.MDPDatastore` loads *training-ready* datasets in zarr
format created with the [mllam-data-prep](https://github.com/mllam/mllam-data-prep)
package.

The `mllam-data-prep` configuration specifies:
- What source datasets to read
- What variables to select
- What dimension/variable transformations to make
- What statistics to calculate for normalization
- How to split data into train/val/test

**Running preprocessing:**

```bash
# Basic usage
python -m mllam_data_prep --config data/danra.datastore.yaml

# Parallel processing for large datasets (≥10GB)
python -m mllam_data_prep --config data/danra.datastore.yaml \
    --dask-distributed-local-core-fraction 0.5
```

**Datastore config example** (`danra.datastore.yaml`):

```yaml
datastore:
  kind: mdp
  config_path: danra.datastore.yaml
```

### NpyFilesDatastoreMEPS

{class}`~neural_lam.datastore.NpyFilesDatastoreMEPS` reads MEPS data from `.npy` files
in the format from neural-lam `v0.1.0`.

:::{note}
This datastore is specific to the MEPS dataset format but can serve as an example
for similar numpy-based datastores.
:::

The full MEPS dataset is available [here](https://nextcloud.liu.se/s/meps). A tiny
subset (`meps_example`) can be downloaded from
[Google Drive](https://drive.google.com/drive/folders/1N6ZT_mkfbdVloVsNs9T5YOrMtxd3jG-j?usp=sharing).

**Standardization (required for npy-file datastores):**

```bash
python -m neural_lam.datastore.npyfilesmeps.compute_standardization_stats <path-to-datastore-config>
```

### Custom DataStores

Create your own by subclassing:

- {class}`~neural_lam.datastore.BaseDatastore` — for any data format
- {class}`~neural_lam.datastore.BaseRegularGridDatastore` — if your data is on a regular grid

## WeatherDataset

The {class}`~neural_lam.weather_dataset.WeatherDataset` class takes a datastore and
handles:

- Temporal sampling to create individual training samples
- Normalization of values
- Conversion to `torch.Tensor` objects

The companion {class}`~neural_lam.weather_dataset.WeatherDataModule` (PyTorch Lightning
`LightningDataModule`) manages train/val/test dataloaders.

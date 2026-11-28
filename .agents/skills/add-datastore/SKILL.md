---
name: add-datastore
description: Implement, register, and test a new Datastore backend using BaseDatastore and DATASTORES.
---

# Add a Datastore Backend

Follow this recipe to implement a new datastore backend in Neural-LAM, register it in `neural_lam.datastore.DATASTORES`, and wire it into the datastore test suite.

## 1. Inherit from `BaseDatastore`
Implement your datastore class in `neural_lam/datastore/<name>.py` subclassing `neural_lam.datastore.base.BaseDatastore`.

### Conventions & Interface Contract
Implement every abstract member of `BaseDatastore`, whose docstrings define the return types and dimension orders, and set a `SHORT_NAME` class attribute (the `kind` in the config). Easy to get wrong:
- `root_path` is a directory, graphs are written to `root_path / "graph" / <name>`.
- `get_xy(category, stacked)` returns one array, `(N_x * N_y, 2)` when stacked.
- `get_dataarray(category, split, standardize)` returns the dimensions in `expected_dim_order(category)`, e.g. `(time, grid_index, state_feature)`, or `None` for a category the dataset does not have.
- `get_standardization_dataarray(category)` returns an `xr.Dataset` with `{category}_mean` and `{category}_std`, plus `state_diff_mean_standardized` and `state_diff_std_standardized` for `state`.
- `boundary_mask` is an `xr.DataArray` on `grid_index`, 1 for boundary and 0 for interior points.
- Graph creation needs `is_on_rectilinear_grid = True`, set it only if `grid_index` unstacks into a complete x/y grid.

### Reference Implementations
Inspect these implementations as canonical references:
- **Smallest in-memory reference**: `tests.dummy_datastore.DummyDatastore` in `tests/dummy_datastore.py`.
- **Production Zarr reference**: `neural_lam.datastore.mdp.MDPDatastore` in `neural_lam/datastore/mdp.py`.
- **Base class and abstract methods**: `neural_lam.datastore.base.BaseDatastore` in `neural_lam/datastore/base.py`.

## 2. Register in `DATASTORE_CLASSES`
Register the class in `neural_lam/datastore/__init__.py`:
- Add the class to `DATASTORE_CLASSES`.
- `DATASTORES` automatically indexes instances by their `SHORT_NAME`.

## 3. Extend Datastore Tests
The tests in `tests/test_datastores.py` and the dataset, graph and training tests are already parametrised over `DATASTORES`, so they pick up the new datastore after step 2. Add `<short_name>=<path to an example config>` to `DATASTORES_EXAMPLES` in `tests/conftest.py` (else they fail with a `KeyError`), and the short name to `REAL_DATA_DATASTORE_KINDS` there if the example data is downloaded.

## 4. Verification Check
Run every test parametrised over the new datastore:
```bash
uv run pytest tests -k <short_name>
```

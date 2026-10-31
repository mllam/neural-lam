---
name: add-datastore
description: Implement, register, and test a new Datastore backend using BaseDatastore and DATASTORES.
---

# Add a Datastore Backend

Follow this recipe to implement a new datastore backend in Neural-LAM, register it in `neural_lam.datastore.DATASTORES`, and wire it into the datastore test suite.

## 1. Inherit from `BaseDatastore`
Implement your datastore class in `neural_lam/datastore/<name>.py` subclassing `neural_lam.datastore.base.BaseDatastore`.

### Conventions & Interface Contract
Subclasses must implement:
- `SHORT_NAME`: string identifier used in CLI and configs (e.g. `DATASTORES[SHORT_NAME]`).
- `root_path`: `pathlib.Path` pointing to the dataset root.
- `step_length`: `datetime.timedelta` representing forecast timestep interval.
- `grid_shape_state`: grid dimensions (e.g. `CartesianGridShape`).
- `coords_projection`: `cartopy.crs.Projection` for geographic projections.
- `get_xy(category)`: returns `(x, y)` coordinate arrays for `"state"` and `"forcing"`.
- `get_vars_names(category)`: returns list of variable names.
- `get_vars_units(category)`: returns list of variable unit strings.
- `get_dataarray(category, split, standardize=False)`: returns `xarray.DataArray` with dimensions `(time, grid_index, var)`.

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
Add test cases in `tests/test_datastores.py`:
- Test property accessors (`step_length`, `grid_shape_state`, `coords_projection`).
- Test array shapes and variable name matching via `DummyDatastore` or test fixtures.

## 4. Verification Check
Run datastore unit tests:
```bash
uv run pytest tests/test_datastores.py
```

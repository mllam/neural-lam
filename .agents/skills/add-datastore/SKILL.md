---
name: add-datastore
description: Implement, register, and test a new Datastore backend in Neural-LAM. Use when adding support for a new data format or dataset source.
---

# Add a Datastore Backend

Follow these instructions to implement a new `BaseDatastore` subclass, register it in `neural_lam.datastore`, and add it to the test parametrization.

## 1. Inherit from `BaseDatastore`
Implement a new datastore class in `neural_lam/datastore/<name>.py` extending `neural_lam.datastore.base.BaseDatastore`:

```python
from datetime import timedelta
from pathlib import Path
import cartopy.crs as ccrs
import numpy as np
import torch
import xarray as xr
from neural_lam.datastore.base import BaseDatastore, CartesianGridShape


class MyCustomDatastore(BaseDatastore):
    """Datastore backend for custom weather dataset."""

    SHORT_NAME = "custom_backend"
    spatial_coordinates = ("x", "y")
    is_on_rectilinear_grid = True

    def __init__(self, config_path: str | Path | None = None) -> None:
        super().__init__()
        self._config_path = Path(config_path) if config_path else Path(".")

    @property
    def root_path(self) -> Path:
        return self._config_path

    @property
    def step_length(self) -> timedelta:
        return timedelta(hours=1)

    @property
    def grid_shape_state(self) -> CartesianGridShape:
        return CartesianGridShape(x=100, y=100)

    @property
    def coords_projection(self) -> ccrs.Projection:
        return ccrs.PlateCarree()

    @property
    def boundary_mask(self) -> np.ndarray:
        # 1 for interior, 0 for boundary
        mask = np.ones(10000, dtype=bool)
        return mask

    @property
    def config(self):
        return None

    def get_xy(self, category: str = "state") -> tuple[np.ndarray, np.ndarray]:
        x = np.linspace(0, 10, 100)
        y = np.linspace(0, 10, 100)
        return x, y

    def get_vars_names(self, category: str = "state") -> list[str]:
        if category == "state":
            return ["temperature", "humidity"]
        return []

    def get_vars_units(self, category: str = "state") -> list[str]:
        return ["K", "kg/kg"] if category == "state" else []

    def get_vars_long_names(self, category: str = "state") -> list[str]:
        return self.get_vars_names(category)

    def get_num_data_vars(self, category: str = "state") -> int:
        return len(self.get_vars_names(category))

    def get_normalization_dataarray(
        self, category: str = "state"
    ) -> xr.DataArray:
        names = self.get_vars_names(category)
        return xr.DataArray(
            np.ones((2, len(names))),
            dims=["stat", "var"],
            coords={"stat": ["mean", "std"], "var": names},
        )

    def get_dataarray(
        self, category: str, split: str, standardize: bool = False
    ) -> xr.DataArray:
        # Return xarray with dims (time, grid_index, var)
        pass
```

## 2. Register in Datastore Registry
Add your datastore class to `DATASTORE_CLASSES` in `neural_lam/datastore/__init__.py`:

```python
from .my_custom import MyCustomDatastore

DATASTORE_CLASSES = [
    MDPDatastore,
    NpyFilesDatastoreMEPS,
    MyCustomDatastore,
]

DATASTORES = {
    datastore.SHORT_NAME: datastore for datastore in DATASTORE_CLASSES
}
```

## 3. Add to Test Suite
Add test parametrization or fixtures in `tests/test_datastores.py`:
- Verify required property types (`step_length`, `grid_shape_state`, `coords_projection`).
- Verify array shapes and variable naming conventions.

## 4. Verification Check
Run datastore unit tests:

```bash
uv run pytest tests/test_datastores.py -k custom_backend
```

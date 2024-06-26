from typing import List

from numpy import ndarray

from .base import BaseCartesianDatastore, CartesianGridShape

import mllam_data_prep as mdp
import xarray as xr
import cartopy.crs as ccrs


class MLLAMDatastore(BaseCartesianDatastore):
    """
    Datastore class for the MLLAM dataset.
    """

    def __init__(self, config_path, n_boundary_points=30):
        self._config_path = config_path
        self._config = mdp.Config.from_yaml_file(config_path)
        self._ds = mdp.create_dataset(config=self._config)
        self._n_boundary_points = n_boundary_points
        
    def step_length(self) -> int:
        da_dt = self._ds["time"].diff("time")
        return da_dt.dt.seconds[0] // 3600
    
    def get_vars_units(self, category: str) -> List[str]:
        return self._ds[f"{category}_unit"].values.tolist()
    
    def get_vars_names(self, category: str) -> List[str]:
        return self._ds[f"{category}_longname"].values.tolist()    
    
    def get_num_data_vars(self, category: str) -> int:
        return len(self._ds[category].data_vars)
    
    def get_dataarray(self, category: str, split: str) -> xr.DataArray:
        # TODO: Implement split handling in mllam-data-prep, for now we hardcode that
        # train will be the first 80%, then validation 10% and test 10%
        da_category = self._ds[category]
        n_samples = len(da_category.time)
        # compute the split indices
        if split == "train":
            i_start, i_end = 0, int(0.8 * n_samples)
        elif split == "val":
            i_start, i_end = int(0.8 * n_samples), int(0.9 * n_samples)
        elif split == "test":
            i_start, i_end = int(0.9 * n_samples), n_samples
        else:
            raise ValueError(f"Unknown split {split}")
        
        da_split = da_category.isel(time=slice(i_start, i_end))
        return da_split
    
    @property
    def boundary_mask(self) -> xr.DataArray:
        da_mask = xr.ones_like(self._ds["state"].isel(time=0).isel(variable=0))
        da_mask.isel(x=slice(0, self._n_boundary_points), y=slice(0, self._n_boundary_points)).values = 0
        return da_mask
    
    @property
    def coords_projection(self) -> ccrs.Projection:
        # TODO: danra doesn't contain projection information yet, but the next version wil
        # for now we hardcode the projection
        # XXX: this is wrong
        return ccrs.PlateCarree()
    
    @property
    def grid_shape_state(self):
        return CartesianGridShape(
            x=self._ds["state"].x.size, y=self._ds["state"].y.size
        )
        
    def get_xy(self, category: str, stacked: bool) -> ndarray:
        da_x = self._ds[category].x
        da_y = self._ds[category].y
        if stacked:
            x, y = xr.broadcast(da_x, da_y)
            return xr.concat([x, y], dim="xy").values
        else:
            return da_x.values, da_y.values
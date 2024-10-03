# Standard library
import datetime
import tempfile
from pathlib import Path
from typing import List, Union

# Third-party
import isodate
import numpy as np
import xarray as xr
from cartopy import crs as ccrs

# First-party
from neural_lam.datastore.base import BaseDatastore


class DummyDatastore(BaseDatastore):
    """
    Dummy datastore for testing purposes. It generates random data for a
    irregular grid with a random boundary mask.
    """

    SHORT_NAME = "dummydata"
    T0 = isodate.parse_datetime("2021-01-01T00:00:00")
    N_FEATURES = dict(state=5, forcing=2, static=1)

    def __init__(
        self, config_path=None, n_grid_points=100, n_timesteps=10
    ) -> None:
        assert (
            config_path is None
        ), "No config file is needed for the dummy datastore"

        self.da_grid = xr.DataArray(
            np.random.rand(n_grid_points),
            dims=("grid_index"),
            coords={
                "grid_index": range(n_grid_points),
            },
        )
        self._num_grid_points = n_grid_points

        dt = datetime.timedelta(hours=self.step_length)
        times = [self.T0 + dt * i for i in range(n_timesteps)]
        self.ds = xr.Dataset(
            coords={"grid_index": range(n_grid_points), "time": times}
        )

        for category, n in self.N_FEATURES.items():
            dims = ["grid_index", f"{category}_feature"]
            shape = [n_grid_points, n]
            if category != "static":
                dims.append("time")
                shape.append(n_timesteps)

            self.ds[category] = xr.DataArray(
                np.random.rand(*shape),
                dims=dims,
                coords={
                    f"{category}_feature": [
                        f"{category}_feat_{i}" for i in range(n)
                    ],
                },
            )

        # add units and long_name to the features
        for category, n in self.N_FEATURES.items():
            self.ds[f"{category}_feature_units"] = ["1"] * n
            self.ds[f"{category}_feature_long_name"] = [
                f"{category} feature {i}" for i in range(n)
            ]

        # pick a random grid point as the boundary
        self.ds["boundary_mask"] = xr.DataArray(
            np.random.choice([0, 1], n_grid_points),
            dims=("grid_index",),
            coords={
                "grid_index": range(n_grid_points),
            },
        )

        # create some lat/lon coordinates randomly sampled around Denmark
        lat = np.random.uniform(54, 58, n_grid_points)
        lon = np.random.uniform(8, 13, n_grid_points)
        self.ds.coords["lat"] = xr.DataArray(
            lat,
            dims=("grid_index",),
            coords={
                "grid_index": range(n_grid_points),
            },
        )
        self.ds.coords["lon"] = xr.DataArray(
            lon,
            dims=("grid_index",),
            coords={
                "grid_index": range(n_grid_points),
            },
        )

        # project the lat/lon coordinates to x/y using the projection
        coords = self.coords_projection.transform_points(
            src_crs=ccrs.PlateCarree(), x=lon, y=lat
        )
        x = coords[:, 0]
        y = coords[:, 1]
        self.ds.coords["x"] = xr.DataArray(
            x,
            dims=("grid_index",),
            coords={
                "grid_index": range(n_grid_points),
            },
        )
        self.ds.coords["y"] = xr.DataArray(
            y,
            dims=("grid_index",),
            coords={
                "grid_index": range(n_grid_points),
            },
        )

        # create a temporary path for the datastore so that graphs etc can be
        # stored with it
        self._tempdir = tempfile.TemporaryDirectory()
        self._root_path = Path(self._tempdir.name)

    @property
    def root_path(self) -> Path:
        """
        The root path to the datastore. It is relative to this that any derived
        files (for example the graph components) are stored.

        Returns
        -------
        pathlib.Path
            The root path to the datastore.

        """
        return self._root_path

    @property
    def config(self) -> dict:
        """The configuration of the datastore.

        Returns
        -------
        collections.abc.Mapping
            The configuration of the datastore, any dict like object can be
            returned.

        """
        return {}

    @property
    def step_length(self) -> int:
        """The step length of the dataset in hours.

        Returns:
            int: The step length in hours.

        """
        return 1

    def get_vars_names(self, category: str) -> list[str]:
        """Get the names of the variables in the given category.

        Parameters
        ----------
        category : str
            The category of the variables (state/forcing/static).

        Returns
        -------
        List[str]
            The names of the variables.

        """
        return self.ds[f"{category}_feature"].values.tolist()

    def get_vars_units(self, category: str) -> list[str]:
        """Get the units of the variables in the given category.

        Parameters
        ----------
        category : str
            The category of the variables (state/forcing/static).

        Returns
        -------
        List[str]
            The units of the variables.

        """
        return self.ds[f"{category}_feature_units"].values.tolist()

    def get_vars_long_names(self, category: str) -> List[str]:
        """Get the long names of the variables in the given category.

        Parameters
        ----------
        category : str
            The category of the variables (state/forcing/static).

        Returns
        -------
        List[str]
            The long names of the variables.

        """
        return self.ds[f"{category}_feature_long_name"].values.tolist()

    def get_num_data_vars(self, category: str) -> int:
        """Get the number of data variables in the given category.

        Parameters
        ----------
        category : str
            The category of the variables (state/forcing/static).

        Returns
        -------
        int
            The number of data variables.

        """
        return self.ds[f"{category}_feature"].size

    def get_standardization_dataarray(self, category: str) -> xr.Dataset:
        """
        Return the standardization (i.e. scaling to mean of 0.0 and standard
        deviation of 1.0) dataarray for the given category. This should contain
        a `{category}_mean` and `{category}_std` variable for each variable in
        the category. For `category=="state"`, the dataarray should also
        contain a `state_diff_mean` and `state_diff_std` variable for the one-
        step differences of the state variables. The returned dataarray should
        at least have dimensions of `({category}_feature)`, but can also
        include for example `grid_index` (if the standardization is done per
        grid point for example).

        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).

        Returns
        -------
        xr.Dataset
            The standardization dataarray for the given category, with variables
            for the mean and standard deviation of the variables (and
            differences for state variables).

        """
        ds_standardization = xr.Dataset()

        ops = ["mean", "std"]
        if category == "state":
            ops += ["diff_mean", "diff_std"]

        for op in ops:
            da_op = xr.ones_like(self.ds[f"{category}_feature"]).astype(float)
            ds_standardization[f"{category}_{op}"] = da_op

        return ds_standardization

    def get_dataarray(
        self, category: str, split: str
    ) -> Union[xr.DataArray, None]:
        """
        Return the processed data (as a single `xr.DataArray`) for the given
        category of data and test/train/val-split that covers all the data (in
        space and time) of a given category (state/forcing/static). A
        datastore must be able to return for the "state" category, but
        "forcing" and "static" are optional (in which case the method should
        return `None`). For the "static" category the `split` is allowed to be
        `None` because the static data is the same for all splits.

        The returned dataarray is expected to at minimum have dimensions of
        `(grid_index, {category}_feature)` so that any spatial dimensions have
        been stacked into a single dimension and all variables and levels have
        been stacked into a single feature dimension named by the `category` of
        data being loaded.

        For categories of data that have a time dimension (i.e. not static
        data), the dataarray is expected additionally have `(analysis_time,
        elapsed_forecast_duration)` dimensions if `is_forecast` is True, or
        `(time)` if `is_forecast` is False.

        If the data is ensemble data, the dataarray is expected to have an
        additional `ensemble_member` dimension.

        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).
        split : str
            The time split to filter the dataset (train/val/test).

        Returns
        -------
        xr.DataArray or None
            The xarray DataArray object with processed dataset.

        """
        return self.ds[category]

    @property
    def boundary_mask(self) -> xr.DataArray:
        """
        Return the boundary mask for the dataset, with spatial dimensions
        stacked. Where the value is 1, the grid point is a boundary point, and
        where the value is 0, the grid point is not a boundary point.

        Returns
        -------
        xr.DataArray
            The boundary mask for the dataset, with dimensions
            `('grid_index',)`.

        """
        return self.ds["boundary_mask"]

    def get_xy(self, category: str) -> np.ndarray:
        """
        Return the x, y coordinates of the dataset as a numpy arrays for a
        given category of data.

        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).

        Returns
        -------
        np.ndarray
            The x, y coordinates of the dataset with shape `[2, n_grid_points]`.
        """
        return np.stack([self.ds["x"].values, self.ds["y"].values], axis=0)

    @property
    def coords_projection(self) -> ccrs.Projection:
        """Return the projection object for the coordinates.

        The projection object is used to plot the coordinates on a map.

        Returns
        -------
        cartopy.crs.Projection:
            The projection object.

        """
        # make a projection centered on Denmark
        return ccrs.LambertAzimuthalEqualArea(
            central_latitude=56, central_longitude=10
        )

    @property
    def num_grid_points(self) -> int:
        """Return the number of grid points in the dataset.

        Returns
        -------
        int
            The number of grid points in the dataset.

        """
        return self._num_grid_points

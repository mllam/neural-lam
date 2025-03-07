# Standard library
import datetime
import tempfile
from functools import cached_property
from pathlib import Path
from typing import List, Union

# Third-party
import isodate
import numpy as np
import xarray as xr
from cartopy import crs as ccrs
from numpy import ndarray

# First-party
from neural_lam.datastore.base import (
    BaseRegularGridDatastore,
    CartesianGridShape,
)


class DummyDatastore(BaseRegularGridDatastore):
    """
    Datastore that creates some dummy data for testing purposes. The data
    consists of state, forcing, and static variables, and is stored in a
    regular grid (using Lambert Azimuthal Equal Area projection). The domain
    is centered on Denmark and has a size of 500x500 km.
    """

    SHORT_NAME = "dummydata"
    T0 = isodate.parse_datetime("2021-01-01T00:00:00")
    N_FEATURES = dict(state=5, forcing=2, static=1)
    CARTESIAN_COORDS = ["x", "y"]

    # center the domain on Denmark
    latlon_center = [56, 10]  # latitude, longitude
    bbox_size_km = [500, 500]  # km

    def __init__(
        self, config_path=None, n_grid_points=10000, n_timesteps=10
    ) -> None:
        """
        Create a dummy datastore with random data.

        Parameters
        ----------
        config_path : None
            No config file is needed for the dummy datastore. This argument is
            only present to match the signature of the other datastores.
        n_grid_points : int
            The number of grid points in the dataset. Must be a perfect square.
        n_timesteps : int
            The number of timesteps in the dataset.
        """
        assert (
            config_path is None
        ), "No config file is needed for the dummy datastore"

        # Ensure n_grid_points is a perfect square
        n_points_1d = int(np.sqrt(n_grid_points))
        assert (
            n_points_1d * n_points_1d == n_grid_points
        ), "n_grid_points must be a perfect square"

        # create equal area grid
        lx, ly = self.bbox_size_km
        x = np.linspace(-lx / 2.0 * 1.0e3, lx / 2.0 * 1.0e3, n_points_1d)
        y = np.linspace(-ly / 2.0 * 1.0e3, ly / 2.0 * 1.0e3, n_points_1d)

        xs, ys = np.meshgrid(x, y)

        # Create lat/lon coordinates using equal area projection
        lon_mesh, lat_mesh = (
            ccrs.PlateCarree()
            .transform_points(
                src_crs=self.coords_projection,
                x=xs.flatten(),
                y=ys.flatten(),
            )[:, :2]
            .T
        )

        # Create base dataset with proper coordinates
        self.ds = xr.Dataset(
            coords={
                "x": (
                    "x",
                    x,
                    {"units": "m"},
                ),  # Use first column for x coordinates
                "y": (
                    "y",
                    y,
                    {"units": "m"},
                ),  # Use first row for y coordinates
                "longitude": (
                    "grid_index",
                    lon_mesh.flatten(),
                    {"units": "degrees_east"},
                ),
                "latitude": (
                    "grid_index",
                    lat_mesh.flatten(),
                    {"units": "degrees_north"},
                ),
            }
        )
        # Create data variables with proper dimensions
        for category, n in self.N_FEATURES.items():
            feature_names = [f"{category}_feat_{i}" for i in range(n)]
            feature_units = ["-" for _ in range(n)]  # Placeholder units
            feature_long_names = [
                f"Long name for {name}" for name in feature_names
            ]

            self.ds[f"{category}_feature"] = feature_names
            self.ds[f"{category}_feature_units"] = (
                f"{category}_feature",
                feature_units,
            )
            self.ds[f"{category}_feature_long_name"] = (
                f"{category}_feature",
                feature_long_names,
            )

            # Define dimensions and create random data
            dims = ["grid_index", f"{category}_feature"]
            if category != "static":
                dims.append("time")
                shape = (n_grid_points, n, n_timesteps)
            else:
                shape = (n_grid_points, n)

            # Create random data
            data = np.random.randn(*shape)

            # Create DataArray with proper dimensions
            self.ds[category] = xr.DataArray(
                data,
                dims=dims,
                coords={
                    f"{category}_feature": feature_names,
                },
            )

            if category != "static":
                dt = datetime.timedelta(hours=self.step_length)
                times = [self.T0 + dt * i for i in range(n_timesteps)]
                self.ds.coords["time"] = times

        # Add boundary mask
        self.ds["boundary_mask"] = xr.DataArray(
            np.random.choice([0, 1], size=(n_points_1d, n_points_1d)),
            dims=["x", "y"],
        )

        # Stack the spatial dimensions into grid_index
        self.ds = self.ds.stack(grid_index=self.CARTESIAN_COORDS)

        # Create temporary directory for storing derived files
        self._tempdir = tempfile.TemporaryDirectory()
        self._root_path = Path(self._tempdir.name)
        self._num_grid_points = n_grid_points

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
        the category.
        For `category=="state"`, the dataarray should also contain a
        `state_diff_mean_standardized` and `state_diff_std_standardized`
        variable for the one-step differences of the state variables.
        The returned dataarray should at least have dimensions of
        `({category}_feature)`, but can also include for example `grid_index`
        (if the standardization is done per grid point for example).

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
            ops += ["diff_mean_standardized", "diff_std_standardized"]

        for op in ops:
            da_op = xr.ones_like(self.ds[f"{category}_feature"]).astype(float)
            ds_standardization[f"{category}_{op}"] = da_op

        return ds_standardization

    def get_dataarray(
        self, category: str, split: str, standardize: bool = False
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
        standardize: bool
            If the dataarray should be returned standardized

        Returns
        -------
        xr.DataArray or None
            The xarray DataArray object with processed dataset.

        """
        dim_order = self.expected_dim_order(category=category)

        da_category = self.ds[category].transpose(*dim_order)

        if standardize:
            return self._standardize_datarray(da_category, category=category)

        return da_category

    @cached_property
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

    def get_xy(self, category: str, stacked: bool) -> ndarray:
        """Return the x, y coordinates of the dataset.

        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).
        stacked : bool
            Whether to stack the x, y coordinates.

        Returns
        -------
        np.ndarray
            The x, y coordinates of the dataset, returned differently based on
            the value of `stacked`:
            - `stacked==True`: shape `(n_grid_points, 2)` where
                               n_grid_points=N_x*N_y.
            - `stacked==False`: shape `(N_x, N_y, 2)`

        """
        # assume variables are stored in dimensions [grid_index, ...]
        ds_category = self.unstack_grid_coords(da_or_ds=self.ds[category])

        da_xs = ds_category.x
        da_ys = ds_category.y

        assert da_xs.ndim == da_ys.ndim == 1, "x and y coordinates must be 1D"

        da_x, da_y = xr.broadcast(da_xs, da_ys)
        da_xy = xr.concat([da_x, da_y], dim="grid_coord")

        if stacked:
            da_xy = da_xy.stack(grid_index=self.CARTESIAN_COORDS).transpose(
                "grid_index",
                "grid_coord",
            )
        else:
            dims = [
                "x",
                "y",
                "grid_coord",
            ]
            da_xy = da_xy.transpose(*dims)

        return da_xy.values

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
        lat_center, lon_center = self.latlon_center
        return ccrs.LambertAzimuthalEqualArea(
            central_latitude=lat_center, central_longitude=lon_center
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

    @cached_property
    def grid_shape_state(self) -> CartesianGridShape:
        """The shape of the grid for the state variables.

        Returns
        -------
        CartesianGridShape:
            The shape of the grid for the state variables, which has `x` and
            `y` attributes.
        """

        n_points_1d = int(np.sqrt(self.num_grid_points))
        return CartesianGridShape(x=n_points_1d, y=n_points_1d)

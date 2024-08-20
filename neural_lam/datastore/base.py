# Standard library
import abc
import collections
import dataclasses
from pathlib import Path
from typing import List, Union

# Third-party
import cartopy.crs as ccrs
import numpy as np
import xarray as xr


class BaseDatastore(abc.ABC):
    """Base class for weather data used in the neural- lam package. A datastore defines
    the interface for accessing weather data by providing methods to access the data in
    a processed format that can be used for training and evaluation of neural networks.

    NOTE: All methods return either primitive types, `numpy.ndarray`,
    `xarray.DataArray` or `xarray.Dataset` objects, not `pytorch.Tensor`
    objects. Conversion to `pytorch.Tensor` objects should be done in the
    `weather_dataset.WeatherDataset` class (which inherits from
    `torch.utils.data.Dataset` and uses the datastore to access the data).

    # Forecast vs analysis data
    If the datastore is used represent forecast rather than analysis data, then
    the `is_forecast` attribute should be set to True, and returned data from
    `get_dataarray` is assumed to have `analysis_time` and `forecast_time` dimensions
    (rather than just `time`).

    # Ensemble vs deterministic data
    If the datastore is used to represent ensemble data, then the `is_ensemble`
    attribute should be set to True, and returned data from `get_dataarray` is
    assumed to have an `ensemble_member` dimension.

    """

    is_ensemble: bool = False
    is_forecast: bool = False

    @property
    @abc.abstractmethod
    def root_path(self) -> Path:
        """The root path to the datastore. It is relative to this that any derived files
        (for example the graph components) are stored.

        Returns
        -------
        pathlib.Path
            The root path to the datastore.

        """
        pass

    @property
    @abc.abstractmethod
    def config(self) -> collections.abc.Mapping:
        """The configuration of the datastore.

        Returns
        -------
        collections.abc.Mapping
            The configuration of the datastore, any dict like object can be returned.

        """
        pass

    @property
    @abc.abstractmethod
    def step_length(self) -> int:
        """The step length of the dataset in hours.

        Returns:
            int: The step length in hours.

        """
        pass

    @abc.abstractmethod
    def get_vars_units(self, category: str) -> List[str]:
        """Get the units of the variables in the given category.

        Parameters
        ----------
        category : str
            The category of the variables.

        Returns
        -------
        List[str]
            The units of the variables.

        """
        pass

    @abc.abstractmethod
    def get_vars_names(self, category: str) -> List[str]:
        """Get the names of the variables in the given category.

        Parameters
        ----------
        category : str
            The category of the variables.

        Returns
        -------
        List[str]
            The names of the variables.

        """
        pass

    @abc.abstractmethod
    def get_num_data_vars(self, category: str) -> int:
        """Get the number of data variables in the given category.

        Parameters
        ----------
        category : str
            The category of the variables.

        Returns
        -------
        int
            The number of data variables.

        """
        pass

    @abc.abstractmethod
    def get_normalization_dataarray(self, category: str) -> xr.Dataset:
        """Return the
        normalization
        dataarray for the
        given category. This
        should contain a
        `{category}_mean` and
        `{category}_std`
        variable for each
        variable in the
        category. For
        `category=="state"`,
        the dataarray should
        also contain a
        `state_diff_mean` and
        `state_diff_std`
        variable for the one-
        step differences of
        the state variables.
        The returned dataarray
        should at least have
        dimensions of `({categ
        ory}_feature)`, but
        can also include for
        example `grid_index`
        (if the normalisation
        is done per grid point
        for example).

        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).

        Returns
        -------
        xr.Dataset
            The normalization dataarray for the given category, with variables
            for the mean and standard deviation of the variables (and
            differences for state variables).

        """
        pass

    @abc.abstractmethod
    def get_dataarray(
        self, category: str, split: str
    ) -> Union[xr.DataArray, None]:
        """Return the
        processed data (as a
        single `xr.DataArray`)
        for the given category
        of data and
        test/train/val-split
        that covers all the
        data (in space and
        time) of a given
        category (state/forcin
        g/static). A datastore
        must be able to return
        for the "state"
        category, but
        "forcing" and "static"
        are optional (in which
        case the method should
        return `None`).

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
        pass

    @property
    @abc.abstractmethod
    def boundary_mask(self) -> xr.DataArray:
        """Return the boundary
        mask for the dataset,
        with spatial
        dimensions stacked.
        Where the value is 1,
        the grid point is a
        boundary point, and
        where the value is 0,
        the grid point is not
        a boundary point.

        Returns
        -------
        xr.DataArray
            The boundary mask for the dataset, with dimensions `('grid_index',)`.

        """
        pass


@dataclasses.dataclass
class CartesianGridShape:
    """Dataclass to store the shape of a grid."""

    x: int
    y: int


class BaseCartesianDatastore(BaseDatastore):
    """Base class for weather
    data stored on a Cartesian
    grid. In addition to the
    methods and attributes
    required for weather data
    in general (see
    `BaseDatastore`) for
    Cartesian gridded source
    data each `grid_index`
    coordinate value is assume
    to have an associated `x`
    and `y`-value so that the
    processed data-arrays can
    be reshaped back into into
    2D xy-gridded arrays.

    In addition the following attributes and methods are required:
    - `coords_projection` (property): Projection object for the coordinates.
    - `grid_shape_state` (property): Shape of the grid for the state variables.
    - `get_xy_extent` (method): Return the extent of the x, y coordinates for a
      given category of data.
    - `get_xy` (method): Return the x, y coordinates of the dataset.

    """

    CARTESIAN_COORDS = ["y", "x"]

    @property
    @abc.abstractmethod
    def coords_projection(self) -> ccrs.Projection:
        """Return the projection object for the coordinates.

        The projection object is used to plot the coordinates on a map.

        Returns
        -------
        cartopy.crs.Projection:
            The projection object.

        """
        pass

    @property
    @abc.abstractmethod
    def grid_shape_state(self) -> CartesianGridShape:
        """The shape of the grid for the state variables.

        Returns
        -------
        CartesianGridShape:
            The shape of the grid for the state variables, which has `x` and
            `y` attributes.

        """
        pass

    @abc.abstractmethod
    def get_xy(self, category: str, stacked: bool) -> np.ndarray:
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
            The x, y coordinates of the dataset, returned differently based on the
            value of `stacked`:
            - `stacked==True`: shape `(2, n_grid_points)` where n_grid_points=N_x*N_y.
            - `stacked==False`: shape `(2, N_y, N_x)`

        """
        pass

    def get_xy_extent(self, category: str) -> List[float]:
        """Return the extent
        of the x, y
        coordinates for a
        given category of
        data. The extent
        should be returned as
        a list of 4 floats
        with `[xmin, xmax,
        ymin, ymax]` which can
        then be used to set
        the extent of a plot.

        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).

        Returns
        -------
        List[float]
            The extent of the x, y coordinates.

        """
        xy = self.get_xy(category, stacked=False)
        extent = [xy[0].min(), xy[0].max(), xy[1].min(), xy[1].max()]
        return [float(v) for v in extent]

    def unstack_grid_coords(
        self, da_or_ds: Union[xr.DataArray, xr.Dataset]
    ) -> Union[xr.DataArray, xr.Dataset]:
        """Stack the spatial grid coordinates into separate `x` and `y` dimensions (the
        names can be set by the `CARTESIAN_COORDS` attribute) to create a 2D grid.

        Parameters
        ----------
        da_or_ds : xr.DataArray or xr.Dataset
            The dataarray or dataset to unstack the grid coordinates of.

        Returns
        -------
        xr.DataArray or xr.Dataset
            The dataarray or dataset with the grid coordinates unstacked.

        """
        return da_or_ds.set_index(grid_index=self.CARTESIAN_COORDS).unstack(
            "grid_index"
        )

    def stack_grid_coords(
        self, da_or_ds: Union[xr.DataArray, xr.Dataset]
    ) -> Union[xr.DataArray, xr.Dataset]:
        """Stack the spatial grid coordinated (by default `x` and `y`, but this can be
        set by the `CARTESIAN_COORDS` attribute) into a single `grid_index` dimension.

        Parameters
        ----------
        da_or_ds : xr.DataArray or xr.Dataset
            The dataarray or dataset to stack the grid coordinates of.

        Returns
        -------
        xr.DataArray or xr.Dataset
            The dataarray or dataset with the grid coordinates stacked.

        """
        return da_or_ds.stack(grid_index=self.CARTESIAN_COORDS)

neural_lam.datastore.base
=========================

.. py:module:: neural_lam.datastore.base




Module Contents
---------------

.. py:class:: BaseDatastore

   Bases: :py:obj:`abc.ABC`


   Base class for weather data used in the neural-lam package. A datastore
   defines the interface for accessing weather data by providing methods to
   access the data in a processed format that can be used for training and
   evaluation of neural networks.

   NOTE: All methods return either primitive types, `numpy.ndarray`,
   `xarray.DataArray` or `xarray.Dataset` objects, not `pytorch.Tensor`
   objects. Conversion to `pytorch.Tensor` objects should be done in the
   `weather_dataset.WeatherDataset` class (which inherits from
   `torch.utils.data.Dataset` and uses the datastore to access the data).

   # Forecast vs analysis data
   If the datastore is used to represent forecast rather than analysis data,
   then the `is_forecast` attribute should be set to True, and returned data
   from `get_dataarray` is assumed to have `analysis_time` and `forecast_time`
   dimensions (rather than just `time`).

   # Ensemble vs deterministic data
   If the datastore is used to present an ensemble of state realisations, for
   example for forecast ensembles, then the `is_ensemble` attribute should be
   set to `True` and returned state data from `get_dataarray` is expected to
   have an `ensemble_member` dimension. If each ensemble member has its own
   forcing values, then `has_ensemble_forcing` should be set to `True`, and
   returned forcing data from `get_dataarray` is expected to have an
   `ensemble_member` dimension; otherwise forcing data is expected not to have
   one.

   # Grid index
   All methods that return data specific to a grid point (like
   `get_dataarray`) should have a single dimension named `grid_index` that
   represents the spatial grid index of the data. The actual x, y coordinates
   of the grid points should be stored in the `x` and `y` coordinates of the
   dataarray or dataset with the `grid_index` dimension as the coordinate for
   each of the `x` and `y` coordinates.


   .. py:attribute:: is_ensemble
      :type:  bool
      :value: False



   .. py:attribute:: has_ensemble_forcing
      :type:  bool
      :value: False



   .. py:attribute:: is_forecast
      :type:  bool
      :value: False



   .. py:property:: root_path
      :type: pathlib.Path

      :abstractmethod:


      The root path to the datastore. It is relative to this that any derived
      files (for example the graph components) are stored.

      Returns
      -------
      pathlib.Path
          The root path to the datastore.




   .. py:property:: config
      :type: collections.abc.Mapping

      :abstractmethod:


      The configuration of the datastore.

      Returns
      -------
      collections.abc.Mapping
          The configuration of the datastore, any dict like object can be
          returned.




   .. py:property:: step_length
      :type: datetime.timedelta

      :abstractmethod:


      The step length of the dataset as a time interval.

      Returns:
          timedelta: The step length as a datetime.timedelta object.




   .. py:method:: get_vars_units(category: str) -> List[str]
      :abstractmethod:


      Get the units of the variables in the given category.

      Parameters
      ----------
      category : str
          The category of the variables (state/forcing/static).

      Returns
      -------
      List[str]
          The units of the variables.




   .. py:method:: get_vars_names(category: str) -> List[str]
      :abstractmethod:


      Get the names of the variables in the given category.

      Parameters
      ----------
      category : str
          The category of the variables (state/forcing/static).

      Returns
      -------
      List[str]
          The names of the variables.




   .. py:method:: get_vars_long_names(category: str) -> List[str]
      :abstractmethod:


      Get the long names of the variables in the given category.

      Parameters
      ----------
      category : str
          The category of the variables (state/forcing/static).

      Returns
      -------
      List[str]
          The long names of the variables.




   .. py:method:: get_num_data_vars(category: str) -> int
      :abstractmethod:


      Get the number of data variables in the given category.

      Parameters
      ----------
      category : str
          The category of the variables (state/forcing/static).

      Returns
      -------
      int
          The number of data variables.




   .. py:method:: get_standardization_dataarray(category: str) -> xarray.Dataset
      :abstractmethod:


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




   .. py:method:: get_dataarray(category: str, split: Optional[str], standardize: bool = False) -> Union[xarray.DataArray, None]
      :abstractmethod:


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

      If we have multiple ensemble members of state data, the returned state
      dataarray is expected to have an additional `ensemble_member`
      dimension. If `has_ensemble_forcing=True`, the returned forcing
      dataarray is expected to have an additional `ensemble_member`
      dimension; otherwise it is expected not to have one.

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




   .. py:property:: boundary_mask
      :type: xarray.DataArray

      :abstractmethod:


      Return the boundary mask for the dataset, with spatial dimensions
      stacked. Where the value is 1, the grid point is a boundary point, and
      where the value is 0, the grid point is not a boundary point.

      Returns
      -------
      xr.DataArray
          The boundary mask for the dataset, with dimensions
          `('grid_index',)`.




   .. py:method:: get_xy(category: str, stacked: bool) -> numpy.ndarray
      :abstractmethod:


      Return the x, y coordinates of the dataset as a numpy arrays for a
      given category of data.

      Parameters
      ----------
      category : str
          The category of the dataset (state/forcing/static).
      stacked : bool
          Whether to stack the x, y coordinates. `stacked=False` is only
          meaningful for grid points on a regular-grid.

      Returns
      -------
      np.ndarray
          The x, y coordinates of the dataset with shape `[n_grid_points, 2]`.



   .. py:property:: coords_projection
      :type: cartopy.crs.Projection

      :abstractmethod:


      Return the projection object for the coordinates.

      The projection object is used to plot the coordinates on a map.

      Returns
      -------
      cartopy.crs.Projection:
          The projection object.




   .. py:method:: get_xy_extent(category: str) -> List[float]

      Return the extent of the x, y coordinates for a given category of data.
      The extent should be returned as a list of 4 floats with `[xmin, xmax,
      ymin, ymax]` which can then be used to set the extent of a plot.

      Parameters
      ----------
      category : str
          The category of the dataset (state/forcing/static).

      Returns
      -------
      List[float]
          The extent of the x, y coordinates.




   .. py:method:: get_lat_lon(category: str) -> numpy.ndarray

      Return stacked longitude/latitude pairs for the requested category.



   .. py:property:: num_grid_points
      :type: int

      :abstractmethod:


      Return the number of grid points in the dataset.

      Returns
      -------
      int
          The number of grid points in the dataset.




   .. py:property:: state_feature_weights_values
      :type: List[float]

      :abstractmethod:


      Return the weights for each state feature as a list of floats. The
      weights are defined by the user in a config file for the datastore.

      Implementations of this method must assert that there is one weight for
      each state feature in the datastore. The weights can be used to scale
      the loss function for each state variable (e.g. via the standard
      deviation of the 1-step differences of the state variables).

      Returns:
          List[float]: The weights for each state feature.



   .. py:method:: expected_dim_order(category: Optional[str] = None) -> tuple[str, Ellipsis]

      Return the expected dimension order for the dataarray or dataset
      returned by `get_dataarray` for the given category of data. The
      dimension order is the order of the dimensions in the dataarray or
      dataset, and is used to check that the data is in the expected format.

      This is necessary so that when stacking and unstacking the spatial grid
      we can ensure that the dimension order is the same as what is returned
      from `get_dataarray`. And also ensures that downstream uses of a
      datastore (e.g. WeatherDataset) sees the data in a common structure.

      If the category is None, then the it assumed that data only represents
      a 1D scalar field varying with grid-index.

      The order is constructed to match the order in `pytorch.Tensor` objects
      that will be constructed from the data so that the last two dimensions
      are always the grid-index and feature dimensions (i.e. the order is
      `[..., grid_index, {category}_feature]`), with any time-related and
      ensemble-number dimension(s) coming before these two.

      Parameters
      ----------
      category : str
          The category of the dataset (state/forcing/static).
      Returns
      -------
      List[str]
          The expected dimension order for the dataarray or dataset.




.. py:class:: CartesianGridShape

   Dataclass to store the shape of a grid.


   .. py:attribute:: x
      :type:  int


   .. py:attribute:: y
      :type:  int


.. py:class:: BaseRegularGridDatastore

   Bases: :py:obj:`BaseDatastore`


   Base class for weather data stored on a regular grid (like a chess-board,
   as opposed to a irregular grid where each cell cannot be indexed by just
   two integers, see https://en.wikipedia.org/wiki/Regular_grid). In addition
   to the methods and attributes required for weather data in general (see
   `BaseDatastore`) for regular-gridded source data each `grid_index`
   coordinate value is assumed to be associated with `x` and `y`-values that
   allow the processed data-arrays can be reshaped back into into 2D
   xy-gridded arrays (to change the name of the spatial coordinates the
   `spatial_coordinates` value should be changed from its default value of
   `("x", "y")`).

   The following methods and attributes must be implemented for datastore that
   represents regular-gridded data:
   - `grid_shape_state` (property): 2D shape of the grid for the state
     variables.
   - `get_xy` (method): Return the x, y coordinates of the dataset, with the
     option to not stack the coordinates (so that they are returned as a 2D
     grid).
   - `get_lat_lon` (method): Return the latitude/longitude coordinates of
     the dataset for convenience when plotting.

   The operation of going from (x,y)-indexed regular grid
   to `grid_index`-indexed data-array is called "stacking" and the reverse
   operation is called "unstacking". This class provides methods to stack and
   unstack the spatial grid coordinates of the data-arrays (called
   `stack_grid_coords` and `unstack_grid_coords` respectively).


   .. py:attribute:: spatial_coordinates
      :value: ('x', 'y')



   .. py:property:: grid_shape_state
      :type: CartesianGridShape

      :abstractmethod:


      The shape of the grid for the state variables.

      Returns
      -------
      CartesianGridShape:
          The shape of the grid for the state variables, which has `x` and
          `y` attributes.




   .. py:method:: get_xy(category: str, stacked: bool) -> numpy.ndarray
      :abstractmethod:


      Return the x, y coordinates of the dataset.

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
          the value of `stacked`: - `stacked==True`: shape `(n_grid_points,
          2)` where
                             n_grid_points=N_x*N_y.
          - `stacked==False`: shape `(N_x, N_y, 2)`



   .. py:method:: unstack_grid_coords(da_or_ds: Union[xarray.DataArray, xarray.Dataset]) -> Union[xarray.DataArray, xarray.Dataset]

      Unstack the spatial grid coordinates from `grid_index` into separate `x`
      and `y` dimensions to create a 2D grid (if the spatial coordinates have
      different names, those are used instead). Only performs unstacking if
      the data is currently stacked (has grid_index dimension).

      Parameters
      ----------
      da_or_ds : xr.DataArray or xr.Dataset
          The dataarray or dataset to unstack the grid coordinates of.

      Returns
      -------
      xr.DataArray or xr.Dataset
          The dataarray or dataset with the grid coordinates unstacked.



   .. py:method:: stack_grid_coords(da_or_ds: Union[xarray.DataArray, xarray.Dataset]) -> Union[xarray.DataArray, xarray.Dataset]

      Stack the spatial grid coordinates (x and y) into a single `grid_index`
      dimension. Only performs stacking if the data is currently unstacked
      (has x and y dimensions).

      Parameters
      ----------
      da_or_ds : xr.DataArray or xr.Dataset
          The dataarray or dataset to stack the grid coordinates of.

      Returns
      -------
      xr.DataArray or xr.Dataset
          The dataarray or dataset with the grid coordinates stacked.



   .. py:property:: num_grid_points
      :type: int


      Return the number of grid points in the dataset.

      Returns
      -------
      int
          The number of grid points in the dataset.





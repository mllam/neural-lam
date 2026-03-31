neural_lam.datastore.npyfilesmeps.store
=======================================

.. py:module:: neural_lam.datastore.npyfilesmeps.store

.. autoapi-nested-parse::

   Numpy-files based datastore to support the MEPS example dataset introduced in
   neural-lam v0.1.0.







Module Contents
---------------

.. py:data:: STATE_FILENAME_FORMAT
   :value: 'nwp_{analysis_time:%Y%m%d%H}_mbr{member_id:03d}.npy'


.. py:data:: TOA_SW_DOWN_FLUX_FILENAME_FORMAT
   :value: 'nwp_toa_downwelling_shortwave_flux_{analysis_time:%Y%m%d%H}.npy'


.. py:data:: OPEN_WATER_FILENAME_FORMAT
   :value: 'wtr_{analysis_time:%Y%m%d%H}.npy'


.. py:class:: NpyFilesDatastoreMEPS(config_path)

   Bases: :py:obj:`neural_lam.datastore.base.BaseRegularGridDatastore`


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


   .. py:attribute:: SHORT_NAME
      :value: 'npyfilesmeps'



   .. py:attribute:: is_forecast
      :value: True



   .. py:attribute:: is_ensemble


   .. py:attribute:: has_ensemble_forcing
      :value: False



   .. py:property:: root_path
      :type: pathlib.Path


      The root path of the datastore on disk. This is the directory relative
      to which graphs and other files can be stored.

      Returns
      -------
      Path
          The root path of the datastore




   .. py:property:: config
      :type: neural_lam.datastore.npyfilesmeps.config.NpyDatastoreConfig


      The configuration for the datastore.

      Returns
      -------
      NpyDatastoreConfig
          The configuration for the datastore.




   .. py:method:: get_dataarray(category: str, split: Optional[str], standardize: bool = False) -> xarray.core.dataarray.DataArray

      Get the data array for the given category and split of data. If the
      category is 'state', the data array will be a concatenation of the data
      arrays for all ensemble members. The data will be loaded as a dask
      array, so that the data isn't actually loaded until it's needed.

      Parameters
      ----------
      category : str
          The category of the data to load. One of 'state', 'forcing', or
          'static'.
      split : str
          The dataset split to load the data for. One of 'train', 'val', or
          'test'.
      standardize: bool
          If the dataarray should be returned standardized

      Returns
      -------
      xr.DataArray
          The data array for the given category and split, with dimensions
          per category:
          state:     `[elapsed_forecast_duration, analysis_time, grid_index,
                      feature, ensemble_member]`
          forcing:   `[elapsed_forecast_duration, analysis_time, grid_index,
                      feature]`
          static:    `[grid_index, feature]`




   .. py:method:: get_vars_units(category: str) -> List[str]

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

      Get the number of data variables in the given category.

      Parameters
      ----------
      category : str
          The category of the variables (state/forcing/static).

      Returns
      -------
      int
          The number of data variables.




   .. py:method:: get_xy(category: str, stacked: bool) -> numpy.ndarray

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
          The x, y coordinates of the dataset (with x first then y second),
          returned differently based on the value of `stacked`:
          - `stacked==True`: shape `(n_grid_points, 2)` where
                                    n_grid_points=N_x*N_y.
          - `stacked==False`: shape `(N_x, N_y, 2)`




   .. py:property:: step_length
      :type: datetime.timedelta


      The length of each time step as a time interval.

      Returns
      -------
      timedelta
          The length of each time step as a datetime.timedelta object.




   .. py:property:: grid_shape_state
      :type: neural_lam.datastore.base.CartesianGridShape


      The shape of the cartesian grid for the state variables.

      Returns
      -------
      CartesianGridShape
          The shape of the cartesian grid for the state variables.




   .. py:property:: boundary_mask
      :type: xarray.DataArray


      The boundary mask for the dataset. This is a binary mask that is 1
      where the grid cell is on the boundary of the domain, and 0 otherwise.

      Returns
      -------
      xr.DataArray
          The boundary mask for the dataset, with dimensions `[grid_index]`.




   .. py:method:: get_standardization_dataarray(category: str) -> xarray.Dataset

      Return the standardization dataarray for the given category. This
      should contain a `{category}_mean` and `{category}_std` variable for
      each variable in the category.
      For `category=="state"`, the dataarray should also contain a
      `state_diff_mean_standardized` and `state_diff_std_standardized`
      variable for the one-step differences of the state variables.

      Parameters
      ----------
      category : str
          The category of the dataset (state/forcing/static).

      Returns
      -------
      xr.Dataset
          The standardization dataarray for the given category, with
          variables for the mean and standard deviation of the variables (and
          differences for state variables).




   .. py:method:: coords_projection() -> cartopy.crs.Projection

      The projection of the spatial coordinates.

      Returns
      -------
      ccrs.Projection
          The projection of the spatial coordinates.





neural_lam.datastore.mdp
========================

.. py:module:: neural_lam.datastore.mdp




Module Contents
---------------

.. py:class:: MDPDatastore(config_path, n_boundary_points=30, reuse_existing=True)

   Bases: :py:obj:`neural_lam.datastore.base.BaseRegularGridDatastore`


   Datastore class for datasets made with the mllam_data_prep library
   (https://github.com/mllam/mllam-data-prep). This class wraps the
   `mllam_data_prep` library to do the necessary transforms to create the
   different categories (state/forcing/static) of data, with the actual
   transform to do being specified in the configuration file.


   .. py:attribute:: SHORT_NAME
      :value: 'mdp'



   .. py:attribute:: is_ensemble


   .. py:attribute:: has_ensemble_forcing


   .. py:attribute:: spatial_coordinates
      :value: None



   .. py:property:: root_path
      :type: pathlib.Path


      The root path of the dataset.

      Returns
      -------
      Path
          The root path of the dataset.




   .. py:property:: config
      :type: mllam_data_prep.Config


      The configuration of the dataset.

      Returns
      -------
      mdp.Config
          The configuration of the dataset.




   .. py:property:: step_length
      :type: datetime.timedelta


      The length of the time steps as a time interval.

      Returns
      -------
      timedelta
          The length of the time steps as a datetime.timedelta object.




   .. py:method:: get_vars_units(category: str) -> List[str]

      Return the units of the variables in the given category.

      Parameters
      ----------
      category : str
          The category of the dataset (state/forcing/static).

      Returns
      -------
      List[str]
          The units of the variables in the given category.




   .. py:method:: get_vars_names(category: str) -> List[str]

      Return the names of the variables in the given category.

      Parameters
      ----------
      category : str
          The category of the dataset (state/forcing/static).

      Returns
      -------
      List[str]
          The names of the variables in the given category.




   .. py:method:: get_vars_long_names(category: str) -> List[str]

      Return the long names of the variables in the given category.

      Parameters
      ----------
      category : str
          The category of the dataset (state/forcing/static).

      Returns
      -------
      List[str]
          The long names of the variables in the given category.




   .. py:method:: get_num_data_vars(category: str) -> int

      Return the number of variables in the given category.

      Parameters
      ----------
      category : str
          The category of the dataset (state/forcing/static).

      Returns
      -------
      int
          The number of variables in the given category.




   .. py:method:: get_dataarray(category: str, split: Optional[str], standardize: bool = False) -> Union[xarray.DataArray, None]

      Return the processed data (as a single `xr.DataArray`) for the given
      category of data and test/train/val-split that covers all the data (in
      space and time) of a given category (state/forcing/static). "state" is
      the only required category, for other categories, the method will
      return `None` if the category is not found in the datastore.

      The returned dataarray will at minimum have dimensions of `(grid_index,
      {category}_feature)` so that any spatial dimensions have been stacked
      into a single dimension and all variables and levels have been stacked
      into a single feature dimension named by the `category` of data being
      loaded.

      For categories of data that have a time dimension (i.e. not static
      data), the dataarray will additionally have `(analysis_time,
      elapsed_forecast_duration)` dimensions if `is_forecast` is True, or
      `(time)` if `is_forecast` is False.

      If we have multiple ensemble members of state data, the returned state
      dataarray will have an additional `ensemble_member` dimension. If
      `has_ensemble_forcing=True`, the returned forcing dataarray will also
      have an additional `ensemble_member` dimension.

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




   .. py:property:: boundary_mask
      :type: xarray.DataArray


      Produce a 0/1 mask for the boundary points of the dataset, these will
      sit at the edges of the domain (in x/y extent) and will be used to mask
      out the boundary points from the loss function and to overwrite the
      boundary points from the prediction. For now this is created when the
      mask is requested, but in the future this could be saved to the zarr
      file.

      Returns
      -------
      xr.DataArray
          A 0/1 mask for the boundary points of the dataset, where 1 is a
          boundary point and 0 is not.




   .. py:property:: coords_projection
      :type: cartopy.crs.Projection


      Return the projection of the coordinates.

      NOTE: currently this expects the projection information to be in the
      `extra` section of the configuration file, with a `projection` key
      containing a `class_name` and `kwargs` for constructing the
      `cartopy.crs.Projection` object. This is a temporary solution until
      the projection information can be parsed in the produced dataset
      itself. `mllam-data-prep` ignores the contents of the `extra` section
      of the config file which is why we need to check that the necessary
      parts are there.

      Returns
      -------
      ccrs.Projection
          The projection of the coordinates.




   .. py:property:: grid_shape_state

      The shape of the cartesian grid for the state variables.

      Returns
      -------
      CartesianGridShape
          The shape of the cartesian grid for the state variables.




   .. py:method:: get_xy(category: str, stacked: bool) -> numpy.ndarray

      Return the x, y coordinates of the dataset (i.e. the Cartesian
      coordinates of the regular gridded dataset).

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




   .. py:method:: get_lat_lon(category: str) -> numpy.ndarray

      Return the longitude, latitude coordinates of the dataset as numpy
      array for a given category of data.
      Override in MDP to use lat/lons directly from xr.Dataset, if available.

      Parameters
      ----------
      category : str
          The category of the dataset (state/forcing/static).

      Returns
      -------
      np.ndarray
          The longitude, latitude coordinates of the dataset
          with shape `[n_grid_points, 2]`.




neural_lam.vis
==============

.. py:module:: neural_lam.vis




Module Contents
---------------

.. py:function:: plot_on_axis(ax, da, datastore, vmin=None, vmax=None, ax_title=None, cmap='plasma', boundary_alpha=None, crop_to_interior=False)

   Plot weather state on given axis using datastore metadata.

   Parameters
   ----------
   ax : matplotlib.axes.Axes
       The axis to plot on. Should have a cartopy projection.
   da : xarray.DataArray
       The data to plot. Should have shape (N_grid,).
   datastore : BaseRegularGridDatastore
       The datastore containing metadata about the grid.
   vmin : float, optional
       Minimum value for color scale.
   vmax : float, optional
       Maximum value for color scale.
   ax_title : str, optional
       Title for the axis.
   cmap : str or matplotlib.colors.Colormap, optional
       Colormap to use for plotting.
   boundary_alpha : float, optional
       If provided, overlay boundary mask with given alpha transparency.
   crop_to_interior : bool, optional
       If True, crop the plot to the interior region.

   Returns
   -------
   matplotlib.collections.QuadMesh
       The mesh object created by pcolormesh.



.. py:function:: plot_error_map(errors, datastore: neural_lam.datastore.base.BaseRegularGridDatastore, title=None)

   Plot a heatmap of errors of different variables at different
   predictions horizons
   errors: (pred_steps, d_f)


.. py:function:: plot_prediction(datastore: neural_lam.datastore.base.BaseRegularGridDatastore, da_prediction: xarray.DataArray, da_target: xarray.DataArray, title=None, vrange=None, boundary_alpha=0.7, crop_to_interior=True, colorbar_label: str = '')

   Plot example prediction and grond truth.

   Each has shape (N_grid,)



.. py:function:: plot_spatial_error(error: torch.Tensor, datastore: neural_lam.datastore.base.BaseRegularGridDatastore, title=None, vrange=None, boundary_alpha=0.7, crop_to_interior=True, colorbar_label: str = '')

   Plot spatial error with projection-aware axes.



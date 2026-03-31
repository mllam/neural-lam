neural_lam.datastore.plot_example
=================================

.. py:module:: neural_lam.datastore.plot_example




Module Contents
---------------

.. py:function:: plot_example_from_datastore(category, datastore, col_dim, split='train', standardize=True, selection={}, index_selection={})

   Create a plot of the data from the datastore.

   Parameters
   ----------
   category : str
       Category of data to plot, one of "state", "forcing", or "static".
   datastore : Datastore
       Datastore to retrieve data from.
   col_dim : str
       Dimension to use for plot facetting into columns. This can be a
       template string that can be formatted with the category name.
   split : str, optional
       Split of data to plot, by default "train".
   standardize : bool, optional
       Whether to standardize the data before plotting, by default True.
   selection : dict, optional
       Selections to apply to the dataarray, for example
       `time="1990-09-03T0:00" would select this single timestep, by default
       {}.
   index_selection: dict, optional
       Index-based selection to apply to the dataarray, for example
       `time=0` would select the first item along the `time` dimension, by
       default {}.

   Returns
   -------
   Figure
       Matplotlib figure object.



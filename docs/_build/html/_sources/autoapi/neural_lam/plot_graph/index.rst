neural_lam.plot_graph
=====================

.. py:module:: neural_lam.plot_graph






Module Contents
---------------

.. py:data:: MESH_HEIGHT
   :value: 0.1


.. py:data:: MESH_LEVEL_DIST
   :value: 0.2


.. py:data:: GRID_HEIGHT
   :value: 0


.. py:function:: plot_graph(grid_pos, hierarchical, graph_ldict, show_axis=False, save=None)

   Build a 3D plotly figure of the graph structure.

   Parameters
   ----------
   grid_pos : np.ndarray
       Grid node positions, shape (N_grid, 2).
   hierarchical : bool
       Whether the loaded graph is hierarchical.
   graph_ldict : dict
       Graph dict as returned by ``utils.load_graph``.
   show_axis : bool
       If True, show the 3D axis.
   save : str or None
       If given, save the figure as an HTML file at this path.

   Returns
   -------
   go.Figure
       The plotly figure object.


.. py:function:: main()

   Plot graph structure in 3D using plotly.



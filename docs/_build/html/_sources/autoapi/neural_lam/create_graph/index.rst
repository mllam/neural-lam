neural_lam.create_graph
=======================

.. py:module:: neural_lam.create_graph




Module Contents
---------------

.. py:function:: plot_graph(graph, title=None)

   Plot a graph with nodes coloured by degree.

   Parameters
   ----------
   graph : torch_geometric.data.Data
       Graph object with edge_index and pos attributes.
   title : str, optional
       Title for the plot. If None, no title is shown.

   Returns
   -------
   tuple
       A (fig, axis) tuple of the matplotlib figure and axis.


.. py:function:: sort_nodes_internally(nx_graph)

   Return a copy of the graph with nodes sorted internally.

   Converts the input DiGraph into a new DiGraph with nodes sorted
   by their keys. This is needed because networkx node ordering
   is used by PyTorch Geometric during conversion.

   Parameters
   ----------
   nx_graph : networkx.DiGraph
       Input directed graph to sort.

   Returns
   -------
   networkx.DiGraph
       New directed graph with nodes sorted by key.


.. py:function:: save_edges(graph, name, base_path)

   Save edge index and edge features of a graph to disk.

   Saves two files: {name}_edge_index.pt and {name}_features.pt,
   where features are the concatenation of edge length and vector
   difference (len, vdiff).

   Parameters
   ----------
   graph : torch_geometric.data.Data
       Graph with edge_index, len, and vdiff attributes.
   name : str
       Name prefix for the saved files.
   base_path : str
       Directory path where files will be saved.


.. py:function:: save_edges_list(graphs, name, base_path)

   Save edge indices and features for a list of graphs to disk.

   Saves two files: {name}_edge_index.pt (list of edge indices) and
   {name}_features.pt (list of edge feature tensors), where features
   are the concatenation of edge length and vector difference.

   Parameters
   ----------
   graphs : list of torch_geometric.data.Data
       List of graphs, each with edge_index, len, and vdiff attributes.
   name : str
       Name prefix for the saved files.
   base_path : str
       Directory path where files will be saved.


.. py:function:: from_networkx_with_start_index(nx_graph, start_index)

   Convert a networkx graph to PyTorch Geometric format with offset indices.

   Converts the graph and shifts all node indices in edge_index by
   start_index, so nodes are correctly indexed in a larger combined graph.

   Parameters
   ----------
   nx_graph : networkx.Graph
       Input graph to convert.
   start_index : int
       Integer offset to add to all node indices in edge_index.

   Returns
   -------
   torch_geometric.data.Data
       PyG graph with edge_index shifted by start_index.


.. py:function:: mk_2d_graph(xy, nx, ny)

   Create a 2D directed graph over a regular grid with diagonal edges.

   Builds a grid graph of nx by ny nodes positioned within the spatial
   extent of xy, avoiding border nodes. Adds both axis-aligned and
   diagonal edges in both directions, storing edge length and vector
   difference as edge attributes.

   Parameters
   ----------
   xy : np.ndarray
       Grid coordinates of shape (Nx, Ny, 2).
   nx : int
       Number of nodes along the x axis.
   ny : int
       Number of nodes along the y axis.

   Returns
   -------
   networkx.DiGraph
       Directed graph with pos, len, and vdiff attributes on nodes and edges.


.. py:function:: prepend_node_index(graph, new_index)

   Relabel graph nodes by prepending a new index level.

   Transforms each node key (i, j) into (new_index, i, j), which is
   used to distinguish nodes across different hierarchy levels in the
   mesh graph.

   Parameters
   ----------
   graph : networkx.Graph
       Input graph with tuple node keys.
   new_index : int
       Index to prepend to each node key.

   Returns
   -------
   networkx.Graph
       Copy of the graph with relabelled node keys.


.. py:function:: create_graph(graph_dir_path: str, xy: numpy.ndarray, n_max_levels: Optional[int] = None, hierarchical: Optional[bool] = False, create_plot: Optional[bool] = False)

   Create graph components from `xy` grid coordinates and store in
   `graph_dir_path`.

   Creates the following files for all graphs:
   - g2m_edge_index.pt  [2, N_g2m_edges]
   - g2m_features.pt    [N_g2m_edges, d_features]
   - m2g_edge_index.pt  [2, N_m2m_edges]
   - m2g_features.pt    [N_m2m_edges, d_features]
   - m2m_edge_index.pt  list of [2, N_m2m_edges_level], length==n_levels
   - m2m_features.pt    list of [N_m2m_edges_level, d_features],
                        length==n_levels
   - mesh_features.pt   list of [N_mesh_nodes_level, d_mesh_static],
                        length==n_levels

   where
     d_features:
           number of features per edge (currently d_features==3, for
           edge-length, x and y)
     N_g2m_edges:
           number of edges in the graph from grid-to-mesh
     N_m2g_edges:
           number of edges in the graph from mesh-to-grid
     N_m2m_edges_level:
           number of edges in the graph from mesh-to-mesh at a given level
           (list index corresponds to the level)
     d_mesh_static:
           number of static features per mesh node (currently
           d_mesh_static==2, for x and y)
     N_mesh_nodes_level:
           number of nodes in the mesh at a given level

   And in addition for hierarchical graphs:
   - mesh_up_edge_index.pt
       list of [2, N_mesh_updown_edges_level], length==n_levels-1
   - mesh_up_features.pt
       list of [N_mesh_updown_edges_level, d_features], length==n_levels-1
   - mesh_down_edge_index.pt
       list of [2, N_mesh_updown_edges_level], length==n_levels-1
   - mesh_down_features.pt
       list of [N_mesh_updown_edges_level, d_features], length==n_levels-1

   where N_mesh_updown_edges_level is the number of edges in the graph from
   mesh-to-mesh between two consecutive levels (list index corresponds index
   of lower level)


   Parameters
   ----------
   graph_dir_path : str
       Path to store the graph components.
   xy : np.ndarray
       Grid coordinates, expected to be of shape (Nx, Ny, 2).
   n_max_levels : int
       Limit multi-scale mesh to given number of levels, from bottom up
       (default: None (no limit)).
   hierarchical : bool
       Generate hierarchical mesh graph (default: False).
   create_plot : bool
       If graphs should be plotted during generation (default: False).

   Returns
   -------
   None



.. py:function:: create_graph_from_datastore(datastore: neural_lam.datastore.base.BaseRegularGridDatastore, output_root_path: str, n_max_levels: Optional[int] = None, hierarchical: bool = False, create_plot: bool = False)

   Create graph components from a datastore and save to disk.

   Extracts grid coordinates from the datastore and calls create_graph
   to build and save all graph components to output_root_path.
   Currently only supports BaseRegularGridDatastore.

   Parameters
   ----------
   datastore : BaseRegularGridDatastore
       Datastore to extract grid coordinates from.
   output_root_path : str
       Directory path where graph components will be saved.
   n_max_levels : int, optional
       Limit multi-scale mesh to given number of levels (default: None).
   hierarchical : bool
       Generate hierarchical mesh graph (default: False).
   create_plot : bool
       If graphs should be plotted during generation (default: False).

   Raises
   ------
   NotImplementedError
       If the datastore is not a BaseRegularGridDatastore.


.. py:function:: cli(input_args=None)

   Command-line interface for graph generation.

   Parses command-line arguments and calls create_graph_from_datastore
   to generate and save graph components for a given neural-lam config.

   Parameters
   ----------
   input_args : list of str, optional
       List of argument strings. If None, reads from sys.argv.



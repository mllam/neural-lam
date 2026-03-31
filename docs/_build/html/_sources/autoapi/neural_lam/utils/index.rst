neural_lam.utils
================

.. py:module:: neural_lam.utils






Module Contents
---------------

.. py:class:: BufferList(buffer_tensors, persistent=True)

   Bases: :py:obj:`torch.nn.Module`


   A list of torch buffer tensors that sit together as a Module with no
   parameters and only buffers.

   This should be replaced by a native torch BufferList once implemented.
   See: https://github.com/pytorch/pytorch/issues/37386


   .. py:attribute:: n_buffers


.. py:function:: zero_index_edge_index(edge_index)

   Make both sender and receiver indices of edge_index start at 0


.. py:function:: zero_index_m2g(m2g_edge_index: torch.Tensor, mesh_static_features: list[torch.Tensor], mesh_first: bool, restore: bool = False) -> torch.Tensor

   Zero-index the m2g (mesh-to-grid) edge index, or undo this operation.

   Special handling is needed since not all mesh nodes may be present.

   Parameters
   ----------
   m2g_edge_index : torch.Tensor
       Edge index tensor of shape (2, N_edges).
   mesh_static_features : list of torch.Tensor
       Mesh node feature tensors.
   mesh_first : bool
       If True, mesh nodes are indexed before grid nodes.
   restore : bool
       If True, undo zero-indexing (restore original indices).

   Returns
   -------
   torch.Tensor
       Edge index tensor with zero-based or restored indices.


.. py:function:: zero_index_g2m(g2m_edge_index: torch.Tensor, mesh_static_features: list[torch.Tensor], mesh_first: bool, restore: bool = False) -> torch.Tensor

   Zero-index the g2m (grid-to-mesh) edge index, or undo this operation.

   Special handling is needed since not all mesh nodes may be present.

   Parameters
   ----------
   g2m_edge_index : torch.Tensor
       Edge index tensor of shape (2, N_edges).
   mesh_static_features : list of torch.Tensor
       Mesh node feature tensors.
   mesh_first : bool
       If True, mesh nodes are indexed before grid nodes.
   restore : bool
       If True, undo zero-indexing (restore original indices).

   Returns
   -------
   torch.Tensor
       Edge index tensor with zero-based or restored indices.


.. py:function:: load_graph(graph_dir_path, device='cpu')

   Load all tensors representing the graph from `graph_dir_path`.

   Needs the following files for all graphs:
   - m2m_edge_index.pt
   - g2m_edge_index.pt
   - m2g_edge_index.pt
   - m2m_features.pt
   - g2m_features.pt
   - m2g_features.pt
   - mesh_features.pt

   And in addition for hierarchical graphs:
   - mesh_up_edge_index.pt
   - mesh_down_edge_index.pt
   - mesh_up_features.pt
   - mesh_down_features.pt

   Parameters
   ----------
   graph_dir_path : str
       Path to directory containing the graph files.
   device : str
       Device to load tensors to.

   Returns
   -------
   hierarchical : bool
       Whether the graph is hierarchical.
   graph : dict
       Dictionary containing the graph tensors, with keys as follows:
       - g2m_edge_index
       - m2g_edge_index
       - m2m_edge_index
       - mesh_up_edge_index
       - mesh_down_edge_index
       - g2m_features
       - m2g_features
       - m2m_features
       - mesh_up_features
       - mesh_down_features
       - mesh_static_features



.. py:function:: make_mlp(blueprint, layer_norm=True)

   Create MLP from list blueprint, with
   input dimensionality: blueprint[0]
   output dimensionality: blueprint[-1] and
   hidden layers of dimensions: blueprint[1], ..., blueprint[-2]

   if layer_norm is True, includes a LayerNorm layer at
   the output (as used in GraphCast)


.. py:function:: has_working_latex()

   Check if LaTeX is available or its toolchain


.. py:function:: fractional_plot_bundle(fraction)

   Get the tueplots bundle, but with figure width as a fraction of
   the page width.


.. py:function:: log_on_rank_zero(msg: str, level: str = 'info', *args, **kwargs)

   Log a message only on rank zero using loguru logger.

   Parameters
   ----------
   msg : str
       The message to log.
   level : str, optional
       The logging level (e.g. "info", "warning", "error"). Default is "info".


.. py:function:: init_training_logger_metrics(training_logger, val_steps)

   Set up logger metrics to track


.. py:function:: setup_training_logger(datastore, args, run_name)

   Set up the training logger (WandB or MLFlow).

   Parameters
   ----------
   datastore : Datastore
       Datastore object.

   args : argparse.Namespace
       Arguments from command line.

   run_name : str
       Name of the run.

   Returns
   -------
   training_logger : pytorch_lightning.loggers.base
       Logger object.

   Notes
   -----
   When ``--wandb_id`` is given, ``resume="allow"`` is set automatically:
   W&B resumes the run if it exists, or creates it with that ID otherwise.
   This allows the same job script to be safely resubmitted on HPC systems.
   The run name is set to ``None`` when resuming to preserve the existing name.


.. py:function:: inverse_softplus(x, beta=1, threshold=20)

   Inverse of torch.nn.functional.softplus

   Input is clamped to approximately positive values of x, and the function is
   linear for inputs above x*beta for numerical stability.

   Note that this torch.clamp will make gradients 0, but this is not a
   problem as values of x that are this close to 0 have gradients of 0 anyhow.


.. py:function:: inverse_sigmoid(x)

   Inverse of torch.sigmoid

   Sigmoid output takes values in [0,1], this makes sure input is just within
   this interval.
   Note that this torch.clamp will make gradients 0, but this is not a problem
   as values of x that are this close to 0 or 1 have gradients of 0 anyhow.


.. py:function:: get_integer_time(tdelta) -> tuple[int, str]

   Get the largest time unit that can represent the given timedelta as an
   integer.

   Returns:
       int: The integer value of the timedelta in the largest time unit, or
               1 if no such unit exists.
       str: The time unit as a string ('weeks', 'days', 'hours', 'minutes',
               'seconds', 'milliseconds', 'microseconds'). If no unit can
               represent the timedelta as an integer, returns 'unknown'.

   Examples:
       >>> from datetime import timedelta
       >>> get_integer_time(timedelta(days=14))
       (2, 'weeks')
       >>> get_integer_time(timedelta(hours=5))
       (5, 'hours')
       >>> get_integer_time(timedelta(milliseconds=1000))
       (1, 'seconds')
       >>> get_integer_time(timedelta(days=0.001))
       (1, 'unknown')



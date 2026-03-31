neural_lam.models.base_hi_graph_model
=====================================

.. py:module:: neural_lam.models.base_hi_graph_model




Module Contents
---------------

.. py:class:: BaseHiGraphModel(args, config: neural_lam.config.NeuralLAMConfig, datastore: neural_lam.datastore.BaseDatastore)

   Bases: :py:obj:`neural_lam.models.base_graph_model.BaseGraphModel`


   Base class for hierarchical graph models.


   .. py:attribute:: num_levels


   .. py:attribute:: level_mesh_sizes


   .. py:attribute:: mesh_embedders


   .. py:attribute:: mesh_same_embedders


   .. py:attribute:: mesh_up_embedders


   .. py:attribute:: mesh_down_embedders


   .. py:attribute:: mesh_init_gnns


   .. py:attribute:: mesh_read_gnns


   .. py:method:: get_num_mesh()

      Compute number of mesh nodes from loaded features,
      and number of mesh nodes that should be ignored in encoding/decoding



   .. py:method:: embedd_mesh_nodes()

      Embed static mesh features
      This embeds only bottom level, rest is done at beginning of
      processing step
      Returns tensor of shape (num_mesh_nodes[0], d_h)



   .. py:method:: process_step(mesh_rep)

      Process step of embedd-process-decode framework
      Processes the representation on the mesh, possible in multiple steps

      mesh_rep: has shape (B, num_mesh_nodes, d_h)
      Returns mesh_rep: (B, num_mesh_nodes, d_h)



   .. py:method:: hi_processor_step(mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep)
      :abstractmethod:


      Internal processor step of hierarchical graph models.
      Between mesh init and read out.

      Each input is list with representations, each with shape

      mesh_rep_levels: (B, num_mesh_nodes[l], d_h)
      mesh_same_rep: (B, M_same[l], d_h)
      mesh_up_rep: (B, M_up[l -> l+1], d_h)
      mesh_down_rep: (B, M_down[l <- l+1], d_h)

      Returns same lists




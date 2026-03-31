neural_lam.models.hi_lam_parallel
=================================

.. py:module:: neural_lam.models.hi_lam_parallel




Module Contents
---------------

.. py:class:: HiLAMParallel(args, config: neural_lam.config.NeuralLAMConfig, datastore: neural_lam.datastore.BaseDatastore)

   Bases: :py:obj:`neural_lam.models.base_hi_graph_model.BaseHiGraphModel`


   Version of HiLAM where all message passing in the hierarchical mesh (up,
   down, inter-level) is ran in parallel.

   This is a somewhat simpler alternative to the sequential message passing
   of Hi-LAM.


   .. py:attribute:: edge_split_sections


   .. py:method:: hi_processor_step(mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep)

      Internal processor step of hierarchical graph models.
      Between mesh init and read out.

      Each input is list with representations, each with shape

      mesh_rep_levels: (B, N_mesh[l], d_h)
      mesh_same_rep: (B, M_same[l], d_h)
      mesh_up_rep: (B, M_up[l -> l+1], d_h)
      mesh_down_rep: (B, M_down[l <- l+1], d_h)

      Returns same lists




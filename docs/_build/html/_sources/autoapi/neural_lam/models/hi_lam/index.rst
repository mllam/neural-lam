neural_lam.models.hi_lam
========================

.. py:module:: neural_lam.models.hi_lam




Module Contents
---------------

.. py:class:: HiLAM(args, config: neural_lam.config.NeuralLAMConfig, datastore: neural_lam.datastore.BaseDatastore)

   Bases: :py:obj:`neural_lam.models.base_hi_graph_model.BaseHiGraphModel`


   Hierarchical graph model with message passing that goes sequentially down
   and up the hierarchy during processing.
   The Hi-LAM model from Oskarsson et al. (2023)


   .. py:attribute:: mesh_down_gnns


   .. py:attribute:: mesh_down_same_gnns


   .. py:attribute:: mesh_up_gnns


   .. py:attribute:: mesh_up_same_gnns


   .. py:method:: make_same_gnns(args)

      Make intra-level GNNs.



   .. py:method:: make_up_gnns(args)

      Make GNNs for processing steps up through the hierarchy.



   .. py:method:: make_down_gnns(args)

      Make GNNs for processing steps down through the hierarchy.



   .. py:method:: mesh_down_step(mesh_rep_levels, mesh_same_rep, mesh_down_rep, down_gnns, same_gnns)

      Run down-part of vertical processing, sequentially alternating between
      processing using down edges and same-level edges.



   .. py:method:: mesh_up_step(mesh_rep_levels, mesh_same_rep, mesh_up_rep, up_gnns, same_gnns)

      Run up-part of vertical processing, sequentially alternating between
      processing using up edges and same-level edges.



   .. py:method:: hi_processor_step(mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep)

      Internal processor step of hierarchical graph models.
      Between mesh init and read out.

      Each input is list with representations, each with shape

      mesh_rep_levels: (B, N_mesh[l], d_h)
      mesh_same_rep: (B, M_same[l], d_h)
      mesh_up_rep: (B, M_up[l -> l+1], d_h)
      mesh_down_rep: (B, M_down[l <- l+1], d_h)

      Returns same lists




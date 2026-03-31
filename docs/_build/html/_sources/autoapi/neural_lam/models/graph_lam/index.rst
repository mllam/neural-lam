neural_lam.models.graph_lam
===========================

.. py:module:: neural_lam.models.graph_lam




Module Contents
---------------

.. py:class:: GraphLAM(args, config: neural_lam.config.NeuralLAMConfig, datastore: neural_lam.datastore.BaseDatastore)

   Bases: :py:obj:`neural_lam.models.base_graph_model.BaseGraphModel`


   Full graph-based LAM model that can be used with different
   (non-hierarchical )graphs. Mainly based on GraphCast, but the model from
   Keisler (2022) is almost identical. Used for GC-LAM and L1-LAM in
   Oskarsson et al. (2023).


   .. py:attribute:: mesh_embedder


   .. py:attribute:: m2m_embedder


   .. py:attribute:: processor


   .. py:method:: get_num_mesh()

      Compute number of mesh nodes from loaded features,
      and number of mesh nodes that should be ignored in encoding/decoding



   .. py:method:: embedd_mesh_nodes()

      Embed static mesh features
      Returns tensor of shape (N_mesh, d_h)



   .. py:method:: process_step(mesh_rep)

      Process step of embedd-process-decode framework
      Processes the representation on the mesh, possible in multiple steps

      mesh_rep: has shape (B, N_mesh, d_h)
      Returns mesh_rep: (B, N_mesh, d_h)




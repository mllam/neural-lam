neural_lam.models.base_graph_model
==================================

.. py:module:: neural_lam.models.base_graph_model




Module Contents
---------------

.. py:class:: BaseGraphModel(args, config: neural_lam.config.NeuralLAMConfig, datastore: neural_lam.datastore.BaseDatastore)

   Bases: :py:obj:`neural_lam.models.ar_model.ARModel`


   Base (abstract) class for graph-based models building on
   the encode-process-decode idea.


   .. py:attribute:: mlp_blueprint_end


   .. py:attribute:: grid_embedder


   .. py:attribute:: g2m_embedder


   .. py:attribute:: m2g_embedder


   .. py:attribute:: g2m_gnn


   .. py:attribute:: encoding_grid_mlp


   .. py:attribute:: m2g_gnn


   .. py:attribute:: output_map


   .. py:method:: prepare_clamping_params(config: neural_lam.config.NeuralLAMConfig, datastore: neural_lam.datastore.BaseDatastore)

      Prepare parameters for clamping predicted values to valid range



   .. py:method:: get_clamped_new_state(state_delta, prev_state)

      Clamp prediction to valid range supplied in config
      Returns the clamped new state after adding delta to original state

      Instead of the new state being computed as
      $X_{t+1} = X_t + \delta = X_t + model(\{X_t,X_{t-1},...\}, forcing)$
      The clamped values will be
      $f(f^{-1}(X_t) + model(\{X_t, X_{t-1},... \}, forcing))$
      Which means the model will learn to output values in the range of the
      inverse clamping function

      state_delta: (B, num_grid_nodes, feature_dim)
      prev_state: (B, num_grid_nodes, feature_dim)



   .. py:method:: get_num_mesh()
      :abstractmethod:


      Compute number of mesh nodes from loaded features,
      and number of mesh nodes that should be ignored in encoding/decoding



   .. py:method:: embedd_mesh_nodes()
      :abstractmethod:


      Embed static mesh features
      Returns tensor of shape (num_mesh_nodes, d_h)



   .. py:method:: process_step(mesh_rep)
      :abstractmethod:


      Process step of embedd-process-decode framework
      Processes the representation on the mesh, possible in multiple steps

      mesh_rep: has shape (B, num_mesh_nodes, d_h)
      Returns mesh_rep: (B, num_mesh_nodes, d_h)



   .. py:method:: predict_step(prev_state, prev_prev_state, forcing)

      Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
      prev_state: (B, num_grid_nodes, feature_dim), X_t
      prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
      forcing: (B, num_grid_nodes, forcing_dim)




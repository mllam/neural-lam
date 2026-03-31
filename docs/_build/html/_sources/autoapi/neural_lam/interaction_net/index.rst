neural_lam.interaction_net
==========================

.. py:module:: neural_lam.interaction_net




Module Contents
---------------

.. py:class:: InteractionNet(edge_index, input_dim, update_edges=True, hidden_layers=1, hidden_dim=None, edge_chunk_sizes=None, aggr_chunk_sizes=None, aggr='sum')

   Bases: :py:obj:`torch_geometric.nn.MessagePassing`


   Implementation of a generic Interaction Network,
   from Battaglia et al. (2016)


   .. py:attribute:: num_rec


   .. py:attribute:: update_edges
      :value: True



   .. py:method:: forward(send_rep, rec_rep, edge_rep)

      Apply interaction network to update the representations of receiver
      nodes, and optionally the edge representations.

      send_rep: (N_send, d_h), vector representations of sender nodes
      rec_rep: (N_rec, d_h), vector representations of receiver nodes
      edge_rep: (M, d_h), vector representations of edges used

      Returns:
      rec_rep: (N_rec, d_h), updated vector representations of receiver nodes
      (optionally) edge_rep: (M, d_h), updated vector representations
          of edges



   .. py:method:: message(x_j, x_i, edge_attr)

      Compute messages from node j to node i.



   .. py:method:: aggregate(inputs, index, ptr, dim_size)

      Overridden aggregation function to:
      * return both aggregated and original messages,
      * only aggregate to number of receiver nodes.



.. py:class:: SplitMLPs(mlps, chunk_sizes)

   Bases: :py:obj:`torch.nn.Module`


   Module that feeds chunks of input through different MLPs.
   Split up input along dim -2 using given chunk sizes and feeds
   each chunk through separate MLPs.


   .. py:attribute:: mlps


   .. py:attribute:: chunk_sizes


   .. py:method:: forward(x)

      Chunk up input and feed through MLPs

      x: (..., N, d), where N = sum(chunk_sizes)

      Returns:
      joined_output: (..., N, d), concatenated results from the MLPs




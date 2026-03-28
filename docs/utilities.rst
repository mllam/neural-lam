Utilities and Tools
===================

Graph Creation and Visualization
---------------------------------

Create graphs for your domain:

.. code-block:: python

   from neural_lam.create_graph import create_hierarchical_graph
   
   graph = create_hierarchical_graph(
       boundary_path="domain_boundary.shp",
       config=graph_config
   )

Visualize graph structure:

.. code-block:: python

   from neural_lam.plot_graph import plot_graph
   
   plot_graph(graph, output_path="graph.png")

Visualization Tools
-------------------

Plotting predictions vs observations:

.. code-block:: python

   from neural_lam.vis import plot_prediction
   
   plot_prediction(
       prediction=pred,
       target=target,
       variable="temperature",
       lead_time=24
   )

Metrics Computation
-------------------

Built-in evaluation metrics:

.. code-block:: python

   from neural_lam.metrics import compute_metrics
   
   metrics = compute_metrics(
       predictions=pred,
       observations=obs,
       variables=variable_names
   )

Common metrics include:

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Pattern Correlation
- Spectral analysis

Loss Weighting
--------------

Apply variable-specific weights during training:

.. code-block:: python

   from neural_lam.loss_weighting import create_loss_weights
   
   weights = create_loss_weights(
       config=weight_config,
       variable_names=vars
   )

Configuration Utilities
----------------------

Load and manage configurations:

.. code-block:: python

   from neural_lam.config import load_config
   
   config = load_config("config.yaml")

Custom Loggers
--------------

Track experiments with custom loggers:

.. code-block:: python

   from neural_lam.custom_loggers import CustomMLFlowLogger
   
   logger = CustomMLFlowLogger(experiment_name="my_experiment")

Interaction Networks
--------------------

Graph attention and interaction mechanisms:

.. code-block:: python

   from neural_lam.interaction_net import InteractionNetwork
   
   interaction_net = InteractionNetwork(
       node_dim=hidden_dim,
       edge_dim=edge_dim
   )

Data Utility Functions
----------------------

Common data operations in ``utils.py``:

- Buffer list management
- Index normalization
- Edge index manipulation
- Device management utilities

See :doc:`api/utilities` for complete function signatures.

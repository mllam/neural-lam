Model Architectures
===================

Overview
--------

Neural-LAM provides several graph-based neural network architectures for weather prediction:

1. **Graph LAM**: Graph-based model from Keisler (2022)
2. **GraphCast**: Implementation of the model by Lam et al. (2023)
3. **Hierarchical LAM**: Multi-level graph neural network from Oskarsson et al. (2023)

All models inherit from ``BaseGraphModel`` or ``BaseHiGraphModel`` for hierarchical models.

Graph LAM
---------

The Graph LAM model uses a single-level graph representation of the weather domain.

Reference: Keisler, R. (2022). "Forecasting Global Weather with Graph Neural Networks"

Key Features:
- Single-level graph representation
- Message-passing neural network
- Suitable for small to medium-sized domains

Usage:

.. code-block:: python

   from neural_lam.models import GraphLAM
   
   model = GraphLAM(model_config)

Hierarchical LAM
----------------

The Hierarchical LAM model uses multi-level graph representations.

Reference: Oskarsson et al. (2023). "Graph-based Neural Weather Prediction for Limited Area Modeling"

Key Features:
- Multi-level hierarchical graphs
- Scalable to larger domains
- Improved efficiency through hierarchical processing
- Supports both sequential and parallel processing

Sequential Processing:

.. code-block:: python

   from neural_lam.models import HiLAM
   
   model = HiLAM(model_config, parallel=False)

Parallel Processing:

.. code-block:: python

   from neural_lam.models import HiLAMParallel
   
   model = HiLAMParallel(model_config)

Model Configuration
-------------------

Models are configured using dataclass-based configuration objects. Key configuration parameters:

- ``num_hidden_layers``: Number of hidden layers
- ``hidden_dim``: Hidden dimension size
- ``num_conv_layers``: Number of graph convolution layers
- ``num_heads``: Number of attention heads (if using attention)
- ``dropout``: Dropout rate for regularization

See :doc:`api/models` for detailed API documentation of each model.

Common Methods
--------------

All models implement standard PyTorch Lightning interfaces:

- ``forward()``: Perform a forward pass
- ``training_step()``: Training step
- ``validation_step()``: Validation step
- ``configure_optimizers()``: Setup optimizer and scheduler

Best Practices
--------------

1. **Model Selection**: Choose Graph LAM for simplicity, Hierarchical LAM for scalability
2. **Hardware**: Use GPU for faster training (CUDA recommended)
3. **Checkpointing**: Save model checkpoints regularly during training
4. **Validation**: Use validation metrics to monitor model overfitting

For more details, see the :doc:`training` guide.

Overview
========

Neural-LAM is a repository of graph-based neural weather prediction models for Limited Area Modeling (LAM).

Key Features
------------

* **Graph-based Neural Networks**: Implements state-of-the-art graph neural network architectures for weather prediction
* **Multiple Model Variants**: Contains implementations of:
  
  - Graph LAM (Keisler, 2022)
  - GraphCast (Lam et al., 2023)
  - Hierarchical LAM (Oskarsson et al., 2023)

* **Modular Design**: Models, graphs, and data are stored separately for easy customization
* **PyTorch & PyTorch Lightning**: Built on modern deep learning frameworks
* **Graph Neural Networks via PyG**: Uses PyTorch Geometric for scalable GNN implementations
* **Experimental Tracking**: Integration with Weights & Biases for tracking experiments

Modularity
----------

The Neural-LAM code is designed to be modular with the following components:

1. **Models**: Different neural network architectures for weather prediction
2. **Graphs**: Graph structures connecting grid points
3. **Data**: Weather datasets and data handling utilities

While highly modular, some constraints apply:

* The graph must be compatible with the model (e.g., hierarchical models require hierarchical graphs)
* Both graph and data are specific to the limited area under consideration

Architecture
------------

.. image:: https://raw.githubusercontent.com/mllam/neural-lam/main/figures/neural_lam_setup.png

Publications
------------

If you use Neural-LAM in your work, please cite the relevant papers:

**Graph-based Neural Weather Prediction for Limited Area Modeling** (Oskarsson et al., 2023)

.. code-block:: bibtex

   @inproceedings{oskarsson2023graphbased,
       title={Graph-based Neural Weather Prediction for Limited Area Modeling},
       author={Oskarsson, Joel and Landelius, Tomas and Lindsten, Fredrik},
       booktitle={NeurIPS 2023 Workshop on Tackling Climate Change with Machine Learning},
       year={2023}
   }

**Probabilistic Weather Forecasting with Hierarchical Graph Neural Networks** (Oskarsson et al., 2024)

.. code-block:: bibtex

   @inproceedings{oskarsson2024probabilistic,
       title={Probabilistic Weather Forecasting with Hierarchical Graph Neural Networks},
       author={Oskarsson, Joel and Landelius, Tomas and Deisenroth, Marc Peter and Lindsten, Fredrik},
       booktitle={Advances in Neural Information Processing Systems},
       volume={37},
       year={2024}
   }

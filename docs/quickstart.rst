Quick Start
===========

Basic Usage
-----------

Loading a Pre-trained Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from neural_lam.models import GraphLAM
   
   model = GraphLAM.load_from_checkpoint("path/to/checkpoint.ckpt")
   model.eval()

Making Predictions
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   with torch.no_grad():
       predictions = model(batch)

Working with Datasets
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neural_lam import WeatherDataset
   
   dataset = WeatherDataset(
       config_path="path/to/datastore.yaml",
       dataset_type="train"
   )
   
   for batch in dataset:
       print(batch.keys())

Training a Model
----------------

Training workflow using PyTorch Lightning:

.. code-block:: python

   import pytorch_lightning as pl
   from neural_lam.models import GraphLAM
   from neural_lam import WeatherDataset
   
   model = GraphLAM(config=model_config)
   
   dataset = WeatherDataset(config_path="datastore.yaml")
   
   trainer = pl.Trainer(
       max_epochs=100,
       accelerator="gpu",
       devices=1
   )
   
   trainer.fit(model, dataset)

Configuration
~~~~~~~~~~~~~

Neural-LAM uses YAML configuration files for datasets and models. See the example configurations in the test data directory.

Graph Creation
--------------

Create a graph for your domain:

.. code-block:: python

   from neural_lam.create_graph import create_hierarchical_graph
   
   graph = create_hierarchical_graph(
       boundary_path="path/to/boundary.shp",
       config_path="path/to/graph_config.yaml"
   )

Evaluation
----------

Evaluate a trained model:

.. code-block:: python

   import pytorch_lightning as pl
   from neural_lam.models import GraphLAM
   
   model = GraphLAM.load_from_checkpoint("checkpoint.ckpt")
   trainer = pl.Trainer(accelerator="gpu")
   
   metrics = trainer.test(model, dataloaders=test_loader)

Next Steps
----------

- Explore the :doc:`api` documentation for detailed function references
- Check :doc:`models` for detailed information about available models
- Read :doc:`data` for data preparation and handling
- Visit `GitHub examples <https://github.com/mllam/neural-lam>`_ for more complex examples

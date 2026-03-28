Training Models
===============

Overview
--------

Neural-LAM uses PyTorch Lightning for training, providing a clean interface for managing:

- Training loops
- Validation and testing
- Checkpointing
- Distributed training
- Experiment logging

Basic Training
--------------

Train a model with default settings:

.. code-block:: python

   import pytorch_lightning as pl
   from neural_lam.models import GraphLAM
   from neural_lam import WeatherDataset
   from torch.utils.data import DataLoader
   
   model = GraphLAM(config)
   
   train_dataset = WeatherDataset(
       config_path="datastore.yaml",
       dataset_type="train"
   )
   val_dataset = WeatherDataset(
       config_path="datastore.yaml",
       dataset_type="val"
   )
   
   train_loader = DataLoader(train_dataset, batch_size=32)
   val_loader = DataLoader(val_dataset, batch_size=32)
   
   trainer = pl.Trainer(
       max_epochs=100,
       accelerator="gpu",
       devices=1
   )
   
   trainer.fit(model, train_loader, val_loader)

Configuration Management
------------------------

Define training configuration:

.. code-block:: python

   from neural_lam.config import ModelConfig
   
   config = ModelConfig(
       num_layers=4,
       hidden_dim=128,
       num_heads=8,
       dropout=0.1,
       learning_rate=1e-3
   )

Experiment Logging
------------------

Log experiments with Weights & Biases:

.. code-block:: python

   from pytorch_lightning.loggers import WandbLogger
   
   logger = WandbLogger(
       project="neural-lam",
       name="experiment-v1"
   )
   
   trainer = pl.Trainer(logger=logger)

Advanced Training Options
-------------------------

Early Stopping
~~~~~~~~~~~~~~

.. code-block:: python

   from pytorch_lightning.callbacks import EarlyStopping
   
   early_stop = EarlyStopping(
       monitor="val_loss",
       patience=10
   )
   
   trainer = pl.Trainer(callbacks=[early_stop])

Learning Rate Scheduling
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def configure_optimizers(self):
       optimizer = torch.optim.Adam(self.parameters())
       scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
           optimizer,
           T_max=self.trainer.max_epochs
       )
       return {
           "optimizer": optimizer,
           "lr_scheduler": {
               "scheduler": scheduler,
               "interval": "epoch"
           }
       }

Multi-GPU Training
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   trainer = pl.Trainer(
       accelerator="gpu",
       devices=4,
       strategy="ddp"
   )

Checkpointing
~~~~~~~~~~~~~

.. code-block:: python

   from pytorch_lightning.callbacks import ModelCheckpoint
   
   checkpoint_callback = ModelCheckpoint(
       monitor="val_loss",
       save_top_k=3,
       mode="min"
   )
   
   trainer = pl.Trainer(callbacks=[checkpoint_callback])

Loss Functions
--------------

Weighted MSE Loss:

.. code-block:: python

   from neural_lam.loss_weighting import WeightedMSELoss
   
   loss_fn = WeightedMSELoss(weight_dict)

Custom loss functions can be implemented by subclassing PyTorch's ``nn.Module``.

Validation and Testing
----------------------

Validate model:

.. code-block:: python

   trainer.validate(model, val_loader)

Test final performance:

.. code-block:: python

   test_dataset = WeatherDataset(
       config_path="datastore.yaml",
       dataset_type="test"
   )
   test_loader = DataLoader(test_dataset, batch_size=32)
   
   results = trainer.test(model, test_loader)

Inference
---------

Load trained model and make predictions:

.. code-block:: python

   model = GraphLAM.load_from_checkpoint("checkpoint.ckpt")
   model.eval()
   
   with torch.no_grad():
       predictions = model(batch)

Best Practices
--------------

1. **Batch Size**: Start with moderate batch sizes (32-64), increase if GPU memory allows
2. **Learning Rate**: Use 1e-3 to 1e-4 for most models
3. **Validation Frequency**: Validate every epoch or every N batches
4. **Checkpointing**: Save best model based on validation loss
5. **Early Stopping**: Use patience of 10-20 epochs
6. **Data Augmentation**: Consider temporal/spatial augmentation for robustness
7. **Mixed Precision**: Enable for faster training on modern GPUs

Troubleshooting
---------------

**Out of Memory**: Reduce batch size or model hidden dimension

**Training is slow**: Use mixed precision training or multi-GPU setup

**Model overfitting**: Increase dropout, add regularization, or use more data

**Unstable training**: Reduce learning rate or use gradient clipping

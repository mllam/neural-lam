neural_lam.custom_loggers
=========================

.. py:module:: neural_lam.custom_loggers




Module Contents
---------------

.. py:class:: CustomMLFlowLogger(experiment_name, tracking_uri, run_name)

   Bases: :py:obj:`pytorch_lightning.loggers.MLFlowLogger`


   Custom MLFlow logger that adds the `log_image()` functionality not
   present in the default implementation from pytorch-lightning as
   of version `2.0.3` at least.


   .. py:property:: save_dir

      Returns the directory where the MLFlow artifacts are saved.
      Used to define the path to save output when using the logger.

      Returns
      -------
      str
          Path to the directory where the artifacts are saved.



   .. py:method:: log_image(key, images, step=None)

      Log a matplotlib figure as an image to MLFlow

      key: str
          Key to log the image under
      images: list
          List of matplotlib figures to log
      step: Union[int, None]
          Step to log the image under. If None, logs under the key directly




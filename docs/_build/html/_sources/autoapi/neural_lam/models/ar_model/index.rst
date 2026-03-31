neural_lam.models.ar_model
==========================

.. py:module:: neural_lam.models.ar_model




Module Contents
---------------

.. py:class:: ARModel(args, config: neural_lam.config.NeuralLAMConfig, datastore: neural_lam.datastore.BaseDatastore)

   Bases: :py:obj:`pytorch_lightning.LightningModule`


   Generic auto-regressive weather model.
   Abstract class that can be extended.


   .. py:attribute:: args


   .. py:attribute:: feature_weights


   .. py:attribute:: output_std


   .. py:attribute:: grid_dim


   .. py:attribute:: loss


   .. py:attribute:: val_metrics
      :type:  Dict[str, List]


   .. py:attribute:: test_metrics
      :type:  Dict[str, List]


   .. py:attribute:: restore_opt


   .. py:attribute:: n_example_pred


   .. py:attribute:: plotted_examples
      :value: 0



   .. py:attribute:: spatial_loss_maps
      :type:  List[Any]
      :value: []



   .. py:method:: configure_optimizers()

      Choose what optimizers and learning-rate schedulers to use in your optimization. Normally you'd need one.
      But in the case of GANs or similar you might have multiple. Optimization with multiple optimizers only works in
      the manual optimization mode.

      Return:
          Any of these 6 options.

          - **Single optimizer**.
          - **List or Tuple** of optimizers.
          - **Two lists** - The first list has multiple optimizers, and the second has multiple LR schedulers
            (or multiple ``lr_scheduler_config``).
          - **Dictionary**, with an ``"optimizer"`` key, and (optionally) a ``"lr_scheduler"``
            key whose value is a single LR scheduler or ``lr_scheduler_config``.
          - **None** - Fit will run without any optimizer.

      The ``lr_scheduler_config`` is a dictionary which contains the scheduler and its associated configuration.
      The default configuration is shown below.

      .. code-block:: python

          lr_scheduler_config = {
              # REQUIRED: The scheduler instance
              "scheduler": lr_scheduler,
              # The unit of the scheduler's step size, could also be 'step'.
              # 'epoch' updates the scheduler on epoch end whereas 'step'
              # updates it after a optimizer update.
              "interval": "epoch",
              # How many epochs/steps should pass between calls to
              # `scheduler.step()`. 1 corresponds to updating the learning
              # rate after every epoch/step.
              "frequency": 1,
              # Metric to monitor for schedulers like `ReduceLROnPlateau`
              "monitor": "val_loss",
              # If set to `True`, will enforce that the value specified 'monitor'
              # is available when the scheduler is updated, thus stopping
              # training if not found. If set to `False`, it will only produce a warning
              "strict": True,
              # If using the `LearningRateMonitor` callback to monitor the
              # learning rate progress, this keyword can be used to specify
              # a custom logged name
              "name": None,
          }

      When there are schedulers in which the ``.step()`` method is conditioned on a value, such as the
      :class:`torch.optim.lr_scheduler.ReduceLROnPlateau` scheduler, Lightning requires that the
      ``lr_scheduler_config`` contains the keyword ``"monitor"`` set to the metric name that the scheduler
      should be conditioned on.

      .. testcode::

          # The ReduceLROnPlateau scheduler requires a monitor
          def configure_optimizers(self):
              optimizer = Adam(...)
              return {
                  "optimizer": optimizer,
                  "lr_scheduler": {
                      "scheduler": ReduceLROnPlateau(optimizer, ...),
                      "monitor": "metric_to_track",
                      "frequency": "indicates how often the metric is updated",
                      # If "monitor" references validation metrics, then "frequency" should be set to a
                      # multiple of "trainer.check_val_every_n_epoch".
                  },
              }


          # In the case of two optimizers, only one using the ReduceLROnPlateau scheduler
          def configure_optimizers(self):
              optimizer1 = Adam(...)
              optimizer2 = SGD(...)
              scheduler1 = ReduceLROnPlateau(optimizer1, ...)
              scheduler2 = LambdaLR(optimizer2, ...)
              return (
                  {
                      "optimizer": optimizer1,
                      "lr_scheduler": {
                          "scheduler": scheduler1,
                          "monitor": "metric_to_track",
                      },
                  },
                  {"optimizer": optimizer2, "lr_scheduler": scheduler2},
              )

      Metrics can be made available to monitor by simply logging it using
      ``self.log('metric_to_track', metric_val)`` in your :class:`~pytorch_lightning.core.LightningModule`.

      Note:
          Some things to know:

          - Lightning calls ``.backward()`` and ``.step()`` automatically in case of automatic optimization.
          - If a learning rate scheduler is specified in ``configure_optimizers()`` with key
            ``"interval"`` (default "epoch") in the scheduler configuration, Lightning will call
            the scheduler's ``.step()`` method automatically in case of automatic optimization.
          - If you use 16-bit precision (``precision=16``), Lightning will automatically handle the optimizer.
          - If you use :class:`torch.optim.LBFGS`, Lightning handles the closure function automatically for you.
          - If you use multiple optimizers, you will have to switch to 'manual optimization' mode and step them
            yourself.
          - If you need to control how often the optimizer steps, override the :meth:`optimizer_step` hook.




   .. py:property:: interior_mask_bool

      Get the interior mask as a boolean (N,) mask.



   .. py:method:: expand_to_batch(x, batch_size)
      :staticmethod:


      Expand tensor with initial batch dimension



   .. py:method:: predict_step(prev_state, prev_prev_state, forcing)
      :abstractmethod:


      Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
      prev_state: (B, num_grid_nodes, feature_dim), X_t prev_prev_state: (B,
      num_grid_nodes, feature_dim), X_{t-1} forcing: (B, num_grid_nodes,
      forcing_dim)



   .. py:method:: unroll_prediction(init_states, forcing_features, true_states)

      Roll out prediction taking multiple autoregressive steps with model
      init_states: (B, 2, num_grid_nodes, d_f) forcing_features: (B,
      pred_steps, num_grid_nodes, d_static_f) true_states: (B, pred_steps,
      num_grid_nodes, d_f)



   .. py:method:: common_step(batch)

      Predict on single batch batch consists of: init_states: (B, 2,
      num_grid_nodes, d_features) target_states: (B, pred_steps,
      num_grid_nodes, d_features) forcing_features: (B, pred_steps,
      num_grid_nodes, d_forcing),
          where index 0 corresponds to index 1 of init_states



   .. py:method:: training_step(batch)

      Train on single batch



   .. py:method:: all_gather_cat(tensor_to_gather)

      Gather tensors across all ranks, and concatenate across dim. 0 (instead
      of stacking in new dim. 0)

      tensor_to_gather: (d1, d2, ...), distributed over K ranks

      returns:
          - single-device strategies: (d1, d2, ...)
          - multi-device strategies: (K*d1, d2, ...)



   .. py:method:: validation_step(batch, batch_idx)

      Run validation on single batch



   .. py:method:: on_validation_epoch_end()

      Compute val metrics at the end of val epoch



   .. py:method:: test_step(batch, batch_idx)

      Run test on single batch



   .. py:method:: plot_examples(batch, n_examples, split, prediction=None)

      Plot the first n_examples forecasts from batch

      batch: batch with data to plot corresponding forecasts for n_examples:
      number of forecasts to plot prediction: (B, pred_steps, num_grid_nodes,
      d_f), existing prediction.
          Generate if None.



   .. py:method:: create_metric_log_dict(metric_tensor, prefix, metric_name)

      Put together a dict with everything to log for one metric. Also saves
      plots as pdf and csv if using test prefix.

      metric_tensor: (pred_steps, d_f), metric values per time and variable
      prefix: string, prefix to use for logging metric_name: string, name of
      the metric

      Return: log_dict: dict with everything to log for given metric



   .. py:method:: aggregate_and_plot_metrics(metrics_dict, prefix)

      Aggregate and create error map plots for all metrics in metrics_dict

      metrics_dict: dictionary with metric_names and list of tensors
          with step-evals.
      prefix: string, prefix to use for logging



   .. py:method:: on_test_epoch_end()

      Compute test metrics and make plots at the end of test epoch. Will
      gather stored tensors and perform plotting and logging on rank 0.



   .. py:method:: on_load_checkpoint(checkpoint)

      Perform any changes to state dict before loading checkpoint




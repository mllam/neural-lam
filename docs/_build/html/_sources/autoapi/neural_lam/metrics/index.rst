neural_lam.metrics
==================

.. py:module:: neural_lam.metrics






Module Contents
---------------

.. py:function:: get_metric(metric_name)

   Get a defined metric with given name

   metric_name: str, name of the metric

   Returns:
   metric: function implementing the metric


.. py:function:: mask_and_reduce_metric(metric_entry_vals, mask, average_grid, sum_vars)

   Masks and (optionally) reduces entry-wise metric values

   (...,) is any number of batch dimensions, potentially different
       but broadcastable
   metric_entry_vals: (..., N, d_state), prediction
   mask: (N,), boolean mask describing which grid nodes to use in metric
   average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
   sum_vars: boolean, if variable dimension -1 should be reduced (sum
       over d_state)

   Returns:
   metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
   depending on reduction arguments.


.. py:function:: wmse(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True)

   Weighted Mean Squared Error

   (...,) is any number of batch dimensions, potentially different
       but broadcastable
   pred: (..., N, d_state), prediction
   target: (..., N, d_state), target
   pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
   mask: (N,), boolean mask describing which grid nodes to use in metric
   average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
   sum_vars: boolean, if variable dimension -1 should be reduced (sum
       over d_state)

   Returns:
   metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
   depending on reduction arguments.


.. py:function:: mse(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True)

   (Unweighted) Mean Squared Error

   (...,) is any number of batch dimensions, potentially different
       but broadcastable
   pred: (..., N, d_state), prediction
   target: (..., N, d_state), target
   pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
   mask: (N,), boolean mask describing which grid nodes to use in metric
   average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
   sum_vars: boolean, if variable dimension -1 should be reduced (sum
       over d_state)

   Returns:
   metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
   depending on reduction arguments.


.. py:function:: wmae(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True)

   Weighted Mean Absolute Error

   (...,) is any number of batch dimensions, potentially different
       but broadcastable
   pred: (..., N, d_state), prediction
   target: (..., N, d_state), target
   pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
   mask: (N,), boolean mask describing which grid nodes to use in metric
   average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
   sum_vars: boolean, if variable dimension -1 should be reduced (sum
       over d_state)

   Returns:
   metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
   depending on reduction arguments.


.. py:function:: mae(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True)

   (Unweighted) Mean Absolute Error

   (...,) is any number of batch dimensions, potentially different
       but broadcastable
   pred: (..., N, d_state), prediction
   target: (..., N, d_state), target
   pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
   mask: (N,), boolean mask describing which grid nodes to use in metric
   average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
   sum_vars: boolean, if variable dimension -1 should be reduced (sum
       over d_state)

   Returns:
   metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
   depending on reduction arguments.


.. py:function:: nll(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True)

   Negative Log Likelihood loss, for isotropic Gaussian likelihood

   (...,) is any number of batch dimensions, potentially different
       but broadcastable
   pred: (..., N, d_state), prediction
   target: (..., N, d_state), target
   pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
   mask: (N,), boolean mask describing which grid nodes to use in metric
   average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
   sum_vars: boolean, if variable dimension -1 should be reduced (sum
       over d_state)

   Returns:
   metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
   depending on reduction arguments.


.. py:function:: crps_gauss(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True)

   (Negative) Continuous Ranked Probability Score (CRPS)
   Closed-form expression based on Gaussian predictive distribution

   (...,) is any number of batch dimensions, potentially different
           but broadcastable
   pred: (..., N, d_state), prediction
   target: (..., N, d_state), target
   pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
   mask: (N,), boolean mask describing which grid nodes to use in metric
   average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
   sum_vars: boolean, if variable dimension -1 should be reduced (sum
       over d_state)

   Returns:
   metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
   depending on reduction arguments.


.. py:data:: DEFINED_METRICS


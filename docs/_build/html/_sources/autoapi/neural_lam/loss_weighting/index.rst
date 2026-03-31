neural_lam.loss_weighting
=========================

.. py:module:: neural_lam.loss_weighting




Module Contents
---------------

.. py:function:: get_manual_state_feature_weights(weighting_config: neural_lam.config.ManualStateFeatureWeighting, datastore: neural_lam.datastore.base.BaseDatastore) -> list[float]

   Return the state feature weights as a list of floats in the order of the
   state features in the datastore.

   Parameters
   ----------
   weighting_config : ManualStateFeatureWeighting
       Configuration object containing the manual state feature weights.
   datastore : BaseDatastore
       Datastore object containing the state features.

   Returns
   -------
   list[float]
       List of floats containing the state feature weights.


.. py:function:: get_uniform_state_feature_weights(datastore: neural_lam.datastore.base.BaseDatastore) -> list[float]

   Return the state feature weights as a list of floats in the order of the
   state features in the datastore.

   The weights are uniform, i.e. 1.0/n_features for each feature.

   Parameters
   ----------
   datastore : BaseDatastore
       Datastore object containing the state features.

   Returns
   -------
   list[float]
       List of floats containing the state feature weights.


.. py:function:: get_state_feature_weighting(config: neural_lam.config.NeuralLAMConfig, datastore: neural_lam.datastore.base.BaseDatastore) -> list[float]

   Return the state feature weights as a list of floats in the order of the
   state features in the datastore. The weights are determined based on the
   configuration in the NeuralLAMConfig object.

   Parameters
   ----------
   config : NeuralLAMConfig
       Configuration object for neural-lam.
   datastore : BaseDatastore
       Datastore object containing the state features.

   Returns
   -------
   list[float]
       List of floats containing the state feature weights.



neural_lam.config
=================

.. py:module:: neural_lam.config








Module Contents
---------------

.. py:class:: DatastoreSelection

   Configuration for selecting a datastore to use with neural-lam.

   Attributes
   ----------
   kind : str
       The kind of datastore to use, currently `mdp` or `npyfilesmeps` are
       implemented.
   config_path : str
       The path to the configuration file for the selected datastore, this is
       assumed to be relative to the configuration file for neural-lam.


   .. py:attribute:: kind
      :type:  str


   .. py:attribute:: config_path
      :type:  str


.. py:class:: ManualStateFeatureWeighting

   Configuration for weighting the state features in the loss function where
   the weights are manually specified.

   Attributes
   ----------
   weights : Dict[str, float]
       Manual weights for the state features.


   .. py:attribute:: weights
      :type:  Dict[str, float]


.. py:class:: UniformFeatureWeighting

   Configuration for weighting the state features in the loss function where
   all state features are weighted equally.


.. py:class:: OutputClamping

   Configuration for clamping the output of the model.

   Attributes
   ----------
   lower : Dict[str, float]
       The minimum value to clamp each output feature to.
   upper : Dict[str, float]
       The maximum value to clamp each output feature to.


   .. py:attribute:: lower
      :type:  Dict[str, float]


   .. py:attribute:: upper
      :type:  Dict[str, float]


.. py:class:: TrainingConfig

   Configuration related to training neural-lam

   Attributes
   ----------
   state_feature_weighting : Union[ManualStateFeatureWeighting,
                                   UnformFeatureWeighting]
       The method to use for weighting the state features in the loss
       function. Defaults to uniform weighting (`UnformFeatureWeighting`, i.e.
       all features are weighted equally).


   .. py:attribute:: state_feature_weighting
      :type:  Union[ManualStateFeatureWeighting, UniformFeatureWeighting]


   .. py:attribute:: output_clamping
      :type:  OutputClamping


.. py:class:: NeuralLAMConfig

   Bases: :py:obj:`dataclass_wizard.JSONWizard`, :py:obj:`dataclass_wizard.YAMLWizard`


   Dataclass for Neural-LAM configuration. This class is used to load and
   store the configuration for using Neural-LAM.

   Attributes
   ----------
   datastore : DatastoreSelection
       The configuration for the datastore to use.
   training : TrainingConfig
       The configuration for training the model.


   .. py:attribute:: datastore
      :type:  DatastoreSelection


   .. py:attribute:: training
      :type:  TrainingConfig


.. py:exception:: InvalidConfigError

   Bases: :py:obj:`Exception`


   Common base class for all non-exit exceptions.


.. py:function:: load_config_and_datastore(config_path: str) -> tuple[NeuralLAMConfig, Union[neural_lam.datastore.MDPDatastore, neural_lam.datastore.NpyFilesDatastoreMEPS]]

   Load the neural-lam configuration and the datastore specified in the
   configuration.

   Parameters
   ----------
   config_path : str
       Path to the Neural-LAM configuration file.

   Returns
   -------
   tuple[NeuralLAMConfig, Union[MDPDatastore, NpyFilesDatastoreMEPS]]
       The Neural-LAM configuration and the loaded datastore.



neural_lam.datastore.npyfilesmeps.config
========================================

.. py:module:: neural_lam.datastore.npyfilesmeps.config




Module Contents
---------------

.. py:class:: Projection

   Represents the projection information for a dataset, including the type
   of projection and its parameters. Capable of creating a cartopy.crs
   projection object.

   Attributes:
       class_name: The class name of the projection, this should be a valid
       cartopy.crs class.
       kwargs: A dictionary of keyword arguments specific to the projection
       type.



   .. py:attribute:: class_name
      :type:  str


   .. py:attribute:: kwargs
      :type:  Dict[str, Any]


.. py:class:: Dataset

   Contains information about the dataset, including variable names, units,
   and descriptions.

   Attributes:
       name: The name of the dataset.
       var_names: A list of variable names in the dataset.
       var_units: A list of units for each variable.
       var_longnames: A list of long, descriptive names for each variable.
       num_forcing_features: The number of forcing features in the dataset.



   .. py:attribute:: name
      :type:  str


   .. py:attribute:: var_names
      :type:  List[str]


   .. py:attribute:: var_units
      :type:  List[str]


   .. py:attribute:: var_longnames
      :type:  List[str]


   .. py:attribute:: num_forcing_features
      :type:  int


   .. py:attribute:: num_timesteps
      :type:  int


   .. py:attribute:: step_length
      :type:  datetime.timedelta


   .. py:attribute:: num_ensemble_members
      :type:  int


   .. py:attribute:: remove_state_features_with_index
      :type:  List[int]
      :value: []



.. py:class:: NpyDatastoreConfig

   Bases: :py:obj:`dataclass_wizard.YAMLWizard`


   Configuration for loading and processing a dataset, including dataset
   details, grid shape, and projection information.

   Attributes:
       dataset: An instance of Dataset containing details about the dataset.
       grid_shape_state: A list representing the shape of the grid state.
       projection: An instance of Projection containing projection details.



   .. py:attribute:: dataset
      :type:  Dataset


   .. py:attribute:: grid_shape_state
      :type:  List[int]


   .. py:attribute:: projection
      :type:  Projection



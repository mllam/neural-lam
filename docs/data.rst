Data Handling
=============

Overview
--------

Neural-LAM provides flexible data handling through pluggable datastore implementations. Data is organized into:

- **State variables**: Model state (e.g., temperature, pressure)
- **Forcing variables**: External forcings (e.g., solar radiation)
- **Static features**: Terrain height, land-sea mask, etc.

Data Sources
------------

The framework supports multiple data sources:

1. **NetCDF-based storage** (NetCDF MDP format)
2. **NPY Files MEPS** (NumPy array storage)
3. **Custom implementations** (extendable base class)

Dataset Structure
-----------------

Data is stored with standardized naming and organization:

- Training/Validation/Test splits
- Normalization statistics (mean, std, diff_mean, diff_std)
- Temporal and spatial dimensions
- Feature-wise metadata

Weather Dataset
---------------

The main interface for accessing data:

.. code-block:: python

   from neural_lam import WeatherDataset
   
   dataset = WeatherDataset(
       config_path="path/to/datastore.yaml",
       dataset_type="train",  # "train", "val", or "test"
       num_time_steps=10
   )

Configuration
~~~~~~~~~~~~~

Dataset configuration is done via YAML files specifying:

- Data location and format
- Variable names and units
- Normalization parameters
- Train/val/test split ratios

Example datastore configuration:

.. code-block:: yaml

   dataset_name: my_weather_data
   root: /path/to/data
   forcing:
     variables: [tcc, tisr]
   state:
     variables: [u10m, v10m, t2m]

Data Preparation
-----------------

For NetCDF data:

.. code-block:: python

   from neural_lam.datastore.mdp import NetCDFMDPDataStore
   
   store = NetCDFMDPDataStore(config)
   store.compute_normalization()

For NumPy data:

.. code-block:: python

   from neural_lam.datastore.npyfilesmeps import NpyFilesMEPSStore
   
   store = NpyFilesMEPSStore(config)

Batching and Loading
---------------------

Data is automatically batched for training:

.. code-block:: python

   from torch.utils.data import DataLoader
   
   loader = DataLoader(
       dataset,
       batch_size=32,
       shuffle=True,
       num_workers=4
   )
   
   for batch in loader:
       predictions = model(batch)

Custom Datastores
------------------

Implement custom datastores by extending ``BaseDataStore``:

.. code-block:: python

   from neural_lam.datastore.base import BaseDataStore
   
   class MyDataStore(BaseDataStore):
       def load_sample(self, sample_index):
           pass
       
       def compute_normalization(self):
           pass

Data Normalization
-------------------

Data is normalized using training set statistics:

- **Mean normalization**: $(x - \mu) / \sigma$
- **Difference normalization**: $(x_{t+1} - x_t - \mu_{diff}) / \sigma_{diff}$

Normalization parameters are computed once and reused for all splits.

Best Practices
---------------

1. **Organization**: Keep data in organized directory structures
2. **Metadata**: Include complete variable metadata (units, sources)
3. **Validation**: Always validate data shapes and value ranges
4. **Normalization**: Compute stats on training data only
5. **Splitting**: Use appropriate train/val/test ratios (e.g., 70/15/15)

For existing datasets like DANRA, see the test data examples for structure.

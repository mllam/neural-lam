# Glossary

```{glossary}
ARModel
    The autoregressive base model implemented in PyTorch Lightning.

BaseDatastore
    The abstract base class for data loaders and datastores.

Datastore
    The overarching data handler component for loading weather data.

Encode-Process-Decode
    An architecture style often used in graph neural networks involving encoding inputs into latent graphs, processing them with message passing, and decoding back to the target space.

Forcing Variables
    External variables providing boundary or context conditions to the {term}`Forecaster`.

Forecaster
    The core component or model responsible for generating predictions over time.

GNN
    Graph Neural Network. A type of neural network designed to operate on graph structures.

GraphLAM
    The main Graph-based Limited Area Model implementation.

HiLAM
    Hierarchical Limited Area Model.

HiLAMParallel
    A parallelized version of {term}`HiLAM`.

LAM
    Limited Area Model. A weather prediction model focused on a specific geographic region rather than global scope.

MDPDatastore
    Datastore designed to read Zarr formats via the {term}`mllam-data-prep` module.

Mesh
    The structured or unstructured graph grid onto which the weather data is projected and processed.

mllam-data-prep
    The tool/module responsible for preparing and formatting raw weather data into a format ingestible by Neural-LAM.

PyG
    PyTorch Geometric, a library for deep learning on irregular input data such as graphs.

State Variables
    The set of variables that describe the current internal state of the weather system in the model.

StepPredictor
    A model component designed to predict the next single time step given a current state.

WeatherDataModule
    The PyTorch Lightning data module encapsulating the {term}`WeatherDataset`.

WeatherDataset
    The PyTorch dataset class representing the prepared weather data.
```

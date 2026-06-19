# Data Flow

## Overview

The Neural-LAM data pipeline is designed to efficiently load raw meteorological data and feed it into the models for training and inference. It abstracts data sources, normalizes variables, and prepares batches using PyTorch Lightning.

## Datastore Abstraction

The {py:class}`neural_lam.datastore.BaseDatastore` provides an abstract interface for data access. It defines methods for fetching input and target tensors (`get_xy`), as well as metadata like variable names and units. Subclasses like `MDPDatastore` implement this interface to support specific formats like Zarr.

## WeatherDataset

The {py:class}`neural_lam.weather_dataset.WeatherDataset` acts as a PyTorch wrapper around a `BaseDatastore`. It handles indexing, temporal batching, and any necessary on-the-fly transformations required before the data reaches the Lightning module.

## WeatherDataModule

The {py:class}`neural_lam.weather_dataset.WeatherDataModule` is a PyTorch Lightning DataModule that encapsulates the `WeatherDataset`s for training, validation, and testing splits. It manages the dataloaders and ensures data is correctly distributed across devices during distributed training.

## Configuration

The entire data pipeline is driven by type-safe YAML configurations defined in {py:mod}`neural_lam.config`. Dataclasses define expected paths, batch sizes, and data normalization statistics, allowing easy experimentation without code changes.

## Data Format Requirements

| Component | Format/Shape Expected |
|-----------|-----------------------|
| Model Input | `(batch, sequence_length, num_grid_nodes, num_features)` |
| Target Output | `(batch, sequence_length, num_grid_nodes, num_features)` |
| Static Features | `(num_grid_nodes, num_static_features)` |

```{mermaid}
flowchart TD
    A["Raw Data Files"] -->|Read by| B["BaseDatastore"]
    B -->|get_xy| C["WeatherDataset"]
    C -->|__getitem__| D["DataLoader"]
    D -->|B, T, N, F| E["WeatherDataModule"]
    E -->|Batch| F["Model Forward Pass"]
```

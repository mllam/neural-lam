# Architecture Overview

## System Design

Neural-LAM is designed with a modular architecture that separates data handling, model architecture, and training logic. This separation of concerns allows for easy experimentation with different graph structures, model components, and datasets without requiring extensive changes to the core system.

The core components include a robust data pipeline utilizing an abstract datastore interface, a flexible set of graph-based neural network models (such as `GraphLAM`, `HiLAM`, and `HiLAMParallel`), and a training module built on top of PyTorch Lightning. This design ensures scalability and ease of use for both researchers and practitioners.

## Data Flow

```{mermaid}
flowchart LR
    A["Raw Data<br/>(zarr / numpy)"] --> B["Datastore<br/>(BaseDatastore)"]
    B --> C["WeatherDataset"]
    C --> D["WeatherDataModule<br/>(Lightning)"]
    D --> E["ARModel<br/>(Autoregressive)"]
    E --> F["StepPredictor<br/>(GNN)"]
    F --> G["Predictions"]
    G --> H["Loss & Metrics"]
```

## Module Map

| Module | Responsibility |
|--------|----------------|
| `datastore` | Handles reading from diverse data sources (e.g., Zarr, NetCDF) via the `BaseDatastore` interface. |
| `weather_dataset` | Wraps the datastore in a PyTorch `Dataset` and Lightning `DataModule` for training. |
| `models` | Contains the core neural network architectures (`ARModel`, `BaseGraphModel`, `GraphLAM`, etc.). |
| `create_graph` | Utility to build the hierarchical mesh graphs used by the GNN models. |
| `config` | Manages the YAML-based configuration via dataclasses. |

## Component Interaction

This diagram is **automatically generated** from the Python source code using `pyreverse`, ensuring it never goes stale!

```{eval-rst}
.. mermaid:: ../_static/uml/classes_models.mmd
```

## Key Design Decisions

- **Modular Datastores**: `BaseDatastore` abstracts away data loading intricacies.
- **PyTorch Lightning**: Used to reduce boilerplate and scale training easily.
- **Hierarchical Graphs**: `create_graph` decouples graph structure generation from model logic.
- **Dataclass Configurations**: Type-safe YAML configurations managed by `dataclass-wizard`.

## See Also

- {doc}`data-flow` for detailed data pipeline documentation
- {doc}`models` for model architecture details
- {doc}`graph-construction` for graph creation details

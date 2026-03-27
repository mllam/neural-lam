# Neural-LAM Architecture & Developer Guide


## High-Level Overview

Neural-LAM is a graph-based neural weather prediction system for Limited Area Modeling (LAM).
The core pipeline is: **Data → Graph → Model → Training → Forecast**.

```mermaid
flowchart TD
    A[Raw Input Data\n(xarray / MEPS / NpyFilesDatastore)]
    --> B[Configuration & Datastore Loading\nload_config_and_datastore\nconfig.py]

    B --> C[Graph Construction\ncreate_graph\nMulti-scale / Hierarchical\nRectangular or Triangular]

    C --> D[Graph Neural Network Model\nGraphLAM or HiLAM\nMessage Passing on Graph]

    D --> E[Autoregressive Forecasting\nAR steps\nEnsemble / Probabilistic Extensions\nLBCs handling]

    E --> F[Loss & Optimization\nState feature weighting\nOutput clamping\nTraining loop]

    F --> G[Output & Evaluation\nForecasts + Visualization]

    style A fill:#e3f2fd,stroke:#1976d2
    style G fill:#f3e5f5,stroke:#7b1fa2

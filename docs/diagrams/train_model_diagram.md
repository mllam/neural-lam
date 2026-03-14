```mermaid
%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80}}}%%
flowchart TD

module["train_model"]

subgraph Parent_Class
end

subgraph Base_Class
end


subgraph Imports
    random["random"]
    weather_dataset["weather_dataset"]
    time["time"]
    models["models"]
    json["json"]
    lightning_fabric["lightning_fabric"]
    argparse["argparse"]
    torch["torch"]
    pytorch_lightning["pytorch_lightning"]
    config["config"]
    loguru["loguru"]
end

    random --> module
    weather_dataset --> module
    time --> module
    models --> module
    json --> module
    lightning_fabric --> module
    argparse --> module
    torch --> module
    pytorch_lightning --> module
    config --> module
    loguru --> module

subgraph Methods
end


classDef parent fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#f1f5f9,font-size:16px
classDef base fill:#78350f,stroke:#f59e0b,stroke-width:1px,color:#fde68a,font-size:16px
classDef import fill:#1f2937,stroke:#6b7280,stroke-width:1.5px,color:#e5e7eb,font-size:16px
classDef method fill:#2d043f,stroke:#7c3aed,stroke-width:1.5px,color:#ede9fe,font-size:16px
classDef callNode fill:#064e3b,stroke:#10b981,stroke-width:1.5px,color:#d1fae5,font-size:16px
class random,weather_dataset,time,models,json,lightning_fabric,argparse,torch,pytorch_lightning,config,loguru import

```

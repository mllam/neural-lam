```mermaid
%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80}}}%%
flowchart TD

module["create_graph"]

subgraph Imports
    typing["typing"]
    torch_geometric["torch_geometric"]
    os["os"]
    numpy["numpy"]
    config["config"]
    scipy["scipy"]
    datastore["datastore"]
    matplotlib["matplotlib"]
    argparse["argparse"]
    networkx["networkx"]
    loguru["loguru"]
    torch["torch"]
end

    typing --> module
    torch_geometric --> module
    os --> module
    numpy --> module
    config --> module
    scipy --> module
    datastore --> module
    matplotlib --> module
    argparse --> module
    networkx --> module
    loguru --> module
    torch --> module

classDef parent fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#f1f5f9,font-size:16px
classDef base fill:#78350f,stroke:#f59e0b,stroke-width:1px,color:#fde68a,font-size:16px
classDef import fill:#1f2937,stroke:#6b7280,stroke-width:1.5px,color:#e5e7eb,font-size:16px
classDef method fill:#2d043f,stroke:#7c3aed,stroke-width:1.5px,color:#ede9fe,font-size:16px
classDef callNode fill:#064e3b,stroke:#10b981,stroke-width:1.5px,color:#d1fae5,font-size:16px
class typing,torch_geometric,os,numpy,config,scipy,datastore,matplotlib,argparse,networkx,loguru,torch import
```

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80}}}%%
flowchart TD

module["create_graph"]

subgraph Parent_Class
end

subgraph Base_Class
end


subgraph Imports
    torch["torch"]
    typing["typing"]
    torch_geometric["torch_geometric"]
    os["os"]
    scipy["scipy"]
    argparse["argparse"]
    config["config"]
    datastore["datastore"]
    networkx["networkx"]
    loguru["loguru"]
    matplotlib["matplotlib"]
    numpy["numpy"]
end

    torch --> module
    typing --> module
    torch_geometric --> module
    os --> module
    scipy --> module
    argparse --> module
    config --> module
    datastore --> module
    networkx --> module
    loguru --> module
    matplotlib --> module
    numpy --> module

subgraph Methods
end


classDef parent fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#f1f5f9,font-size:16px
classDef base fill:#78350f,stroke:#f59e0b,stroke-width:1px,color:#fde68a,font-size:16px
classDef import fill:#1f2937,stroke:#6b7280,stroke-width:1.5px,color:#e5e7eb,font-size:16px
classDef method fill:#2d043f,stroke:#7c3aed,stroke-width:1.5px,color:#ede9fe,font-size:16px
classDef callNode fill:#064e3b,stroke:#10b981,stroke-width:1.5px,color:#d1fae5,font-size:16px
class torch,typing,torch_geometric,os,scipy,argparse,config,datastore,networkx,loguru,matplotlib,numpy import

```
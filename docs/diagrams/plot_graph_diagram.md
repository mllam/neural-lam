```mermaid
%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80}}}%%
flowchart TD

module["plot_graph"]

subgraph Parent_Class
end

subgraph Base_Class
end


subgraph Imports
    config["config"]
    plotly["plotly"]
    argparse["argparse"]
    os["os"]
    torch_geometric["torch_geometric"]
    numpy["numpy"]
end

    config --> module
    plotly --> module
    argparse --> module
    os --> module
    torch_geometric --> module
    numpy --> module

subgraph Methods
end


classDef parent fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#f1f5f9,font-size:16px
classDef base fill:#78350f,stroke:#f59e0b,stroke-width:1px,color:#fde68a,font-size:16px
classDef import fill:#1f2937,stroke:#6b7280,stroke-width:1.5px,color:#e5e7eb,font-size:16px
classDef method fill:#2d043f,stroke:#7c3aed,stroke-width:1.5px,color:#ede9fe,font-size:16px
classDef callNode fill:#064e3b,stroke:#10b981,stroke-width:1.5px,color:#d1fae5,font-size:16px
class config,plotly,argparse,os,torch_geometric,numpy import

```

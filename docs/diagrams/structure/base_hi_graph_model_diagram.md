```mermaid
%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80}}}%%
flowchart TD

module["base_hi_graph_model"]

subgraph Parent_Class
    BaseGraphModel["BaseGraphModel"]
end

subgraph Base_Class
    BaseHiGraphModel["BaseHiGraphModel"]
end

    BaseGraphModel --> BaseHiGraphModel

subgraph Imports
    base_graph_model["base_graph_model"]
    datastore["datastore"]
    torch["torch"]
    interaction_net["interaction_net"]
    config["config"]
end

    base_graph_model --> module
    datastore --> module
    torch --> module
    interaction_net --> module
    config --> module
    module --> BaseHiGraphModel

subgraph Methods
    BaseHiGraphModel_get_num_mesh["mesh()"]
    BaseHiGraphModel_embedd_mesh_nodes["nodes()"]
    BaseHiGraphModel_process_step["step()"]
    BaseHiGraphModel_hi_processor_step["step()"]
end

    BaseHiGraphModel --> BaseHiGraphModel_get_num_mesh
    BaseHiGraphModel --> BaseHiGraphModel_embedd_mesh_nodes
    BaseHiGraphModel --> BaseHiGraphModel_process_step
    BaseHiGraphModel --> BaseHiGraphModel_hi_processor_step

classDef parent fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#f1f5f9,font-size:16px
classDef base fill:#78350f,stroke:#f59e0b,stroke-width:1px,color:#fde68a,font-size:16px
classDef import fill:#1f2937,stroke:#6b7280,stroke-width:1.5px,color:#e5e7eb,font-size:16px
classDef method fill:#2d043f,stroke:#7c3aed,stroke-width:1.5px,color:#ede9fe,font-size:16px
classDef callNode fill:#064e3b,stroke:#10b981,stroke-width:1.5px,color:#d1fae5,font-size:16px
class BaseGraphModel parent
class BaseHiGraphModel base
class base_graph_model,datastore,torch,interaction_net,config import
class BaseHiGraphModel_get_num_mesh,BaseHiGraphModel_embedd_mesh_nodes,BaseHiGraphModel_process_step,BaseHiGraphModel_hi_processor_step method
```

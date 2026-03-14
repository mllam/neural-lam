```mermaid
%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80}}}%%
flowchart TD

module["utils"]

subgraph Parent_Class
    Module["Module"]
end

subgraph Base_Class
    BufferList["BufferList"]
end

    Module --> BufferList

subgraph Imports
    tueplots["tueplots"]
    loguru["loguru"]
    torch["torch"]
    warnings["warnings"]
    shutil["shutil"]
    os["os"]
    pytorch_lightning["pytorch_lightning"]
    functools["functools"]
    tempfile["tempfile"]
    custom_loggers["custom_loggers"]
    subprocess["subprocess"]
    pathlib["pathlib"]
end

    tueplots --> module
    loguru --> module
    torch --> module
    warnings --> module
    shutil --> module
    os --> module
    pytorch_lightning --> module
    functools --> module
    tempfile --> module
    custom_loggers --> module
    subprocess --> module
    pathlib --> module
    module --> BufferList

subgraph Methods
    BufferList___getitem__["__getitem__()"]
    BufferList___len__["__len__()"]
    BufferList___iter__["__iter__()"]
end

    BufferList --> BufferList___getitem__
    BufferList --> BufferList___len__
    BufferList --> BufferList___iter__

classDef parent fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#f1f5f9,font-size:16px
classDef base fill:#78350f,stroke:#f59e0b,stroke-width:1px,color:#fde68a,font-size:16px
classDef import fill:#1f2937,stroke:#6b7280,stroke-width:1.5px,color:#e5e7eb,font-size:16px
classDef method fill:#2d043f,stroke:#7c3aed,stroke-width:1.5px,color:#ede9fe,font-size:16px
classDef callNode fill:#064e3b,stroke:#10b981,stroke-width:1.5px,color:#d1fae5,font-size:16px
class Module parent
class BufferList base
class tueplots,loguru,torch,warnings,shutil,os,pytorch_lightning,functools,tempfile,custom_loggers,subprocess,pathlib import
class BufferList___getitem__,BufferList___len__,BufferList___iter__ method

```

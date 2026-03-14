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
    pathlib["pathlib"]
    custom_loggers["custom_loggers"]
    os["os"]
    tempfile["tempfile"]
    shutil["shutil"]
    subprocess["subprocess"]
    torch["torch"]
    tueplots["tueplots"]
    pytorch_lightning["pytorch_lightning"]
    loguru["loguru"]
    warnings["warnings"]
    functools["functools"]
end

    pathlib --> module
    custom_loggers --> module
    os --> module
    tempfile --> module
    shutil --> module
    subprocess --> module
    torch --> module
    tueplots --> module
    pytorch_lightning --> module
    loguru --> module
    warnings --> module
    functools --> module
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
class pathlib,custom_loggers,os,tempfile,shutil,subprocess,torch,tueplots,pytorch_lightning,loguru,warnings,functools import
class BufferList___getitem__,BufferList___len__,BufferList___iter__ method

```

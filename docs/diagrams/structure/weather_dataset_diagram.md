```mermaid
%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80}}}%%
flowchart TD

module["weather_dataset"]

subgraph Parent_Class
    Dataset["Dataset"]
    LightningDataModule["LightningDataModule"]
end

subgraph Base_Class
    WeatherDataset["WeatherDataset"]
    WeatherDataModule["WeatherDataModule"]
end

    Dataset --> WeatherDataset
    LightningDataModule --> WeatherDataModule

subgraph Imports
    xarray["xarray"]
    warnings["warnings"]
    typing["typing"]
    neural_lam["neural_lam"]
    torch["torch"]
    datetime["datetime"]
    loguru["loguru"]
    numpy["numpy"]
    pytorch_lightning["pytorch_lightning"]
end

    xarray --> module
    warnings --> module
    typing --> module
    neural_lam --> module
    torch --> module
    datetime --> module
    loguru --> module
    numpy --> module
    pytorch_lightning --> module
    module --> WeatherDataset
    module --> WeatherDataModule

subgraph Methods
    WeatherDataset__compute_std_safe["safe()"]
    WeatherDataset___len__["()"]
    WeatherDataset__slice_state_time["time()"]
    WeatherDataset__slice_forcing_time["time()"]
    WeatherDataset__build_item_dataarrays["dataarrays()"]
    WeatherDataset___getitem__["()"]
    WeatherDataset___iter__["()"]
    WeatherDataset_create_dataarray_from_tensor["tensor()"]
    WeatherDataModule_setup["setup()"]
    WeatherDataModule_train_dataloader["dataloader()"]
    WeatherDataModule_val_dataloader["dataloader()"]
    WeatherDataModule_test_dataloader["dataloader()"]
end

    WeatherDataset --> WeatherDataset__compute_std_safe
    WeatherDataset --> WeatherDataset___len__
    WeatherDataset --> WeatherDataset__slice_state_time
    WeatherDataset --> WeatherDataset__slice_forcing_time
    WeatherDataset --> WeatherDataset__build_item_dataarrays
    WeatherDataset --> WeatherDataset___getitem__
    WeatherDataset --> WeatherDataset___iter__
    WeatherDataset --> WeatherDataset_create_dataarray_from_tensor
    WeatherDataModule --> WeatherDataModule_setup
    WeatherDataModule --> WeatherDataModule_train_dataloader
    WeatherDataModule --> WeatherDataModule_val_dataloader
    WeatherDataModule --> WeatherDataModule_test_dataloader

classDef parent fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#f1f5f9,font-size:16px
classDef base fill:#78350f,stroke:#f59e0b,stroke-width:1px,color:#fde68a,font-size:16px
classDef import fill:#1f2937,stroke:#6b7280,stroke-width:1.5px,color:#e5e7eb,font-size:16px
classDef method fill:#2d043f,stroke:#7c3aed,stroke-width:1.5px,color:#ede9fe,font-size:16px
classDef callNode fill:#064e3b,stroke:#10b981,stroke-width:1.5px,color:#d1fae5,font-size:16px
class Dataset,LightningDataModule parent
class WeatherDataset,WeatherDataModule base
class xarray,warnings,typing,neural_lam,torch,datetime,loguru,numpy,pytorch_lightning import
class WeatherDataset__compute_std_safe,WeatherDataset___len__,WeatherDataset__slice_state_time,WeatherDataset__slice_forcing_time,WeatherDataset__build_item_dataarrays,WeatherDataset___getitem__,WeatherDataset___iter__,WeatherDataset_create_dataarray_from_tensor,WeatherDataModule_setup,WeatherDataModule_train_dataloader,WeatherDataModule_val_dataloader,WeatherDataModule_test_dataloader method
```

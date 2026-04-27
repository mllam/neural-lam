```mermaid
%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80}}}%%
flowchart TD

module["store"]

subgraph Parent_Class
    BaseRegularGridDatastore["BaseRegularGridDatastore"]
end

subgraph Base_Class
    NpyFilesDatastoreMEPS["NpyFilesDatastoreMEPS"]
end

    BaseRegularGridDatastore --> NpyFilesDatastoreMEPS

subgraph Imports
    functools["functools"]
    typing["typing"]
    datetime["datetime"]
    pathlib["pathlib"]
    dask["dask"]
    numpy["numpy"]
    parse["parse"]
    config["config"]
    base["base"]
    re["re"]
    warnings["warnings"]
    xarray["xarray"]
    cartopy["cartopy"]
    torch["torch"]
end

    functools --> module
    typing --> module
    datetime --> module
    pathlib --> module
    dask --> module
    numpy --> module
    parse --> module
    config --> module
    base --> module
    re --> module
    warnings --> module
    xarray --> module
    cartopy --> module
    torch --> module
    module --> NpyFilesDatastoreMEPS

subgraph Methods
    NpyFilesDatastoreMEPS_root_path["path()"]
    NpyFilesDatastoreMEPS_config["config()"]
    NpyFilesDatastoreMEPS_get_dataarray["dataarray()"]
    NpyFilesDatastoreMEPS__get_single_timeseries_dataarray["dataarray()"]
    NpyFilesDatastoreMEPS__get_analysis_times["times()"]
    NpyFilesDatastoreMEPS__calc_datetime_forcing_features["features()"]
    NpyFilesDatastoreMEPS_get_vars_units["units()"]
    NpyFilesDatastoreMEPS_get_vars_names["names()"]
    NpyFilesDatastoreMEPS_get_vars_long_names["names()"]
    NpyFilesDatastoreMEPS_get_num_data_vars["vars()"]
    NpyFilesDatastoreMEPS_get_xy["xy()"]
    NpyFilesDatastoreMEPS_step_length["length()"]
    NpyFilesDatastoreMEPS_grid_shape_state["state()"]
    NpyFilesDatastoreMEPS_boundary_mask["mask()"]
    NpyFilesDatastoreMEPS_get_standardization_dataarray["dataarray()"]
    NpyFilesDatastoreMEPS_coords_projection["projection()"]
end

    NpyFilesDatastoreMEPS --> NpyFilesDatastoreMEPS_root_path
    NpyFilesDatastoreMEPS --> NpyFilesDatastoreMEPS_config
    NpyFilesDatastoreMEPS --> NpyFilesDatastoreMEPS_get_dataarray
    NpyFilesDatastoreMEPS --> NpyFilesDatastoreMEPS__get_single_timeseries_dataarray
    NpyFilesDatastoreMEPS --> NpyFilesDatastoreMEPS__get_analysis_times
    NpyFilesDatastoreMEPS --> NpyFilesDatastoreMEPS__calc_datetime_forcing_features
    NpyFilesDatastoreMEPS --> NpyFilesDatastoreMEPS_get_vars_units
    NpyFilesDatastoreMEPS --> NpyFilesDatastoreMEPS_get_vars_names
    NpyFilesDatastoreMEPS --> NpyFilesDatastoreMEPS_get_vars_long_names
    NpyFilesDatastoreMEPS --> NpyFilesDatastoreMEPS_get_num_data_vars
    NpyFilesDatastoreMEPS --> NpyFilesDatastoreMEPS_get_xy
    NpyFilesDatastoreMEPS --> NpyFilesDatastoreMEPS_step_length
    NpyFilesDatastoreMEPS --> NpyFilesDatastoreMEPS_grid_shape_state
    NpyFilesDatastoreMEPS --> NpyFilesDatastoreMEPS_boundary_mask
    NpyFilesDatastoreMEPS --> NpyFilesDatastoreMEPS_get_standardization_dataarray
    NpyFilesDatastoreMEPS --> NpyFilesDatastoreMEPS_coords_projection

classDef parent fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#f1f5f9,font-size:16px
classDef base fill:#78350f,stroke:#f59e0b,stroke-width:1px,color:#fde68a,font-size:16px
classDef import fill:#1f2937,stroke:#6b7280,stroke-width:1.5px,color:#e5e7eb,font-size:16px
classDef method fill:#2d043f,stroke:#7c3aed,stroke-width:1.5px,color:#ede9fe,font-size:16px
classDef callNode fill:#064e3b,stroke:#10b981,stroke-width:1.5px,color:#d1fae5,font-size:16px
class BaseRegularGridDatastore parent
class NpyFilesDatastoreMEPS base
class functools,typing,datetime,pathlib,dask,numpy,parse,config,base,re,warnings,xarray,cartopy,torch import
class NpyFilesDatastoreMEPS_root_path,NpyFilesDatastoreMEPS_config,NpyFilesDatastoreMEPS_get_dataarray,NpyFilesDatastoreMEPS__get_single_timeseries_dataarray,NpyFilesDatastoreMEPS__get_analysis_times,NpyFilesDatastoreMEPS__calc_datetime_forcing_features,NpyFilesDatastoreMEPS_get_vars_units,NpyFilesDatastoreMEPS_get_vars_names,NpyFilesDatastoreMEPS_get_vars_long_names,NpyFilesDatastoreMEPS_get_num_data_vars,NpyFilesDatastoreMEPS_get_xy,NpyFilesDatastoreMEPS_step_length,NpyFilesDatastoreMEPS_grid_shape_state,NpyFilesDatastoreMEPS_boundary_mask,NpyFilesDatastoreMEPS_get_standardization_dataarray,NpyFilesDatastoreMEPS_coords_projection method
```

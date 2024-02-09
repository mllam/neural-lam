## Format of data directory
It is possible to store multiple datasets in the `data` directory.
Each dataset contains a set of files with static features and a set of samples.
The samples are split into different sub-directories for training, validation and testing.
The directory structure is shown with examples below.
Script names within parenthesis denote the script used to generate the file.
```
data
├── dataset1
│   ├── samples                             - Directory with data samples
│   │   ├── train                           - Training data
│   │   │   ├── nwp_2022040100_mbr000.npy  - A time series sample
│   │   │   ├── nwp_2022040100_mbr001.npy
│   │   │   ├── ...
│   │   │   ├── nwp_2022043012_mbr001.npy
│   │   │   ├── nwp_toa_downwelling_shortwave_flux_2022040100.npy   - Solar flux forcing (x TODO: temporarily omitting)
│   │   │   ├── nwp_toa_downwelling_shortwave_flux_2022040112.npy
│   │   │   ├── ...
│   │   │   ├── nwp_toa_downwelling_shortwave_flux_2022043012.npy
│   │   │   ├── wtr_2022040100.npy          - Open water features for one sample (x TODO: temporarily omitting)
│   │   │   ├── wtr_2022040112.npy
│   │   │   ├── ...
│   │   │   └── wtr_202204012.npy
│   │   ├── val                             - Validation data
│   │   └── test                            - Test data
│   └── static                              - Directory with graph information and static features
│       ├── nwp_xy.npy                      - Coordinates of grid nodes (part of dataset)                                   x
│       ├── surface_geopotential.npy        - Geopotential at surface of grid nodes (part of dataset)                       x TODO: temporarily omitting this from grid_features
│       ├── border_mask.npy                 - Mask with True for grid nodes that are part of border (part of dataset)       x TODO: temp. omitting
│       ├── grid_features.pt                - Static features of grid nodes (create_grid_features.py)                       x TODO: add other features in
│       ├── parameter_mean.pt               - Means of state parameters (create_parameter_weights.py)                       x TODO: setting as 0 for now
│       ├── parameter_std.pt                - Std.-dev. of state parameters (create_parameter_weights.py)                   x TODO: setting as 1 for now
│       ├── diff_mean.pt                    - Means of one-step differences (create_parameter_weights.py)                   x TODO: setting as 0 for now
│       ├── diff_std.pt                     - Std.-dev. of one-step differences (create_parameter_weights.py)               x TODO: setting as 1 for now
│       ├── flux_stats.pt                   - Mean and std.-dev. of solar flux forcing (create_parameter_weights.py)        x TODO: setting as 0, 1 for now
│       └── parameter_weights.npy           - Loss weights for different state parameters (create_parameter_weights.py)     
├── dataset2
├── ...
└── datasetN
```


## Details

```
MEPS dataset:
    N_t' = 65 (total number of time steps in forecast. 1hr between time steps)
    N_t = 65//subsample_step (= 21 for 3h steps) 
        (subsample_step: how many hours ahead for one prediction. N_t is thus the total number of subsampled forecasts) 
    dim_x = 268 (width)
    dim_y = 238 (height)
    N_grid = dim_x * dim_y = 268 * 238 = 63784 (total number of grid nodes)
    d_features = 17 (d_features' = 18) (number of atmospheric features)
    d_forcing = 5 (number of forcing features)

nwp_xy.npy
    - xy coordinates
    - (2, N_x, N_y)

border_mask.npy
    - xy coordinates
    - (2, N_x, N_y)

nwp_2022040100_mbr000.npy
    - A time series sample
    - (N_t', dim_x, dim_y, d_features')

nwp_toa_downwelling_shortwave_flux_2022040100.npy   
    - Solar flux forcing term
    - 
```

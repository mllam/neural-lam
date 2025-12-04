# End-to-End Reproduction (COSMO)

This concise guide walks you through reproducing the COSMO experiments from scratch. If you get
stuck, ask in Slack (#neural-lam): [Slack
invite](https://join.slack.com/t/ml-lam/shared_invite/zt-2t112zvm8-Vt6aBvhX7nYa6Kbj_LkCBQ)

## Word of Warning

Reproducing the experiments is resource-intensive (GPU, storage, memory and time). Consider using
SLURM on a cluster to submit jobs. Or work inside a suitable environment/container and use tools
like `code tunnel` to run commands interactively. For a lightweight test, consider using the COSMO
sample dataset and reduced settings (see `docs/reproduce_paper_sample.md`).

## Step 0: Download Resources & Set Path

Download the sample dataset and checkpoints into a local `WORKDIR` directory (~7TB):

```zsh
# Neural-LAM will read/write data from here
## --------> CHANGE THIS PATH <--------
WORKDIR=/path/to/your/working/directory
mkdir -p $WORKDIR
```

Obtain the full COSMO dataset by contacting the authors and placing it under `$WORKDIR/cosmo_ml_dataset.zarr`.

### Published checkpoint (optional)

* Published COSMO checkpoint (~1min for 201MB): [Zenodo](https://zenodo.org/records/15131838) \
`curl -L -o $WORKDIR/cosmo_model.ckpt "https://zenodo.org/api/records/15131838/files/cosmo_model.ckpt/content"`

## Step 1: Environment and installs

Required Python: >=3.10 & <=3.12.8. The paper reproduction was tested on v3.12.8 with this setup:

```zsh
# clone neural-lam repo research branch (e.g into home directory)
git clone --branch research https://github.com/joeloskarsson/neural-lam-dev.git
cd neural-lam-dev
SRCDIR=$(pwd)

# Using uv to manage Python versions and venvs (use what you prefer - pip/pdm/conda...)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu126 # match your cuda version
uv pip install ".[graph]" "zarr<3" # experiments were originally run with zarr<3

# mllam-data-prep (paper tag)
git clone --branch building-ml-lams https://github.com/sadamov/mllam-data-prep.git
uv pip install mllam-data-prep

# weather-model-graphs (paper tag)
git clone --branch building-ml-lams https://github.com/joeloskarsson/weather-model-graphs.git
uv pip install weather-model-graphs
```

## Step 2: Prepare configs and artifacts

Copy the COSMO scripts/configs/artifacts into WORKDIR so everything is co-located

```zsh
cp "$SRCDIR/scripts/cosmo_interior_config.yaml" "$WORKDIR/"
cp "$SRCDIR/scripts/cosmo_era5_config.yaml" "$WORKDIR/"
cp "$SRCDIR/scripts/cosmo_ifs_config.yaml" "$WORKDIR/"
cp "$SRCDIR/scripts/cosmo_build_graphs.sh" "$WORKDIR/"
cp "$SRCDIR/scripts/cosmo_model_config_era5.yaml" "$WORKDIR/"
cp "$SRCDIR/scripts/cosmo_model_config_ifs.yaml" "$WORKDIR/"
cp -r "$SRCDIR/scripts/artifacts/cosmo_land_sea_mask.zarr" "$WORKDIR/cosmo_land_sea_mask.zarr"
mkdir -p "$WORKDIR/figures"
cp "$SRCDIR/figures/earth_texture.jpeg" "$WORKDIR/figures/"
```

## Step 3: ERA5 & IFS download

Run these from `$SRCDIR`; outputs go under `$WORKDIR`.

```zsh
# ERA5 WeatherBench2 subset -> Zarr - (~30min for 17GB)
python scripts/era_download.py \
    --start "2015-01-01T00" --end "2020-12-31T00" \
    --output-name cosmo_era5.zarr \
    --output-dir "$WORKDIR"

# IFS WeatherBench2 subset -> Zarr (~1h for 92GB)
python scripts/ifs_download.py \
    --start "2019-08-30T00" --end "2020-11-02T00" \
    --output-name cosmo_ifs.zarr \
    --output-dir "$WORKDIR"
```

## Step 4: Impute missing IFS values

There are missing values in the IFS dataset that need to be addressed.

```zsh
python scripts/interp_na_ifs.py "$WORKDIR/cosmo_ifs.zarr"
```

## Step 5: Land–sea mask (LSM)

The land-sea mask was already generated for your convenience and is available under
`scripts/artifacts/`. You already copied it as `$WORKDIR/cosmo_land_sea_mask.zarr`. If you want to
regenerate it, you can use the following commands:

<details>
<summary>Generate land–sea mask with micromamba (click to expand)</summary>

```zsh
# I have issues with GDAL system library (rasterio backend), hence I install it manually here
# with micromamba
if [ ! -x bin/micromamba ]; then
    ARCH=$(uname -m)
    case "$ARCH" in
        x86_64)
            MICROMAMBA_PKG=linux-64
            ;;
        aarch64)
            MICROMAMBA_PKG=linux-aarch64
            ;;
        *)
            echo "Unsupported architecture: $ARCH" >&2
            exit 1
            ;;
    esac
    curl -Ls "https://micro.mamba.pm/api/micromamba/${MICROMAMBA_PKG}/latest" | tar -xvj bin/micromamba
fi

if ! bin/micromamba env list | awk '{print $1}' | grep -q '^lam-lsm$'; then
    bin/micromamba create -n lam-lsm -c conda-forge python=3.12 gdal rasterio dask geopandas cartopy pyproj xarray shapely affine zarr -y
else
    echo "Reusing existing lam-lsm environment."
fi

bin/micromamba run -n lam-lsm python scripts/create_land_sea_mask.py \
    --source_zarr "$WORKDIR/cosmo_sample.zarr" \
    --output_zarr "$WORKDIR/cosmo_land_sea_mask.zarr"

# delete mamba env again (optional)
bin/micromamba env remove -n lam-lsm -y
rm -rf bin
```

</details>

## Step 6: Preprocess datastores with mllam-data-prep

This step builds the `.zarr` datastores that Neural-LAM will consume.
From now on we will work exclusively in the `$WORKDIR`.

```zsh
cd "$WORKDIR"
python -m mllam_data_prep --show-progress cosmo_interior_config.yaml # ~2h
python -m mllam_data_prep --show-progress cosmo_era5_config.yaml # ~10min
python -m mllam_data_prep --show-progress cosmo_ifs_config.yaml # ~20min

```

### Step 7: Build graphs

```zsh
bash cosmo_build_graphs.sh # ~5min

python -m neural_lam.plot_graph \
    --config_path cosmo_model_config_era5.yaml \
    --graph_name rectangular_hierarchical \
    --save cosmo_graph.html \
    --mesh_level_dist 0.1 \
    --edge_width 0.1 \
    --mesh_edge_width 0.4 \
    --grid_node_size 0.2 \
    --mesh_node_size 0.8 \
    --boundary_grid_color grey \
    --mesh_color orange
```

Graph plotting is slow; use Firefox and optionally reduce the plotted region with
`--corner_filter_radius`. Earth texture lives at `figures/earth_texture.jpeg`.

Result should look like:
![cosmo_graph](../figures/cosmo_graph.png)

## Step 8: Train model

The following trains the hierarchical model on the interior dataset for 200 epochs.
Usually we used 64 nodes with 4 GH200 for such experiments.

```zsh
python -m neural_lam.train_model \
    --config_path "$WORKDIR/cosmo_model_config_era5.yaml" \
    --model hi_lam \
    --graph_name rectangular_hierarchical \
    --hidden_dim 300 \
    --hidden_dim_grid 150 \
    --time_delta_enc_dim 32 \
    --processor_layers 2 \
    --batch_size 1 \
    --min_lr 0.001 \
    --epochs 200 \
    --val_interval 10 \
    --val_steps_to_log 1 2 3 4 \
    --ar_steps_eval 4 \
    --precision bf16-mixed \
    --plot_vars "T_2M" \
    --num_workers 8 \
    --num_nodes 64
```

GPU-memory used will be around 45GB; system memory around 100GB.

## Step 9: Finetune from checkpoint

To finetune from a checkpoint (e.g., the one created in Step 8) we first need to locate the `CHECKPOINT` from the run above. Then you need to replace <your_run> below with the actual folder from the run.

```zsh
CHECKPOINT=$WORKDIR/saved_models/<your_run>/last.ckpt # replace <your_run>

python -m neural_lam.train_model \
    --config_path "$WORKDIR/cosmo_model_config_era5.yaml" \
    --model hi_lam \
    --graph_name rectangular_hierarchical \
    --hidden_dim 300 \
    --hidden_dim_grid 150 \
    --time_delta_enc_dim 32 \
    --processor_layers 2 \
    --batch_size 1 \
    --lr 0.0001 \
    --min_lr 0.0001 \
    --epochs 200 \
    --val_interval 10 \
    --val_steps_to_log 1 2 3 4 8 12 16 20 24 \
    --ar_steps_train 12 \
    --ar_steps_eval 24 \
    --precision bf16-mixed \
    --plot_vars "T_2M" \
    --grad_checkpointing \
    --num_workers 4 \
    --num_nodes 64 \
    --restore_opt \
    --load ${CHECKPOINT}
```

## Step 10: Evaluate and save forecasts to Zarr

Evaluation is using `train_model` with the `--eval` flag and requires the `CHECKPOINT_FT` from above
provided via the `--load` flag. This demo shows eval on the validation set, for eval on the test set
change `--eval test`, that's it.

```zsh
CHECKPOINT=$WORKDIR/saved_models/<your_finetune_run>/last.ckpt # replace <your_finetune_run>

python -m neural_lam.train_model \
    --model hi_lam \
    --epochs 1 \
    --eval test \
    --n_example_pred 1 \
    --ar_steps_eval 120 \
    --val_steps_to_log 1 12 24 36 48 60 72 84 96 108 120 \
    --hidden_dim 300 \
    --hidden_dim_grid 150 \
    --time_delta_enc_dim 32 \
    --processor_layers 2 \
    --num_nodes 64 \
    --batch_size 1 \
    --plot_vars "T_2M" "U_10M" \
    --precision bf16-mixed \
    --graph_name rectangular_hierarchical \
    --config_path "$WORKDIR/cosmo_model_config_ifs.yaml" \
    --load ${CHECKPOINT_FT}
    --save_eval_to_zarr_path "$WORKDIR/cosmo_sample_forecasts.zarr"
```

This should produce the desired forecasts and save them to the specified Zarr format under `$WORKDIR/cosmo_sample_forecasts.zarr`.

To compare your results with the original wandb runs from the paper you can have a look at this report:
[https://api.wandb.ai/links/jo-research-team/6h61s1m6](https://api.wandb.ai/links/jo-research-team/6h61s1m6)

Awesome, you have successfully reproduced the paper's results. Good luck with your experiments!

# Standard library
import os

# Third-party
import cartopy.crs as ccrs
import numpy as np
import torch
import xarray as xr
import yaml
from torch import nn
from tueplots import bundles, figsizes


class BufferList(nn.Module):
    """
    A list of torch buffer tensors that sit together as a Module with no
    parameters and only buffers.

    This should be replaced by a native torch BufferList once implemented.
    See: https://github.com/pytorch/pytorch/issues/37386
    """

    def __init__(self, buffer_tensors, persistent=True):
        super().__init__()
        self.n_buffers = len(buffer_tensors)
        for buffer_i, tensor in enumerate(buffer_tensors):
            self.register_buffer(f"b{buffer_i}", tensor, persistent=persistent)

    def __getitem__(self, key):
        return getattr(self, f"b{key}")

    def __len__(self):
        return self.n_buffers

    def __iter__(self):
        return (self[i] for i in range(len(self)))


def load_graph(graph_name, device="cpu"):
    """
    Load all tensors representing the graph
    """
    # Define helper lambda function
    graph_dir_path = os.path.join("graphs", graph_name)

    def loads_file(fn):
        return torch.load(os.path.join(graph_dir_path, fn), map_location=device)

    # Load edges (edge_index)
    m2m_edge_index = BufferList(
        loads_file("m2m_edge_index.pt"), persistent=False
    )  # List of (2, M_m2m[l])
    g2m_edge_index = loads_file("g2m_edge_index.pt")  # (2, M_g2m)
    m2g_edge_index = loads_file("m2g_edge_index.pt")  # (2, M_m2g)

    n_levels = len(m2m_edge_index)
    hierarchical = n_levels > 1  # Nor just single level mesh graph

    # Load static edge features
    m2m_features = loads_file("m2m_features.pt")  # List of (M_m2m[l], d_edge_f)
    g2m_features = loads_file("g2m_features.pt")  # (M_g2m, d_edge_f)
    m2g_features = loads_file("m2g_features.pt")  # (M_m2g, d_edge_f)

    # Normalize by dividing with longest edge (found in m2m)
    longest_edge = max(
        torch.max(level_features[:, 0]) for level_features in m2m_features
    )  # Col. 0 is length
    m2m_features = BufferList(
        [level_features / longest_edge for level_features in m2m_features],
        persistent=False,
    )
    g2m_features = g2m_features / longest_edge
    m2g_features = m2g_features / longest_edge

    # Load static node features
    mesh_static_features = loads_file(
        "mesh_features.pt"
    )  # List of (N_mesh[l], d_mesh_static)

    # Some checks for consistency
    assert (
        len(m2m_features) == n_levels
    ), "Inconsistent number of levels in mesh"
    assert (
        len(mesh_static_features) == n_levels
    ), "Inconsistent number of levels in mesh"

    if hierarchical:
        # Load up and down edges and features
        mesh_up_edge_index = BufferList(
            loads_file("mesh_up_edge_index.pt"), persistent=False
        )  # List of (2, M_up[l])
        mesh_down_edge_index = BufferList(
            loads_file("mesh_down_edge_index.pt"), persistent=False
        )  # List of (2, M_down[l])

        mesh_up_features = loads_file(
            "mesh_up_features.pt"
        )  # List of (M_up[l], d_edge_f)
        mesh_down_features = loads_file(
            "mesh_down_features.pt"
        )  # List of (M_down[l], d_edge_f)

        # Rescale
        mesh_up_features = BufferList(
            [
                edge_features / longest_edge
                for edge_features in mesh_up_features
            ],
            persistent=False,
        )
        mesh_down_features = BufferList(
            [
                edge_features / longest_edge
                for edge_features in mesh_down_features
            ],
            persistent=False,
        )

        mesh_static_features = BufferList(
            mesh_static_features, persistent=False
        )
    else:
        # Extract single mesh level
        m2m_edge_index = m2m_edge_index[0]
        m2m_features = m2m_features[0]
        mesh_static_features = mesh_static_features[0]

        (
            mesh_up_edge_index,
            mesh_down_edge_index,
            mesh_up_features,
            mesh_down_features,
        ) = ([], [], [], [])

    return hierarchical, {
        "g2m_edge_index": g2m_edge_index,
        "m2g_edge_index": m2g_edge_index,
        "m2m_edge_index": m2m_edge_index,
        "mesh_up_edge_index": mesh_up_edge_index,
        "mesh_down_edge_index": mesh_down_edge_index,
        "g2m_features": g2m_features,
        "m2g_features": m2g_features,
        "m2m_features": m2m_features,
        "mesh_up_features": mesh_up_features,
        "mesh_down_features": mesh_down_features,
        "mesh_static_features": mesh_static_features,
    }


def make_mlp(blueprint, layer_norm=True):
    """
    Create MLP from list blueprint, with
    input dimensionality: blueprint[0]
    output dimensionality: blueprint[-1] and
    hidden layers of dimensions: blueprint[1], ..., blueprint[-2]

    if layer_norm is True, includes a LayerNorm layer at
    the output (as used in GraphCast)
    """
    hidden_layers = len(blueprint) - 2
    assert hidden_layers >= 0, "Invalid MLP blueprint"

    layers = []
    for layer_i, (dim1, dim2) in enumerate(zip(blueprint[:-1], blueprint[1:])):
        layers.append(nn.Linear(dim1, dim2))
        if layer_i != hidden_layers:
            layers.append(nn.SiLU())  # Swish activation

    # Optionally add layer norm to output
    if layer_norm:
        layers.append(nn.LayerNorm(blueprint[-1]))

    return nn.Sequential(*layers)


def fractional_plot_bundle(fraction):
    """
    Get the tueplots bundle, but with figure width as a fraction of
    the page width.
    """
    bundle = bundles.neurips2023(usetex=True, family="serif")
    bundle.update(figsizes.neurips2023())
    original_figsize = bundle["figure.figsize"]
    bundle["figure.figsize"] = (
        original_figsize[0] / fraction,
        original_figsize[1],
    )
    return bundle


def init_wandb_metrics(wandb_logger, val_steps):
    """
    Set up wandb metrics to track
    """
    experiment = wandb_logger.experiment
    experiment.define_metric("val_mean_loss", summary="min")
    for step in val_steps:
        experiment.define_metric(f"val_loss_unroll{step}", summary="min")


class ConfigLoader:
    """
    Class for loading configuration files.

    This class loads a YAML configuration file and provides a way to access
    its values as attributes.
    """

    def __init__(self, config_path, values=None):
        self.config_path = config_path
        if values is None:
            self.values = self.load_config()
        else:
            self.values = values

    def load_config(self):
        """Load configuration file."""
        with open(self.config_path, encoding="utf-8", mode="r") as file:
            return yaml.safe_load(file)

    def __getattr__(self, name):
        keys = name.split(".")
        value = self.values
        for key in keys:
            if key in value:
                value = value[key]
            else:
                return None
        if isinstance(value, dict):
            return ConfigLoader(None, values=value)
        return value

    def __getitem__(self, key):
        value = self.values[key]
        if isinstance(value, dict):
            return ConfigLoader(None, values=value)
        return value

    def __contains__(self, key):
        return key in self.values

    def param_names(self):
        """Return parameter names."""
        surface_names = self.values["state"]["surface"]
        atmosphere_names = [
            f"{var}_{level}"
            for var in self.values["state"]["atmosphere"]
            for level in self.values["state"]["levels"]
        ]
        return surface_names + atmosphere_names

    def param_units(self):
        """Return parameter units."""
        surface_units = self.values["state"]["surface_units"]
        atmosphere_units = [
            unit
            for unit in self.values["state"]["atmosphere_units"]
            for _ in self.values["state"]["levels"]
        ]
        return surface_units + atmosphere_units

    def num_data_vars(self, key):
        """Return the number of data variables for a given key."""
        surface_vars = len(self.values[key]["surface"])
        atmosphere_vars = len(self.values[key]["atmosphere"])
        levels = len(self.values[key]["levels"])
        return surface_vars + atmosphere_vars * levels

    def projection(self):
        """Return the projection."""
        proj_config = self.values["projections"]["class"]
        proj_class = getattr(ccrs, proj_config["proj_class"])
        proj_params = proj_config["proj_params"]
        return proj_class(**proj_params)

    def open_zarr(self, dataset_name):
        """Open a dataset specified by the dataset name."""
        dataset_path = self.zarrs[dataset_name].path
        if dataset_path is None or not os.path.exists(dataset_path):
            print(f"Dataset '{dataset_name}' not found at path: {dataset_path}")
            return None
        dataset = xr.open_zarr(dataset_path, consolidated=True)
        return dataset

    def load_normalization_stats(self):
        """Load normalization statistics from Zarr archive."""
        normalization_path = "normalization.zarr"
        if not os.path.exists(normalization_path):
            print(
                f"Normalization statistics not found at "
                f"path: {normalization_path}"
            )
            return None
        normalization_stats = xr.open_zarr(
            normalization_path, consolidated=True
        )
        return normalization_stats

    def process_dataset(self, dataset_name, split="train", stack=True):
        """Process a single dataset specified by the dataset name."""

        dataset = self.open_zarr(dataset_name)
        if dataset is None:
            return None

        start, end = (
            self.splits[split].start,
            self.splits[split].end,
        )
        dataset = dataset.sel(time=slice(start, end))
        dataset = dataset.rename_dims(
            {
                v: k
                for k, v in self.zarrs[dataset_name].dims.values.items()
                if k not in dataset.dims
            }
        )

        vars_surface = []
        if self[dataset_name].surface:
            vars_surface = dataset[self[dataset_name].surface]

        vars_atmosphere = []
        if self[dataset_name].atmosphere:
            vars_atmosphere = xr.merge(
                [
                    dataset[var]
                    .sel(level=level, drop=True)
                    .rename(f"{var}_{level}")
                    for var in self[dataset_name].atmosphere
                    for level in self[dataset_name].levels
                ]
            )

        if vars_surface and vars_atmosphere:
            dataset = xr.merge([vars_surface, vars_atmosphere])
        elif vars_surface:
            dataset = vars_surface
        elif vars_atmosphere:
            dataset = vars_atmosphere
        else:
            print(f"No variables found in dataset {dataset_name}")
            return None

        if not all(
            lat_lon in self.zarrs[dataset_name].dims.values.values()
            for lat_lon in self.zarrs[
                dataset_name
            ].lat_lon_names.values.values()
        ):
            lat_name = self.zarrs[dataset_name].lat_lon_names.lat
            lon_name = self.zarrs[dataset_name].lat_lon_names.lon
            if dataset[lat_name].ndim == 2:
                dataset[lat_name] = dataset[lat_name].isel(x=0, drop=True)
            if dataset[lon_name].ndim == 2:
                dataset[lon_name] = dataset[lon_name].isel(y=0, drop=True)
            dataset = dataset.assign_coords(
                x=dataset[lon_name], y=dataset[lat_name]
            )

        if stack:
            dataset = self.stack_grid(dataset)

        return dataset

    def stack_grid(self, dataset):
        """Stack grid dimensions."""
        dataset = dataset.squeeze().stack(grid=("x", "y")).to_array()

        if "time" in dataset.dims:
            dataset = dataset.transpose("time", "grid", "variable")
        else:
            dataset = dataset.transpose("grid", "variable")
        return dataset

    def get_nwp_xy(self):
        """Get the x and y coordinates for the NWP grid."""
        x = self.process_dataset("static", stack=False).x.values
        y = self.process_dataset("static", stack=False).y.values
        xx, yy = np.meshgrid(y, x)
        xy = np.stack((xx, yy), axis=0)

        return xy

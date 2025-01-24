# Standard library
import os
import shutil

# Third-party
import numpy as np
import torch
from pytorch_lightning.utilities import rank_zero_only
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


def load_graph(graph_dir_path, device="cpu"):
    """Load all tensors representing the graph from `graph_dir_path`.

    Needs the following files for all graphs:
    - m2m_edge_index.pt
    - g2m_edge_index.pt
    - m2g_edge_index.pt
    - m2m_features.pt
    - g2m_features.pt
    - m2g_features.pt
    - mesh_features.pt

    And in addition for hierarchical graphs:
    - mesh_up_edge_index.pt
    - mesh_down_edge_index.pt
    - mesh_up_features.pt
    - mesh_down_features.pt

    Parameters
    ----------
    graph_dir_path : str
        Path to directory containing the graph files.
    device : str
        Device to load tensors to.

    Returns
    -------
    hierarchical : bool
        Whether the graph is hierarchical.
    graph : dict
        Dictionary containing the graph tensors, with keys as follows:
        - g2m_edge_index
        - m2g_edge_index
        - m2m_edge_index
        - mesh_up_edge_index
        - mesh_down_edge_index
        - g2m_features
        - m2g_features
        - m2m_features
        - mesh_up_features
        - mesh_down_features
        - mesh_static_features

    """

    def loads_file(fn):
        return torch.load(
            os.path.join(graph_dir_path, fn),
            map_location=device,
            weights_only=True,
        )

    # Load edges (edge_index)
    m2m_edge_index = BufferList(
        loads_file("m2m_edge_index.pt"), persistent=False
    )  # List of (2, M_m2m[l])
    g2m_edge_index = loads_file("g2m_edge_index.pt")  # (2, M_g2m)
    m2g_edge_index = loads_file("m2g_edge_index.pt")  # (2, M_m2g)

    n_levels = len(m2m_edge_index)
    hierarchical = n_levels > 1  # Nor just single level mesh graph

    # Load static edge features
    # List of (M_m2m[l], d_edge_f)
    m2m_features = loads_file("m2m_features.pt")
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
    # If latex is not available, some visualizations might not render
    # correctly, but will at least not raise an error. Alternatively, use
    # unicode raised numbers.
    usetex = True if shutil.which("latex") else False
    bundle = bundles.neurips2023(usetex=usetex, family="serif")
    bundle.update(figsizes.neurips2023())
    original_figsize = bundle["figure.figsize"]
    bundle["figure.figsize"] = (
        original_figsize[0] / fraction,
        original_figsize[1],
    )
    return bundle


@rank_zero_only
def rank_zero_print(*args, **kwargs):
    """Print only from rank 0 process"""
    print(*args, **kwargs)


def init_wandb_metrics(wandb_logger, val_steps):
    """
    Set up wandb metrics to track
    """
    experiment = wandb_logger.experiment
    experiment.define_metric("val_mean_loss", summary="min")
    for step in val_steps:
        experiment.define_metric(f"val_loss_unroll{step}", summary="min")


def get_time_step(times):
    """Calculate the time step from a time dataarray.

    Parameters
    ----------
    times : xr.DataArray
        The time dataarray to calculate the time step from.

    Returns
    -------
    time_step : float
        The time step in the the datetime-format of the times dataarray.
    """
    time_diffs = np.diff(times)
    if not np.all(time_diffs == time_diffs[0]):
        raise ValueError(
            "Inconsistent time steps in data. "
            f"Found different time steps: {np.unique(time_diffs)}"
        )
    return time_diffs[0]


def check_time_overlap(
    da1,
    da2,
    da1_is_forecast=False,
    da2_is_forecast=False,
    num_past_steps=1,
    num_future_steps=1,
):
    """Check that the time coverage of two dataarrays overlap.

    Parameters
    ----------
    da1 : xr.DataArray
        The first dataarray to check.
    da2 : xr.DataArray
        The second dataarray to check.
    da1_is_forecast : bool, optional
        Whether the first dataarray is forecast data.
    da2_is_forecast : bool, optional
        Whether the second dataarray is forecast data.
    num_past_steps : int, optional
        Number of past forcing steps.
    num_future_steps : int, optional
        Number of future forcing steps.

    Raises
    ------
    ValueError
        If the time coverage of the dataarrays does not overlap.
    """

    if da1_is_forecast:
        times_da1 = da1.analysis_time
    else:
        times_da1 = da1.time
    time_min_da1 = times_da1.min().values
    time_max_da1 = times_da1.max().values

    if da2_is_forecast:
        times_da2 = da2.analysis_time
        _ = get_time_step(da2.elapsed_forecast_duration)
    else:
        times_da2 = da2.time
        time_step_da2 = get_time_step(times_da2.values)

    time_min_da2 = times_da2.min().values
    time_max_da2 = times_da2.max().values

    # Calculate required bounds for da2 using its time step
    da2_required_time_min = time_min_da1 - num_past_steps * time_step_da2
    da2_required_time_max = time_max_da1 + num_future_steps * time_step_da2

    if time_min_da2 > da2_required_time_min:
        raise ValueError(
            f"The second DataArray (e.g. 'boundary forcing') starts too late."
            f"Required start: {da2_required_time_min}, "
            f"but DataArray starts at {time_min_da2}."
        )

    if time_max_da2 < da2_required_time_max:
        raise ValueError(
            f"The second DataArray (e.g. 'boundary forcing') ends too early."
            f"Required end: {da2_required_time_max}, "
            f"but DataArray ends at {time_max_da2}."
        )


def crop_time_if_needed(
    da1, da2, da1_is_forecast=False, da2_is_forecast=False, num_past_steps=1
):
    """
    Slice away the first few timesteps from the first DataArray (e.g. 'state')
    if the second DataArray (e.g. boundary forcing) does not cover that range
    (including num_past_steps).

    Parameters
    ----------
    da1 : xr.DataArray
        The first DataArray to crop.
    da2 : xr.DataArray
        The second DataArray to compare against.
    da1_is_forecast : bool, optional
        Whether the first dataarray is forecast data.
    da2_is_forecast : bool, optional
        Whether the second dataarray is forecast data.
    num_past_steps : int
        Number of past time steps to consider.

    Return
    ------
    da1 : xr.DataArray
        The cropped first DataArray and print a warning if any steps are
        removed.
    """
    if da1 is None or da2 is None:
        return da1

    try:
        check_time_overlap(
            da1,
            da2,
            da1_is_forecast,
            da2_is_forecast,
            num_past_steps,
            num_future_steps=0,
        )
        return da1
    except ValueError:
        # If da2 coverage is insufficient, remove earliest da1 times
        # until coverage is possible. Figure out how many steps to remove.
        if da1_is_forecast:
            da1_tvals = da1.analysis_time.values
        else:
            da1_tvals = da1.time.values
        if da2_is_forecast:
            da2_tvals = da2.analysis_time.values
        else:
            da2_tvals = da2.time.values

        if da1_tvals[0] < da2_tvals[0]:
            # Calculate how many steps to remove skip just enough steps so that:
            if da2_is_forecast:
                # The windowing for forecast type data happens in the
                # elapsed_forecast_duration dimension, so we can omit it here.
                required_min = da2_tvals[0]
            else:
                dt = get_time_step(da2_tvals)
                required_min = da2_tvals[0] + num_past_steps * dt
            first_valid_idx = (da1_tvals >= required_min).argmax()
            n_removed = first_valid_idx
            if n_removed > 0:
                print(
                    f"Warning: removing {n_removed} da1 (e.g. 'state') "
                    f"timesteps to align with da2 (e.g. 'boundary forcing') "
                    f"coverage."
                )
                da1 = da1.isel(time=slice(first_valid_idx, None))
        return da1

# Standard library
from datetime import timedelta
from pathlib import Path

# Third-party
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless test runs

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pytest
import torch

# First-party
from neural_lam import config as nlconfig
from neural_lam import vis
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.weather_dataset import WeatherDataset
from tests.dummy_datastore import DummyDatastore

# Create output directory for test figures
TEST_OUTPUT_DIR = Path(__file__).parent / "test_outputs" / "plotting"
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class HeatmapDatastore:
    """Minimal datastore stub for error-heatmap plotting tests."""

    def __init__(self, n_vars, step_length=timedelta(hours=1)):
        self._n_vars = n_vars
        self.step_length = step_length

    def get_vars_names(self, category):
        assert category == "state"
        return [f"state_var_{i}" for i in range(self._n_vars)]

    def get_vars_units(self, category):
        assert category == "state"
        return ["unit"] * self._n_vars


@pytest.fixture
def model_and_batch(tmp_path, time_step, time_unit):
    """Setup a model and dataset for testing plot_examples"""
    # Create timedelta with specified step length
    step_length_kwargs = {time_unit: time_step}
    step_length = timedelta(**step_length_kwargs)

    # Create datastore with specified step_length
    datastore = DummyDatastore(step_length=step_length)

    # Create minimal model args
    class ModelArgs:
        output_std = False
        loss = "mse"
        restore_opt = False
        n_example_pred = 2
        graph = "1level"
        hidden_dim = 4
        hidden_layers = 1
        processor_layers = 1
        mesh_aggr = "sum"
        lr = 1.0e-3
        val_steps_to_log = [1, 2]
        metrics_watch = []
        num_past_forcing_steps = 0
        num_future_forcing_steps = 0
        var_leads_metrics_watch = {}

    graph_dir_path = Path(datastore.root_path) / "graph" / "1level"
    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            n_max_levels=1,
        )

    # Create config
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME,
            config_path=datastore.root_path,
        ),
    )

    # Create model
    model = GraphLAM(
        args=ModelArgs(),
        config=config,
        datastore=datastore,
    )

    # Create dataset to get a sample batch
    dataset = WeatherDataset(
        datastore=datastore,
        split="train",
        ar_steps=2,
        num_past_forcing_steps=0,
        num_future_forcing_steps=0,
    )

    # Get a batch (just use one sample)
    sample = dataset[0]
    batch = tuple(torch.stack([item]) for item in sample)  # Add batch dimension

    return model, batch, datastore, tmp_path


@pytest.mark.parametrize(
    "time_step,time_unit",
    [
        (1, "hours"),
        (3, "hours"),
        (6, "hours"),
        (1, "minutes"),
    ],
)
@pytest.mark.parametrize("t_i", [1, 2])
def test_plot_examples_integration_saves_figure(
    model_and_batch, time_step, time_unit, t_i
):
    """Integration test that saves actual figure for manual inspection"""
    model, batch, datastore, tmp_path = model_and_batch

    # Reset plotted examples counter
    model.plotted_examples = 0

    # Verify that the model correctly inferred time step from datastore
    assert (
        model.time_step_int == time_step
    ), f"Expected time_step_int={time_step}, got {model.time_step_int}"
    assert (
        model.time_step_unit == time_unit
    ), f"Expected time_step_unit={time_unit}, got {model.time_step_unit}"

    # Generate prediction
    prediction, target, _, _ = model.common_step(batch)

    # Rescale to original data scale
    prediction_rescaled = prediction * model.state_std + model.state_mean
    target_rescaled = target * model.state_std + model.state_mean

    # Get first example
    pred_slice = prediction_rescaled[0].detach()  # Detach from graph
    target_slice = target_rescaled[0].detach()
    time_slice = batch[3][0]

    # Create DataArrays
    dataset = WeatherDataset(datastore=datastore, split="train")

    time = np.array(time_slice.cpu(), dtype="datetime64[ns]")

    da_prediction = dataset.create_dataarray_from_tensor(
        tensor=pred_slice, time=time, category="state"
    ).unstack("grid_index")

    da_target = dataset.create_dataarray_from_tensor(
        tensor=target_slice, time=time, category="state"
    ).unstack("grid_index")

    # Get vranges
    var_vmin = (
        torch.minimum(
            pred_slice.flatten(0, 1).min(dim=0)[0],
            target_slice.flatten(0, 1).min(dim=0)[0],
        )
        .cpu()
        .numpy()
    )
    var_vmax = (
        torch.maximum(
            pred_slice.flatten(0, 1).max(dim=0)[0],
            target_slice.flatten(0, 1).max(dim=0)[0],
        )
        .cpu()
        .numpy()
    )
    var_vranges = list(zip(var_vmin, var_vmax))

    # Create plot for specified timestep and first variable
    var_names = datastore.get_vars_names("state")
    var_units = datastore.get_vars_units("state")

    fig = vis.plot_prediction(
        datastore=datastore,
        title=f"{var_names[0]} ({var_units[0]}), t={t_i}"
        f"({(time_step * t_i)} {time_unit})",
        vrange=var_vranges[0],
        da_prediction=da_prediction.isel(
            state_feature=0, time=t_i - 1
        ).squeeze(),
        da_target=da_target.isel(state_feature=0, time=t_i - 1).squeeze(),
    )

    # Save for inspection
    output_path = (
        TEST_OUTPUT_DIR
        / f"ar_model_integration_t{t_i}_{time_step}{time_unit}.png"
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved integration test figure to: {output_path}")

    plt.close(fig)

    # Verify the figure was created
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    assert output_path.exists()


def test_plot_error_heatmap_uses_global_color_scale():
    """Heatmap colors should encode absolute values across all variables."""
    errors = torch.tensor(
        [
            [1.0, 100.0, 10.0],
            [2.0, 80.0, 5.0],
            [3.0, 60.0, 2.5],
        ]
    )  # (pred_steps, d_f)
    datastore = HeatmapDatastore(n_vars=errors.shape[1])

    fig = vis.plot_error_heatmap(errors, datastore=datastore)
    ax = fig.axes[0]
    image = ax.images[0]

    np.testing.assert_allclose(image.get_array(), errors.T.numpy())
    assert image.norm.vmin == 0.0
    assert image.norm.vmax == pytest.approx(errors.max().item())
    assert len(fig.axes) == 2  # main axis + colorbar axis

    plt.close(fig)


def test_plot_error_heatmap_adapts_figure_and_font_sizes():
    """Dense heatmaps should get more space and smaller text."""
    small_errors = torch.ones((4, 5))
    large_errors = torch.ones((20, 30))

    small_fig = vis.plot_error_heatmap(
        small_errors, datastore=HeatmapDatastore(n_vars=small_errors.shape[1])
    )
    large_fig = vis.plot_error_heatmap(
        large_errors, datastore=HeatmapDatastore(n_vars=large_errors.shape[1])
    )

    small_ax = small_fig.axes[0]
    large_ax = large_fig.axes[0]

    assert large_fig.get_size_inches()[0] > small_fig.get_size_inches()[0]
    assert large_fig.get_size_inches()[1] > small_fig.get_size_inches()[1]
    assert (
        large_ax.get_yticklabels()[0].get_fontsize()
        < small_ax.get_yticklabels()[0].get_fontsize()
    )
    assert large_ax.texts[0].get_fontsize() < small_ax.texts[0].get_fontsize()
    assert large_ax.get_xticklabels()[0].get_rotation() == 45.0

    plt.close(small_fig)
    plt.close(large_fig)


def test_plot_error_heatmap_skips_annotations_for_very_dense_grids():
    """Very dense heatmaps should omit in-cell text to stay readable."""
    dense_errors = torch.ones((40, 50))
    fig = vis.plot_error_heatmap(
        dense_errors, datastore=HeatmapDatastore(n_vars=dense_errors.shape[1])
    )
    ax = fig.axes[0]

    # No text annotations should be drawn when cells are too small
    assert len(ax.texts) == 0
    # Figure should still grow beyond the old 18-inch cap
    assert fig.get_size_inches()[0] > 18.0

    plt.close(fig)


def test_plot_error_map_deprecated_wrapper():
    """The old plot_error_map name should still work but emit a warning."""
    errors = torch.ones((3, 4))
    datastore = HeatmapDatastore(n_vars=errors.shape[1])

    with pytest.warns(DeprecationWarning, match="plot_error_heatmap"):
        fig = vis.plot_error_map(errors, datastore=datastore)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)

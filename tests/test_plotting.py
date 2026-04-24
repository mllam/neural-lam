# Standard library
from datetime import timedelta
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch

# Third-party
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import xarray as xr
from cartopy import crs as ccrs

# First-party
from neural_lam import config as nlconfig
from neural_lam import vis
from neural_lam.create_graph_with_wmg import create_graph_from_datastore
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.weather_dataset import WeatherDataset
from tests.conftest import init_datastore_example
from tests.dummy_datastore import DummyDatastore

# Create output directory for test figures
TEST_OUTPUT_DIR = Path(__file__).parent / "test_outputs" / "plotting"
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(autouse=True)
def mock_cartopy_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Prevent cartopy from downloading Natural Earth map data during tests.
    Monkeypatches the GeoAxes methods used in vis.plot_on_axis.
    """
    # Third-party
    from cartopy.mpl.geoaxes import GeoAxes

    monkeypatch.setattr(GeoAxes, "coastlines", lambda *args, **kwargs: None)
    monkeypatch.setattr(GeoAxes, "add_feature", lambda *args, **kwargs: None)


@pytest.fixture(autouse=True)
def close_all_figures_after_test() -> Iterator[None]:
    """Ensure test-created matplotlib figures are always cleaned up."""
    yield
    plt.close("all")


def test_plot_prediction() -> None:
    """Check prediction plot structure, titles and shared color scaling."""
    datastore = init_datastore_example("dummydata")
    n_grid = datastore.num_grid_points

    da_pred = xr.DataArray(np.linspace(0.0, 1.0, n_grid))
    da_target = xr.DataArray(np.linspace(1.0, 2.0, n_grid))

    expected_vmin = float(np.nanmin([da_pred.values, da_target.values]))
    expected_vmax = float(np.nanmax([da_pred.values, da_target.values]))

    fig = vis.plot_prediction(
        datastore=datastore,
        da_prediction=da_pred,
        da_target=da_target,
        title="Test Prediction",
        vrange=(expected_vmin, expected_vmax),
        boundary_alpha=None,
        crop_to_interior=False,
    )

    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 3

    ground_truth_ax, prediction_ax, _ = fig.axes
    assert ground_truth_ax.get_title() == "Ground Truth"
    assert prediction_ax.get_title() == "Prediction"
    assert fig._suptitle.get_text() == "Test Prediction"

    assert len(ground_truth_ax.collections) == 1
    assert len(prediction_ax.collections) == 1

    assert ground_truth_ax.collections[0].norm.vmin == expected_vmin
    assert ground_truth_ax.collections[0].norm.vmax == expected_vmax
    assert prediction_ax.collections[0].norm.vmin == expected_vmin
    assert prediction_ax.collections[0].norm.vmax == expected_vmax


def test_plot_error_map() -> None:
    """Check error heatmap content, labels and annotations."""
    datastore = init_datastore_example("dummydata")
    d_f = len(datastore.get_vars_names(category="state"))
    pred_steps = 4

    errors = torch.arange(1, pred_steps * d_f + 1, dtype=torch.float32).reshape(
        pred_steps, d_f
    )

    fig = vis.plot_error_map(
        errors=errors,
        datastore=datastore,
        title="Test Error Map",
    )

    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 1

    ax = fig.axes[0]
    assert len(ax.images) == 1
    assert ax.images[0].get_array().shape == (d_f, pred_steps)
    assert ax.get_xlabel() == "Lead time (h)"
    assert ax.get_title() == "Test Error Map"

    expected_x_ticklabels = [str(step) for step in range(1, pred_steps + 1)]
    actual_x_ticklabels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert actual_x_ticklabels == expected_x_ticklabels

    var_names = datastore.get_vars_names(category="state")
    var_units = datastore.get_vars_units(category="state")
    expected_y_ticklabels = [
        f"{name} ({unit})" for name, unit in zip(var_names, var_units)
    ]
    actual_y_ticklabels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert actual_y_ticklabels == expected_y_ticklabels

    assert len(ax.texts) == pred_steps * d_f


def test_plot_spatial_error() -> None:
    """Check that plot_spatial_error runs without error and returns a Figure."""
    datastore = init_datastore_example("dummydata")
    n_grid = datastore.num_grid_points

    error = torch.linspace(0.0, 1.0, n_grid)

    fig = vis.plot_spatial_error(
        error=error,
        datastore=datastore,
        title="Test Spatial Error",
        boundary_alpha=None,
        crop_to_interior=False,
    )

    assert isinstance(fig, matplotlib.figure.Figure)
    # GeoAxes + colorbar axes
    assert len(fig.axes) == 2
    assert fig.texts[0].get_text() == "Test Spatial Error"


def test_plot_spatial_error_crop_to_interior_changes_extent() -> None:
    """Check interior cropping forwards interior lon/lat bounds to
    set_extent."""
    datastore = init_datastore_example("dummydata")
    n_grid = datastore.num_grid_points
    grid_shape = (datastore.grid_shape_state.x, datastore.grid_shape_state.y)

    boundary_mask = np.ones(grid_shape, dtype=int)
    boundary_mask[2:-2, 2:-2] = 0
    datastore.ds["boundary_mask"] = xr.DataArray(
        boundary_mask.reshape(n_grid), dims=["grid_index"]
    )
    datastore.__dict__.pop("boundary_mask", None)

    lats_lons = datastore.get_lat_lon("state")
    lons = lats_lons[:, 0].reshape(grid_shape)
    lats = lats_lons[:, 1].reshape(grid_shape)
    interior = boundary_mask == 0

    expected_min_lon = float(lons[interior].min())
    expected_max_lon = float(lons[interior].max())
    expected_min_lat = float(lats[interior].min())
    expected_max_lat = float(lats[interior].max())

    error = torch.linspace(0.0, 1.0, n_grid)
    with patch(
        "cartopy.mpl.geoaxes.GeoAxes.set_extent", autospec=True
    ) as set_extent_mock:
        vis.plot_spatial_error(
            error=error,
            datastore=datastore,
            boundary_alpha=None,
            crop_to_interior=True,
        )

    assert set_extent_mock.call_count == 1
    called_extent = set_extent_mock.call_args.args[1]
    called_crs = set_extent_mock.call_args.kwargs["crs"]

    assert called_extent[0] == pytest.approx(expected_min_lon)
    assert called_extent[1] == pytest.approx(expected_max_lon)
    assert called_extent[2] == pytest.approx(expected_min_lat)
    assert called_extent[3] == pytest.approx(expected_max_lat)
    assert isinstance(called_crs, ccrs.PlateCarree)


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
        create_gif = False
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
            archetype="keisler",
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
        title=f"{var_names[0]}, t={t_i} ({time_step * t_i} {time_unit})",
        colorbar_label=var_units[0],
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


@pytest.mark.parametrize(
    "time_step,time_unit",
    [(1, "hours")],
)
def test_plot_examples_gif_integration(model_and_batch, monkeypatch):
    model, batch, datastore, tmp_path = model_and_batch

    # Enable the GIF path and reset the example counter
    model.args.create_gif = True
    model.plotted_examples = 0

    # Minimal logger: plot_examples only reads save_dir and optionally calls
    # log_image
    class _SimpleLogger:
        save_dir = str(tmp_path)

    simple_logger = _SimpleLogger()
    monkeypatch.setattr(
        type(model), "logger", property(lambda self: simple_logger)
    )

    with torch.no_grad():
        prediction, _, _, _ = model.common_step(batch)
        model.plot_examples(
            batch, n_examples=1, prediction=prediction, split="train"
        )

    var_names = datastore.get_vars_names("state")
    pred_steps = batch[1].shape[1]
    example_i = 1
    plot_dir = tmp_path / f"example_plots_{example_i}"

    assert plot_dir.is_dir(), "Plot directory was not created"

    for var_name in var_names:
        # Every time-step must have a PNG frame
        for t_i in range(1, pred_steps + 1):
            png = (
                plot_dir
                / f"{var_name}_example_{example_i}_prediction_t_{t_i:02d}.png"
            )
            assert png.exists(), f"Missing PNG frame: {png.name}"

        # One GIF per variable must exist and be a valid GIF file
        gif = plot_dir / f"{var_name}_example_{example_i}_prediction.gif"
        assert gif.exists(), f"Missing GIF: {gif.name}"
        assert gif.read_bytes()[:3] == b"GIF", f"{gif.name} is not a valid GIF"


# Shared ModelArgs for metrics_watch regression tests (issue #302).
# Kept at module level to avoid copy-paste duplication across tests.
class _MetricsWatchModelArgs:
    output_std = False
    loss = "mse"
    restore_opt = False
    n_example_pred = 1
    graph = "1level"
    hidden_dim = 4
    hidden_layers = 1
    processor_layers = 1
    mesh_aggr = "sum"
    lr = 1.0e-3
    val_steps_to_log = [1, 2]
    metrics_watch = ["val_rmse"]
    var_leads_metrics_watch = {0: [1]}
    num_past_forcing_steps = 0
    num_future_forcing_steps = 0


def test_create_metric_log_dict_with_metrics_watch(tmp_path):
    """
    Regression test for issue #302: AssertionError when using --metrics_watch.

    Previously, aggregate_and_plot_metrics asserted all log_dict values were
    plt.Figure, which failed when --metrics_watch added scalar tensor values.
    This test verifies that create_metric_log_dict correctly returns a single
    dict containing both plt.Figure and scalar entries.
    """
    datastore = DummyDatastore()
    num_state_vars = datastore.get_num_data_vars(category="state")

    graph_dir_path = Path(datastore.root_path) / "graph" / "1level"
    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            n_max_levels=1,
        )

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME,
            config_path=datastore.root_path,
        ),
    )

    model = GraphLAM(
        args=_MetricsWatchModelArgs(),
        config=config,
        datastore=datastore,
    )

    # Create a dummy metric tensor: (pred_steps=2, d_f=num_state_vars)
    metric_tensor = torch.rand(2, num_state_vars)

    # This call should not raise an AssertionError (the original bug)
    log_dict = model.create_metric_log_dict(
        metric_tensor, prefix="val", metric_name="rmse"
    )

    # Verify log_dict contains the error-map figure
    assert "val_rmse" in log_dict
    assert isinstance(log_dict["val_rmse"], plt.Figure)

    # Verify log_dict also contains the watched scalar metric
    var_names = datastore.get_vars_names(category="state")
    expected_key = f"val_rmse_{var_names[0]}_step_1"
    assert expected_key in log_dict, (
        f"Expected key '{expected_key}' in log_dict, "
        f"got keys: {list(log_dict.keys())}"
    )

    # Verify figure entries are plt.Figure and scalar entries are tensors
    for key, value in log_dict.items():
        assert isinstance(
            value, (plt.Figure, torch.Tensor)
        ), f"Unexpected value type for key '{key}': {type(value)}"

    plt.close("all")


def test_aggregate_and_plot_metrics_with_metrics_watch(tmp_path):
    """
    Integration test for issue #302: exercises the full watched-metrics path
    through aggregate_and_plot_metrics(), which is the exact crash site of the
    original AssertionError.

    Previously, aggregate_and_plot_metrics asserted all values in the log dict
    were plt.Figure objects, which failed when --metrics_watch added scalar
    tensor values. This test ensures the full pipeline works without crashing.
    """
    datastore = DummyDatastore()
    num_state_vars = datastore.get_num_data_vars(category="state")

    graph_dir_path = Path(datastore.root_path) / "graph" / "1level"
    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            n_max_levels=1,
        )

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME,
            config_path=datastore.root_path,
        ),
    )

    model = GraphLAM(
        args=_MetricsWatchModelArgs(),
        config=config,
        datastore=datastore,
    )

    # Mock the trainer to simulate rank-0 single-process execution
    mock_trainer = MagicMock()
    mock_trainer.is_global_zero = True
    mock_trainer.sanity_checking = False
    mock_trainer.current_epoch = 0
    model._trainer = mock_trainer

    # Mock logger so log_image calls don't fail.
    # In Lightning, self.logger resolves to self._trainer.logger,
    # so we must attach the mock there.
    mock_logger = MagicMock()
    mock_trainer.logger = mock_logger

    # Patch all_gather_cat to be a no-op (single process)
    model.all_gather_cat = lambda x: x

    # Capture scalar metrics logged via self.log()
    logged_scalars = {}

    def capture_log(key, value, **kwargs):
        logged_scalars[key] = value

    model.log = capture_log

    # Build a fake metrics_dict with MSE entries:
    # shape (N_eval=2, pred_steps=2, d_f=num_state_vars)
    metrics_dict = {"mse": [torch.rand(1, 2, num_state_vars) for _ in range(2)]}

    # This is the exact crash site: should NOT raise AssertionError
    model.aggregate_and_plot_metrics(metrics_dict, prefix="val")

    # Verify that log_image was called (figures were logged)
    mock_logger.log_image.assert_called()

    # Verify that scalar metrics were captured via self.log()
    assert len(logged_scalars) > 0, (
        "Expected scalar metrics to be logged via self.log() "
        "when metrics_watch is configured"
    )

    # Verify the expected watched-metric key is present
    var_names = datastore.get_vars_names(category="state")
    expected_key = f"val_rmse_{var_names[0]}_step_1"
    assert expected_key in logged_scalars, (
        f"Expected key '{expected_key}' in logged scalars, "
        f"got keys: {list(logged_scalars.keys())}"
    )

    plt.close("all")

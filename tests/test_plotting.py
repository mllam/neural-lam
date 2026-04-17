# Standard library
from datetime import timedelta
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

# Third-party
import matplotlib
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
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.weather_dataset import WeatherDataset
from tests.conftest import init_datastore_example
from tests.dummy_datastore import DummyDatastore

# Create output directory for test figures
TEST_OUTPUT_DIR = Path(__file__).parent / "test_outputs" / "plotting"
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session", autouse=True)
def _set_agg_backend():
    """Use non-interactive backend for all plotting tests."""
    plt.switch_backend("Agg")


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


class HeatmapDatastore:
    """Minimal datastore stub for error-heatmap plotting tests."""

    def __init__(
        self,
        n_vars,
        step_length=timedelta(hours=1),
        state_std=None,
        state_diff_std_standardized=None,
    ):
        self._n_vars = n_vars
        self.step_length = step_length
        self._state_std = (
            np.asarray(state_std, dtype=float)
            if state_std is not None
            else np.ones(n_vars, dtype=float)
        )
        self._state_diff_std_standardized = (
            np.asarray(state_diff_std_standardized, dtype=float)
            if state_diff_std_standardized is not None
            else np.ones(n_vars, dtype=float)
        )

    def get_vars_names(self, category):
        assert category == "state"
        return [f"state_var_{i}" for i in range(self._n_vars)]

    def get_vars_units(self, category):
        assert category == "state"
        return ["unit"] * self._n_vars

    def get_standardization_dataarray(self, category):
        assert category == "state"
        return xr.Dataset(
            {
                "state_std": (("state_feature",), self._state_std),
                "state_diff_std_standardized": (
                    ("state_feature",),
                    self._state_diff_std_standardized,
                ),
            }
        )


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
    """Check the deprecated error-heatmap wrapper still renders correctly."""
    datastore = init_datastore_example("dummydata")
    d_f = len(datastore.get_vars_names(category="state"))
    pred_steps = 4

    errors = torch.arange(1, pred_steps * d_f + 1, dtype=torch.float32).reshape(
        pred_steps, d_f
    )

    with pytest.warns(DeprecationWarning, match="plot_error_heatmap"):
        fig = vis.plot_error_map(
            errors=errors,
            datastore=datastore,
            title="Test Error Map",
        )

    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 2

    ax, colorbar_ax = fig.axes
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
    assert colorbar_ax.get_ylabel() != ""


def test_plot_error_heatmap_state_std_normalization():
    """state_std mode: colors are RMSE / state_std, cross-variable comparable."""
    errors = torch.tensor(
        [
            [1.0, 100.0, 10.0],
            [2.0, 80.0, 5.0],
            [3.0, 60.0, 2.5],
        ]
    )
    datastore = HeatmapDatastore(
        n_vars=errors.shape[1],
        state_std=[1.0, 100.0, 10.0],
        state_diff_std_standardized=[1.0, 2.0, 0.5],
    )

    fig = vis.plot_error_heatmap(errors, datastore=datastore, normalization="state_std")
    ax = fig.axes[0]
    image = ax.images[0]
    colorbar = fig.axes[1]

    # color = errors.T / state_std[:, None]
    expected = errors.T.numpy() / np.array([[1.0], [100.0], [10.0]])
    np.testing.assert_allclose(image.get_array(), expected)
    assert image.norm.vmin == pytest.approx(0.0)
    assert image.norm.vmax == pytest.approx(expected.max())
    assert len(fig.axes) == 2
    assert colorbar.get_ylabel() == "RMSE / state_std"

    plt.close(fig)


def test_plot_error_heatmap_one_step_normalization():
    """one_step mode: colors are RMSE / diff_std (physical), cross-variable comparable."""
    errors = torch.tensor(
        [
            [1.0, 100.0, 10.0],
            [2.0, 80.0, 5.0],
            [3.0, 60.0, 2.5],
        ]
    )
    datastore = HeatmapDatastore(
        n_vars=errors.shape[1],
        state_std=[1.0, 100.0, 10.0],
        state_diff_std_standardized=[1.0, 2.0, 0.5],
    )

    fig = vis.plot_error_heatmap(errors, datastore=datastore, normalization="one_step")
    ax = fig.axes[0]
    image = ax.images[0]
    colorbar = fig.axes[1]

    # physical diff_std = state_std * state_diff_std_standardized = [1, 200, 5]
    expected = errors.T.numpy() / np.array([[1.0], [200.0], [5.0]])
    np.testing.assert_allclose(image.get_array(), expected)
    assert image.norm.vmin == pytest.approx(0.0)
    assert image.norm.vmax == pytest.approx(expected.max())
    assert colorbar.get_ylabel() == "Error / Std(1-step change)"

    plt.close(fig)


def test_plot_error_heatmap_falls_back_to_per_var_scale_without_stats():
    """When stats are unavailable colors fall back to per-variable max normalization."""

    class NoStatsHeatmapDatastore(HeatmapDatastore):
        def get_standardization_dataarray(self, category):
            raise KeyError("Missing standardization stats")

    errors = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    datastore = NoStatsHeatmapDatastore(n_vars=errors.shape[1])

    with pytest.warns(UserWarning, match="falling back to per-variable scale"):
        fig = vis.plot_error_heatmap(errors, datastore=datastore)
    ax = fig.axes[0]
    image = ax.images[0]
    colorbar = fig.axes[1]

    # errors_np after transpose: var0=[1,3], var1=[2,4]; max per var: [3,4]
    expected = np.array([[1 / 3, 3 / 3], [2 / 4, 4 / 4]])
    np.testing.assert_allclose(image.get_array(), expected)
    assert image.norm.vmin == pytest.approx(0.0)
    assert image.norm.vmax == pytest.approx(1.0)
    assert "[fallback]" in colorbar.get_ylabel()

    plt.close(fig)


def test_plot_error_heatmap_one_step_falls_back_when_diff_std_absent():
    """one_step mode falls back to per-var max when diff_std is missing, never to state_std."""

    class StateStdOnlyDatastore(HeatmapDatastore):
        def get_standardization_dataarray(self, category):
            return (
                super()
                .get_standardization_dataarray(category)
                .drop_vars("state_diff_std_standardized")
            )

    errors = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    datastore = StateStdOnlyDatastore(n_vars=2, state_std=[2.0, 1.0])

    with pytest.warns(UserWarning, match="falling back to per-variable scale"):
        fig = vis.plot_error_heatmap(
            errors, datastore=datastore, normalization="one_step"
        )
    colorbar = fig.axes[1]
    assert "[fallback]" in colorbar.get_ylabel()
    plt.close(fig)


def test_plot_error_heatmap_state_std_ignores_diff_std():
    """state_std mode uses only state_std; presence or absence of diff_std is irrelevant."""

    class StateStdOnlyDatastore(HeatmapDatastore):
        def get_standardization_dataarray(self, category):
            return (
                super()
                .get_standardization_dataarray(category)
                .drop_vars("state_diff_std_standardized")
            )

    errors = torch.tensor([[2.0, 4.0], [1.0, 3.0]])
    datastore = StateStdOnlyDatastore(n_vars=2, state_std=[2.0, 1.0])
    fig = vis.plot_error_heatmap(errors, datastore=datastore, normalization="state_std")
    ax = fig.axes[0]
    image = ax.images[0]

    # errors_np after transpose: var0=[2,1], var1=[4,3]; state_std=[2,1]
    expected = np.array([[2 / 2, 1 / 2], [4 / 1, 3 / 1]])
    np.testing.assert_allclose(image.get_array(), expected)
    assert image.norm.vmin == pytest.approx(0.0)
    assert image.norm.vmax == pytest.approx(4.0)
    assert fig.axes[1].get_ylabel() == "RMSE / state_std"
    plt.close(fig)


def test_plot_error_heatmap_adapts_layout_for_grid_size():
    """Dense heatmaps adapt size, font scale, and annotation density."""
    small_fig = vis.plot_error_heatmap(
        torch.ones((4, 5)), datastore=HeatmapDatastore(n_vars=5)
    )
    large_fig = vis.plot_error_heatmap(
        torch.ones((20, 30)), datastore=HeatmapDatastore(n_vars=30)
    )
    dense_fig = vis.plot_error_heatmap(
        torch.ones((40, 50)), datastore=HeatmapDatastore(n_vars=50)
    )

    assert large_fig.get_size_inches()[0] > small_fig.get_size_inches()[0]
    assert (
        large_fig.axes[0].get_yticklabels()[0].get_fontsize()
        < small_fig.axes[0].get_yticklabels()[0].get_fontsize()
    )
    assert large_fig.axes[0].get_xticklabels()[0].get_rotation() == 45.0
    assert len(dense_fig.axes[0].texts) == 0
    assert dense_fig.get_size_inches()[0] > 18.0

    plt.close(small_fig)
    plt.close(large_fig)
    plt.close(dense_fig)


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
    assert len(fig.axes) == 2
    assert fig.texts[0].get_text() == "Test Spatial Error"


def test_plot_spatial_error_crop_to_interior_changes_extent() -> None:
    """Check interior cropping forwards interior lon/lat bounds."""
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
    """Setup a model and dataset for testing plot_examples."""
    # Create timedelta with specified step length.
    step_length_kwargs = {time_unit: time_step}
    step_length = timedelta(**step_length_kwargs)

    # Create datastore with specified step length.
    datastore = DummyDatastore(step_length=step_length)

    # Create minimal model args.
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

    # Create graph files if they do not already exist.
    graph_dir_path = Path(datastore.root_path) / "graph" / "1level"
    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            n_max_levels=1,
        )

    # Create config.
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME,
            config_path=datastore.root_path,
        ),
    )

    # Create model.
    model = GraphLAM(
        args=ModelArgs(),
        config=config,
        datastore=datastore,
    )

    # Create dataset to get a sample batch.
    dataset = WeatherDataset(
        datastore=datastore,
        split="train",
        ar_steps=2,
        num_past_forcing_steps=0,
        num_future_forcing_steps=0,
    )

    # Get a batch (just use one sample).
    sample = dataset[0]
    batch = tuple(torch.stack([item]) for item in sample)

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
    """Integration test that saves an actual figure for manual inspection."""
    model, batch, datastore, tmp_path = model_and_batch

    # Reset plotted examples counter.
    model.plotted_examples = 0

    # Verify that the model correctly inferred time step from the datastore.
    assert (
        model.time_step_int == time_step
    ), f"Expected time_step_int={time_step}, got {model.time_step_int}"
    assert (
        model.time_step_unit == time_unit
    ), f"Expected time_step_unit={time_unit}, got {model.time_step_unit}"

    # Generate prediction.
    prediction, target, _, _ = model.common_step(batch)

    # Rescale to original data scale.
    prediction_rescaled = prediction * model.state_std + model.state_mean
    target_rescaled = target * model.state_std + model.state_mean

    # Get first example.
    pred_slice = prediction_rescaled[0].detach()
    target_slice = target_rescaled[0].detach()
    time_slice = batch[3][0]

    # Create DataArrays.
    dataset = WeatherDataset(datastore=datastore, split="train")

    time = np.array(time_slice.cpu(), dtype="datetime64[ns]")

    da_prediction = dataset.create_dataarray_from_tensor(
        tensor=pred_slice, time=time, category="state"
    ).unstack("grid_index")

    da_target = dataset.create_dataarray_from_tensor(
        tensor=target_slice, time=time, category="state"
    ).unstack("grid_index")

    # Get vranges.
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

    var_names = datastore.get_vars_names("state")
    var_units = datastore.get_vars_units("state")

    # Create plot for specified timestep and first variable.
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

    # Save for inspection.
    output_path = (
        TEST_OUTPUT_DIR
        / f"ar_model_integration_t{t_i}_{time_step}{time_unit}.png"
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved integration test figure to: {output_path}")

    plt.close(fig)

    # Verify the figure was created.
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

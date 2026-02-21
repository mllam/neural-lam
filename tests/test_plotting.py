# Standard library
from datetime import timedelta
from pathlib import Path

# Third-party
import imageio
import matplotlib.pyplot as plt
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


# ---------------------------------------------------------------------------
# GIF generation tests
# ---------------------------------------------------------------------------

_GIF_N_PRED_STEPS = 2


class _MockLogger:

    def __init__(self, save_dir):
        self._save_dir = str(save_dir)

    @property
    def save_dir(self):
        return self._save_dir

    def log_image(self, key, images, step=None):
        pass


@pytest.fixture
def gif_model_and_batch(tmp_path, monkeypatch):
    """
    GraphLAM model + single-sample batch with a mock logger.

    Follows the same setup pattern as ``model_and_batch``.
    ``GeoAxes.coastlines`` is patched to avoid network calls to Natural Earth
    during cartopy rendering.
    """
    # Suppress cartopy Natural Earth downloads – no network needed in tests
    monkeypatch.setattr(
        "cartopy.mpl.geoaxes.GeoAxes.coastlines",
        lambda *args, **kwargs: None,
    )

    datastore = DummyDatastore()

    class ModelArgs:
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

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME,
            config_path=datastore.root_path,
        ),
    )
    model = GraphLAM(args=ModelArgs(), config=config, datastore=datastore)

    dataset = WeatherDataset(
        datastore=datastore,
        split="train",
        ar_steps=_GIF_N_PRED_STEPS,
        num_past_forcing_steps=0,
        num_future_forcing_steps=0,
    )
    sample = dataset[0]
    batch = tuple(torch.stack([item]) for item in sample)

    mock_logger = _MockLogger(save_dir=tmp_path)
    monkeypatch.setattr(
        type(model), "logger", property(lambda self: mock_logger)
    )
    model.plotted_examples = 0

    return model, batch, datastore, mock_logger


def test_gif_created_per_variable(gif_model_and_batch):
    """A non-empty GIF file must exist for every state variable."""
    model, batch, datastore, mock_logger = gif_model_and_batch

    with torch.no_grad():
        prediction, _, _, _ = model.common_step(batch)
        model.plot_examples(
            batch, n_examples=1, split="train", prediction=prediction
        )

    plot_dir = Path(mock_logger.save_dir) / "example_plots_1"
    for var_name in datastore.get_vars_names("state"):
        gif_path = plot_dir / f"{var_name}_example_1_prediction.gif"
        assert gif_path.exists(), f"GIF missing for '{var_name}'"
        assert gif_path.stat().st_size > 0, f"GIF empty for '{var_name}'"


def test_png_frames_saved_per_variable_and_timestep(gif_model_and_batch):
    """One PNG frame must be saved for every (variable, timestep) pair."""
    model, batch, datastore, mock_logger = gif_model_and_batch

    with torch.no_grad():
        prediction, _, _, _ = model.common_step(batch)
        model.plot_examples(
            batch, n_examples=1, split="train", prediction=prediction
        )

    plot_dir = Path(mock_logger.save_dir) / "example_plots_1"
    assert plot_dir.is_dir(), "Per-example plot directory not created"

    for var_name in datastore.get_vars_names("state"):
        for t_i in range(1, _GIF_N_PRED_STEPS + 1):
            png = plot_dir / f"{var_name}_example_1_prediction_t_{t_i:02d}.png"
            assert png.exists(), (
                f"Missing PNG for '{var_name}' at t={t_i}"
            )


def test_gif_frame_count_matches_pred_steps(gif_model_and_batch):
    """Each GIF must contain exactly _GIF_N_PRED_STEPS frames."""
    model, batch, datastore, mock_logger = gif_model_and_batch

    with torch.no_grad():
        prediction, _, _, _ = model.common_step(batch)
        model.plot_examples(
            batch, n_examples=1, split="train", prediction=prediction
        )

    plot_dir = Path(mock_logger.save_dir) / "example_plots_1"
    for var_name in datastore.get_vars_names("state"):
        gif_path = plot_dir / f"{var_name}_example_1_prediction.gif"
        reader = imageio.get_reader(str(gif_path))
        n_frames = sum(1 for _ in reader)
        reader.close()
        assert n_frames == _GIF_N_PRED_STEPS, (
            f"GIF for '{var_name}': expected {_GIF_N_PRED_STEPS} frames, "
            f"got {n_frames}"
        )

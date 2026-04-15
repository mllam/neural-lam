# Standard library
from datetime import timedelta
from pathlib import Path

# Third-party
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


@pytest.mark.parametrize("n_examples", [1, 2])
@pytest.mark.parametrize("n_pred_steps", [1, 3])
def test_plot_examples_calls_logger_and_saves_pt_files(
    tmp_path, n_examples, n_pred_steps
):
    """Unit test for vis.plot_examples."""
    # Standard library
    from unittest.mock import MagicMock

    # Third-party
    import isodate

    datastore = DummyDatastore()
    n_state_features = datastore.get_num_data_vars("state")
    n_grid = datastore.num_grid_points

    prediction = torch.randn(n_examples, n_pred_steps, n_grid, n_state_features)
    target = torch.randn(n_examples, n_pred_steps, n_grid, n_state_features)

    t0_ns = int(isodate.parse_datetime("2021-01-01T00:00:00").timestamp() * 1e9)
    step_ns = int(datastore.step_length.total_seconds() * 1e9)
    time_batch = torch.tensor(
        [
            [t0_ns + i * step_ns + j * step_ns for j in range(n_pred_steps)]
            for i in range(n_examples)
        ],
        dtype=torch.int64,
    )
    mock_logger = MagicMock()
    mock_logger.save_dir = str(tmp_path)

    vis.plot_examples(
        datastore=datastore,
        logger=mock_logger,
        prediction=prediction,
        target=target,
        time_batch=time_batch,
        first_example_idx=0,
    )

    expected_log_calls = n_examples * n_pred_steps * n_state_features
    assert mock_logger.log_image.call_count == expected_log_calls
    for i in range(n_examples):
        assert (tmp_path / f"example_pred_{i}.pt").exists()
        assert (tmp_path / f"example_target_{i}.pt").exists()
    plt.close("all")


def test_plot_examples_first_example_idx_offset(tmp_path):
    """Verify first_example_idx offsets saved filenames correctly."""
    # Standard library
    from unittest.mock import MagicMock

    # Third-party
    import isodate

    datastore = DummyDatastore()
    n_state_features = datastore.get_num_data_vars("state")
    n_grid = datastore.num_grid_points

    prediction = torch.randn(1, 1, n_grid, n_state_features)
    target = torch.randn(1, 1, n_grid, n_state_features)

    t0_ns = int(isodate.parse_datetime("2021-01-01T00:00:00").timestamp() * 1e9)
    step_ns = int(datastore.step_length.total_seconds() * 1e9)
    time_batch = torch.tensor([[t0_ns + step_ns]], dtype=torch.int64)

    mock_logger = MagicMock()
    mock_logger.save_dir = str(tmp_path)

    vis.plot_examples(
        datastore=datastore,
        logger=mock_logger,
        prediction=prediction,
        target=target,
        time_batch=time_batch,
        first_example_idx=3,
    )

    assert (tmp_path / "example_pred_3.pt").exists()
    assert not (tmp_path / "example_pred_0.pt").exists()
    plt.close("all")

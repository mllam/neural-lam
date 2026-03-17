# Standard library
from pathlib import Path

# Third-party
import pytest
import pytorch_lightning as pl
import torch
import wandb

# First-party
from neural_lam import config as nlconfig
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.datastore import DATASTORES
from neural_lam.datastore.base import BaseRegularGridDatastore
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.weather_dataset import WeatherDataModule
from tests.conftest import init_datastore_example


def run_simple_training(datastore, set_output_std):
    """
    Run one epoch of a simple model training setup using the given datastore.

    Parameters
    ----------
    datastore : BaseRegularGridDatastore
        Datastore to load data from for training
    set_output_std : bool
        If --output_std should be set during training
    """

    if torch.cuda.is_available():
        device_name = "cuda"
        torch.set_float32_matmul_precision(
            "high"
        )  # Allows using Tensor Cores on A100s
    else:
        device_name = "cpu"

    trainer = pl.Trainer(
        max_epochs=1,
        deterministic=True,
        accelerator=device_name,
        # XXX: `devices` has to be set to 2 otherwise
        # neural_lam.models.ar_model.ARModel.aggregate_and_plot_metrics fails
        # because it expects to aggregate over multiple devices
        devices=2,
        log_every_n_steps=1,
        # use `detect_anomaly` to ensure that we don't have NaNs popping up
        # during training
        detect_anomaly=True,
    )

    graph_name = "1level"

    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name

    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            n_max_levels=1,
        )

    data_module = WeatherDataModule(
        datastore=datastore,
        ar_steps_train=3,
        ar_steps_eval=5,
        standardize=True,
        batch_size=2,
        num_workers=1,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
    )

    class ModelArgs:
        output_std = set_output_std
        loss = "mse"
        restore_opt = False
        n_example_pred = 1
        # XXX: this should be superfluous when we have already defined the
        # model object no?
        graph = graph_name
        hidden_dim = 4
        hidden_layers = 1
        processor_layers = 2
        mesh_aggr = "sum"
        lr = 1.0e-3
        val_steps_to_log = [1, 3]
        metrics_watch = []
        num_past_forcing_steps = 1
        num_future_forcing_steps = 1

    model_args = ModelArgs()

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )

    model = GraphLAM(  # noqa
        args=model_args,
        datastore=datastore,
        config=config,
    )
    wandb.init()
    trainer.fit(model=model, datamodule=data_module)


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_training(datastore_name):
    datastore = init_datastore_example(datastore_name)

    if not isinstance(datastore, BaseRegularGridDatastore):
        pytest.skip(
            f"Skipping test for {datastore_name} as it is not a regular "
            "grid datastore."
        )

    run_simple_training(datastore, set_output_std=False)


def test_training_output_std():
    datastore = init_datastore_example("mdp")  # Test only with mdp datastore
    run_simple_training(datastore, set_output_std=True)


# ---------------------------------------------------------------------------
# Unit tests for specific bug fixes that don't require a full training loop
# ---------------------------------------------------------------------------


def _build_model(datastore, val_steps_to_log, metrics_watch=None,
                 var_leads_metrics_watch=None):
    """
    Instantiate a minimal GraphLAM on *datastore* without starting a trainer.
    The graph directory is created if it does not already exist.
    """
    graph_name = "1level"
    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name
    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            n_max_levels=1,
        )

    class ModelArgs:
        output_std = False
        loss = "mse"
        restore_opt = False
        n_example_pred = 0
        graph = graph_name
        hidden_dim = 4
        hidden_layers = 1
        processor_layers = 2
        mesh_aggr = "sum"
        lr = 1.0e-3
        num_past_forcing_steps = 1
        num_future_forcing_steps = 1

    ModelArgs.val_steps_to_log = val_steps_to_log
    ModelArgs.metrics_watch = metrics_watch or []
    ModelArgs.var_leads_metrics_watch = var_leads_metrics_watch or {}

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )
    return GraphLAM(args=ModelArgs(), datastore=datastore, config=config)


def test_create_metric_log_dict_with_metrics_watch():
    """Regression test: aggregate_and_plot_metrics must not raise when
    metrics_watch and var_leads_metrics_watch are configured, and must
    dispatch figures to log_image() and scalars to log_metrics().

    Before the fix, aggregate_and_plot_metrics had:
        assert all(isinstance(value, plt.Figure) for _, value in log_dict.items())
    which crashed as soon as a scalar was added for a watched variable.

    This test drives aggregate_and_plot_metrics end-to-end with a stubbed
    logger and verifies that log_image is called for Figure entries and
    log_metrics is called for scalar entries without raising.
    """
    from unittest.mock import MagicMock, patch

    import matplotlib.pyplot as plt

    datastore = init_datastore_example("dummydata")
    n_state = datastore.get_num_data_vars("state")
    pred_steps = 3

    model = _build_model(
        datastore,
        val_steps_to_log=[1, 2, 3],
        metrics_watch=["val_rmse"],
        var_leads_metrics_watch={0: [1, 3]},
    )

    # Seed val_metrics with a synthetic batch of MSE values
    model.val_metrics["mse"].append(
        torch.ones(1, pred_steps, n_state, dtype=torch.float32)
    )

    # Stub trainer so that is_global_zero=True and sanity_checking=False
    mock_trainer = MagicMock()
    mock_trainer.is_global_zero = True
    mock_trainer.sanity_checking = False
    mock_trainer.current_epoch = 0

    # Stub logger with inspectable log_image / log_metrics
    class StubLogger:
        def __init__(self):
            self.log_image = MagicMock(name="log_image")
            self.log_metrics = MagicMock(name="log_metrics")

    stub_logger = StubLogger()
    mock_trainer.logger = stub_logger
    model._trainer = mock_trainer

    # all_gather_cat is a no-op in single-process testing
    model.all_gather_cat = lambda t: t

    mock_fig = MagicMock(spec=plt.Figure)
    with patch("neural_lam.vis.plot_error_map", return_value=mock_fig):
        # Before the fix this raised AssertionError; now it must not
        model.aggregate_and_plot_metrics(model.val_metrics, prefix="val")

    assert stub_logger.log_image.called, (
        "Expected log_image to be called for the error-map Figure"
    )
    assert stub_logger.log_metrics.called, (
        "Expected log_metrics to be called for scalar watched-metric entries"
    )


def test_val_steps_to_log_guard_prevents_index_error():
    """Regression test: test_step must not raise IndexError when
    val_steps_to_log contains steps that exceed the prediction horizon, and
    must only store spatial-loss maps for in-range steps.

    validation_step already had this guard; test_step did not. The fix adds
    the guard in two places: the test_log_dict dict-comprehension, and the
    spatial-loss index list.

    This test calls the real test_step with a mocked common_step so that
    removing the guard from ar_model.py would cause this test to fail.
    """
    from unittest.mock import MagicMock

    datastore = init_datastore_example("dummydata")
    n_state = datastore.get_num_data_vars("state")
    n_grid = datastore.num_grid_points
    pred_steps = 3

    model = _build_model(
        datastore,
        val_steps_to_log=[1, 2, 5, 10],  # steps 5 and 10 exceed the 3-step horizon
    )

    # Stub trainer — skip distributed gather and all plotting branches
    mock_trainer = MagicMock()
    mock_trainer.is_global_zero = False
    model._trainer = mock_trainer

    # log_dict requires a live trainer; stub it out
    model.log_dict = MagicMock()

    # Mock common_step to return well-shaped synthetic tensors
    model.common_step = MagicMock(
        return_value=(
            torch.zeros(1, pred_steps, n_grid, n_state),  # prediction
            torch.zeros(1, pred_steps, n_grid, n_state),  # target
            model.per_var_std,  # pred_std (constant per-var std, shape (d_f,))
            None,  # batch_times
        )
    )

    # Must not raise IndexError
    model.test_step(None, 0)

    # Only steps 1 and 2 are within the 3-step horizon
    assert len(model.spatial_loss_maps) == 1, (
        "Expected exactly one spatial_loss_maps entry after one test step"
    )
    n_logged = model.spatial_loss_maps[0].shape[1]
    assert n_logged == 2, (
        f"Expected 2 in-range steps logged (1, 2), got {n_logged}"
    )

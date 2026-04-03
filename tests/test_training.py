# Standard library
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Third-party
import matplotlib.pyplot as plt
import pytest
import pytorch_lightning as pl
import torch
import wandb

# First-party
from neural_lam import config as nlconfig
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.datastore import DATASTORES
from neural_lam.datastore.base import BaseRegularGridDatastore
from neural_lam.models.ar_model import ARModel
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.weather_dataset import WeatherDataModule
from tests.conftest import init_datastore_example
from tests.dummy_datastore import DummyDatastore


def run_simple_training(datastore, set_output_std, metrics_watch=None):
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

    _mw = metrics_watch or []
    _vlmw = {0: [1]} if _mw else {}

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
        metrics_watch = _mw
        var_leads_metrics_watch = _vlmw
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
    wandb.init(mode="disabled")  # Disable wandb for offline test run
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


def test_all_gather_cat_single_device():
    """
    Test that all_gather_cat preserves tensor shape on single-device runs.
    On a single device, all_gather returns the tensor unchanged (no new
    leading dim), so all_gather_cat should not flatten any existing dims.
    """

    class MockModule:
        """Minimal object with mocked single-device all_gather."""

        def all_gather(self, tensor_to_gather, sync_grads=False):
            # Single-device behavior: return tensor unchanged
            return tensor_to_gather

    module = MockModule()
    # Bind the real ARModel.all_gather_cat to our mock
    module.all_gather_cat = ARModel.all_gather_cat.__get__(module, MockModule)

    # Simulate a 3D metric tensor: (N_eval, pred_steps, d_f)
    tensor = torch.randn(4, 3, 5)
    result = module.all_gather_cat(tensor)

    # On single device, shape must be preserved
    assert result.shape == tensor.shape, (
        f"all_gather_cat changed shape on single device: "
        f"{tensor.shape} -> {result.shape}"
    )
    assert torch.equal(result, tensor)


def test_all_gather_cat_multi_device_simulation():
    """
    Test that all_gather_cat correctly flattens when all_gather adds a
    leading dimension (simulating multi-device behavior).
    """

    class MockModule:
        """Object with mocked multi-device all_gather."""

        def all_gather(self, tensor, sync_grads=False):
            # Simulate 2-GPU all_gather: prepend a dim of size 2
            return torch.stack([tensor, tensor], dim=0)

    module = MockModule()
    # Bind the real ARModel.all_gather_cat to our mock
    module.all_gather_cat = ARModel.all_gather_cat.__get__(module, MockModule)

    tensor = torch.randn(4, 3, 5)  # (N_eval, pred_steps, d_f)
    result = module.all_gather_cat(tensor)

    # Should flatten (2, 4, 3, 5) -> (8, 3, 5)
    assert result.shape == (
        8,
        3,
        5,
    ), f"all_gather_cat wrong shape on multi-device: {result.shape}"
    # Validate values match expected concatenation along dim 0
    expected = torch.cat([tensor, tensor], dim=0)
    assert torch.equal(result, expected), (
        "all_gather_cat produced incorrectly ordered/combined values "
        "on multi-device simulation"
    )


class MockARModel(ARModel):
    def predict_step(self, prev_state, prev_prev_state, forcing):
        pred_state = torch.zeros_like(prev_state)
        pred_std = torch.ones_like(prev_state) if self.output_std else None
        return pred_state, pred_std


def test_on_test_epoch_end_resets_test_state(tmp_path):
    datastore = DummyDatastore()

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
        var_leads_metrics_watch = {}
        num_past_forcing_steps = 0
        num_future_forcing_steps = 0

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )
    model = MockARModel(args=ModelArgs(), datastore=datastore, config=config)

    model.test_metrics = {
        "mse": [torch.ones(1, 2, datastore.N_FEATURES["state"])],
        "mae": [torch.ones(1, 2, datastore.N_FEATURES["state"])],
    }
    model.spatial_loss_maps = [
        torch.ones(
            1, len(model.args.val_steps_to_log), datastore.num_grid_points
        )
    ]
    model.plotted_examples = model.n_example_pred
    model.matched_metrics = {"test_rmse"}
    model.aggregate_and_plot_metrics = lambda metrics_dict, prefix: None
    model.all_gather_cat = lambda tensor: tensor
    model._trainer = SimpleNamespace(
        is_global_zero=True,
        sanity_checking=False,
        current_epoch=0,
        logger=SimpleNamespace(
            save_dir=str(tmp_path),
            log_image=lambda **kwargs: None,
        ),
    )

    with patch(
        "neural_lam.models.ar_model.vis.plot_spatial_error",
        side_effect=lambda *args, **kwargs: plt.figure(),
    ):
        model.on_test_epoch_end()

    assert all(
        len(metric_vals) == 0 for metric_vals in model.test_metrics.values()
    )
    assert model.spatial_loss_maps == []
    assert model.plotted_examples == 0
    assert model.matched_metrics == set()

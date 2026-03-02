# Standard library
from pathlib import Path

# Third-party
import pytorch_lightning as pl
import torch

# First-party
from neural_lam import config as nlconfig
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.models.graph_lam import GraphLAM
from tests.dummy_datastore import DummyDatastore


def test_datastore_not_in_checkpoint(tmp_path):
    """
    Test for issue #148: Ensure that the datastore object is not pickled
    into the PyTorch Lightning checkpoint's hyperparameters.
    """
    datastore = DummyDatastore()

    # Minimal model args
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
        val_steps_to_log = [1]
        metrics_watch = []
        num_past_forcing_steps = 0
        num_future_forcing_steps = 0
        var_leads_metrics_watch = {}

    # Create minimal graph
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

    # 1. Assert it is ignored in Lightning's in-memory hparams
    assert "datastore" not in model.hparams

    # 2. Save an actual checkpoint to ensure it's not serialized
    trainer = pl.Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        max_epochs=0,
        logger=False,
        enable_checkpointing=False,
    )
    # Manually attach model to trainer
    trainer.strategy.connect(model)

    ckpt_path = tmp_path / "test.ckpt"
    trainer.save_checkpoint(ckpt_path, weights_only=False)

    # Load and verify
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert "hyper_parameters" in ckpt
    assert "datastore" not in ckpt["hyper_parameters"]

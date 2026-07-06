# Standard library
from pathlib import Path

# Third-party
import pytorch_lightning as pl
import torch

# First-party
from neural_lam import config as nlconfig
from neural_lam import migrations
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.models import ARForecaster, ForecasterModule, GraphLAM
from tests.dummy_datastore import DummyDatastore


def _build_forecaster_module(datastore: DummyDatastore) -> ForecasterModule:
    """Build the minimal `ForecasterModule` the checkpoint tests need."""
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

    predictor = GraphLAM(
        datastore=datastore,
        graph_name="1level",
        hidden_dim=4,
        hidden_layers=1,
        processor_layers=1,
        mesh_aggr="sum",
        num_past_forcing_steps=0,
        num_future_forcing_steps=0,
        output_std=False,
        output_clamping_lower=config.training.output_clamping.lower,
        output_clamping_upper=config.training.output_clamping.upper,
    )
    forecaster = ARForecaster(predictor, datastore)
    return ForecasterModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
        loss="mse",
        lr=1.0e-3,
        n_example_pred=1,
        val_steps_to_log=[1],
    )


def test_saved_checkpoint_excludes_datastore_and_forecaster(tmp_path):
    """
    Regression check for issue #148: heavy non-pickle-safe objects
    (`datastore`, `forecaster`) must be excluded from the saved Lightning
    hyperparameters so that checkpoints stay small and portable, and so
    that `load_from_checkpoint` requires them to be passed in explicitly.
    """
    model = _build_forecaster_module(DummyDatastore())

    # Lightning's in-memory hparams must already drop these.
    assert "datastore" not in model.hparams
    assert "forecaster" not in model.hparams

    # And the on-disk checkpoint round-trip must agree.
    trainer = pl.Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        max_epochs=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.strategy.connect(model)

    ckpt_path = tmp_path / "test.ckpt"
    trainer.save_checkpoint(ckpt_path, weights_only=False)

    # In-process checkpoint, trusted source.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "hyper_parameters" in ckpt
    assert "datastore" not in ckpt["hyper_parameters"]
    assert "forecaster" not in ckpt["hyper_parameters"]


def test_saved_checkpoint_is_stamped_with_current_schema_version(tmp_path):
    """`on_save_checkpoint` must stamp every new checkpoint so that a
    future `on_load_checkpoint` knows whether it needs to migrate it."""
    model = _build_forecaster_module(DummyDatastore())

    trainer = pl.Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        max_epochs=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.strategy.connect(model)

    ckpt_path = tmp_path / "test.ckpt"
    trainer.save_checkpoint(ckpt_path, weights_only=False)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert (
        ckpt["neural_lam_checkpoint_version"]
        == migrations.CURRENT_CHECKPOINT_VERSION
    )


def test_pre_refactor_checkpoint_loads_via_migration(tmp_path):
    """A checkpoint from before #208 (flat `ARModel`, no version stamp,
    `g2m_gnn.grid_mlp` naming) must still load, matching the current
    model's state dict once migrated."""
    model = _build_forecaster_module(DummyDatastore())
    current_state_dict = model.state_dict()

    # Reverse this PR's own migrations to build a legacy state dict: strip
    # the 'forecaster.predictor.' prefix and rename encoding_grid_mlp back
    # to its pre-refactor name.
    legacy_state_dict = {}
    for key, value in current_state_dict.items():
        legacy_key = key.removeprefix("forecaster.predictor.")
        legacy_key = legacy_key.replace("encoding_grid_mlp", "g2m_gnn.grid_mlp")
        legacy_state_dict[legacy_key] = value

    checkpoint = {"state_dict": legacy_state_dict}
    model.on_load_checkpoint(checkpoint)

    assert checkpoint["state_dict"].keys() == current_state_dict.keys()
    assert (
        checkpoint["neural_lam_checkpoint_version"]
        == migrations.CURRENT_CHECKPOINT_VERSION
    )

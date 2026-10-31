# Standard library
import urllib.error
from pathlib import Path

# Third-party
import pooch
import pytest
import pytorch_lightning as pl
import requests
import torch

# First-party
from neural_lam import config as nlconfig
from neural_lam import migrations
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.models import ARForecaster, ForecasterModule, GraphLAM
from neural_lam.train_model import load_forecaster_module_from_checkpoint

# Local
from .conftest import (
    DATASTORE_EXAMPLES_ROOT_PATH,
    S3_BUCKET_NAME,
    S3_ENDPOINT_URL,
)
from .dummy_datastore import DummyDatastore


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
            config_path=str(datastore.root_path),
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


REFERENCE_CHECKPOINT_KEY = (
    "mllam-testdata/neural-lam/model-checkpoints/v0.7.0/"
    "danra_winds_graph_lam_1level_v1.ckpt"
)
REFERENCE_CHECKPOINT_URL = (
    f"{S3_ENDPOINT_URL}/{S3_BUCKET_NAME}/{REFERENCE_CHECKPOINT_KEY}"
)
REFERENCE_CHECKPOINT_HASH = (
    "75ba32cb7b4d1a13dc36033a20ccb3edfae2c3d9f520d637b0d89be3b48c9555"
)


def test_v0_7_0_reference_checkpoint_loads_via_pooch(tmp_path):
    """
    Verify that an authentic v0.7.0 release checkpoint fetched via pooch
    loads cleanly and runs a single forward step.
    """
    try:
        ckpt_file = pooch.retrieve(
            url=REFERENCE_CHECKPOINT_URL,
            known_hash=REFERENCE_CHECKPOINT_HASH,
            path=tmp_path,
            fname="danra_winds_graph_lam_1level_v1.ckpt",
        )
    except (
        requests.exceptions.RequestException,
        urllib.error.URLError,
    ) as e:
        pytest.skip(f"Reference checkpoint not available on S3: {e}")

    danra_config_path = (
        DATASTORE_EXAMPLES_ROOT_PATH / "mdp/danra_100m_winds/config.yaml"
    )
    config, datastore = nlconfig.load_config_and_datastore(danra_config_path)
    loaded_model = load_forecaster_module_from_checkpoint(
        ckpt_path=ckpt_file,
        config=config,
        datastore=datastore,
    )

    assert loaded_model is not None
    predictor = loaded_model.forecaster.predictor
    assert predictor.__class__.__name__ == "GraphLAM"

    # Single forward step
    B, num_grid_nodes = 1, predictor.num_grid_nodes
    d_state = datastore.get_num_data_vars(category="state")
    d_forcing = datastore.get_num_data_vars(category="forcing") * (
        predictor.num_past_forcing_steps
        + predictor.num_future_forcing_steps
        + 1
    )
    init_states = torch.ones(B, 2, num_grid_nodes, d_state)
    forcing_features = torch.ones(B, 1, num_grid_nodes, d_forcing)
    boundary_states = torch.ones(B, 1, num_grid_nodes, d_state)

    with torch.no_grad():
        pred = loaded_model.forecaster(
            init_states, forcing_features, boundary_states
        )

    assert pred.shape[0] == B
    assert pred.shape[-2] == num_grid_nodes

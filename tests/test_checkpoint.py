# Standard library
import subprocess
import sys
from pathlib import Path

# Third-party
import pytorch_lightning as pl
import torch

# First-party
from neural_lam import config as nlconfig
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.models import ARForecaster, ForecasterModule, GraphLAM
from tests.dummy_datastore import DummyDatastore


def test_saved_checkpoint_excludes_datastore_and_forecaster(tmp_path):
    """
    Regression check for issue #148: heavy non-pickle-safe objects
    (`datastore`, `forecaster`) must be excluded from the saved Lightning
    hyperparameters so that checkpoints stay small and portable, and so
    that `load_from_checkpoint` requires them to be passed in explicitly.
    """
    datastore = DummyDatastore()

    # Build the minimum graph the GraphLAM predictor needs.
    graph_dir_path = Path(datastore.root_path) / "graph" / "1level"
    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            n_max_levels=1,
        )

    config = nlconfig.NeuralLAMConfig(
        datastores={
            "main": nlconfig.DatastoreSelection(
                kind=datastore.SHORT_NAME,
                config_path=datastore.root_path,
            )
        },
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
    model = ForecasterModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
        loss="mse",
        lr=1.0e-3,
        n_example_pred=1,
        val_steps_to_log=[1],
    )

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
    trainer.save_checkpoint(ckpt_path)

    # In-process checkpoint, trusted source.
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert "hyper_parameters" in ckpt
    assert "datastore" not in ckpt["hyper_parameters"]
    assert "forecaster" not in ckpt["hyper_parameters"]


def test_load_from_checkpoint_without_importing_train_model(tmp_path):
    """
    Regression check for #799: ForecasterModule.load_from_checkpoint must
    succeed with PyTorch's default weights_only=True unpickler even when
    neural_lam.train_model has not been imported.
    """
    datastore = DummyDatastore()
    graph_dir_path = Path(datastore.root_path) / "graph" / "1level"
    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            n_max_levels=1,
        )

    config = nlconfig.NeuralLAMConfig(
        datastores={
            "main": nlconfig.DatastoreSelection(
                kind=datastore.SHORT_NAME,
                config_path=datastore.root_path,
            )
        },
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
    model = ForecasterModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
        loss="mse",
        lr=1.0e-3,
        n_example_pred=1,
        val_steps_to_log=[1],
    )

    trainer = pl.Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        max_epochs=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.strategy.connect(model)

    ckpt_path = tmp_path / "model.ckpt"
    trainer.save_checkpoint(ckpt_path)

    # In a fresh subprocess that never imported neural_lam.train_model:
    code = f"""
import sys
from pathlib import Path
from tests.conftest import DATASTORES
from tests.dummy_datastore import DummyDatastore
from neural_lam import config as nlconfig
from neural_lam.models import ARForecaster, ForecasterModule, GraphLAM

assert "neural_lam.train_model" not in sys.modules

datastore = DummyDatastore()
datastore._root_path = Path(r"{datastore.root_path}")
config = nlconfig.NeuralLAMConfig(
    datastores={{
        "main": nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME,
            config_path=datastore.root_path,
        )
    }},
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

loaded = ForecasterModule.load_from_checkpoint(
    r"{ckpt_path}",
    map_location="cpu",
    forecaster=forecaster,
    datastore=datastore,
)
assert isinstance(loaded, ForecasterModule)
assert "neural_lam.train_model" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0

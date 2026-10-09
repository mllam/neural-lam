# Standard library
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

# Third-party
import pytest
import pytorch_lightning as pl
import torch
import wandb
from torch.utils.data import DataLoader, TensorDataset

# First-party
from neural_lam import config as nlconfig
from neural_lam.callbacks import EMACallback
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.models import MODELS, ARForecaster, ForecasterModule
from neural_lam.train_model import main
from neural_lam.weather_dataset import WeatherDataModule
from tests.conftest import init_datastore_example


class TinyModule(pl.LightningModule):
    """Small module to exercise the EMA callback quickly in CI."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.layer(x)

    def _loss(self, batch):
        x, y = batch
        return torch.nn.functional.mse_loss(self(x), y)

    def training_step(self, batch, batch_idx):
        return self._loss(batch)

    def validation_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log("val_mean_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        return self._loss(batch)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class WeightProbe(pl.Callback):
    """Records one model weight at hook boundaries.

    Placed after `EMACallback` in the callbacks list, so its hooks run once
    the EMA callback has already swapped weights in or out.
    """

    def __init__(self, param_name="layer.weight"):
        self.param_name = param_name
        self.after_train_batch = None
        self.during_evaluation = []
        self.after_evaluation = None

    def _weight(self, pl_module):
        params = dict(pl_module.named_parameters())
        return params[self.param_name].detach().clone()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.after_train_batch = self._weight(pl_module)

    def on_validation_start(self, trainer, pl_module):
        self.during_evaluation.append(self._weight(pl_module))

    def on_validation_end(self, trainer, pl_module):
        self.after_evaluation = self._weight(pl_module)

    def on_test_start(self, trainer, pl_module):
        self.during_evaluation.append(self._weight(pl_module))

    def on_test_end(self, trainer, pl_module):
        self.after_evaluation = self._weight(pl_module)


def tiny_dataloader():
    """Loader over a handful of random samples for `TinyModule`."""
    return DataLoader(
        TensorDataset(torch.randn(8, 1), torch.randn(8, 1)), batch_size=2
    )


def tiny_trainer(callbacks, **kwargs):
    """Single-epoch CPU trainer, with sanity checking off by default.

    Checkpointing is off unless a test asks for it, so that Lightning does
    not drop a default `ModelCheckpoint` into the working directory.
    """
    kwargs.setdefault("max_epochs", 1)
    kwargs.setdefault("num_sanity_val_steps", 0)
    kwargs.setdefault("enable_checkpointing", False)
    return pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=callbacks,
        **kwargs,
    )


# --- Decay validation ---------------------------------------------------------


@pytest.mark.parametrize("decay", [-0.1, 1.0, 1.5])
def test_decay_outside_unit_interval_rejected(decay):
    """A decay outside [0, 1) is rejected when the callback is built."""
    with pytest.raises(ValueError, match=r"EMA decay must be in"):
        EMACallback(decay=decay)


@pytest.mark.parametrize("decay", [0.0, 0.999, 0.9999])
def test_decay_inside_unit_interval_accepted(decay):
    """Decays in [0, 1) are accepted and kept as given."""
    assert EMACallback(decay=decay).decay == decay


# --- Averaging ----------------------------------------------------------------


def test_update_matches_closed_form():
    """The running average matches `decay * ema + (1 - decay) * param`."""
    decay = 0.9
    module = TinyModule()
    callback = EMACallback(decay=decay)
    callback.on_train_start(None, module)

    expected = {
        name: param.detach().clone()
        for name, param in module.named_parameters()
    }

    for step in range(1, 4):
        with torch.no_grad():
            for param in module.parameters():
                param.add_(float(step))
        callback.on_train_batch_end(None, module, None, None, step)
        for name, param in module.named_parameters():
            expected[name] = (
                decay * expected[name] + (1.0 - decay) * param.detach()
            )

    assert set(callback.ema_state) == set(expected)
    for name, shadow in callback.ema_state.items():
        assert torch.allclose(shadow, expected[name], atol=1e-6)


def test_averaged_weights_cover_every_parameter():
    """Every parameter gets shadow weights, and no buffer does."""
    module = TinyModule()
    module.register_buffer("some_constant", torch.ones(3))
    callback = EMACallback(decay=0.9)
    callback.on_train_start(None, module)

    assert set(callback.ema_state) == {"layer.weight", "layer.bias"}


def test_on_train_start_keeps_restored_weights():
    """Restored shadow weights are not overwritten when training starts."""
    module = TinyModule()
    callback = EMACallback(decay=0.9)
    restored = {
        name: torch.full_like(param, 7.0)
        for name, param in module.named_parameters()
    }
    callback.load_state_dict({"ema_state": restored})

    callback.on_train_start(None, module)

    for name, shadow in callback.ema_state.items():
        assert torch.equal(shadow, restored[name])


def test_missing_shadow_weights_raise():
    """A restored state that does not cover the model is reported clearly."""
    module = TinyModule()
    callback = EMACallback(decay=0.9)
    callback.load_state_dict({"ema_state": {"layer.weight": torch.zeros(1, 1)}})

    with pytest.raises(ValueError, match=r"does not match the current model"):
        callback.on_train_batch_end(None, module, None, None, 0)


# --- Device placement ---------------------------------------------------------


def test_shadow_weights_follow_parameter_device():
    """Placement is deferred to first use, then follows each parameter.

    Lightning restores callback state before the strategy moves the module
    onto its accelerator, so `load_state_dict` must not resolve a device
    itself - the model is still on the CPU at that point. The `meta` device
    stands in for an accelerator so this runs on CPU-only machines too.
    """
    module = TinyModule().to("meta")
    callback = EMACallback(decay=0.9)
    callback.load_state_dict(
        {
            "ema_state": {
                name: torch.zeros(param.shape, device="cpu")
                for name, param in module.named_parameters()
            }
        }
    )

    assert all(t.device.type == "cpu" for t in callback.ema_state.values())

    callback.on_train_batch_end(None, module, None, None, 0)

    assert all(t.device.type == "meta" for t in callback.ema_state.values())


# --- Evaluation ---------------------------------------------------------------


def test_validation_uses_averaged_weights_and_restores_raw():
    """Validation runs on the average; training weights are put back after."""
    module = TinyModule()
    ema = EMACallback(decay=0.5)
    probe = WeightProbe()
    loader = tiny_dataloader()

    tiny_trainer([ema, probe]).fit(
        module, train_dataloaders=loader, val_dataloaders=loader
    )

    assert probe.during_evaluation
    assert torch.equal(
        probe.during_evaluation[-1], ema.ema_state["layer.weight"]
    )
    assert torch.equal(probe.after_evaluation, probe.after_train_batch)
    assert not torch.equal(
        probe.during_evaluation[-1], probe.after_train_batch
    )


def test_sanity_check_without_averaged_weights_is_quiet():
    """Sanity checking runs before the average exists, which is expected."""
    module = TinyModule()
    loader = tiny_dataloader()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tiny_trainer([EMACallback(decay=0.5)], num_sanity_val_steps=1).fit(
            module, train_dataloaders=loader, val_dataloaders=loader
        )

    assert not [w for w in caught if "No EMA weights" in str(w.message)]


def test_evaluation_without_averaged_weights_warns():
    """Evaluating with no average available says so rather than staying mute."""
    module = TinyModule()
    loader = tiny_dataloader()

    with pytest.warns(UserWarning, match=r"No EMA weights available"):
        tiny_trainer([EMACallback(decay=0.5)]).test(module, dataloaders=loader)


# --- Checkpointing ------------------------------------------------------------


def run_and_checkpoint(tmp_path, decay=0.5):
    """Fit `TinyModule` with EMA enabled, returning the module and checkpoint."""
    module = TinyModule()
    ema = EMACallback(decay=decay)
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=str(Path(tmp_path) / "checkpoints"),
        filename="min_val_loss",
        monitor="val_mean_loss",
        mode="min",
        save_top_k=1,
        save_on_train_epoch_end=False,
    )
    loader = tiny_dataloader()

    tiny_trainer([checkpoint, ema], enable_checkpointing=True).fit(
        module, train_dataloaders=loader, val_dataloaders=loader
    )

    return module, ema, Path(checkpoint.best_model_path)


def test_checkpoint_holds_raw_weights_and_averaged_weights(tmp_path):
    """The state dict keeps the raw weights, the average sits beside it.

    Lightning orders `ModelCheckpoint` last among the callbacks, so the raw
    weights are always back in place by the time the checkpoint is written.
    """
    module, ema, ckpt_path = run_and_checkpoint(tmp_path)
    checkpoint = torch.load(
        ckpt_path, map_location="cpu", weights_only=False
    )

    saved = checkpoint["state_dict"]["layer.weight"]
    assert torch.equal(saved, module.layer.weight.detach())

    stored = checkpoint["callbacks"]["EMACallback"]["ema_state"]
    assert set(stored) == {"layer.weight", "layer.bias"}
    assert all(shadow.device.type == "cpu" for shadow in stored.values())
    assert torch.equal(stored["layer.weight"], ema.ema_state["layer.weight"])
    assert not torch.equal(saved, stored["layer.weight"])


def test_averaged_weights_restored_on_resume(tmp_path):
    """Resuming a run picks the average back up where it left off."""
    _, ema, ckpt_path = run_and_checkpoint(tmp_path)

    resumed = EMACallback(decay=0.5)
    tiny_trainer([resumed]).fit(
        TinyModule(),
        train_dataloaders=tiny_dataloader(),
        val_dataloaders=tiny_dataloader(),
        ckpt_path=str(ckpt_path),
    )

    assert set(resumed.ema_state) == set(ema.ema_state)
    for name, shadow in ema.ema_state.items():
        assert torch.equal(resumed.ema_state[name], shadow)


def test_averaged_weights_used_when_evaluating_a_checkpoint(tmp_path):
    """Evaluating a checkpoint tests the average, not the raw weights."""
    saved_module, ema, ckpt_path = run_and_checkpoint(tmp_path)

    probe = WeightProbe()
    tiny_trainer([EMACallback(decay=0.5), probe]).test(
        TinyModule(), dataloaders=tiny_dataloader(), ckpt_path=str(ckpt_path)
    )

    assert probe.during_evaluation
    assert torch.equal(
        probe.during_evaluation[-1], ema.ema_state["layer.weight"]
    )
    # ...and the raw weights out of the checkpoint are put back afterwards
    assert torch.equal(
        probe.after_evaluation, saved_module.layer.weight.detach()
    )


# --- CLI wiring ---------------------------------------------------------------


def trainer_callbacks_for(extra_args, tmp_path):
    """Run `main` far enough to capture the callbacks it hands the trainer."""
    captured = {}

    def capture_trainer(**kwargs):
        captured.update(kwargs)
        raise SystemExit(0)

    argv = [
        "--config_path",
        "dummy.yaml",
        "--runs_root",
        str(tmp_path),
        "--logger_run_name",
        "ema-test",
        *extra_args,
    ]

    with (
        patch(
            "neural_lam.train_model.load_config_and_datastore",
            return_value=(MagicMock(), MagicMock()),
        ),
        patch("neural_lam.train_model.WeatherDataModule"),
        patch("neural_lam.train_model.build_predictor"),
        patch("neural_lam.train_model.ARForecaster"),
        patch("neural_lam.train_model.ForecasterModule"),
        patch("neural_lam.train_model.utils.setup_training_logger"),
        patch("neural_lam.train_model.pl.Trainer", capture_trainer),
        pytest.raises(SystemExit),
    ):
        getattr(main, "__wrapped__", main)(argv)

    return captured["callbacks"]


def test_ema_decay_adds_the_callback(tmp_path):
    """`--ema_decay` reaches the trainer as a configured `EMACallback`."""
    callbacks = trainer_callbacks_for(["--ema_decay", "0.99"], tmp_path)

    added = [cb for cb in callbacks if isinstance(cb, EMACallback)]
    assert len(added) == 1
    assert added[0].decay == 0.99


def test_no_ema_callback_without_the_flag(tmp_path):
    """Leaving `--ema_decay` unset keeps the trainer free of the callback."""
    callbacks = trainer_callbacks_for([], tmp_path)

    assert not any(isinstance(cb, EMACallback) for cb in callbacks)


def test_ema_decay_outside_unit_interval_fails_the_run(tmp_path):
    """An out-of-range `--ema_decay` stops the run rather than being ignored."""
    with pytest.raises(ValueError, match=r"EMA decay must be in"):
        trainer_callbacks_for(["--ema_decay", "1.0"], tmp_path)


# --- Integration --------------------------------------------------------------


@pytest.mark.slow
def test_ema_over_forecaster_module(tmp_path):
    """The callback averages a real `ForecasterModule` through a fit run."""
    datastore = init_datastore_example("dummydata")

    graph_name = "1level"
    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name
    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            n_max_levels=1,
        )

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )
    data_module = WeatherDataModule(
        datastore=datastore,
        ar_steps_train=3,
        ar_steps_eval=5,
        batch_size=2,
        num_workers=1,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
    )
    predictor = MODELS["graph_lam"](
        datastore=datastore,
        graph_name=graph_name,
        hidden_dim=4,
        hidden_layers=1,
        processor_layers=2,
        mesh_aggr="sum",
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        output_std=False,
        output_clamping_lower=config.training.output_clamping.lower,
        output_clamping_upper=config.training.output_clamping.upper,
    )
    model = ForecasterModule(
        forecaster=ARForecaster(predictor, datastore),
        config=config,
        datastore=datastore,
        loss="mse",
        lr=1.0e-3,
        restore_opt=False,
        n_example_pred=1,
        val_steps_to_log=[1, 3],
    )

    ema = EMACallback(decay=0.9)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        default_root_dir=str(tmp_path),
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        callbacks=[ema],
    )

    wandb.init(mode="disabled")
    trainer.fit(model=model, datamodule=data_module)

    assert set(ema.ema_state) == {name for name, _ in model.named_parameters()}
    assert all(
        torch.isfinite(shadow).all() for shadow in ema.ema_state.values()
    )
    # The raw weights are back in the model once validation is over
    assert any(
        not torch.equal(param.detach(), ema.ema_state[name])
        for name, param in model.named_parameters()
    )

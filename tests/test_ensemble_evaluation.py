"""Ensemble scoring and epoch reduction without external forecast data."""

# Standard library
from unittest.mock import MagicMock, Mock

# Third-party
import pytest
import torch

# First-party
from neural_lam.config import NeuralLAMConfig
from neural_lam.models import ProbabilisticForecastingModule
from tests.dummy_datastore import DummyDatastore
from tests.test_probabilistic_forecaster import (
    ConcreteProbabilisticARForecaster,
    NoisyStepPredictor,
)


@pytest.fixture
def evaluation():
    datastore = DummyDatastore(n_grid_points=25)
    config = NeuralLAMConfig(datastores={})
    forecaster = ConcreteProbabilisticARForecaster(
        NoisyStepPredictor(datastore), datastore, config=config
    )
    model = ProbabilisticForecastingModule(
        forecaster, config, datastore, eval_ensemble_size=3
    )
    torch.manual_seed(2)
    init = torch.randn(2, 2, 25, 5)
    target = torch.randn(2, 4, 25, 5)
    forcing = torch.randn(2, 4, 25, 2)
    ensemble = torch.randn(2, 3, 4, 25, 5)
    mask = model.interior_mask_bool
    ensemble[:, :, :, ~mask] = 1e6
    forecaster.sample_ensemble = Mock(
        return_value=(ensemble, torch.full_like(ensemble, 500))
    )
    forecaster.compute_training_loss = Mock(
        return_value=(torch.tensor(0.0), {})
    )
    model.log_dict = Mock()
    return model, (init, target, forcing, None, torch.zeros(2, 4)), ensemble


@pytest.mark.parametrize("phase", ["val", "test"])
def test_evaluation_scores_one_ensemble_on_interior_nodes(evaluation, phase):
    model, batch, ensemble = evaluation
    target = batch[1]
    mask = model.interior_mask_bool
    if phase == "val":
        model.validation_step(batch, 0)
        collected = model.val_metrics
    else:
        model.test_step(batch, 0)
        collected = model.test_metrics
    assert set(collected) == {"ens_mse", "crps_ens"}
    expected_mse = (ensemble.mean(1) - target).square()[:, :, mask].mean(-2)
    first = (ensemble - target.unsqueeze(1)).abs().mean(1)
    pairs = (ensemble.unsqueeze(2) - ensemble.unsqueeze(1)).abs().sum((1, 2))
    expected_crps = (first - pairs / 12)[:, :, mask].mean(-2)
    for name, expected in {
        "ens_mse": expected_mse,
        "crps_ens": expected_crps,
    }.items():
        torch.testing.assert_close(collected[name][0], expected)
        assert collected[name][0].shape == (2, 4, 5)
    model.forecaster.sample_ensemble.assert_called_once()


@pytest.mark.parametrize("prefix", ["val", "test"])
def test_epoch_metrics_root_after_averaging_and_restore_units(
    evaluation, prefix
):
    model, _, _ = evaluation
    trainer = MagicMock(
        is_global_zero=True, sanity_checking=False, current_epoch=0
    )
    model._trainer = trainer
    model.all_gather_cat = lambda values: values
    captured = {}

    def capture(values, phase, name):
        captured[f"{phase}_{name}"] = values
        return {}

    model.create_metric_log_dict = capture
    # Unequal batches must weight individual examples, not batch means.
    squared_values = [torch.ones(1, 2, 5), torch.full((2, 2, 5), 9.0)]
    model.aggregate_and_plot_metrics(
        {
            "ens_mse": squared_values,
            "crps_ens": squared_values,
        },
        prefix,
    )
    state_std = torch.tensor(
        model.datastore.get_standardization_dataarray("state").state_std.values,
        dtype=torch.float32,
    )
    mean = torch.full((2, 5), 19 / 3)
    torch.testing.assert_close(
        captured[f"{prefix}_ens_rmse"], mean.sqrt() * state_std
    )
    torch.testing.assert_close(captured[f"{prefix}_crps_ens"], mean * state_std)


def test_single_member_evaluation_keeps_mean_error(evaluation):
    model, batch, ensemble = evaluation
    model = ProbabilisticForecastingModule(
        model.forecaster,
        model.hparams.config,
        model.datastore,
        eval_ensemble_size=1,
    )
    model.log_dict = Mock()
    model.forecaster.sample_ensemble = Mock(
        return_value=(ensemble[:, :1], None)
    )
    model.validation_step(batch, 0)
    assert set(model.val_metrics) == {"ens_mse"}
    assert torch.isfinite(model.val_metrics["ens_mse"][0]).all()

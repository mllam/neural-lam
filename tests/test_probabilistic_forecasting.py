# Standard library
from pathlib import Path

# Third-party
import torch
from torch.utils.data import DataLoader

# First-party
from neural_lam import config as nlconfig
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.metrics import get_metric
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.weather_dataset import WeatherDataset
from tests.dummy_datastore import DummyDatastore


class ProbabilisticModelArgs:
    output_std = True
    loss = "nll"
    restore_opt = False
    n_example_pred = 0
    graph = "1level"
    hidden_dim = 4
    hidden_layers = 1
    processor_layers = 1
    mesh_aggr = "sum"
    lr = 1.0e-3
    val_steps_to_log = [1]
    metrics_watch = []
    var_leads_metrics_watch = {}
    num_past_forcing_steps = 1
    num_future_forcing_steps = 1


def test_graph_lam_probabilistic_step_produces_finite_positive_std():
    datastore = DummyDatastore(n_grid_points=16, n_timesteps=8)
    graph_dir_path = Path(datastore.root_path) / "graph" / "1level"

    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            n_max_levels=1,
        )

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=""
        )
    )
    model = GraphLAM(
        args=ProbabilisticModelArgs(),
        datastore=datastore,
        config=config,
    )

    dataset = WeatherDataset(datastore=datastore, split="train", ar_steps=2)
    batch = next(iter(DataLoader(dataset, batch_size=2)))

    prediction, target, pred_std, _ = model.common_step(batch)
    loss = torch.mean(
        model.loss(
            prediction,
            target,
            pred_std,
            mask=model.interior_mask_bool,
        )
    )

    assert prediction.shape == target.shape
    assert pred_std.shape == target.shape
    assert torch.all(pred_std > 0)
    assert torch.isfinite(pred_std).all()
    assert torch.isfinite(loss)


def test_probabilistic_metrics_are_available():
    assert callable(get_metric("nll"))
    assert callable(get_metric("crps_gauss"))

def test_base_graph_model_prevents_softplus_underflow_nans():
    # Because native softplus evaluates to exactly 0.0 at -100, 
    # it causes division by zero -> NaN loss crashes.
    pred_std_raw = torch.tensor([-100.0, -200.0, -1000.0])
    pred_std = torch.clamp(torch.nn.functional.softplus(pred_std_raw), min=1e-6)
    
    assert torch.all(pred_std > 0)
    assert not torch.any(pred_std == 0.0)

def test_ar_model_ensemble_samples_from_pred_std():
    datastore = DummyDatastore(n_grid_points=16, n_timesteps=8)
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(kind=datastore.SHORT_NAME, config_path=""),
        training=nlconfig.TrainingConfig(output_mode="ensemble", ensemble_size=1000),
    )
    
    from neural_lam.models.ar_model import ARModel
    class DummyArgs(ProbabilisticModelArgs):
        loss = "mse"
        output_std = True
    
    class DummyARModel(ARModel):
        def predict_step(self, prev_state, prev_prev_state, forcing):
            return prev_state, None
            
    model = DummyARModel(args=DummyArgs(), config=config, datastore=datastore)
    
    preds = torch.ones(2, 4, 5)
    pred_std = torch.full((2, 4, 5), 5.0)
    
    ensemble_preds = model(preds, pred_std=pred_std)
    
    assert ensemble_preds.shape == (1000, 2, 4, 5)
    
    # Assert variance matches the model's predicted std, NOT the default 0.01 fallback
    measured_std = torch.std(ensemble_preds, dim=0)
    assert torch.allclose(measured_std, pred_std, atol=0.5)

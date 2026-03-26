# Standard library
from datetime import timedelta

# Third-party
import torch

# First-party
from neural_lam.models2.ar_forecaster import ARForecaster
from neural_lam.models2.forecaster_module import ForecasterModule


class MockDatastore:
    """
    Minimal datastore stub for ForecasterModule initialization.
    """

    step_length = timedelta(hours=1)


class MockPredictor(torch.nn.Module):
    """
    Minimal one-step predictor used for models2 core tests.
    """

    def __init__(
        self, output_std=False, num_grid_nodes=4, d_state=2, boundary_nodes=None
    ):
        super().__init__()
        self.output_std = output_std
        self.num_grid_nodes = num_grid_nodes
        self.d_state = d_state

        if boundary_nodes is None:
            boundary_nodes = [0, num_grid_nodes - 1]

        boundary_mask = torch.zeros(num_grid_nodes, 1, dtype=torch.float32)
        boundary_mask[boundary_nodes] = 1.0
        self.register_buffer("boundary_mask", boundary_mask, persistent=False)
        self.register_buffer(
            "interior_mask", 1.0 - boundary_mask, persistent=False
        )

        self.register_buffer(
            "state_mean", torch.zeros(d_state), persistent=False
        )
        self.register_buffer("state_std", torch.ones(d_state), persistent=False)
        self.register_buffer(
            "per_var_std",
            0.5 * torch.ones(d_state),
            persistent=False,
        )

    def predict_step(self, prev_state, prev_prev_state, forcing):
        _ = prev_prev_state
        _ = forcing
        pred_state = prev_state + 2.0
        pred_std = torch.ones_like(pred_state) * 0.25 if self.output_std else None
        return pred_state, pred_std


class MockArgs:
    """
    Minimal args holder for ForecasterModule tests.
    """

    loss = "mse"
    lr = 1e-3
    restore_opt = False
    n_example_pred = 0
    val_steps_to_log = [1, 2, 3]
    metrics_watch = []
    var_leads_metrics_watch = {}


def test_ar_forecaster_boundary_and_std_fallback():
    predictor = MockPredictor(output_std=False, num_grid_nodes=4, d_state=1)
    forecaster = ARForecaster(step_predictor=predictor)

    # B=1, conditioning states are 10 and 20.
    init_states = torch.tensor([[[[10.0], [10.0], [10.0], [10.0]], [[20.0], [20.0], [20.0], [20.0]]]])
    forcing_features = torch.zeros(1, 2, 4, 3)

    # Boundary values at each step should overwrite nodes 0 and 3.
    true_states = torch.tensor(
        [[[[100.0], [0.0], [0.0], [200.0]], [[300.0], [0.0], [0.0], [400.0]]]]
    )

    prediction, pred_std = forecaster(
        init_states=init_states,
        forcing_features=forcing_features,
        true_states=true_states,
    )

    # Interior node values: step1 -> 22, step2 -> 24.
    assert prediction.shape == (1, 2, 4, 1)
    assert torch.allclose(prediction[0, 0, 1:3, 0], torch.tensor([22.0, 22.0]))
    assert torch.allclose(prediction[0, 1, 1:3, 0], torch.tensor([24.0, 24.0]))
    # Boundary nodes are overwritten from true_states.
    assert torch.allclose(prediction[0, 0, [0, 3], 0], torch.tensor([100.0, 200.0]))
    assert torch.allclose(prediction[0, 1, [0, 3], 0], torch.tensor([300.0, 400.0]))
    # Matches ARModel contract for output_std=False.
    assert torch.allclose(pred_std, predictor.per_var_std)


def test_ar_forecaster_output_std_with_ensemble_shape():
    predictor = MockPredictor(output_std=True, num_grid_nodes=3, d_state=2)
    forecaster = ARForecaster(step_predictor=predictor)

    init_states = torch.zeros(2, 2, 3, 2)
    forcing_features = torch.zeros(2, 4, 3, 1)
    true_states = torch.zeros(2, 4, 3, 2)

    prediction, pred_std = forecaster(
        init_states=init_states,
        forcing_features=forcing_features,
        true_states=true_states,
        ensemble_size=3,
    )

    assert prediction.shape == (2, 3, 4, 3, 2)
    assert pred_std.shape == (2, 3, 4, 3, 2)


def test_forecaster_module_common_step_and_training_step():
    predictor = MockPredictor(output_std=False, num_grid_nodes=4, d_state=1)
    forecaster = ARForecaster(step_predictor=predictor)
    module = ForecasterModule(
        args=MockArgs(), forecaster=forecaster, datastore=MockDatastore()
    )

    init_states = torch.zeros(2, 2, 4, 1)
    target_states = torch.ones(2, 3, 4, 1)
    forcing_features = torch.zeros(2, 3, 4, 1)
    batch_times = torch.zeros(2, 3, dtype=torch.long)
    batch = (init_states, target_states, forcing_features, batch_times)

    prediction, target, pred_std, times = module.common_step(batch)
    assert prediction.shape == target.shape == (2, 3, 4, 1)
    assert pred_std.shape == (1,)
    assert torch.equal(times, batch_times)

    batch_loss = module.training_step(batch)
    assert batch_loss.ndim == 0
    assert torch.isfinite(batch_loss)


def test_forecaster_module_all_gather_cat_single_and_multi_device_sim():
    class MockModule:
        def all_gather(self, tensor, sync_grads=False):
            _ = sync_grads
            return tensor

    single = MockModule()
    single.all_gather_cat = ForecasterModule.all_gather_cat.__get__(
        single, MockModule
    )
    tensor = torch.randn(4, 3, 5)
    result = single.all_gather_cat(tensor)
    assert result.shape == tensor.shape
    assert torch.equal(result, tensor)

    class MultiMockModule:
        def all_gather(self, tensor, sync_grads=False):
            _ = sync_grads
            return torch.stack([tensor, tensor], dim=0)

    multi = MultiMockModule()
    multi.all_gather_cat = ForecasterModule.all_gather_cat.__get__(
        multi, MultiMockModule
    )
    result = multi.all_gather_cat(tensor)
    assert result.shape == (8, 3, 5)
    assert torch.equal(result, torch.cat([tensor, tensor], dim=0))

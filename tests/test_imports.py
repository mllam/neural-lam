# First-party
import neural_lam
import neural_lam.vis
from neural_lam.forecaster import Forecaster
from neural_lam.step_predictor import StepPredictor
import torch.nn as nn
import torch
from neural_lam import metrics

def test_import():
    assert neural_lam is not None
    assert neural_lam.vis is not None
    assert neural_lam.__version__ is not None


def test_step_predictor_interface():
    """StepPredictor is importable, is an nn.Module subclass, and cannot
    be instantiated directly (it is abstract)."""

    assert issubclass(StepPredictor, nn.Module)
    assert StepPredictor.output_std is False

    # Cannot instantiate the abstract base class
    try:
        StepPredictor()
        assert False, "Expected TypeError when instantiating abstract class"
    except TypeError:
        pass  # expected


def test_forecaster_interface():
    """Forecaster is importable, is an nn.Module subclass, cannot be
    instantiated directly, and exposes the concrete compute_loss method."""

    assert issubclass(Forecaster, nn.Module)
    assert callable(Forecaster.compute_loss)

    # Cannot instantiate the abstract base class
    try:
        Forecaster()  # noqa: F841
        assert False, "Expected TypeError when instantiating abstract class"
    except TypeError:
        pass  # expected


def test_forecaster_compute_loss_concrete():
    """compute_loss works end-to-end with a trivial concrete Forecaster."""

    class _TrivialForecaster(Forecaster):
        """Minimal concrete subclass to exercise compute_loss."""

        def forward(self, init_states, forcing_features, true_states):
            # Just return the true states as the prediction (zero loss)
            return true_states, torch.ones_like(true_states)

    forecaster = _TrivialForecaster()

    B, T, N, d_f = 2, 3, 10, 4
    prediction = torch.zeros(B, T, N, d_f)
    target = torch.zeros(B, T, N, d_f)
    pred_std = torch.ones(d_f)  # constant per-variable std

    loss = forecaster.compute_loss(
        prediction=prediction,
        target=target,
        pred_std=pred_std,
        loss_fn=metrics.mse,
        mask=None,
    )

    assert loss.shape == torch.Size([]), "Expected scalar loss"
    assert loss.item() == 0.0, "Expected zero loss for identical pred/target"


def test_step_predictor_concrete_forward():
    """A minimal concrete StepPredictor can be instantiated and forward
    raises NotImplementedError only if the subclass forgets to override it.
    A proper override should be callable normally."""

    class _PassThroughPredictor(StepPredictor):
        """Always returns prev_state unchanged (identity predictor)."""

        def forward(self, prev_state, prev_prev_state, forcing):
            return prev_state, None

    predictor = _PassThroughPredictor()
    assert isinstance(predictor, nn.Module)
    assert predictor.output_std is False

    B, N, d_f, d_forcing = 2, 10, 4, 6
    prev_state = torch.zeros(B, N, d_f)
    prev_prev_state = torch.zeros(B, N, d_f)
    forcing = torch.zeros(B, N, d_forcing)

    new_state, pred_std = predictor(prev_state, prev_prev_state, forcing)

    assert new_state.shape == (B, N, d_f)
    assert pred_std is None

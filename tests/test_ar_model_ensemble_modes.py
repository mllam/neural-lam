# Third-party
import torch

# First-party
from neural_lam.models.ar_model import ARModel


class DummyARModel(ARModel):
    """Lightweight ARModel subclass for testing ensemble modes.

    Bypass heavy ARModel initialization by setting only the attributes
    needed by `generate_ensemble` and `unroll_prediction`.
    """

    def __init__(
        self,
        ensemble_size=4,
        ensemble_mode="sar",
        perturbation_scale=0.0,
        ic_perturbation_scale=0.5,
        num_grid_nodes=10,
        d_f=2,
    ):
        # Do not call super().__init__ -- instead set minimal attributes
        self.ensemble_size = ensemble_size
        self.ensemble_mode = ensemble_mode
        self.perturbation_scale = perturbation_scale
        self.ic_perturbation_scale = ic_perturbation_scale
        self.output_std = False

        # simple per-var diff std (used to scale SAR noise)
        self.diff_std = torch.ones(d_f) * 0.1
        # per-variable std fallback used by ARModel when output_std is False
        self.per_var_std = self.diff_std.clone()

        # boundary/interior masks
        boundary = torch.zeros(num_grid_nodes, 1)
        # make first and last node boundaries
        boundary[0, 0] = 1.0
        boundary[-1, 0] = 1.0
        self.register_buffer = lambda name, val, persistent=False: setattr(self, name, val)
        self.register_buffer("boundary_mask", boundary, persistent=False)
        self.register_buffer("interior_mask", 1.0 - boundary, persistent=False)

    def predict_step(self, prev_state, prev_prev_state, forcing):
        # simple linear dynamics: next = prev + 0.1 * tendency + small forcing
        tendency = prev_state - prev_prev_state
        next_state = prev_state + 0.1 * tendency + 0.01 * forcing
        pred_std = None
        return next_state, pred_std


def _make_dummy_batch(B=2, T=5, N=10, F=2):
    init_prev_prev = torch.randn(B, N, F)
    init_prev = init_prev_prev + 0.05 * torch.randn_like(init_prev_prev)
    init_states = torch.stack([init_prev_prev, init_prev], dim=1)  # (B,2,N,F)
    forcing = torch.randn(B, T, N, 1)
    # true_states (targets) used only for boundary overwrite here
    true_states = torch.randn(B, T, N, F)
    return init_states, forcing, true_states


def test_lagged_ic_produces_diversity():
    model = DummyARModel(ensemble_size=5, ensemble_mode="lagged_ic", ic_perturbation_scale=0.8)
    init_states, forcing, true_states = _make_dummy_batch()

    ensemble = model.generate_ensemble(init_states, forcing, true_states)
    # ensemble shape: (E, B, T, N, F)
    assert ensemble.shape[0] == model.ensemble_size

    # members must not be identical
    all_same = all(
        torch.allclose(ensemble[0], ensemble[i]) for i in range(1, model.ensemble_size)
    )
    assert not all_same, "lagged_ic members must differ"


def test_hybrid_includes_sar_spread():
    model = DummyARModel(ensemble_size=4, ensemble_mode="hybrid", perturbation_scale=0.5, ic_perturbation_scale=0.5)
    init_states, forcing, true_states = _make_dummy_batch()

    ensemble = model.generate_ensemble(init_states, forcing, true_states)
    # ensure members differ and shape is correct
    assert ensemble.shape[0] == model.ensemble_size
    diffs = [torch.norm(ensemble[i] - ensemble[0]) for i in range(1, model.ensemble_size)]
    # expect non-zero distances
    assert all(d > 0 for d in diffs)

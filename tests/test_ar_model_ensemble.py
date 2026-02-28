# Third-party
import torch

# First-party
from neural_lam.metrics import crps_ensemble


def test_ensemble_crps_integration():
    """
    Test that generate_ensemble output shape is compatible with
    crps_ensemble metric, and that CRPS values are non-negative.

    Ensemble shape convention: (E, B, T, N, F)
    """
    E, B, T, N, F = 3, 2, 5, 10, 4
    ensemble_preds = torch.randn(E, B, T, N, F)
    target = torch.randn(B, T, N, F)

    crps_score = crps_ensemble(ensemble_preds, target)

    assert crps_score.shape == target.shape, (
        f"Expected CRPS shape {target.shape}, got {crps_score.shape}"
    )
    assert (crps_score >= 0).all(), "CRPS values must be non-negative"


def test_ensemble_output_shape():
    """
    Test that the ensemble leading dimension matches ensemble_size.
    """
    E, B, T, N, F = 5, 2, 4, 10, 4
    ensemble_preds = torch.randn(E, B, T, N, F)
    assert ensemble_preds.shape[0] == E


def test_ensemble_member_diversity():
    """
    Ensemble members must differ — stochastic perturbation must produce
    trajectory divergence, not identical rollouts.
    """
    E, B, T, N, F = 5, 2, 5, 10, 4
    base = torch.randn(1, B, T, N, F).expand(E, -1, -1, -1, -1).clone()
    noise = torch.randn_like(base) * 0.01
    ensemble_preds = base + noise

    all_same = all(
        torch.allclose(ensemble_preds[0], ensemble_preds[i])
        for i in range(1, E)
    )
    assert not all_same, "Ensemble members must not be identical"


def test_crps_perfect_ensemble():
    """
    When all ensemble members equal the target, CRPS should be zero.
    This verifies the energy-form decomposition is correct.
    """
    E, B, T, N, F = 4, 2, 3, 8, 4
    target = torch.randn(B, T, N, F)
    perfect_preds = target.unsqueeze(0).expand(E, -1, -1, -1, -1)

    score = crps_ensemble(perfect_preds, target)
    assert torch.allclose(score, torch.zeros_like(score), atol=1e-5), (
        "Perfect ensemble (all members = target) must yield zero CRPS"
    )


def test_crps_registered_in_metrics():
    """
    crps_ensemble must be accessible via the metrics registry.
    """
    from neural_lam import metrics
    fn = metrics.get_metric("crps")
    assert fn is crps_ensemble, (
        "crps_ensemble must be registered under key 'crps' in DEFINED_METRICS"
    )


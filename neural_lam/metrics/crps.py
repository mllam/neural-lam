import torch

def crps_ensemble(predictions: torch.Tensor, target: torch.Tensor):
    """
    Continuous Ranked Probability Score for ensemble forecasts.

    predictions: (E, B, T, N, F)
    target: (B, T, N, F)

    Returns:
        CRPS tensor averaged across ensemble dimension.
    """
    ensemble_size = predictions.shape[0]

    # term 1
    term1 = torch.mean(torch.abs(predictions - target.unsqueeze(0)), dim=0)

    # term 2
    pairwise_diff = torch.abs(
        predictions.unsqueeze(0) - predictions.unsqueeze(1)
    )
    term2 = torch.mean(pairwise_diff, dim=(0, 1)) / 2

    return term1 - term2

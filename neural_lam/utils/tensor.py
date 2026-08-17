"""Numerically stable inverses of common activation functions."""

# Third-party
import torch


def inverse_softplus(
    x: torch.Tensor, beta: float = 1.0, threshold: float = 20.0
) -> torch.Tensor:
    """
    Inverse of :func:`torch.nn.functional.softplus`.

    For most inputs this function is exact up to numerical precision. The
    input is clamped to ensure numerical stability: values above
    ``threshold / beta`` are treated as linear (which is exact in that
    regime), and values near zero are clamped to avoid ``log`` of
    non-positive numbers. Only near the lower clamping bound does the
    result deviate from the true inverse.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor whose softplus inverse should be computed.
    beta : float, optional
        Softplus ``beta`` parameter that controls the sharpness. Default ``1``.
    threshold : float, optional
        Threshold above which the function is treated as linear for numerical
        stability. Default ``20``.

    Returns
    -------
    torch.Tensor
        Tensor containing the inverse-softplus values.

    Notes
    -----
    ``torch.clamp`` will zero the gradients near the bounds, but values this
    close to zero or ``threshold / beta`` already have negligible gradients.
    """
    x_clamped = torch.clamp(
        x, min=torch.log(torch.tensor(1e-6 + 1)) / beta, max=threshold / beta
    )

    non_linear_part = torch.log(torch.expm1(x_clamped * beta)) / beta

    below_threshold = x * beta <= threshold

    x = torch.where(condition=below_threshold, input=non_linear_part, other=x)

    return x


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse of ``torch.sigmoid`` with clamping for numerical stability.

    Sigmoid output takes values in ``[0, 1]``; we clamp the input slightly
    within that open interval before applying ``log(x / (1 - x))``.

    Note that ``torch.clamp`` will make gradients 0 near the bounds, but
    this is not a problem as values of x that are this close to 0 or 1
    have gradients of 0 anyhow.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor assumed to contain logits after a sigmoid.

    Returns
    -------
    torch.Tensor
        Tensor containing ``log(x / (1 - x))`` after clamping away from the
        saturation limits.

    Notes
    -----
    ``torch.clamp`` zeroes gradients for values at the bounds, but values this
    close to 0 or 1 already have negligible gradients.
    """
    x_clamped = torch.clamp(x, min=1e-6, max=1 - 1e-6)
    return torch.log(x_clamped / (1 - x_clamped))
